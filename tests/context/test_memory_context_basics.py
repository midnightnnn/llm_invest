from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from arena.config import Settings
from arena.context import ContextBuilder
from arena.memory.policy import normalize_memory_policy
from arena.models import AccountSnapshot, Position, utc_now

from tests.context.helpers import (
    FakeRepo,
    FakeMemory,
    FakeBoard,
    FakeVectorStore,
    _settings,
)


def test_context_builder_skips_initial_memory_without_vector_candidates() -> None:
    repo = FakeRepo()
    memory = FakeMemory(
        recent_rows=[
            {
                "event_id": "evt_recent",
                "created_at": "2026-02-18T00:00:00Z",
                "agent_id": "gpt",
                "event_type": "trade_execution",
                "summary": "최근 체결 요약",
                "score": 0.2,
                "payload_json": '{"x":1}',
            }
        ],
        top_rows=[
            {
                "event_id": "evt_top",
                "created_at": "2026-01-10T00:00:00Z",
                "agent_id": "gpt",
                "event_type": "react_tools_summary",
                "summary": "장기적으로 중요했던 도구 사용 패턴",
                "score": 1.0,
                "payload_json": '{"x":3}',
            },
        ],
    )
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"] == []
    assert context["memory_context"] == ""
    assert "Long-horizon compounding" in context["investment_style_context"] or "low turnover" in context["investment_style_context"]


def test_context_builder_skips_vector_rows_without_event_ids() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"summary": "orphan vector row", "score": 0.8, "created_at": "2026-02-22T00:00:00Z"},
                ]
            }
        ),
    )
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"] == []
    assert context["memory_context"] == ""


def test_context_builder_builds_opportunity_query_for_high_cash_state() -> None:
    builder = ContextBuilder(repo=FakeRepo(), memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    snapshot = AccountSnapshot(
        cash_krw=800_000,
        total_equity_krw=1_000_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=200_000,
            )
        },
    )

    query = builder._build_opportunity_memory_query(snapshot)

    assert query is not None
    assert "new entry opportunity" in query
    assert "opportunity cost compare" in query


def test_context_builder_merge_memory_tracks_reserves_opportunity_slots() -> None:
    builder = ContextBuilder(repo=FakeRepo(), memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    primary_rows = [
        {"event_id": f"h{i}", "retrieval_score": 1.0 - (i * 0.01), "importance_score": 0.8}
        for i in range(5)
    ]
    opportunity_rows = [
        {"event_id": "o1", "retrieval_score": 0.10, "importance_score": 0.3},
        {"event_id": "o2", "retrieval_score": 0.09, "importance_score": 0.2},
    ]

    merged = builder._merge_memory_query_tracks(
        primary_rows=primary_rows,
        opportunity_rows=opportunity_rows,
        total_limit=6,
    )

    assert [row["event_id"] for row in merged[:6]] == ["h0", "h1", "h2", "h3", "o1", "o2"]


def test_context_builder_compresses_memories_into_typed_sections() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_reflect", "summary": "Momentum reflection", "score": 0.6, "created_at": "2026-02-20T00:00:00Z"},
                    {"event_id": "evt_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": "2026-02-22T00:00:00Z"},
                    {"event_id": "evt_note", "summary": "AAPL support note", "score": 0.5, "created_at": "2026-02-23T00:00:00Z"},
                ]
            }
        ),
    )
    repo.memory_by_id = {
        "evt_reflect": {
            "event_id": "evt_reflect",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "Momentum chase after earnings spikes usually fades.",
            "importance_score": 0.6,
            "score": 0.6,
            "payload_json": "{}",
        },
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": "2026-02-22T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "outcome_score": 0.8,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
        "evt_note": {
            "event_id": "evt_note",
            "created_at": "2026-02-23T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL held support while semis rolled over.",
            "importance_score": 0.5,
            "score": 0.5,
            "payload_json": '{"source":"manual_note"}',
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert "Neutral Lessons:" in context["memory_context"]
    assert "Momentum chase" in context["memory_context"]
    assert "AAPL BUY" in context["memory_context"]
    assert "status=FILLED" not in context["memory_context"]
    assert "broker=filled" not in context["memory_context"]
    assert "AAPL held support" in context["memory_context"]


def test_context_builder_reserves_candidate_memory_track() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(
        cash_krw=500_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=120_000,
            )
        },
    )
    settings = _settings()
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    market_rows = repo.latest_market_features(["AAPL"], limit=settings.context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=settings.context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    opportunity_query = seed_builder._build_opportunity_memory_query(snapshot)
    assert opportunity_query is not None
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_aapl_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": "2026-02-22T00:00:00Z"},
                    {"event_id": "evt_aapl_note", "summary": "AAPL risk", "score": 0.6, "created_at": "2026-02-23T00:00:00Z"},
                    {"event_id": "evt_aapl_extra", "summary": "AAPL extra", "score": 0.5, "created_at": "2026-02-24T00:00:00Z"},
                ],
                opportunity_query: [
                    {"event_id": "evt_neutral", "summary": "avoid one-day breakouts", "score": 0.6, "created_at": "2026-02-21T00:00:00Z"},
                ],
            }
        ),
    )
    repo.memory_by_id = {
        "evt_aapl_trade": {
            "event_id": "evt_aapl_trade",
            "created_at": "2026-02-22T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
        "evt_aapl_note": {
            "event_id": "evt_aapl_note",
            "created_at": "2026-02-23T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL valuation risk note.",
            "importance_score": 0.6,
            "score": 0.6,
            "payload_json": '{"ticker":"AAPL"}',
        },
        "evt_aapl_extra": {
            "event_id": "evt_aapl_extra",
            "created_at": "2026-02-24T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL duplicate exposure note.",
            "importance_score": 0.5,
            "score": 0.5,
            "payload_json": '{"ticker":"AAPL"}',
        },
        "evt_neutral": {
            "event_id": "evt_neutral",
            "created_at": "2026-02-21T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "Avoid treating one-day breakouts as confirmation.",
            "importance_score": 0.6,
            "score": 0.6,
            "payload_json": "{}",
        },
    }
    repo.candidate_memory_rows = [
        {
            "event_id": "cand_msft",
            "created_at": "2026-02-24T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "candidate_screen_hit",
            "summary": "MSFT candidate_screen_hit: surfaced by screen_market:value; evidence is screen-only.",
            "importance_score": 0.25,
            "score": 0.25,
            "payload_json": '{"source":"candidate_discovery","ticker":"MSFT","evidence_level":"screened_only"}',
        }
    ]
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert "Portfolio Memory:" in context["memory_context"]
    assert "Candidate Memory:" in context["memory_context"]
    assert "Neutral Lessons:" in context["memory_context"]
    assert any(row["event_id"] == "cand_msft" and row["memory_track"] == "candidate" for row in context["memory_events"])
    assert sum(1 for row in context["memory_events"] if "AAPL" in row.get("tickers", [])) <= 2
