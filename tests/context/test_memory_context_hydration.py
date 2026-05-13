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


def test_context_builder_builds_environment_queries_from_research_briefings() -> None:
    repo = FakeRepo()
    repo.research_briefings = [
        {
            "category": "global_market",
            "headline": "Sticky inflation keeps higher-for-longer rates in play",
            "summary": "Bond yields remain elevated and broad risk appetite is fragile.",
        },
        {
            "category": "geopolitical",
            "headline": "Shipping disruptions lift energy and logistics risk",
            "summary": "Geopolitical tension is pushing oil and freight volatility higher.",
        },
    ]
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
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

    queries = builder._build_memory_search_queries("gpt", snapshot, [])

    assert any("portfolio state" in query for query in queries)
    assert any("macro regime" in query and "higher-for-longer" in query for query in queries)
    assert any("geopolitical risk" in query and "Shipping disruptions" in query for query in queries)


def test_context_builder_hydrates_vector_hits_and_prefers_ticker_overlap() -> None:
    repo = FakeRepo()
    repo.memory_by_id = {
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=3 status=FILLED policy=ok broker=filled",
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
        "evt_reflect": {
            "event_id": "evt_reflect",
            "created_at": "2026-02-24T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "Avoid averaging down on broken cyclicals.",
            "score": 0.4,
            "payload_json": "{}",
        },
    }
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
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
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    market_rows = repo.latest_market_features(["AAPL"], limit=_settings().context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=_settings().context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    vector_store = FakeVectorStore(
        results_by_query={
            queries[0]: [
                {"event_id": "evt_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": "2026-02-20T00:00:00Z"},
                {"event_id": "evt_reflect", "summary": "cyclicals", "score": 0.4, "created_at": "2026-02-24T00:00:00Z"},
            ]
        }
    )
    memory = FakeMemory(recent_rows=[], vector_store=vector_store)
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_trade"
    assert context["memory_events"][0]["tickers"] == ["AAPL"]
    assert context["memory_events"][0]["canonical_tickers"] == ["AAPL"]
    assert context["memory_events"][0]["derived_tickers"] == []
    assert context["memory_events"][0]["side"] == "BUY"
    assert context["memory_events"][0]["canonical_side"] == "BUY"
    assert context["memory_events"][0]["derived_side"] == ""


def test_context_builder_falls_back_to_raw_vector_rows_when_bq_hydration_fails() -> None:
    class FailingHydrateRepo(FakeRepo):
        def memory_events_by_ids(self, *, agent_id, event_ids, trading_mode="paper", tenant_id=None):
            _ = (agent_id, event_ids, trading_mode, tenant_id)
            raise RuntimeError("bq unavailable")

    repo = FailingHydrateRepo()
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
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
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    market_rows = repo.latest_market_features(["AAPL"], limit=_settings().context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=_settings().context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    vector_store = FakeVectorStore(
        results_by_query={
            queries[0]: [
                {
                    "event_id": "evt_trade",
                    "agent_id": "gpt",
                    "event_type": "trade_execution",
                    "summary": "AAPL BUY qty=3 status=FILLED policy=ok broker=filled",
                    "score": 0.7,
                    "created_at": "2026-02-20T00:00:00Z",
                }
            ]
        }
    )
    memory = FakeMemory(recent_rows=[], vector_store=vector_store)
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_trade"
    assert context["memory_events"][0]["event_type"] == "trade_execution"
    assert context["memory_events"][0]["tickers"] == ["AAPL"]
    assert context["memory_events"][0]["canonical_tickers"] == []
    assert context["memory_events"][0]["derived_tickers"] == ["AAPL"]
    assert context["memory_events"][0]["ticker_source"] == "summary_regex"
    assert context["memory_events"][0]["side"] == "BUY"
    assert context["memory_events"][0]["canonical_side"] == ""
    assert context["memory_events"][0]["derived_side"] == "BUY"
    assert context["memory_events"][0]["side_source"] == "summary_keyword"
    assert "Portfolio Memory:" in context["memory_context"]
    assert "~AAPL" in context["memory_context"]
    assert "~BUY" in context["memory_context"]
    assert "prior entry" in context["memory_context"]
    assert "qty=3" not in context["memory_context"]


def test_context_builder_returns_empty_memory_when_state_query_has_no_vector_hits() -> None:
    repo = FakeRepo()
    repo.memory_by_id = {
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=3 status=FILLED policy=ok broker=filled",
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
    }
    repo.ticker_memory_rows = [
        {
            "event_id": "evt_bq_only",
            "created_at": "2026-02-24T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "Fallback-only BQ row should not appear when vector works.",
            "score": 0.9,
            "payload_json": "{}",
        }
    ]
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
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
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    market_rows = repo.latest_market_features(["AAPL"], limit=_settings().context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=_settings().context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    vector_store = FakeVectorStore(results_by_query={queries[0]: []})
    memory = FakeMemory(
        recent_rows=[
            {
                "event_id": "evt_recent_only",
                "created_at": "2026-02-25T00:00:00Z",
                "agent_id": "gpt",
                "event_type": "manual_note",
                "summary": "Recent-only fallback row should stay out.",
                "score": 1.0,
                "payload_json": "{}",
            }
        ],
        vector_store=vector_store,
    )
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)
    assert context["memory_events"] == []
    assert context["memory_context"] == ""


def test_context_builder_renders_research_detail_json_as_compact_digest() -> None:
    repo = FakeRepo()
    repo.research_briefings = [
        {
            "briefing_id": "brf_vz",
            "created_at": "2026-05-12T04:09:05Z",
            "ticker": "VZ",
            "category": "held",
            "headline": "Verbose headline should not be repeated",
            "summary": "EPS beat; FY26 guide raised; postpaid adds positive.",
            "detail_json": {
                "summary": "EPS beat; FY26 guide raised; postpaid adds positive.",
                "risks": ["valuation after rally", "integration execution"],
                "sentiment": "positive",
            },
        }
    ]
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "VZ": Position(
                ticker="VZ",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=120_000,
            )
        },
    )
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["research_context"] == (
        "- [VZ 2026-05-12 +] EPS beat; FY26 guide raised; postpaid adds positive. "
        "Risk: valuation after rally"
    )
    assert context["research_briefings"][0]["detail_json"]["risks"] == [
        "valuation after rally",
        "integration execution",
    ]


def test_context_builder_renders_legacy_research_as_compact_fallback() -> None:
    repo = FakeRepo()
    repo.research_briefings = [
        {
            "briefing_id": "brf_global_old",
            "created_at": "2026-05-09T06:42:27Z",
            "ticker": "GLOBAL",
            "category": "global_market",
            "headline": "GLOBAL 리서치 브리핑",
            "summary": "**글로벌 증시:**\n* 미국 증시는 강세였고 중동 리스크로 유가와 금리가 상승했습니다.",
        }
    ]
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_000_000, positions={})
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["research_context"] == (
        "- [GLOBAL 2026-05-09] 글로벌 증시: 미국 증시는 강세였고 중동 리스크로 유가와 금리가 상승했습니다."
    )
