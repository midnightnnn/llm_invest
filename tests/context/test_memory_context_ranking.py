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


def test_context_builder_appends_graph_decision_paths_when_enabled() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    settings = _settings()
    settings.memory_policy = normalize_memory_policy({"graph": {"enabled": True, "max_expansion_hops": 1, "max_expanded_nodes": 6}})
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [{"event_id": "evt_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": "2026-02-22T00:00:00Z"}]
            }
        ),
    )
    repo.memory_by_id = {
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": "2026-02-22T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "score": 0.7,
            "graph_node_id": "mem:evt_trade",
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY","intent_id":"intent_1"}}',
        }
    }
    repo.graph_neighbors_rows = [
        {
            "seed_node_id": "mem:evt_trade",
            "direction": "incoming",
            "neighbor_node_id": "intent:intent_1",
            "edge_id": "edge:precedes:intent:intent_1:evt_trade",
            "edge_created_at": "2026-02-22T00:00:00Z",
            "edge_type": "PRECEDES",
            "edge_strength": 0.9,
            "confidence": 1.0,
            "node_created_at": "2026-02-22T00:00:00Z",
            "node_kind": "order_intent",
            "source_table": "agent_order_intents",
            "source_id": "intent_1",
            "agent_id": "gpt",
            "node_trading_mode": "paper",
            "cycle_id": "cycle_1",
            "summary": "BUY AAPL qty=2 rationale=setup",
            "ticker": "AAPL",
        },
        {
            "seed_node_id": "mem:evt_trade",
            "direction": "incoming",
            "neighbor_node_id": "exec:ord_1",
            "edge_id": "edge:resulted_in:exec:ord_1:evt_trade",
            "edge_created_at": "2026-02-22T00:00:01Z",
            "edge_type": "RESULTED_IN",
            "edge_strength": 1.0,
            "confidence": 1.0,
            "node_created_at": "2026-02-22T00:00:01Z",
            "node_kind": "execution_report",
            "source_table": "execution_reports",
            "source_id": "ord_1",
            "agent_id": "gpt",
            "node_trading_mode": "paper",
            "cycle_id": "cycle_1",
            "summary": "FILLED BUY AAPL filled=2.0000 avg=100000",
            "ticker": "AAPL",
        },
    ]
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert "Decision Paths:" not in context["memory_context"]
    assert "preceded by order intent AAPL" not in context["memory_context"]
    assert "resulted from execution report AAPL" not in context["memory_context"]
    assert context["graph_context"].startswith("Decision Paths:")
    assert "preceded by order intent AAPL" in context["graph_context"]
    assert "resulted from execution report AAPL" in context["graph_context"]
    assert "qty=2" not in context["graph_context"]
    assert "filled=2.0000" not in context["graph_context"]
    assert context["graph_events"]
    assert "2026-02-22" in context["memory_context"]


def test_context_builder_ticker_display_uses_context_tags_and_avoids_plain_words() -> None:
    repo = FakeRepo()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())

    reflection = builder._normalize_memory_row(
        {
            "event_id": "evt_reflect",
            "created_at": "2026-03-10T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "Momentum breakouts in bull/low-vol tech regimes work best when breadth confirms.",
            "importance_score": 0.8,
            "score": 0.8,
            "context_tags_json": {"tickers": ["AAPL"], "regimes": ["bull"], "strategies": ["momentum"]},
        }
    )
    note = builder._normalize_memory_row(
        {
            "event_id": "evt_note",
            "created_at": "2026-03-12T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL liquidity looked fake around the open; wait for second push confirmation.",
            "importance_score": 0.5,
            "score": 0.5,
        }
    )

    assert reflection["tickers"] == ["AAPL"]
    assert reflection["canonical_tickers"] == ["AAPL"]
    assert reflection["derived_tickers"] == []
    assert reflection["ticker_source"] == "context_tags"
    assert "IN" not in reflection["tickers"]
    assert "BULL" not in reflection["tickers"]
    assert note["tickers"] == ["AAPL"]
    assert note["canonical_tickers"] == []
    assert note["derived_tickers"] == ["AAPL"]
    assert note["ticker_source"] == "summary_regex"
    assert note["side"] == ""


def test_normalize_memory_row_treats_naive_datetime_as_utc() -> None:
    repo = FakeRepo()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())

    row = builder._normalize_memory_row(
        {
            "event_id": "evt_naive",
            "created_at": datetime(2026, 3, 12),
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "DuckDB returns naive timestamps.",
            "score": 0.5,
        }
    )

    assert row["created_at"].tzinfo == timezone.utc
    assert isinstance(row["age_days"], int)


def test_context_builder_keeps_summary_fallback_derived_but_out_of_ticker_bonus() -> None:
    repo = FakeRepo()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())

    derived = builder._normalize_memory_row(
        {
            "event_id": "evt_note",
            "created_at": "2026-03-12T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL liquidity looked fake around the open; wait for confirmation.",
            "importance_score": 0.5,
            "score": 0.5,
        }
    )
    canonical = builder._normalize_memory_row(
        {
            "event_id": "evt_trade",
            "created_at": "2026-03-12T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.5,
            "score": 0.5,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        }
    )

    assert builder._memory_ticker_bonus(derived, {"AAPL"}) == 0.0
    assert builder._memory_ticker_bonus(canonical, {"AAPL"}) > 0.0
    assert "~AAPL" in builder._format_memory_line(derived)
    assert "~BUY" not in builder._format_memory_line(derived)
    canonical_line = builder._format_memory_line(canonical)
    assert "prior entry" in canonical_line
    assert "status=FILLED" not in canonical_line


def test_context_builder_uses_temporal_tiers_when_hierarchy_enabled() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {"hierarchy": {"enabled": True, "working_ttl_hours": 24, "episodic_ttl_days": 60}}
    )
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    now = utc_now()
    working_created = (now - timedelta(hours=3)).isoformat()
    reflection_created = (now - timedelta(days=6)).isoformat()
    trade_created = (now - timedelta(days=4)).isoformat()
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_working", "summary": "tool trace", "score": 0.8, "created_at": working_created},
                    {"event_id": "evt_reflect", "summary": "reflection", "score": 0.6, "created_at": reflection_created},
                    {"event_id": "evt_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": trade_created},
                ]
            }
        ),
    )
    repo.memory_by_id = {
        "evt_working": {
            "event_id": "evt_working",
            "created_at": working_created,
            "agent_id": "gpt",
            "event_type": "react_tools_summary",
            "memory_tier": "working",
            "summary": "Technical signals and screen_market were called repeatedly.",
            "importance_score": 0.8,
            "score": 0.8,
            "payload_json": '{"tool_events":[{"tool":"technical_signals"}]}',
        },
        "evt_reflect": {
            "event_id": "evt_reflect",
            "created_at": reflection_created,
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "memory_tier": "semantic",
            "summary": "Avoid chasing weak breadth breakouts without confirmation.",
            "importance_score": 0.6,
            "score": 0.6,
            "payload_json": "{}",
        },
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": trade_created,
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "memory_tier": "episodic",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "outcome_score": 0.8,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert [row["event_id"] for row in context["memory_events"]] == ["evt_reflect", "evt_trade"]
    assert "Neutral Lessons:" in context["memory_context"]
    assert "Avoid chasing weak breadth" in context["memory_context"]
    assert "AAPL BUY" in context["memory_context"]
    assert "status=FILLED" not in context["memory_context"]
    assert "Past Lessons:" not in context["memory_context"]
    assert "Technical signals and screen_market" not in context["memory_context"]


def test_context_builder_prefers_memories_with_matching_context_tags() -> None:
    class TagRepo(FakeRepo):
        def latest_market_features(self, tickers, limit, sources=None):
            _ = (tickers, limit, sources)
            return [
                {
                    "ticker": "AAPL",
                    "close_price_krw": 1000,
                    "ret_20d": 0.14,
                    "ret_5d": 0.04,
                    "volatility_20d": 0.10,
                }
            ]

    repo = TagRepo()
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
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {
            "tagging": {
                "enabled": True,
                "regime_bonus": 0.35,
                "strategy_bonus": 0.25,
                "sector_bonus": 0.15,
            }
        }
    )
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, repo.latest_market_features(["AAPL"], limit=8))
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_miss", "summary": "mismatch", "score": 0.7, "created_at": "2026-02-23T00:00:00Z"},
                    {"event_id": "evt_match", "summary": "match", "score": 0.7, "created_at": "2026-02-23T00:00:00Z"},
                ]
            }
        ),
    )
    repo.memory_by_id = {
        "evt_miss": {
            "event_id": "evt_miss",
            "created_at": "2026-02-23T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
            "primary_regime": "bear",
            "primary_strategy_tag": "mean_reversion",
            "primary_sector": "energy",
            "context_tags_json": '{"regimes":["bear"],"strategies":["mean_reversion"],"sectors":["energy"],"tickers":["AAPL"]}',
        },
        "evt_match": {
            "event_id": "evt_match",
            "created_at": "2026-02-23T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
            "primary_regime": "bull",
            "primary_strategy_tag": "breakout",
            "primary_sector": "tech",
            "context_tags_json": '{"regimes":["bull","low_vol"],"strategies":["momentum","breakout"],"sectors":["tech"],"tickers":["AAPL"]}',
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_match"
    assert context["memory_events"][0]["retrieval_score"] > context["memory_events"][1]["retrieval_score"]


def test_context_builder_prefers_memories_with_stronger_effective_score_when_bonus_enabled() -> None:
    repo = FakeRepo()
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
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {
            "forgetting": {"enabled": True},
            "cleanup": {"min_score": 0.30},
            "retrieval": {
                "reranking": {
                    "effective_score_bonus_scale": 0.12,
                    "effective_score_bonus_cap": 0.12,
                }
            },
        }
    )
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    market_rows = repo.latest_market_features(["AAPL"], limit=settings.context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=settings.context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_low", "summary": "low effective", "score": 0.7, "created_at": "2026-02-23T00:00:00Z"},
                    {"event_id": "evt_high", "summary": "high effective", "score": 0.7, "created_at": "2026-02-23T00:00:00Z"},
                ]
            }
        ),
    )
    base_trade = {
        "created_at": "2026-02-23T00:00:00Z",
        "agent_id": "gpt",
        "event_type": "trade_execution",
        "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
        "importance_score": 0.7,
        "score": 0.7,
        "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
    }
    repo.memory_by_id = {
        "evt_low": {
            "event_id": "evt_low",
            **base_trade,
            "effective_score": 0.32,
        },
        "evt_high": {
            "event_id": "evt_high",
            **base_trade,
            "effective_score": 0.90,
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_high"
    assert context["memory_events"][0]["effective_score"] == 0.9
    assert context["memory_events"][0]["retrieval_score"] > context["memory_events"][1]["retrieval_score"]


def test_context_builder_logs_memory_access_when_forgetting_enabled() -> None:
    repo = FakeRepo()
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {"forgetting": {"enabled": True, "access_log_enabled": True}}
    )
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_reflect", "summary": "reflection", "score": 0.6, "created_at": "2026-02-20T00:00:00Z"},
                    {"event_id": "evt_trade", "summary": "trade", "score": 0.7, "created_at": "2026-02-22T00:00:00Z"},
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
            "summary": "Favor patient entries when breadth is narrowing.",
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
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot, cycle_id="cycle_access")

    assert len(context["memory_events"]) == 2
    assert len(repo.memory_access_rows) == 2
    assert all(row["access_type"] == "retrieval" for row in repo.memory_access_rows)
    assert all(row["used_in_prompt"] is True for row in repo.memory_access_rows)
    assert all(row["cycle_id"] == "cycle_access" for row in repo.memory_access_rows)
