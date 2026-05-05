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

def test_relation_triples_shadow_mode_does_not_affect_retrieval() -> None:
    repo = FakeRepo()
    repo.relation_candidate_rows = [
        {
            "event_id": "evt_relation",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "AAPL relation candidate should stay shadowed.",
            "score": 0.9,
            "importance_score": 0.9,
            "payload_json": "{}",
            "relation_predicate": "contains",
            "relation_object_type": "ticker",
            "relation_object_label": "AAPL",
            "relation_confidence": 0.95,
            "relation_evidence_text": "AAPL relation candidate should stay shadowed.",
        }
    ]
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=1, avg_price_krw=100_000, market_price_krw=120_000)
        },
    )
    builder = ContextBuilder(repo=repo, memory=FakeMemory(vector_store=FakeVectorStore()), board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert repo.relation_candidate_calls == []
    assert context["memory_events"] == []
    assert context["relation_context"] == ""


def test_relation_triples_boost_mode_adds_relation_candidates() -> None:
    repo = FakeRepo()
    repo.relation_candidate_rows = [
        {
            "event_id": "evt_relation",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "AAPL relation candidate captures a prior risk lesson.",
            "score": 0.5,
            "importance_score": 0.5,
            "payload_json": "{}",
            "relation_predicate": "contains",
            "relation_object_type": "ticker",
            "relation_object_label": "AAPL",
            "relation_confidence": 0.9,
            "relation_evidence_text": "AAPL relation candidate captures a prior risk lesson.",
        }
    ]
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {"graph": {"semantic_triples": {"mode": "boost", "min_confidence": 0.8, "boost_bonus_base": 0.2}}}
    )
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=1, avg_price_krw=100_000, market_price_krw=120_000)
        },
    )
    builder = ContextBuilder(repo=repo, memory=FakeMemory(vector_store=FakeVectorStore()), board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert repo.relation_candidate_calls
    assert "ticker:AAPL" in repo.relation_candidate_calls[0]["seed_node_ids"]
    assert [row["event_id"] for row in context["memory_events"]] == ["evt_relation"]
    assert context["memory_events"][0]["relation_boost"] == pytest.approx(0.18)
    assert context["relation_context"] == ""


def test_relation_triples_inject_mode_adds_relation_context() -> None:
    repo = FakeRepo()
    repo.relation_candidate_rows = [
        {
            "event_id": "evt_relation",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "AAPL relation candidate captures a prior risk lesson.",
            "score": 0.5,
            "importance_score": 0.5,
            "payload_json": "{}",
            "relation_predicate": "contains",
            "relation_object_type": "ticker",
            "relation_object_label": "AAPL",
            "relation_confidence": 0.9,
            "relation_evidence_text": "AAPL relation candidate captures a prior risk lesson.",
        }
    ]
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {"graph": {"semantic_triples": {"mode": "inject", "max_relation_context_items": 2}}}
    )
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=1, avg_price_krw=100_000, market_price_krw=120_000)
        },
    )
    builder = ContextBuilder(repo=repo, memory=FakeMemory(vector_store=FakeVectorStore()), board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_relation"
    assert context["relation_context"].startswith("Relation Hints:")
    assert "contains ticker AAPL" in context["relation_context"]
