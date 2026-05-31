from __future__ import annotations

import math

import pytest

from arena.memory.policy import normalize_memory_policy
from arena.memory.store import MemoryStore
from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, RiskDecision, Side
from tests.memory.memory_store_helpers import _FakeRepo, _FakeVectorStore


def test_record_execution_summary_includes_policy_and_broker_reason() -> None:
    repo = _FakeRepo()
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())

    intent = OrderIntent(
        agent_id="gpt",
        ticker="WMT",
        side=Side.BUY,
        quantity=1.149,
        price_krw=120_000,
        rationale="test",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.ERROR,
        order_id="err_123",
        filled_qty=0.0,
        avg_price_krw=0.0,
        message="market is closed",
    )

    store.record_execution(intent=intent, decision=decision, report=report)

    assert len(repo.events) == 1
    summary = str(repo.events[0].summary)
    assert "status=ERROR" in summary
    assert "policy=approved" in summary
    assert "broker=market is closed" in summary


def test_record_execution_indexes_only_filled_or_simulated() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    intent = OrderIntent(
        agent_id="gpt",
        ticker="WMT",
        side=Side.BUY,
        quantity=1.0,
        price_krw=120_000,
        rationale="test",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])

    store.record_execution(
        intent=intent,
        decision=decision,
        report=ExecutionReport(
            status=ExecutionStatus.REJECTED,
            order_id="rej_1",
            filled_qty=0.0,
            avg_price_krw=0.0,
            message="policy reject",
        ),
    )
    store.record_execution(
        intent=intent,
        decision=decision,
        report=ExecutionReport(
            status=ExecutionStatus.FILLED,
            order_id="fill_1",
            filled_qty=1.0,
            avg_price_krw=120_000.0,
            message="filled",
        ),
    )

    assert [row["event_type"] for row in vector_store.saved] == ["trade_execution"]


def test_record_candidate_memories_persists_bounded_nonheld_screen_hits() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(
        repo=repo,
        vector_store=vector_store,
        trading_mode="paper",
        memory_policy=normalize_memory_policy({}),
    )

    written = store.record_candidate_memories(
        agent_id="gpt",
        candidate_ledger={
            "AAPL": {"source_tools": {"screen_market:momentum"}, "discovery_count": 1},
            "MSFT": {
                "source_tools": {"screen_market:value"},
                "discovery_count": 1,
                "last_seen_rank": 2,
                "discovery_evidence": {"reason_for": "Valuation support", "score": 1.2},
            },
        },
        held_tickers={"AAPL"},
        cycle_id="cycle_candidate",
        phase="execution",
    )

    assert written == 1
    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.event_type == "candidate_screen_hit"
    assert event.payload["ticker"] == "MSFT"
    assert event.payload["cycle_id"] == "cycle_candidate"
    assert event.payload["structured_memory"]["v"] == "candidate_memory_v1"
    assert event.payload["structured_memory"]["t"] == "MSFT"
    assert event.payload["structured_memory"]["src"] == ["screen_market:value"]
    assert event.payload["structured_memory"]["rank"] == 2
    assert event.payload["structured_memory"]["score"] == 1.2
    assert event.payload["structured_memory"]["why"] == "Valuation support"
    assert event.semantic_key.startswith("candidate:gpt:paper:MSFT:")
    assert event.expires_at is not None
    assert vector_store.saved[0]["event_type"] == "candidate_screen_hit"


def test_record_execution_updates_existing_order_memory() -> None:
    repo = _FakeRepo()
    repo.trade_memory_by_order_id["ord_keep"] = {
        "event_id": "mem_existing",
        "created_at": None,
    }
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())

    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2.0,
        price_krw=100_000,
        rationale="reconcile",
        intent_id="intent_x",
    )
    decision = RiskDecision(allowed=True, reason="reconciled", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_keep",
        filled_qty=2.0,
        avg_price_krw=100_000.0,
        message="reconciled",
    )

    store.record_execution(intent=intent, decision=decision, report=report)

    assert len(repo.events) == 0
    assert len(repo.event_updates) == 1
    updated = repo.event_updates[0]
    assert updated["event_id"] == "mem_existing"
    assert "status=FILLED" in str(updated["summary"])
    assert float(updated["score"]) == pytest.approx(0.75)
    assert float(updated["importance_score"]) == pytest.approx(0.75)
    assert float(updated["outcome_score"]) == pytest.approx(0.5)


def _expected_tanh_score(pnl_ratio: float) -> float:
    """tanh 기반 score 공식의 기대값을 계산한다."""
    return max(0.1, min(0.5 + 0.5 * math.tanh(pnl_ratio * 3), 1.0))


def test_record_execution_sells_trigger_buy_score_feedback() -> None:
    repo = _FakeRepo()
    repo.buy_memories = [
        {"event_id": "buy1", "payload_json": '{"intent": {"price_krw": 100.0}}', "score": 1.0}
    ]
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())

    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.SELL,
        quantity=1.0,
        price_krw=150.0,
        rationale="take profit",
    )
    decision = RiskDecision(allowed=True, reason="ok", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ok_1",
        filled_qty=1.0,
        avg_price_krw=150.0,
        message="filled",
    )

    # +50% profit
    store.record_execution(intent=intent, decision=decision, report=report)
    expected = _expected_tanh_score(0.5)
    assert abs(repo.score_updates.get("buy1", 0) - expected) < 0.01

    # -10% loss
    repo.score_updates.clear()
    report.avg_price_krw = 90.0
    store.record_execution(intent=intent, decision=decision, report=report)
    expected = _expected_tanh_score(-0.1)
    assert abs(repo.score_updates.get("buy1", 0) - expected) < 0.01


def test_score_formula_boundary_values() -> None:
    """Score 공식의 변별력을 검증한다."""
    s_50_profit = _expected_tanh_score(0.5)
    s_10_profit = _expected_tanh_score(0.1)
    s_zero = _expected_tanh_score(0.0)
    s_10_loss = _expected_tanh_score(-0.1)
    s_50_loss = _expected_tanh_score(-0.5)

    assert s_50_profit > s_10_profit, "+50% > +10%"
    assert s_10_profit > s_zero, "+10% > 0%"
    assert s_zero > s_10_loss, "0% > -10%"
    assert s_10_loss > s_50_loss, "-10% > -50%"
    assert s_zero == 0.5, "0% pnl = 0.5 score"
    assert s_50_loss >= 0.1, "min score >= 0.1"
    assert s_50_profit <= 1.0, "max score <= 1.0"
