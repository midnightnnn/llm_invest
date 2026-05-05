from __future__ import annotations

import json

from arena.memory.store import MemoryStore
from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, RiskDecision, Side
from tests.memory.memory_store_helpers import _FakeRepo, _FakeVectorStore


def test_record_thesis_lifecycle_opens_new_thesis_on_filled_buy() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2.0,
        price_krw=100_000,
        rationale="AI demand and margin recovery",
        strategy_refs=["momentum"],
        intent_id="intent_open",
        cycle_id="cycle_open",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_open",
        filled_qty=2.0,
        avg_price_krw=100_000.0,
        message="filled",
    )

    store.record_thesis_lifecycle(intent=intent, decision=decision, report=report, snapshot_before=None)

    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.event_type == "thesis_open"
    assert event.semantic_key
    assert event.payload["thesis_id"] == event.semantic_key
    assert event.payload["position_action"] == "entry"
    assert vector_store.saved == []


def test_record_thesis_lifecycle_preserves_full_thesis_summary() -> None:
    repo = _FakeRepo()
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())
    long_rationale = (
        "AAPL의 서비스 매출 성장과 잉여현금흐름 안정성이 AAPL 매수 thesis를 지지한다. "
        + "sleeve context를 감안해 목표 비중을 유지 가능한 범위에서 올린다. " * 12
        + "THESIS_TAIL"
    )
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2.0,
        price_krw=100_000,
        rationale=long_rationale,
        strategy_refs=["momentum"],
        intent_id="intent_open_long",
        cycle_id="cycle_open_long",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_open_long",
        filled_qty=2.0,
        avg_price_krw=100_000.0,
        message="filled",
    )

    store.record_thesis_lifecycle(intent=intent, decision=decision, report=report, snapshot_before=None)

    assert repo.events[0].payload["thesis_summary"] == long_rationale
    assert repo.events[0].payload["thesis_summary"].endswith("THESIS_TAIL")


def test_record_thesis_lifecycle_skips_non_material_active_buy() -> None:
    repo = _FakeRepo()
    repo.active_thesis_rows["AAPL"] = {
        "event_id": "mem_active",
        "event_type": "thesis_open",
        "semantic_key": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
        "payload_json": json.dumps(
            {
                "thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
                "ticker": "AAPL",
                "thesis_summary": "AI demand and margin recovery",
                "strategy_refs": ["momentum"],
            }
        ),
    }
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=101_000,
        rationale="AI demand and margin recovery",
        strategy_refs=["momentum"],
        intent_id="intent_add",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_add",
        filled_qty=1.0,
        avg_price_krw=101_000.0,
        message="filled",
    )

    store.record_thesis_lifecycle(intent=intent, decision=decision, report=report, snapshot_before=None)

    assert repo.events == []


def test_record_thesis_lifecycle_invalidates_on_thesis_broken_sell() -> None:
    repo = _FakeRepo()
    repo.active_thesis_rows["AAPL"] = {
        "event_id": "mem_active",
        "event_type": "thesis_update",
        "semantic_key": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
        "payload_json": json.dumps(
            {
                "thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
                "ticker": "AAPL",
                "state": "active",
                "thesis_summary": "AI demand and margin recovery",
                "strategy_refs": ["momentum"],
                "entry_cycle_id": "cycle_old",
            }
        ),
    }
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.SELL,
        quantity=1.0,
        price_krw=99_000,
        rationale="Guidance cut broke the thesis",
        strategy_refs=["thesis_broken"],
        intent_id="intent_sell",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_sell",
        filled_qty=1.0,
        avg_price_krw=99_000.0,
        message="filled",
    )

    snapshot = {
        "positions": {
            "AAPL": {"quantity": 2.0},
        }
    }
    store.record_thesis_lifecycle(intent=intent, decision=decision, report=report, snapshot_before=snapshot)

    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.event_type == "thesis_invalidated"
    assert event.payload["position_action"] == "trim"
    assert vector_store.saved[0]["event_type"] == "thesis_invalidated"


def test_record_thesis_lifecycle_realizes_full_exit() -> None:
    repo = _FakeRepo()
    repo.active_thesis_rows["AAPL"] = {
        "event_id": "mem_active",
        "event_type": "thesis_open",
        "semantic_key": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
        "payload_json": json.dumps(
            {
                "thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
                "ticker": "AAPL",
                "state": "open",
                "thesis_summary": "AI demand and margin recovery",
                "strategy_refs": ["momentum"],
                "entry_cycle_id": "cycle_old",
            }
        ),
    }
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.SELL,
        quantity=2.0,
        price_krw=110_000,
        rationale="Target multiple reached",
        strategy_refs=["profit_taking"],
        intent_id="intent_exit",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_exit",
        filled_qty=2.0,
        avg_price_krw=110_000.0,
        message="filled",
    )

    snapshot = {
        "positions": {
            "AAPL": {"quantity": 2.0},
        }
    }
    store.record_thesis_lifecycle(intent=intent, decision=decision, report=report, snapshot_before=snapshot)

    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.event_type == "thesis_realized"
    assert event.payload["position_action"] == "exit"
