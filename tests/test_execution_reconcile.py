from __future__ import annotations

import logging

from arena.execution.gateway import ExecutionGateway
from arena.models import AccountSnapshot, ExecutionReport, ExecutionStatus, RiskDecision, Side, utc_now


class _Repo:
    def __init__(self) -> None:
        self.writes = []

    def recent_submitted_reports(self, *, limit: int, lookback_hours: int, trading_mode: str | None = None):
        _ = (limit, lookback_hours, trading_mode)
        return [
            {
                "order_id": "ord_1",
                "intent_id": "intent_1",
                "created_at": utc_now(),
                "trading_mode": trading_mode or "paper",
                "agent_id": "gpt",
                "ticker": "AAPL",
                "side": "BUY",
                "requested_qty": 2.5,
                "avg_price_krw": 100_000.0,
                "status": "SUBMITTED",
            }
        ]

    def write_execution_report(self, intent, report):
        self.writes.append((intent, report))

    def recent_turnover_krw(self, *args, **kwargs):
        _ = (args, kwargs)
        return 0.0

    def recent_intent_count(self, *args, **kwargs):
        _ = (args, kwargs)
        return 0

    def last_trade_time(self, *args, **kwargs):
        _ = (args, kwargs)
        return None

    def write_order_intent(self, intent, decision):
        _ = (intent, decision)

    def resolve_tenant_id(self, tenant_id=None):
        _ = tenant_id
        return "tenant-k"


class _Broker:
    def reconcile_submitted(self, **kwargs):
        _ = kwargs
        return ExecutionReport(
            status=ExecutionStatus.FILLED,
            order_id="ord_1",
            filled_qty=2.0,
            avg_price_krw=101_000.0,
            message="reconciled",
            created_at=utc_now(),
        )


class _FilledBroker:
    def place_order(self, intent, *, fx_rate=None):
        _ = (intent, fx_rate)
        return ExecutionReport(
            status=ExecutionStatus.FILLED,
            order_id="ord_1",
            filled_qty=2.0,
            avg_price_krw=101_000.0,
            message="filled",
            created_at=utc_now(),
        )


class _MemoryStore:
    def __init__(self) -> None:
        self.calls = []
        self.thesis_calls = []

    def record_execution(self, *, intent, decision, report):
        self.calls.append((intent, decision, report))

    def record_thesis_lifecycle(self, *, intent, decision, report, snapshot_before=None):
        self.thesis_calls.append((intent, decision, report, snapshot_before))


class _RiskEngine:
    def __init__(self) -> None:
        self.settings = type("Settings", (), {"trading_mode": "live"})()

    def evaluate(self, **kwargs):
        _ = kwargs
        return RiskDecision(allowed=True, reason="ok", policy_hits=[])


class _ErrorBroker:
    def place_order(self, intent, *, fx_rate=None):
        _ = (intent, fx_rate)
        return ExecutionReport(
            status=ExecutionStatus.ERROR,
            order_id="err_1",
            filled_qty=0.0,
            avg_price_krw=0.0,
            message="boom",
            created_at=utc_now(),
        )


def test_reconcile_submitted_orders_updates_execution_row() -> None:
    repo = _Repo()
    gateway = ExecutionGateway(repo=repo, risk_engine=object(), broker=_Broker(), memory_store=object())

    updated = gateway.reconcile_submitted_orders(limit=50, lookback_hours=24)

    assert updated == 1
    assert len(repo.writes) == 1
    intent, report = repo.writes[0]
    assert intent.intent_id == "intent_1"
    assert report.order_id == "ord_1"
    assert report.status.value == "FILLED"


def test_reconcile_submitted_orders_syncs_memory_store() -> None:
    repo = _Repo()
    memory = _MemoryStore()
    gateway = ExecutionGateway(repo=repo, risk_engine=object(), broker=_Broker(), memory_store=memory)

    updated = gateway.reconcile_submitted_orders(limit=50, lookback_hours=24)

    assert updated == 1
    assert len(memory.calls) == 1
    intent, decision, report = memory.calls[0]
    assert intent.intent_id == "intent_1"
    assert decision.reason == "reconciled"
    assert report.status.value == "FILLED"
    assert len(memory.thesis_calls) == 1


def test_process_order_syncs_thesis_lifecycle_when_available() -> None:
    repo = _Repo()
    memory = _MemoryStore()
    gateway = ExecutionGateway(repo=repo, risk_engine=_RiskEngine(), broker=_FilledBroker(), memory_store=memory)
    snapshot = AccountSnapshot(cash_krw=1_000_000.0, total_equity_krw=1_000_000.0, positions={})
    intent = type(
        "Intent",
        (),
        {
            "agent_id": "gpt",
            "ticker": "AAPL",
            "trading_mode": "live",
            "exchange_code": "NASD",
            "instrument_id": "NASD:AAPL",
            "side": Side.BUY,
            "quantity": 2.0,
            "price_krw": 100_000.0,
            "price_native": 75.0,
            "quote_currency": "USD",
            "fx_rate": 1333.0,
            "rationale": "test",
            "strategy_refs": ["momentum"],
            "created_at": utc_now(),
            "intent_id": "intent_live",
            "cycle_id": "cycle_live",
            "notional_krw": 200_000.0,
        },
    )()

    report = gateway.process(intent, snapshot)

    assert report.status == ExecutionStatus.FILLED
    assert len(memory.thesis_calls) == 1
    thesis_intent, thesis_decision, thesis_report, thesis_snapshot = memory.thesis_calls[0]
    assert thesis_intent.intent_id == "intent_live"
    assert thesis_decision.allowed is True
    assert thesis_report.order_id == "ord_1"
    assert thesis_snapshot is snapshot


def test_process_order_error_logs_tenant(caplog) -> None:
    repo = _Repo()
    memory = _MemoryStore()
    gateway = ExecutionGateway(repo=repo, risk_engine=_RiskEngine(), broker=_ErrorBroker(), memory_store=memory)
    snapshot = AccountSnapshot(cash_krw=1_000_000.0, total_equity_krw=1_000_000.0, positions={})
    intent = type(
        "Intent",
        (),
        {
            "agent_id": "claude",
            "ticker": "001510",
            "trading_mode": "live",
            "exchange_code": "KRX",
            "instrument_id": "KRX:001510",
            "side": Side.BUY,
            "quantity": 10.0,
            "price_krw": 2375.0,
            "price_native": 2375.0,
            "quote_currency": "KRW",
            "fx_rate": 1.0,
            "rationale": "test",
            "strategy_refs": [],
            "created_at": utc_now(),
            "intent_id": "intent_1",
            "cycle_id": "",
            "notional_krw": 23_750.0,
        },
    )()

    with caplog.at_level(logging.ERROR):
        report = gateway.process(intent, snapshot)

    assert report.status == ExecutionStatus.ERROR
    assert "tenant=tenant-k" in caplog.text
