from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.config import Settings
from arena.models import AccountSnapshot, Position
from arena.reconciliation import StateReconciliationService, StateRecoveryService

from tests.trading.reconciliation_helpers import (
    _FakeRepo,
    _settings,
)

def test_orphan_execution_report_fallback_us_market_timing() -> None:
    """Broker trade occurred_at before checkpoint but execution_report created_at after.

    This reproduces the US-market timing gap where KIS reports the trade timestamp
    in US session time (before the batch checkpoint), causing broker_trade_events
    to fall outside the replay window.  execution_reports are the primary source
    for AI trades so the trade is applied directly without needing a fallback.
    """
    checkpoint_at = datetime(2026, 3, 24, 20, 35, 53, tzinfo=timezone.utc)
    broker_trade_at = datetime(2026, 3, 24, 5, 46, 56, tzinfo=timezone.utc)  # before checkpoint
    exec_report_at = datetime(2026, 3, 25, 19, 13, 4, tzinfo=timezone.utc)  # after checkpoint

    repo = _FakeRepo()
    repo.snapshot_at = checkpoint_at
    repo.snapshot = AccountSnapshot(
        cash_krw=10_000_000.0,
        total_equity_krw=12_000_000.0,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=2.0, avg_price_krw=100_000.0, market_price_krw=120_000.0),
        },
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": checkpoint_at,
            "cash_krw": 5_000_000.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}, {"ticker": "MRVL", "quantity": 1}],
            "source": "test",
        },
        "gemini": {
            "agent_id": "gemini",
            "event_id": "chk_gemini",
            "checkpoint_at": checkpoint_at,
            "cash_krw": 5_000_000.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}],
            "source": "test",
        },
    }

    # Broker trade event BEFORE checkpoint (won't be picked up by since filter)
    repo.broker_trade_rows = [
        {
            "event_id": "bt_mrvl_sell",
            "occurred_at": broker_trade_at,
            "broker_order_id": "0030558700",
            "ticker": "MRVL",
            "side": "SELL",
            "quantity": 1.0,
            "price_krw": 120_000.0,
            "status": "FILLED",
        },
    ]

    # Execution report AFTER checkpoint (will be picked up)
    repo.filled_execution_rows = [
        {
            "order_id": "0030558700",
            "created_at": exec_report_at,
            "agent_id": "gpt",
            "ticker": "MRVL",
            "side": "SELL",
            "filled_qty": 1.0,
            "status": "FILLED",
        },
    ]

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt", "gemini"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True, f"Expected ok but got issues: {[(i.issue_type, i.entity_key, i.severity) for i in result.issues]}"

    # execution_reports are the primary source — no fallback needed
    fallback_issues = [i for i in result.issues if i.issue_type == "execution_report_fallback_applied"]
    assert len(fallback_issues) == 0


def test_no_orphan_when_broker_trade_in_window() -> None:
    """When broker_trade_events are within the replay window, no fallback needed."""
    checkpoint_at = datetime(2026, 3, 24, 20, 0, 0, tzinfo=timezone.utc)
    trade_at = datetime(2026, 3, 25, 14, 30, 0, tzinfo=timezone.utc)  # after checkpoint

    repo = _FakeRepo()
    repo.snapshot_at = checkpoint_at
    repo.snapshot = AccountSnapshot(
        cash_krw=10_000_000.0,
        total_equity_krw=12_000_000.0,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=2.0, avg_price_krw=100_000.0, market_price_krw=120_000.0),
        },
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": checkpoint_at,
            "cash_krw": 5_000_000.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}, {"ticker": "MRVL", "quantity": 1}],
            "source": "test",
        },
        "gemini": {
            "agent_id": "gemini",
            "event_id": "chk_gemini",
            "checkpoint_at": checkpoint_at,
            "cash_krw": 5_000_000.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}],
            "source": "test",
        },
    }

    repo.broker_trade_rows = [
        {
            "event_id": "bt_mrvl_sell",
            "occurred_at": trade_at,
            "broker_order_id": "0030558700",
            "ticker": "MRVL",
            "side": "SELL",
            "quantity": 1.0,
            "price_krw": 120_000.0,
            "status": "FILLED",
        },
    ]
    repo.filled_execution_rows = [
        {
            "order_id": "0030558700",
            "created_at": trade_at,
            "agent_id": "gpt",
            "ticker": "MRVL",
            "side": "SELL",
            "filled_qty": 1.0,
            "status": "FILLED",
        },
    ]

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt", "gemini"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    fallback_issues = [i for i in result.issues if i.issue_type == "execution_report_fallback_applied"]
    assert len(fallback_issues) == 0
