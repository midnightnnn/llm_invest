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

def test_recovery_rebuilds_checkpoints_from_current_state_and_reconciles() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=200_000.0,
        total_equity_krw=320_000.0,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=1.0, avg_price_krw=100_000.0, market_price_krw=120_000.0),
        },
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt_old",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [{"ticker": "MSFT", "quantity": 1.0}],
            "source": "stale",
        },
    }
    repo.agent_snapshots = {
        "gpt": AccountSnapshot(
            cash_krw=200_000.0,
            total_equity_krw=320_000.0,
            positions={
                "AAPL": Position(ticker="AAPL", quantity=1.0, avg_price_krw=100_000.0, market_price_krw=120_000.0),
            },
        )
    }

    result = StateRecoveryService(settings=_settings(), repo=repo).recover_and_reconcile(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert result.applied_checkpoints == 1
    assert result.after.ok is True
    assert repo.checkpoint_configs["gpt"]["source"] == "recovery_rebuild"


def test_recovery_can_skip_checkpoint_rebuild() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=200_000.0,
        total_equity_krw=320_000.0,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=1.0, avg_price_krw=100_000.0, market_price_krw=120_000.0),
        },
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt_old",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [{"ticker": "MSFT", "quantity": 1.0}],
            "source": "stale",
        },
    }

    result = StateRecoveryService(settings=_settings(), repo=repo).recover_and_reconcile(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
        allow_checkpoint_rebuild=False,
    )

    assert result.ok is False
    assert result.applied_checkpoints == 0
    assert result.after is result.before
    assert result.recoveries == ["checkpoint_rebuild_disabled"]
    assert repo.checkpoint_configs["gpt"]["source"] == "stale"
