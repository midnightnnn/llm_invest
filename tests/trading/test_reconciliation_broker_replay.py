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

def test_reconciliation_replays_broker_trade_events_since_seed() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=500_000.0,
        positions={"AAPL": Position(ticker="AAPL", quantity=2.0, avg_price_krw=100_000.0, market_price_krw=120_000.0)},
    )
    repo.sleeve_configs = {
        "gpt": {
            "agent_id": "gpt",
            "initialized_at": repo.snapshot_at,
            "initial_cash_krw": 0.0,
            "initial_positions_json": '[{"ticker":"AAPL","quantity":1}]',
        },
    }
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}],
            "source": "test",
        },
    }
    repo.broker_trade_rows = [
        {
            "event_id": "evt_buy",
            "occurred_at": repo.snapshot_at,
            "broker_order_id": "ORDER-1",
            "ticker": "AAPL",
            "side": "BUY",
            "quantity": 1.0,
        }
    ]
    repo.filled_execution_rows = [
        {
            "order_id": "ORDER-1",
            "created_at": repo.snapshot_at,
            "ticker": "AAPL",
            "side": "BUY",
            "filled_qty": 1.0,
        }
    ]

    result = StateReconciliationService(settings=_settings(), repo=repo, cash_reconciliation_enabled=True).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert result.summary["ledger_ticker_count"] == 1
    assert result.summary["seed_source"] == "agent_state_checkpoints"


def test_reconciliation_excludes_external_broker_carry_without_ai_evidence() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=0.0,
        total_equity_krw=1_000_000.0,
        positions={"PLTD": Position(ticker="PLTD", quantity=197.0, avg_price_krw=1.0, market_price_krw=1.0)},
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [],
            "source": "test",
        },
    }

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert any(issue.issue_type == "external_broker_position_excluded" for issue in result.issues)
    assert all(issue.issue_type != "position_quantity_mismatch" for issue in result.issues)
    assert repo.reconciliation_runs[0]["summary"]["external_carry_ticker_count"] == 1


def test_reconciliation_excludes_unmatched_broker_trade_delta_as_external() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=0.0,
        total_equity_krw=500_000.0,
        positions={"AAPL": Position(ticker="AAPL", quantity=2.0, avg_price_krw=100_000.0, market_price_krw=120_000.0)},
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}],
            "source": "test",
        },
    }
    repo.broker_trade_rows = [
        {
            "event_id": "evt_manual",
            "occurred_at": repo.snapshot_at,
            "broker_order_id": "MANUAL-1",
            "ticker": "AAPL",
            "side": "BUY",
            "quantity": 1.0,
        }
    ]

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert any(issue.issue_type == "external_broker_trade_excluded" for issue in result.issues)
    assert all(issue.issue_type != "position_quantity_mismatch" for issue in result.issues)
    assert repo.reconciliation_runs[0]["summary"]["external_trade_ticker_count"] == 1


def test_reconciliation_excludes_broker_overlap_above_ai_ledger_as_external() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=0.0,
        total_equity_krw=500_000.0,
        positions={"AAPL": Position(ticker="AAPL", quantity=2.0, avg_price_krw=100_000.0, market_price_krw=120_000.0)},
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}],
            "source": "test",
        },
    }

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert any(issue.issue_type == "external_broker_position_overlap_excluded" for issue in result.issues)
    assert all(issue.issue_type != "position_quantity_mismatch" for issue in result.issues)
    assert repo.reconciliation_runs[0]["summary"]["external_overlap_ticker_count"] == 1


def test_reconciliation_flags_checkpoint_seed_timestamp_mismatch() -> None:
    repo = _FakeRepo()
    older = datetime(2026, 3, 11, 23, 59, tzinfo=timezone.utc)
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=500_000.0,
        positions={"AAPL": Position(ticker="AAPL", quantity=2.0, avg_price_krw=100_000.0, market_price_krw=120_000.0)},
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": older,
            "cash_krw": 0.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}],
            "source": "test",
        },
        "gemini": {
            "agent_id": "gemini",
            "event_id": "chk_gemini",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}],
            "source": "test",
        },
    }

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt", "gemini"],
        tenant_id="midnightnnn",
    )

    assert result.ok is False
    assert any(issue.issue_type == "checkpoint_seed_timestamp_mismatch" for issue in result.issues)


def test_reconciliation_default_does_not_bootstrap_checkpoints_from_legacy_sleeves() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=300_000.0,
        positions={"AAPL": Position(ticker="AAPL", quantity=1.0, avg_price_krw=100_000.0, market_price_krw=120_000.0)},
    )
    repo.sleeve_configs = {
        "gpt": {
            "agent_id": "gpt",
            "initialized_at": repo.snapshot_at,
            "initial_cash_krw": 50_000.0,
            "initial_positions_json": '[{"ticker":"AAPL","quantity":1}]',
        },
    }

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
        auto_recover=True,
    )

    assert result.ok is True
    assert result.status == "recovered"
    assert "ensure_agent_state_checkpoints" in result.recoveries
    assert "bootstrap_agent_state_checkpoints" not in result.recoveries
    assert repo.checkpoint_configs["gpt"]["source"] == "ensure"
    assert repo.reconciliation_runs[0]["summary"]["seed_source"] == "agent_state_checkpoints"


def test_reconciliation_can_explicitly_bootstrap_checkpoints_from_legacy_sleeves() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=300_000.0,
        positions={"AAPL": Position(ticker="AAPL", quantity=1.0, avg_price_krw=100_000.0, market_price_krw=120_000.0)},
    )
    repo.sleeve_configs = {
        "gpt": {
            "agent_id": "gpt",
            "initialized_at": repo.snapshot_at,
            "initial_cash_krw": 50_000.0,
            "initial_positions_json": '[{"ticker":"AAPL","quantity":1}]',
        },
    }

    result = StateReconciliationService(
        settings=_settings(),
        repo=repo,
        allow_legacy_sleeve_seed=True,
    ).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
        auto_recover=True,
    )

    assert result.ok is True
    assert result.status == "recovered"
    assert "bootstrap_agent_state_checkpoints" in result.recoveries
    assert repo.checkpoint_configs["gpt"]["source"] == "legacy_agent_sleeve"
    assert repo.reconciliation_runs[0]["summary"]["seed_source"] == "agent_state_checkpoints"


def test_reconciliation_reports_broker_cash_unallocated_as_warning() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=300_000.0,
        total_equity_krw=500_000.0,
        positions={},
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [],
            "source": "test",
        },
    }
    repo.agent_snapshots = {
        "gpt": AccountSnapshot(cash_krw=100_000.0, total_equity_krw=100_000.0, positions={}),
    }

    result = StateReconciliationService(settings=_settings(), repo=repo, cash_reconciliation_enabled=True).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert any(issue.issue_type == "broker_cash_unallocated" for issue in result.issues)
    assert result.summary["broker_cash_krw"] == pytest.approx(300_000.0)
    assert result.summary["derived_agent_cash_krw"] == pytest.approx(100_000.0)
    assert result.summary["unallocated_cash_krw"] == pytest.approx(200_000.0)


def test_reconciliation_warns_when_agent_cash_exceeds_broker_cash() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=100_000.0,
        positions={},
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [],
            "source": "test",
        },
    }
    repo.agent_snapshots = {
        "gpt": AccountSnapshot(cash_krw=150_000.0, total_equity_krw=150_000.0, positions={}),
    }

    result = StateReconciliationService(settings=_settings(), repo=repo, cash_reconciliation_enabled=True).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    issue = next(issue for issue in result.issues if issue.issue_type == "broker_cash_overallocated")
    assert issue.severity == "warning"


def test_reconciliation_allows_small_cash_overallocation_within_tolerance() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=100_000.0,
        positions={},
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [],
            "source": "test",
        },
    }
    repo.agent_snapshots = {
        "gpt": AccountSnapshot(cash_krw=100_500.0, total_equity_krw=100_500.0, positions={}),
    }

    result = StateReconciliationService(
        settings=_settings(),
        repo=repo,
        cash_reconciliation_enabled=True,
        cash_tolerance_krw=1_000.0,
    ).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert any(issue.issue_type == "broker_cash_overallocated" for issue in result.issues)
    assert any(issue.severity == "warning" for issue in result.issues)
    assert result.issues[0].detail["within_tolerance"] is True
    assert repo.reconciliation_runs[0]["summary"]["cash_tolerance_krw"] == pytest.approx(1_000.0)


def test_reconciliation_marks_inferred_cash_coverage_in_summary_and_issue_detail() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=300_000.0,
        total_equity_krw=300_000.0,
        positions={},
    )
    repo.checkpoint_configs = {
        "gpt": {
            "agent_id": "gpt",
            "event_id": "chk_gpt",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [],
            "source": "test",
        },
    }
    repo.agent_snapshots = {
        "gpt": AccountSnapshot(cash_krw=100_000.0, total_equity_krw=100_000.0, positions={}),
    }
    repo.broker_cash_event_rows = [
        {
            "event_id": "cash_1",
            "occurred_at": repo.snapshot_at,
            "currency": "KRW",
            "amount_krw": 50_000.0,
            "source": "account_cash_history_residual",
            "raw_payload_json": {"inferred": True},
        }
    ]

    result = StateReconciliationService(settings=_settings(), repo=repo, cash_reconciliation_enabled=True).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.summary["cash_event_basis"] == "inferred_only"
    assert result.summary["inferred_cash_event_count"] == 1
    issue = next(issue for issue in result.issues if issue.issue_type == "broker_cash_unallocated")
    assert issue.detail is not None
    assert issue.detail["cash_event_basis"] == "inferred_only"
    assert issue.detail["inferred_cash_event_count"] == 1
