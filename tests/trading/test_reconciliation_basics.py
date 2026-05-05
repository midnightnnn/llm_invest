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

def test_cash_event_coverage_summary_logs_structured_failure(caplog) -> None:
    class BrokenCashRepo(_FakeRepo):
        def broker_cash_events_since(self, *, since, tenant_id=None):
            _ = (since, tenant_id)
            raise RuntimeError("coverage boom")

    service = StateReconciliationService(settings=_settings(), repo=BrokenCashRepo())

    with caplog.at_level("WARNING"):
        summary = service._cash_event_coverage_summary(
            since=datetime(2026, 3, 12, 1, 0, tzinfo=timezone.utc),
            tenant_id="midnightnnn",
        )

    assert summary["cash_event_load_error"] == "coverage boom"
    record = next(item for item in caplog.records if getattr(item, "event", "") == "broker_cash_coverage_load_skipped")
    assert record.tenant_id == "midnightnnn"
    assert record.stage == "cash_event_coverage"
    assert record.err_type == "RuntimeError"
    assert record.err == "coverage boom"


def test_reconciliation_ok_with_excluded_ticker() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=300_000.0,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=2.0, avg_price_krw=100_000.0, market_price_krw=120_000.0),
            "PLTD": Position(ticker="PLTD", quantity=197.0, avg_price_krw=1.0, market_price_krw=1.0),
        },
    )
    repo.sleeve_configs = {
        "gpt": {
            "agent_id": "gpt",
            "initialized_at": repo.snapshot_at,
            "initial_cash_krw": 0.0,
            "initial_positions_json": '[{"ticker":"AAPL","quantity":1}]',
        },
        "gemini": {
            "agent_id": "gemini",
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
        "gemini": {
            "agent_id": "gemini",
            "event_id": "chk_gemini",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [{"ticker": "AAPL", "quantity": 1}],
            "source": "test",
        },
    }

    result = StateReconciliationService(settings=_settings(), repo=repo, excluded_tickers=["PLTD"]).reconcile_positions(
        agent_ids=["gpt", "gemini"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert result.status == "ok"
    assert result.issues == []
    assert repo.reconciliation_runs[0]["status"] == "ok"


def test_reconciliation_prefers_checkpoint_ensure_over_legacy_sleeves() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=2_000_000.0,
        total_equity_krw=2_000_000.0,
        positions={},
    )

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt", "gemini"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert repo.ensure_checkpoint_calls
    assert repo.ensure_calls == []
    assert "ensure_agent_state_checkpoints" in result.recoveries
    assert repo.reconciliation_runs[0]["summary"]["excluded_tickers"] == []
    assert repo.reconciliation_runs[0]["summary"]["seed_source"] == "agent_state_checkpoints"
    assert repo.reconciliation_issues == []


def test_reconciliation_prefers_checkpoint_seed_over_legacy_sleeves() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=300_000.0,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=1.0, avg_price_krw=100_000.0, market_price_krw=120_000.0),
            "MSFT": Position(ticker="MSFT", quantity=1.0, avg_price_krw=100_000.0, market_price_krw=120_000.0),
        },
    )
    repo.sleeve_configs = {
        "gpt": {
            "agent_id": "gpt",
            "initialized_at": repo.snapshot_at,
            "initial_cash_krw": 0.0,
            "initial_positions_json": '[{"ticker":"AAPL","quantity":1}]',
        },
        "gemini": {
            "agent_id": "gemini",
            "initialized_at": repo.snapshot_at,
            "initial_cash_krw": 0.0,
            "initial_positions_json": '[{"ticker":"MSFT","quantity":1}]',
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
        "gemini": {
            "agent_id": "gemini",
            "event_id": "chk_gemini",
            "checkpoint_at": repo.snapshot_at,
            "cash_krw": 0.0,
            "positions_json": [{"ticker": "MSFT", "quantity": 1}],
            "source": "test",
        },
    }

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt", "gemini"],
        tenant_id="midnightnnn",
        auto_recover=True,
    )

    assert result.ok is True
    assert result.status == "ok"
    assert result.recoveries == []
    assert repo.ensure_calls == []
    assert repo.reconciliation_runs[0]["status"] == "ok"
    assert repo.reconciliation_runs[0]["summary"]["seed_source"] == "agent_state_checkpoints"


def test_reconciliation_records_position_shortfall_and_fails() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=300_000.0,
        positions={},
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

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is False
    assert result.status == "failed"
    assert len(result.issues) == 1
    assert result.issues[0].issue_type == "position_quantity_mismatch"
    assert repo.reconciliation_runs[0]["status"] == "failed"
    assert repo.reconciliation_issues[0]["issue_type"] == "position_quantity_mismatch"


def test_reconciliation_allows_position_mismatch_within_tolerance_as_warning() -> None:
    repo = _FakeRepo()
    repo.snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=300_000.0,
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

    result = StateReconciliationService(settings=_settings(), repo=repo, qty_tolerance=1.0).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
    )

    assert result.ok is True
    assert result.status == "ok"
    assert len(result.issues) == 1
    assert result.issues[0].issue_type == "position_quantity_mismatch"
    assert result.issues[0].severity == "warning"
    assert result.issues[0].detail["within_tolerance"] is True
    assert repo.reconciliation_runs[0]["summary"]["qty_tolerance"] == pytest.approx(1.0)


def test_reconciliation_uses_sync_callback_when_snapshot_missing() -> None:
    repo = _FakeRepo()
    repo.sleeve_configs = {
        "gpt": {
            "agent_id": "gpt",
            "initialized_at": repo.snapshot_at,
            "initial_cash_krw": 0.0,
            "initial_positions_json": "[]",
        },
    }
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

    def _sync_snapshot():
        repo.snapshot = AccountSnapshot(cash_krw=0.0, total_equity_krw=0.0, positions={})
        return repo.snapshot

    result = StateReconciliationService(settings=_settings(), repo=repo).reconcile_positions(
        agent_ids=["gpt"],
        tenant_id="midnightnnn",
        auto_recover=True,
        sync_account_snapshot=_sync_snapshot,
    )

    assert result.ok is True
    assert "sync_account_snapshot" in result.recoveries
