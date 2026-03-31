from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.config import Settings
from arena.models import AccountSnapshot, Position
from arena.reconciliation import StateReconciliationService, StateRecoveryService


class _FakeRepo:
    def __init__(self) -> None:
        self.dataset_fqn = "proj.ds"
        self.tenant_id = "local"
        self.snapshot_at = datetime(2026, 3, 12, 1, 0, tzinfo=timezone.utc)
        self.snapshot: AccountSnapshot | None = None
        self.sleeve_configs: dict[str, dict] = {}
        self.checkpoint_configs: dict[str, dict] = {}
        self.broker_trade_rows: list[dict] = []
        self.filled_execution_rows: list[dict] = []
        self.manual_adjustment_rows: list[dict] = []
        self.manual_cash_adjustment_rows: list[dict] = []
        self.broker_cash_event_rows: list[dict] = []
        self.agent_snapshots: dict[str, AccountSnapshot] = {}
        self.reconciliation_runs: list[dict] = []
        self.reconciliation_issues: list[dict] = []
        self.ensure_calls: list[dict] = []
        self.ensure_checkpoint_calls: list[dict] = []

    def latest_account_snapshot(self, *, tenant_id=None):
        _ = tenant_id
        return self.snapshot

    def latest_agent_sleeves(self, *, agent_ids, tenant_id=None):
        _ = tenant_id
        return {agent_id: self.sleeve_configs[agent_id] for agent_id in agent_ids if agent_id in self.sleeve_configs}

    def latest_agent_state_checkpoints(self, *, agent_ids, tenant_id=None):
        _ = tenant_id
        return {
            agent_id: self.checkpoint_configs[agent_id]
            for agent_id in agent_ids
            if agent_id in self.checkpoint_configs
        }

    def ensure_agent_sleeves(self, *, agent_ids, total_cash_krw, capital_per_agent=None, tenant_id=None, initialized_at=None):
        _ = (tenant_id, initialized_at)
        self.ensure_calls.append(
            {
                "agent_ids": list(agent_ids),
                "total_cash_krw": float(total_cash_krw),
                "capital_per_agent": dict(capital_per_agent) if capital_per_agent else None,
            }
        )
        ts = self.snapshot_at
        for agent_id in agent_ids:
            self.sleeve_configs.setdefault(
                agent_id,
                {
                    "agent_id": agent_id,
                    "initialized_at": ts,
                    "initial_cash_krw": 1_000_000.0,
                    "initial_positions_json": "[]",
                },
            )
        return self.latest_agent_sleeves(agent_ids=agent_ids)

    def ensure_agent_state_checkpoints(self, *, agent_ids, total_cash_krw, capital_per_agent=None, tenant_id=None, checkpoint_at=None):
        _ = (tenant_id, checkpoint_at)
        self.ensure_checkpoint_calls.append(
            {
                "agent_ids": list(agent_ids),
                "total_cash_krw": float(total_cash_krw),
                "capital_per_agent": dict(capital_per_agent) if capital_per_agent else None,
            }
        )
        ts = self.snapshot_at
        for agent_id in agent_ids:
            self.checkpoint_configs.setdefault(
                agent_id,
                {
                    "agent_id": agent_id,
                    "event_id": f"chk_{agent_id}",
                    "checkpoint_at": ts,
                    "cash_krw": 1_000_000.0,
                    "positions_json": [],
                    "source": "ensure",
                },
            )
        return self.latest_agent_state_checkpoints(agent_ids=agent_ids)

    def append_agent_state_checkpoints(self, rows, *, tenant_id=None):
        _ = tenant_id
        for row in rows:
            agent_id = str(row.get("agent_id") or "").strip()
            if agent_id:
                self.checkpoint_configs[agent_id] = dict(row)

    def broker_trade_events_since(self, *, since, tenant_id=None, statuses=None):
        _ = (tenant_id, statuses)
        return [row for row in self.broker_trade_rows if row["occurred_at"] >= since]

    def filled_execution_reports_since(self, *, since, tenant_id=None):
        _ = tenant_id
        return [row for row in self.filled_execution_rows if row["created_at"] >= since]

    def manual_position_adjustments_since(self, *, since, tenant_id=None):
        _ = tenant_id
        return [row for row in self.manual_adjustment_rows if row["occurred_at"] >= since]

    def manual_cash_adjustments_since(self, *, agent_id, since, tenant_id=None):
        _ = tenant_id
        return [
            row
            for row in self.manual_cash_adjustment_rows
            if row["occurred_at"] >= since and str(row.get("agent_id") or "").strip() == str(agent_id).strip()
        ]

    def broker_cash_events_since(self, *, since, tenant_id=None):
        _ = tenant_id
        return [row for row in self.broker_cash_event_rows if row["occurred_at"] >= since]

    def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True, tenant_id=None):
        _ = (sources, include_simulated, tenant_id)
        snapshot = self.agent_snapshots.get(
            str(agent_id),
            AccountSnapshot(cash_krw=0.0, total_equity_krw=0.0, positions={}),
        )
        return snapshot, float(snapshot.total_equity_krw), {"agent_id": str(agent_id), "seed_source": "test"}

    def append_reconciliation_run(self, **kwargs):
        self.reconciliation_runs.append(dict(kwargs))

    def append_reconciliation_issues(self, rows, *, tenant_id=None):
        _ = tenant_id
        self.reconciliation_issues.extend(dict(row) for row in rows)

    def fetch_rows(self, sql: str, params=None):
        _ = params
        if "FROM `proj.ds.account_snapshots`" in sql:
            return [{"snapshot_at": self.snapshot_at}] if self.snapshot is not None else []
        return []


def _settings() -> Settings:
    return Settings(
        google_cloud_project="p",
        bq_dataset="d",
        bq_location="loc",
        agent_ids=["gpt", "gemini"],
        agent_mode="adk",
        base_currency="KRW",
        sleeve_capital_krw=1_000_000.0,
        log_level="INFO",
        log_format="rich",
        trading_mode="live",
        kis_order_endpoint="",
        kis_api_key="k",
        kis_api_secret="s",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no="12345678",
        kis_account_product_code="01",
        kis_account_key_suffix="",
        kis_env="real",
        kis_target_market="nasdaq",
        kis_overseas_quote_excd="NAS",
        kis_overseas_order_excd="NASD",
        kis_us_natn_cd="840",
        kis_us_tr_mket_cd="01",
        kis_secret_name="",
        kis_secret_version="latest",
        kis_http_timeout_seconds=20,
        kis_http_max_retries=0,
        kis_http_backoff_base_seconds=0.1,
        kis_http_backoff_max_seconds=0.1,
        kis_confirm_fills=False,
        kis_confirm_timeout_seconds=25,
        kis_confirm_poll_seconds=2.0,
        usd_krw_rate=1460.0,
        market_sync_history_days=60,
        max_order_krw=350_000.0,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=0.35,
        min_cash_buffer_ratio=0.10,
        ticker_cooldown_seconds=120,
        max_daily_orders=0,
        estimated_fee_bps=10.0,
        context_max_board_posts=24,
        context_max_memory_events=32,
        context_max_market_rows=64,
        openai_api_key="",
        openai_model="gpt-5.2",
        gemini_api_key="",
        gemini_model="gemini-3-pro-preview",
        research_gemini_model="gemini-2.5-flash",
        llm_timeout_seconds=25,
        default_universe=["AAPL"],
        allow_live_trading=False,
    )


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


def test_reconciliation_bootstraps_checkpoints_from_legacy_sleeves_for_backward_compatibility() -> None:
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
