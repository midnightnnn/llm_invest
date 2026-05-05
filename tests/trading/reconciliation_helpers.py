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
