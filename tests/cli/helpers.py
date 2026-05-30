from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace

import arena.cli as cli
import pytest
from arena.config import load_settings

class _FakeRepo:
    def __init__(
        self,
        *,
        row: dict | None = None,
        rows_by_tenant: dict[str, dict] | None = None,
        tenants: list[str] | None = None,
        latest_config_map: dict[str, str] | None = None,
        latest_agents_config_map: dict[str, str] | None = None,
    ) -> None:
        self._row = row
        self._rows_by_tenant = {
            str(key).strip().lower(): dict(value)
            for key, value in dict(rows_by_tenant or {}).items()
            if str(key).strip()
        }
        self._tenants = list(tenants or [])
        self._latest_config_map = {
            str(key).strip().lower(): str(value)
            for key, value in dict(latest_config_map or {}).items()
            if str(key).strip()
        }
        self._latest_agents_config_map = {
            str(key).strip().lower(): str(value)
            for key, value in dict(latest_agents_config_map or {}).items()
            if str(key).strip()
        }
        self.latest_tenant: str | None = None
        self.run_status_rows: list[dict[str, object]] = []

    def latest_runtime_credentials(self, *, tenant_id: str) -> dict | None:
        self.latest_tenant = tenant_id
        tenant = str(tenant_id or "").strip().lower()
        if tenant in self._rows_by_tenant:
            return dict(self._rows_by_tenant[tenant])
        return self._row

    def list_runtime_tenants(self, *, limit: int = 200) -> list[str]:
        _ = limit
        return list(self._tenants)

    def latest_config_values(self, *, config_key: str, tenant_ids: list[str] | None = None) -> dict[str, str]:
        ids = [str(token).strip().lower() for token in (tenant_ids or []) if str(token).strip()]
        if config_key == "kis_target_market":
            return {tenant: self._latest_config_map.get(tenant, "") for tenant in ids}
        if config_key == "agents_config":
            return {tenant: self._latest_agents_config_map.get(tenant, "") for tenant in ids}
        return {tenant: "" for tenant in ids}

    def append_tenant_run_status(self, **kwargs) -> None:
        self.run_status_rows.append(dict(kwargs))


def _stub_shared_prep_environment(
    monkeypatch,
    settings,
    repo,
    calls: list,
    *,
    phase: str = "open_cycle",
    session_ready: tuple[bool, dict] = (True, {"session_id": "sp_test"}),
    forecast_rows: int = 42,
    ranker_scores: int = 7,
    ranker_status: str = "ok",
    sync_result: object | None = None,
) -> None:
    from arena.cli_commands import run_pipeline as run_pipeline_mod

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "_apply_tenant_runtime_credentials", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_batch_phase", lambda *args, **kwargs: (phase, None))
    effective_sync_result = sync_result
    if effective_sync_result is None:
        effective_sync_result = SimpleNamespace(inserted_rows=11, attempted_tickers=11, failed_tickers=[])

    def _fake_batch_market_sync(*args, **kwargs):
        _ = (args, kwargs)
        calls.append(("sync", None))
        return effective_sync_result

    monkeypatch.setattr(cli, "_batch_market_sync", _fake_batch_market_sync)

    def _fake_forecast(args):
        calls.append(("forecast", args.horizon))
        return SimpleNamespace(rows_written=forecast_rows, run_date="2026-03-13", tickers_used=10, used_neuralforecast=True, model_names=["nbeatsx"], note="")

    def _fake_ranker(args):
        calls.append(("ranker", args.horizon))
        return SimpleNamespace(
            status=ranker_status,
            ranker_version="v-test",
            training_rows=100,
            validation_rows=10,
            scoring_rows=50,
            scores_written=ranker_scores,
            oos_ic_20d=0.1,
            oos_hit_rate_20d=0.55,
            note="",
        )

    monkeypatch.setattr(cli, "cmd_build_forecasts", _fake_forecast)
    monkeypatch.setattr(cli, "cmd_refresh_fundamentals_derived", lambda args: calls.append(("fundamentals", args.lookback_days)))
    monkeypatch.setattr(cli, "cmd_build_opportunity_ranker", _fake_ranker)
    monkeypatch.setattr(cli, "_dispatch_agent_job", lambda settings, job_name: calls.append(("dispatch", job_name)))
    monkeypatch.setattr(
        run_pipeline_mod,
        "_shared_prep_session_ready",
        lambda *args, **kwargs: session_ready,
    )
    monkeypatch.setattr(
        run_pipeline_mod,
        "_record_shared_prep_session",
        lambda *args, **kwargs: calls.append(("marker", kwargs.get("stage"), kwargs.get("status"))),
    )
    # Default: no same-day intraday taint so stage=slow/all tests proceed.
    # Individual tests can override this to exercise the abort path.
    monkeypatch.setattr(
        run_pipeline_mod,
        "_same_day_quote_rows_present",
        lambda *args, **kwargs: (False, {"count": 0}),
    )
    # Default: upstream daily feed fresh. Tests can override for stale abort.
    monkeypatch.setattr(
        run_pipeline_mod,
        "_upstream_market_freshness",
        lambda *args, **kwargs: (True, {"age_days": 0}),
    )
    monkeypatch.setattr(
        run_pipeline_mod,
        "_refresh_macro_indicators_for_prep",
        lambda *args, **kwargs: calls.append(("macro", kwargs.get("stage", ""))),
    )

    class _FakeMarketSyncResult:
        inserted_rows = 10
        attempted_tickers = 10
        failed_tickers: list = []

    def _fake_market_service_factory(**kwargs):
        service_settings = kwargs.get("settings")

        class _S:
            def sync_market_features(self_inner):
                calls.append(("daily_sync", None))
                return _FakeMarketSyncResult()

            def sync_market_features_for_tickers(self_inner, tickers):
                calls.append(("held_coverage", getattr(service_settings, "kis_target_market", ""), tuple(tickers)))
                return _FakeMarketSyncResult()

        return _S()

    monkeypatch.setattr(cli, "MarketDataSyncService", _fake_market_service_factory)
    monkeypatch.setattr(
        cli,
        "nasdaq_window",
        lambda now=None: SimpleNamespace(
            now_local=SimpleNamespace(weekday=lambda: 4),
            trading_date=date(2026, 3, 13),
        ),
    )
    monkeypatch.setattr(cli, "is_nasdaq_holiday", lambda d: False)
