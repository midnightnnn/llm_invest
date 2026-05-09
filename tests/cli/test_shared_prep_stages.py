from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import arena.cli as cli
import pytest
from arena.config import load_settings

from tests.cli.helpers import _FakeRepo, _stub_shared_prep_environment


def test_cmd_run_shared_prep_dispatches_agent_job(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    repo = _Repo()

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "_apply_tenant_runtime_credentials", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_batch_phase", lambda *args, **kwargs: ("open_cycle", None))
    monkeypatch.setattr(cli, "_batch_market_sync", lambda *args, **kwargs: calls.append(("sync", None)))

    def _forecast_fn(args):
        calls.append(("forecast", args.horizon))
        return SimpleNamespace(
            rows_written=42, run_date="2026-03-13", tickers_used=10,
            used_neuralforecast=True, model_names=["nbeatsx"], note="",
        )

    def _ranker_fn(args):
        calls.append(("ranker", args.horizon))
        return SimpleNamespace(
            status="ok", ranker_version="v", training_rows=100,
            validation_rows=10, scoring_rows=50, scores_written=7,
            oos_ic_20d=0.1, oos_hit_rate_20d=0.55, note="",
        )

    monkeypatch.setattr(cli, "cmd_build_forecasts", _forecast_fn)
    monkeypatch.setattr(cli, "cmd_build_opportunity_ranker", _ranker_fn)
    monkeypatch.setattr(cli, "_dispatch_agent_job", lambda settings, job_name: calls.append(("dispatch", job_name)))
    monkeypatch.setattr(
        run_pipeline_mod,
        "_same_day_quote_rows_present",
        lambda *args, **kwargs: (False, {"count": 0}),
    )
    monkeypatch.setattr(
        run_pipeline_mod,
        "_upstream_market_freshness",
        lambda *args, **kwargs: (True, {"age_days": 0}),
    )
    monkeypatch.setattr(
        cli,
        "nasdaq_window",
        lambda now=None: SimpleNamespace(
            now_local=SimpleNamespace(weekday=lambda: 4),
            trading_date=date(2026, 3, 13),
        ),
    )
    monkeypatch.setattr(cli, "is_nasdaq_holiday", lambda d: False)

    cli.cmd_run_shared_prep(live=True, market_override="us", dispatch_job="agent-us")

    assert ("sync", None) in calls
    assert ("forecast", 20) in calls
    assert ("ranker", 20) in calls
    assert ("dispatch", "agent-us") in calls


def test_cmd_run_shared_prep_slow_runs_only_ml_and_skips_dispatch(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="slow"
    )

    stages = [c[0] for c in calls]
    assert "sync" not in stages, "slow stage must skip sync-market"
    assert "forecast" in stages
    assert "fundamentals" in stages
    assert "ranker" in stages
    assert "dispatch" not in stages, "slow stage must not dispatch downstream agent"
    assert any(
        c[0] == "marker" and c[1] == "slow" and c[2] == "ok" for c in calls
    ), "slow stage must record an ok session marker"


def test_cmd_run_shared_prep_local_simulated_reuses_existing_market_data(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "local"
    settings.bq_dataset = "llm_arena"
    settings.bq_location = "local"
    settings.kis_target_market = "us"
    settings.arena_mode = "local"
    settings.distribution_mode = "simulated_only"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls, phase="general")

    cli.cmd_run_shared_prep(live=False, market_override="us", stage="all")

    stages = [c[0] for c in calls]
    assert "sync" not in stages
    assert "daily_sync" not in stages
    assert "forecast" in stages
    assert "ranker" in stages


def test_cmd_run_shared_prep_fast_runs_only_sync_and_dispatches(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="fast"
    )

    stages = [c[0] for c in calls]
    assert "sync" in stages
    assert "forecast" not in stages, "fast stage must skip ML forecast"
    assert "fundamentals" not in stages
    assert "ranker" not in stages
    assert ("dispatch", "agent-us") in calls
    assert any(
        c[0] == "marker" and c[1] == "fast" and c[2] == "ok" for c in calls
    ), "fast stage must record an ok session marker after dispatch"


def test_cmd_run_shared_prep_all_live_path_not_blocked_by_self_sync_quotes(monkeypatch) -> None:
    """stage='all' runs sync BEFORE ML on purpose. The taint guard must not
    misinterpret quote rows that this very invocation just wrote as external
    contamination; otherwise the legacy single-shot/live deploy path breaks.
    """
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)
    # Simulate the realistic condition: _batch_market_sync wrote today's
    # quote rows. A naive guard would treat these as taint and SystemExit(4).
    monkeypatch.setattr(
        run_pipeline_mod,
        "_same_day_quote_rows_present",
        lambda *args, **kwargs: (True, {"count": 500, "market": "us"}),
    )

    # No SystemExit expected — stage='all' legacy flow must complete.
    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="all"
    )

    stages = [c[0] for c in calls]
    assert "sync" in stages
    assert "forecast" in stages, "stage='all' must run ML even when same-day quotes exist"
    assert "ranker" in stages
    assert ("dispatch", "agent-us") in calls


def test_cmd_run_shared_prep_slow_runs_daily_sync_then_ml(monkeypatch) -> None:
    """Slow stage must trigger MarketDataSyncService.sync_market_features()
    BEFORE the ML steps so the daily EOD feed is refreshed — the live
    scheduler phases otherwise never populate daily rows.
    """
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="", stage="slow"
    )

    stages = [c[0] for c in calls]
    # Daily sync must run before forecast/ranker.
    assert "daily_sync" in stages
    daily_idx = stages.index("daily_sync")
    forecast_idx = stages.index("forecast")
    assert daily_idx < forecast_idx, (
        f"daily_sync must precede forecast: daily@{daily_idx} forecast@{forecast_idx}"
    )
    # Marker is ok because freshness default is fresh.
    assert any(c[0] == "marker" and c[2] == "ok" for c in calls)


def test_cmd_run_shared_prep_slow_syncs_account_held_coverage_before_ml(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            assert all_tenants is True
            if market == "us,kospi,kosdaq":
                return ["AAPL", "053580"]
            return ["AAPL"]

    repo = _Repo()
    _stub_shared_prep_environment(monkeypatch, settings, repo, calls)

    class _FakeMarketSyncResult:
        inserted_rows = 1
        attempted_tickers = 1
        failed_tickers: list = []

    def _fake_market_service_factory(**kwargs):
        service_settings = kwargs["settings"]

        class _S:
            def sync_market_features(self_inner):
                calls.append(("daily_sync", service_settings.kis_target_market))
                return _FakeMarketSyncResult()

            def sync_market_features_for_tickers(self_inner, tickers):
                calls.append(("held_coverage", service_settings.kis_target_market, tuple(tickers)))
                return _FakeMarketSyncResult()

        return _S()

    monkeypatch.setattr(cli, "MarketDataSyncService", _fake_market_service_factory)

    cli.cmd_run_shared_prep(live=True, market_override="us", dispatch_job="", stage="slow")

    stages = [c[0] for c in calls]
    assert "held_coverage" in stages
    assert stages.index("held_coverage") < stages.index("forecast")
    assert ("held_coverage", "us,kospi,kosdaq", ("AAPL", "053580")) in calls
