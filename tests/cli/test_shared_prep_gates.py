from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import arena.cli as cli
import pytest
from arena.config import load_settings

from tests.cli.helpers import _FakeRepo, _stub_shared_prep_environment


def test_cmd_run_shared_prep_fast_aborts_when_artifacts_stale(monkeypatch) -> None:
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

    _stub_shared_prep_environment(
        monkeypatch,
        settings,
        _Repo(),
        calls,
        session_ready=(False, {"reason": "no_session", "market": "us"}),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="agent-us", stage="fast"
        )

    assert exc_info.value.code == 3
    stages = [c[0] for c in calls]
    assert "sync" not in stages, "fast gate must abort BEFORE sync when session is not ready"
    assert "dispatch" not in stages, "fast stage must not dispatch when slow session is not ready"
    assert not any(c[0] == "marker" for c in calls), "no marker on aborted fast run"


def test_cmd_run_shared_prep_fast_aborts_when_quote_sync_zero_rows(monkeypatch) -> None:
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

    _stub_shared_prep_environment(
        monkeypatch,
        settings,
        _Repo(),
        calls,
        sync_result=SimpleNamespace(inserted_rows=0, attempted_tickers=405, failed_tickers=["AAPL"]),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="agent-us", stage="fast"
        )

    assert exc_info.value.code == 7
    stages = [c[0] for c in calls]
    assert "sync" in stages, "fast stage must attempt quote sync before zero-row abort"
    assert "dispatch" not in stages, "fast stage must not dispatch when quote sync wrote zero rows"
    assert not any(c[0] == "marker" for c in calls), "no fast marker on zero-row abort"


def test_cmd_run_shared_prep_slow_records_non_ok_when_forecast_empty(monkeypatch) -> None:
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

    # Simulate build_and_store_stacked_forecasts writing zero rows (e.g., all
    # upstream data missing). Ranker still returns 'ok' with scores, but the
    # combined readiness must be not-ok because forecast was empty.
    _stub_shared_prep_environment(
        monkeypatch, settings, _Repo(), calls,
        forecast_rows=0, ranker_scores=5, ranker_status="ok",
    )

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="slow"
    )

    marker_entries = [c for c in calls if c[0] == "marker"]
    assert marker_entries, "slow stage must always record a marker"
    # status must NOT be 'ok' because forecast_rows_written == 0
    assert all(c[2] != "ok" for c in marker_entries), (
        f"forecast=0 must downgrade slow marker status; got {marker_entries}"
    )


def test_cmd_run_shared_prep_fast_without_dispatch_still_runs_gate(monkeypatch) -> None:
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

    # Same-session marker missing; the fast gate must fire even though
    # dispatch_job is empty (manual/operator invocation path).
    _stub_shared_prep_environment(
        monkeypatch, settings, _Repo(), calls,
        session_ready=(False, {"reason": "no_session", "market": "us"}),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="", stage="fast"
        )

    assert exc_info.value.code == 3
    stages = [c[0] for c in calls]
    assert "sync" not in stages, "fast gate must abort BEFORE sync even without dispatch_job"
    assert "dispatch" not in stages
    assert not any(c[0] == "marker" for c in calls)


def test_cmd_run_shared_prep_slow_aborts_on_same_day_intraday_quote(monkeypatch) -> None:
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
    # Override taint check to report an intraday quote row already present.
    monkeypatch.setattr(
        run_pipeline_mod,
        "_same_day_quote_rows_present",
        lambda *args, **kwargs: (True, {"count": 1, "market": "us"}),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="", stage="slow"
        )

    assert exc_info.value.code == 4
    stages = [c[0] for c in calls]
    # ML must not run, no marker must be recorded.
    assert "forecast" not in stages, "slow must not run forecast when intraday quote is present"
    assert "ranker" not in stages
    assert not any(c[0] == "marker" for c in calls)


def test_cmd_run_shared_prep_rejects_multi_market_config(monkeypatch) -> None:
    """Shared-prep is single-market: mixed KIS_TARGET_MARKET (e.g., 'us,kospi')
    must be rejected so readiness markers and taint checks cannot silently
    ignore one of the markets.
    """
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us,kospi"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="", dispatch_job="agent-x", stage="all"
        )

    assert exc_info.value.code == 5
    stages = [c[0] for c in calls]
    assert "sync" not in stages, "multi-market reject must happen before any sync"
    assert "forecast" not in stages
    assert "ranker" not in stages
    assert "dispatch" not in stages


def test_cmd_run_shared_prep_all_refuses_dispatch_on_partial_status(monkeypatch) -> None:
    """stage='all' must fail-closed when its own prep status is not 'ok'.

    The previous flow recorded a 'partial' marker and then dispatched the
    agent anyway — a known-bad prep would still launch live trading.
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

    # forecast_rows=0 downgrades status to 'partial' even though ranker
    # claims ok. stage='all' must refuse to dispatch in that case.
    _stub_shared_prep_environment(
        monkeypatch, settings, _Repo(), calls,
        forecast_rows=0, ranker_scores=5, ranker_status="ok",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="agent-us", stage="all"
        )

    assert exc_info.value.code == 6
    # Marker should have been recorded (so operators see the failure reason),
    # but dispatch must NOT have happened.
    marker_entries = [c for c in calls if c[0] == "marker"]
    assert any(c[2] != "ok" for c in marker_entries), (
        f"expected a non-ok marker; got {marker_entries}"
    )
    assert "dispatch" not in [c[0] for c in calls]


def test_cmd_run_shared_prep_slow_aborts_when_upstream_stale(monkeypatch) -> None:
    """When daily EOD data is far behind (e.g., feed broken for weeks),
    refuse before training the ranker — otherwise it silently learns on
    stale prices and the fast gate cannot detect that.
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
    monkeypatch.setattr(
        run_pipeline_mod,
        "_upstream_market_freshness",
        lambda *args, **kwargs: (False, {
            "reason": "stale_daily",
            "market": "us",
            "age_days": 27,
            "threshold_days": 5,
        }),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="", stage="slow"
        )

    assert exc_info.value.code == 7
    stages = [c[0] for c in calls]
    assert "daily_sync" in stages, "daily sync must have been attempted"
    assert "forecast" not in stages, "forecast must not run when upstream is stale"
    assert "ranker" not in stages
    assert not any(c[0] == "marker" for c in calls)


def test_cmd_run_shared_prep_slow_aborts_when_daily_sync_fails(monkeypatch) -> None:
    """If MarketDataSyncService raises, slow must abort before ML."""
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

    def _boom_factory(**kwargs):
        class _S:
            def sync_market_features(self_inner):
                calls.append(("daily_sync_attempted", None))
                raise RuntimeError("KIS api down")

        return _S()

    monkeypatch.setattr(cli, "MarketDataSyncService", _boom_factory)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="", stage="slow"
        )

    assert exc_info.value.code == 8
    stages = [c[0] for c in calls]
    assert "daily_sync_attempted" in stages
    assert "forecast" not in stages
    assert "ranker" not in stages


def test_cmd_run_shared_prep_slow_refuses_ml_on_seed_phase(monkeypatch) -> None:
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

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls, phase="seed")

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="slow"
    )

    stages = [c[0] for c in calls]
    # ML must still be refused on seed, but the slow path MUST run a daily
    # EOD sync so the feed can bootstrap out of the seed state. Without this,
    # an empty/sparse deployment would deadlock — slow can never populate
    # daily rows and phase would stay 'seed' forever.
    assert "sync" not in stages, "no quote sync on seed"
    assert "daily_sync" in stages, "seed+slow must run daily EOD sync to bootstrap"
    assert "forecast" not in stages, "seed must still refuse ML"
    assert "fundamentals" not in stages
    assert "ranker" not in stages
    assert "dispatch" not in stages
