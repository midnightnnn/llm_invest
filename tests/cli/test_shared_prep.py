from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
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


def test_shared_prep_session_ready_accepts_all_marker(monkeypatch) -> None:
    """_shared_prep_session_ready must treat a 'stage=all' marker as valid so
    legacy single-shot runs can hand off to a subsequent fast invocation.
    """
    from datetime import datetime as _dt, timezone as _tz

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    class _Repo:
        def __init__(self) -> None:
            self.queried_stages: list[str] = []

        def get_latest_shared_prep_session(self, *, market, trading_date, stage):
            self.queried_stages.append(stage)
            if stage == "all":
                return {
                    "session_id": "sp_all_ok",
                    "market": market,
                    "trading_date": trading_date,
                    "stage": "all",
                    "status": "ok",
                    "forecast_rows_written": 100,
                    "ranker_scores_written": 25,
                    "created_at": _dt(2026, 3, 13, 5, 0, tzinfo=_tz.utc),
                }
            return None  # no slow marker

    repo = _Repo()
    ready, info = run_pipeline_mod._shared_prep_session_ready(
        repo, market="us", trading_date="2026-03-13"
    )

    assert ready is True, info
    assert info["session_id"] == "sp_all_ok"
    assert info["matched_stage"] == "all"
    assert set(repo.queried_stages) == {"slow", "all"}


def test_shared_prep_session_ready_prefers_newer_of_slow_and_all(monkeypatch) -> None:
    """When both markers exist, the most recent created_at wins."""
    from datetime import datetime as _dt, timezone as _tz

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    slow_row = {
        "session_id": "sp_slow_old",
        "stage": "slow",
        "status": "ok",
        "forecast_rows_written": 50,
        "ranker_scores_written": 10,
        "created_at": _dt(2026, 3, 13, 1, 0, tzinfo=_tz.utc),
    }
    all_row = {
        "session_id": "sp_all_new",
        "stage": "all",
        "status": "ok",
        "forecast_rows_written": 60,
        "ranker_scores_written": 20,
        "created_at": _dt(2026, 3, 13, 4, 30, tzinfo=_tz.utc),
    }

    class _Repo:
        def get_latest_shared_prep_session(self, *, market, trading_date, stage):
            return slow_row if stage == "slow" else all_row

    ready, info = run_pipeline_mod._shared_prep_session_ready(
        _Repo(), market="us", trading_date="2026-03-13"
    )

    assert ready is True
    assert info["session_id"] == "sp_all_new"
    assert info["matched_stage"] == "all"


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


def test_canonical_market_key_normalizes_us_aliases() -> None:
    from arena.cli_commands.run_pipeline import _canonical_market_key

    assert _canonical_market_key("us") == "us"
    assert _canonical_market_key("NASDAQ") == "us"
    assert _canonical_market_key("nyse") == "us"
    assert _canonical_market_key("AMEX") == "us"
    assert _canonical_market_key("kospi") == "kospi"
    assert _canonical_market_key("kosdaq") == "kospi"
    assert _canonical_market_key("KR") == "kospi"
    assert _canonical_market_key("us,kospi") == "us", "comma-separated picks first"
    assert _canonical_market_key("") == ""
    assert _canonical_market_key("unknown_venue") == "unknown_venue"


def test_trading_date_handles_us_aliases_without_utc_fallback(monkeypatch) -> None:
    """_trading_date_for_market must route nasdaq/nyse/amex to America/New_York.

    Previously the raw key path would silently fall back to UTC, so a late-evening
    US session could compute a different civil date and store/look up markers
    under the wrong day.
    """
    from datetime import date as _date
    from arena.cli_commands.run_pipeline import _trading_date_for_market

    results = {
        "us": _trading_date_for_market("us"),
        "nasdaq": _trading_date_for_market("nasdaq"),
        "nyse": _trading_date_for_market("nyse"),
        "amex": _trading_date_for_market("amex"),
        "kospi": _trading_date_for_market("kospi"),
        "kosdaq": _trading_date_for_market("kosdaq"),
    }

    for key, value in results.items():
        assert isinstance(value, _date), f"{key} must return a date, got {value!r}"
    # All US aliases must agree on the same civil date (same TZ).
    us_dates = {results[k] for k in ("us", "nasdaq", "nyse", "amex")}
    assert len(us_dates) == 1, f"US aliases diverged: {results}"
    kr_dates = {results[k] for k in ("kospi", "kosdaq")}
    assert len(kr_dates) == 1, f"KR aliases diverged: {results}"


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


def test_shared_prep_session_ready_isolates_us_subexchanges(monkeypatch) -> None:
    """Exchange-level isolation: a 'nasdaq' prep marker must NOT satisfy a
    'nyse' readiness check (and vice versa), because forecast/ranker prep
    scopes itself by the raw KIS_TARGET_MARKET token.
    """
    from datetime import datetime as _dt, timezone as _tz
    from arena.cli_commands import run_pipeline as run_pipeline_mod

    nasdaq_row = {
        "session_id": "sp_nasdaq",
        "market": "nasdaq",
        "stage": "slow",
        "status": "ok",
        "forecast_rows_written": 40,
        "ranker_scores_written": 10,
        "created_at": _dt(2026, 3, 13, 5, 0, tzinfo=_tz.utc),
    }

    class _Repo:
        def __init__(self) -> None:
            self.queries: list[dict[str, object]] = []

        def get_latest_shared_prep_session(self, *, market, trading_date, stage):
            self.queries.append({"market": market, "stage": stage})
            # Only return the nasdaq marker when queried for nasdaq.
            if market == "nasdaq":
                return nasdaq_row
            return None

    repo = _Repo()

    ready_nasdaq, info_nasdaq = run_pipeline_mod._shared_prep_session_ready(
        repo, market="nasdaq", trading_date="2026-03-13"
    )
    assert ready_nasdaq is True
    assert info_nasdaq["session_id"] == "sp_nasdaq"

    ready_nyse, info_nyse = run_pipeline_mod._shared_prep_session_ready(
        repo, market="nyse", trading_date="2026-03-13"
    )
    assert ready_nyse is False
    assert info_nyse["reason"] == "no_session"

    # Queries must have been market-scoped, not canonicalized to 'us'.
    queried_markets = {q["market"] for q in repo.queries}
    assert queried_markets == {"nasdaq", "nyse"}


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
