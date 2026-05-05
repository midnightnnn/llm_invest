from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import arena.cli as cli
import pytest
from arena.config import load_settings

from tests.cli.helpers import _FakeRepo, _stub_shared_prep_environment


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
