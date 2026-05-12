from __future__ import annotations

from datetime import date

import pytest

from arena.cli_commands.local_demo import cmd_seed_local_demo
from arena.config import load_settings
from arena.data.local.repository import LocalRepository
from arena.forecasting import stacked as stacked_mod
from arena.recommendation import build_and_store_opportunity_ranker


pytest.importorskip("duckdb")


def _repo(tmp_path, monkeypatch) -> LocalRepository:
    db_path = tmp_path / "arena.duckdb"
    monkeypatch.setenv("ARENA_MODE", "local")
    monkeypatch.setenv("ARENA_LOCAL_DB_PATH", str(db_path))
    monkeypatch.setenv("KIS_TARGET_MARKET", "us")
    cmd_seed_local_demo(days=120)
    repo = LocalRepository(tenant_id="local", db_path=str(db_path), settings=load_settings())
    repo.ensure_tables()
    return repo


def test_local_forecast_store_methods_fallback_from_live_sources(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    try:
        tickers = repo.latest_universe_candidate_tickers(limit=10, markets=["us"])
        assert {"AAPL", "MSFT"}.issubset(set(tickers))

        frame = repo.get_daily_close_frame(
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 1),
            end=date(2100, 1, 1),
            sources=["open_trading_us"],
        )
        assert not frame.empty
        assert {"AAPL", "MSFT"}.issubset(set(frame.columns))

        written = repo.replace_predicted_returns(
            [
                {
                    "run_date": date.today(),
                    "ticker": "AAPL",
                    "exp_return_period": 0.04,
                    "forecast_horizon": 20,
                    "forecast_model": "ensemble_wmae",
                    "is_stacked": True,
                    "prob_up": 0.67,
                    "model_votes_up": 4,
                    "model_votes_total": 6,
                    "consensus": "MODEL_UP",
                }
            ],
            run_date=date.today(),
        )
        assert written == 1
        rows = repo.get_predicted_returns(tickers=["AAPL"], mode="stacked", staleness_days=30)
        assert rows and rows[0]["ticker"] == "AAPL"
    finally:
        repo.session.close()


def test_local_signal_ranker_and_prep_markers(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    try:
        settings = load_settings()
        settings.arena_mode = "local"
        settings.kis_target_market = "us"
        monkeypatch.setattr(
            stacked_mod,
            "_require_forecasting_dependencies",
            lambda: (_ for _ in ()).throw(RuntimeError("missing test forecasting deps")),
        )

        forecast = stacked_mod.build_and_store_stacked_forecasts(
            repo,
            settings,
            lookback_days=100,
            horizon=5,
            min_series_length=80,
        )
        assert forecast.rows_written > 0

        result = build_and_store_opportunity_ranker(
            repo,
            settings,
            lookback_days=100,
            horizon_days=5,
            max_scoring_rows=20,
            min_ic_dates=10,
            min_valid_signals=3,
        )

        assert result.status == "ok"
        assert result.scores_written > 0
        learned = repo.latest_opportunity_ranker_scores(limit=5, markets=["us"], max_age_hours=24 * 14)
        assert learned

        inserted = repo.insert_shared_prep_session(
            {
                "session_id": "sp_test",
                "market": "us",
                "trading_date": date.today(),
                "stage": "all",
                "status": "ok",
                "forecast_rows_written": 1,
                "ranker_scores_written": result.scores_written,
                "detail_json": {"test": True},
            }
        )
        assert inserted == 1
        session = repo.get_latest_shared_prep_session(market="us", trading_date=date.today(), stage="all")
        assert session and session["status"] == "ok"
    finally:
        repo.session.close()
