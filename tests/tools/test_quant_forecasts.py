from __future__ import annotations

from datetime import datetime, timezone
import math
from types import SimpleNamespace
from typing import Literal, get_args, get_origin, get_type_hints

import pytest

from arena.config import Settings
from arena.tools.quant_tools import QuantTools

from tests.tools.quant_helpers import (
    _literal_args,
    _stable_quant_tool_now,
    FakeRepo,
    FakeOpenTradingClient,
    _settings,
)

def test_forecast_returns_reads_predictions() -> None:
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=_settings())
    rows = qt.forecast_returns()
    assert len(rows) >= 1
    assert rows[0]["run_date"]
    assert "ticker" in rows[0]
    assert "exp_return_period" in rows[0]
    assert all(r["ticker"] in {"AAPL", "MSFT", "TSLA", "PLTD"} for r in rows)
    assert repo.last_forecast_mode == "all"


def test_forecast_returns_prefers_dynamic_candidate_tickers_from_context() -> None:
    class _Repo(FakeRepo):
        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

    repo = _Repo()
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=repo, settings=settings)
    context = {
        "target_market": "nasdaq",
        "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
        "_candidate_tickers": ["TSLA"],
    }
    qt.set_context(context)
    context["_candidate_tickers"] = ["PLTD", "TSLA"]

    rows = qt.forecast_returns()

    assert set(repo.last_forecast_tickers) == {"AAPL", "PLTD", "TSLA"}
    assert {row["ticker"] for row in rows} == {"AAPL", "PLTD", "TSLA"}


def test_forecast_returns_prefers_opportunity_working_set_over_raw_candidate_list() -> None:
    class _Repo(FakeRepo):
        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

    repo = _Repo()
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=repo, settings=settings)
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
            "_candidate_tickers": ["TSLA"],
            "opportunity_working_set": [{"ticker": "PLTD", "status": "pending"}],
        }
    )

    rows = qt.forecast_returns()

    assert set(repo.last_forecast_tickers) == {"AAPL", "PLTD"}
    assert {row["ticker"] for row in rows} == {"AAPL", "PLTD"}


def test_forecast_returns_prefers_full_discovered_basket_over_working_set() -> None:
    class _Repo(FakeRepo):
        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

    repo = _Repo()
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=repo, settings=settings)
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
            "_candidate_tickers": ["TSLA"],
            "_discovered_candidate_tickers": ["PLTD", "TSLA", "MSFT"],
            "opportunity_working_set": [{"ticker": "PLTD", "status": "pending"}],
        }
    )

    rows = qt.forecast_returns()

    assert set(repo.last_forecast_tickers) == {"AAPL", "PLTD", "TSLA", "MSFT"}
    assert {row["ticker"] for row in rows} == {"AAPL", "PLTD", "TSLA", "MSFT"}


def test_forecast_returns_compacts_model_rows_by_ticker() -> None:
    repo = FakeRepo()
    repo._preds = [
        {
            "run_date": "2026-03-12",
            "ticker": "AAPL",
            "exp_return_period": 0.1257810646,
            "forecast_horizon": 20,
            "forecast_model": "iTransformer",
            "is_stacked": False,
            "forecast_score": -0.0325,
            "prob_up": 1.0,
            "model_votes_up": 4,
            "model_votes_total": 4,
            "consensus": "STRONG_BUY",
        },
        {
            "run_date": "2026-03-12",
            "ticker": "AAPL",
            "exp_return_period": 0.0777303991,
            "forecast_horizon": 20,
            "forecast_model": "PatchTST",
            "is_stacked": False,
            "forecast_score": -0.0317,
            "prob_up": 1.0,
            "model_votes_up": 4,
            "model_votes_total": 4,
            "consensus": "STRONG_BUY",
        },
        {
            "run_date": "2026-03-12",
            "ticker": "AAPL",
            "exp_return_period": 0.0697741524,
            "forecast_horizon": 20,
            "forecast_model": "ensemble_avg",
            "is_stacked": True,
            "forecast_score": -0.0307,
            "prob_up": 1.0,
            "model_votes_up": 4,
            "model_votes_total": 4,
            "consensus": "STRONG_BUY",
        },
        {
            "run_date": "2026-03-12",
            "ticker": "AAPL",
            "exp_return_period": 0.0683418329,
            "forecast_horizon": 20,
            "forecast_model": "ensemble_wmae",
            "is_stacked": True,
            "forecast_score": -0.0304,
            "prob_up": 1.0,
            "model_votes_up": 4,
            "model_votes_total": 4,
            "consensus": "STRONG_BUY",
        },
    ]
    qt = QuantTools(repo=repo, settings=_settings())

    rows = qt.forecast_returns(tickers=["AAPL"])

    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "AAPL"
    assert row["forecast_model"] == "ensemble_wmae"
    assert row["is_stacked"] is True
    assert row["consensus"] == "STRONG_BUY"
    assert row["model_votes_up"] == 4
    assert row["model_votes_total"] == 4
    assert len(row["stacked_models"]) == 2
    assert len(row["base_models"]) == 2
    assert row["best_base_model"] == "iTransformer"
    assert row["best_base_return"] == 0.1257810646


def test_forecast_returns_forwards_mode_setting() -> None:
    settings = _settings()
    settings.forecast_mode = "stacked"
    settings.forecast_table = "my_proj.llm_arena.predicted_expected_returns"
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=settings)
    _ = qt.forecast_returns()
    assert repo.last_forecast_mode == "stacked"
    assert repo.last_forecast_table == "my_proj.llm_arena.predicted_expected_returns"


def test_forecast_returns_invalid_mode_falls_back_to_default_mode() -> None:
    settings = _settings()
    settings.forecast_mode = "all"
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=settings)

    rows = qt.forecast_returns(tickers=["AAPL"], forecast_mode="balanced")

    assert rows
    assert repo.last_forecast_mode == "all"


def test_forecast_returns_auto_build_retries_with_relaxed_config(monkeypatch) -> None:
    repo = FakeRepo()
    repo._preds = []
    qt = QuantTools(repo=repo, settings=_settings())
    monkeypatch.setenv("ARENA_FORECAST_AUTO_BUILD", "true")

    attempts: list[dict] = []

    def _fake_build(repo_obj, settings_obj, **cfg):
        _ = settings_obj
        attempts.append(dict(cfg))
        if len(attempts) == 1:
            return SimpleNamespace(rows_written=0, tickers_used=0, used_neuralforecast=False, note="insufficient series length")
        repo_obj._preds = [{"run_date": "2026-02-24", "ticker": "AAPL", "exp_return_period": 0.02, "forecast_horizon": 20}]
        return SimpleNamespace(rows_written=1, tickers_used=1, used_neuralforecast=False, note="ok")

    monkeypatch.setattr("arena.forecasting.build_and_store_stacked_forecasts", _fake_build)

    rows = qt.forecast_returns(tickers=["AAPL"])

    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert len(attempts) == 2
    assert int(attempts[0]["min_series_length"]) == 160
    assert int(attempts[1]["min_series_length"]) == 90


def test_forecast_returns_returns_empty_when_all_auto_build_attempts_fail(monkeypatch) -> None:
    from arena.tools import quant_tools as _qt_mod
    _qt_mod._forecast_built_dates.clear()

    repo = FakeRepo()
    repo._preds = []
    qt = QuantTools(repo=repo, settings=_settings())
    monkeypatch.setenv("ARENA_FORECAST_AUTO_BUILD", "true")

    attempts: list[dict] = []

    def _fake_build(repo_obj, settings_obj, **cfg):
        _ = (repo_obj, settings_obj)
        attempts.append(dict(cfg))
        return SimpleNamespace(rows_written=0, tickers_used=0, used_neuralforecast=False, note="insufficient series length")

    monkeypatch.setattr("arena.forecasting.build_and_store_stacked_forecasts", _fake_build)

    rows = qt.forecast_returns(tickers=["AAPL"])

    assert rows == []
    assert len(attempts) == 3


def test_forecast_returns_returns_empty_when_auto_build_disabled(monkeypatch) -> None:
    repo = FakeRepo()
    repo._preds = []
    qt = QuantTools(repo=repo, settings=_settings())
    monkeypatch.delenv("ARENA_FORECAST_AUTO_BUILD", raising=False)

    def _should_not_call(self):
        raise AssertionError("auto-build should not run when disabled")

    monkeypatch.setattr(QuantTools, "_auto_build_forecasts_if_needed", _should_not_call)
    rows = qt.forecast_returns(tickers=["AAPL", "MSFT"])

    assert rows == []
