from __future__ import annotations

from arena.tools.quant_tools import QuantTools

from tests.tools.quant_helpers import _stable_quant_tool_now, FakeRepo, _settings


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
