from __future__ import annotations

from types import SimpleNamespace

from arena.tools.quant_tools import QuantTools

from tests.tools.quant_helpers import _stable_quant_tool_now, FakeRepo, _settings


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
