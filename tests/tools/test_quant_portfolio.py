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

def test_optimize_portfolio_sharpe() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="sharpe", lookback_days=20)
    assert out["strategy"] == "max_sharpe"
    w = out["weights"]
    assert set(w.keys()) == {"AAPL", "MSFT"}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert "backtest_mdd" in out
    assert out["backtest_mdd"]["value"] <= 0.0


def test_optimize_portfolio_risk_parity() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    assert out["strategy"] == "hrp"
    w = out["weights"]
    assert set(w.keys()) == {"AAPL", "MSFT"}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert "backtest_mdd" in out


def test_optimize_portfolio_forecast() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="forecast", lookback_days=20, mu_confidence=0.3)
    assert out["strategy"] == "forecast_max_sharpe"
    w = out["weights"]
    assert set(w.keys()) == {"AAPL", "MSFT"}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert "backtest_mdd" in out


def test_optimize_portfolio_forecast_heuristic_fallback() -> None:
    repo = FakeRepo()
    repo._preds = []  # no BQ forecast data
    qt = QuantTools(repo=repo, settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="forecast", lookback_days=20)
    # With zero forecast coverage, tool degrades to HRP instead of silently
    # running forecast_max_sharpe with empty predicted_mu.
    assert out["strategy"] == "hrp"
    assert out["status"] == "degraded"
    assert "forecast_coverage_insufficient" in out["degraded_reasons"]
    assert out["forecast_coverage"] == 0.0
    assert out["strategy_requested"] == "forecast"
    w = out["weights"]
    assert set(w.keys()) == {"AAPL", "MSFT"}
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_optimize_portfolio_invalid_strategy() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="invalid_xyz")
    assert "error" in out


def test_optimize_portfolio_partial_excludes_short_history() -> None:
    class _Repo(FakeRepo):
        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            out = super().get_daily_closes(tickers=tickers, lookback_days=lookback_days, sources=sources)
            out["TSLA"] = [100.0, 101.0, 102.0]  # insufficient (<10)
            return out

    qt = QuantTools(repo=_Repo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT", "TSLA"], strategy="risk_parity", lookback_days=20)
    assert out["status"] == "ok"
    assert out["data_quality"]["status"] == "partial"
    assert out["data_quality"]["usable_tickers"] == 2
    assert any(
        e["ticker"] == "TSLA" and e["reason"] == "insufficient_history"
        for e in out["data_quality"]["excluded"]
    )
    assert set(out["weights"].keys()) == {"AAPL", "MSFT"}


def test_optimize_portfolio_unusable_returns_graceful_error() -> None:
    class _Repo(FakeRepo):
        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            return {}  # no data for any ticker

    qt = QuantTools(repo=_Repo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="sharpe", lookback_days=20)
    assert out["status"] == "unusable"
    assert out["data_quality"]["status"] == "unusable"
    assert out["data_quality"]["usable_tickers"] == 0
    assert len(out["data_quality"]["excluded"]) == 2
    assert "error" in out


def test_optimize_portfolio_decision_summary_without_context_is_suggestion() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    ds = out["decision_summary"]
    assert ds["headline_code"] == "no_current_portfolio"
    assert ds["confidence"] == "low"
    assert ds["turnover"] == 0.0


def test_portfolio_weights_require_market_price_not_avg_cost() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    qt.set_context({
        "target_market": "nasdaq",
        "portfolio": {"positions": {"TSLA": {"quantity": 1.0, "avg_price_krw": 100.0}}},
    })

    weights, stock_mv, cash = qt._portfolio_weights()

    assert weights == {}
    assert stock_mv == 0.0
    assert cash == 0.0


def test_optimize_portfolio_decision_summary_rotate() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    qt.set_context({
        "target_market": "nasdaq",
        "portfolio": {"positions": {"TSLA": {"quantity": 1.0, "avg_price_krw": 100.0}}},
        "market_features": [{"ticker": "TSLA", "close_price_krw": 100.0}],
    })
    out = qt.optimize_portfolio(["AAPL", "MSFT", "TSLA"], strategy="risk_parity", lookback_days=20)
    ds = out["decision_summary"]
    # Starting from 100% TSLA, HRP spreads across AAPL/MSFT → rotate (both BUY and SELL).
    assert ds["headline_code"] == "rotate"
    assert ds["confidence"] in {"medium", "high"}
    assert ds["turnover"] > 0.03
    # Canonical vocabulary — no hype words.
    for bad in ("strong", "guaranteed", "best", "must"):
        assert bad not in ds["headline"].lower()


def test_optimize_portfolio_evidence_gaps_emitted() -> None:
    class _Repo(FakeRepo):
        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            out = super().get_daily_closes(tickers=tickers, lookback_days=lookback_days, sources=sources)
            out["TSLA"] = [100.0, 101.0]  # insufficient
            return out

    qt = QuantTools(repo=_Repo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT", "TSLA"], strategy="forecast", lookback_days=20)
    assert "some_tickers_excluded" in out.get("evidence_gaps", [])
    notes = out["validation_notes"]
    assert any("timing" in n.lower() for n in notes)


def test_optimize_portfolio_binding_forecast_preserves_forecast_basis() -> None:
    # Binding constraints on a forecast allocation must recompute stats on the
    # forecast-blended mu basis — not historical mu — so the reported
    # expected_return_daily/sharpe_daily stay coherent with the optimizer.
    import numpy as np
    from arena.tools.allocation import blend_forecast_mu, recompute_stats

    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=_settings())
    out = qt.optimize_portfolio(
        ["AAPL", "MSFT", "PLTD"],
        strategy="forecast",
        mu_confidence=1.0,
        lookback_days=20,
        max_weight=0.40,  # binding: PLTD would otherwise dominate
    )
    assert "constraints_applied" in out

    tickers = out["tickers"]
    closes = repo.get_daily_closes(tickers=tickers, lookback_days=21)
    aligned = np.stack([np.array(closes[t], dtype=float) for t in tickers], axis=1)
    rets = (aligned[1:] / aligned[:-1]) - 1.0
    predicted_mu = {p["ticker"]: p["exp_return_period"] for p in repo._preds}
    mu_blended = blend_forecast_mu(tickers, rets, predicted_mu, mu_confidence=1.0)

    exp_ret_forecast, vol_forecast, sharpe_forecast = recompute_stats(
        tickers, out["weights"], rets, mu_override=mu_blended,
    )
    exp_ret_hist, _, _ = recompute_stats(tickers, out["weights"], rets)

    # Output must match forecast-basis recompute, not historical-only.
    assert out["expected_return_daily"] == pytest.approx(exp_ret_forecast, abs=1e-6)
    assert out["volatility_daily"] == pytest.approx(vol_forecast, abs=1e-6)
    assert out["sharpe_daily"] == pytest.approx(sharpe_forecast, abs=1e-3)
    # Forecast vs historical basis differ materially in this fixture.
    assert abs(exp_ret_forecast - exp_ret_hist) > 1e-6


def test_optimize_portfolio_non_binding_constraint_preserves_forecast_stats() -> None:
    # When a constraint is passed but does not bind (e.g. max_weight=1.0),
    # weights must be unchanged AND stats must match the unconstrained call.
    # Matters most for strategy='forecast' whose mu blends historical + forecast.
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    base = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="forecast", lookback_days=20, mu_confidence=0.3)
    held = qt.optimize_portfolio(
        ["AAPL", "MSFT"], strategy="forecast", lookback_days=20, mu_confidence=0.3,
        max_weight=1.0,  # non-binding
    )
    assert held["weights"] == base["weights"]
    assert held["expected_return_daily"] == base["expected_return_daily"]
    assert held["sharpe_daily"] == base["sharpe_daily"]
    assert held["volatility_daily"] == base["volatility_daily"]
    assert "constraints_applied" not in held


def test_optimize_portfolio_cash_buffer_recomputes_stats() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    base = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    buffered = qt.optimize_portfolio(
        ["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20, cash_buffer=0.50,
    )
    # Weights scaled down, expected_return + volatility should scale accordingly.
    assert sum(buffered["weights"].values()) == pytest.approx(0.50, abs=1e-6)
    # Rounded to 6 decimals in the output — tolerate rounding noise.
    assert buffered["expected_return_daily"] == pytest.approx(base["expected_return_daily"] * 0.5, abs=1e-6)
    assert buffered["volatility_daily"] == pytest.approx(base["volatility_daily"] * 0.5, abs=1e-6)


def test_optimize_portfolio_min_weight_preserves_backtest_mdd() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(
        ["AAPL", "MSFT", "TSLA"],
        strategy="risk_parity",
        lookback_days=20,
        min_weight=0.40,  # drops at least one name
    )
    # Shape mismatch would silently drop backtest_mdd — assert it survives.
    assert "backtest_mdd" in out
    assert out["backtest_mdd"]["value"] <= 0.0


def test_optimize_portfolio_aligned_portfolio_headline_is_hold() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    # First compute the optimizer's target weights with no context.
    suggestion = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    target = suggestion["weights"]
    # Now set a portfolio exactly matching the target weights (quantity * avg_price = weight).
    qt.set_context({
        "target_market": "nasdaq",
        "market_features": [
            {"ticker": "AAPL", "close_price_krw": 1.0},
            {"ticker": "MSFT", "close_price_krw": 1.0},
        ],
        "portfolio": {"positions": {
            "AAPL": {"quantity": target["AAPL"] * 100.0, "avg_price_krw": 1.0},
            "MSFT": {"quantity": target["MSFT"] * 100.0, "avg_price_krw": 1.0},
        }},
    })
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    assert out["rebalance_orders"] == []
    assert out["decision_summary"]["headline_code"] == "hold"


def test_optimize_portfolio_single_usable_ticker() -> None:
    class _Repo(FakeRepo):
        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            out = super().get_daily_closes(tickers=tickers, lookback_days=lookback_days, sources=sources)
            out.pop("MSFT", None)  # only AAPL usable
            return out

    qt = QuantTools(repo=_Repo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="sharpe", lookback_days=20)
    assert out["status"] == "degraded"
    assert "single_usable_ticker" in out["degraded_reasons"]
    assert out["strategy"] == "single_name"
    assert out["weights"] == {"AAPL": 1.0}
    assert out["data_quality"]["usable_tickers"] == 1
