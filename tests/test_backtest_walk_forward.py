from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from arena.backtest.walk_forward import WalkForwardConfig, stabilize_weights, walk_forward_backtest


def test_stabilize_weights_caps_delta() -> None:
    prev = {"AAPL": 0.9, "MSFT": 0.1}
    new = {"AAPL": 0.0, "MSFT": 1.0}

    w = stabilize_weights(new, prev, alpha=1.0, max_delta=0.05, hysteresis_abs=0.0)

    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["AAPL"] == 0.85
    assert abs(w["MSFT"] - 0.15) < 1e-12


def test_stabilize_weights_hysteresis_band() -> None:
    prev = {"AAPL": 0.5, "MSFT": 0.5}
    new = {"AAPL": 0.51, "MSFT": 0.49}

    w = stabilize_weights(new, prev, alpha=1.0, max_delta=1.0, hysteresis_abs=0.02)

    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert w["AAPL"] == prev["AAPL"]
    assert w["MSFT"] == prev["MSFT"]


def test_walk_forward_backtest_runs_min_vol() -> None:
    idx = pd.bdate_range("2024-01-02", periods=140)
    rng = np.random.default_rng(123)

    returns = pd.DataFrame(
        rng.normal(loc=0.0004, scale=0.015, size=(len(idx), 3)),
        index=idx,
        columns=["AAPL", "MSFT", "NVDA"],
    )

    cfg = WalkForwardConfig(
        start=idx[25].date(),
        end=idx[-1].date(),
        lookback_days=20,
        min_obs=15,
        rebalance_freq="W-FRI",
        fee_bps=5.0,
        smooth_alpha=0.5,
        max_weight_delta=0.2,
        hysteresis_abs=0.0,
    )

    nav_df, alloc_df, summary = walk_forward_backtest(returns, config=cfg, strategy="min_vol")

    assert not nav_df.empty
    assert set(nav_df.columns) == {"nav_date", "nav", "daily_return", "cum_return", "drawdown"}

    assert nav_df["nav"].iloc[0] > 0
    assert nav_df["nav"].iloc[-1] > 0

    assert summary["strategy"] == "min_vol"
    assert summary["start"] == cfg.start.isoformat()
    assert summary["end"] == cfg.end.isoformat()

    # Drawdown is defined <= 0.
    assert float(nav_df["drawdown"].max()) <= 1e-9

    # Allocations are optional, but when present should have required columns.
    assert set(alloc_df.columns) == {"rebalance_date", "ticker", "weight", "turnover", "cost_ratio"}


def test_walk_forward_backtest_rejects_unknown_strategy() -> None:
    idx = pd.bdate_range("2024-01-02", periods=80)
    returns = pd.DataFrame(0.001, index=idx, columns=["AAPL", "MSFT", "NVDA"])

    cfg = WalkForwardConfig(start=idx[10].date(), end=idx[-1].date(), lookback_days=10, min_obs=5)

    try:
        walk_forward_backtest(returns, config=cfg, strategy="does_not_exist")
    except ValueError as exc:
        assert "unsupported strategy" in str(exc)
    else:
        raise AssertionError("expected ValueError")
