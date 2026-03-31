from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from arena.tools.allocation import (
    TRADING_DAYS,
    optimize_blend,
    optimize_forecast_sharpe,
    optimize_hrp,
    optimize_max_sharpe,
    optimize_min_vol,
)


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Walk-forward backtest configuration.

    - Training window: fixed-length lookback (trading days) ending at rebalance_date-1.
    - Execution window: from rebalance_date to the end of the current rebalance period.

    Costs are modeled as: cost_ratio = fee_bps/10000 * sum(|w_new - w_prev|).
    """

    start: date
    end: date

    lookback_days: int = 60
    rebalance_freq: str = "W-FRI"  # pandas offset alias (e.g. W-FRI, M, Q)

    min_obs: int = 60

    # Stabilization (approximation of PortfolioP's smoothing / hysteresis)
    smooth_alpha: float = 0.30
    max_weight_delta: float = 0.05
    hysteresis_abs: float = 0.01

    # Trading frictions
    fee_bps: float = 10.0

    # Strategy params
    risk_free_rate: float = 0.04  # annual
    mu_confidence: float = 0.30
    sharpe_ratio: float = 0.60
    hrp_ratio: float = 0.40


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    total = 0.0
    for k, v in (weights or {}).items():
        key = str(k).strip().upper()
        if not key:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x):
            continue
        x = max(0.0, x)
        cleaned[key] = x
        total += x

    if total <= 0.0:
        # If everything is zero, return empty so the caller can interpret as cash.
        return {}

    return {k: float(v) / total for k, v in cleaned.items() if v > 0}


def stabilize_weights(
    new_weights: dict[str, float],
    prev_weights: dict[str, float] | None,
    *,
    alpha: float,
    max_delta: float,
    hysteresis_abs: float,
) -> dict[str, float]:
    """Stabilizes weights to reduce turnover.

    This is a lightweight approximation of PortfolioP's stabilization.
    """

    new_w = _normalize_weights(new_weights)
    prev_w = _normalize_weights(prev_weights or {})

    if not prev_w:
        return new_w
    if not new_w:
        return prev_w

    a = max(0.0, min(float(alpha), 1.0))
    md = max(0.0, float(max_delta))
    hy = max(0.0, float(hysteresis_abs))

    keys = sorted(set(prev_w) | set(new_w))
    blended: dict[str, float] = {}

    for k in keys:
        p = float(prev_w.get(k, 0.0))
        n = float(new_w.get(k, 0.0))
        v = (1.0 - a) * p + a * n

        # Cap per-step change.
        v = max(p - md, min(p + md, v))

        # Hysteresis band: ignore tiny changes.
        if abs(v - p) < hy:
            v = p

        blended[k] = max(0.0, float(v))

    return _normalize_weights(blended)


def make_period_cuts(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """Returns a list of segment end timestamps aligned to existing index values."""
    if index.empty:
        return []

    idx = pd.DatetimeIndex(index).sort_values()
    s = pd.Series(idx, index=idx)

    cuts: list[pd.Timestamp] = []
    try:
        for _, group in s.groupby(pd.Grouper(freq=str(freq))):
            if group.empty:
                continue
            cuts.append(pd.Timestamp(group.iloc[-1]))
    except Exception:
        # Fallback: one segment only.
        cuts = [pd.Timestamp(idx[-1])]

    last = pd.Timestamp(idx[-1])
    if not cuts:
        cuts = [last]
    elif cuts[-1] != last:
        cuts.append(last)

    # Uniq + sorted
    return sorted(set(pd.to_datetime(cuts)))


def _strategy_weights(
    strategy: str,
    tickers: list[str],
    train_daily_returns: np.ndarray,
    *,
    cfg: WalkForwardConfig,
    predicted_mu: dict[str, float] | None,
) -> dict[str, float]:
    key = str(strategy or "").strip().lower()

    if key in {"max_sharpe", "max-sharpe", "mpt", "markowitz"}:
        return optimize_max_sharpe(tickers, train_daily_returns, risk_free_rate=cfg.risk_free_rate).weights

    if key in {"min_vol", "min-vol", "minimum_vol", "minimum-vol"}:
        return optimize_min_vol(tickers, train_daily_returns, risk_free_rate=cfg.risk_free_rate).weights

    if key in {"hrp"}:
        return optimize_hrp(tickers, train_daily_returns, risk_free_rate=cfg.risk_free_rate).weights

    if key in {"blend", "blend60_40", "blend_60_40", "blend_sharpe_hrp"}:
        return optimize_blend(
            tickers,
            train_daily_returns,
            risk_free_rate=cfg.risk_free_rate,
            sharpe_ratio=cfg.sharpe_ratio,
            hrp_ratio=cfg.hrp_ratio,
        ).weights

    if key in {"forecast", "forecast_max_sharpe", "fcast", "fcast_max_sharpe"}:
        mu = predicted_mu or {}
        if not mu:
            # Without forecasts, fall back to max-sharpe.
            return optimize_max_sharpe(tickers, train_daily_returns, risk_free_rate=cfg.risk_free_rate).weights
        return optimize_forecast_sharpe(
            tickers,
            train_daily_returns,
            mu,
            risk_free_rate=cfg.risk_free_rate,
            mu_confidence=cfg.mu_confidence,
        ).weights

    raise ValueError(f"unsupported strategy: {strategy}")


def walk_forward_backtest(
    returns: pd.DataFrame,
    *,
    config: WalkForwardConfig,
    strategy: str,
    predicted_mu: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Runs a fixed-lookback, walk-forward backtest for one allocation strategy.

    Parameters
    - returns: daily returns, index is datetime-like, columns are tickers.
      The DataFrame may contain history prior to config.start for training.

    Returns
    - nav_df: columns [nav_date, nav, daily_return, cum_return, drawdown]
    - alloc_df: columns [rebalance_date, ticker, weight, turnover, cost_ratio]
    - summary: dict with total_return, mdd, annual_vol, etc.
    """

    if returns is None or returns.empty:
        nav_df = pd.DataFrame(columns=["nav_date", "nav", "daily_return", "cum_return", "drawdown"])
        alloc_df = pd.DataFrame(columns=["rebalance_date", "ticker", "weight", "turnover", "cost_ratio"])
        return nav_df, alloc_df, {"strategy": strategy, "error": "empty returns"}

    rets = returns.copy()
    rets.index = pd.to_datetime(rets.index)
    rets = rets.sort_index()

    start_ts = pd.Timestamp(config.start)
    end_ts = pd.Timestamp(config.end)
    sim = rets.loc[(rets.index >= start_ts) & (rets.index <= end_ts)]

    if sim.empty:
        nav_df = pd.DataFrame(columns=["nav_date", "nav", "daily_return", "cum_return", "drawdown"])
        alloc_df = pd.DataFrame(columns=["rebalance_date", "ticker", "weight", "turnover", "cost_ratio"])
        return nav_df, alloc_df, {"strategy": strategy, "error": "no rows in simulation window"}

    cuts = make_period_cuts(pd.DatetimeIndex(sim.index), config.rebalance_freq)
    if not cuts:
        cuts = [pd.Timestamp(sim.index[-1])]

    lookback = max(2, int(config.lookback_days))
    min_obs = max(2, int(config.min_obs))

    nav = 1.0
    nav_parts: list[pd.Series] = []
    alloc_rows: list[dict[str, Any]] = []

    w_prev: dict[str, float] = {}

    start_pos = 0
    idx_sim = pd.DatetimeIndex(sim.index)

    for cut in cuts:
        if start_pos >= len(idx_sim):
            break

        try:
            end_pos = int(idx_sim.get_loc(cut))
        except KeyError:
            # If cut is not present, advance to the next available date.
            end_pos = len(idx_sim) - 1

        seg = sim.iloc[start_pos : end_pos + 1]
        if seg.empty:
            start_pos = end_pos + 1
            continue

        rebalance_ts = pd.Timestamp(seg.index[0])

        # Training window ends on rebalance_ts - 1 (previous trading row in rets).
        try:
            rets_loc = int(rets.index.get_loc(rebalance_ts))
        except KeyError:
            # Should not happen if sim is a slice of rets, but keep safe.
            rets_loc = int(rets.index.searchsorted(rebalance_ts))

        train_end = rets_loc - 1
        if train_end < 0:
            w_new = w_prev
        else:
            train_start = max(0, train_end - lookback + 1)
            train = rets.iloc[train_start : train_end + 1]

            # Keep tickers with enough observations.
            valid_cols = train.notna().sum(axis=0) >= min_obs
            train = train.loc[:, valid_cols]
            train = train.fillna(0.0)

            tickers = [str(c).strip().upper() for c in train.columns if str(c).strip()]
            if len(tickers) >= 2 and len(train) >= min_obs:
                daily = train.to_numpy(dtype=float)
                try:
                    w_raw = _strategy_weights(strategy, tickers, daily, cfg=config, predicted_mu=predicted_mu)
                except ValueError:
                    raise
                except Exception:
                    w_raw = {}

                if w_raw:
                    w_new = stabilize_weights(
                        w_raw,
                        w_prev,
                        alpha=config.smooth_alpha,
                        max_delta=config.max_weight_delta,
                        hysteresis_abs=config.hysteresis_abs,
                    )
                else:
                    w_new = w_prev
            else:
                w_new = w_prev

        # Turnover + transaction cost at rebalance.
        keys = set(w_prev) | set(w_new)
        turnover = float(sum(abs(float(w_new.get(k, 0.0)) - float(w_prev.get(k, 0.0))) for k in keys))
        fee = max(0.0, float(config.fee_bps)) / 10000.0
        cost_ratio = fee * turnover if w_prev else 0.0
        nav *= max(0.0, 1.0 - cost_ratio)

        if w_new:
            for tkr, wt in w_new.items():
                if float(wt) <= 1e-12:
                    continue
                alloc_rows.append(
                    {
                        "rebalance_date": rebalance_ts.date().isoformat(),
                        "ticker": tkr,
                        "weight": float(wt),
                        "turnover": turnover,
                        "cost_ratio": cost_ratio,
                    }
                )

        # Apply weights across this segment.
        if w_new:
            active_cols = [c for c in seg.columns if str(c).strip().upper() in w_new]
            if active_cols:
                w_vec = np.array([float(w_new[str(c).strip().upper()]) for c in active_cols], dtype=float)
                w_sum = float(np.sum(w_vec))
                if w_sum > 0:
                    w_vec = w_vec / w_sum
                seg_ret = seg[active_cols].fillna(0.0).to_numpy(dtype=float) @ w_vec
                daily_ret = pd.Series(seg_ret, index=seg.index)
            else:
                daily_ret = pd.Series(0.0, index=seg.index)
        else:
            daily_ret = pd.Series(0.0, index=seg.index)

        seg_nav = (1.0 + daily_ret).cumprod() * nav
        if not seg_nav.empty:
            nav = float(seg_nav.iloc[-1])
            nav_parts.append(seg_nav)

        w_prev = dict(w_new)
        start_pos = end_pos + 1

    if not nav_parts:
        nav_df = pd.DataFrame(columns=["nav_date", "nav", "daily_return", "cum_return", "drawdown"])
        alloc_df = pd.DataFrame(alloc_rows, columns=["rebalance_date", "ticker", "weight", "turnover", "cost_ratio"])
        return nav_df, alloc_df, {"strategy": strategy, "error": "no segments"}

    nav_series = pd.concat(nav_parts)
    nav_series = nav_series[~nav_series.index.duplicated(keep="first")]

    nav_df = pd.DataFrame({"nav_date": nav_series.index.date, "nav": nav_series.values})
    nav_df["daily_return"] = pd.Series(nav_series.values).pct_change().fillna(0.0).values
    nav_df["cum_return"] = (nav_df["nav"] / float(nav_df["nav"].iloc[0])) - 1.0

    peak = pd.Series(nav_df["nav"]).cummax()
    nav_df["drawdown"] = (pd.Series(nav_df["nav"]) / peak) - 1.0

    alloc_df = pd.DataFrame(alloc_rows, columns=["rebalance_date", "ticker", "weight", "turnover", "cost_ratio"])

    total_return = float(nav_df["cum_return"].iloc[-1]) if not nav_df.empty else 0.0
    mdd = float(nav_df["drawdown"].min()) if not nav_df.empty else 0.0

    n_days = max(1, int(len(nav_df)))
    cagr = (1.0 + total_return) ** (float(TRADING_DAYS) / float(n_days)) - 1.0 if total_return > -0.999 else -1.0

    vol = float(pd.Series(nav_df["daily_return"]).std(ddof=0) * math.sqrt(float(TRADING_DAYS))) if n_days > 2 else 0.0

    summary: dict[str, Any] = {
        "strategy": str(strategy),
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "rebalance_freq": str(config.rebalance_freq),
        "lookback_days": int(config.lookback_days),
        "fee_bps": float(config.fee_bps),
        "total_return": total_return,
        "cagr": float(cagr),
        "annual_vol": vol,
        "max_drawdown": mdd,
        "rebalance_count": int(len(alloc_df["rebalance_date"].unique())) if not alloc_df.empty else 0,
        "avg_turnover": float(alloc_df["turnover"].mean()) if not alloc_df.empty else 0.0,
        "avg_cost_ratio": float(alloc_df["cost_ratio"].mean()) if not alloc_df.empty else 0.0,
    }

    return nav_df, alloc_df, summary
