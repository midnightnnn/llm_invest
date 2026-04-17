from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.optimize import minimize
    from scipy.spatial.distance import squareform

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    leaves_list = None
    linkage = None
    squareform = None
    minimize = None
    _HAS_SCIPY = False


TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class AllocationResult:
    """Allocation optimization output (daily units)."""

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    strategy: str


def sanitize_cov(cov: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """Symmetrize + clean NaN/Inf + PSD-fix via eigenvalue clipping."""
    arr = np.asarray(cov, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("cov must be a square 2D matrix")
    if arr.size == 0:
        return arr

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = (arr + arr.T) / 2.0

    vals, vecs = np.linalg.eigh(arr)
    vals = np.clip(vals, float(eps), None)
    psd = vecs @ np.diag(vals) @ vecs.T
    psd = (psd + psd.T) / 2.0
    return psd


def _to_daily_rf(risk_free_rate: float, days: int = TRADING_DAYS) -> float:
    if days <= 0:
        return 0.0
    return (1.0 + float(risk_free_rate)) ** (1.0 / float(days)) - 1.0


def blend_forecast_mu(
    tickers: list[str],
    daily_returns: np.ndarray,
    predicted_mu: dict[str, float],
    *,
    mu_confidence: float,
    forecast_horizon: int = 20,
) -> np.ndarray:
    """Returns the blended mu vector used by optimize_forecast_sharpe.

    Exposed so callers can recompute stats on constrained weights using the
    same return basis the optimizer used, rather than historical-only mu.
    """
    x = np.asarray(daily_returns, dtype=float)
    mu_hist = np.nanmean(x, axis=0)
    conf = max(0.0, min(float(mu_confidence), 1.0))
    horizon = max(1, int(forecast_horizon))
    mu = np.array(mu_hist, dtype=float)
    for i, t in enumerate(tickers):
        if t not in predicted_mu:
            continue
        try:
            period_ret = float(predicted_mu[t])
        except (TypeError, ValueError):
            continue
        daily = (1.0 + period_ret) ** (1.0 / float(horizon)) - 1.0
        mu[i] = (1.0 - conf) * mu_hist[i] + conf * daily
    return mu


def recompute_stats(
    tickers: list[str],
    weights: dict[str, float],
    daily_returns: np.ndarray,
    *,
    risk_free_rate: float = 0.04,
    mu_override: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Recomputes (expected_return, volatility, sharpe) for arbitrary weights.

    Used when post-optimizer constraints (cap/floor/cash_buffer) alter weights,
    so the reported stats match the actually-returned allocation.

    Pass `mu_override` to recompute on a non-historical basis (e.g. the blended
    forecast mu from blend_forecast_mu); otherwise historical mu is used.
    """
    x = np.asarray(daily_returns, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(tickers) or x.shape[1] == 0:
        return 0.0, 0.0, 0.0
    if mu_override is not None and np.asarray(mu_override).shape == (len(tickers),):
        mu = np.asarray(mu_override, dtype=float)
    else:
        mu = np.nanmean(x, axis=0)
    rf_daily = _to_daily_rf(risk_free_rate)
    w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
    if x.shape[1] == 1:
        # np.cov on a single series is scalar; handle directly.
        var = float(np.nanvar(x[:, 0], ddof=1)) if x.shape[0] > 1 else 0.0
        ret = float(mu[0] * w[0])
        vol = math.sqrt(max(var, 0.0)) * abs(float(w[0]))
        sharpe = 0.0
        if vol > 0:
            sharpe = (ret - rf_daily) / vol
        return ret, vol, float(sharpe)
    cov = sanitize_cov(np.cov(np.nan_to_num(x, nan=0.0), rowvar=False))
    return _portfolio_stats(w, mu, cov, rf_daily)


def apply_weight_constraints(
    weights: dict[str, float],
    *,
    max_weight: float | None = None,
    min_weight: float | None = None,
    cash_buffer: float | None = None,
) -> dict[str, float]:
    """Applies drop-min / cap-max / cash-buffer to a weight dict.

    Semantics:
        - min_weight: drop names below threshold, renormalize remainder to sum=1.
        - max_weight: cap each name at threshold; redistribute excess pro-rata to
          uncapped names, iterating until stable.
        - cash_buffer: scale final weights so equities sum to (1 - cash_buffer).
    Any of the arguments may be None to skip that constraint.
    """
    w = {str(k): float(v) for k, v in weights.items() if float(v) > 0.0}

    if min_weight is not None and float(min_weight) > 0.0:
        thresh = float(min_weight)
        w = {k: v for k, v in w.items() if v >= thresh}
        s = sum(w.values())
        if s > 0:
            w = {k: v / s for k, v in w.items()}

    if max_weight is not None and float(max_weight) > 0.0 and w:
        cap = float(max_weight)
        for _ in range(len(w) + 2):
            over = [k for k, v in w.items() if v > cap + 1e-12]
            if not over:
                break
            excess = sum(w[k] - cap for k in over)
            for k in over:
                w[k] = cap
            uncapped = {k: v for k, v in w.items() if v < cap - 1e-12}
            base = sum(uncapped.values())
            if base <= 0:
                break  # all names at cap — sum may stay < 1, caller must cope
            for k in list(uncapped.keys()):
                w[k] += excess * (uncapped[k] / base)

    if cash_buffer is not None and float(cash_buffer) > 0.0:
        scale = max(0.0, 1.0 - float(cash_buffer))
        w = {k: v * scale for k, v in w.items()}

    return w


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, 1.0)
    s = float(np.sum(w))
    if s <= 0:
        return np.full_like(w, 1.0 / max(1, w.size), dtype=float)
    return w / s


def _portfolio_stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf_daily: float) -> tuple[float, float, float]:
    """Returns (ret, vol, sharpe) in daily units."""
    ret = float(np.dot(mu, w))
    var = float(np.dot(w, np.dot(cov, w)))
    vol = math.sqrt(max(var, 0.0))
    sharpe = 0.0
    if vol > 0:
        sharpe = (ret - rf_daily) / vol
    return ret, vol, float(sharpe)


def _solve_long_only(*, objective_fn, w0: np.ndarray) -> np.ndarray:
    """Runs SLSQP with long-only weights; falls back to w0 if scipy unavailable/fails."""
    w0 = _normalize_weights(w0)
    if not _HAS_SCIPY:
        return w0

    n = int(w0.size)
    bounds = [(0.0, 1.0) for _ in range(n)]
    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0}]

    try:
        res = minimize(objective_fn, w0, method="SLSQP", bounds=bounds, constraints=cons)
        w = np.array(res.x if getattr(res, "success", False) else w0, dtype=float)
        return _normalize_weights(w)
    except Exception:  # pragma: no cover
        return w0


def _max_sharpe_weights(*, mu: np.ndarray, cov: np.ndarray, rf_daily: float) -> np.ndarray:
    n = int(mu.size)
    w0 = np.full(n, 1.0 / max(1, n), dtype=float)

    def objective(w: np.ndarray) -> float:
        ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf_daily)
        if not np.isfinite(ret) or not np.isfinite(vol) or vol <= 0:
            return 1e6
        if not np.isfinite(sharpe):
            return 1e6
        return -float(sharpe)

    return _solve_long_only(objective_fn=objective, w0=w0)


def _min_vol_weights(*, cov: np.ndarray) -> np.ndarray:
    n = int(cov.shape[0])
    w0 = np.full(n, 1.0 / max(1, n), dtype=float)

    def objective(w: np.ndarray) -> float:
        var = float(np.dot(w, np.dot(cov, w)))
        if not np.isfinite(var):
            return 1e6
        return math.sqrt(max(var, 0.0))

    return _solve_long_only(objective_fn=objective, w0=w0)


def optimize_max_sharpe(
    tickers: list[str],
    daily_returns: np.ndarray,
    *,
    risk_free_rate: float = 0.04,
) -> AllocationResult:
    """Long-only max-Sharpe allocation from daily returns (daily units)."""
    x = np.asarray(daily_returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("daily_returns must be a 2D matrix")

    n = int(x.shape[1])
    if n == 0 or len(tickers) != n:
        raise ValueError("tickers and daily_returns shape mismatch")

    mu = np.nanmean(x, axis=0)
    cov = np.cov(np.nan_to_num(x, nan=0.0), rowvar=False)
    cov = sanitize_cov(cov)
    rf_daily = _to_daily_rf(risk_free_rate)

    w = _max_sharpe_weights(mu=mu, cov=cov, rf_daily=rf_daily)
    ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf_daily)
    weights = {t: float(wi) for t, wi in zip(tickers, w)}
    return AllocationResult(weights=weights, expected_return=ret, volatility=vol, sharpe=sharpe, strategy="max_sharpe")


def optimize_min_vol(
    tickers: list[str],
    daily_returns: np.ndarray,
    *,
    risk_free_rate: float = 0.04,
) -> AllocationResult:
    """Long-only minimum-volatility allocation from daily returns (daily units)."""
    x = np.asarray(daily_returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("daily_returns must be a 2D matrix")

    n = int(x.shape[1])
    if n == 0 or len(tickers) != n:
        raise ValueError("tickers and daily_returns shape mismatch")

    mu = np.nanmean(x, axis=0)
    cov = np.cov(np.nan_to_num(x, nan=0.0), rowvar=False)
    cov = sanitize_cov(cov)
    rf_daily = _to_daily_rf(risk_free_rate)

    w = _min_vol_weights(cov=cov)
    ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf_daily)
    weights = {t: float(wi) for t, wi in zip(tickers, w)}
    return AllocationResult(weights=weights, expected_return=ret, volatility=vol, sharpe=sharpe, strategy="min_vol")


def _ivp_weights(cov: np.ndarray) -> np.ndarray:
    diag = np.diag(cov).astype(float)
    inv = np.zeros_like(diag, dtype=float)
    ok = diag > 0
    inv[ok] = 1.0 / diag[ok]
    s = float(np.sum(inv))
    if s <= 0:
        return np.full_like(inv, 1.0 / max(1, inv.size), dtype=float)
    return inv / s


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov).astype(float), 0.0, None))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return (corr + corr.T) / 2.0


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    w = _ivp_weights(sub)
    return float(np.dot(w, np.dot(sub, w)))


def _hrp_bisect_weights(cov: np.ndarray, ordered_items: list[int]) -> np.ndarray:
    # weights aligned to ordered_items
    w = np.ones(len(ordered_items), dtype=float)

    clusters: list[list[int]] = [ordered_items]
    while clusters:
        clusters = [c for c in clusters if len(c) > 1]
        if not clusters:
            break

        new_clusters: list[list[int]] = []
        for c in clusters:
            split = len(c) // 2
            c1 = c[:split]
            c2 = c[split:]

            v1 = _cluster_variance(cov, c1)
            v2 = _cluster_variance(cov, c2)
            denom = v1 + v2
            if denom <= 0 or not np.isfinite(denom):
                alpha = 0.5
            else:
                alpha = 1.0 - (v1 / denom)

            # Apply weight split to ordered indices
            c1_set = set(c1)
            c2_set = set(c2)
            for i, item in enumerate(ordered_items):
                if item in c1_set:
                    w[i] *= alpha
                elif item in c2_set:
                    w[i] *= 1.0 - alpha

            new_clusters.extend([c1, c2])
        clusters = new_clusters

    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 0:
        return np.full_like(w, 1.0 / max(1, w.size), dtype=float)
    return w / s


def optimize_hrp(
    tickers: list[str],
    daily_returns: np.ndarray,
    *,
    risk_free_rate: float = 0.04,
) -> AllocationResult:
    """Hierarchical Risk Parity allocation (daily units)."""
    x = np.asarray(daily_returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("daily_returns must be a 2D matrix")

    n = int(x.shape[1])
    if n == 0 or len(tickers) != n:
        raise ValueError("tickers and daily_returns shape mismatch")

    mu = np.nanmean(x, axis=0)
    cov = np.cov(np.nan_to_num(x, nan=0.0), rowvar=False)
    cov = sanitize_cov(cov)
    rf_daily = _to_daily_rf(risk_free_rate)

    if not _HAS_SCIPY:
        w = _ivp_weights(cov)
        ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf_daily)
        weights = {t: float(wi) for t, wi in zip(tickers, w)}
        return AllocationResult(weights=weights, expected_return=ret, volatility=vol, sharpe=sharpe, strategy="hrp")

    corr = _corr_from_cov(cov)
    dist = np.sqrt(0.5 * (1.0 - corr))
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    dist = np.clip(dist, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    order = list(leaves_list(link))

    w_ordered = _hrp_bisect_weights(cov, order)

    w = np.zeros(n, dtype=float)
    for rank, idx in enumerate(order):
        w[idx] = w_ordered[rank]
    w = _normalize_weights(w)

    ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf_daily)
    weights = {t: float(wi) for t, wi in zip(tickers, w)}
    return AllocationResult(weights=weights, expected_return=ret, volatility=vol, sharpe=sharpe, strategy="hrp")


def optimize_blend(
    tickers: list[str],
    daily_returns: np.ndarray,
    *,
    risk_free_rate: float = 0.04,
    sharpe_ratio: float = 0.6,
    hrp_ratio: float = 0.4,
) -> AllocationResult:
    """Blend allocation: Max-Sharpe + HRP, re-normalized."""
    a = optimize_max_sharpe(tickers, daily_returns, risk_free_rate=risk_free_rate)
    b = optimize_hrp(tickers, daily_returns, risk_free_rate=risk_free_rate)

    s = max(0.0, float(sharpe_ratio))
    h = max(0.0, float(hrp_ratio))
    denom = s + h
    if denom <= 0:
        s, h, denom = 0.5, 0.5, 1.0
    s /= denom
    h /= denom

    w = np.array([s * float(a.weights[t]) + h * float(b.weights[t]) for t in tickers], dtype=float)
    w = _normalize_weights(w)

    x = np.asarray(daily_returns, dtype=float)
    mu = np.nanmean(x, axis=0)
    cov = sanitize_cov(np.cov(np.nan_to_num(x, nan=0.0), rowvar=False))
    rf_daily = _to_daily_rf(risk_free_rate)
    ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf_daily)

    weights = {t: float(wi) for t, wi in zip(tickers, w)}
    return AllocationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        strategy="blend_sharpe_hrp",
    )


def optimize_forecast_sharpe(
    tickers: list[str],
    daily_returns: np.ndarray,
    predicted_mu: dict[str, float],
    *,
    risk_free_rate: float = 0.04,
    mu_confidence: float = 0.3,
    forecast_horizon: int = 20,
) -> AllocationResult:
    """Max-Sharpe with blended mu: historical mu + (confidence * forecast period return)."""
    x = np.asarray(daily_returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("daily_returns must be a 2D matrix")

    n = int(x.shape[1])
    if n == 0 or len(tickers) != n:
        raise ValueError("tickers and daily_returns shape mismatch")

    mu_hist = np.nanmean(x, axis=0)
    conf = float(mu_confidence)
    conf = max(0.0, min(conf, 1.0))
    horizon = max(1, int(forecast_horizon))

    mu = np.array(mu_hist, dtype=float)
    for i, t in enumerate(tickers):
        if t not in predicted_mu:
            continue
        try:
            period_ret = float(predicted_mu[t])
        except (TypeError, ValueError):
            continue
        daily = (1.0 + period_ret) ** (1.0 / float(horizon)) - 1.0
        mu[i] = (1.0 - conf) * mu_hist[i] + conf * daily

    cov = sanitize_cov(np.cov(np.nan_to_num(x, nan=0.0), rowvar=False))
    rf_daily = _to_daily_rf(risk_free_rate)

    w = _max_sharpe_weights(mu=mu, cov=cov, rf_daily=rf_daily)
    ret, vol, sharpe = _portfolio_stats(w, mu, cov, rf_daily)
    weights = {t: float(wi) for t, wi in zip(tickers, w)}
    return AllocationResult(
        weights=weights,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        strategy="forecast_max_sharpe",
    )
