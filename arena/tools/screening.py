from __future__ import annotations

import numpy as np

DISCOVERY_BUCKETS: tuple[str, ...] = (
    "momentum",
    "pullback",
    "recovery",
    "defensive",
    "value",
)


def daily_returns_from_closes(closes: list[float]) -> np.ndarray:
    """Computes simple daily returns from close series."""
    arr = np.array(closes, dtype=float)
    if arr.size < 2:
        return np.array([], dtype=float)
    prev = arr[:-1]
    now = arr[1:]
    out = np.zeros_like(now)
    mask = prev > 0
    out[mask] = (now[mask] / prev[mask]) - 1.0
    return out


def momentum_scores(
    closes_by_ticker: dict[str, list[float]],
    *,
    windows: list[int],
    vol_adjust: bool,
) -> list[dict]:
    """Builds multi-window momentum scores and returns per ticker rows."""
    tickers = sorted(closes_by_ticker.keys())
    if not tickers:
        return []

    vols: dict[str, float] = {}
    for t in tickers:
        r = daily_returns_from_closes(closes_by_ticker[t])
        vols[t] = float(np.std(r[-20:])) if r.size else 0.0

    raw_by_window: dict[int, list[tuple[str, float]]] = {}
    for w in windows:
        vals: list[tuple[str, float]] = []
        for t in tickers:
            closes = closes_by_ticker[t]
            if len(closes) <= w:
                vals.append((t, 0.0))
                continue
            base = closes[-(w + 1)]
            now = closes[-1]
            r = (now / base) - 1.0 if base > 0 else 0.0
            if vol_adjust:
                v = max(vols.get(t, 0.0), 1e-8)
                r = r / v
            vals.append((t, float(r)))
        raw_by_window[w] = vals

    scores: dict[str, float] = {t: 0.0 for t in tickers}
    for w, vals in raw_by_window.items():
        xs = np.array([v for _, v in vals], dtype=float)
        mean = float(np.mean(xs))
        std = float(np.std(xs))
        if std <= 1e-12:
            continue
        for t, v in vals:
            scores[t] += (v - mean) / std

    rows = []
    for t in tickers:
        rows.append(
            {
                "ticker": t,
                "score": float(scores[t]),
                "vol_20d": float(vols.get(t, 0.0)),
            }
        )
    rows.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return rows


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        text = str(value).strip().replace(",", "")
        if not text:
            return float(default)
        return float(text)
    except (TypeError, ValueError):
        return float(default)


def _zscore_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array([float(v) for v in values.values()], dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(std) or std <= 1e-12:
        return {ticker: 0.0 for ticker in values}
    return {
        ticker: float((float(value) - mean) / std)
        for ticker, value in values.items()
    }


def _inverse_rank_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: float(item[1]))
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    denom = max(1, len(ordered) - 1)
    return {
        ticker: float((denom - idx) / denom)
        for idx, (ticker, _) in enumerate(ordered)
    }


def _append_metric(row: dict[str, object], key: str, value: object) -> None:
    if value is None:
        return
    try:
        val = float(value)
    except (TypeError, ValueError):
        return
    if not np.isfinite(val):
        return
    row[key] = float(val)


def _screen_reason_risk(
    *,
    bucket: str,
    ret_5d: float,
    ret_20d: float,
    volatility_20d: float,
    fundamentals: dict[str, object] | None = None,
) -> str:
    """Builds a compact, data-backed caution note for screened candidates."""
    notes: list[str] = []
    bucket_token = str(bucket or "").strip().lower()

    if volatility_20d >= 0.50:
        notes.append(f"High 20d volatility {volatility_20d:.3f}.")
    elif volatility_20d >= 0.30:
        notes.append(f"Elevated 20d volatility {volatility_20d:.3f}.")

    if ret_5d < 0.0:
        notes.append(f"5d return is cooling at {ret_5d:+.2%}.")
    elif bucket_token == "momentum" and ret_20d >= 0.25:
        notes.append(f"20d move is already stretched at {ret_20d:+.2%}.")

    if bucket_token == "value":
        fund = fundamentals or {}
        per = _safe_float(fund.get("per"), default=np.nan)
        pbr = _safe_float(fund.get("pbr"), default=np.nan)
        roe = _safe_float(fund.get("roe"), default=np.nan)
        debt = _safe_float(fund.get("debt_ratio"), default=np.nan)
        if not np.isfinite(per) and not np.isfinite(pbr):
            notes.append("Valuation snapshot is incomplete.")
        if np.isfinite(roe) and roe <= 0.0:
            notes.append(f"ROE is weak at {roe:.2f}.")
        if np.isfinite(debt) and debt >= 200.0:
            notes.append(f"Debt ratio is elevated at {debt:.1f}.")

    if not notes:
        notes.append("Screen-only evidence; confirm with forecast/technical/fundamental tools before initiating.")
    return " ".join(notes)


def build_discovery_rows(
    latest_rows: list[dict[str, object]],
    *,
    momentum_rows: list[dict[str, object]] | None = None,
    fundamentals_rows: list[dict[str, object]] | None = None,
    bucket: str | None = None,
    top_n: int = 10,
    per_bucket: int | None = None,
    order: str = "desc",
) -> list[dict[str, object]]:
    """Builds multi-style discovery rows from a runtime universe snapshot."""
    requested_bucket = str(bucket or "balanced").strip().lower() or "balanced"
    direction = "asc" if str(order or "").strip().lower() == "asc" else "desc"

    row_map: dict[str, dict[str, object]] = {}
    for row in latest_rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        row_map[ticker] = dict(row)
    if not row_map:
        return []

    momentum_map = {
        str(row.get("ticker") or "").strip().upper(): dict(row)
        for row in (momentum_rows or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }
    fundamentals_map = {
        str(row.get("ticker") or "").strip().upper(): dict(row)
        for row in (fundamentals_rows or [])
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    }

    ret5 = {ticker: _safe_float(row.get("ret_5d")) for ticker, row in row_map.items()}
    ret20 = {ticker: _safe_float(row.get("ret_20d")) for ticker, row in row_map.items()}
    vol20 = {
        ticker: max(0.0, _safe_float(row.get("volatility_20d"), default=0.0))
        for ticker, row in row_map.items()
    }
    sentiment = {ticker: _safe_float(row.get("sentiment_score")) for ticker, row in row_map.items()}
    momentum = {
        ticker: _safe_float((momentum_map.get(ticker) or {}).get("score"))
        for ticker in row_map
    }

    ret5_z = _zscore_map(ret5)
    ret20_z = _zscore_map(ret20)
    vol20_z = _zscore_map(vol20)
    sentiment_z = _zscore_map(sentiment)
    momentum_z = _zscore_map(momentum)

    positive_per = {
        ticker: value
        for ticker, value in {
            t: _safe_float((fundamentals_map.get(t) or {}).get("per"), default=np.nan)
            for t in row_map
        }.items()
        if np.isfinite(value) and value > 0.0
    }
    positive_pbr = {
        ticker: value
        for ticker, value in {
            t: _safe_float((fundamentals_map.get(t) or {}).get("pbr"), default=np.nan)
            for t in row_map
        }.items()
        if np.isfinite(value) and value > 0.0
    }
    roe_raw = {
        ticker: value
        for ticker, value in {
            t: _safe_float((fundamentals_map.get(t) or {}).get("roe"), default=np.nan)
            for t in row_map
        }.items()
        if np.isfinite(value)
    }
    debt_raw = {
        ticker: value
        for ticker, value in {
            t: _safe_float((fundamentals_map.get(t) or {}).get("debt_ratio"), default=np.nan)
            for t in row_map
        }.items()
        if np.isfinite(value)
    }
    growth_raw: dict[str, float] = {}
    for ticker in row_map:
        raw_values = [
            _safe_float((fundamentals_map.get(ticker) or {}).get("operating_profit_growth"), default=np.nan),
            _safe_float((fundamentals_map.get(ticker) or {}).get("net_profit_growth"), default=np.nan),
        ]
        finite = [float(value) for value in raw_values if np.isfinite(value)]
        if finite:
            growth_raw[ticker] = float(np.mean(np.array(finite, dtype=float)))

    cheap_per = _inverse_rank_map(positive_per)
    cheap_pbr = _inverse_rank_map(positive_pbr)
    roe_z = _zscore_map(roe_raw)
    debt_z = _zscore_map(debt_raw)
    growth_z = _zscore_map(growth_raw)

    bucket_rows: dict[str, list[dict[str, object]]] = {name: [] for name in DISCOVERY_BUCKETS}
    for ticker, feature_row in row_map.items():
        base_row = {
            "ticker": ticker,
            "as_of_ts": feature_row.get("as_of_ts"),
            "exchange_code": feature_row.get("exchange_code"),
            "instrument_id": feature_row.get("instrument_id"),
            "close_price_krw": feature_row.get("close_price_krw"),
            "ret_5d": ret5.get(ticker),
            "ret_20d": ret20.get(ticker),
            "volatility_20d": vol20.get(ticker),
            "sentiment_score": sentiment.get(ticker),
            "source": feature_row.get("source"),
        }

        momentum_score = (
            0.55 * momentum_z.get(ticker, 0.0)
            + 0.25 * ret20_z.get(ticker, 0.0)
            + 0.10 * ret5_z.get(ticker, 0.0)
            + 0.10 * sentiment_z.get(ticker, 0.0)
            - 0.05 * vol20_z.get(ticker, 0.0)
        )
        row = dict(base_row)
        row["bucket"] = "momentum"
        row["score"] = round(float(momentum_score), 6)
        reason_for = (
            f"Multi-window momentum {momentum.get(ticker, 0.0):+.2f}, "
            f"20d return {ret20.get(ticker, 0.0):+.2%}"
        )
        row["reason"] = reason_for
        row["reason_for"] = reason_for
        row["reason_risk"] = _screen_reason_risk(
            bucket="momentum",
            ret_5d=ret5.get(ticker, 0.0),
            ret_20d=ret20.get(ticker, 0.0),
            volatility_20d=vol20.get(ticker, 0.0),
        )
        row["evidence_level"] = "screened_only"
        bucket_rows["momentum"].append(row)

        pullback_score = (
            0.50 * momentum_z.get(ticker, 0.0)
            + 0.35 * ret20_z.get(ticker, 0.0)
            - 0.55 * ret5_z.get(ticker, 0.0)
            + 0.10 * sentiment_z.get(ticker, 0.0)
            - 0.10 * vol20_z.get(ticker, 0.0)
        )
        row = dict(base_row)
        row["bucket"] = "pullback"
        row["score"] = round(float(pullback_score), 6)
        reason_for = (
            f"Uptrend intact but recent pullback: 20d {ret20.get(ticker, 0.0):+.2%}, "
            f"5d {ret5.get(ticker, 0.0):+.2%}"
        )
        row["reason"] = reason_for
        row["reason_for"] = reason_for
        row["reason_risk"] = _screen_reason_risk(
            bucket="pullback",
            ret_5d=ret5.get(ticker, 0.0),
            ret_20d=ret20.get(ticker, 0.0),
            volatility_20d=vol20.get(ticker, 0.0),
        )
        row["evidence_level"] = "screened_only"
        bucket_rows["pullback"].append(row)

        recovery_score = (
            0.45 * ret5_z.get(ticker, 0.0)
            - 0.30 * ret20_z.get(ticker, 0.0)
            + 0.20 * momentum_z.get(ticker, 0.0)
            + 0.20 * sentiment_z.get(ticker, 0.0)
            - 0.10 * vol20_z.get(ticker, 0.0)
        )
        row = dict(base_row)
        row["bucket"] = "recovery"
        row["score"] = round(float(recovery_score), 6)
        reason_for = (
            f"Recent recovery bias: 5d {ret5.get(ticker, 0.0):+.2%}, "
            f"20d {ret20.get(ticker, 0.0):+.2%}, sentiment {sentiment.get(ticker, 0.0):+.2f}"
        )
        row["reason"] = reason_for
        row["reason_for"] = reason_for
        row["reason_risk"] = _screen_reason_risk(
            bucket="recovery",
            ret_5d=ret5.get(ticker, 0.0),
            ret_20d=ret20.get(ticker, 0.0),
            volatility_20d=vol20.get(ticker, 0.0),
        )
        row["evidence_level"] = "screened_only"
        bucket_rows["recovery"].append(row)

        defensive_score = (
            -0.70 * vol20_z.get(ticker, 0.0)
            + 0.20 * ret20_z.get(ticker, 0.0)
            + 0.15 * sentiment_z.get(ticker, 0.0)
            + 0.10 * ret5_z.get(ticker, 0.0)
        )
        row = dict(base_row)
        row["bucket"] = "defensive"
        row["score"] = round(float(defensive_score), 6)
        reason_for = (
            f"Lower-volatility profile: vol20 {vol20.get(ticker, 0.0):.3f}, "
            f"20d {ret20.get(ticker, 0.0):+.2%}"
        )
        row["reason"] = reason_for
        row["reason_for"] = reason_for
        row["reason_risk"] = _screen_reason_risk(
            bucket="defensive",
            ret_5d=ret5.get(ticker, 0.0),
            ret_20d=ret20.get(ticker, 0.0),
            volatility_20d=vol20.get(ticker, 0.0),
        )
        row["evidence_level"] = "screened_only"
        bucket_rows["defensive"].append(row)

        fundamentals = fundamentals_map.get(ticker) or {}
        if fundamentals:
            eps = fundamentals.get("eps")
            bps = fundamentals.get("bps")
            per = fundamentals.get("per")
            pbr = fundamentals.get("pbr")
            quality_guard = 0.0
            if eps is not None and _safe_float(eps, default=-1.0) > 0.0:
                quality_guard += 0.35
            if bps is not None and _safe_float(bps, default=-1.0) > 0.0:
                quality_guard += 0.25
            if fundamentals.get("roe") is not None and _safe_float(fundamentals.get("roe"), default=-99.0) > 0.0:
                quality_guard += 0.20
            if fundamentals.get("debt_ratio") is None or _safe_float(fundamentals.get("debt_ratio"), default=1000.0) < 250.0:
                quality_guard += 0.20
            value_score = (
                0.45 * cheap_per.get(ticker, 0.0)
                + 0.35 * cheap_pbr.get(ticker, 0.0)
                + 0.20 * roe_z.get(ticker, 0.0)
                - 0.15 * debt_z.get(ticker, 0.0)
                + 0.10 * growth_z.get(ticker, 0.0)
                + quality_guard
            )
            row = dict(base_row)
            row["bucket"] = "value"
            row["score"] = round(float(value_score), 6)
            reason_for = (
                f"Valuation support: PER {per if per is not None else 'n/a'}, "
                f"PBR {pbr if pbr is not None else 'n/a'}"
            )
            row["reason"] = reason_for
            row["reason_for"] = reason_for
            row["reason_risk"] = _screen_reason_risk(
                bucket="value",
                ret_5d=ret5.get(ticker, 0.0),
                ret_20d=ret20.get(ticker, 0.0),
                volatility_20d=vol20.get(ticker, 0.0),
                fundamentals=fundamentals,
            )
            row["evidence_level"] = "screened_only"
            _append_metric(row, "per", per)
            _append_metric(row, "pbr", pbr)
            _append_metric(row, "eps", eps)
            _append_metric(row, "bps", bps)
            _append_metric(row, "roe", fundamentals.get("roe"))
            _append_metric(row, "debt_ratio", fundamentals.get("debt_ratio"))
            bucket_rows["value"].append(row)

    reverse = direction != "asc"
    for name, rows in bucket_rows.items():
        rows.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                float(item.get("ret_20d") or 0.0),
                -float(item.get("volatility_20d") or 0.0),
                str(item.get("ticker") or ""),
            ),
            reverse=reverse,
        )
        for idx, row in enumerate(rows, start=1):
            row["bucket_rank"] = idx

    if requested_bucket not in {"", "balanced"}:
        if requested_bucket not in bucket_rows:
            return []
        rows = bucket_rows[requested_bucket]
        return rows[: max(1, min(int(top_n), 100))]

    selected_buckets = [name for name in DISCOVERY_BUCKETS if bucket_rows.get(name)]
    if not selected_buckets:
        return []
    bucket_cap = max(2, min(int(per_bucket or top_n), 30))
    trimmed = {
        name: rows[:bucket_cap]
        for name, rows in bucket_rows.items()
        if name in selected_buckets
    }

    out: list[dict[str, object]] = []
    seen: set[str] = set()
    round_idx = 0
    limit = max(1, min(int(top_n), 100))
    while len(out) < limit and any(round_idx < len(rows) for rows in trimmed.values()):
        for name in selected_buckets:
            rows = trimmed.get(name) or []
            if round_idx >= len(rows):
                continue
            row = rows[round_idx]
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            out.append(row)
            if len(out) >= limit:
                break
        round_idx += 1
    return out
