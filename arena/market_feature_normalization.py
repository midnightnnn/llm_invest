from __future__ import annotations

import logging
import math
from statistics import stdev
from typing import Any

logger = logging.getLogger(__name__)


def daily_history_sources(sources: list[str] | None) -> list[str] | None:
    """Restricts raw close lookups to daily sources when live quote sources are present."""
    if not sources:
        return None
    daily = [str(source) for source in sources if not str(source).endswith("_quote")]
    return daily or list(sources)


def _finite_float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _daily_returns_from_closes(closes: list[float]) -> list[float]:
    out: list[float] = []
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        now = closes[idx]
        if prev > 0:
            out.append((now / prev) - 1.0)
    return out


def close_window_return(closes: list[float], window: int) -> float | None:
    """Computes a trailing close-to-close return when enough raw closes exist."""
    if len(closes) <= window:
        return None
    base = closes[-(window + 1)]
    now = closes[-1]
    if base <= 0:
        return None
    return float((now / base) - 1.0)


def close_volatility_20d(closes: list[float]) -> float | None:
    """Computes 20-trading-day daily return volatility from raw closes."""
    rets = _daily_returns_from_closes(closes)
    if len(rets) < 20:
        return None
    if len(rets[-20:]) < 2:
        return None
    vol = float(stdev(rets[-20:]))
    if not math.isfinite(vol):
        return None
    return vol


def normalize_market_feature_rows_from_closes(
    latest_rows: list[dict[str, Any]],
    closes_by_ticker: dict[str, list[float]],
    *,
    include_quality: bool = False,
) -> list[dict[str, Any]]:
    """Overlays prompt/tool market features with deterministic raw-close features.

    Database snapshots remain the source of record for price, source, FX, and
    metadata. Raw daily closes are the source of truth for derived return fields
    whenever enough history exists.
    """
    if not latest_rows or not closes_by_ticker:
        return latest_rows

    out: list[dict[str, Any]] = []
    for row in latest_rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        closes = closes_by_ticker.get(ticker) or []
        patched = dict(row)
        quality: dict[str, Any] = {"daily_close_points": len(closes)}

        derived = {
            "ret_5d": close_window_return(closes, 5),
            "ret_20d": close_window_return(closes, 20),
            "volatility_20d": close_volatility_20d(closes),
        }
        for key, value in derived.items():
            if value is not None:
                patched[key] = value
                quality[key] = "raw_close"
            elif _finite_float_or_none(row.get(key)) is not None:
                quality[key] = "stored"
            else:
                quality[key] = "missing"

        if include_quality:
            patched["_feature_quality"] = quality
        out.append(patched)
    return out


def normalize_market_feature_rows(
    latest_rows: list[dict[str, Any]],
    *,
    repo: Any,
    sources: list[str] | None = None,
    lookback_days: int = 22,
    include_quality: bool = False,
) -> list[dict[str, Any]]:
    """Loads raw daily closes and normalizes derived return fields for rows."""
    if not latest_rows:
        return latest_rows
    loader = getattr(repo, "get_daily_closes", None)
    if not callable(loader):
        return latest_rows

    tickers = [
        str(row.get("ticker") or "").strip().upper()
        for row in latest_rows
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    ]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return latest_rows

    try:
        closes = loader(
            tickers=tickers,
            lookback_days=max(22, int(lookback_days)),
            sources=daily_history_sources(sources),
        ) or {}
    except Exception as exc:
        logger.warning("[yellow]market feature raw-close normalization skipped[/yellow] err=%s", str(exc))
        return latest_rows

    return normalize_market_feature_rows_from_closes(
        latest_rows,
        closes,
        include_quality=include_quality,
    )
