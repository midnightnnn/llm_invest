"""Builds semantic search queries from tool results for REACT-time memory injection."""
from __future__ import annotations

from typing import Any

MEMORY_INJECTABLE_TOOLS: set[str] = {
    "technical_signals",
    "screen_market",
    "forecast_returns",
    "get_fundamentals",
    "optimize_portfolio",
}


def _top_tickers(rows: list[dict[str, Any]], key: str = "ticker", n: int = 3) -> str:
    """Extracts up to *n* ticker symbols from a list of row dicts."""
    tickers: list[str] = []
    for row in rows:
        t = str(row.get(key) or "").strip()
        if t and t not in tickers:
            tickers.append(t)
        if len(tickers) >= n:
            break
    return " ".join(tickers)


def build_memory_query(tool_name: str, args: dict[str, Any], result: Any) -> str | None:
    """Returns a semantic search query derived from tool output, or None to skip."""
    if tool_name not in MEMORY_INJECTABLE_TOOLS:
        return None

    # Skip error results
    if isinstance(result, dict) and result.get("error"):
        return None

    try:
        return _BUILDERS[tool_name](args, result)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-tool query builders
# ---------------------------------------------------------------------------

def _technical_signals(args: dict, result: Any) -> str | None:
    if not isinstance(result, dict):
        return None

    rows = result.get("rows")
    if isinstance(rows, list):
        tickers = _top_tickers(rows)
        return f"technical signals {tickers}" if tickers else None

    ticker = str(result.get("ticker") or args.get("ticker") or "").strip()
    if not ticker:
        return None
    rsi_state = str(result.get("rsi_state") or "").strip()
    trend_state = str(result.get("trend_state") or "").strip()
    macd = result.get("macd") or {}
    macd_state = str(macd.get("state") or "").strip() if isinstance(macd, dict) else ""
    parts = [p for p in [ticker, rsi_state, trend_state, macd_state] if p]
    return " ".join(parts) if len(parts) >= 2 else None


def _screen_market(args: dict, result: Any) -> str | None:
    rows = result if isinstance(result, list) else []
    if not rows:
        return None
    tickers = _top_tickers(rows)
    requested_bucket = str(args.get("bucket") or "").strip().lower()
    buckets: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bucket = str(row.get("bucket") or "").strip().lower()
        if bucket and bucket not in buckets:
            buckets.append(bucket)
        if len(buckets) >= 3:
            break
    bucket_phrase = " ".join(buckets[:3]) if buckets else requested_bucket
    parts = ["market screening"]
    if bucket_phrase:
        parts.append(bucket_phrase)
    if tickers:
        parts.append(tickers)
    query = " ".join(part for part in parts if part).strip()
    return query or None

def _forecast_returns(args: dict, result: Any) -> str | None:
    rows = result if isinstance(result, list) else []
    if not rows:
        return None
    tickers = _top_tickers(rows)
    return f"return forecast {tickers}" if tickers else None


def _get_fundamentals(args: dict, result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    rows = result.get("rows") or []
    if not rows or not isinstance(rows, list):
        return None
    first = rows[0] if rows else {}
    ticker = str(first.get("ticker") or "").strip()
    if not ticker:
        return None
    market = str(first.get("market") or "").strip()
    if market == "kospi":
        roe = first.get("roe")
        debt = first.get("debt_ratio")
        roe_str = f"ROE {roe}" if roe is not None else ""
        debt_str = f"debt {debt}%" if debt is not None else ""
        parts = [p for p in [f"{ticker} valuation", roe_str, debt_str] if p]
    else:
        per = first.get("per")
        pbr = first.get("pbr")
        per_str = f"PER {per}" if per is not None else ""
        pbr_str = f"PBR {pbr}" if pbr is not None else ""
        parts = [p for p in [f"{ticker} valuation", per_str, pbr_str] if p]
    return " ".join(parts)


def _optimize_portfolio(args: dict, result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    strategy = str(result.get("strategy") or args.get("strategy") or "").strip()
    tickers = result.get("tickers") or []
    ticker_str = " ".join(str(t) for t in tickers[:5])
    return f"portfolio optimization {strategy} {ticker_str}".strip() or None


_BUILDERS: dict[str, Any] = {
    "technical_signals": _technical_signals,
    "screen_market": _screen_market,
    "forecast_returns": _forecast_returns,
    "get_fundamentals": _get_fundamentals,
    "optimize_portfolio": _optimize_portfolio,
}
