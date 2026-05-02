"""Builds semantic search queries from tool results for REACT-time memory injection."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

MEMORY_INJECTABLE_TOOLS: set[str] = {
    "technical_signals",
    "recommend_opportunities",
    "screen_market",
    "forecast_returns",
    "get_fundamentals",
    "optimize_portfolio",
    "index_snapshot",
    "fear_greed_index",
    "earnings_calendar",
    "fetch_reddit_sentiment",
    "fetch_sec_filings",
    "macro_snapshot",
    "get_research_briefing",
}


@dataclass(frozen=True)
class MemoryQuerySpec:
    """Typed retrieval request for REACT-time memory injection."""

    tool_name: str
    key_type: str
    keys: tuple[str, ...]
    query: str
    context_keys: tuple[str, ...] = ()

    def search_text(self) -> str:
        typed_parts = [f"{self.key_type}:{key}" for key in self.keys]
        context_parts = [f"context:{key}" for key in self.context_keys]
        return " ".join(
            part
            for part in [
                self.query,
                f"tool:{self.tool_name}",
                f"key_type:{self.key_type}",
                " ".join(typed_parts),
                " ".join(context_parts),
            ]
            if part
        ).strip()


def _dedupe(tokens: list[Any], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for token in tokens:
        clean = str(token or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if limit is not None and len(out) >= limit:
            break
    return out


def _ticker_list_from_args_result(args: dict, result: Any, rows: list[Any]) -> list[str]:
    tickers: list[Any] = []
    if isinstance(args.get("tickers"), list):
        tickers.extend(args.get("tickers") or [])
    if args.get("ticker"):
        tickers.append(args.get("ticker"))
    if isinstance(result, dict):
        if isinstance(result.get("tickers"), list):
            tickers.extend(result.get("tickers") or [])
        if result.get("ticker"):
            tickers.append(result.get("ticker"))
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("ticker") or row.get("symbol")
        if symbol:
            tickers.append(symbol)
    return _dedupe([str(t).strip().upper() for t in tickers if str(t).strip()], limit=6)


def _normalize_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9_\-/ ]+", " ", text)
    return re.sub(r"[\s/]+", "_", text).strip("_")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value_from_indicator(item: Any) -> float | None:
    if isinstance(item, dict):
        return _float_or_none(item.get("value"))
    return _float_or_none(item)


def _make_spec(
    tool_name: str,
    key_type: str,
    keys: list[Any],
    parts: list[Any],
    *,
    context_keys: list[Any] | None = None,
) -> MemoryQuerySpec | None:
    clean_parts = [str(part or "").strip() for part in parts if str(part or "").strip()]
    query = " ".join(clean_parts).strip()
    clean_keys = tuple(_dedupe(keys))
    if not query or not clean_keys:
        return None
    return MemoryQuerySpec(
        tool_name=str(tool_name or "").strip(),
        key_type=str(key_type or "").strip(),
        keys=clean_keys,
        query=query,
        context_keys=tuple(_dedupe(context_keys or [])),
    )


def _top_tickers(rows: list[dict[str, Any]], key: str = "ticker", n: int = 3) -> str:
    """Extracts up to *n* ticker symbols from a list of row dicts."""
    return " ".join(_top_ticker_list(rows, key=key, n=n))


def _top_ticker_list(rows: list[dict[str, Any]], key: str = "ticker", n: int = 3) -> list[str]:
    tickers: list[str] = []
    for row in rows:
        t = str(row.get(key) or row.get("symbol") or "").strip().upper()
        if t and t not in tickers:
            tickers.append(t)
        if len(tickers) >= n:
            break
    return tickers


def build_memory_query(tool_name: str, args: dict[str, Any], result: Any) -> str | None:
    """Returns a semantic search query derived from tool output, or None to skip."""
    spec = build_memory_query_spec(tool_name, args, result)
    return spec.query if spec is not None else None


def build_memory_query_spec(tool_name: str, args: dict[str, Any], result: Any) -> MemoryQuerySpec | None:
    """Returns a typed semantic search spec derived from tool output, or None to skip."""
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

def _technical_signals(args: dict, result: Any) -> MemoryQuerySpec | None:
    if not isinstance(result, dict):
        return None

    rows = result.get("rows")
    if isinstance(rows, list):
        ticker_list = _top_ticker_list(rows)
        tickers = " ".join(ticker_list)
        if not tickers:
            return None
        return _make_spec("technical_signals", "ticker", ticker_list, ["technical signals", tickers])

    ticker = str(result.get("ticker") or args.get("ticker") or "").strip()
    if not ticker:
        return None
    rsi_state = str(result.get("rsi_state") or "").strip()
    trend_state = str(result.get("trend_state") or "").strip()
    macd = result.get("macd") or {}
    macd_state = str(macd.get("state") or "").strip() if isinstance(macd, dict) else ""
    parts = [p for p in [ticker, rsi_state, trend_state, macd_state] if p]
    return _make_spec("technical_signals", "ticker", [ticker], parts) if len(parts) >= 2 else None


def _screen_market(args: dict, result: Any) -> MemoryQuerySpec | None:
    rows = result if isinstance(result, list) else []
    if not rows:
        return None
    ticker_list = _top_ticker_list(rows)
    tickers = " ".join(ticker_list)
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
    return _make_spec("screen_market", "ticker_strategy", [*buckets[:3], *ticker_list], [query])


def _recommend_opportunities(args: dict, result: Any) -> MemoryQuerySpec | None:
    if not isinstance(result, dict):
        return None
    rows = result.get("recommendations") or result.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return None
    ticker_list = _top_ticker_list(rows)
    tickers = " ".join(ticker_list)
    profiles: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "").strip().lower()
        if profile and profile not in profiles:
            profiles.append(profile)
        if len(profiles) >= 3:
            break
    parts = ["validated opportunities"]
    if profiles:
        parts.append(" ".join(profiles[:3]))
    if tickers:
        parts.append(tickers)
    query = " ".join(part for part in parts if part).strip()
    return _make_spec("recommend_opportunities", "ticker_strategy", [*profiles[:3], *ticker_list], [query])

def _forecast_returns(args: dict, result: Any) -> MemoryQuerySpec | None:
    rows = result if isinstance(result, list) else []
    if not rows:
        return None
    ticker_list = _top_ticker_list(rows)
    tickers = " ".join(ticker_list)
    return _make_spec("forecast_returns", "ticker", ticker_list, ["return forecast", tickers]) if tickers else None


def _get_fundamentals(args: dict, result: Any) -> MemoryQuerySpec | None:
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
    return _make_spec("get_fundamentals", "ticker", [ticker], parts)


def _optimize_portfolio(args: dict, result: Any) -> MemoryQuerySpec | None:
    if not isinstance(result, dict):
        return None
    strategy = str(result.get("strategy") or args.get("strategy") or "").strip()
    tickers = result.get("tickers") or []
    ticker_str = " ".join(str(t) for t in tickers[:5])
    keys = [strategy, *[str(t).strip().upper() for t in tickers[:5] if str(t).strip()]]
    return _make_spec("optimize_portfolio", "ticker_strategy", keys, ["portfolio optimization", strategy, ticker_str])


def _macro_snapshot(args: dict, result: Any) -> MemoryQuerySpec | None:
    if not isinstance(result, dict):
        return None
    indicators = result.get("indicators") or {}
    if not isinstance(indicators, dict) or not indicators:
        return None
    keys: list[str] = []
    fed = _value_from_indicator(indicators.get("fed_funds_rate") or indicators.get("bok_base_rate"))
    if fed is not None and fed >= 4.0:
        keys.append("high_rates")
    y10 = _value_from_indicator(
        indicators.get("treasury_10y")
        or indicators.get("kr_treasury_5y")
        or indicators.get("kr_treasury_3y")
    )
    if y10 is not None and y10 >= 4.5:
        keys.append("high_yields")
    for name, item in indicators.items():
        if "spread" in str(name).lower():
            spread = _value_from_indicator(item)
            if spread is not None and spread < 0:
                keys.append("yield_curve_inverted")
                break
    cpi = _value_from_indicator(indicators.get("cpi_yoy"))
    if cpi is not None and cpi >= 3.0:
        keys.append("inflation_elevated")
    usd_krw = _value_from_indicator(indicators.get("usd_krw"))
    if usd_krw is not None and usd_krw >= 1350:
        keys.append("strong_usd")
    if not keys:
        keys.append("macro_regime")
    indicator_names = [str(name).strip() for name in indicators.keys() if str(name).strip()][:8]
    return _make_spec("macro_snapshot", "regime", keys, ["macro regime", " ".join(keys), " ".join(indicator_names)])


def _fear_greed_index(args: dict, result: Any) -> MemoryQuerySpec | None:
    if not isinstance(result, dict):
        return None
    keys: list[str] = []
    regime_label = _normalize_key(result.get("regime_label"))
    if regime_label in {"risk_off", "risk_on", "neutral"}:
        keys.append(regime_label)
    regime_text = _normalize_key(result.get("regime"))
    score = _float_or_none(result.get("fear_greed_score") or result.get("regime_score"))
    vol_close = _float_or_none(result.get("volatility_close"))
    if vol_close is not None:
        if vol_close >= 25:
            keys.append("high_vol")
        elif 0 < vol_close <= 14:
            keys.append("low_vol")
    if "extreme_fear" in regime_text or (score is not None and score <= 20):
        keys.append("extreme_fear")
    elif "fear" in regime_text or (score is not None and score < 40):
        keys.append("fear")
    if "extreme_greed" in regime_text or (score is not None and score >= 80):
        keys.append("extreme_greed")
    elif "greed" in regime_text or (score is not None and score > 60):
        keys.append("greed")
    if not keys:
        return None
    vol_index = str(result.get("volatility_index") or "").strip().upper()
    return _make_spec("fear_greed_index", "regime", keys, ["market regime", " ".join(keys), vol_index])


def _index_snapshot(args: dict, result: Any) -> MemoryQuerySpec | None:
    if not isinstance(result, dict):
        return None
    rows = result.get("indices") or []
    if not isinstance(rows, list) or not rows:
        return None
    keys: list[str] = []
    symbols: list[str] = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if symbol:
            symbols.append(symbol)
        row_type = _normalize_key(row.get("type") or row.get("name"))
        value = _float_or_none(row.get("close") if row.get("close") is not None else row.get("value"))
        if symbol in {"VIX", "VKOSPI"} and value is not None and value >= 25:
            keys.extend(["risk_off", "high_vol"])
        if row_type in {"bond_yield", "yield", "rates"} or symbol in {"TNX", "DGS10", "US10Y"}:
            keys.append("rates")
            if value is not None and value >= 4.5:
                keys.append("high_yields")
        if row_type in {"index", "market_index"}:
            keys.append("market_index")
    if not keys and symbols:
        keys.append("market_index")
    return _make_spec("index_snapshot", "regime", keys, ["market index regime", " ".join(keys), " ".join(symbols[:5])])


_THEME_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ai_capex", ("ai capex", "gpu capex", "datacenter capex", "data center capex")),
    ("supply_chain", ("supply chain", "supply-chain", "bottleneck")),
    ("geopolitical", ("geopolitical", "war", "sanction", "tariff")),
    ("rates", ("rate cut", "rate hike", "yields", "treasury")),
    ("earnings_revision", ("earnings revision", "estimate revision", "guidance")),
    ("retail_sentiment", ("wallstreetbets", "reddit", "retail sentiment", "meme stock")),
)


def _theme_keys_from_text(text: str) -> list[str]:
    haystack = str(text or "").lower()
    keys: list[str] = []
    for key, patterns in _THEME_PATTERNS:
        if any(pattern in haystack for pattern in patterns):
            keys.append(key)
    return keys


def _get_research_briefing(args: dict, result: Any) -> MemoryQuerySpec | None:
    rows = result if isinstance(result, list) else []
    if not rows and isinstance(result, dict):
        rows = result.get("rows") or result.get("briefings") or []
    if not isinstance(rows, list) or not rows:
        return None
    categories: list[str] = []
    tickers: list[str] = []
    text_parts: list[str] = []
    for row in rows[:6]:
        if not isinstance(row, dict):
            continue
        category = _normalize_key(row.get("category"))
        if category:
            categories.append(category)
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker:
            tickers.append(ticker)
        text_parts.extend([str(row.get("headline") or ""), str(row.get("summary") or "")])
    themes = _theme_keys_from_text(" ".join(text_parts))
    keys = [*categories[:3], *themes[:4]]
    query_parts = [*keys, *_dedupe(tickers, limit=3)]
    return _make_spec(
        "get_research_briefing",
        "theme",
        keys,
        ["research theme", " ".join(query_parts)],
        context_keys=tickers[:3],
    )


def _earnings_calendar(args: dict, result: Any) -> MemoryQuerySpec | None:
    if not isinstance(result, dict):
        return None
    rows = result.get("rows") or []
    if not isinstance(rows, list):
        rows = []
    event_classes: list[str] = []
    for row in rows[:12]:
        if not isinstance(row, dict):
            continue
        event_type = _normalize_key(row.get("event_type"))
        if event_type:
            event_classes.append(event_type)
    keys = _dedupe(event_classes, limit=4)
    if not keys:
        return None
    ticker_keys = _ticker_list_from_args_result(args, result, rows)[:3]
    return _make_spec(
        "earnings_calendar",
        "event_class",
        keys,
        ["calendar event", " ".join([*keys, *ticker_keys])],
        context_keys=ticker_keys,
    )


def _fetch_reddit_sentiment(args: dict, result: Any) -> MemoryQuerySpec | None:
    if isinstance(result, dict):
        rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    else:
        rows = result if isinstance(result, list) else []
    if not isinstance(rows, list):
        return None
    ticker_keys = _ticker_list_from_args_result(args, result, rows)
    text = " ".join(
        " ".join([str(row.get("title") or ""), str(row.get("selftext_snippet") or ""), str(row.get("subreddit") or "")])
        for row in rows[:8]
        if isinstance(row, dict)
    )
    themes = _theme_keys_from_text(text)
    keys = [token for token in ["social_sentiment", *themes] if token]
    return _make_spec(
        "fetch_reddit_sentiment",
        "theme",
        keys,
        ["social sentiment", " ".join(ticker_keys), " ".join(themes)],
        context_keys=ticker_keys,
    )


def _fetch_sec_filings(args: dict, result: Any) -> MemoryQuerySpec | None:
    if isinstance(result, dict):
        rows = result.get("rows") if isinstance(result.get("rows"), list) else []
    else:
        rows = result if isinstance(result, list) else []
    if not isinstance(rows, list):
        return None
    form_types: list[str] = []
    entities: list[str] = []
    for row in rows[:8]:
        if not isinstance(row, dict):
            continue
        form = _normalize_key(row.get("form_type") or row.get("form"))
        if form:
            form_types.append(form)
        entity = str(row.get("entity") or "").strip()
        if entity:
            entities.append(entity)
    ticker = str(args.get("ticker") or "").strip().upper()
    form_arg = _normalize_key(args.get("filing_type"))
    if form_arg:
        form_types.insert(0, form_arg)
    keys = _dedupe(form_types, limit=4)
    if not keys:
        return None
    ticker_keys = _ticker_list_from_args_result(args, result, rows)
    if ticker and ticker not in ticker_keys:
        ticker_keys.insert(0, ticker)
    context_keys = [*ticker_keys[:3], *_dedupe(entities, limit=3)]
    return _make_spec(
        "fetch_sec_filings",
        "event_class",
        keys,
        ["filing event", " ".join([*keys, *ticker_keys[:3]]), " ".join(_dedupe(entities, limit=3))],
        context_keys=context_keys,
    )


_BUILDERS: dict[str, Any] = {
    "technical_signals": _technical_signals,
    "recommend_opportunities": _recommend_opportunities,
    "screen_market": _screen_market,
    "forecast_returns": _forecast_returns,
    "get_fundamentals": _get_fundamentals,
    "optimize_portfolio": _optimize_portfolio,
    "index_snapshot": _index_snapshot,
    "fear_greed_index": _fear_greed_index,
    "earnings_calendar": _earnings_calendar,
    "fetch_reddit_sentiment": _fetch_reddit_sentiment,
    "fetch_sec_filings": _fetch_sec_filings,
    "macro_snapshot": _macro_snapshot,
    "get_research_briefing": _get_research_briefing,
}
