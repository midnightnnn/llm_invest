from __future__ import annotations

import re
from typing import Any

from arena.tools.sector_map import SECTOR_BY_TICKER

_REGIME_ALIASES = {
    "bull": "bull",
    "bullish": "bull",
    "risk_on": "bull",
    "uptrend": "bull",
    "bear": "bear",
    "bearish": "bear",
    "risk_off": "bear",
    "defensive": "bear",
    "sideways": "sideways",
    "neutral": "sideways",
    "range": "sideways",
    "range_bound": "sideways",
    "choppy": "sideways",
    "high_vol": "high_vol",
    "high_volatility": "high_vol",
    "volatile": "high_vol",
    "low_vol": "low_vol",
    "low_volatility": "low_vol",
    "calm": "low_vol",
}

_STRATEGY_ALIASES = {
    "momentum": "momentum",
    "trend": "momentum",
    "trend_following": "momentum",
    "continuation": "momentum",
    "breakout": "breakout",
    "breakouts": "breakout",
    "mean_reversion": "mean_reversion",
    "reversion": "mean_reversion",
    "oversold": "mean_reversion",
    "dip_buy": "mean_reversion",
    "pullback": "mean_reversion",
    "sizing": "sizing",
    "position_sizing": "sizing",
    "rebalancing": "rebalancing",
}

_SECTOR_ALIASES = {
    "technology": "tech",
    "tech": "tech",
    "energy": "energy",
    "health_care": "healthcare",
    "healthcare": "healthcare",
    "health": "healthcare",
    "financial": "financials",
    "financials": "financials",
    "consumer": "consumer",
    "consumer_discretionary": "consumer",
    "consumer_staples": "consumer",
    "industrials": "industrials",
    "industrial": "industrials",
    "materials": "materials",
    "utilities": "utilities",
    "communication_services": "communication_services",
    "communication": "communication_services",
    "real_estate": "real_estate",
}

_TICKER_TOKEN_RE = re.compile(r"\b(?:[A-Z][A-Z0-9]{1,5}|\d{6})\b")
_IGNORED_TICKER_TOKENS = {"BUY", "SELL", "HOLD", "USD", "KRW"}


def _dedupe(tokens: list[str], *, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        clean = str(token or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        result.append(clean)
        if limit is not None and len(result) >= limit:
            break
    return result


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if value is None:
        return []
    if isinstance(value, str) and not value.strip():
        return []
    return [value]


def normalize_tag_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    return re.sub(r"[\s_]+", "_", text).strip("_")


def canonical_regime_tag(value: Any) -> str:
    return _REGIME_ALIASES.get(normalize_tag_token(value), "")


def canonical_strategy_tag(value: Any) -> str:
    return _STRATEGY_ALIASES.get(normalize_tag_token(value), "")


def canonical_sector_tag(value: Any) -> str:
    return _SECTOR_ALIASES.get(normalize_tag_token(value), "")


def sector_tag_for_ticker(ticker: Any) -> str:
    token = str(ticker or "").strip().upper()
    if not token:
        return ""
    return canonical_sector_tag(SECTOR_BY_TICKER.get(token, ""))


def _collect_tickers_from_value(value: Any, tickers: list[str], *, depth: int = 0) -> None:
    if depth > 3 or len(tickers) >= 6:
        return
    if isinstance(value, dict):
        ticker = value.get("ticker")
        if isinstance(ticker, str):
            token = ticker.strip().upper()
            if token:
                tickers.append(token)
        raw_tickers = value.get("tickers")
        if isinstance(raw_tickers, list):
            for token in raw_tickers:
                clean = str(token or "").strip().upper()
                if clean:
                    tickers.append(clean)
        for nested in value.values():
            _collect_tickers_from_value(nested, tickers, depth=depth + 1)
        return
    if isinstance(value, list):
        for item in value[:8]:
            _collect_tickers_from_value(item, tickers, depth=depth + 1)


def _extract_tickers(payload: dict[str, Any], summary: str) -> list[str]:
    tickers: list[str] = []
    _collect_tickers_from_value(payload, tickers)
    if summary:
        for token in _TICKER_TOKEN_RE.findall(summary):
            if token in _IGNORED_TICKER_TOKENS:
                continue
            tickers.append(token)
    return _dedupe([str(token or "").strip().upper() for token in tickers if str(token or "").strip()], limit=4)


def _classify_tag(token: Any, *, regimes: list[str], strategies: list[str], sectors: list[str], extras: list[str]) -> None:
    raw = normalize_tag_token(token)
    if not raw:
        return
    regime = canonical_regime_tag(raw)
    if regime:
        regimes.append(regime)
        return
    strategy = canonical_strategy_tag(raw)
    if strategy:
        strategies.append(strategy)
        return
    sector = canonical_sector_tag(raw)
    if sector:
        sectors.append(sector)
        return
    extras.append(raw)


def normalize_context_tags(
    raw: Any,
    *,
    primary_regime: Any = None,
    primary_strategy_tag: Any = None,
    primary_sector: Any = None,
    max_tags: int = 6,
) -> dict[str, Any]:
    regimes: list[str] = []
    strategies: list[str] = []
    sectors: list[str] = []
    extras: list[str] = []
    tickers: list[str] = []
    source = ""

    if isinstance(raw, dict):
        source = str(raw.get("source") or "").strip()
        for token in _sequence(raw.get("regimes")):
            _classify_tag(token, regimes=regimes, strategies=strategies, sectors=sectors, extras=extras)
        for token in _sequence(raw.get("strategies")):
            _classify_tag(token, regimes=regimes, strategies=strategies, sectors=sectors, extras=extras)
        for token in _sequence(raw.get("sectors")):
            _classify_tag(token, regimes=regimes, strategies=strategies, sectors=sectors, extras=extras)
        for token in _sequence(raw.get("tags")) + _sequence(raw.get("legacy_tags")):
            _classify_tag(token, regimes=regimes, strategies=strategies, sectors=sectors, extras=extras)
        tickers = [
            str(token or "").strip().upper()
            for token in _sequence(raw.get("tickers"))
            if str(token or "").strip()
        ]
    else:
        for token in _sequence(raw):
            _classify_tag(token, regimes=regimes, strategies=strategies, sectors=sectors, extras=extras)

    _classify_tag(primary_regime, regimes=regimes, strategies=strategies, sectors=sectors, extras=extras)
    _classify_tag(primary_strategy_tag, regimes=regimes, strategies=strategies, sectors=sectors, extras=extras)
    _classify_tag(primary_sector, regimes=regimes, strategies=strategies, sectors=sectors, extras=extras)

    regimes = _dedupe(regimes)
    strategies = _dedupe(strategies)
    sectors = _dedupe(sectors)
    extras = _dedupe(extras)
    tickers = _dedupe(tickers, limit=4)
    tags = _dedupe(regimes + strategies + sectors + extras, limit=max(1, int(max_tags)))

    result: dict[str, Any] = {}
    if tags:
        result["tags"] = tags
    if regimes:
        result["regimes"] = regimes
    if strategies:
        result["strategies"] = strategies
    if sectors:
        result["sectors"] = sectors
    if tickers:
        result["tickers"] = tickers
    if source:
        result["source"] = source
    return result


def extract_context_tags(
    *,
    event_type: str,
    summary: str,
    payload: dict[str, Any] | None,
    max_tags: int = 6,
) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    primary_regime = data.get("primary_regime") or data.get("market_regime") or data.get("regime")
    primary_strategy_tag = data.get("primary_strategy_tag") or data.get("strategy_tag") or data.get("strategy")
    primary_sector = data.get("primary_sector") or data.get("sector")
    normalized = normalize_context_tags(
        data.get("context_tags"),
        primary_regime=primary_regime,
        primary_strategy_tag=primary_strategy_tag,
        primary_sector=primary_sector,
        max_tags=max_tags,
    )

    regimes = list(normalized.get("regimes") or [])
    strategies = list(normalized.get("strategies") or [])
    sectors = list(normalized.get("sectors") or [])
    extras = [
        str(token or "").strip()
        for token in (normalized.get("tags") or [])
        if str(token or "").strip() not in set(regimes + strategies + sectors)
    ]
    tickers = list(normalized.get("tickers") or [])

    for token in _sequence(data.get("tags")):
        _classify_tag(token, regimes=regimes, strategies=strategies, sectors=sectors, extras=extras)

    rationale = ""
    intent = data.get("intent")
    if isinstance(intent, dict):
        rationale = str(intent.get("rationale") or "").strip()

    text = normalize_tag_token(" ".join(part for part in [summary, rationale, str(event_type or "")] if part))
    text_space = f" {text.replace('_', ' ')} "
    for token, canonical in _REGIME_ALIASES.items():
        phrase = token.replace("_", " ")
        if f" {phrase} " in text_space:
            regimes.append(canonical)
    for token, canonical in _STRATEGY_ALIASES.items():
        phrase = token.replace("_", " ")
        if f" {phrase} " in text_space:
            strategies.append(canonical)

    tickers.extend(_extract_tickers(data, summary))
    for ticker in tickers:
        sector = sector_tag_for_ticker(ticker)
        if sector:
            sectors.append(sector)

    regimes = _dedupe(regimes)
    strategies = _dedupe(strategies)
    sectors = _dedupe(sectors)
    extras = _dedupe(extras)
    tickers = _dedupe(tickers, limit=4)

    source = str(normalized.get("source") or "").strip()
    if not source:
        if normalized or data.get("tags"):
            source = "payload"
        elif regimes or strategies or sectors or tickers:
            source = "heuristic"

    return normalize_context_tags(
        {
            "tags": regimes + strategies + sectors + extras,
            "regimes": regimes,
            "strategies": strategies,
            "sectors": sectors,
            "tickers": tickers,
            "source": source,
        },
        max_tags=max_tags,
    )
