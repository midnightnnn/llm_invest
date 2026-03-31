from __future__ import annotations

from collections.abc import Iterable

_US_TARGET_TO_ORDER_EXCHANGE: dict[str, str] = {
    "nasdaq": "NASD",
    "nyse": "NYSE",
    "amex": "AMEX",
}
_CANONICAL_US_QUOTE_EXCHANGES: tuple[str, ...] = ("NAS", "NYS", "AMS")
_CANONICAL_US_ORDER_EXCHANGES: tuple[str, ...] = ("NASD", "NYSE", "AMEX")


def _iter_tokens(values: tuple[object, ...]) -> Iterable[object]:
    for value in values:
        if isinstance(value, (list, tuple, set)):
            yield from value
            continue
        yield value


def normalize_us_quote_exchange(exchange_code: object) -> str:
    token = str(exchange_code or "").strip().upper()
    if token in {"NAS", "NASD"}:
        return "NAS"
    if token in {"NYS", "NYSE"}:
        return "NYS"
    if token in {"AMS", "AMEX"}:
        return "AMS"
    return ""


def normalize_us_order_exchange(exchange_code: object) -> str:
    token = str(exchange_code or "").strip().upper()
    if token in {"NAS", "NASD"}:
        return "NASD"
    if token in {"NYS", "NYSE"}:
        return "NYSE"
    if token in {"AMS", "AMEX"}:
        return "AMEX"
    return ""


def quote_to_order_exchange(exchange_code: object) -> str:
    return normalize_us_order_exchange(exchange_code)


def order_to_quote_exchange(exchange_code: object) -> str:
    return normalize_us_quote_exchange(exchange_code)


def instrument_id_us_order_exchange(instrument_id: object) -> str:
    token = str(instrument_id or "").strip()
    if not token or ":" not in token:
        return ""
    prefix = token.split(":", 1)[0]
    return normalize_us_order_exchange(prefix)


def parse_target_markets(target_market: object) -> list[str]:
    parts = [str(part).strip().lower() for part in str(target_market or "").split(",")]
    return list(dict.fromkeys([part for part in parts if part]))


def target_market_default_us_order_exchange(target_market: object) -> str:
    markets = [market for market in parse_target_markets(target_market) if market in {"nasdaq", "nyse", "amex", "us"}]
    if len(markets) != 1:
        return ""
    return _US_TARGET_TO_ORDER_EXCHANGE.get(markets[0], "")


def target_market_default_us_quote_exchange(target_market: object) -> str:
    return order_to_quote_exchange(target_market_default_us_order_exchange(target_market))


def us_quote_exchange_candidates(*tokens: object) -> list[str]:
    out: list[str] = []
    for token in _iter_tokens(tokens):
        normalized = normalize_us_quote_exchange(token)
        if normalized and normalized not in out:
            out.append(normalized)
    for token in _CANONICAL_US_QUOTE_EXCHANGES:
        if token not in out:
            out.append(token)
    return out


def us_order_exchange_candidates(*tokens: object) -> list[str]:
    out: list[str] = []
    for token in _iter_tokens(tokens):
        normalized = normalize_us_order_exchange(token)
        if normalized and normalized not in out:
            out.append(normalized)
    for token in _CANONICAL_US_ORDER_EXCHANGES:
        if token not in out:
            out.append(token)
    return out
