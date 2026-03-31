from __future__ import annotations

from collections.abc import Iterable

US_MARKETS: frozenset[str] = frozenset({"nasdaq", "nyse", "amex", "us"})
KOSPI_MARKETS: frozenset[str] = frozenset({"kospi"})

LIVE_MARKET_SOURCES_BY_MARKET: dict[str, tuple[str, ...]] = {
    "nasdaq": ("open_trading_nasdaq_quote", "open_trading_nasdaq", "open_trading_us_quote", "open_trading_us"),
    "nyse": ("open_trading_nyse_quote", "open_trading_nyse", "open_trading_us_quote", "open_trading_us"),
    "amex": ("open_trading_amex_quote", "open_trading_amex", "open_trading_us_quote", "open_trading_us"),
    "us": (
        "open_trading_us_quote",
        "open_trading_us",
        "open_trading_nasdaq_quote",
        "open_trading_nasdaq",
        "open_trading_nyse_quote",
        "open_trading_nyse",
        "open_trading_amex_quote",
        "open_trading_amex",
    ),
    "kospi": ("open_trading_kospi_quote", "open_trading_kospi"),
}


def parse_markets(markets: str | Iterable[str] | None) -> list[str]:
    """Parses comma-separated or iterable market values into a deduped list."""
    if markets is None:
        return []
    if isinstance(markets, str):
        tokens = markets.split(",")
    elif isinstance(markets, (set, frozenset)):
        tokens = sorted(str(token) for token in markets)
    else:
        tokens = list(markets)

    out: list[str] = []
    for token in tokens:
        market = str(token or "").strip().lower()
        if market and market not in out:
            out.append(market)
    return out


def live_market_sources_for_markets(markets: str | Iterable[str] | None) -> list[str]:
    """Returns quote-aware live market sources in stable priority order."""
    combined: list[str] = []
    for market in parse_markets(markets):
        for source in LIVE_MARKET_SOURCES_BY_MARKET.get(market, ()):
            if source not in combined:
                combined.append(source)
    return combined
