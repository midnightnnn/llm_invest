from __future__ import annotations

from arena.market_sources import live_market_sources_for_markets, parse_markets


def test_parse_markets_preserves_order_and_dedupes() -> None:
    assert parse_markets(" nasdaq, kospi, nasdaq ,, us ") == ["nasdaq", "kospi", "us"]


def test_live_market_sources_for_combo_include_quote_sources_first() -> None:
    assert live_market_sources_for_markets(["nasdaq", "kospi"]) == [
        "open_trading_nasdaq_quote",
        "open_trading_nasdaq",
        "open_trading_us_quote",
        "open_trading_us",
        "open_trading_kospi_quote",
        "open_trading_kospi",
    ]
