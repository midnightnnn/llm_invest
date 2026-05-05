from __future__ import annotations

import pytest

from arena.agents.adk_order_support import fetch_market_row_from_bq, resolve_order_price
from arena.config import load_settings
from tests.adk.order_support_helpers import _RepoForMarketLookup


def test_fetch_market_row_from_bq_uses_live_kospi_sources() -> None:
    settings = load_settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "kospi"
    repo = _RepoForMarketLookup(
        [
            {"ticker": "005930", "close_price_krw": 70500.0, "close_price_native": 70500.0},
        ]
    )

    row = fetch_market_row_from_bq(repo, settings, "005930")

    assert row is not None
    assert row["ticker"] == "005930"
    assert repo.calls[0]["sources"] == ["open_trading_kospi_quote", "open_trading_kospi"]


def test_resolve_order_price_prefers_live_fx_for_us_quotes() -> None:
    settings = load_settings()
    settings.kis_target_market = "nasdaq"
    settings.usd_krw_rate = 1300.0

    price_krw, native_price, quote_currency, fx_rate = resolve_order_price(
        settings,
        market_row={
            "close_price_native": 10.0,
            "fx_rate_used": 1250.0,
        },
        portfolio={"usd_krw_rate": 1400.0},
    )

    assert price_krw == pytest.approx(14000.0)
    assert native_price == pytest.approx(10.0)
    assert quote_currency == "USD"
    assert fx_rate == pytest.approx(1400.0)


def test_resolve_order_price_returns_zero_when_us_fx_is_missing() -> None:
    settings = load_settings()
    settings.kis_target_market = "nasdaq"
    settings.usd_krw_rate = 1300.0

    price_krw, native_price, quote_currency, fx_rate = resolve_order_price(
        settings,
        market_row={
            "close_price_native": 10.0,
            "fx_rate_used": 0.0,
        },
        portfolio={"usd_krw_rate": 0.0},
    )

    assert price_krw == pytest.approx(0.0)
    assert native_price == pytest.approx(10.0)
    assert quote_currency == "USD"
    assert fx_rate == pytest.approx(0.0)


def test_resolve_order_price_multi_market_infers_usd_from_exchange_identity() -> None:
    settings = load_settings()
    settings.kis_target_market = "us,kospi"

    price_krw, native_price, quote_currency, fx_rate = resolve_order_price(
        settings,
        market_row={
            "ticker": "AAPL",
            "exchange_code": "NAS",
            "instrument_id": "NASD:AAPL",
            "close_price_native": 100.0,
            "fx_rate_used": 1300.0,
        },
        portfolio={},
    )

    assert price_krw == 130000.0
    assert native_price == 100.0
    assert quote_currency == "USD"
    assert fx_rate == 1300.0
