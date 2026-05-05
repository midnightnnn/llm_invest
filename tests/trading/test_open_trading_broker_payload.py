from __future__ import annotations

import pytest

from arena.broker.open_trading import KISOpenTradingBroker, _normalize_us_order_exchange
from arena.models import OrderIntent, Side
from tests.trading.open_trading_broker_helpers import _settings


def test_to_order_payload_applies_buy_slippage() -> None:
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.2,
        price_krw=130_000,
        rationale="test",
        fx_rate=1300.0,
    )

    market, qty, local_limit, limit_krw, bps, order_exchange, fx_rate = broker._to_order_payload(intent)

    assert market == "us"
    assert qty == 1
    assert bps == 10.0
    assert order_exchange == "NASD"
    assert fx_rate == 1300.0
    assert local_limit == 100.1
    assert limit_krw == pytest.approx(130130.0)


def test_to_order_payload_prefers_explicit_fx_rate() -> None:
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=145_000,
        rationale="test",
        fx_rate=1300.0,
    )

    market, qty, local_limit, limit_krw, bps, order_exchange, fx_rate = broker._to_order_payload(
        intent,
        fx_rate=1450.0,
    )

    assert market == "us"
    assert qty == 1
    assert bps == 10.0
    assert order_exchange == "NASD"
    assert fx_rate == 1450.0
    assert local_limit == pytest.approx(100.1)
    assert limit_krw == pytest.approx(145145.0)


def test_to_order_payload_requires_exchange_for_mixed_us_market() -> None:
    settings = _settings()
    settings.kis_target_market = "us"
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=145_000,
        rationale="test",
        fx_rate=1300.0,
    )

    with pytest.raises(ValueError, match="unable to resolve US order exchange"):
        broker._to_order_payload(intent)


def test_to_order_payload_accepts_combo_market_with_us_exchange() -> None:
    settings = _settings()
    settings.kis_target_market = "us,kospi"
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=145_000,
        rationale="test",
        fx_rate=1300.0,
        exchange_code="NASD",
    )

    market, qty, _local_limit, _limit_krw, _bps, order_exchange, fx_rate = broker._to_order_payload(intent)

    assert market == "us"
    assert qty == 1
    assert order_exchange == "NASD"
    assert fx_rate == 1300.0


def test_to_order_payload_accepts_combo_market_with_krx_exchange() -> None:
    settings = _settings()
    settings.kis_target_market = "us,kospi"
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="005930",
        side=Side.BUY,
        quantity=1.0,
        price_krw=70_000,
        rationale="test",
        exchange_code="KRX",
        quote_currency="KRW",
    )

    market, qty, local_limit, limit_krw, _bps, order_exchange, fx_rate = broker._to_order_payload(intent)

    assert market == "kospi"
    assert qty == 1
    assert local_limit == 70100.0
    assert limit_krw == 70100.0
    assert order_exchange == "KRX"
    assert fx_rate == 1.0


def test_live_slippage_bps_is_capped() -> None:
    s = _settings()
    s.live_slippage_bps_base = 5.0
    s.live_slippage_bps_impact = 20.0
    s.live_slippage_bps_max = 30.0
    broker = KISOpenTradingBroker(settings=s)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=100.0,
        price_krw=130_000,
        rationale="test",
    )

    assert broker._live_slippage_bps(intent) == 30.0


def test_resolved_fx_rate_rejects_missing_rate() -> None:
    """Orders must not proceed with a stale config default — ValueError expected."""
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=130_000,
        rationale="test",
    )
    with pytest.raises(ValueError, match="No live USD/KRW rate"):
        broker._resolved_fx_rate(intent)


def test_resolved_fx_rate_uses_api_value_even_below_config() -> None:
    """API FX rate 1455 should be used even when config default is 1460."""
    settings = _settings()
    settings.usd_krw_rate = 1460.0
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=130_000,
        rationale="test",
        fx_rate=1455.0,
    )
    assert broker._resolved_fx_rate(intent) == 1455.0


def test_normalize_us_order_exchange_requires_resolvable_code() -> None:
    with pytest.raises(ValueError, match="unable to resolve US order exchange"):
        _normalize_us_order_exchange("UNKNOWN", "")
