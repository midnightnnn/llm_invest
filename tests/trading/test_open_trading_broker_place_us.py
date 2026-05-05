from __future__ import annotations

import pytest

from arena.broker.open_trading import KISOpenTradingBroker
from arena.models import OrderIntent, Side
from tests.trading.open_trading_broker_helpers import _settings


def test_place_order_adjusts_limit_to_live_price(monkeypatch) -> None:
    """When live ask > stale limit, place_order re-anchors limit to live price + slippage."""
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="GILD",
        side=Side.BUY,
        quantity=2.0,
        price_krw=130_000,
        rationale="test",
        fx_rate=1300.0,
        exchange_code="NASD",
    )

    # Stale limit from _to_order_payload: 130_000 * (1 + 10/10000) / 1300 = 100.10 USD
    # Live price is 102.50 USD — higher than stale limit
    monkeypatch.setattr(
        broker.client, "get_overseas_price",
        lambda **kw: {"last": "102.50"},
    )

    captured: dict = {}

    def _fake_place(**kwargs):
        captured.update(kwargs)
        return {"output": {"ODNO": "test123"}, "msg1": "ok"}

    monkeypatch.setattr(broker.client, "place_overseas_order", _fake_place)

    report = broker.place_order(intent)

    submitted_limit = captured["limit_price"]
    # Should be ≥ live price (102.50), not stale limit (100.10)
    assert submitted_limit >= 102.50
    # Should include slippage buffer on top of live price
    expected = 102.50 * (1.0 + 10.0 / 10_000.0)  # ≈ 102.6025 → rounded to 102.61
    assert submitted_limit == pytest.approx(expected, abs=0.01)
    assert report.order_id == "test123"


def test_place_order_keeps_limit_when_live_lower(monkeypatch) -> None:
    """When live price < stale limit for BUY, no adjustment needed."""
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=130_000,
        rationale="test",
        fx_rate=1300.0,
        exchange_code="NASD",
    )

    # Live price 99.00 is below stale limit 100.10 — no adjustment needed
    monkeypatch.setattr(
        broker.client, "get_overseas_price",
        lambda **kw: {"last": "99.00"},
    )

    captured: dict = {}

    def _fake_place(**kwargs):
        captured.update(kwargs)
        return {"output": {"ODNO": "test456"}, "msg1": "ok"}

    monkeypatch.setattr(broker.client, "place_overseas_order", _fake_place)

    broker.place_order(intent)

    # Should remain at stale limit (100.10), not adjusted down
    assert captured["limit_price"] == pytest.approx(100.10, abs=0.01)


def test_place_order_tolerates_live_price_failure(monkeypatch) -> None:
    """When live quote API fails, order proceeds with stale limit."""
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=130_000,
        rationale="test",
        fx_rate=1300.0,
        exchange_code="NASD",
    )

    def _fail(**kw):
        raise ConnectionError("API down")

    monkeypatch.setattr(broker.client, "get_overseas_price", _fail)

    captured: dict = {}

    def _fake_place(**kwargs):
        captured.update(kwargs)
        return {"output": {"ODNO": "test789"}, "msg1": "ok"}

    monkeypatch.setattr(broker.client, "place_overseas_order", _fake_place)

    report = broker.place_order(intent)

    # Should fall back to stale limit
    assert captured["limit_price"] == pytest.approx(100.10, abs=0.01)
    assert report.order_id == "test789"
