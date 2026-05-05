from __future__ import annotations

import logging

from arena.broker.open_trading import KISOpenTradingBroker
from arena.models import OrderIntent, Side
from tests.trading.open_trading_broker_helpers import _settings


def test_to_order_payload_rounds_kospi_buy_to_tick() -> None:
    settings = _settings()
    settings.kis_target_market = "kospi"
    settings.live_slippage_bps_base = 8.0
    settings.live_slippage_bps_impact = 12.0
    settings.live_slippage_bps_max = 80.0
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="claude",
        ticker="001510",
        side=Side.BUY,
        quantity=94.0,
        price_krw=2375.0,
        rationale="test",
        fx_rate=1.0,
    )

    market, qty, local_limit, limit_krw, *_rest = broker._to_order_payload(intent)

    assert market == "kospi"
    assert qty == 94
    assert local_limit == 2380.0
    assert limit_krw == 2380.0


def test_place_order_adjusts_kospi_limit_to_live_tick_and_logs_tenant(monkeypatch, caplog) -> None:
    settings = _settings()
    settings.kis_target_market = "kospi"
    settings.live_slippage_bps_base = 8.0
    settings.live_slippage_bps_impact = 12.0
    settings.live_slippage_bps_max = 80.0
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="claude",
        ticker="001510",
        side=Side.BUY,
        quantity=94.0,
        price_krw=2375.0,
        rationale="test",
        fx_rate=1.0,
    )

    monkeypatch.setenv("ARENA_TENANT_ID", "cxznms")
    monkeypatch.setattr(
        broker.client,
        "get_domestic_price",
        lambda **kw: {"stck_prpr": "2390"},
    )

    captured: dict = {}

    def _fake_place(**kwargs):
        captured.update(kwargs)
        return {"output": {"ODNO": "krx123"}, "msg1": "ok"}

    monkeypatch.setattr(broker.client, "place_domestic_cash_order", _fake_place)

    with caplog.at_level(logging.INFO):
        report = broker.place_order(intent)

    assert captured["limit_price"] == 2395.0
    assert "tenant=cxznms" in caplog.text
    assert report.order_id == "krx123"


def test_place_order_error_preserves_attempted_kospi_limit(monkeypatch) -> None:
    settings = _settings()
    settings.kis_target_market = "kospi"
    settings.live_slippage_bps_base = 8.0
    settings.live_slippage_bps_impact = 12.0
    settings.live_slippage_bps_max = 80.0
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="claude",
        ticker="001510",
        side=Side.SELL,
        quantity=10.0,
        price_krw=1880.0,
        rationale="거래정지 리스크 축소.",
        fx_rate=1.0,
    )

    monkeypatch.setattr(
        broker.client,
        "get_domestic_price",
        lambda **kw: {"stck_prpr": "1863"},
    )

    def _fail_place(**kwargs):
        raise RuntimeError("거래정지종목(주식)은 취소주문만 가능(정정불가)합니다.")

    monkeypatch.setattr(broker.client, "place_domestic_cash_order", _fail_place)

    report = broker.place_order(intent)

    assert report.status.value == "ERROR"
    assert report.avg_price_krw == 1861.0
    assert report.avg_price_native == 1861.0
    assert "거래정지종목" in report.message
