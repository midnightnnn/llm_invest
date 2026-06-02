from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.broker.open_trading import KISOpenTradingBroker
from tests.trading.open_trading_broker_helpers import _settings


def test_query_fill_once_converts_nasdaq_price_to_krw(monkeypatch) -> None:
    broker = KISOpenTradingBroker(settings=_settings())

    def _fake_inquire(**kwargs):
        _ = kwargs
        return [{"ODNO": "123", "CCLD_QTY": "2", "CCLD_UNPR": "50.5"}]

    monkeypatch.setattr(broker.client, "inquire_overseas_ccnl", _fake_inquire)

    report = broker._query_fill_once(
        market="us",
        order_id="123",
        ticker="AAPL",
        qty=2,
        fallback_price_krw=100_000,
        message="confirmed",
        fx_rate=1300.0,
    )

    assert report is not None
    assert report.filled_qty == 2
    assert report.avg_price_krw == 50.5 * 1300.0


def test_query_fill_once_uses_explicit_fx_rate(monkeypatch) -> None:
    broker = KISOpenTradingBroker(settings=_settings())

    def _fake_inquire(**kwargs):
        _ = kwargs
        return [{"ODNO": "123", "CCLD_QTY": "2", "CCLD_UNPR": "50.5"}]

    monkeypatch.setattr(broker.client, "inquire_overseas_ccnl", _fake_inquire)

    report = broker._query_fill_once(
        market="us",
        order_id="123",
        ticker="AAPL",
        qty=2,
        fallback_price_krw=100_000,
        message="confirmed",
        fx_rate=1450.0,
    )

    assert report is not None
    assert report.avg_price_krw == pytest.approx(50.5 * 1450.0)
    assert report.fx_rate == pytest.approx(1450.0)


def test_query_fill_once_scans_multiple_us_exchanges(monkeypatch) -> None:
    broker = KISOpenTradingBroker(settings=_settings())
    calls: list[str] = []

    def _fake_inquire(**kwargs):
        exchange = str(kwargs.get("exchange_code") or "")
        calls.append(exchange)
        if exchange == "NYSE":
            return [{"ODNO": "777", "CCLD_QTY": "1", "CCLD_UNPR": "20.0"}]
        return []

    monkeypatch.setattr(broker.client, "inquire_overseas_ccnl", _fake_inquire)

    report = broker._query_fill_once(
        market="us",
        order_id="777",
        ticker="EXC",
        qty=1,
        fallback_price_krw=100_000,
        message="confirmed",
        exchange_code="NASD",
        fx_rate=1300.0,
    )

    assert report is not None
    assert report.status.value == "FILLED"
    assert report.filled_qty == 1
    assert "NASD" in calls
    assert "NYSE" in calls


def test_reconcile_submitted_uses_fill_lookup(monkeypatch) -> None:
    broker = KISOpenTradingBroker(settings=_settings())

    def _fake_query_fill_once(**kwargs):
        _ = kwargs
        from arena.models import ExecutionReport, ExecutionStatus, utc_now

        return ExecutionReport(
            status=ExecutionStatus.FILLED,
            order_id="abc",
            filled_qty=3,
            avg_price_krw=99_000,
            message="reconciled",
            created_at=utc_now(),
        )

    monkeypatch.setattr(broker, "_query_fill_once", _fake_query_fill_once)

    report = broker.reconcile_submitted(
        order_id="abc",
        ticker="AAPL",
        side="BUY",
        requested_qty=3.9,
        fallback_price_krw=98_000,
    )

    assert report is not None
    assert report.status.value == "FILLED"


def test_reconcile_submitted_scans_from_submitted_kst_date_for_kospi(monkeypatch) -> None:
    settings = _settings()
    settings.kis_target_market = "kospi"
    broker = KISOpenTradingBroker(settings=settings)
    calls: list[dict[str, str]] = []

    def _fake_inquire(**kwargs):
        calls.append(dict(kwargs))
        return [{"odno": "0058661300", "tot_ccld_qty": "50", "avg_prvs": "2130"}]

    monkeypatch.setattr(broker.client, "inquire_domestic_daily_ccld", _fake_inquire)

    report = broker.reconcile_submitted(
        order_id="0058661300",
        ticker="010580",
        side="BUY",
        requested_qty=50,
        fallback_price_krw=2130,
        submitted_at=datetime(2026, 6, 1, 5, 40, 1, tzinfo=timezone.utc),
    )

    assert report is not None
    assert report.status.value == "FILLED"
    assert calls[0]["start_date"] == "20260601"
    assert calls[0]["end_date"] >= "20260601"
