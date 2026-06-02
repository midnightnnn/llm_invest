from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.config import Settings
from arena.open_trading.sync import AccountSyncService, BrokerCashSyncService, BrokerTradeSyncService, MarketDataSyncService

from tests.trading.open_trading_sync_helpers import (
    FakeRepo,
    FakeClient,
    FakeBrokerTradeRepo,
    FakeBrokerCashRepo,
    FakeBrokerTradeClient,
    _settings,
)

def test_broker_trade_sync_normalizes_overseas_rows_and_is_idempotent() -> None:
    repo = FakeBrokerTradeRepo()
    settings = _settings("us", ["AAPL"])
    client = FakeBrokerTradeClient(
        overseas={
            "NASD": [
                {
                    "ODNO": "12345",
                    "PDNO": "AAPL",
                    "SLL_BUY_DVSN": "02",
                    "CCLD_QTY": "2",
                    "CCLD_UNPR": "100.50",
                    "ORD_DT": "20260311",
                    "ORD_TMD": "153045",
                    "bass_exrt": "1460.2",
                }
            ]
        }
    )
    service = BrokerTradeSyncService(settings=settings, repo=repo, client=client)

    first = service.sync_broker_trade_events(days=3)

    assert first.inserted_events == 1
    assert first.scanned_rows == 1
    assert first.skipped_existing == 0
    assert first.failed_scopes == []
    assert client.overseas_calls == ["NASD", "NYSE", "AMEX"]
    assert len(repo.appended_trade_rows) == 1
    row = repo.appended_trade_rows[0]
    assert row["broker_order_id"] == "12345"
    assert row["ticker"] == "AAPL"
    assert row["exchange_code"] == "NASD"
    assert row["instrument_id"] == "NASD:AAPL"
    assert row["side"] == "BUY"
    assert row["quantity"] == pytest.approx(2.0)
    assert row["price_native"] == pytest.approx(100.50)
    assert row["fx_rate"] == pytest.approx(1460.2)
    assert row["price_krw"] == pytest.approx(100.50 * 1460.2)
    assert row["quote_currency"] == "USD"
    assert row["source"] == "kis_inquire_overseas_ccnl"
    assert row["occurred_at"] == datetime(2026, 3, 11, 15, 30, 45, tzinfo=timezone.utc)

    second = service.sync_broker_trade_events(days=3)

    assert second.inserted_events == 0
    assert second.scanned_rows == 1
    assert second.skipped_existing == 1
    assert len(repo.appended_trade_rows) == 1


def test_broker_trade_sync_uses_api_fx_even_below_config() -> None:
    """When API bass_exrt=1455 and config=1460, API value must win (not max())."""
    repo = FakeBrokerTradeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_rate = 1460.0
    client = FakeBrokerTradeClient(
        overseas={
            "NASD": [
                {
                    "ODNO": "99",
                    "PDNO": "AAPL",
                    "SLL_BUY_DVSN_CD": "02",
                    "TOT_CCLD_QTY": "1",
                    "AVG_UNPR": "200.0",
                    "ORD_DT": "20260315",
                    "ORD_TMD": "100000",
                    "bass_exrt": "1455.0",
                }
            ]
        }
    )
    service = BrokerTradeSyncService(settings=settings, repo=repo, client=client)
    service.sync_broker_trade_events(days=1)

    assert len(repo.appended_trade_rows) == 1
    row = repo.appended_trade_rows[0]
    assert row["fx_rate"] == pytest.approx(1455.0)
    assert row["price_krw"] == pytest.approx(200.0 * 1455.0)


def test_broker_trade_sync_skips_when_no_fx_available() -> None:
    """When neither API nor period_trans provides FX, the row is skipped (not recorded with bad rate)."""
    repo = FakeBrokerTradeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_rate = 1460.0
    client = FakeBrokerTradeClient(
        overseas={
            "NASD": [
                {
                    "ODNO": "100",
                    "PDNO": "AAPL",
                    "SLL_BUY_DVSN_CD": "02",
                    "TOT_CCLD_QTY": "1",
                    "AVG_UNPR": "200.0",
                    "ORD_DT": "20260315",
                    "ORD_TMD": "100000",
                }
            ]
        }
    )
    service = BrokerTradeSyncService(settings=settings, repo=repo, client=client)
    result = service.sync_broker_trade_events(days=1)

    assert len(repo.appended_trade_rows) == 0
    assert result.inserted_events == 0


def test_broker_trade_sync_normalizes_domestic_rows() -> None:
    repo = FakeBrokerTradeRepo()
    settings = _settings("kospi", ["005930"])
    client = FakeBrokerTradeClient(
        domestic=[
            {
                "ODNO": "8899",
                "PDNO": "005930",
                "SLL_BUY_DVSN_CD": "01",
                "TOT_CCLD_QTY": "3",
                "AVG_UNPR": "71200",
                "ORD_DT": "20260311",
                "ORD_TMD": "091500",
            }
        ]
    )
    service = BrokerTradeSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_broker_trade_events(days=2)

    assert result.inserted_events == 1
    assert result.scanned_rows == 1
    assert client.domestic_calls
    row = repo.appended_trade_rows[0]
    assert row["broker_order_id"] == "8899"
    assert row["ticker"] == "005930"
    assert row["exchange_code"] == "KRX"
    assert row["instrument_id"] == "KRX:005930"
    assert row["side"] == "SELL"
    assert row["quantity"] == pytest.approx(3.0)
    assert row["price_native"] == pytest.approx(71200.0)
    assert row["price_krw"] == pytest.approx(71200.0)
    assert row["quote_currency"] == "KRW"
    assert row["fx_rate"] == pytest.approx(1.0)
    assert row["source"] == "kis_inquire_domestic_daily_ccld"


def test_broker_trade_sync_normalizes_domestic_kis_price_aliases_and_kst_time() -> None:
    repo = FakeBrokerTradeRepo()
    settings = _settings("kospi", ["010580"])
    client = FakeBrokerTradeClient(
        domestic=[
            {
                "odno": "0058661300",
                "pdno": "010580",
                "sll_buy_dvsn_cd": "02",
                "tot_ccld_qty": "50",
                "tot_ccld_amt": "106500",
                "avg_prvs": "2130",
                "ord_dt": "20260601",
                "ord_tmd": "144001",
            }
        ]
    )
    service = BrokerTradeSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_broker_trade_events(days=2)

    assert result.inserted_events == 1
    row = repo.appended_trade_rows[0]
    assert row["broker_order_id"] == "0058661300"
    assert row["ticker"] == "010580"
    assert row["side"] == "BUY"
    assert row["quantity"] == pytest.approx(50.0)
    assert row["price_native"] == pytest.approx(2130.0)
    assert row["price_krw"] == pytest.approx(2130.0)
    assert row["occurred_at"] == datetime(2026, 6, 1, 5, 40, 1, tzinfo=timezone.utc)


def test_broker_trade_sync_reports_failed_scopes_without_failing_other_markets() -> None:
    repo = FakeBrokerTradeRepo()
    settings = _settings("nasdaq,kospi", ["AAPL", "005930"])
    client = FakeBrokerTradeClient(
        domestic=[
            {
                "ODNO": "1001",
                "PDNO": "005930",
                "SLL_BUY_DVSN_CD": "02",
                "CCLD_QTY": "1",
                "CCLD_UNPR": "70000",
                "ORD_DT": "20260311",
            }
        ],
        failed_exchanges={"NASD"},
    )
    service = BrokerTradeSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_broker_trade_events(days=2)

    assert result.inserted_events == 1
    assert result.failed_scopes == ["us:NASD"]
    assert repo.appended_trade_rows[0]["ticker"] == "005930"
