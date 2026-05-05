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

def test_broker_cash_sync_normalizes_overseas_rows() -> None:
    repo = FakeBrokerCashRepo()
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
    service = BrokerCashSyncService(settings=settings, repo=repo, client=client)

    first = service.sync_broker_cash_events(days=3)

    assert first.inserted_events == 1
    assert first.scanned_rows == 1
    assert first.skipped_existing == 0
    row = repo.appended_cash_rows[0]
    assert row["currency"] == "USD"
    assert row["event_type"] == "TRADE_SETTLEMENT"
    assert row["amount_native"] == pytest.approx(-201.0)
    assert row["amount_krw"] == pytest.approx(-201.0 * 1460.2)
    assert row["source"] == "kis_inquire_overseas_ccnl"

    second = service.sync_broker_cash_events(days=3)
    assert second.inserted_events == 0
    assert second.skipped_existing == 1


def test_broker_cash_sync_skips_us_rows_without_fx() -> None:
    repo = FakeBrokerCashRepo()
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
                }
            ]
        }
    )
    service = BrokerCashSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_broker_cash_events(days=3)

    assert result.inserted_events == 0
    assert repo.appended_cash_rows == []


def test_broker_cash_sync_normalizes_domestic_rows() -> None:
    repo = FakeBrokerCashRepo()
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
    service = BrokerCashSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_broker_cash_events(days=2)

    assert result.inserted_events == 1
    assert result.scanned_rows == 1
    row = repo.appended_cash_rows[0]
    assert row["currency"] == "KRW"
    assert row["fx_rate"] == pytest.approx(1.0)
    assert row["amount_native"] == pytest.approx(3.0 * 71200.0)
    assert row["amount_krw"] == pytest.approx(3.0 * 71200.0)
    assert row["source"] == "kis_inquire_domestic_daily_ccld"


def test_broker_cash_sync_normalizes_overseas_fee_rows() -> None:
    repo = FakeBrokerCashRepo()
    settings = _settings("us", ["AAPL"])
    client = FakeBrokerTradeClient(
        overseas_period_trans={
            "NASD": [
                {
                    "trad_dt": "20260311",
                    "pdno": "AAPL",
                    "dmst_frcr_fee1": "1.25",
                    "frcr_fee1": "0.75",
                    "dmst_wcrc_fee": "150",
                    "ovrs_wcrc_fee": "50",
                    "erlm_exrt": "1465.0",
                }
            ]
        }
    )
    service = BrokerCashSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_broker_cash_events(days=3)

    assert result.inserted_events == 2
    assert {row["currency"] for row in repo.appended_cash_rows} == {"USD", "KRW"}
    usd_row = next(row for row in repo.appended_cash_rows if row["currency"] == "USD")
    krw_row = next(row for row in repo.appended_cash_rows if row["currency"] == "KRW")
    assert usd_row["event_type"] == "BROKER_FEE"
    assert usd_row["amount_native"] == pytest.approx(-2.0)
    assert usd_row["amount_krw"] == pytest.approx(-2.0 * 1465.0)
    assert krw_row["event_type"] == "BROKER_FEE"
    assert krw_row["amount_krw"] == pytest.approx(-200.0)


def test_broker_cash_sync_normalizes_domestic_fee_tax_rows() -> None:
    repo = FakeBrokerCashRepo()
    settings = _settings("kospi", ["005930"])
    client = FakeBrokerTradeClient(
        domestic_period_profit=[
            {
                "trad_dt": "20260311",
                "fee": "1250",
                "tl_tax": "900",
                "loan_int": "100",
            }
        ]
    )
    service = BrokerCashSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_broker_cash_events(days=2)

    assert result.inserted_events == 3
    event_types = {row["event_type"] for row in repo.appended_cash_rows}
    assert event_types == {"BROKER_FEE", "BROKER_TAX", "BROKER_INTEREST"}
    assert sum(float(row["amount_krw"]) for row in repo.appended_cash_rows) == pytest.approx(-(1250.0 + 900.0 + 100.0))


def test_broker_cash_sync_derives_residual_deposit_withdraw_from_account_snapshots() -> None:
    repo = FakeBrokerCashRepo()
    repo.cash_history_rows = [
        {
            "snapshot_at": datetime(2026, 3, 10, 20, 0, tzinfo=timezone.utc),
            "cash_krw": 1_460_000.0,
            "cash_foreign": 1_000.0,
            "cash_foreign_currency": "USD",
            "usd_krw_rate": 1460.0,
        },
        {
            "snapshot_at": datetime(2026, 3, 11, 20, 0, tzinfo=timezone.utc),
            "cash_krw": 1_831_250.0,
            "cash_foreign": 1_250.0,
            "cash_foreign_currency": "USD",
            "usd_krw_rate": 1465.0,
        },
        {
            "snapshot_at": datetime(2026, 3, 12, 20, 0, tzinfo=timezone.utc),
            "cash_krw": 1_831_250.0,
            "cash_foreign": 1_250.0,
            "cash_foreign_currency": "USD",
            "usd_krw_rate": 1465.0,
        },
    ]
    settings = _settings("us", ["AAPL"])
    client = FakeBrokerTradeClient()
    service = BrokerCashSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_broker_cash_events(days=3)

    assert result.inserted_events == 1
    row = repo.appended_cash_rows[0]
    assert row["event_type"] == "DEPOSIT"
    assert row["currency"] == "USD"
    assert row["amount_native"] == pytest.approx(250.0)
    assert row["amount_krw"] == pytest.approx(250.0 * 1465.0)
    assert row["source"] == "account_cash_history_residual"
    assert row["raw_payload_json"]["inferred"] is True
    assert row["raw_payload_json"]["inference_reason"] == "account_cash_history_residual"
