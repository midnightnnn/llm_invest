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

def test_account_sync_overseas_persists_snapshot() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    snapshot = AccountSyncService(settings=settings, repo=repo, client=FakeClient()).sync_account_snapshot()

    assert snapshot.cash_krw == 1_000_000
    assert "AAPL" in snapshot.positions
    assert snapshot.usd_krw_rate == pytest.approx(1300.0)
    assert repo.snapshot is snapshot


def test_account_sync_overseas_raises_without_live_fx() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])

    class MissingFxClient(FakeClient):
        def get_overseas_present_balance(self, *, tr_mket_cd=None, max_pages=8):
            _ = (tr_mket_cd, max_pages)
            return (
                [
                    {
                        "pdno": "AAPL",
                        "cblc_qty13": "2",
                        "ccld_qty_smtl1": "2",
                        "ord_psbl_qty1": "2",
                        "avg_unpr3": "100",
                        "ovrs_now_pric1": "120",
                        "ovrs_excg_cd": "NASD",
                        "tr_crcy_cd": "USD",
                    }
                ],
                [],
                [{"tot_dncl_amt": "1000000", "frcr_use_psbl_amt": "1000", "tot_asst_amt": "1312000"}],
            )

    with pytest.raises(RuntimeError, match="USD/KRW FX symbol not configured"):
        AccountSyncService(settings=settings, repo=repo, client=MissingFxClient()).sync_account_snapshot()


def test_account_sync_domestic_uses_orderable_cash_without_summary_fallback() -> None:
    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    snapshot = AccountSyncService(settings=settings, repo=repo, client=FakeClient()).sync_account_snapshot()

    assert snapshot.cash_krw == pytest.approx(500000.0)
    assert "005930" in snapshot.positions


def test_account_sync_domestic_failure_preserves_orderable_cash_cause() -> None:
    class FailingOrderableCashClient(FakeClient):
        def get_domestic_orderable_cash(self):
            raise RuntimeError("KIS rt_cd=1 msg_cd=OPSQ0002 msg=invalid input path=/uapi/domestic-stock/v1/trading/inquire-psbl-order")

    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])

    with pytest.raises(RuntimeError, match="domestic orderable cash query failed: KIS rt_cd=1 msg_cd=OPSQ0002"):
        AccountSyncService(settings=settings, repo=repo, client=FailingOrderableCashClient()).sync_account_snapshot()


def test_account_sync_overseas_prefers_current_quantity_over_carry_quantity() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["CCEP"])
    client = FakeClient()

    def _present_balance(*, tr_mket_cd=None, max_pages=8):
        _ = (tr_mket_cd, max_pages)
        return (
            [
                {
                    "pdno": "CCEP",
                    "cblc_qty13": "6",
                    "thdt_buy_ccld_qty1": "1",
                    "thdt_sll_ccld_qty1": "0",
                    "ccld_qty_smtl1": "7",
                    "ord_psbl_qty1": "7",
                    "avg_unpr3": "102.9826",
                    "ovrs_now_pric1": "100.46",
                    "bass_exrt": "1479.8",
                    "ovrs_excg_cd": "NASD",
                    "buy_crcy_cd": "USD",
                }
            ],
            [],
            [{"tot_dncl_amt": "2420707", "tot_asst_amt": "12680718"}],
        )

    client.get_overseas_present_balance = _present_balance
    snapshot = AccountSyncService(settings=settings, repo=repo, client=client).sync_account_snapshot()

    assert snapshot.positions["CCEP"].quantity == pytest.approx(7.0)


def test_account_sync_overseas_reads_native_usd_cash_from_currency_rows() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    client = FakeClient()

    def _present_balance(*, tr_mket_cd=None, max_pages=8):
        _ = (tr_mket_cd, max_pages)
        return (
            [
                {
                    "pdno": "AAPL",
                    "cblc_qty13": "2",
                    "ccld_qty_smtl1": "2",
                    "ord_psbl_qty1": "2",
                    "avg_unpr3": "100",
                    "ovrs_now_pric1": "120",
                    "bass_exrt": "1476.1",
                    "ovrs_excg_cd": "NASD",
                    "tr_crcy_cd": "USD",
                }
            ],
            [
                {
                    "crcy_cd": "USD",
                    "frcr_dncl_amt_2": "179.750000",
                    "frcr_drwg_psbl_amt_1": "179.750000",
                    "frcr_evlu_amt2": "265328.000000",
                }
            ],
            [{"tot_dncl_amt": "1173554", "frcr_use_psbl_amt": "265328.00", "tot_asst_amt": "1523554"}],
        )

    client.get_overseas_present_balance = _present_balance
    snapshot = AccountSyncService(settings=settings, repo=repo, client=client).sync_account_snapshot()

    assert snapshot.cash_krw == pytest.approx(1_173_554.0)
    assert snapshot.cash_foreign == pytest.approx(179.75)
    assert snapshot.cash_foreign_currency == "USD"
    assert snapshot.usd_krw_rate == pytest.approx(1476.1)


def test_account_sync_combined_preserves_usd_fx_rate() -> None:
    repo = FakeRepo()
    settings = _settings("us,kospi", ["AAPL", "005930"])
    snapshot = AccountSyncService(settings=settings, repo=repo, client=FakeClient()).sync_account_snapshot()

    assert snapshot.usd_krw_rate == pytest.approx(1300.0)
    assert snapshot.cash_foreign_currency == "USD"


def test_account_sync_combined_does_not_double_count_shared_krw_cash() -> None:
    repo = FakeRepo()
    settings = _settings("us,kospi", ["AAPL", "005930"])
    snapshot = AccountSyncService(settings=settings, repo=repo, client=FakeClient()).sync_account_snapshot()

    position_value = sum(pos.market_value_krw() for pos in snapshot.positions.values())
    assert snapshot.cash_krw == pytest.approx(1_000_000.0)
    assert snapshot.total_equity_krw == pytest.approx(1_000_000.0 + position_value)


def test_account_sync_overseas_merges_multi_exchange_balances() -> None:
    repo = FakeRepo()
    settings = _settings("us", ["AAPL", "VZ"])
    client = FakeClient()

    def _present_balance(*, tr_mket_cd=None, max_pages=8):
        _ = max_pages
        if tr_mket_cd == "02":
            return (
                [
                    {
                        "pdno": "VZ",
                        "cblc_qty13": "3",
                        "ccld_qty_smtl1": "3",
                        "ord_psbl_qty1": "3",
                        "avg_unpr3": "49.54",
                        "ovrs_now_pric1": "50.27",
                        "bass_exrt": "1499.7",
                        "ovrs_excg_cd": "NYSE",
                        "tr_crcy_cd": "USD",
                    }
                ],
                [],
                [{"tot_dncl_amt": "1000000", "frcr_use_psbl_amt": "1000", "tot_asst_amt": "1200000"}],
            )
        if tr_mket_cd == "03":
            return ([], [], [{"tot_dncl_amt": "1000000", "frcr_use_psbl_amt": "1000", "tot_asst_amt": "1000000"}])
        return (
            [
                {
                    "pdno": "AAPL",
                    "cblc_qty13": "2",
                    "ccld_qty_smtl1": "2",
                    "ord_psbl_qty1": "2",
                    "avg_unpr3": "100",
                    "ovrs_now_pric1": "120",
                    "bass_exrt": "1300",
                    "ovrs_excg_cd": "NASD",
                    "tr_crcy_cd": "USD",
                }
            ],
            [],
            [{"tot_dncl_amt": "1000000", "frcr_use_psbl_amt": "1000", "tot_asst_amt": "1312000"}],
        )

    client.get_overseas_present_balance = _present_balance
    snapshot = AccountSyncService(settings=settings, repo=repo, client=client).sync_account_snapshot()

    assert "AAPL" in snapshot.positions
    assert "VZ" in snapshot.positions
    assert snapshot.positions["VZ"].exchange_code == "NYSE"
    assert snapshot.positions["VZ"].quantity == pytest.approx(3.0)
    assert snapshot.total_equity_krw == pytest.approx(1_312_000 + snapshot.positions["VZ"].market_value_krw())


def test_account_sync_overseas_probes_missing_exchange_code() -> None:
    repo = FakeRepo()
    settings = _settings("us", ["KO"])

    class ProbeClient(FakeClient):
        def get_overseas_present_balance(self, *, tr_mket_cd=None, max_pages=8):
            _ = (tr_mket_cd, max_pages)
            return (
                [
                    {
                        "pdno": "KO",
                        "cblc_qty13": "3",
                        "avg_unpr3": "75.0",
                        "ovrs_now_pric1": "77.34",
                        "bass_exrt": "1310",
                    }
                ],
                [],
                [{"tot_dncl_amt": "1000000", "tot_asst_amt": "1303762"}],
            )

        def get_overseas_price(self, ticker, excd):
            if ticker == "KO" and excd == "NYS":
                return {"last": "77.34", "rate": "0.2", "rsym": "DNYSKO"}
            return {"last": "", "rate": "", "rsym": ""}

    snapshot = AccountSyncService(settings=settings, repo=repo, client=ProbeClient()).sync_account_snapshot()

    assert snapshot.positions["KO"].exchange_code == "NYSE"
    assert snapshot.positions["KO"].instrument_id == "NYSE:KO"


def test_account_sync_overseas_logs_instrument_map_failure(caplog) -> None:
    class RepoWithBrokenInstrumentMap(FakeRepo):
        def latest_instrument_map(self, tickers):
            self.latest_instrument_map_calls.append(list(tickers))
            raise RuntimeError("boom")

    repo = RepoWithBrokenInstrumentMap()
    settings = _settings("nasdaq", ["AAPL"])

    with caplog.at_level("WARNING"):
        snapshot = AccountSyncService(settings=settings, repo=repo, client=FakeClient()).sync_account_snapshot()

    assert snapshot.positions["AAPL"].exchange_code == "NASD"
    assert "instrument_map load failed" in caplog.text
    record = next(item for item in caplog.records if getattr(item, "event", "") == "instrument_map_load_failed")
    assert record.market == "nasdaq"
    assert record.stage == "load_instrument_map"
    assert record.ticker_count == 1
    assert record.err_type == "RuntimeError"
    assert record.err == "boom"
