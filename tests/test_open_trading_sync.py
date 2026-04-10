from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.config import Settings
from arena.open_trading.sync import AccountSyncService, BrokerCashSyncService, BrokerTradeSyncService, MarketDataSyncService


class FakeRepo:
    def __init__(self):
        self.rows = []
        self.snapshot = None
        self._latest_dates = {}
        self._spans = {}
        self.latest_instrument_map_calls = []
        self.latest_market_features_calls = []

    def insert_market_features(self, rows):
        self.rows.extend(rows)

    def write_account_snapshot(self, snapshot):
        self.snapshot = snapshot

    def latest_feature_dates(self, tickers, source):
        _ = (tickers, source)
        return dict(self._latest_dates)

    def feature_date_spans(self, tickers, source):
        _ = (tickers, source)
        return dict(self._spans)

    def latest_instrument_map(self, tickers):
        self.latest_instrument_map_calls.append(list(tickers))
        return {}

    def latest_market_features(self, *, tickers, limit, sources=None):
        self.latest_market_features_calls.append(
            {
                "tickers": list(tickers),
                "limit": limit,
                "sources": list(sources or []),
            }
        )
        rows = []
        for ticker in tickers:
            token = str(ticker).strip().upper()
            if not token:
                continue
            is_kospi = token.isdigit() and len(token) == 6
            rows.append(
                {
                    "ticker": token,
                    "exchange_code": "KRX" if is_kospi else "NASD",
                    "instrument_id": f"{'KRX' if is_kospi else 'NASD'}:{token}",
                    "ret_5d": 0.01,
                    "ret_20d": 0.04,
                    "volatility_20d": 0.12,
                    "sentiment_score": 0.1,
                }
            )
        return rows[:limit]

    def latest_missing_daily_feature_tickers(self, *, sources=None, limit=1000):
        _ = (sources, limit)
        return []


class FakeClient:
    def __init__(self):
        self.domestic_daily_requests = []
        self.overseas_daily_requests = []

    def get_usd_krw_daily_chart(self, *, symbol, start_date="", end_date="", market_div_code="X", period="D", max_pages=8):
        _ = (symbol, start_date, end_date, market_div_code, period, max_pages)
        return [
            {"stck_bsop_date": "20260101", "ovrs_nmix_prpr": "1295"},
            {"stck_bsop_date": "20260102", "ovrs_nmix_prpr": "1297"},
            {"stck_bsop_date": "20260103", "ovrs_nmix_prpr": "1300"},
            {"stck_bsop_date": "20260104", "ovrs_nmix_prpr": "1302"},
            {"stck_bsop_date": "20260105", "ovrs_nmix_prpr": "1304"},
            {"stck_bsop_date": "20260106", "ovrs_nmix_prpr": "1306"},
        ]

    def get_overseas_price(self, ticker, excd):
        return {"last": "100", "rate": "1.2"}

    def get_overseas_price_detail(self, ticker, excd):
        _ = (ticker, excd)
        return {"curr": "USD", "t_rate": "1311"}

    def get_overseas_daily_price(self, ticker, excd, bymd, gubn, modp):
        self.overseas_daily_requests.append(
            {
                "ticker": ticker,
                "excd": excd,
                "bymd": bymd,
                "gubn": gubn,
                "modp": modp,
            }
        )
        return [
            {"xymd": "20260101", "clos": "90"},
            {"xymd": "20260102", "clos": "95"},
            {"xymd": "20260103", "clos": "100"},
            {"xymd": "20260104", "clos": "105"},
            {"xymd": "20260105", "clos": "110"},
            {"xymd": "20260106", "clos": "120"},
        ]

    def search_overseas_stocks(self, *, excd="NAS", max_pages=1, **kwargs):
        # Return minimal discovery rows so _discover_us_symbols() works in tests.
        if excd == "NAS":
            return [{"symb": "AAPL", "valx": "3000000"}]
        return []

    def get_domestic_price(self, ticker, market_div_code):
        return {"stck_prpr": "70000", "prdy_ctrt": "0.8"}

    def get_domestic_daily_price(self, ticker, start_date, end_date, market_div_code, period_div_code, org_adj_prc):
        self.domestic_daily_requests.append(
            {
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "market_div_code": market_div_code,
                "period_div_code": period_div_code,
                "org_adj_prc": org_adj_prc,
            }
        )
        return [
            {"stck_bsop_date": "20260101", "stck_clpr": "66000"},
            {"stck_bsop_date": "20260102", "stck_clpr": "67000"},
            {"stck_bsop_date": "20260103", "stck_clpr": "68000"},
            {"stck_bsop_date": "20260104", "stck_clpr": "69000"},
            {"stck_bsop_date": "20260105", "stck_clpr": "70000"},
            {"stck_bsop_date": "20260106", "stck_clpr": "71000"},
        ]

    def get_domestic_market_cap_ranking(self, *, market_scope="0001", div_cls_code="0"):
        _ = (market_scope, div_cls_code)
        return [
            {"mksc_shrn_iscd": "005930"},
            {"mksc_shrn_iscd": "000660"},
        ]

    def get_domestic_top_interest_stock(self, *, market_scope="0001"):
        _ = (market_scope,)
        return [
            {"mksc_shrn_iscd": "005930"},
            {"mksc_shrn_iscd": "373220"},
        ]

    def get_domestic_volume_rank(self, *, market_scope="0001"):
        _ = (market_scope,)
        return [
            {"mksc_shrn_iscd": "003280"},
        ]

    def get_overseas_present_balance(self, *, tr_mket_cd=None, max_pages=8):
        _ = (tr_mket_cd, max_pages)
        return (
            [
                {
                    "pdno": "AAPL",
                    "cblc_qty13": "2",
                    "avg_unpr3": "100",
                    "ovrs_now_pric1": "120",
                    "bass_exrt": "1300",
                }
            ],
            [],
            [{"tot_dncl_amt": "1000000", "tot_asst_amt": "1312000"}],
        )

    def get_domestic_balance(self, inqr_dvsn):
        return (
            [
                {
                    "pdno": "005930",
                    "hldg_qty": "3",
                    "pchs_avg_pric": "65000",
                    "prpr": "70000",
                }
            ],
            [{"dnca_tot_amt": "500000", "tot_evlu_amt": "710000"}],
        )

    def get_domestic_orderable_cash(self):
        return 500000.0


class FakeBrokerTradeRepo:
    def __init__(self):
        self.appended_trade_rows = []
        self.existing_ids = set()

    def existing_event_ids(self, table_name, event_ids, tenant_id=None):
        _ = tenant_id
        assert table_name == "broker_trade_events"
        return {token for token in event_ids if token in self.existing_ids}

    def append_broker_trade_events(self, rows, tenant_id=None):
        _ = tenant_id
        self.appended_trade_rows.extend(rows)
        self.existing_ids.update(str(row.get("event_id") or "") for row in rows if str(row.get("event_id") or ""))


class FakeBrokerCashRepo:
    def __init__(self):
        self.appended_cash_rows = []
        self.existing_ids = set()
        self.cash_history_rows = []
        self.existing_cash_rows = []

    def existing_event_ids(self, table_name, event_ids, tenant_id=None):
        _ = tenant_id
        assert table_name == "broker_cash_events"
        return {token for token in event_ids if token in self.existing_ids}

    def append_broker_cash_events(self, rows, tenant_id=None):
        _ = tenant_id
        self.appended_cash_rows.extend(rows)
        self.existing_ids.update(str(row.get("event_id") or "") for row in rows if str(row.get("event_id") or ""))

    def account_cash_history(self, *, start_at, end_at=None, tenant_id=None):
        _ = (start_at, end_at, tenant_id)
        return list(self.cash_history_rows)

    def broker_cash_events_since(self, *, since, tenant_id=None):
        _ = (since, tenant_id)
        return list(self.existing_cash_rows)


class FakeBrokerTradeClient:
    def __init__(self, *, overseas=None, domestic=None, failed_exchanges=None, overseas_period_trans=None, domestic_period_profit=None):
        self.overseas = overseas or {}
        self.domestic = domestic or []
        self.failed_exchanges = set(failed_exchanges or [])
        self.overseas_period_trans = overseas_period_trans or {}
        self.domestic_period_profit = domestic_period_profit or []
        self.overseas_calls = []
        self.domestic_calls = []
        self.overseas_period_trans_calls = []
        self.domestic_period_profit_calls = []

    def inquire_overseas_ccnl(self, *, days=7, pdno="", exchange_code=None, sort_sqn="DS", max_pages=8):
        _ = (days, pdno, sort_sqn, max_pages)
        exchange = str(exchange_code or "").upper()
        self.overseas_calls.append(exchange)
        if exchange in self.failed_exchanges:
            raise RuntimeError(f"boom:{exchange}")
        return list(self.overseas.get(exchange, []))

    def inquire_domestic_daily_ccld(self, *, start_date, end_date, pdno="", odno="", max_pages=8):
        _ = (pdno, odno, max_pages)
        self.domestic_calls.append((start_date, end_date))
        return list(self.domestic)

    def inquire_overseas_period_trans(self, *, start_date, end_date, exchange_code=None, pdno="", sll_buy_dvsn_cd="00", loan_dvsn_cd="", max_pages=8):
        _ = (pdno, sll_buy_dvsn_cd, loan_dvsn_cd, max_pages)
        exchange = str(exchange_code or "").upper()
        self.overseas_period_trans_calls.append((exchange, start_date, end_date))
        return list(self.overseas_period_trans.get(exchange, [])), []

    def inquire_domestic_period_profit(self, *, start_date, end_date, sort_dvsn="00", inqr_dvsn="00", cblc_dvsn="00", pdno="", max_pages=8):
        _ = (sort_dvsn, inqr_dvsn, cblc_dvsn, pdno, max_pages)
        self.domestic_period_profit_calls.append((start_date, end_date))
        return list(self.domestic_period_profit), []


def _settings(target_market: str, universe: list[str]) -> Settings:
    return Settings(
        google_cloud_project="p",
        bq_dataset="d",
        bq_location="loc",
        agent_ids=["gpt"],
        agent_mode="adk",
        base_currency="KRW",
        sleeve_capital_krw=2_000_000,
        log_level="INFO",
        log_format="rich",
        trading_mode="paper",
        kis_order_endpoint="",
        kis_api_key="k",
        kis_api_secret="s",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no="1234567801",
        kis_account_product_code="01",
        kis_account_key_suffix="",
        kis_env="real",
        kis_target_market=target_market,
        kis_overseas_quote_excd="NAS",
        kis_overseas_order_excd="NASD",
        kis_us_natn_cd="840",
        kis_us_tr_mket_cd="01",
        kis_secret_name="",
        kis_secret_version="latest",
        kis_http_timeout_seconds=20,
        kis_http_max_retries=0,
        kis_http_backoff_base_seconds=0.1,
        kis_http_backoff_max_seconds=0.1,
        kis_confirm_fills=False,
        kis_confirm_timeout_seconds=25,
        kis_confirm_poll_seconds=2.0,
        usd_krw_rate=1300.0,
        market_sync_history_days=60,
        max_order_krw=350000,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=0.35,
        min_cash_buffer_ratio=0.1,
        ticker_cooldown_seconds=120,
        max_daily_orders=0,
        estimated_fee_bps=10.0,
        context_max_board_posts=24,
        context_max_memory_events=32,
        context_max_market_rows=64,
        openai_api_key="",
        openai_model="gpt-5.2",
        gemini_api_key="",
        gemini_model="gemini-3-pro-preview",
        research_gemini_model="gemini-2.5-flash",
        llm_timeout_seconds=25,
        default_universe=universe,
        allow_live_trading=False,
    )


def test_market_sync_nasdaq_builds_rows() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())
    result = service.sync_market_features()

    # Discovery returns AAPL from NAS + benchmarks SPY/QQQ/DIA = 4 tickers
    assert result.attempted_tickers == 4
    assert result.inserted_rows == 24
    assert len(repo.rows) == 24
    assert {"AAPL", "SPY", "QQQ", "DIA"} <= {r["ticker"] for r in repo.rows}
    assert repo.rows[-1]["close_price_krw"] > 0
    aapl_rows = [r for r in repo.rows if r["ticker"] == "AAPL"]
    assert aapl_rows[-1]["close_price_native"] == pytest.approx(120.0)
    assert aapl_rows[-1]["quote_currency"] == "USD"
    assert aapl_rows[-1]["fx_rate_used"] == pytest.approx(1306.0)
    assert aapl_rows[-1]["close_price_krw"] == pytest.approx(120.0 * 1306.0)
    assert repo.latest_instrument_map_calls == [["AAPL", "SPY", "QQQ", "DIA"]]


def test_market_sync_nasdaq_fails_without_live_fx() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())

    result = service.sync_market_features()

    assert result.inserted_rows == 0
    assert "AAPL" in result.failed_tickers
    assert not repo.rows


def test_market_sync_kospi_builds_rows() -> None:
    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)
    result = service.sync_market_features()

    assert result.attempted_tickers == 5
    assert result.inserted_rows == 30
    assert {r["ticker"] for r in repo.rows} == {"005930", "000660", "003280", "069500", "373220"}
    assert repo.rows[-1]["source"] == "open_trading_kospi"
    assert repo.rows[-1]["quote_currency"] == "KRW"
    assert repo.rows[-1]["fx_rate_used"] == pytest.approx(1.0)
    assert client.domestic_daily_requests


def test_discover_kospi_symbols_backfills_name_for_already_seen_ticker() -> None:
    class NamedClient(FakeClient):
        def get_domestic_market_cap_ranking(self, *, market_scope="0001", div_cls_code="0"):
            _ = (market_scope, div_cls_code)
            return [
                {"mksc_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자"},
                {"mksc_shrn_iscd": "000660", "hts_kor_isnm": "SK하이닉스"},
            ]

    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    service = MarketDataSyncService(settings=settings, repo=repo, client=NamedClient())

    symbols = service._discover_kospi_symbols()

    assert {"ticker": "005930", "quote_excd": "KRX"} in symbols
    assert service._kospi_ticker_names["005930"] == "삼성전자"


def test_market_sync_kospi_requests_long_history_for_forecast_bootstrap() -> None:
    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    service.sync_market_features()

    assert client.domestic_daily_requests
    req = client.domestic_daily_requests[0]
    start = datetime.strptime(req["start_date"], "%Y%m%d")
    end = datetime.strptime(req["end_date"], "%Y%m%d")
    assert (end - start).days >= 360


def test_market_sync_kospi_forces_backfill_when_existing_history_is_too_shallow() -> None:
    repo = FakeRepo()
    repo._latest_dates = {"005930": datetime.strptime("20260306", "%Y%m%d").date()}
    repo._spans = {
        "005930": {
            "min_d": datetime.strptime("20251002", "%Y%m%d").date(),
            "max_d": datetime.strptime("20260306", "%Y%m%d").date(),
            "row_count": 100,
        }
    }
    settings = _settings("kospi", ["005930"])
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_features()

    assert result.inserted_rows == 30
    assert client.domestic_daily_requests
    assert len([r for r in repo.rows if r["ticker"] == "005930"]) == 6


def test_market_sync_us_forces_backfill_when_existing_history_is_too_shallow() -> None:
    repo = FakeRepo()
    repo._latest_dates = {"AAPL": datetime.strptime("20260106", "%Y%m%d").date()}
    repo._spans = {
        "AAPL": {
            "min_d": datetime.strptime("20260106", "%Y%m%d").date(),
            "max_d": datetime.strptime("20260106", "%Y%m%d").date(),
            "row_count": 1,
        }
    }
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_features()

    assert result.inserted_rows == 24
    assert client.overseas_daily_requests
    assert len([r for r in repo.rows if r["ticker"] == "AAPL"]) == 6


def test_market_sync_us_includes_existing_tickers_missing_daily_features() -> None:
    class RepoWithMissingFeatureTicker(FakeRepo):
        def latest_missing_daily_feature_tickers(self, *, sources=None, limit=1000):
            _ = (sources, limit)
            return [
                {
                    "ticker": "MISS",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:MISS",
                    "source": "open_trading_us_quote",
                }
            ]

    class LongHistoryClient(FakeClient):
        def get_overseas_daily_price(self, ticker, excd, bymd, gubn, modp):
            self.overseas_daily_requests.append(
                {
                    "ticker": ticker,
                    "excd": excd,
                    "bymd": bymd,
                    "gubn": gubn,
                    "modp": modp,
                }
            )
            return [
                {"xymd": f"202601{idx:02d}", "clos": str(90 + idx)}
                for idx in range(1, 26)
            ]

    repo = RepoWithMissingFeatureTicker()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    client = LongHistoryClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_features()

    assert result.attempted_tickers == 5
    assert result.inserted_rows == 125
    miss_rows = [row for row in repo.rows if row["ticker"] == "MISS"]
    assert miss_rows
    assert miss_rows[-1]["ret_5d"] is not None
    assert miss_rows[-1]["ret_20d"] is not None
    assert miss_rows[-1]["volatility_20d"] is not None


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


def test_quote_sync_us_rows_include_native_price_and_fx() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())

    result = service.sync_market_quotes()

    assert result.inserted_rows == 4
    aapl_rows = [r for r in repo.rows if r["ticker"] == "AAPL"]
    assert aapl_rows
    assert aapl_rows[-1]["close_price_native"] == pytest.approx(100.0)
    assert aapl_rows[-1]["fx_rate_used"] == pytest.approx(1311.0)
    assert aapl_rows[-1]["close_price_krw"] == pytest.approx(131100.0)
    assert aapl_rows[-1]["ret_20d"] == pytest.approx(0.04)
    assert aapl_rows[-1]["volatility_20d"] == pytest.approx(0.12)


def test_quote_sync_us_skips_quote_rows_when_daily_features_are_missing() -> None:
    class RepoWithoutDailyBase(FakeRepo):
        def latest_market_features(self, *, tickers, limit, sources=None):
            self.latest_market_features_calls.append(
                {
                    "tickers": list(tickers),
                    "limit": limit,
                    "sources": list(sources or []),
                }
            )
            return []

    repo = RepoWithoutDailyBase()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_quotes()

    assert result.inserted_rows == 0
    assert sorted(result.failed_tickers) == ["AAPL", "DIA", "QQQ", "SPY"]
    assert client.overseas_daily_requests == []
    assert repo.rows == []


def test_quote_sync_us_held_ticker_probes_exchange() -> None:
    class RepoWithHoldings(FakeRepo):
        def get_all_held_tickers(self):
            return ["KO"]

    class ProbeClient(FakeClient):
        def search_overseas_stocks(self, *, excd="NAS", max_pages=1, **kwargs):
            _ = (excd, max_pages, kwargs)
            return []

        def get_overseas_price(self, ticker, excd):
            if ticker == "KO" and excd == "NYS":
                return {"last": "77.34", "rate": "0.2", "rsym": "DNYSKO"}
            if ticker in {"SPY", "QQQ", "DIA"} and excd == "NAS":
                return {"last": "100", "rate": "1.2", "rsym": f"DNAS{ticker}"}
            return {"last": "", "rate": "", "rsym": ""}

    repo = RepoWithHoldings()
    settings = _settings("us", ["KO"])
    settings.usd_krw_fx_symbol = "USDKRW"
    service = MarketDataSyncService(settings=settings, repo=repo, client=ProbeClient())

    result = service.sync_market_quotes()

    assert result.inserted_rows == 4
    ko_rows = [row for row in repo.rows if row["ticker"] == "KO"]
    assert ko_rows
    assert ko_rows[-1]["exchange_code"] == "NYSE"
    assert ko_rows[-1]["instrument_id"] == "NYSE:KO"


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
