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
