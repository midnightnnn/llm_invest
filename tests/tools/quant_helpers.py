from __future__ import annotations

from datetime import datetime, timezone
import math
from types import SimpleNamespace
from typing import Literal, get_args, get_origin, get_type_hints

import pytest

from arena.config import Settings
from arena.tools.quant_tools import QuantTools

def _literal_args(annotation) -> set[object]:
    if get_origin(annotation) is Literal:
        return set(get_args(annotation))
    values: set[object] = set()
    for arg in get_args(annotation):
        values.update(_literal_args(arg))
    return values


@pytest.fixture(autouse=True)
def _stable_quant_tool_now(monkeypatch) -> None:
    import arena.tools.quant_tools as qt_module

    monkeypatch.setattr(
        qt_module,
        "_utc_now",
        lambda: datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
    )


class FakeRepo:
    def __init__(self):
        self.last_screen_kwargs = None
        self.last_market_kwargs = None
        self.last_fundamentals_kwargs = None
        self.last_close_kwargs = None
        self.last_universe_limit = None
        self.universe_rows = ["AAPL", "MSFT", "TSLA", "PLTD"]
        self._features = [
            {
                "as_of_ts": "2026-01-01T00:00:00+00:00",
                "ticker": "AAPL",
                "ret_20d": 0.20,
                "ret_5d": 0.05,
                "volatility_20d": 0.10,
                "sentiment_score": 0.2,
                "close_price_krw": 1000.0,
                "source": "seed_demo",
            },
            {
                "as_of_ts": "2026-01-01T00:00:00+00:00",
                "ticker": "MSFT",
                "ret_20d": 0.10,
                "ret_5d": 0.02,
                "volatility_20d": 0.08,
                "sentiment_score": 0.1,
                "close_price_krw": 900.0,
                "source": "seed_demo",
            },
            {
                "as_of_ts": "2026-01-01T00:00:00+00:00",
                "ticker": "TSLA",
                "ret_20d": -0.05,
                "ret_5d": -0.01,
                "volatility_20d": 0.25,
                "sentiment_score": -0.1,
                "close_price_krw": 500.0,
                "source": "seed_demo",
            },
            {
                "as_of_ts": "2026-01-01T00:00:00+00:00",
                "ticker": "PLTD",
                "ret_20d": 0.95,
                "ret_5d": 0.30,
                "volatility_20d": 0.60,
                "sentiment_score": 0.8,
                "close_price_krw": 1200.0,
                "source": "seed_demo",
            },
        ]
        self._preds = [
            {"run_date": "2026-01-02", "ticker": "AAPL", "exp_return_period": 0.02, "forecast_horizon": 20},
            {"run_date": "2026-01-02", "ticker": "MSFT", "exp_return_period": 0.015, "forecast_horizon": 20},
            {"run_date": "2026-01-02", "ticker": "TSLA", "exp_return_period": 0.03, "forecast_horizon": 20},
            {"run_date": "2026-01-02", "ticker": "PLTD", "exp_return_period": 0.08, "forecast_horizon": 20},
        ]
        self._fundamentals = [
            {"ticker": "AAPL", "market": "us", "per": 28.0, "pbr": 9.0, "eps": 6.0, "bps": 20.0, "roe": 18.0, "debt_ratio": 120.0},
            {"ticker": "MSFT", "market": "us", "per": 14.0, "pbr": 2.1, "eps": 12.0, "bps": 40.0, "roe": 21.0, "debt_ratio": 60.0},
            {"ticker": "TSLA", "market": "us", "per": 90.0, "pbr": 14.0, "eps": 2.0, "bps": 14.0, "roe": 6.0, "debt_ratio": 180.0},
            {"ticker": "PLTD", "market": "us", "per": 120.0, "pbr": 18.0, "eps": 1.0, "bps": 8.0, "roe": 4.0, "debt_ratio": 220.0},
        ]
        self.last_forecast_mode = None
        self.last_forecast_table = None

    def screen_latest_features(self, **kwargs):
        self.last_screen_kwargs = dict(kwargs)
        sort_by = kwargs.get("sort_by", "ret_20d")
        order = kwargs.get("order", "desc")
        top_n = int(kwargs.get("top_n", 10))
        rows = list(self._features)
        allowed = kwargs.get("tickers")
        if allowed is not None:
            allow = {str(t).strip().upper() for t in allowed if str(t).strip()}
            rows = [r for r in rows if str(r.get("ticker", "")).upper() in allow]
        reverse = str(order).lower() != "asc"
        if sort_by == "as_of_ts":
            rows.sort(key=lambda r: str(r.get(sort_by) or ""), reverse=reverse)
        else:
            rows.sort(key=lambda r: float(r.get(sort_by) or 0.0), reverse=reverse)
        return rows[:top_n]

    def get_daily_closes(self, *, tickers, lookback_days, sources=None):
        self.last_close_kwargs = {
            "tickers": list(tickers),
            "lookback_days": lookback_days,
            "sources": list(sources) if sources is not None else None,
        }
        _ = sources
        n = int(lookback_days)
        out = {}
        for t in tickers:
            base = 100.0
            if t == "AAPL":
                base = 100.0
            if t == "MSFT":
                base = 80.0
            if t == "TSLA":
                base = 60.0
            out[t] = [base + i for i in range(max(n, 12))]
        return out

    def latest_market_features(self, tickers, limit, sources=None):
        self.last_market_kwargs = {
            "tickers": list(tickers),
            "limit": limit,
            "sources": list(sources) if sources is not None else None,
        }
        rows = list(self._features)
        allow = {str(t).strip().upper() for t in tickers if str(t).strip()}
        if allow:
            rows = [r for r in rows if str(r.get("ticker", "")).upper() in allow]
        return rows[:limit]

    def latest_fundamentals_snapshot(self, *, tickers=None, limit=500):
        self.last_fundamentals_kwargs = {
            "tickers": list(tickers) if tickers is not None else None,
            "limit": limit,
        }
        rows = list(self._fundamentals)
        if tickers:
            allow = {str(t).strip().upper() for t in tickers if str(t).strip()}
            rows = [r for r in rows if str(r.get("ticker", "")).upper() in allow]
        return rows[:limit]

    def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
        _ = limit
        self.last_forecast_mode = mode
        self.last_forecast_table = table_id
        self.last_staleness_days = staleness_days
        rows = list(self._preds)
        if tickers:
            want = {str(t).strip().upper() for t in tickers}
            rows = [r for r in rows if str(r.get("ticker", "")).upper() in want]
        return rows

    def latest_universe_candidate_tickers(self, *, limit=200):
        self.last_universe_limit = limit
        return list(self.universe_rows[:limit])


class FakeOpenTradingClient:
    def __init__(self) -> None:
        self.overseas_price_detail_calls: list[tuple[str, str | None]] = []

    def get_overseas_price_detail(self, ticker: str, excd: str | None = None):
        self.overseas_price_detail_calls.append((ticker, excd))
        exchange = str(excd or "").strip().upper()
        data = {
            ("AAPL", "NAS"): {"curr": "USD", "last": "201.12", "tomv": "3000000", "perx": "31.5", "pbrx": "45.2", "epsx": "6.38", "bpsx": "4.45", "e_ordyn": "Y"},
            ("MSFT", "NAS"): {"curr": "USD", "last": "425.50", "tomv": "3200000", "perx": "34.0", "pbrx": "12.1", "epsx": "12.50", "bpsx": "35.12", "e_ordyn": "Y"},
            ("AAPL", "NYS"): {"curr": "USD", "last": "", "tomv": "", "perx": "", "pbrx": "", "epsx": "", "bpsx": "", "e_ordyn": ""},
            ("MSFT", "NYS"): {"curr": "USD", "last": "", "tomv": "", "perx": "", "pbrx": "", "epsx": "", "bpsx": "", "e_ordyn": ""},
            ("AAPL", "AMS"): {"curr": "USD", "last": "", "tomv": "", "perx": "", "pbrx": "", "epsx": "", "bpsx": "", "e_ordyn": ""},
            ("MSFT", "AMS"): {"curr": "USD", "last": "", "tomv": "", "perx": "", "pbrx": "", "epsx": "", "bpsx": "", "e_ordyn": ""},
        }
        key = (ticker, exchange or "NAS")
        if key not in data:
            raise RuntimeError("ticker not found")
        return data[key]

    def search_overseas_stocks(
        self,
        *,
        excd: str | None = None,
        price_min: float | None = None,
        price_max: float | None = None,
        per_min: float | None = None,
        per_max: float | None = None,
        eps_min: float | None = None,
        eps_max: float | None = None,
        max_pages: int = 4,
    ):
        _ = (excd, price_min, price_max, per_min, per_max, eps_min, eps_max, max_pages)
        return [
            {"symb": "AAPL", "excd": "NAS", "last": "201.12", "per": "31.5", "eps": "6.38", "valx": "3000000", "e_ordyn": "Y"},
            {"symb": "MSFT", "excd": "NAS", "last": "425.50", "per": "34.0", "eps": "12.50", "valx": "3200000", "e_ordyn": "Y"},
            {"symb": "XYZ", "excd": "NAS", "last": "12.11", "per": "9.9", "eps": "1.2", "valx": "1000", "e_ordyn": "N"},
        ]


def _settings() -> Settings:
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
        kis_api_key="",
        kis_api_secret="",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no="",
        kis_account_product_code="01",
        kis_account_key_suffix="",
        kis_env="demo",
        kis_target_market="nasdaq",
        kis_overseas_quote_excd="NAS",
        kis_overseas_order_excd="NASD",
        kis_us_natn_cd="840",
        kis_us_tr_mket_cd="01",
        kis_secret_name="KISAPI",
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
        max_order_krw=350_000,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=0.35,
        min_cash_buffer_ratio=0.10,
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
        default_universe=["AAPL", "MSFT", "TSLA"],
        allow_live_trading=False,
        autonomy_working_set_enabled=True,
        autonomy_tool_default_candidates_enabled=True,
        autonomy_opportunity_context_enabled=True,
    )
