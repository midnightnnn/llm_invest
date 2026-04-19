from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from arena.config import Settings
from arena.tools.quant_tools import QuantTools


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


def test_screen_market_returns_rows() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    rows = qt.screen_market(top_n=2)
    assert len(rows) == 2
    assert rows[0]["ticker"]
    assert rows[0]["bucket"] in {"momentum", "pullback", "recovery", "defensive", "value"}
    assert "score" in rows[0]
    assert rows[0]["reason_for"]
    assert rows[0]["reason_risk"]
    assert rows[0]["evidence_level"] == "screened_only"


def test_screen_market_excludes_quote_only_rows_without_history_features() -> None:
    class _SparseRepo(FakeRepo):
        def __init__(self):
            super().__init__()
            self._features = [
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "MISSING",
                    "ret_20d": None,
                    "ret_5d": None,
                    "volatility_20d": None,
                    "sentiment_score": 1.0,
                    "close_price_krw": 1000.0,
                    "source": "open_trading_us_quote",
                },
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "ZERO",
                    "ret_20d": 0.0,
                    "ret_5d": 0.0,
                    "volatility_20d": 0.0,
                    "sentiment_score": 0.0,
                    "close_price_krw": 1000.0,
                    "source": "open_trading_us",
                },
            ]
            self.universe_rows = ["MISSING", "ZERO"]

        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            out = super().get_daily_closes(tickers=tickers, lookback_days=lookback_days, sources=sources)
            out.pop("MISSING", None)
            return out

    settings = _settings()
    settings.default_universe = ["MISSING", "ZERO"]
    qt = QuantTools(repo=_SparseRepo(), settings=settings)

    rows = qt.screen_market(bucket="defensive", top_n=5)

    assert {row["ticker"] for row in rows} == {"ZERO"}


def test_screen_market_overlays_stored_returns_with_raw_close_features() -> None:
    class _ZeroFeatureRepo(FakeRepo):
        def __init__(self):
            super().__init__()
            for row in self._features:
                row["ret_5d"] = 0.0
                row["ret_20d"] = 0.0
                row["volatility_20d"] = 0.0

    qt = QuantTools(repo=_ZeroFeatureRepo(), settings=_settings())

    rows = qt.screen_market(bucket="defensive", top_n=10)

    aapl = next(row for row in rows if row["ticker"] == "AAPL")
    closes = [100.0 + i for i in range(128)]
    assert math.isclose(aapl["ret_5d"], (closes[-1] / closes[-6]) - 1.0)
    assert math.isclose(aapl["ret_20d"], (closes[-1] / closes[-21]) - 1.0)
    assert aapl["volatility_20d"] > 0.0


def test_target_universe_filters_us_markets_to_alpha_tickers() -> None:
    settings = _settings()
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL", "EXC", "005930", "123456"]
    qt = QuantTools(repo=FakeRepo(), settings=settings)
    assert qt._target_universe() == ["AAPL", "EXC"]


def test_target_universe_loads_latest_universe_candidates_when_default_empty() -> None:
    settings = _settings()
    settings.kis_target_market = "us"
    settings.default_universe = []
    settings.universe_run_top_n = 3
    repo = FakeRepo()
    repo.universe_rows = ["AAPL", "005930", "MSFT", "123456"]
    qt = QuantTools(repo=repo, settings=settings)

    assert qt._target_universe() == ["AAPL", "MSFT"]
    assert repo.last_universe_limit == 3


def test_sources_for_us_include_quote_and_legacy_daily_sources() -> None:
    settings = _settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "us"
    qt = QuantTools(repo=FakeRepo(), settings=settings)
    assert qt._sources() == [
        "open_trading_us_quote",
        "open_trading_us",
        "open_trading_nasdaq_quote",
        "open_trading_nasdaq",
        "open_trading_nyse_quote",
        "open_trading_nyse",
        "open_trading_amex_quote",
        "open_trading_amex",
    ]


def test_screen_market_live_us_passes_quote_sources() -> None:
    settings = _settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "us"
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=settings)

    qt.screen_market(top_n=2)

    assert repo.last_market_kwargs is not None
    assert repo.last_market_kwargs["sources"] == [
        "open_trading_us_quote",
        "open_trading_us",
        "open_trading_nasdaq_quote",
        "open_trading_nasdaq",
        "open_trading_nyse_quote",
        "open_trading_nyse",
        "open_trading_amex_quote",
        "open_trading_amex",
    ]
    assert repo.last_close_kwargs is not None
    assert repo.last_close_kwargs["sources"] == [
        "open_trading_us",
        "open_trading_nasdaq",
        "open_trading_nyse",
        "open_trading_amex",
    ]


def test_screen_market_legacy_sort_mode_still_uses_bq_screen() -> None:
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=_settings())

    rows = qt.screen_market(sort_by="ret_20d", top_n=2)

    assert repo.last_screen_kwargs is not None
    assert len(rows) == 2
    assert rows[0]["ticker"] == "AAPL"


def test_screen_market_explicit_bucket_ignores_legacy_sort_mode() -> None:
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=_settings())

    rows = qt.screen_market(bucket="defensive", sort_by="ret_20d", top_n=2)

    assert rows
    assert all(row["bucket"] == "defensive" for row in rows)
    assert repo.last_screen_kwargs is None


def test_screen_market_momentum_bucket_outputs_scores() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    rows = qt.screen_market(bucket="momentum", top_n=3)
    assert rows
    assert "ticker" in rows[0]
    assert "score" in rows[0]
    assert rows[0]["bucket"] == "momentum"
    assert rows[0]["reason_for"].startswith("Multi-window momentum")
    assert "Screen-only evidence" in rows[0]["reason_risk"] or "volatility" in rows[0]["reason_risk"]


def test_screen_market_momentum_bucket_live_us_passes_quote_sources() -> None:
    settings = _settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "us"
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=settings)

    qt.screen_market(bucket="momentum", top_n=3)

    assert repo.last_close_kwargs is not None
    assert repo.last_close_kwargs["sources"] == [
        "open_trading_us",
        "open_trading_nasdaq",
        "open_trading_nyse",
        "open_trading_amex",
    ]


def test_screen_market_momentum_bucket_scans_target_universe_without_prescreen_cut() -> None:
    class _Repo(FakeRepo):
        def screen_latest_features(self, **kwargs):
            raise AssertionError("screen_market momentum bucket should not use legacy prescreen path")
            return super().screen_latest_features(**kwargs)

    qt = QuantTools(repo=_Repo(), settings=_settings())
    rows = qt.screen_market(bucket="momentum", top_n=3)
    assert rows
    assert len(rows) <= 3


def test_screen_market_value_bucket_prefers_snapshot_valuation() -> None:
    class _ValueRepo(FakeRepo):
        def __init__(self):
            super().__init__()
            self._features = [
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "CHEAP",
                    "ret_20d": -0.04,
                    "ret_5d": -0.01,
                    "volatility_20d": 0.12,
                    "sentiment_score": 0.0,
                    "close_price_krw": 1000.0,
                    "source": "seed_demo",
                },
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "EXPNSV",
                    "ret_20d": 0.08,
                    "ret_5d": 0.02,
                    "volatility_20d": 0.15,
                    "sentiment_score": 0.1,
                    "close_price_krw": 1100.0,
                    "source": "seed_demo",
                },
            ]
            self.universe_rows = ["CHEAP", "EXPNSV"]
            self._fundamentals = [
                {"ticker": "CHEAP", "market": "us", "per": 7.0, "pbr": 0.9, "eps": 4.0, "bps": 15.0, "roe": 16.0, "debt_ratio": 55.0},
                {"ticker": "EXPNSV", "market": "us", "per": 42.0, "pbr": 7.5, "eps": 2.0, "bps": 8.0, "roe": 7.0, "debt_ratio": 180.0},
            ]

        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            self.last_close_kwargs = {
                "tickers": list(tickers),
                "lookback_days": lookback_days,
                "sources": list(sources) if sources is not None else None,
            }
            _ = sources
            return {str(t).upper(): [100.0 + i for i in range(max(int(lookback_days), 12))] for t in tickers}

    settings = _settings()
    settings.default_universe = ["CHEAP", "EXPNSV"]
    repo = _ValueRepo()
    qt = QuantTools(repo=repo, settings=settings)

    rows = qt.screen_market(bucket="value", top_n=2)

    assert [row["ticker"] for row in rows] == ["CHEAP", "EXPNSV"]
    assert rows[0]["bucket"] == "value"
    assert rows[0]["reason_for"].startswith("Valuation support")
    assert rows[0]["reason_risk"]
    assert repo.last_fundamentals_kwargs is not None


def test_recommend_opportunities_uses_precomputed_learned_scores() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.last_ranker_kwargs = None

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, per_profile_limit=None):
            self.last_ranker_kwargs = {
                "limit": limit,
                "max_age_hours": max_age_hours,
                "tickers": tickers,
                "profiles": profiles,
                "buckets": buckets,
                "per_profile_limit": per_profile_limit,
            }
            return [
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "learned",
                    "ticker": "MSFT",
                    "profile": "defensive",
                    "bucket": "defensive",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.041,
                    "predicted_excess_return_20d": 0.032,
                    "prob_outperform_20d": 0.61,
                    "predicted_drawdown_20d": -0.045,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {"ret_20d": 0.08, "forecast_exp_return": 0.03},
                    "explanation_json": {"top_features": ["forecast_exp_return", "screen_score"]},
                }
            ]

    repo = _Repo()
    qt = QuantTools(repo=repo, settings=_settings())

    out = qt.recommend_opportunities(top_n=3)

    assert out["status"] == "ok"
    assert out["ranker"]["score_source"] == "learned"
    assert out["recommendations"][0]["ticker"] == "MSFT"
    assert out["recommendations"][0]["score_source"] == "learned"
    assert out["recommendations"][0]["predicted_excess_return_20d"] == 0.032
    assert repo.last_ranker_kwargs["limit"] == 3
    assert repo.last_ranker_kwargs["per_profile_limit"] == 3
    assert out["diagnostics"]["selection_scope"]["mode"] == "ranked_union"
    assert out["diagnostics"]["selection_scope"]["global_limit"] == 3
    assert out["diagnostics"]["selection_scope"]["per_profile_limit"] == 3
    assert out["diagnostics"]["selection_scope"]["loaded_rows"] == 1


def test_recommend_opportunities_uses_bucket_filter_with_profile_context() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.last_ranker_kwargs = None

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, per_profile_limit=None):
            self.last_ranker_kwargs = {
                "limit": limit,
                "max_age_hours": max_age_hours,
                "tickers": tickers,
                "profiles": profiles,
                "buckets": buckets,
                "per_profile_limit": per_profile_limit,
            }
            return [
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "learned_ic",
                    "ticker": "MSFT",
                    "profile": "defensive",
                    "bucket": "defensive",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.041,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {},
                    "explanation_json": {},
                }
            ]

    repo = _Repo()
    qt = QuantTools(repo=repo, settings=_settings())

    out = qt.recommend_opportunities(top_n=8, buckets=["defensive"])

    assert out["status"] == "ok"
    assert repo.last_ranker_kwargs["limit"] == 8
    assert repo.last_ranker_kwargs["per_profile_limit"] == 8
    assert repo.last_ranker_kwargs["buckets"] == ["defensive"]
    assert out["diagnostics"]["selection_scope"]["requested_buckets"] == ["defensive"]


def test_recommend_opportunities_learned_missing_is_not_silent_heuristic_fallback() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())

    out = qt.recommend_opportunities(top_n=3)

    assert out["status"] == "unusable"
    assert out["recommendations"] == []
    assert out["ranker"]["score_source"] == "missing"


def test_optimize_portfolio_sharpe() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="sharpe", lookback_days=20)
    assert out["strategy"] == "max_sharpe"
    w = out["weights"]
    assert set(w.keys()) == {"AAPL", "MSFT"}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert "backtest_mdd" in out
    assert out["backtest_mdd"]["value"] <= 0.0


def test_optimize_portfolio_risk_parity() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    assert out["strategy"] == "hrp"
    w = out["weights"]
    assert set(w.keys()) == {"AAPL", "MSFT"}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert "backtest_mdd" in out


def test_optimize_portfolio_forecast() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="forecast", lookback_days=20, mu_confidence=0.3)
    assert out["strategy"] == "forecast_max_sharpe"
    w = out["weights"]
    assert set(w.keys()) == {"AAPL", "MSFT"}
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert "backtest_mdd" in out


def test_optimize_portfolio_forecast_heuristic_fallback() -> None:
    repo = FakeRepo()
    repo._preds = []  # no BQ forecast data
    qt = QuantTools(repo=repo, settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="forecast", lookback_days=20)
    # With zero forecast coverage, tool degrades to HRP instead of silently
    # running forecast_max_sharpe with empty predicted_mu.
    assert out["strategy"] == "hrp"
    assert out["status"] == "degraded"
    assert "forecast_coverage_insufficient" in out["degraded_reasons"]
    assert out["forecast_coverage"] == 0.0
    assert out["strategy_requested"] == "forecast"
    w = out["weights"]
    assert set(w.keys()) == {"AAPL", "MSFT"}
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_optimize_portfolio_invalid_strategy() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="invalid_xyz")
    assert "error" in out


def test_optimize_portfolio_partial_excludes_short_history() -> None:
    class _Repo(FakeRepo):
        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            out = super().get_daily_closes(tickers=tickers, lookback_days=lookback_days, sources=sources)
            out["TSLA"] = [100.0, 101.0, 102.0]  # insufficient (<10)
            return out

    qt = QuantTools(repo=_Repo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT", "TSLA"], strategy="risk_parity", lookback_days=20)
    assert out["status"] == "ok"
    assert out["data_quality"]["status"] == "partial"
    assert out["data_quality"]["usable_tickers"] == 2
    assert any(
        e["ticker"] == "TSLA" and e["reason"] == "insufficient_history"
        for e in out["data_quality"]["excluded"]
    )
    assert set(out["weights"].keys()) == {"AAPL", "MSFT"}


def test_optimize_portfolio_unusable_returns_graceful_error() -> None:
    class _Repo(FakeRepo):
        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            return {}  # no data for any ticker

    qt = QuantTools(repo=_Repo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="sharpe", lookback_days=20)
    assert out["status"] == "unusable"
    assert out["data_quality"]["status"] == "unusable"
    assert out["data_quality"]["usable_tickers"] == 0
    assert len(out["data_quality"]["excluded"]) == 2
    assert "error" in out


def test_optimize_portfolio_decision_summary_without_context_is_suggestion() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    ds = out["decision_summary"]
    assert ds["headline_code"] == "no_current_portfolio"
    assert ds["confidence"] == "low"
    assert ds["turnover"] == 0.0


def test_optimize_portfolio_decision_summary_rotate() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    qt.set_context({
        "target_market": "nasdaq",
        "portfolio": {"positions": {"TSLA": {"quantity": 1.0, "avg_price_krw": 100.0}}},
    })
    out = qt.optimize_portfolio(["AAPL", "MSFT", "TSLA"], strategy="risk_parity", lookback_days=20)
    ds = out["decision_summary"]
    # Starting from 100% TSLA, HRP spreads across AAPL/MSFT → rotate (both BUY and SELL).
    assert ds["headline_code"] == "rotate"
    assert ds["confidence"] in {"medium", "high"}
    assert ds["turnover"] > 0.03
    # Canonical vocabulary — no hype words.
    for bad in ("strong", "guaranteed", "best", "must"):
        assert bad not in ds["headline"].lower()


def test_optimize_portfolio_evidence_gaps_emitted() -> None:
    class _Repo(FakeRepo):
        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            out = super().get_daily_closes(tickers=tickers, lookback_days=lookback_days, sources=sources)
            out["TSLA"] = [100.0, 101.0]  # insufficient
            return out

    qt = QuantTools(repo=_Repo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT", "TSLA"], strategy="forecast", lookback_days=20)
    assert "some_tickers_excluded" in out.get("evidence_gaps", [])
    notes = out["validation_notes"]
    assert any("timing" in n.lower() for n in notes)


def test_optimize_portfolio_binding_forecast_preserves_forecast_basis() -> None:
    # Binding constraints on a forecast allocation must recompute stats on the
    # forecast-blended mu basis — not historical mu — so the reported
    # expected_return_daily/sharpe_daily stay coherent with the optimizer.
    import numpy as np
    from arena.tools.allocation import blend_forecast_mu, recompute_stats

    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=_settings())
    out = qt.optimize_portfolio(
        ["AAPL", "MSFT", "PLTD"],
        strategy="forecast",
        mu_confidence=1.0,
        lookback_days=20,
        max_weight=0.40,  # binding: PLTD would otherwise dominate
    )
    assert "constraints_applied" in out

    tickers = out["tickers"]
    closes = repo.get_daily_closes(tickers=tickers, lookback_days=21)
    aligned = np.stack([np.array(closes[t], dtype=float) for t in tickers], axis=1)
    rets = (aligned[1:] / aligned[:-1]) - 1.0
    predicted_mu = {p["ticker"]: p["exp_return_period"] for p in repo._preds}
    mu_blended = blend_forecast_mu(tickers, rets, predicted_mu, mu_confidence=1.0)

    exp_ret_forecast, vol_forecast, sharpe_forecast = recompute_stats(
        tickers, out["weights"], rets, mu_override=mu_blended,
    )
    exp_ret_hist, _, _ = recompute_stats(tickers, out["weights"], rets)

    # Output must match forecast-basis recompute, not historical-only.
    assert out["expected_return_daily"] == pytest.approx(exp_ret_forecast, abs=1e-6)
    assert out["volatility_daily"] == pytest.approx(vol_forecast, abs=1e-6)
    assert out["sharpe_daily"] == pytest.approx(sharpe_forecast, abs=1e-3)
    # Forecast vs historical basis differ materially in this fixture.
    assert abs(exp_ret_forecast - exp_ret_hist) > 1e-6


def test_optimize_portfolio_non_binding_constraint_preserves_forecast_stats() -> None:
    # When a constraint is passed but does not bind (e.g. max_weight=1.0),
    # weights must be unchanged AND stats must match the unconstrained call.
    # Matters most for strategy='forecast' whose mu blends historical + forecast.
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    base = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="forecast", lookback_days=20, mu_confidence=0.3)
    held = qt.optimize_portfolio(
        ["AAPL", "MSFT"], strategy="forecast", lookback_days=20, mu_confidence=0.3,
        max_weight=1.0,  # non-binding
    )
    assert held["weights"] == base["weights"]
    assert held["expected_return_daily"] == base["expected_return_daily"]
    assert held["sharpe_daily"] == base["sharpe_daily"]
    assert held["volatility_daily"] == base["volatility_daily"]
    assert "constraints_applied" not in held


def test_optimize_portfolio_cash_buffer_recomputes_stats() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    base = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    buffered = qt.optimize_portfolio(
        ["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20, cash_buffer=0.50,
    )
    # Weights scaled down, expected_return + volatility should scale accordingly.
    assert sum(buffered["weights"].values()) == pytest.approx(0.50, abs=1e-6)
    # Rounded to 6 decimals in the output — tolerate rounding noise.
    assert buffered["expected_return_daily"] == pytest.approx(base["expected_return_daily"] * 0.5, abs=1e-6)
    assert buffered["volatility_daily"] == pytest.approx(base["volatility_daily"] * 0.5, abs=1e-6)


def test_optimize_portfolio_min_weight_preserves_backtest_mdd() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    out = qt.optimize_portfolio(
        ["AAPL", "MSFT", "TSLA"],
        strategy="risk_parity",
        lookback_days=20,
        min_weight=0.40,  # drops at least one name
    )
    # Shape mismatch would silently drop backtest_mdd — assert it survives.
    assert "backtest_mdd" in out
    assert out["backtest_mdd"]["value"] <= 0.0


def test_optimize_portfolio_aligned_portfolio_headline_is_hold() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    # First compute the optimizer's target weights with no context.
    suggestion = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    target = suggestion["weights"]
    # Now set a portfolio exactly matching the target weights (quantity * avg_price = weight).
    qt.set_context({
        "target_market": "nasdaq",
        "portfolio": {"positions": {
            "AAPL": {"quantity": target["AAPL"] * 100.0, "avg_price_krw": 1.0},
            "MSFT": {"quantity": target["MSFT"] * 100.0, "avg_price_krw": 1.0},
        }},
    })
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="risk_parity", lookback_days=20)
    assert out["rebalance_orders"] == []
    assert out["decision_summary"]["headline_code"] == "hold"


def test_optimize_portfolio_single_usable_ticker() -> None:
    class _Repo(FakeRepo):
        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            out = super().get_daily_closes(tickers=tickers, lookback_days=lookback_days, sources=sources)
            out.pop("MSFT", None)  # only AAPL usable
            return out

    qt = QuantTools(repo=_Repo(), settings=_settings())
    out = qt.optimize_portfolio(["AAPL", "MSFT"], strategy="sharpe", lookback_days=20)
    assert out["status"] == "degraded"
    assert "single_usable_ticker" in out["degraded_reasons"]
    assert out["strategy"] == "single_name"
    assert out["weights"] == {"AAPL": 1.0}
    assert out["data_quality"]["usable_tickers"] == 1


def test_forecast_returns_reads_predictions() -> None:
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=_settings())
    rows = qt.forecast_returns()
    assert len(rows) >= 1
    assert rows[0]["run_date"]
    assert "ticker" in rows[0]
    assert "exp_return_period" in rows[0]
    assert all(r["ticker"] in {"AAPL", "MSFT", "TSLA", "PLTD"} for r in rows)
    assert repo.last_forecast_mode == "all"


def test_forecast_returns_prefers_dynamic_candidate_tickers_from_context() -> None:
    class _Repo(FakeRepo):
        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

    repo = _Repo()
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=repo, settings=settings)
    context = {
        "target_market": "nasdaq",
        "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
        "_candidate_tickers": ["TSLA"],
    }
    qt.set_context(context)
    context["_candidate_tickers"] = ["PLTD", "TSLA"]

    rows = qt.forecast_returns()

    assert set(repo.last_forecast_tickers) == {"AAPL", "PLTD", "TSLA"}
    assert {row["ticker"] for row in rows} == {"AAPL", "PLTD", "TSLA"}


def test_forecast_returns_prefers_opportunity_working_set_over_raw_candidate_list() -> None:
    class _Repo(FakeRepo):
        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

    repo = _Repo()
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=repo, settings=settings)
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
            "_candidate_tickers": ["TSLA"],
            "opportunity_working_set": [{"ticker": "PLTD", "status": "pending"}],
        }
    )

    rows = qt.forecast_returns()

    assert set(repo.last_forecast_tickers) == {"AAPL", "PLTD"}
    assert {row["ticker"] for row in rows} == {"AAPL", "PLTD"}


def test_forecast_returns_prefers_full_discovered_basket_over_working_set() -> None:
    class _Repo(FakeRepo):
        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

    repo = _Repo()
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=repo, settings=settings)
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
            "_candidate_tickers": ["TSLA"],
            "_discovered_candidate_tickers": ["PLTD", "TSLA", "MSFT"],
            "opportunity_working_set": [{"ticker": "PLTD", "status": "pending"}],
        }
    )

    rows = qt.forecast_returns()

    assert set(repo.last_forecast_tickers) == {"AAPL", "PLTD", "TSLA", "MSFT"}
    assert {row["ticker"] for row in rows} == {"AAPL", "PLTD", "TSLA", "MSFT"}


def test_forecast_returns_compacts_model_rows_by_ticker() -> None:
    repo = FakeRepo()
    repo._preds = [
        {
            "run_date": "2026-03-12",
            "ticker": "AAPL",
            "exp_return_period": 0.1257810646,
            "forecast_horizon": 20,
            "forecast_model": "iTransformer",
            "is_stacked": False,
            "forecast_score": -0.0325,
            "prob_up": 1.0,
            "model_votes_up": 4,
            "model_votes_total": 4,
            "consensus": "STRONG_BUY",
        },
        {
            "run_date": "2026-03-12",
            "ticker": "AAPL",
            "exp_return_period": 0.0777303991,
            "forecast_horizon": 20,
            "forecast_model": "PatchTST",
            "is_stacked": False,
            "forecast_score": -0.0317,
            "prob_up": 1.0,
            "model_votes_up": 4,
            "model_votes_total": 4,
            "consensus": "STRONG_BUY",
        },
        {
            "run_date": "2026-03-12",
            "ticker": "AAPL",
            "exp_return_period": 0.0697741524,
            "forecast_horizon": 20,
            "forecast_model": "ensemble_avg",
            "is_stacked": True,
            "forecast_score": -0.0307,
            "prob_up": 1.0,
            "model_votes_up": 4,
            "model_votes_total": 4,
            "consensus": "STRONG_BUY",
        },
        {
            "run_date": "2026-03-12",
            "ticker": "AAPL",
            "exp_return_period": 0.0683418329,
            "forecast_horizon": 20,
            "forecast_model": "ensemble_wmae",
            "is_stacked": True,
            "forecast_score": -0.0304,
            "prob_up": 1.0,
            "model_votes_up": 4,
            "model_votes_total": 4,
            "consensus": "STRONG_BUY",
        },
    ]
    qt = QuantTools(repo=repo, settings=_settings())

    rows = qt.forecast_returns(tickers=["AAPL"])

    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "AAPL"
    assert row["forecast_model"] == "ensemble_wmae"
    assert row["is_stacked"] is True
    assert row["consensus"] == "STRONG_BUY"
    assert row["model_votes_up"] == 4
    assert row["model_votes_total"] == 4
    assert len(row["stacked_models"]) == 2
    assert len(row["base_models"]) == 2
    assert row["best_base_model"] == "iTransformer"
    assert row["best_base_return"] == 0.1257810646


def test_forecast_returns_forwards_mode_setting() -> None:
    settings = _settings()
    settings.forecast_mode = "stacked"
    settings.forecast_table = "my_proj.llm_arena.predicted_expected_returns"
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=settings)
    _ = qt.forecast_returns()
    assert repo.last_forecast_mode == "stacked"
    assert repo.last_forecast_table == "my_proj.llm_arena.predicted_expected_returns"


def test_forecast_returns_invalid_mode_falls_back_to_default_mode() -> None:
    settings = _settings()
    settings.forecast_mode = "all"
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=settings)

    rows = qt.forecast_returns(tickers=["AAPL"], forecast_mode="balanced")

    assert rows
    assert repo.last_forecast_mode == "all"


def test_forecast_returns_auto_build_retries_with_relaxed_config(monkeypatch) -> None:
    repo = FakeRepo()
    repo._preds = []
    qt = QuantTools(repo=repo, settings=_settings())
    monkeypatch.setenv("ARENA_FORECAST_AUTO_BUILD", "true")

    attempts: list[dict] = []

    def _fake_build(repo_obj, settings_obj, **cfg):
        _ = settings_obj
        attempts.append(dict(cfg))
        if len(attempts) == 1:
            return SimpleNamespace(rows_written=0, tickers_used=0, used_neuralforecast=False, note="insufficient series length")
        repo_obj._preds = [{"run_date": "2026-02-24", "ticker": "AAPL", "exp_return_period": 0.02, "forecast_horizon": 20}]
        return SimpleNamespace(rows_written=1, tickers_used=1, used_neuralforecast=False, note="ok")

    monkeypatch.setattr("arena.forecasting.build_and_store_stacked_forecasts", _fake_build)

    rows = qt.forecast_returns(tickers=["AAPL"])

    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert len(attempts) == 2
    assert int(attempts[0]["min_series_length"]) == 160
    assert int(attempts[1]["min_series_length"]) == 90


def test_forecast_returns_returns_empty_when_all_auto_build_attempts_fail(monkeypatch) -> None:
    from arena.tools import quant_tools as _qt_mod
    _qt_mod._forecast_built_dates.clear()

    repo = FakeRepo()
    repo._preds = []
    qt = QuantTools(repo=repo, settings=_settings())
    monkeypatch.setenv("ARENA_FORECAST_AUTO_BUILD", "true")

    attempts: list[dict] = []

    def _fake_build(repo_obj, settings_obj, **cfg):
        _ = (repo_obj, settings_obj)
        attempts.append(dict(cfg))
        return SimpleNamespace(rows_written=0, tickers_used=0, used_neuralforecast=False, note="insufficient series length")

    monkeypatch.setattr("arena.forecasting.build_and_store_stacked_forecasts", _fake_build)

    rows = qt.forecast_returns(tickers=["AAPL"])

    assert rows == []
    assert len(attempts) == 3


def test_forecast_returns_returns_empty_when_auto_build_disabled(monkeypatch) -> None:
    repo = FakeRepo()
    repo._preds = []
    qt = QuantTools(repo=repo, settings=_settings())
    monkeypatch.delenv("ARENA_FORECAST_AUTO_BUILD", raising=False)

    def _should_not_call(self):
        raise AssertionError("auto-build should not run when disabled")

    monkeypatch.setattr(QuantTools, "_auto_build_forecasts_if_needed", _should_not_call)
    rows = qt.forecast_returns(tickers=["AAPL", "MSFT"])

    assert rows == []


def test_screen_market_excludes_tickers_outside_default_universe() -> None:
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT"]
    qt = QuantTools(repo=FakeRepo(), settings=settings)
    rows = qt.screen_market(top_n=10)
    tickers = [str(r.get("ticker", "")).upper() for r in rows]
    assert "PLTD" not in tickers


def test_sector_summary_groups() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    rows = qt.sector_summary("20d")
    assert rows
    assert "sector" in rows[0]
    assert "avg_ret" in rows[0]


def test_get_fundamentals_filters_to_target_universe() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings(), ot_client=FakeOpenTradingClient())
    out = qt.get_fundamentals(["AAPL", "XYZ"], excd="NAS", max_items=10)
    assert out["eligible"] == ["AAPL"]
    assert out["excluded"] == ["XYZ"]
    assert out["rows"]
    assert out["rows"][0]["ticker"] == "AAPL"
    assert out["rows"][0]["per"] == 31.5


def test_get_fundamentals_defaults_to_opportunity_working_set() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings(), ot_client=FakeOpenTradingClient())
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
            "opportunity_working_set": [{"ticker": "MSFT", "status": "pending"}],
        }
    )

    out = qt.get_fundamentals(max_items=10)

    assert out["eligible"] == ["MSFT"]
    assert out["rows"]
    assert out["rows"][0]["ticker"] == "MSFT"


def test_get_fundamentals_normalizes_generic_us_exchange() -> None:
    client = FakeOpenTradingClient()
    qt = QuantTools(repo=FakeRepo(), settings=_settings(), ot_client=client)

    out = qt.get_fundamentals(["AAPL"], excd="US", max_items=10)

    assert out["rows"]
    assert out["rows"][0]["ticker"] == "AAPL"
    assert out["rows"][0]["exchange"] == "NAS"
    assert out["rows"][0]["per"] == 31.5
    assert client.overseas_price_detail_calls == [("AAPL", "NAS")]
