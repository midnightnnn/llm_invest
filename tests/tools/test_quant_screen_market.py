from __future__ import annotations

from datetime import datetime, timezone
import math
from types import SimpleNamespace
from typing import Literal, get_args, get_origin, get_type_hints

import pytest

from arena.config import Settings
from arena.tools.quant_tools import QuantTools

from tests.tools.quant_helpers import (
    _literal_args,
    _stable_quant_tool_now,
    FakeRepo,
    FakeOpenTradingClient,
    _settings,
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


def test_screen_market_excludes_tickers_outside_default_universe() -> None:
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT"]
    qt = QuantTools(repo=FakeRepo(), settings=settings)
    rows = qt.screen_market(top_n=10)
    tickers = [str(r.get("ticker", "")).upper() for r in rows]
    assert "PLTD" not in tickers
