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

def test_sector_summary_groups() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())
    rows = qt.sector_summary("20d")
    assert rows
    assert "sector" in rows[0]
    assert "avg_ret" in rows[0]


def test_sector_summary_uses_instrument_master_classification() -> None:
    class _Repo(FakeRepo):
        def latest_instrument_map(self, tickers):
            return {
                "AAPL": {"sector": "Technology"},
                "MSFT": {"sector": "Technology"},
                "TSLA": {"sector": "Consumer Discretionary"},
            }

    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=_Repo(), settings=settings)

    rows = qt.sector_summary("20d")

    by_sector = {row["sector"]: row for row in rows}
    assert by_sector["Technology"]["tickers"] == ["AAPL", "MSFT"]
    assert by_sector["Consumer Discretionary"]["tickers"] == ["TSLA"]
    assert by_sector["Unknown"]["tickers"] == ["PLTD"]


def test_sector_summary_market_scope_us_narrows_multi_market_agent() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self._features.extend(
                [
                    {
                        "as_of_ts": "2026-01-01T00:00:00+00:00",
                        "ticker": "005930",
                        "ret_20d": 0.18,
                        "ret_5d": 0.04,
                        "volatility_20d": 0.09,
                        "sentiment_score": 0.3,
                        "close_price_krw": 75000.0,
                        "source": "open_trading_kospi_quote",
                    },
                    {
                        "as_of_ts": "2026-01-01T00:00:00+00:00",
                        "ticker": "000660",
                        "ret_20d": 0.12,
                        "ret_5d": 0.03,
                        "volatility_20d": 0.11,
                        "sentiment_score": 0.2,
                        "close_price_krw": 190000.0,
                        "source": "open_trading_kospi_quote",
                    },
                ]
            )

    settings = _settings()
    settings.kis_target_market = "us,kospi,kosdaq"
    settings.default_universe = ["AAPL", "MSFT", "005930", "000660"]
    repo = _Repo()
    qt = QuantTools(repo=repo, settings=settings)

    rows = qt.sector_summary("20d", market_scope="us")

    assert rows
    assert repo.last_screen_kwargs["tickers"] == ["AAPL", "MSFT"]
    assert all(not str(ticker).isdigit() for row in rows for ticker in row["tickers"])


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
