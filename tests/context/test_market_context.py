from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from arena.config import Settings
from arena.context import ContextBuilder
from arena.memory.policy import normalize_memory_policy
from arena.models import AccountSnapshot, Position, utc_now

from tests.context.helpers import (
    FakeRepo,
    FakeMemory,
    FakeBoard,
    FakeVectorStore,
    _settings,
)

def test_context_builder_falls_back_to_default_universe() -> None:
    repo = FakeRepo()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "PLTD": Position(
                ticker="PLTD",
                quantity=1,
                avg_price_krw=10_000,
                market_price_krw=12_000,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["market_features"]
    assert context["market_features"][0]["ticker"] == "AAPL"
    assert repo.calls == [["PLTD"], ["AAPL", "MSFT"]]
    assert context["risk_policy"]["max_order_krw"] == 350_000
    assert context["risk_policy"]["min_cash_buffer_ratio"] == 0.10
    assert context["risk_policy"]["max_daily_orders"] is None
    assert context["risk_policy"]["max_daily_orders_unlimited"] is True
    assert context["risk_policy"]["sleeve_capital_krw"] == 2_000_000
    assert context["order_budget"]["cash_krw"] == 1_000_000
    assert context["order_budget"]["min_cash_required_krw"] == 120_000
    assert context["order_budget"]["max_buy_notional_by_sleeve_krw"] == 880_000
    assert context["order_budget"]["max_buy_notional_krw"] == 350_000
    assert context["order_budget"]["daily_orders_cap"] is None
    assert context["sleeve_state"]["target_sleeve_krw"] == 2_000_000
    assert context["sleeve_state"]["current_equity_krw"] == 1_200_000
    assert "Positions:" not in context["performance_context"]
    assert "Budget " not in context["performance_context"]
    assert "Daily orders" not in context["performance_context"]
    assert "Cash " not in context["performance_context"]


def test_context_builder_normalizes_market_features_from_raw_daily_closes() -> None:
    class RepoWithRawCloses(FakeRepo):
        def __init__(self):
            super().__init__()
            self.close_sources = None

        def latest_market_features(self, tickers, limit, sources=None):
            _ = (limit, sources)
            self.calls.append(list(tickers))
            return [
                {
                    "ticker": "AAPL",
                    "close_price_krw": 1000.0,
                    "ret_5d": 0.0,
                    "ret_20d": 0.0,
                    "volatility_20d": 0.0,
                }
            ]

        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            _ = (tickers, lookback_days)
            self.close_sources = list(sources or [])
            return {"AAPL": [100.0 + idx for idx in range(22)]}

    settings = _settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "nasdaq"
    repo = RepoWithRawCloses()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=10_000,
                market_price_krw=12_000,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    row = context["market_features"][0]
    closes = [100.0 + idx for idx in range(22)]
    assert math.isclose(row["ret_5d"], (closes[-1] / closes[-6]) - 1.0)
    assert math.isclose(row["ret_20d"], (closes[-1] / closes[-21]) - 1.0)
    assert row["volatility_20d"] > 0.0
    assert repo.close_sources == ["open_trading_nasdaq", "open_trading_us"]


def test_context_builder_loads_ticker_names_for_current_positions() -> None:
    repo = FakeRepo()
    repo.ticker_name_rows = {"025860": "남해화학"}
    settings = _settings()
    settings.kis_target_market = "kospi"
    settings.default_universe = []
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "025860": Position(
                ticker="025860",
                quantity=2,
                avg_price_krw=8_000,
                market_price_krw=8_270,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert repo.ticker_name_map_calls
    assert repo.ticker_name_map_calls[-1][0] == ["025860"]
    assert context["ticker_names"]["025860"] == "남해화학"
    assert context["performance"]["positions"][0]["ticker_name"] == "남해화학"
    assert context["sleeve_state"]["sleeve_remaining_krw"] == 880_000
    assert context["sleeve_state"]["over_target"] is False
    assert context["sleeve_state"]["buy_blocked"] is False
    assert "Long-horizon compounding" in context["investment_style_context"]


def test_context_builder_uses_krw_display_when_us_fx_is_unavailable() -> None:
    repo = FakeRepo()
    settings = _settings()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        usd_krw_rate=0.0,
        cash_foreign=500.0,
        cash_foreign_currency="USD",
        positions={},
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["performance"]["display_currency"] == "KRW"
    assert "usd_krw_rate" not in context["order_budget"]
    assert context["order_budget"]["cash_usd"] == pytest.approx(500.0)


def test_context_builder_falls_back_to_runtime_universe_candidates() -> None:
    repo = FakeRepo()
    settings = _settings()
    settings.default_universe = []
    settings.universe_run_top_n = 2
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "PLTD": Position(
                ticker="PLTD",
                quantity=1,
                avg_price_krw=10_000,
                market_price_krw=12_000,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["market_features"]
    assert context["market_features"][0]["ticker"] == "AAPL"
    assert repo.calls == [["PLTD"], ["AAPL", "MSFT"]]
    assert repo.last_universe_limit == 2


def test_context_builder_includes_active_thesis_context_for_holdings() -> None:
    repo = FakeRepo()
    repo.active_thesis_rows["AAPL"] = {
        "event_id": "mem_thesis",
        "event_type": "thesis_update",
        "summary": "AAPL thesis update action=add status=FILLED thesis=AI demand and margin recovery",
        "payload_json": json.dumps(
            {
                "thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_open",
                "ticker": "AAPL",
                "state": "active",
                "thesis_summary": "AI demand and margin recovery",
                "strategy_refs": ["momentum", "earnings_growth"],
            }
        ),
    }
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=2,
                avg_price_krw=100_000,
                market_price_krw=105_000,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert "Active Thesis:" in context["active_thesis_context"]
    assert "AAPL" in context["active_thesis_context"]
    assert "AI demand and margin recovery" in context["active_thesis_context"]
    assert context["active_theses"][0]["event_type"] == "thesis_update"
