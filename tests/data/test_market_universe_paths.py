from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from arena.data.bigquery.ledger_store import LedgerStore
from arena.data.bigquery.market_store import MarketStore
from arena.data.bigquery.sleeve_store import SleeveStore
from arena.models import AccountSnapshot, Position


# ---------------------------------------------------------------------------
# Shared fake infrastructure
# ---------------------------------------------------------------------------

from tests.data.strict_path_helpers import (
    _InsertClient,
    _FakeSession,
    _ForecastSchemaField,
    _ForecastSchemaClient,
    _LoadJob,
    _MarketWriteClient,
    _make_market_store,
    _make_forecast_query_store,
    _make_market_write_store,
    _SleeveStoreForBuild,
    _SleeveStoreForInit,
    _SleeveStoreForRetarget,
    _LedgerStoreForCapitalReplay,
    _MarketStoreForReplay,
    _SleeveStoreForCapitalReplay,
    _SleeveStoreForCapitalRetarget,
    _LedgerStoreForCapitalRetarget,
    _NavSleeveStore,
    _ActualBasisSleeveStore,
    _make_capital_replay_store,
)

def test_upsert_instrument_master_persists_ticker_name() -> None:
    session = _FakeSession(client=_InsertClient())
    store = MarketStore(session)

    inserted = store.upsert_instrument_master(
        [
            {
                "instrument_id": "KRX:025860",
                "ticker": "025860",
                "ticker_name": "남해화학",
                "exchange_code": "KRX",
                "currency": "KRW",
                "lot_size": 1,
                "tick_size": 1.0,
                "tradable": True,
                "status": "ACTIVE",
            }
        ]
    )

    assert inserted == 1
    assert session.client.payloads[0]["ticker_name"] == "남해화학"


def test_ticker_name_map_falls_back_to_instrument_master() -> None:
    store = _make_market_store(
        responses=[
            [
                {"ticker": "025860", "ticker_name": None},
                {"ticker": "005930", "ticker_name": "삼성전자"},
            ],
            [
                {"ticker": "025860", "ticker_name": "남해화학"},
            ],
        ]
    )

    out = store.ticker_name_map(tickers=["025860", "005930"], limit=10)

    assert out == {"025860": "남해화학", "005930": "삼성전자"}


def test_rebuild_universe_candidates_skips_rows_without_daily_history_features() -> None:
    session = _FakeSession(
        responses=[
            [
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "MISSING",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:MISSING",
                    "ret_20d": None,
                    "ret_5d": None,
                    "volatility_20d": None,
                    "sentiment_score": 1.0,
                },
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "ZERO",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:ZERO",
                    "ret_20d": 0.0,
                    "ret_5d": 0.0,
                    "volatility_20d": 0.0,
                    "sentiment_score": 0.0,
                },
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "GOOD",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:GOOD",
                    "ret_20d": 0.1,
                    "ret_5d": 0.02,
                    "volatility_20d": 0.12,
                    "sentiment_score": 0.2,
                },
            ],
            [],
        ],
        client=_InsertClient(),
    )
    store = MarketStore(session)

    out = store.rebuild_universe_candidates(top_n=10, allowed_tickers=["MISSING", "ZERO", "GOOD"])

    assert out["count"] == 2
    written = {row["ticker"] for row in session.client.payloads}
    assert written == {"ZERO", "GOOD"}


def test_rebuild_universe_candidates_supplements_allowed_tickers_from_latest_market_features() -> None:
    def row(ticker: str, ret_20d: float) -> dict[str, object]:
        return {
            "as_of_ts": "2026-01-01T00:00:00+00:00",
            "ticker": ticker,
            "exchange_code": "NASD",
            "instrument_id": f"NASD:{ticker}",
            "ret_20d": ret_20d,
            "ret_5d": ret_20d / 2,
            "volatility_20d": 0.12,
            "sentiment_score": 0.0,
        }

    session = _FakeSession(
        responses=[
            [row("ALLOW", 0.03)],
            [row("EXTRA1", 0.08), row("EXTRA2", 0.06)],
        ],
        client=_InsertClient(),
    )
    store = MarketStore(session)

    out = store.rebuild_universe_candidates(top_n=3, per_exchange_cap=3, allowed_tickers=["ALLOW"])

    assert out["count"] == 3
    written = {row["ticker"] for row in session.client.payloads}
    assert written == {"ALLOW", "EXTRA1", "EXTRA2"}
    assert session.call_pairs[0][1]["tickers"] == ["ALLOW"]
    assert "tickers" not in session.call_pairs[1][1]


def test_latest_universe_candidate_tickers_scopes_latest_run_by_market() -> None:
    session = _FakeSession(
        responses=[
            [
                {"ticker": "AAPL"},
                {"ticker": "MSFT"},
            ]
        ]
    )
    store = MarketStore(session)

    out = store.latest_universe_candidate_tickers(limit=10, markets=["nasdaq"])

    assert out == ["AAPL", "MSFT"]
    sql, params = session.call_pairs[0]
    assert "FROM scoped" in sql
    assert "IN UNNEST(@markets)" in sql
    assert params["markets"] == ["us"]
    assert params["limit"] == 10


# ---------------------------------------------------------------------------
# SleeveStore subclasses (override methods the tests customise)
# ---------------------------------------------------------------------------
