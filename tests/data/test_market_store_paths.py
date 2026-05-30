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

def test_latest_close_prices_propagates_latest_table_failure() -> None:
    store = _make_market_store([RuntimeError("bq down")])

    with pytest.raises(RuntimeError, match="bq down"):
        store.latest_close_prices(tickers=["AAPL"])

    assert len(store.session.calls) == 1
    assert "market_features_latest" in store.session.calls[0]


def test_latest_market_features_does_not_retry_legacy_on_empty_rows() -> None:
    store = _make_market_store([[]])

    rows = store.latest_market_features(tickers=["AAPL"], limit=5)

    assert rows == []
    assert len(store.session.calls) == 1
    assert "market_features_latest" in store.session.calls[0]
    assert "ret_5d IS NOT NULL AND ret_20d IS NOT NULL AND volatility_20d IS NOT NULL" in store.session.calls[0]


def test_latest_missing_daily_feature_tickers_queries_newest_incomplete_snapshots() -> None:
    store = _make_market_store(
        [
            [
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "MISS",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:MISS",
                    "source": "open_trading_us_quote",
                }
            ]
        ]
    )

    rows = store.latest_missing_daily_feature_tickers(sources=["open_trading_us_quote"], limit=5)

    assert rows[0]["ticker"] == "MISS"
    sql, params = store.session.call_pairs[-1]
    assert "market_features_latest" in sql
    assert "ret_5d IS NULL OR ret_20d IS NULL OR volatility_20d IS NULL" in sql
    assert "has_complete_features = 0" in sql
    assert params["sources"] == ["open_trading_us_quote"]


def test_screen_latest_features_does_not_retry_legacy_on_empty_rows() -> None:
    store = _make_market_store([[]])

    rows = store.screen_latest_features(top_n=3)

    assert rows == []
    assert len(store.session.calls) == 1
    assert "market_features_latest" in store.session.calls[0]
    assert "ret_5d IS NOT NULL" in store.session.calls[0]
    assert "ret_20d IS NOT NULL" in store.session.calls[0]
    assert "volatility_20d IS NOT NULL" in store.session.calls[0]


def test_refresh_signal_daily_values_uses_point_in_time_sources() -> None:
    store = _make_market_store([])

    out = store.refresh_signal_daily_values(
        lookback_days=180,
        horizon_days=20,
        sources=["open_trading_us"],
        market="us",
    )

    assert out == 0
    sql, params = store.session.call_pairs[-1]
    assert "signal_daily_values" in sql
    assert "market_features" in sql
    assert "predicted_expected_returns" in sql
    assert "f.run_date <= c.as_of_date" in sql
    assert "fundamentals_derived_daily" in sql
    assert "d.latest_announcement_date <= w.as_of_date" in sql
    assert params["sources"] == ["open_trading_us"]
    assert params["market"] == "us"


def test_latest_opportunity_ranker_scores_reads_fresh_latest_batch() -> None:
    store = _make_market_store([[{"ticker": "AAPL", "recommendation_score": 0.2}]])

    rows = store.latest_opportunity_ranker_scores(
        tickers=["AAPL"],
        profiles=["aggressive"],
        buckets=["momentum"],
        per_profile_limit=2,
        limit=3,
        max_age_hours=12,
    )

    assert rows[0]["ticker"] == "AAPL"
    sql, params = store.session.call_pairs[-1]
    assert "opportunity_ranker_scores_latest" in sql
    assert "latest_batch" in sql
    assert "score_source IN UNNEST(@score_sources)" in sql
    assert "s.score_source IN UNNEST(@score_sources)" in sql
    assert "s.ticker IN UNNEST(@tickers)" in sql
    assert "s.profile IN UNNEST(@profiles)" in sql
    assert "s.bucket IN UNNEST(@buckets)" in sql
    assert "profile_rn <= @per_profile_limit" in sql
    assert "global_rn <= @limit" in sql
    assert "PARTITION BY market" in sql
    assert "s.market = b.market" in sql
    assert params["max_age_hours"] == 12
    assert params["buckets"] == ["momentum"]
    assert params["score_sources"] == ["joint_policy_v1"]
    assert params["per_profile_limit"] == 2
    assert params["max_return_rows"] == 19
    assert "markets" not in params


def test_load_signal_policy_training_rows_reads_label_ready_values() -> None:
    store = _make_market_store([[{"ticker": "AAPL", "fwd_excess_return_20d": 0.02}]])

    rows = store.load_signal_policy_training_rows(lookback_days=180, market="us")

    assert rows[0]["ticker"] == "AAPL"
    sql, params = store.session.call_pairs[-1]
    assert "signal_daily_values" in sql
    assert "label_ready" in sql
    assert "fwd_excess_return_20d IS NOT NULL" in sql
    assert "signal_momentum_20d" in sql
    assert "signal_low_debt" in sql
    assert params["lookback_days"] == 180
    assert params["market"] == "us"


def test_latest_opportunity_ranker_scores_filters_by_market() -> None:
    store = _make_market_store([[{"ticker": "AAPL", "market": "us"}]])

    rows = store.latest_opportunity_ranker_scores(
        markets=["us"],
        limit=5,
        max_age_hours=6,
    )

    assert rows[0]["ticker"] == "AAPL"
    sql, params = store.session.call_pairs[-1]
    assert "s.market IN UNNEST(@markets)" in sql
    assert "market IN UNNEST(@markets)" in sql
    assert params["markets"] == ["us"]


def test_insert_opportunity_ranker_scores_latest_appends_json_rows() -> None:
    store = _make_market_write_store()

    inserted = store.insert_opportunity_ranker_scores_latest(
        [
            {
                "as_of_date": "2026-04-17",
                "computed_at": "2026-04-18T00:00:00+00:00",
                "ranker_version": "ranker",
                "score_source": "learned",
                "ticker": "aapl",
                "recommendation_score": 0.12,
                "feature_json": '{"ret_20d": 0.1}',
                "explanation_json": {"top_features": ["ret_20d"]},
            }
        ]
    )

    assert inserted == 1
    table_id, rows = store.session.client.loads[-1]
    assert table_id == "proj.ds.opportunity_ranker_scores_latest"
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["feature_json"] == {"ret_20d": 0.1}


def test_latest_fundamentals_snapshot_does_not_retry_on_empty_rows() -> None:
    store = _make_market_store([[]])

    rows = store.latest_fundamentals_snapshot(tickers=["AAPL"], limit=5)

    assert rows == []
    assert len(store.session.calls) == 1
    assert "fundamentals_snapshot_latest" in store.session.calls[0]


def test_get_daily_closes_deduplicates_same_timestamp_rows() -> None:
    store = _make_market_store(
        [
            [
                {"as_of_ts": "2026-02-20T00:00:00+00:00", "ticker": "AAPL", "close_price_krw": 100.0},
                {"as_of_ts": "2026-02-20T00:00:00+00:00", "ticker": "AAPL", "close_price_krw": 101.0},
                {"as_of_ts": "2026-02-21T00:00:00+00:00", "ticker": "AAPL", "close_price_krw": 102.0},
            ]
        ]
    )

    rows = store.get_daily_closes(tickers=["AAPL"], lookback_days=10)

    assert rows["AAPL"] == [101.0, 102.0]


def test_get_daily_close_frame_deduplicates_same_day_rows() -> None:
    store = _make_market_store(
        [
            [
                {"d": "2026-02-20", "ticker": "AAPL", "close_price": 100.0},
                {"d": "2026-02-20", "ticker": "AAPL", "close_price": 101.0},
                {"d": "2026-02-21", "ticker": "AAPL", "close_price": 102.0},
                {"d": "2026-02-20", "ticker": "MSFT", "close_price": 200.0},
            ]
        ]
    )

    frame = store.get_daily_close_frame(
        tickers=["AAPL", "MSFT"],
        start=date(2026, 2, 20),
        end=date(2026, 2, 21),
    )

    assert float(frame.loc["2026-02-20", "AAPL"]) == 101.0
    assert float(frame.loc["2026-02-21", "AAPL"]) == 102.0
    assert float(frame.loc["2026-02-20", "MSFT"]) == 200.0


def test_get_daily_close_frame_supports_native_price_field() -> None:
    store = _make_market_store(
        [
            [
                {"d": "2026-02-20", "ticker": "AAPL", "close_price": 100.0},
                {"d": "2026-02-21", "ticker": "AAPL", "close_price": 105.0},
            ]
        ]
    )

    frame = store.get_daily_close_frame(
        tickers=["AAPL"],
        start=date(2026, 2, 20),
        end=date(2026, 2, 21),
        price_field="close_price_native",
    )

    assert "close_price_native AS close_price" in store.session.calls[0]
    assert float(frame.loc["2026-02-21", "AAPL"]) == 105.0


def test_insert_market_features_appends_via_load_job_without_delete() -> None:
    store = _make_market_write_store()

    store.insert_market_features(
        [
            {
                "ticker": "005930",
                "source": "open_trading_kospi",
                "exchange_code": "KRX",
                "instrument_id": "KRX:005930",
                "as_of_ts": "2026-03-07T00:00:00+00:00",
                "close_price_krw": 100.0,
            }
        ]
    )

    assert len(store.session.client.loads) == 1
    table_id, rows = store.session.client.loads[0]
    assert table_id == "proj.ds.market_features"
    assert rows[0]["ticker"] == "005930"
    assert rows[0]["source"] == "open_trading_kospi"
    assert "ingested_at" in rows[0]


def test_earliest_market_feature_date_queries_min_as_of_date() -> None:
    store = _make_market_store([[{"start_date": date(2026, 3, 1)}]])

    out = store.earliest_market_feature_date()

    assert out == date(2026, 3, 1)
    sql, _params = store.session.call_pairs[-1]
    assert "MIN(DATE(as_of_ts)) AS start_date" in sql
    assert "market_features" in sql


def test_latest_macro_indicator_observation_date_filters_by_sources() -> None:
    store = _make_market_store([[{"latest_date": date(2026, 5, 29)}]])

    out = store.latest_macro_indicator_observation_date(sources=["fred", "ecos"])

    assert out == date(2026, 5, 29)
    sql, params = store.session.call_pairs[-1]
    assert "MAX(observation_date) AS latest_date" in sql
    assert "FROM `proj.ds.macro_indicator_observations`" in sql
    assert "source IN UNNEST(@sources)" in sql
    assert params == {"sources": ["fred", "ecos"]}


def test_macro_indicator_observation_history_queries_filtered_window() -> None:
    store = _make_market_store(
        [[{"source": "ecos", "indicator_key": "usd_krw", "observation_date": date(2026, 5, 30), "value": 1410.0}]]
    )

    rows = store.macro_indicator_observation_history(
        sources=["ecos"],
        markets=["kr"],
        indicator_keys=["usd_krw"],
        start_date=date(2026, 1, 1),
        end_date=date(2026, 5, 30),
        limit=100,
    )

    assert rows[0]["indicator_key"] == "usd_krw"
    sql, params = store.session.call_pairs[-1]
    assert "FROM `proj.ds.macro_indicator_observations`" in sql
    assert "source IN UNNEST(@sources)" in sql
    assert "market IN UNNEST(@markets)" in sql
    assert "indicator_key IN UNNEST(@indicator_keys)" in sql
    assert "observation_date >= @start_date" in sql
    assert "observation_date <= @end_date" in sql
    assert "LIMIT @limit" in sql
    assert params == {
        "sources": ["ecos"],
        "markets": ["kr"],
        "indicator_keys": ["usd_krw"],
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 5, 30),
        "limit": 100,
    }


def test_insert_macro_indicator_observations_appends_via_load_job_without_delete() -> None:
    store = _make_market_write_store()

    written = store.insert_macro_indicator_observations(
        [
            {
                "observed_at": "2026-05-30T00:00:00+00:00",
                "as_of_date": "2026-05-29",
                "source": "fred",
                "indicator_key": "treasury_10y",
                "indicator_name": "US 10Y Treasury Yield",
                "group_name": "rates_curve",
                "market": "us",
                "source_series_id": "DGS10",
                "frequency": "daily",
                "observation_date": "2026-05-29",
                "value": 4.5,
                "unit": "%",
                "raw_json": {"date": "2026-05-29", "value": "4.50"},
            }
        ]
    )

    assert written == 1
    assert len(store.session.client.loads) == 1
    table_id, rows = store.session.client.loads[0]
    assert table_id == "proj.ds.macro_indicator_observations"
    assert rows[0]["source"] == "fred"
    assert rows[0]["indicator_key"] == "treasury_10y"
    assert rows[0]["source_series_id"] == "DGS10"
    assert rows[0]["source_item_code"] is None
    assert rows[0]["value"] == 4.5
    assert rows[0]["raw_json"] == {"date": "2026-05-29", "value": "4.50"}


def test_delete_macro_indicator_observations_filters_by_range_and_source() -> None:
    store = _make_market_store([])

    store.delete_macro_indicator_observations(
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 3),
        sources=["fred", "ecos"],
    )

    sql, params = store.session.call_pairs[-1]
    assert "DELETE FROM `proj.ds.macro_indicator_observations`" in sql
    assert "observation_date BETWEEN @start_date AND @end_date" in sql
    assert "source IN UNNEST(@sources)" in sql
    assert params == {"start_date": date(2026, 3, 1), "end_date": date(2026, 3, 3), "sources": ["fred", "ecos"]}


def test_insert_market_features_latest_appends_via_load_job_without_delete() -> None:
    store = _make_market_write_store()

    written = store.insert_market_features_latest(
        [
            {
                "ticker": "005930",
                "source": "open_trading_kospi",
                "exchange_code": "KRX",
                "instrument_id": "KRX:005930",
                "as_of_ts": "2026-03-07T00:00:00+00:00",
                "close_price_krw": 100.0,
            }
        ]
    )

    assert written == 1
    assert len(store.session.client.loads) == 1
    table_id, rows = store.session.client.loads[0]
    assert table_id == "proj.ds.market_features_latest"
    assert rows[0]["ticker"] == "005930"
    assert "updated_at" in rows[0]


def test_insert_fundamentals_snapshot_latest_appends_via_load_job_without_delete() -> None:
    store = _make_market_write_store()

    written = store.insert_fundamentals_snapshot_latest(
        [
            {
                "ticker": "AAPL",
                "market": "us",
                "exchange_code": "NASD",
                "instrument_id": "NASD:AAPL",
                "currency": "USD",
                "as_of_ts": "2026-03-07T00:00:00+00:00",
                "per": 28.5,
                "pbr": 7.2,
                "eps": 6.15,
                "bps": 24.8,
                "source": "open_trading_us_price_detail",
            }
        ]
    )

    assert written == 1
    assert len(store.session.client.loads) == 1
    table_id, rows = store.session.client.loads[0]
    assert table_id == "proj.ds.fundamentals_snapshot_latest"
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["market"] == "us"
    assert "updated_at" in rows[0]


def test_replace_predicted_returns_appends_run_batch_without_delete() -> None:
    store = _make_market_write_store()

    written = store.replace_predicted_returns(
        [
            {
                "run_date": "2026-03-14",
                "ticker": "AAPL",
                "exp_return_period": 0.12,
                "forecast_horizon": 20,
                "forecast_model": "ensemble_wmae",
                "is_stacked": True,
            },
            {
                "run_date": "2026-03-14",
                "ticker": "MSFT",
                "exp_return_period": 0.08,
                "forecast_horizon": 20,
                "forecast_model": "ensemble_wmae",
                "is_stacked": True,
            },
        ],
        run_date=date(2026, 3, 14),
    )

    assert written == 2
    assert len(store.session.client.loads) == 1
    table_id, rows = store.session.client.loads[0]
    assert table_id == "proj.ds.predicted_expected_returns"
    assert rows[0]["forecast_run_id"].startswith("fc_")
    assert rows[1]["forecast_run_id"] == rows[0]["forecast_run_id"]


def test_get_predicted_returns_prefers_latest_forecast_batch_when_run_id_exists() -> None:
    store = _make_forecast_query_store(
        [
            {
                "run_date": "2026-03-14",
                "ticker": "AAPL",
                "exp_return_period": 0.12,
                "forecast_horizon": 20,
                "forecast_model": "ensemble_wmae",
                "is_stacked": True,
            }
        ],
        columns=[
            "run_date",
            "forecast_run_id",
            "ticker",
            "exp_return_period",
            "forecast_horizon",
            "forecast_model",
            "is_stacked",
            "created_at",
        ],
    )

    rows = store.get_predicted_returns(tickers=["AAPL"], limit=5, mode="stacked")

    assert rows[0]["ticker"] == "AAPL"
    assert store.session.call_pairs
    sql, params = store.session.call_pairs[-1]
    assert "latest_batch" in sql
    assert "forecast_run_id" in sql
    assert params["tickers"] == ["AAPL"]


def test_get_predicted_returns_scopes_latest_forecast_batch_per_requested_ticker() -> None:
    store = _make_forecast_query_store(
        [
            {"ticker": "AAPL", "exp_return_period": 0.12},
            {"ticker": "005930", "exp_return_period": 0.08},
        ],
        columns=[
            "run_date",
            "forecast_run_id",
            "ticker",
            "exp_return_period",
            "forecast_model",
            "is_stacked",
            "created_at",
        ],
    )

    rows = store.get_predicted_returns(tickers=["AAPL", "005930"], limit=5, mode="stacked")

    assert [row["ticker"] for row in rows] == ["AAPL", "005930"]
    sql, params = store.session.call_pairs[-1]
    assert "PARTITION BY ticker" in sql
    assert "ON r.ticker = b.ticker" in sql
    assert "r.ticker IN UNNEST(@tickers)" in sql
    assert params["tickers"] == ["AAPL", "005930"]


# ===================================================================
# Tests — Sleeve Store: build_agent_sleeve_snapshot
# ===================================================================
