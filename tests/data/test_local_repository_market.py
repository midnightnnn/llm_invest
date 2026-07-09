from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from tests.data.local_repository_helpers import _now, _seed_market_features_latest, repo


def test_latest_close_prices_returns_only_positive(repo):
    now = _now()
    _seed_market_features_latest(repo, [
        {"as_of_ts": now, "ticker": "AAPL", "close_price_krw": 247000.0, "source": "test", "updated_at": now},
        {"as_of_ts": now, "ticker": "MSFT", "close_price_krw": 601000.0, "source": "test", "updated_at": now},
        {"as_of_ts": now, "ticker": "ZERO", "close_price_krw": 0.0, "source": "test", "updated_at": now},
    ])

    out = repo.latest_close_prices(tickers=["aapl", "MSFT", "ZERO", "MISSING"])
    assert out == {"AAPL": 247000.0, "MSFT": 601000.0}


def test_latest_close_prices_dedups_to_latest_row(repo):
    older = datetime(2026, 4, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 4, 28, tzinfo=timezone.utc)
    _seed_market_features_latest(repo, [
        {"as_of_ts": older, "ticker": "AAPL", "close_price_krw": 100.0, "source": "t", "updated_at": older},
        {"as_of_ts": newer, "ticker": "AAPL", "close_price_krw": 200.0, "source": "t", "updated_at": newer},
    ])
    out = repo.latest_close_prices(tickers=["AAPL"])
    assert out == {"AAPL": 200.0}


def test_latest_close_prices_with_currency_includes_native(repo):
    now = _now()
    _seed_market_features_latest(repo, [
        {
            "as_of_ts": now, "ticker": "AAPL", "close_price_krw": 247000.0,
            "close_price_native": 178.5, "quote_currency": "USD",
            "fx_rate_used": 1383.0, "source": "test", "updated_at": now,
        },
    ])
    out = repo.latest_close_prices_with_currency(tickers=["AAPL"])
    row = out["AAPL"]
    assert row["close_price_krw"] == 247000.0
    assert row["close_price_native"] == 178.5
    assert row["quote_currency"] == "USD"
    assert row["fx_rate_used"] == 1383.0


def test_latest_market_features_returns_full_rows(repo):
    now = _now()
    _seed_market_features_latest(repo, [
        {
            "as_of_ts": now, "ticker": "TSLA", "close_price_krw": 308000.0,
            "ret_5d": -0.024, "ret_20d": -0.011, "volatility_20d": 0.039,
            "sentiment_score": -0.08, "source": "test", "updated_at": now,
        },
    ])
    rows = repo.latest_market_features(tickers=["TSLA"], limit=10)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "TSLA"
    assert rows[0]["ret_5d"] == pytest.approx(-0.024)


def test_local_macro_indicator_observations_insert_and_market_feature_start(repo):
    repo.insert_market_features(
        [
            {
                "ticker": "AAPL",
                "as_of_ts": datetime(2026, 3, 2, 15, tzinfo=timezone.utc),
                "close_price_krw": 100.0,
                "source": "test",
            },
            {
                "ticker": "MSFT",
                "as_of_ts": datetime(2026, 3, 1, 15, tzinfo=timezone.utc),
                "close_price_krw": 200.0,
                "source": "test",
            },
        ]
    )

    assert repo.earliest_market_feature_date() == date(2026, 3, 1)

    written = repo.insert_macro_indicator_observations(
        [
            {
                "observed_at": datetime(2026, 5, 30, tzinfo=timezone.utc),
                "as_of_date": date(2026, 5, 29),
                "source": "fred",
                "indicator_key": "treasury_10y",
                "indicator_name": "US 10Y Treasury Yield",
                "group_name": "rates_curve",
                "market": "us",
                "source_series_id": "DGS10",
                "frequency": "daily",
                "observation_date": date(2026, 5, 29),
                "value": 4.5,
                "unit": "%",
                "raw_json": {"date": "2026-05-29", "value": "4.50"},
            }
        ]
    )

    assert written == 1
    rows = repo.fetch_rows(
        "SELECT source, indicator_key, source_series_id, observation_date, value, raw_json "
        "FROM macro_indicator_observations"
    )
    assert rows == [
        {
            "source": "fred",
            "indicator_key": "treasury_10y",
            "source_series_id": "DGS10",
            "observation_date": date(2026, 5, 29),
            "value": 4.5,
            "raw_json": '{"date": "2026-05-29", "value": "4.50"}',
        }
    ]

    assert repo.latest_macro_indicator_observation_date(sources=["fred"]) == date(2026, 5, 29)
    assert repo.latest_macro_indicator_observation_date(sources=["ecos"]) is None
    history = repo.macro_indicator_observation_history(
        sources=["fred"],
        indicator_keys=["treasury_10y"],
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 30),
    )
    assert len(history) == 1
    assert history[0]["indicator_key"] == "treasury_10y"
    assert history[0]["observation_date"] == date(2026, 5, 29)

    repo.delete_macro_indicator_observations(
        start_date=date(2026, 5, 29),
        end_date=date(2026, 5, 29),
        sources=["fred"],
    )
    assert repo.fetch_rows("SELECT COUNT(*) AS n FROM macro_indicator_observations")[0]["n"] == 0
    assert repo.latest_macro_indicator_observation_date(sources=["fred"]) is None


def test_ticker_name_map_uses_instrument_master(repo):
    now = _now()
    repo.execute(
        """
        INSERT INTO instrument_master (instrument_id, ticker, ticker_name, exchange_code, currency, updated_at)
        VALUES
          ('NASD:AAPL', 'AAPL', 'Apple Inc.', 'NASD', 'USD', ?),
          ('NASD:MSFT', 'MSFT', 'Microsoft Corp.', 'NASD', 'USD', ?)
        """,
        [now, now],
    )
    out = repo.ticker_name_map(tickers=["AAPL", "MSFT", "MISSING"])
    assert out == {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp."}


def test_latest_instrument_map_returns_dicts(repo):
    now = _now()
    repo.execute(
        """
        INSERT INTO instrument_master (
          instrument_id, ticker, ticker_name, exchange_code, currency,
          sector, industry_code, industry_name, classification_source, updated_at
        )
        VALUES ('NASD:NVDA', 'NVDA', 'NVIDIA', 'NASD', 'USD', 'Technology', '06', 'Technology', 'sec_edgar', ?)
        """,
        [now],
    )
    out = repo.latest_instrument_map(["NVDA"])
    assert "NVDA" in out
    assert out["NVDA"]["exchange_code"] == "NASD"
    assert out["NVDA"]["currency"] == "USD"
    assert out["NVDA"]["sector"] == "Technology"
    assert out["NVDA"]["industry_code"] == "06"
    assert out["NVDA"]["industry_name"] == "Technology"
    assert out["NVDA"]["classification_source"] == "sec_edgar"
