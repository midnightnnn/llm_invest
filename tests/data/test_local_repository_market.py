from __future__ import annotations

from datetime import datetime, timezone

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
        INSERT INTO instrument_master (instrument_id, ticker, ticker_name, exchange_code, currency, updated_at)
        VALUES ('NASD:NVDA', 'NVDA', 'NVIDIA', 'NASD', 'USD', ?)
        """,
        [now],
    )
    out = repo.latest_instrument_map(["NVDA"])
    assert "NVDA" in out
    assert out["NVDA"]["exchange_code"] == "NASD"
    assert out["NVDA"]["currency"] == "USD"
