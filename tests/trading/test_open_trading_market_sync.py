from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.config import Settings
from arena.open_trading.sync import AccountSyncService, BrokerCashSyncService, BrokerTradeSyncService, MarketDataSyncService

from tests.trading.open_trading_sync_helpers import (
    FakeRepo,
    FakeClient,
    FakeBrokerTradeRepo,
    FakeBrokerCashRepo,
    FakeBrokerTradeClient,
    _settings,
)

def test_market_sync_nasdaq_builds_rows() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())
    result = service.sync_market_features()

    # Discovery returns AAPL from NAS + benchmarks SPY/QQQ/DIA = 4 tickers
    assert result.attempted_tickers == 4
    assert result.inserted_rows == 24
    assert len(repo.rows) == 24
    assert {"AAPL", "SPY", "QQQ", "DIA"} <= {r["ticker"] for r in repo.rows}
    assert repo.rows[-1]["close_price_krw"] > 0
    aapl_rows = [r for r in repo.rows if r["ticker"] == "AAPL"]
    assert aapl_rows[-1]["close_price_native"] == pytest.approx(120.0)
    assert aapl_rows[-1]["quote_currency"] == "USD"
    assert aapl_rows[-1]["fx_rate_used"] == pytest.approx(1306.0)
    assert aapl_rows[-1]["close_price_krw"] == pytest.approx(120.0 * 1306.0)
    assert repo.latest_instrument_map_calls == [["AAPL", "SPY", "QQQ", "DIA"]]


def test_market_sync_nasdaq_fails_without_live_fx() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())

    result = service.sync_market_features()

    assert result.inserted_rows == 0
    assert "AAPL" in result.failed_tickers
    assert not repo.rows


def test_market_sync_kospi_builds_rows() -> None:
    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    settings.universe_per_exchange_cap = 5
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)
    result = service.sync_market_features()

    assert result.attempted_tickers == 5
    assert result.inserted_rows == 30
    assert {r["ticker"] for r in repo.rows} == {"005930", "000660", "003280", "069500", "373220"}
    assert repo.rows[-1]["source"] == "open_trading_kospi"
    assert repo.rows[-1]["quote_currency"] == "KRW"
    assert repo.rows[-1]["fx_rate_used"] == pytest.approx(1.0)
    assert client.domestic_daily_requests


def test_discover_kospi_symbols_backfills_name_for_already_seen_ticker() -> None:
    class NamedClient(FakeClient):
        def get_domestic_market_cap_ranking(self, *, market_scope="0001", div_cls_code="0"):
            _ = (market_scope, div_cls_code)
            return [
                {"mksc_shrn_iscd": "005930", "hts_kor_isnm": "삼성전자"},
                {"mksc_shrn_iscd": "000660", "hts_kor_isnm": "SK하이닉스"},
            ]

    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    service = MarketDataSyncService(settings=settings, repo=repo, client=NamedClient())

    symbols = service._discover_kospi_symbols()

    assert {"ticker": "005930", "quote_excd": "KRX"} in symbols
    assert service._kospi_ticker_names["005930"] == "삼성전자"


def test_discover_kospi_symbols_uses_static_sector_map_to_reach_cap() -> None:
    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    settings.universe_per_exchange_cap = 8
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())

    symbols = service._discover_kospi_symbols()

    tickers = [symbol["ticker"] for symbol in symbols]
    assert len(tickers) == 8
    assert {"005930", "000660", "003280", "069500", "373220"} <= set(tickers)
    assert all(ticker.isdigit() and len(ticker) == 6 for ticker in tickers)


def test_market_sync_kospi_requests_long_history_for_forecast_bootstrap() -> None:
    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    settings.universe_per_exchange_cap = 5
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    service.sync_market_features()

    assert client.domestic_daily_requests
    req = client.domestic_daily_requests[0]
    start = datetime.strptime(req["start_date"], "%Y%m%d")
    end = datetime.strptime(req["end_date"], "%Y%m%d")
    assert (end - start).days >= 360


def test_market_sync_kospi_forces_backfill_when_existing_history_is_too_shallow() -> None:
    repo = FakeRepo()
    repo._latest_dates = {"005930": datetime.strptime("20260306", "%Y%m%d").date()}
    repo._spans = {
        "005930": {
            "min_d": datetime.strptime("20251002", "%Y%m%d").date(),
            "max_d": datetime.strptime("20260306", "%Y%m%d").date(),
            "row_count": 100,
        }
    }
    settings = _settings("kospi", ["005930"])
    settings.universe_per_exchange_cap = 5
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_features()

    assert result.inserted_rows == 30
    assert client.domestic_daily_requests
    assert len([r for r in repo.rows if r["ticker"] == "005930"]) == 6


def test_market_sync_us_forces_backfill_when_existing_history_is_too_shallow() -> None:
    repo = FakeRepo()
    repo._latest_dates = {"AAPL": datetime.strptime("20260106", "%Y%m%d").date()}
    repo._spans = {
        "AAPL": {
            "min_d": datetime.strptime("20260106", "%Y%m%d").date(),
            "max_d": datetime.strptime("20260106", "%Y%m%d").date(),
            "row_count": 1,
        }
    }
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_features()

    assert result.inserted_rows == 24
    assert client.overseas_daily_requests
    assert len([r for r in repo.rows if r["ticker"] == "AAPL"]) == 6


def test_market_sync_us_includes_existing_tickers_missing_daily_features() -> None:
    class RepoWithMissingFeatureTicker(FakeRepo):
        def latest_missing_daily_feature_tickers(self, *, sources=None, limit=1000):
            _ = (sources, limit)
            return [
                {
                    "ticker": "MISS",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:MISS",
                    "source": "open_trading_us_quote",
                }
            ]

    class LongHistoryClient(FakeClient):
        def get_usd_krw_daily_chart(
            self,
            *,
            symbol,
            start_date="",
            end_date="",
            market_div_code="X",
            period="D",
            max_pages=8,
        ):
            _ = (symbol, start_date, end_date, market_div_code, period, max_pages)
            return [
                {"stck_bsop_date": f"202601{idx:02d}", "ovrs_nmix_prpr": str(1300 + idx)}
                for idx in range(1, 26)
            ]

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
                {"xymd": f"202601{idx:02d}", "clos": str(90 + idx)}
                for idx in range(1, 26)
            ]

    repo = RepoWithMissingFeatureTicker()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    client = LongHistoryClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_features()

    assert result.attempted_tickers == 5
    assert result.inserted_rows == 125
    miss_rows = [row for row in repo.rows if row["ticker"] == "MISS"]
    assert miss_rows
    assert miss_rows[-1]["ret_5d"] is not None
    assert miss_rows[-1]["ret_20d"] is not None
    assert miss_rows[-1]["volatility_20d"] is not None


def test_quote_sync_us_rows_include_native_price_and_fx() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())

    result = service.sync_market_quotes()

    assert result.inserted_rows == 4
    aapl_rows = [r for r in repo.rows if r["ticker"] == "AAPL"]
    assert aapl_rows
    assert aapl_rows[-1]["close_price_native"] == pytest.approx(100.0)
    assert aapl_rows[-1]["fx_rate_used"] == pytest.approx(1311.0)
    assert aapl_rows[-1]["close_price_krw"] == pytest.approx(131100.0)
    assert aapl_rows[-1]["ret_20d"] == pytest.approx(0.04)
    assert aapl_rows[-1]["volatility_20d"] == pytest.approx(0.12)


def test_quote_sync_us_skips_quote_rows_when_daily_features_are_missing() -> None:
    class RepoWithoutDailyBase(FakeRepo):
        def latest_market_features(self, *, tickers, limit, sources=None):
            self.latest_market_features_calls.append(
                {
                    "tickers": list(tickers),
                    "limit": limit,
                    "sources": list(sources or []),
                }
            )
            return []

    repo = RepoWithoutDailyBase()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_quotes()

    assert result.inserted_rows == 0
    assert sorted(result.failed_tickers) == ["AAPL", "DIA", "QQQ", "SPY"]
    assert client.overseas_daily_requests == []
    assert repo.rows == []


def test_quote_sync_us_skips_quote_rows_when_daily_features_are_stale() -> None:
    class RepoWithStaleDailyBase(FakeRepo):
        def latest_market_features(self, *, tickers, limit, sources=None):
            rows = super().latest_market_features(tickers=tickers, limit=limit, sources=sources)
            for row in rows:
                row["as_of_ts"] = datetime(2026, 1, 1, tzinfo=timezone.utc)
            return rows

    repo = RepoWithStaleDailyBase()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_quotes()

    assert result.inserted_rows == 0
    assert sorted(result.failed_tickers) == ["AAPL", "DIA", "QQQ", "SPY"]
    assert repo.rows == []


def test_quote_sync_us_held_ticker_probes_exchange() -> None:
    class RepoWithHoldings(FakeRepo):
        def get_all_held_tickers(self):
            return ["KO"]

    class ProbeClient(FakeClient):
        def search_overseas_stocks(self, *, excd="NAS", max_pages=1, **kwargs):
            _ = (excd, max_pages, kwargs)
            return []

        def get_overseas_price(self, ticker, excd):
            if ticker == "KO" and excd == "NYS":
                return {"last": "77.34", "rate": "0.2", "rsym": "DNYSKO"}
            if ticker in {"SPY", "QQQ", "DIA"} and excd == "NAS":
                return {"last": "100", "rate": "1.2", "rsym": f"DNAS{ticker}"}
            return {"last": "", "rate": "", "rsym": ""}

    repo = RepoWithHoldings()
    settings = _settings("us", ["KO"])
    settings.usd_krw_fx_symbol = "USDKRW"
    service = MarketDataSyncService(settings=settings, repo=repo, client=ProbeClient())

    result = service.sync_market_quotes()

    assert result.inserted_rows == 4
    ko_rows = [row for row in repo.rows if row["ticker"] == "KO"]
    assert ko_rows
    assert ko_rows[-1]["exchange_code"] == "NYSE"
    assert ko_rows[-1]["instrument_id"] == "NYSE:KO"
