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

    # Discovery returns AAPL from NAS + benchmarks + representative asset ETFs.
    assert result.attempted_tickers == 9
    assert result.inserted_rows == 54
    assert len(repo.rows) == 54
    assert {"AAPL", "SPY", "QQQ", "DIA", "GLD", "SLV", "USO", "TLT", "UUP"} <= {r["ticker"] for r in repo.rows}
    assert repo.rows[-1]["close_price_krw"] > 0
    aapl_rows = [r for r in repo.rows if r["ticker"] == "AAPL"]
    assert aapl_rows[-1]["close_price_native"] == pytest.approx(120.0)
    assert aapl_rows[-1]["quote_currency"] == "USD"
    assert aapl_rows[-1]["fx_rate_used"] == pytest.approx(1306.0)
    assert aapl_rows[-1]["close_price_krw"] == pytest.approx(120.0 * 1306.0)
    assert repo.latest_instrument_map_calls == [["AAPL", "SPY", "QQQ", "DIA", "GLD", "SLV", "USO", "TLT", "UUP"]]
    rank_meta = repo.rebuild_universe_calls[-1]["universe_rank_metadata"]
    assert rank_meta["AAPL"]["source"] == "market_cap"
    assert rank_meta["AAPL"]["market_cap_rank"] == 1
    assert rank_meta["SPY"]["source"] == "benchmark"
    assert rank_meta["QQQ"]["source"] == "benchmark"
    assert rank_meta["DIA"]["source"] == "benchmark"
    assert rank_meta["GLD"]["source"] == "asset_benchmark"
    assert rank_meta["GLD"]["asset_class"] == "gold"
    assert rank_meta["UUP"]["source"] == "asset_benchmark"
    assert rank_meta["UUP"]["asset_class"] == "usd_currency"


def test_discover_us_symbols_uses_us_specific_cap_override() -> None:
    class WideUsClient(FakeClient):
        def search_overseas_stocks(self, *, excd="NAS", max_pages=1, **kwargs):
            _ = (max_pages, kwargs)
            prefix = "N" if excd == "NAS" else "Y"
            return [
                {"symb": f"{prefix}{idx}", "valx": str(1_000_000 - idx)}
                for idx in range(1, 5)
            ]

    repo = FakeRepo()
    settings = _settings("us", [])
    settings.universe_per_exchange_cap = 500
    settings.us_universe_per_exchange_cap = 2
    service = MarketDataSyncService(settings=settings, repo=repo, client=WideUsClient())

    symbols = service._discover_us_symbols()

    discovered = [row["ticker"] for row in symbols]
    assert discovered[:4] == ["N1", "N2", "Y1", "Y2"]
    assert "N3" not in discovered
    assert "Y3" not in discovered


def test_market_sync_nasdaq_fails_without_live_fx() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())

    result = service.sync_market_features()

    assert result.inserted_rows == 0
    assert "AAPL" in result.failed_tickers
    assert not repo.rows


def test_market_sync_nasdaq_uses_price_detail_fx_when_daily_fx_empty() -> None:
    class EmptyFxClient(FakeClient):
        def get_usd_krw_daily_chart(self, *, symbol, start_date="", end_date="", market_div_code="X", period="D", max_pages=8):
            _ = (symbol, start_date, end_date, market_div_code, period, max_pages)
            return []

        def get_overseas_price_detail(self, ticker, excd):
            _ = (ticker, excd)
            return {"curr": "USD", "p_rate": "1448.5", "t_rate": "1451.25"}

    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    service = MarketDataSyncService(settings=settings, repo=repo, client=EmptyFxClient())

    result = service.sync_market_features()

    assert result.inserted_rows > 0
    assert "AAPL" not in result.failed_tickers
    aapl_rows = [r for r in repo.rows if r["ticker"] == "AAPL"]
    assert aapl_rows[-2]["fx_rate_used"] == pytest.approx(1448.5)
    assert aapl_rows[-1]["fx_rate_used"] == pytest.approx(1451.25)


def test_market_sync_nasdaq_uses_ecos_daily_fx_without_calling_kis_fx(monkeypatch) -> None:
    class EmptyKisFxLongHistoryClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.kis_daily_fx_calls = 0

        def get_usd_krw_daily_chart(self, *, symbol, start_date="", end_date="", market_div_code="X", period="D", max_pages=8):
            _ = (symbol, start_date, end_date, market_div_code, period, max_pages)
            self.kis_daily_fx_calls += 1
            return []

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

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            _ = (params, timeout)
            assert "ecos.bok.or.kr/api/StatisticSearch" in url
            return FakeResponse(
                {
                    "StatisticSearch": {
                        "row": [
                            {"TIME": f"202601{idx:02d}", "DATA_VALUE": str(1300 + idx)}
                            for idx in range(1, 26)
                        ]
                    }
                }
            )

    import arena.open_trading.sync as sync_module

    monkeypatch.setattr(sync_module.requests, "Session", lambda: FakeSession())
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    settings.ecos_api_key = "ecos-key"
    client = EmptyKisFxLongHistoryClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_features()

    assert client.kis_daily_fx_calls == 0
    assert result.inserted_rows == 225
    aapl_rows = [r for r in repo.rows if r["ticker"] == "AAPL"]
    assert len(aapl_rows) == 25
    assert aapl_rows[-1]["fx_rate_used"] == pytest.approx(1325.0)
    assert aapl_rows[-1]["ret_20d"] is not None
    assert aapl_rows[-1]["volatility_20d"] is not None


def test_market_sync_nasdaq_falls_back_to_fred_daily_fx_when_ecos_empty(monkeypatch) -> None:
    class EmptyKisFxLongHistoryClient(FakeClient):
        def get_usd_krw_daily_chart(self, *, symbol, start_date="", end_date="", market_div_code="X", period="D", max_pages=8):
            _ = (symbol, start_date, end_date, market_div_code, period, max_pages)
            return []

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

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            if "ecos.bok.or.kr" in url:
                return FakeResponse({"StatisticSearch": {"row": []}})
            assert "api.stlouisfed.org/fred/series/observations" in url
            assert (params or {}).get("series_id") == "DEXKOUS"
            return FakeResponse(
                {
                    "observations": [
                        {"date": f"2026-01-{idx:02d}", "value": str(1400 + idx)}
                        for idx in range(1, 26)
                    ]
                }
            )

    import arena.open_trading.sync as sync_module

    monkeypatch.setattr(sync_module.requests, "Session", lambda: FakeSession())
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    settings.ecos_api_key = "ecos-key"
    settings.fred_api_key = "fred-key"
    service = MarketDataSyncService(settings=settings, repo=repo, client=EmptyKisFxLongHistoryClient())

    result = service.sync_market_features()

    assert result.inserted_rows == 225
    aapl_rows = [r for r in repo.rows if r["ticker"] == "AAPL"]
    assert len(aapl_rows) == 25
    assert aapl_rows[-1]["fx_rate_used"] == pytest.approx(1425.0)
    assert aapl_rows[-1]["ret_20d"] is not None
    assert aapl_rows[-1]["volatility_20d"] is not None


def test_market_sync_kospi_builds_rows() -> None:
    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    settings.universe_per_exchange_cap = 10
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)
    result = service.sync_market_features()

    assert result.attempted_tickers == 10
    assert result.inserted_rows == 60
    assert {r["ticker"] for r in repo.rows} == {
        "005930",
        "000660",
        "003280",
        "069500",
        "373220",
        "132030",
        "144600",
        "261220",
        "304660",
        "261240",
    }
    assert repo.rows[-1]["source"] == "open_trading_kospi"
    assert repo.rows[-1]["quote_currency"] == "KRW"
    assert repo.rows[-1]["fx_rate_used"] == pytest.approx(1.0)
    assert client.domestic_daily_requests
    rank_meta = repo.rebuild_universe_calls[-1]["universe_rank_metadata"]
    assert rank_meta["005930"]["source"] == "default"
    assert rank_meta["000660"]["source"] == "market_cap"
    assert rank_meta["000660"]["market_cap_rank"] == 2
    assert rank_meta["003280"]["source"] == "volume_rank"
    assert rank_meta["069500"]["source"] == "benchmark"
    assert rank_meta["132030"]["source"] == "asset_benchmark"
    assert rank_meta["132030"]["asset_class"] == "gold"
    assert rank_meta["261240"]["source"] == "asset_benchmark"
    assert rank_meta["261240"]["asset_class"] == "usd_currency"


def test_market_sync_kosdaq_treats_six_digit_holdings_as_domestic() -> None:
    repo = FakeRepo()
    settings = _settings("kosdaq", ["053580"])
    settings.universe_per_exchange_cap = 3
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_features()

    assert result.attempted_tickers >= 1
    assert "053580" in {row["ticker"] for row in repo.rows}
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


def test_discover_kospi_symbols_does_not_use_static_sector_map_to_reach_cap() -> None:
    repo = FakeRepo()
    settings = _settings("kospi", ["005930"])
    settings.universe_per_exchange_cap = 12
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())

    symbols = service._discover_kospi_symbols()

    tickers = [symbol["ticker"] for symbol in symbols]
    assert tickers == [
        "005930",
        "069500",
        "132030",
        "144600",
        "261220",
        "304660",
        "261240",
        "000660",
        "373220",
        "003280",
    ]
    assert all(ticker.isdigit() and len(ticker) == 6 for ticker in tickers)


def test_discover_kospi_symbols_uses_official_master_to_fill_market_cap_cap() -> None:
    class MasterClient(FakeClient):
        def get_domestic_kospi_master_rows(self):
            return [
                {
                    "ticker": f"{100000 + idx:06d}",
                    "name": f"시총{idx}",
                    "market_cap": float(1_000_000 - idx),
                    "volume": float(10_000 + idx),
                }
                for idx in range(1, 20)
            ]

        def get_domestic_market_cap_ranking(self, *, market_scope="0001", div_cls_code="0"):
            _ = (market_scope, div_cls_code)
            return []

        def get_domestic_top_interest_stock(self, *, market_scope="0001"):
            _ = (market_scope,)
            return []

        def get_domestic_volume_rank(self, *, market_scope="0001"):
            _ = (market_scope,)
            return []

    repo = FakeRepo()
    settings = _settings("kospi", [])
    settings.universe_per_exchange_cap = 12
    service = MarketDataSyncService(settings=settings, repo=repo, client=MasterClient())

    symbols = service._discover_kospi_symbols()

    tickers = [symbol["ticker"] for symbol in symbols]
    assert tickers == [
        "069500",
        "132030",
        "144600",
        "261220",
        "304660",
        "261240",
        "100001",
        "100002",
        "100003",
        "100004",
        "100005",
        "100006",
    ]
    rank_meta = service._universe_rank_metadata
    assert rank_meta["100001"]["source"] == "market_cap"
    assert rank_meta["100001"]["market_cap_rank"] == 1
    assert rank_meta["100001"]["market_cap_value"] == pytest.approx(999999.0)
    assert rank_meta["100001"]["volume_value"] == pytest.approx(10001.0)
    assert rank_meta["100006"]["market_cap_rank"] == 6
    assert service._kospi_ticker_names["100001"] == "시총1"


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

    assert result.inserted_rows == 42
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

    assert result.inserted_rows == 54
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

    assert result.attempted_tickers == 10
    assert result.inserted_rows == 250
    miss_rows = [row for row in repo.rows if row["ticker"] == "MISS"]
    assert miss_rows
    assert miss_rows[-1]["ret_5d"] is not None
    assert miss_rows[-1]["ret_20d"] is not None
    assert miss_rows[-1]["volatility_20d"] is not None


def test_market_sync_for_tickers_syncs_account_held_us_and_domestic_only() -> None:
    repo = FakeRepo()
    settings = _settings("us", [])
    settings.usd_krw_fx_symbol = "USDKRW"
    client = FakeClient()
    service = MarketDataSyncService(settings=settings, repo=repo, client=client)

    result = service.sync_market_features_for_tickers(["AAPL", "053580", "AAPL"])

    assert result.attempted_tickers == 2
    assert result.inserted_rows == 12
    assert {row["ticker"] for row in repo.rows} == {"AAPL", "053580"}
    assert client.overseas_daily_requests
    assert client.domestic_daily_requests


def test_quote_sync_us_rows_include_native_price_and_fx() -> None:
    repo = FakeRepo()
    settings = _settings("nasdaq", ["AAPL"])
    settings.usd_krw_fx_symbol = "USDKRW"
    service = MarketDataSyncService(settings=settings, repo=repo, client=FakeClient())

    result = service.sync_market_quotes()

    assert result.inserted_rows == 9
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
    assert sorted(result.failed_tickers) == ["AAPL", "DIA", "GLD", "QQQ", "SLV", "SPY", "TLT", "USO", "UUP"]
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
    assert sorted(result.failed_tickers) == ["AAPL", "DIA", "GLD", "QQQ", "SLV", "SPY", "TLT", "USO", "UUP"]
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
