from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _Session:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> _Response:
        _ = timeout
        self.calls.append((url, dict(params or {})))
        if "fred/series/observations" in url:
            series_id = str((params or {}).get("series_id") or "")
            if series_id == "DGS10":
                return _Response(
                    {
                        "observations": [
                            {"date": "2026-02-28", "value": "3.90"},
                            {"date": "2026-03-01", "value": "4.00"},
                            {"date": "2026-03-02", "value": "."},
                            {"date": "2026-03-03", "value": "4.10"},
                        ]
                    }
                )
            return _Response({"observations": []})
        if "/731Y003/D/20260301/20260303/0000003" in url:
            return _Response(
                {
                    "StatisticSearch": {
                        "row": [
                            {
                                "STAT_CODE": "731Y003",
                                "ITEM_CODE1": "0000003",
                                "ITEM_NAME1": "원/달러(종가 15:30)",
                                "TIME": "20260303",
                                "DATA_VALUE": "1381.25",
                            }
                        ]
                    }
                }
            )
        if "/731Y001/" in url or "/731Y003/" in url:
            return _Response({})
        if "/901Y009/M/202603/202603/0" in url:
            return _Response(
                {
                    "StatisticSearch": {
                        "row": [
                            {
                                "STAT_CODE": "901Y009",
                                "ITEM_CODE1": "0",
                                "ITEM_NAME1": "총지수",
                                "TIME": "202603",
                                "DATA_VALUE": "118.8",
                            },
                            {
                                "STAT_CODE": "901Y009",
                                "ITEM_CODE1": "0",
                                "ITEM_NAME1": "총지수",
                                "TIME": "202603",
                                "DATA_VALUE": "118.8",
                            }
                        ]
                    }
                }
            )
        if "/200Y102/Q/2026Q1/2026Q1/10111" in url:
            return _Response(
                {
                    "StatisticSearch": {
                        "row": [
                            {
                                "STAT_CODE": "200Y102",
                                "ITEM_CODE1": "10111",
                                "ITEM_NAME1": "국내총생산(GDP)(실질, 계절조정, 전기비)",
                                "TIME": "2026Q1",
                                "DATA_VALUE": "1.694",
                            }
                        ]
                    }
                }
            )
        if "/200Y102/" in url:
            return _Response({})
        if "/512Y013/M/202603/202603/99988/AA" in url:
            return _Response(
                {
                    "StatisticSearch": {
                        "row": [
                            {
                                "STAT_CODE": "512Y013",
                                "ITEM_CODE1": "99988",
                                "ITEM_NAME1": "전 산 업",
                                "ITEM_CODE2": "AA",
                                "ITEM_NAME2": "업황실적BSI 1)",
                                "TIME": "202603",
                                "DATA_VALUE": "72",
                            }
                        ]
                    }
                }
            )
        if "/512Y013/" in url:
            return _Response({})
        return _Response({})


class _Repo:
    def __init__(self) -> None:
        self.inserted: list[dict[str, Any]] = []
        self.deleted: list[tuple[date, date, list[str]]] = []

    def earliest_market_feature_date(self) -> date:
        return date(2026, 3, 1)

    def delete_macro_indicator_observations(self, start_date: date, end_date: date, sources: list[str]) -> None:
        self.deleted.append((start_date, end_date, sources))

    def insert_macro_indicator_observations(self, rows: list[dict[str, Any]]) -> int:
        self.inserted.extend(rows)
        return len(rows)


def test_macro_backfill_uses_market_feature_start_and_writes_fred_and_ecos_rows() -> None:
    from arena.macro_backfill import MacroBackfillService

    settings = SimpleNamespace(fred_api_key="fred-key", ecos_api_key="ecos-key")
    repo = _Repo()
    session = _Session()
    service = MacroBackfillService(
        settings=settings,
        repo=repo,
        session=session,
        now=lambda: datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc),
    )

    result = service.backfill(end_date=date(2026, 3, 3))

    assert result.start_date == date(2026, 3, 1)
    assert result.end_date == date(2026, 3, 3)
    assert result.inserted == 6
    assert result.source_counts == {"fred": 2, "ecos": 4}
    assert repo.deleted == [(date(2026, 1, 1), date(2026, 3, 3), ["ecos", "fred"])]

    fred_call = next(params for url, params in session.calls if params and params.get("series_id") == "DGS10")
    assert fred_call["observation_start"] == "2026-03-01"
    assert fred_call["observation_end"] == "2026-03-03"

    keys = [(row["source"], row["indicator_key"], row["observation_date"], row["value"]) for row in repo.inserted]
    assert ("fred", "treasury_10y", date(2026, 2, 28), 3.9) not in keys
    assert ("fred", "treasury_10y", date(2026, 3, 1), 4.0) in keys
    assert ("fred", "treasury_10y", date(2026, 3, 3), 4.1) in keys
    assert ("ecos", "usd_krw", date(2026, 3, 3), 1381.25) in keys
    assert ("ecos", "kr_cpi", date(2026, 3, 1), 118.8) in keys
    assert ("ecos", "kr_gdp_growth", date(2026, 1, 1), 1.694) in keys
    assert ("ecos", "kr_all_industry_bsi_actual", date(2026, 3, 1), 72.0) in keys
    ecos_row = next(row for row in repo.inserted if row["source"] == "ecos")
    assert ecos_row["market"] == "kr"
    assert any("/512Y013/M/202603/202603/99988/AA" in url for url, _params in session.calls)


def test_macro_backfill_dry_run_does_not_insert_rows() -> None:
    from arena.macro_backfill import MacroBackfillService

    settings = SimpleNamespace(fred_api_key="fred-key", ecos_api_key="")
    repo = _Repo()
    session = _Session()
    service = MacroBackfillService(
        settings=settings,
        repo=repo,
        session=session,
        now=lambda: datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc),
    )

    result = service.backfill(start_date=date(2026, 3, 1), end_date=date(2026, 3, 3), dry_run=True)

    assert result.inserted == 0
    assert result.discovered == 2
    assert repo.deleted == []
    assert repo.inserted == []


def test_ecos_historical_specs_cover_major_key_stat_groups() -> None:
    from arena.macro_backfill import ECOS_HISTORICAL_SPECS

    specs = {spec.key: spec for spec in ECOS_HISTORICAL_SPECS}

    assert len(specs) >= 45
    assert len(specs) == len(ECOS_HISTORICAL_SPECS)
    for key in [
        "bok_base_rate",
        "call_rate",
        "kr_treasury_3y",
        "usd_krw",
        "jpy_krw",
        "kospi_index",
        "kr_cpi",
        "kr_gdp_growth",
        "kr_consumer_sentiment_index",
        "kr_current_account",
        "kr_house_price_index",
    ]:
        assert key in specs


def test_macro_backfill_incremental_replaces_recent_window() -> None:
    from arena.macro_backfill import MacroBackfillService

    class _IncrementalRepo(_Repo):
        def latest_macro_indicator_observation_date(self, sources=None):
            assert sources == ["ecos", "fred"]
            return date(2026, 3, 2)

    settings = SimpleNamespace(fred_api_key="fred-key", ecos_api_key="")
    repo = _IncrementalRepo()
    session = _Session()
    service = MacroBackfillService(
        settings=settings,
        repo=repo,
        session=session,
        now=lambda: datetime(2026, 3, 4, 0, 0, tzinfo=timezone.utc),
    )

    result = service.refresh_incremental(end_date=date(2026, 3, 4), replace_days=1)

    assert result.start_date == date(2026, 3, 1)
    assert result.end_date == date(2026, 3, 4)
    assert repo.deleted == [(date(2026, 3, 1), date(2026, 3, 4), ["fred"])]
