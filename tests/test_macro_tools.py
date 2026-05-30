from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from arena.tools.macro_tools import MacroTools

_FRED_RESPONSE_DFF = {
    "observations": [
        {"date": "2026-02-21", "value": "4.33"},
        {"date": "2026-02-20", "value": "4.33"},
    ]
}

_FRED_RESPONSE_CPI = {
    "observations": [
        {"date": "2026-01-01", "value": "315.5"},
        {"date": "2025-12-01", "value": "314.8"},
        {"date": "2025-11-01", "value": "314.0"},
        {"date": "2025-10-01", "value": "313.2"},
        {"date": "2025-09-01", "value": "312.5"},
        {"date": "2025-08-01", "value": "311.8"},
        {"date": "2025-07-01", "value": "311.0"},
        {"date": "2025-06-01", "value": "310.3"},
        {"date": "2025-05-01", "value": "309.5"},
        {"date": "2025-04-01", "value": "308.8"},
        {"date": "2025-03-01", "value": "308.0"},
        {"date": "2025-02-01", "value": "307.3"},
        {"date": "2025-01-01", "value": "306.9"},
    ]
}

_FRED_RESPONSE_UNRATE = {
    "observations": [{"date": "2026-01-01", "value": "3.9"}]
}

_FRED_RESPONSE_DGS10 = {
    "observations": [{"date": "2026-02-21", "value": "4.28"}]
}

_FRED_RESPONSE_DGS2 = {
    "observations": [{"date": "2026-02-21", "value": "4.15"}]
}

_FRED_RESPONSE_MANEMP = {
    "observations": [{"date": "2026-01-01", "value": "12800"}]
}

_FRED_RESPONSE_YOY = {
    "observations": [
        {"date": "2026-01-01", "value": "110.0"},
        {"date": "2025-12-01", "value": "109.0"},
        {"date": "2025-11-01", "value": "108.0"},
        {"date": "2025-10-01", "value": "107.0"},
        {"date": "2025-09-01", "value": "106.0"},
        {"date": "2025-08-01", "value": "105.0"},
        {"date": "2025-07-01", "value": "104.0"},
        {"date": "2025-06-01", "value": "103.0"},
        {"date": "2025-05-01", "value": "102.0"},
        {"date": "2025-04-01", "value": "101.0"},
        {"date": "2025-03-01", "value": "100.5"},
        {"date": "2025-02-01", "value": "100.2"},
        {"date": "2025-01-01", "value": "100.0"},
    ]
}

_FRED_EMPTY = {"observations": [{"date": "2026-01-01", "value": "."}]}

_SERIES_RESPONSES = {
    "DFF": _FRED_RESPONSE_DFF,
    "CPIAUCSL": _FRED_RESPONSE_CPI,
    "UNRATE": _FRED_RESPONSE_UNRATE,
    "DGS10": _FRED_RESPONSE_DGS10,
    "DGS2": _FRED_RESPONSE_DGS2,
    "MANEMP": _FRED_RESPONSE_MANEMP,
    "SOFR": {"observations": [{"date": "2026-02-21", "value": "4.31"}]},
    "IORB": {"observations": [{"date": "2026-02-21", "value": "4.40"}]},
    "DGS3MO": {"observations": [{"date": "2026-02-21", "value": "4.42"}]},
    "DGS30": {"observations": [{"date": "2026-02-21", "value": "4.55"}]},
    "CPILFESL": _FRED_RESPONSE_YOY,
    "PCEPI": _FRED_RESPONSE_YOY,
    "PCEPILFE": _FRED_RESPONSE_YOY,
    "T5YIE": {"observations": [{"date": "2026-02-21", "value": "2.25"}]},
    "T10YIE": {"observations": [{"date": "2026-02-21", "value": "2.35"}]},
    "PAYEMS": {"observations": [{"date": "2026-01-01", "value": "158000"}]},
    "ICSA": {"observations": [{"date": "2026-02-14", "value": "220000"}]},
    "CIVPART": {"observations": [{"date": "2026-01-01", "value": "62.5"}]},
    "CES0500000003": _FRED_RESPONSE_YOY,
    "GDPC1": {"observations": [{"date": "2025-10-01", "value": "23200"}]},
    "INDPRO": _FRED_RESPONSE_YOY,
    "RSAFS": _FRED_RESPONSE_YOY,
    "DGORDER": _FRED_RESPONSE_YOY,
    "M2SL": _FRED_RESPONSE_YOY,
    "WALCL": {"observations": [{"date": "2026-02-18", "value": "6750000"}]},
    "RRPONTSYD": {"observations": [{"date": "2026-02-21", "value": "120000"}]},
    "STLFSI4": {"observations": [{"date": "2026-02-14", "value": "-0.25"}]},
    "BAMLH0A0HYM2": {"observations": [{"date": "2026-02-20", "value": "3.45"}]},
    "BAMLC0A0CM": {"observations": [{"date": "2026-02-20", "value": "1.25"}]},
    "SP500": {"observations": [{"date": "2026-02-21", "value": "6100.5"}]},
    "NASDAQCOM": {"observations": [{"date": "2026-02-21", "value": "19000.5"}]},
    "DJIA": {"observations": [{"date": "2026-02-21", "value": "44500.5"}]},
    "VIXCLS": {"observations": [{"date": "2026-02-21", "value": "15.2"}]},
    "DCOILWTICO": {"observations": [{"date": "2026-02-20", "value": "78.4"}]},
    "DTWEXBGS": {"observations": [{"date": "2026-02-20", "value": "122.4"}]},
    "DEXKOUS": {"observations": [{"date": "2026-02-20", "value": "1340.5"}]},
    "HOUST": {"observations": [{"date": "2026-01-01", "value": "1450"}]},
    "PERMIT": {"observations": [{"date": "2026-01-01", "value": "1500"}]},
    "CSUSHPINSA": _FRED_RESPONSE_YOY,
    "MORTGAGE30US": {"observations": [{"date": "2026-02-19", "value": "6.85"}]},
}


class _MacroHistoryRepo:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = []

    def macro_indicator_observation_history(self, **kwargs):
        self.calls.append(dict(kwargs))
        indicator_keys = set(kwargs.get("indicator_keys") or [])
        rows = list(self.rows)
        if indicator_keys:
            rows = [row for row in rows if row.get("indicator_key") in indicator_keys]
        start_date = kwargs.get("start_date")
        if start_date:
            rows = [row for row in rows if row.get("observation_date") >= start_date]
        return rows


def _macro_row(
    key: str,
    value: float,
    obs_date: date,
    *,
    source: str = "ecos",
    group: str = "external",
    unit: str = "index",
    frequency: str = "monthly",
    market: str = "kr",
    series_id: str = "TEST",
    item_code: str = "0",
):
    return {
        "observed_at": "2026-05-30T00:00:00+00:00",
        "as_of_date": obs_date,
        "source": source,
        "indicator_key": key,
        "indicator_name": key.replace("_", " ").title(),
        "group_name": group,
        "market": market,
        "source_series_id": series_id,
        "source_item_code": item_code,
        "frequency": frequency,
        "observation_date": obs_date,
        "value": value,
        "unit": unit,
    }


@pytest.fixture
def mt():
    settings = MagicMock()
    settings.fred_api_key = "test-key-123"
    settings.ecos_api_key = ""
    settings.kis_target_market = "us"
    tool = MacroTools(settings=settings, http_timeout=5)
    tool.set_context({"target_market": "us"})
    return tool


@pytest.fixture
def mt_no_key():
    settings = MagicMock()
    settings.fred_api_key = ""
    settings.ecos_api_key = ""
    settings.kis_target_market = "us"
    tool = MacroTools(settings=settings, http_timeout=5)
    tool.set_context({"target_market": "us"})
    return tool


class TestMacroSnapshot:
    @patch.object(MacroTools, "_fetch_series")
    def test_parses_all_indicators(self, mock_fetch, mt: MacroTools):
        def side_effect(series_id, *, limit=2):
            return _SERIES_RESPONSES.get(series_id, _FRED_EMPTY).get("observations", [])

        mock_fetch.side_effect = side_effect

        result = mt.macro_snapshot()
        assert result["source"] == "fred"
        assert "as_of" in result

        ind = result["indicators"]
        assert ind["fed_funds_rate"]["value"] == 4.33
        assert ind["unemployment_rate"]["value"] == 3.9
        assert ind["treasury_10y"]["value"] == 4.28
        assert ind["treasury_2y"]["value"] == 4.15
        assert ind["yield_spread_10y_2y"]["value"] == 0.13

    @patch.object(MacroTools, "_fetch_series")
    def test_us_macro_snapshot_includes_expanded_fred_candidates(self, mock_fetch, mt: MacroTools):
        def side_effect(series_id, *, limit=2):
            return _SERIES_RESPONSES.get(series_id, _FRED_EMPTY).get("observations", [])

        mock_fetch.side_effect = side_effect

        result = mt.macro_snapshot()

        ind = result["indicators"]
        assert ind["sofr"]["value"] == 4.31
        assert ind["treasury_3m"]["value"] == 4.42
        assert ind["core_cpi_yoy"]["value"] == 10.0
        assert ind["initial_jobless_claims"]["value"] == 220000
        assert ind["real_gdp"]["value"] == 23200
        assert ind["m2_money_supply"]["value"] == 110.0
        assert ind["high_yield_oas"]["value"] == 3.45
        assert ind["vix"]["value"] == 15.2
        assert ind["case_shiller_home_price"]["value"] == 110.0
        assert ind["yield_spread_10y_3m"]["value"] == -0.14
        assert ind["credit_spread_hy_corp"]["value"] == 2.2
        assert result["groups"]["policy_rates"]["sofr"]["series_id"] == "SOFR"
        assert result["groups"]["housing"]["case_shiller_home_price"]["source"] == "fred"
        assert result["coverage"]["fred"]["returned"] >= 35

    def test_no_api_key(self, mt_no_key: MacroTools):
        result = mt_no_key.macro_snapshot()
        # No key → error at top level or us_error in indicators
        assert "error" in result or "us_error" in result.get("indicators", {})

    @patch.object(MacroTools, "_fetch_series", return_value=[])
    def test_handles_api_failure(self, mock_fetch, mt: MacroTools):
        result = mt.macro_snapshot()
        # No data fetched → error or empty indicators
        assert "error" in result or result.get("indicators") == {}

    def test_repo_backed_macro_snapshot_returns_brief_regime_card_with_focus(self):
        rows = [
            _macro_row("usd_krw", 1320.0, date(2025, 1, 1), group="external", unit="KRW per USD", frequency="daily", series_id="731Y003", item_code="0000003"),
            _macro_row("usd_krw", 1360.0, date(2026, 2, 28), group="external", unit="KRW per USD", frequency="daily", series_id="731Y003", item_code="0000003"),
            _macro_row("usd_krw", 1410.0, date(2026, 5, 30), group="external", unit="KRW per USD", frequency="daily", series_id="731Y003", item_code="0000003"),
            _macro_row("kr_current_account", 1200.0, date(2025, 5, 1), group="external", unit="USD 1M", series_id="301Y013"),
            _macro_row("kr_current_account", 2100.0, date(2026, 5, 1), group="external", unit="USD 1M", series_id="301Y013"),
            _macro_row("kr_treasury_3y", 3.9, date(2026, 2, 28), group="rates_curve", unit="%", frequency="daily", series_id="817Y002"),
            _macro_row("kr_treasury_3y", 3.2, date(2026, 5, 30), group="rates_curve", unit="%", frequency="daily", series_id="817Y002"),
        ]
        settings = MagicMock()
        settings.fred_api_key = ""
        settings.ecos_api_key = "ecos-key"
        settings.kis_target_market = "kospi"
        repo = _MacroHistoryRepo(rows)
        tool = MacroTools(settings=settings, repo=repo, http_timeout=5)
        tool.set_context({"target_market": "kospi"})

        result = tool.macro_snapshot(depth="brief", focus=["fx_external"], max_indicators=2)

        assert result["depth"] == "brief"
        assert result["data_mode"] == "historical"
        assert result["coverage"]["ecos"]["returned"] == 2
        assert result["regime_card"]["fx_external"] == "pressure_high"
        assert result["market_implications"]["usd_exposure"] == "positive_but_expensive"
        assert list(result["groups"]) == ["external"]
        assert "usd_krw" in result["indicators"]
        usd = result["groups"]["external"]["evidence"][0]
        assert usd["k"] == "usd_krw"
        assert usd["v"] == 1410.0
        assert usd["chg_3m"] == 50.0
        assert usd["z"] > 0
        assert "series" not in usd
        assert repo.calls and repo.calls[0]["lookback_days"] == 540

    def test_repo_backed_macro_snapshot_full_drilldown_includes_series_for_requested_indicator(self):
        rows = [
            _macro_row("usd_krw", 1320.0, date(2026, 1, 1), group="external", unit="KRW per USD", frequency="daily", series_id="731Y003", item_code="0000003"),
            _macro_row("usd_krw", 1360.0, date(2026, 2, 1), group="external", unit="KRW per USD", frequency="daily", series_id="731Y003", item_code="0000003"),
            _macro_row("usd_krw", 1410.0, date(2026, 3, 1), group="external", unit="KRW per USD", frequency="daily", series_id="731Y003", item_code="0000003"),
            _macro_row("kr_cpi", 118.0, date(2026, 3, 1), group="inflation", unit="index", series_id="901Y009"),
        ]
        settings = MagicMock()
        settings.fred_api_key = ""
        settings.ecos_api_key = "ecos-key"
        settings.kis_target_market = "kospi"
        repo = _MacroHistoryRepo(rows)
        tool = MacroTools(settings=settings, repo=repo, http_timeout=5)
        tool.set_context({"target_market": "kospi"})

        result = tool.macro_snapshot(
            depth="full",
            indicators=["usd_krw"],
            include_series=True,
            max_points=2,
        )

        assert result["depth"] == "full"
        assert list(result["indicators"]) == ["usd_krw"]
        usd = result["groups"]["external"]["evidence"][0]
        assert usd["series"] == [
            {"d": "2026-02-01", "v": 1360.0},
            {"d": "2026-03-01", "v": 1410.0},
        ]
        assert result["omitted"]["indicator_count"] == 1


class TestCpiYoY:
    @patch.object(MacroTools, "_fetch_series")
    def test_computes_yoy(self, mock_fetch, mt: MacroTools):
        mock_fetch.return_value = _FRED_RESPONSE_CPI["observations"]
        _, yoy = mt._compute_cpi_yoy()
        assert yoy is not None
        assert abs(yoy - 2.80) < 0.2


class TestEcosSnapshot:
    def test_kr_macro_snapshot_includes_expanded_key_statistics(self):
        settings = MagicMock()
        settings.fred_api_key = ""
        settings.ecos_api_key = "ecos-key-123"
        settings.kis_target_market = "kospi"
        tool = MacroTools(settings=settings, http_timeout=5)
        tool.set_context({"target_market": "kospi"})
        rows = [
                {
                    "CLASS_NAME": "시장금리",
                    "KEYSTAT_NAME": "국고채수익률(3년)",
                    "DATA_VALUE": "3.731",
                    "CYCLE": "20260529",
                    "UNIT_NAME": "%",
                },
                {
                    "CLASS_NAME": "시장금리",
                    "KEYSTAT_NAME": "국고채수익률(5년)",
                    "DATA_VALUE": "3.924",
                    "CYCLE": "20260529",
                    "UNIT_NAME": "%",
                },
                {
                    "CLASS_NAME": "시장금리",
                    "KEYSTAT_NAME": "회사채수익률(3년,AA-)",
                    "DATA_VALUE": "4.353",
                    "CYCLE": "20260529",
                    "UNIT_NAME": "%",
                },
                {
                    "CLASS_NAME": "여수신금리",
                    "KEYSTAT_NAME": "예금은행 수신금리",
                    "DATA_VALUE": "2.92",
                    "CYCLE": "202604",
                    "UNIT_NAME": "%",
                },
                {
                    "CLASS_NAME": "여수신금리",
                    "KEYSTAT_NAME": "예금은행 대출금리",
                    "DATA_VALUE": "4.20",
                    "CYCLE": "202604",
                    "UNIT_NAME": "%",
                },
                {
                    "CLASS_NAME": "통화량",
                    "KEYSTAT_NAME": "M2(광의통화, 평잔)",
                    "DATA_VALUE": "4143515.8",
                    "CYCLE": "202603",
                    "UNIT_NAME": "십억원",
                },
                {
                    "CLASS_NAME": "예금/대출금",
                    "KEYSTAT_NAME": "가계신용",
                    "DATA_VALUE": "1993110.8",
                    "CYCLE": "2026Q1",
                    "UNIT_NAME": "십억원",
                },
                {
                    "CLASS_NAME": "환율",
                    "KEYSTAT_NAME": "원/엔(100엔) 환율(매매기준율)",
                    "DATA_VALUE": "945.56",
                    "CYCLE": "20260529",
                    "UNIT_NAME": "원",
                },
                {
                    "CLASS_NAME": "생산",
                    "KEYSTAT_NAME": "전산업생산지수",
                    "DATA_VALUE": "117.8",
                    "CYCLE": "202604",
                    "UNIT_NAME": "2020=100",
                },
                {
                    "CLASS_NAME": "심리지표",
                    "KEYSTAT_NAME": "소비자심리지수",
                    "DATA_VALUE": "106.1",
                    "CYCLE": "202605",
                    "UNIT_NAME": "",
                },
                {
                    "CLASS_NAME": "대외채권/채무",
                    "KEYSTAT_NAME": "외환보유액",
                    "DATA_VALUE": "427875470",
                    "CYCLE": "202604",
                    "UNIT_NAME": "천달러",
                },
                {
                    "CLASS_NAME": "소비자/생산자 물가",
                    "KEYSTAT_NAME": "농산물 및 석유류제외 소비자물가지수",
                    "DATA_VALUE": "117.38",
                    "CYCLE": "202604",
                    "UNIT_NAME": "2020=100",
                },
                {
                    "CLASS_NAME": "부동산 가격",
                    "KEYSTAT_NAME": "주택매매가격지수",
                    "DATA_VALUE": "101.4",
                    "CYCLE": "202601",
                    "UNIT_NAME": "2025.03=100",
                },
                {
                    "CLASS_NAME": "국제원자재가격",
                    "KEYSTAT_NAME": "Dubai유(현물)",
                    "DATA_VALUE": "105.3",
                    "CYCLE": "202604",
                    "UNIT_NAME": "달러/배럴",
                },
            ]

        with patch.object(MacroTools, "_fetch_ecos_key_stats", return_value=rows):
            result = tool.macro_snapshot()

        ind = result["indicators"]
        assert ind["kr_m2_money_supply"]["value"] == 4143515.8
        assert ind["kr_household_credit"]["value"] == 1993110.8
        assert ind["jpy_krw"]["value"] == 945.56
        assert ind["kr_all_industry_production"]["value"] == 117.8
        assert ind["kr_consumer_sentiment_index"]["value"] == 106.1
        assert ind["kr_fx_reserves"]["value"] == 427875470
        assert ind["kr_core_cpi_ex_food_energy"]["value"] == 117.38
        assert ind["kr_house_price_index"]["value"] == 101.4
        assert ind["dubai_oil"]["value"] == 105.3
        assert ind["kr_yield_spread_5y_3y"]["value"] == 0.193
        assert ind["kr_credit_spread_corp_aa_3y"]["value"] == 0.622
        assert ind["kr_bank_loan_deposit_spread"]["value"] == 1.28
        assert result["groups"]["money"]["kr_m2_money_supply"]["source"] == "ecos"
        assert result["groups"]["housing"]["kr_house_price_index"]["name"] == "주택매매가격지수"
        assert result["coverage"]["ecos"]["returned"] >= 14


class TestParseValue:
    def test_normal(self):
        assert MacroTools._parse_value({"date": "2026-01-01", "value": "4.33"}) == ("2026-01-01", 4.33)

    def test_missing(self):
        assert MacroTools._parse_value({"date": "2026-01-01", "value": "."}) == ("2026-01-01", None)

    def test_empty(self):
        assert MacroTools._parse_value({"date": "", "value": ""}) == ("", None)
