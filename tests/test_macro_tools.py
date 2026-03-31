from __future__ import annotations

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

_FRED_EMPTY = {"observations": [{"date": "2026-01-01", "value": "."}]}

_SERIES_RESPONSES = {
    "DFF": _FRED_RESPONSE_DFF,
    "CPIAUCSL": _FRED_RESPONSE_CPI,
    "UNRATE": _FRED_RESPONSE_UNRATE,
    "DGS10": _FRED_RESPONSE_DGS10,
    "DGS2": _FRED_RESPONSE_DGS2,
    "MANEMP": _FRED_RESPONSE_MANEMP,
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

    def test_no_api_key(self, mt_no_key: MacroTools):
        result = mt_no_key.macro_snapshot()
        # No key → error at top level or us_error in indicators
        assert "error" in result or "us_error" in result.get("indicators", {})

    @patch.object(MacroTools, "_fetch_series", return_value=[])
    def test_handles_api_failure(self, mock_fetch, mt: MacroTools):
        result = mt.macro_snapshot()
        # No data fetched → error or empty indicators
        assert "error" in result or result.get("indicators") == {}


class TestCpiYoY:
    @patch.object(MacroTools, "_fetch_series")
    def test_computes_yoy(self, mock_fetch, mt: MacroTools):
        mock_fetch.return_value = _FRED_RESPONSE_CPI["observations"]
        _, yoy = mt._compute_cpi_yoy()
        assert yoy is not None
        assert abs(yoy - 2.80) < 0.2


class TestParseValue:
    def test_normal(self):
        assert MacroTools._parse_value({"date": "2026-01-01", "value": "4.33"}) == ("2026-01-01", 4.33)

    def test_missing(self):
        assert MacroTools._parse_value({"date": "2026-01-01", "value": "."}) == ("2026-01-01", None)

    def test_empty(self):
        assert MacroTools._parse_value({"date": "", "value": ""}) == ("", None)
