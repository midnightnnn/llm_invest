from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from arena.tools.quant_tools import QuantTools


_KOSPI_ROWS = [
    {"bstp_nmix_prca": "2650.30", "xymd": "20260220"},
    {"bstp_nmix_prca": "2640.10", "xymd": "20260219"},
    {"bstp_nmix_prca": "2630.00", "xymd": "20260218"},
    {"bstp_nmix_prca": "2620.50", "xymd": "20260217"},
    {"bstp_nmix_prca": "2610.00", "xymd": "20260214"},
    {"bstp_nmix_prca": "2600.00", "xymd": "20260213"},
]


@pytest.fixture
def qt():
    settings = MagicMock()
    settings.kis_target_market = "nasdaq"
    settings.trading_mode = "paper"
    settings.default_universe = ["AAPL", "MSFT"]
    settings.fred_api_key = "test_key"
    settings.kis_http_timeout_seconds = 10
    repo = MagicMock()
    return QuantTools(repo=repo, settings=settings)


@pytest.fixture
def qt_kospi():
    settings = MagicMock()
    settings.kis_target_market = "kospi"
    settings.trading_mode = "paper"
    settings.default_universe = ["005930", "000660"]
    repo = MagicMock()
    return QuantTools(repo=repo, settings=settings)


class TestIndexSnapshot:
    @patch("arena.tools.quant_tools.QuantTools._fetch_fred_latest")
    def test_us_indices_via_fred(self, mock_fred, qt: QuantTools):
        """US indices (SPX, COMP, DJI) are fetched from FRED."""
        mock_fred.return_value = ("2026-03-14", 6100.50)

        result = qt.index_snapshot(indices=["SPX"])
        assert len(result["indices"]) == 1
        spx = result["indices"][0]
        assert spx["symbol"] == "SPX"
        assert spx["value"] == 6100.50
        assert spx["type"] == "index"
        mock_fred.assert_called_once_with("SP500")

    @patch("arena.tools.quant_tools.QuantTools._fetch_fred_latest")
    def test_fred_unavailable(self, mock_fred, qt: QuantTools):
        """FRED returning None → error entry."""
        mock_fred.return_value = ("", None)

        result = qt.index_snapshot(indices=["SPX"])
        assert result["indices"] == []
        assert any(e["symbol"] == "SPX" for e in (result["errors"] or []))

    @patch("arena.open_trading.client.OpenTradingClient")
    def test_kospi_via_kis(self, MockClient, qt_kospi: QuantTools):
        """Korean indices still use KIS domestic API."""
        mock_client = MagicMock()
        mock_client.get_domestic_index_daily.return_value = _KOSPI_ROWS
        qt_kospi.ot_client = mock_client

        result = qt_kospi.index_snapshot(indices=["KOSPI"])
        assert len(result["indices"]) == 1
        kospi = result["indices"][0]
        assert kospi["symbol"] == "KOSPI"
        assert kospi["close"] == 2650.30
        assert "change_1d" in kospi

    @patch("arena.open_trading.client.OpenTradingClient")
    def test_kospi_empty_response(self, MockClient, qt_kospi: QuantTools):
        mock_client = MagicMock()
        mock_client.get_domestic_index_daily.return_value = []
        qt_kospi.ot_client = mock_client

        result = qt_kospi.index_snapshot(indices=["KOSPI"])
        assert result["indices"] == []
        assert result["errors"] is not None

    @patch("arena.open_trading.client.OpenTradingClient")
    def test_kospi_api_error(self, MockClient, qt_kospi: QuantTools):
        mock_client = MagicMock()
        mock_client.get_domestic_index_daily.side_effect = Exception("timeout")
        qt_kospi.ot_client = mock_client

        result = qt_kospi.index_snapshot(indices=["KOSPI"])
        assert result["indices"] == []
        assert len(result["errors"]) == 1

    def test_rejects_stock_tickers(self, qt: QuantTools):
        """Individual stock tickers must be rejected — only index symbols allowed."""
        result = qt.index_snapshot(indices=["AAPL", "TSLA", "MSFT"])
        assert result["indices"] == []
        assert result["source"] == "none"
        assert any("Invalid" in e["error"] for e in result["errors"])

    @patch("arena.tools.quant_tools.QuantTools._fetch_fred_latest")
    def test_mixed_valid_invalid(self, mock_fred, qt: QuantTools):
        """Valid indices are kept; invalid tickers are silently dropped."""
        mock_fred.return_value = ("2026-03-14", 6100.50)

        result = qt.index_snapshot(indices=["SPX", "AAPL", "NVDA"])
        assert len(result["indices"]) == 1
        assert result["indices"][0]["symbol"] == "SPX"
