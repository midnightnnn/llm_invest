from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from arena.tools.sentiment_tools import SentimentTools, _safe_get

_SAMPLE_REDDIT = {
    "data": {
        "children": [
            {
                "data": {
                    "title": "AAPL to the moon!",
                    "score": 142,
                    "num_comments": 37,
                    "created_utc": 1739577600.0,
                    "permalink": "/r/wallstreetbets/comments/abc123/aapl/",
                    "selftext": "Apple is going up big time",
                }
            },
        ]
    }
}

_SAMPLE_TICKERS = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

_SAMPLE_SUBMISSIONS = {
    "name": "Apple Inc.",
    "filings": {
        "recent": {
            "form": ["10-K", "8-K", "10-Q"],
            "filingDate": ["2025-11-01", "2025-10-15", "2025-08-01"],
            "accessionNumber": ["0001-23-456789", "0001-23-000001", "0001-23-000002"],
            "primaryDocument": ["aapl-20251231.htm", "form8k.htm", "aapl-10q.htm"],
            "primaryDocDescription": ["FORM 10-K", "FORM 8-K", "FORM 10-Q"],
        }
    },
}


@pytest.fixture
def st():
    settings = MagicMock()
    settings.kis_target_market = "nasdaq"
    settings.autonomy_tool_default_candidates_enabled = True
    return SentimentTools(settings=settings, http_timeout=5)


class TestFetchReddit:
    @patch("arena.tools.sentiment_tools._safe_get")
    def test_parses_reddit_json(self, mock_get, st: SentimentTools):
        resp = MagicMock()
        resp.json.return_value = _SAMPLE_REDDIT
        mock_get.return_value = resp

        result = st.fetch_reddit_sentiment("AAPL", max_posts=5)
        assert len(result) >= 1
        assert result[0]["title"] == "AAPL to the moon!"
        assert result[0]["score"] == 142
        assert "subreddit" in result[0]

    @patch("arena.tools.sentiment_tools._safe_get", return_value=None)
    def test_returns_empty_on_failure(self, mock_get, st: SentimentTools):
        result = st.fetch_reddit_sentiment("AAPL")
        assert result == []

    def test_empty_ticker(self, st: SentimentTools):
        assert st.fetch_reddit_sentiment("") == []


class TestFetchSec:
    @patch("arena.tools.sentiment_tools._safe_get")
    def test_parses_submissions(self, mock_get, st: SentimentTools):
        resp_tickers = MagicMock()
        resp_tickers.json.return_value = _SAMPLE_TICKERS

        resp_subs = MagicMock()
        resp_subs.json.return_value = _SAMPLE_SUBMISSIONS

        mock_get.side_effect = [resp_tickers, resp_subs]

        result = st.fetch_sec_filings("AAPL", filing_type="10-K")
        assert len(result) == 1
        assert result[0]["form_type"] == "10-K"
        assert result[0]["filed_date"] == "2025-11-01"
        assert result[0]["entity"] == "Apple Inc."
        assert "url" in result[0] and result[0]["url"]

    @patch("arena.tools.sentiment_tools._safe_get", return_value=None)
    def test_returns_error_on_total_failure(self, mock_get, st: SentimentTools):
        result = st.fetch_sec_filings("AAPL")
        assert len(result) == 1
        assert "error" in result[0]

    @patch("arena.tools.sentiment_tools._safe_get")
    def test_defaults_to_opportunity_working_set_ticker(self, mock_get, st: SentimentTools):
        resp_tickers = MagicMock()
        resp_tickers.json.return_value = _SAMPLE_TICKERS

        resp_subs = MagicMock()
        resp_subs.json.return_value = _SAMPLE_SUBMISSIONS

        mock_get.side_effect = [resp_tickers, resp_subs]
        st.set_context({"opportunity_working_set": [{"ticker": "AAPL", "status": "pending"}]})

        result = st.fetch_sec_filings()

        assert len(result) == 1
        assert result[0]["form_type"] == "10-K"

    def test_empty_ticker(self, st: SentimentTools):
        assert st.fetch_sec_filings("") == []


class TestSafeGet:
    @patch("arena.tools.sentiment_tools.requests.get", side_effect=ConnectionError("offline"))
    def test_returns_none_on_error(self, mock_get):
        assert _safe_get("https://example.com") is None
