"""Tests for comma-separated multi-market ("us,kospi") configurations.

Validates that all layers correctly parse, route, and filter when
kis_target_market contains multiple markets.
"""

from __future__ import annotations

import pytest

from arena.cli import _parse_cli_markets, _has_us, _has_kr
from arena.context import ContextBuilder
from arena.forecasting.stacked import _parse_markets
from arena.config import Settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_US_TICKERS = ["AAPL", "MSFT", "TSLA"]
_KR_TICKERS = ["005930", "000660", "069500"]
_MIXED = _US_TICKERS + _KR_TICKERS


def _settings(market: str = "us,kospi") -> Settings:
    return Settings(
        google_cloud_project="p",
        bq_dataset="d",
        bq_location="loc",
        agent_ids=["gpt"],
        agent_mode="adk",
        base_currency="KRW",
        sleeve_capital_krw=2_000_000,
        log_level="INFO",
        log_format="rich",
        trading_mode="paper",
        kis_order_endpoint="",
        kis_api_key="",
        kis_api_secret="",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no="",
        kis_account_product_code="01",
        kis_account_key_suffix="",
        kis_env="real",
        kis_target_market=market,
        kis_overseas_quote_excd="NAS",
        kis_overseas_order_excd="NASD",
        kis_us_natn_cd="840",
        kis_us_tr_mket_cd="01",
        kis_secret_name="KISAPI",
        kis_secret_version="latest",
        kis_http_timeout_seconds=20,
        kis_http_max_retries=0,
        kis_http_backoff_base_seconds=0.1,
        kis_http_backoff_max_seconds=0.1,
        kis_confirm_fills=False,
        kis_confirm_timeout_seconds=25,
        kis_confirm_poll_seconds=2.0,
        usd_krw_rate=1300.0,
        market_sync_history_days=60,
        max_order_krw=350_000,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=0.35,
        min_cash_buffer_ratio=0.10,
        ticker_cooldown_seconds=120,
        max_daily_orders=0,
        estimated_fee_bps=10.0,
        context_max_board_posts=24,
        context_max_memory_events=32,
        context_max_market_rows=64,
        openai_api_key="",
        openai_model="gpt-5.2",
        gemini_api_key="",
        gemini_model="gemini-3-pro-preview",
        research_gemini_model="gemini-2.5-flash",
        llm_timeout_seconds=25,
        default_universe=_MIXED,
        allow_live_trading=False,
    )


# ── Fake collaborators ──

class FakeRepo:
    tenant_id = "test"

    def __init__(self):
        self._rows: list[dict] = []

    def recent_market_features(self, *, limit, source=None):
        return self._rows

    def recent_memory_events(self, *, agent_id, limit, event_type=None):
        return []

    def recent_board_posts(self, *, limit, authors=None):
        return []

    def portfolio_history(self, *, agent_id, limit):
        return []


class FakeMemory:
    def search_similar_memories(self, *, query, **kwargs):
        return []


class FakeBoard:
    def recent_posts(self, *, limit, authors=None):
        return []


# ===========================================================================
# 1. CLI market parsing
# ===========================================================================

class TestCLIMultiMarketParsing:
    def test_parse_combo_us_kospi(self):
        s = _settings("us,kospi")
        markets = _parse_cli_markets(s)
        assert markets == {"us", "kospi"}

    def test_parse_combo_nasdaq_kospi(self):
        s = _settings("nasdaq,kospi")
        markets = _parse_cli_markets(s)
        assert markets == {"nasdaq", "kospi"}

    def test_has_us_true_for_combo(self):
        assert _has_us({"us", "kospi"}) is True

    def test_has_kr_true_for_combo(self):
        assert _has_kr({"us", "kospi"}) is True

    def test_parse_single_us(self):
        s = _settings("nasdaq")
        markets = _parse_cli_markets(s)
        assert markets == {"nasdaq"}
        assert _has_us(markets)
        assert not _has_kr(markets)

    def test_parse_single_kospi(self):
        s = _settings("kospi")
        markets = _parse_cli_markets(s)
        assert markets == {"kospi"}
        assert not _has_us(markets)
        assert _has_kr(markets)

    def test_parse_whitespace_handling(self):
        s = _settings(" nasdaq , kospi ")
        markets = _parse_cli_markets(s)
        assert markets == {"nasdaq", "kospi"}

    def test_parse_triple_market(self):
        s = _settings("nasdaq,nyse,kospi")
        markets = _parse_cli_markets(s)
        assert markets == {"nasdaq", "nyse", "kospi"}
        assert _has_us(markets)
        assert _has_kr(markets)


# ===========================================================================
# 2. ContextBuilder ticker filtering
# ===========================================================================

class TestContextBuilderMultiMarket:
    def test_combo_keeps_all_tickers(self):
        s = _settings("us,kospi")
        builder = ContextBuilder(
            repo=FakeRepo(), memory=FakeMemory(), board=FakeBoard(), settings=s,
        )
        filtered = builder._filter_tickers(_MIXED)
        assert set(filtered) == set(_MIXED)

    def test_us_only_excludes_kr_tickers(self):
        s = _settings("nasdaq")
        builder = ContextBuilder(
            repo=FakeRepo(), memory=FakeMemory(), board=FakeBoard(), settings=s,
        )
        filtered = builder._filter_tickers(_MIXED)
        for t in _KR_TICKERS:
            assert t not in filtered
        for t in _US_TICKERS:
            assert t in filtered

    def test_kospi_only_excludes_us_tickers(self):
        s = _settings("kospi")
        builder = ContextBuilder(
            repo=FakeRepo(), memory=FakeMemory(), board=FakeBoard(), settings=s,
        )
        filtered = builder._filter_tickers(_MIXED)
        for t in _US_TICKERS:
            assert t not in filtered
        for t in _KR_TICKERS:
            assert t in filtered

    def test_combo_market_sources_live(self):
        s = _settings("nasdaq,kospi")
        s.trading_mode = "live"
        builder = ContextBuilder(
            repo=FakeRepo(), memory=FakeMemory(), board=FakeBoard(), settings=s,
        )
        sources = builder._market_sources()
        assert sources is not None
        assert "open_trading_nasdaq" in sources or "open_trading_nasdaq_quote" in sources
        assert "open_trading_kospi" in sources or "open_trading_kospi_quote" in sources

    def test_combo_market_sources_paper_returns_none(self):
        s = _settings("us,kospi")
        builder = ContextBuilder(
            repo=FakeRepo(), memory=FakeMemory(), board=FakeBoard(), settings=s,
        )
        assert builder._market_sources() is None


# ===========================================================================
# 3. Forecasting market parsing
# ===========================================================================

class TestForecastingMultiMarket:
    def test_parse_markets_combo(self):
        s = _settings("nasdaq,kospi")
        markets = _parse_markets(s)
        assert markets == {"nasdaq", "kospi"}

    def test_parse_markets_single(self):
        s = _settings("kospi")
        markets = _parse_markets(s)
        assert markets == {"kospi"}


# ===========================================================================
# 4. Tool layer market detection
# ===========================================================================

class TestToolMultiMarket:
    def test_quant_effective_markets_combo(self):
        from arena.tools.quant_tools import QuantTools
        s = _settings("us,kospi")
        qt = QuantTools(settings=s, repo=FakeRepo())
        markets = qt._effective_markets()
        assert "us" in markets
        assert "kospi" in markets

    def test_quant_has_both(self):
        from arena.tools.quant_tools import QuantTools
        s = _settings("nasdaq,kospi")
        qt = QuantTools(settings=s, repo=FakeRepo())
        assert qt._has_us_market()
        assert qt._has_kospi_market()

    def test_macro_effective_markets_combo(self):
        from arena.tools.macro_tools import MacroTools
        s = _settings("us,kospi")
        mt = MacroTools(settings=s)
        markets = mt._effective_markets()
        assert "us" in markets
        assert "kospi" in markets

    def test_macro_has_both(self):
        from arena.tools.macro_tools import MacroTools
        s = _settings("nasdaq,kospi")
        mt = MacroTools(settings=s)
        assert mt._has_us_market()
        assert mt._has_kospi_market()

    def test_sentiment_effective_markets_combo(self):
        from arena.tools.sentiment_tools import SentimentTools
        s = _settings("us,kospi")
        st = SentimentTools(settings=s)
        markets = st._effective_markets()
        assert "us" in markets
        assert "kospi" in markets
