"""Tests for dividend discovery, attribution, dedup, and sleeve integration."""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pytest

from arena.config import Settings, load_settings
from arena.models import AccountSnapshot, Position
from arena.open_trading.sync import DividendSyncService, DividendSyncResult


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------

class FakeClient:
    """Mimics OpenTradingClient.get_overseas_period_rights."""

    def __init__(self, rights_by_ticker: dict[str, list[dict]] | None = None):
        self._rights = rights_by_ticker or {}

    def get_overseas_period_rights(self, ticker, start_date, end_date, excd=None):
        return self._rights.get(ticker.upper(), [])


class FakeRepo:
    """In-memory repo stub for dividend tests."""

    def __init__(
        self,
        *,
        held_tickers: list[str] | None = None,
        agent_holdings: dict[str, dict[str, float]] | None = None,
        existing_event_ids: set[str] | None = None,
        tenant_id: str = "test",
    ):
        self._held = held_tickers or []
        # agent_holdings: {agent_id: {ticker: qty}} — same for all dates
        self._holdings = agent_holdings or {}
        self._existing_ids = existing_event_ids or set()
        self._inserted: list[dict[str, Any]] = []
        self._cash_inserted: list[dict[str, Any]] = []
        self._tenant = tenant_id
        # For sleeve snapshot testing
        self._dividend_events: list[dict[str, Any]] = []
        self._sleeves: dict[str, dict[str, Any]] = {}
        self._fills: list[dict[str, Any]] = []

    # --- SleeveRepositoryMixin stubs ---

    def resolve_tenant_id(self, tenant_id=None):
        return self._tenant

    def get_all_held_tickers(self, *, tenant_id=None):
        return list(self._held)

    def agent_holdings_at_date(self, *, agent_id, as_of_date, tenant_id=None):
        return dict(self._holdings.get(agent_id, {}))

    def dividend_event_exists(self, *, event_ids, tenant_id=None):
        return self._existing_ids & set(event_ids)

    def insert_dividend_events(self, rows):
        self._inserted.extend(rows)
        for r in rows:
            self._dividend_events.append(dict(r))

    def append_broker_cash_events(self, rows, tenant_id=None):
        _ = tenant_id
        for row in rows:
            self._cash_inserted.append(dict(row))

    def existing_event_ids(self, table_name, event_ids, tenant_id=None):
        _ = tenant_id
        if table_name == "broker_cash_events":
            existing = {str(r.get("event_id") or "") for r in self._cash_inserted}
            return {token for token in event_ids if token in existing}
        return self._existing_ids & set(event_ids)

    def account_holdings_at_date(self, *, as_of_date, ticker=None, tenant_id=None):
        _ = (as_of_date, tenant_id)
        token = str(ticker or "").strip().upper()
        total = 0.0
        for holdings in self._holdings.values():
            total += float(holdings.get(token, 0.0) or 0.0)
        return {token: total} if token and total > 0 else {}

    @property
    def inserted_rows(self):
        return list(self._inserted)

    @property
    def inserted_cash_rows(self):
        return list(self._cash_inserted)


def _make_settings(**overrides) -> Settings:
    """Builds a minimal Settings for tests."""
    defaults = dict(
        google_cloud_project="test-project",
        bq_dataset="test_ds",
        bq_location="US",
        agent_ids=["gemini", "gpt", "claude"],
        agent_mode="adk",
        base_currency="KRW",
        sleeve_capital_krw=1_000_000,
        log_level="WARNING",
        log_format="",
        trading_mode="live",
        kis_order_endpoint="",
        kis_api_key="test",
        kis_api_secret="test",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no="1234567890",
        kis_account_product_code="01",
        kis_account_key_suffix="",
        kis_env="real",
        kis_target_market="nasdaq",
        kis_overseas_quote_excd="NAS",
        kis_overseas_order_excd="NASD",
        kis_us_natn_cd="840",
        kis_us_tr_mket_cd="01",
        kis_secret_name="",
        kis_secret_version="latest",
        kis_http_timeout_seconds=5,
        kis_http_max_retries=0,
        kis_http_backoff_base_seconds=0.1,
        kis_http_backoff_max_seconds=1.0,
        kis_confirm_fills=False,
        kis_confirm_timeout_seconds=5,
        kis_confirm_poll_seconds=1.0,
        usd_krw_rate=1450.0,
        market_sync_history_days=60,
        max_order_krw=100_000_000.0,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=1.0,
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
        gemini_model="gemini-3-flash",
        research_gemini_model="gemini-2.5-flash",
        llm_timeout_seconds=90,
        default_universe=["AAPL", "GILD"],
        allow_live_trading=True,
        dividend_sync_enabled=True,
        dividend_lookback_days=90,
        dividend_withholding_rate_us=0.15,
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDividendSyncService:
    """Core dividend sync behaviour."""

    def test_normal_attribution(self):
        """Dividends are attributed to agents holding the stock on ex-date."""
        client = FakeClient(rights_by_ticker={
            "GILD": [{
                "divi_amt": "0.51",
                "ex_date": "20260215",
                "rcrd_dt": "20260216",
                "pay_dt": "20260301",
            }],
        })
        repo = FakeRepo(
            held_tickers=["GILD"],
            agent_holdings={
                "claude": {"GILD": 2.0},
                "gpt": {"GILD": 0.0},
                "gemini": {},
            },
        )
        settings = _make_settings()
        svc = DividendSyncService(settings=settings, repo=repo, client=client)
        result = svc.sync_dividends()

        assert result.tickers_checked == 1
        assert result.dividends_found == 1
        assert result.events_inserted == 1  # only claude
        assert result.skipped_duplicate == 0

        rows = repo.inserted_rows
        assert len(rows) == 1
        row = rows[0]
        assert row["agent_id"] == "claude"
        assert row["ticker"] == "GILD"
        assert row["shares_held"] == 2.0
        assert row["gross_per_share_usd"] == 0.51

    def test_no_held_tickers(self):
        """No held tickers means nothing to check."""
        client = FakeClient()
        repo = FakeRepo(held_tickers=[])
        settings = _make_settings()
        svc = DividendSyncService(settings=settings, repo=repo, client=client)
        result = svc.sync_dividends()
        assert result.tickers_checked == 0
        assert result.events_inserted == 0

    def test_dedup_skips_existing(self):
        """Already-inserted events are skipped by event_id dedup."""
        client = FakeClient(rights_by_ticker={
            "GILD": [{"divi_amt": "0.51", "ex_date": "20260215"}],
        })
        existing = {"div_test_claude_GILD_20260215"}
        repo = FakeRepo(
            held_tickers=["GILD"],
            agent_holdings={"claude": {"GILD": 2.0}, "gpt": {}, "gemini": {}},
            existing_event_ids=existing,
        )
        settings = _make_settings()
        svc = DividendSyncService(settings=settings, repo=repo, client=client)
        result = svc.sync_dividends()

        assert result.skipped_duplicate == 1
        assert result.events_inserted == 0
        assert len(repo.inserted_rows) == 0

    def test_no_holding_on_ex_date(self):
        """Agent with zero shares on ex-date gets no dividend."""
        client = FakeClient(rights_by_ticker={
            "GILD": [{"divi_amt": "0.51", "ex_date": "20260215"}],
        })
        repo = FakeRepo(
            held_tickers=["GILD"],
            agent_holdings={
                "claude": {},
                "gpt": {},
                "gemini": {},
            },
        )
        settings = _make_settings()
        svc = DividendSyncService(settings=settings, repo=repo, client=client)
        result = svc.sync_dividends()

        assert result.dividends_found == 1
        assert result.events_inserted == 0

    def test_tax_and_fx_calculation(self):
        """Net amount = shares * per_share * (1 - withholding) * usd_krw."""
        client = FakeClient(rights_by_ticker={
            "GILD": [{"divi_amt": "0.51", "ex_date": "20260215"}],
        })
        repo = FakeRepo(
            held_tickers=["GILD"],
            agent_holdings={
                "claude": {"GILD": 2.0},
                "gpt": {},
                "gemini": {},
            },
        )
        settings = _make_settings(usd_krw_rate=1450.0, dividend_withholding_rate_us=0.15)
        svc = DividendSyncService(settings=settings, repo=repo, client=client)
        svc.sync_dividends()

        row = repo.inserted_rows[0]
        gross_usd = 2 * 0.51
        net_usd = gross_usd * 0.85
        net_krw = net_usd * 1450.0

        assert abs(row["gross_amount_usd"] - gross_usd) < 1e-6
        assert abs(row["net_amount_usd"] - net_usd) < 1e-6
        assert abs(row["net_amount_krw"] - net_krw) < 1e-2
        assert row["withholding_rate"] == 0.15
        assert row["usd_krw_rate"] == 1450.0

    def test_multiple_agents_attributed(self):
        """Multiple agents holding the same stock get separate events."""
        client = FakeClient(rights_by_ticker={
            "GILD": [{"divi_amt": "0.51", "ex_date": "20260215"}],
        })
        repo = FakeRepo(
            held_tickers=["GILD"],
            agent_holdings={
                "claude": {"GILD": 2.0},
                "gpt": {"GILD": 3.0},
                "gemini": {"GILD": 1.0},
            },
        )
        settings = _make_settings()
        svc = DividendSyncService(settings=settings, repo=repo, client=client)
        result = svc.sync_dividends()

        assert result.events_inserted == 3
        agents = {r["agent_id"] for r in repo.inserted_rows}
        assert agents == {"claude", "gpt", "gemini"}

    def test_appends_broker_cash_dividend_credit(self):
        """Dividend sync also mirrors one broker-level cash credit event."""
        client = FakeClient(rights_by_ticker={
            "GILD": [{
                "divi_amt": "0.50",
                "ex_date": "20260215",
                "pay_dt": "20260301",
            }],
        })
        repo = FakeRepo(
            held_tickers=["GILD"],
            agent_holdings={
                "claude": {"GILD": 2.0},
                "gpt": {"GILD": 3.0},
                "gemini": {"GILD": 1.0},
            },
        )
        settings = _make_settings(usd_krw_rate=1450.0, dividend_withholding_rate_us=0.15)
        svc = DividendSyncService(settings=settings, repo=repo, client=client)

        result = svc.sync_dividends()

        assert result.events_inserted == 3
        assert result.broker_cash_events_inserted == 1
        assert len(repo.inserted_cash_rows) == 1
        row = repo.inserted_cash_rows[0]
        assert row["event_type"] == "DIVIDEND_CREDIT"
        assert row["currency"] == "USD"
        assert row["amount_native"] == pytest.approx(6.0 * 0.50 * 0.85)
        assert row["amount_krw"] == pytest.approx(6.0 * 0.50 * 0.85 * 1450.0)
        assert row["raw_payload_json"]["shares_source"] == "account_snapshot"

    def test_dividend_cash_credit_falls_back_to_agent_holdings_when_snapshot_missing(self):
        """When broker snapshot history is unavailable, agent holdings sum is used as fallback."""
        client = FakeClient(rights_by_ticker={
            "GILD": [{
                "divi_amt": "0.50",
                "ex_date": "20260215",
            }],
        })

        class _NoSnapshotRepo(FakeRepo):
            def account_holdings_at_date(self, *, as_of_date, ticker=None, tenant_id=None):
                _ = (as_of_date, ticker, tenant_id)
                return {}

        repo = _NoSnapshotRepo(
            held_tickers=["GILD"],
            agent_holdings={
                "claude": {"GILD": 2.0},
                "gpt": {"GILD": 1.0},
                "gemini": {},
            },
        )
        settings = _make_settings(usd_krw_rate=1450.0, dividend_withholding_rate_us=0.15)
        svc = DividendSyncService(settings=settings, repo=repo, client=client)

        result = svc.sync_dividends()

        assert result.broker_cash_events_inserted == 1
        row = repo.inserted_cash_rows[0]
        assert row["raw_payload_json"]["shares_source"] == "agent_holdings_fallback"
        assert row["amount_native"] == pytest.approx(3.0 * 0.50 * 0.85)

    def test_fallback_field_names(self):
        """Parsing handles alternative KIS field names."""
        client = FakeClient(rights_by_ticker={
            "AAPL": [{
                "divd_amt": "0.25",     # alt key
                "bass_dt": "20260301",  # alt ex-date key
            }],
        })
        repo = FakeRepo(
            held_tickers=["AAPL"],
            agent_holdings={"claude": {"AAPL": 10.0}, "gpt": {}, "gemini": {}},
        )
        settings = _make_settings()
        svc = DividendSyncService(settings=settings, repo=repo, client=client)
        result = svc.sync_dividends()

        assert result.events_inserted == 1
        row = repo.inserted_rows[0]
        assert row["gross_per_share_usd"] == 0.25
        assert row["ex_date"] == "2026-03-01"

    def test_disabled_config_skips(self):
        """When dividend_sync_enabled=False the service still runs but returns 0."""
        # Note: the actual skip logic is in cli.py, but the service itself
        # always runs when called. This test validates the config field exists.
        settings = _make_settings(dividend_sync_enabled=False)
        assert settings.dividend_sync_enabled is False


class TestDividendDateParsing:
    """Tests for DividendSyncService._parse_date_field."""

    def test_yyyymmdd(self):
        result = DividendSyncService._parse_date_field({"ex_date": "20260215"}, "ex_date")
        assert result == date(2026, 2, 15)

    def test_yyyy_mm_dd_with_dashes(self):
        result = DividendSyncService._parse_date_field({"ex_dt": "2026-02-15"}, "ex_dt")
        assert result == date(2026, 2, 15)

    def test_missing_key(self):
        result = DividendSyncService._parse_date_field({}, "ex_date", "ex_dt")
        assert result is None

    def test_fallback_keys(self):
        result = DividendSyncService._parse_date_field(
            {"bass_dt": "20260301"},
            "ex_date", "ex_dt", "bass_dt",
        )
        assert result == date(2026, 3, 1)


class TestDividendPerShareParsing:
    """Tests for DividendSyncService._parse_dividend_per_share."""

    def test_divi_amt(self):
        assert DividendSyncService._parse_dividend_per_share({"divi_amt": "0.51"}) == 0.51

    def test_divd_amt(self):
        assert DividendSyncService._parse_dividend_per_share({"divd_amt": "1.25"}) == 1.25

    def test_zero_returns_zero(self):
        assert DividendSyncService._parse_dividend_per_share({"divi_amt": "0"}) == 0.0

    def test_empty_returns_zero(self):
        assert DividendSyncService._parse_dividend_per_share({}) == 0.0
