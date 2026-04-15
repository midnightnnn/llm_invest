from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import inspect
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from arena.agents.adk_agents import (
    AdkTradingAgent,
    _ADKDecisionRunner,
    _apply_tool_schema_metadata,
    _compact_tool_result_for_prompt,
    _ContextTools,
    _has_credentials,
    _is_gemini_quota_error,
    _load_disabled_tool_ids,
    _resolve_disabled_tool_ids,
    _resolve_model,
    _system_prompt,
    _user_prompt,
)
from arena.agents.adk_models import (
    _is_vertex_model_access_error,
    _normalize_vertex_anthropic_model,
)
from arena.agents.adk_agent_flow import (
    draft_phase_output,
    extract_decision_payload,
    retry_policy_from_env,
)
from arena.agents.adk_decision_flow import (
    build_tool_summary_memory_record,
    parse_board_response,
    prepare_decision_prompt,
)
from arena.agents.adk_order_support import (
    build_order_intents,
    fetch_market_row_from_bq,
    format_orders_summary,
    resolve_order_price,
)
from arena.agents.adk_runner_bootstrap import resolve_max_tool_events, runner_identity
from arena.agents.adk_runner_runtime import collect_response_text
from arena.config import AgentConfig, load_settings
from arena.memory.query_builders import build_memory_query
from arena.models import BoardPost, ExecutionReport, ExecutionStatus, OrderIntent, Side, utc_now
from arena.tools.registry import ToolEntry, ToolRegistry


def test_normalize_vertex_anthropic_alias_sonnet_46() -> None:
    out = _normalize_vertex_anthropic_model("claude-sonnet-4-6")
    assert out == "vertex_ai/claude-sonnet-4-5"


def test_normalize_vertex_anthropic_keeps_versioned_model() -> None:
    out = _normalize_vertex_anthropic_model("vertex_ai/claude-sonnet-4-5@20250929")
    assert out == "vertex_ai/claude-sonnet-4-5@20250929"


def test_vertex_model_access_error_detects_not_found_access() -> None:
    exc = RuntimeError(
        "litellm.NotFoundError: Vertex_aiException - "
        "Publisher Model `projects/x/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-6` "
        "was not found or your project does not have access to it."
    )
    assert _is_vertex_model_access_error(exc) is True


def test_vertex_model_access_error_detects_quota_exhausted() -> None:
    exc = RuntimeError(
        "litellm.RateLimitError: Vertex_aiException - "
        "429 RESOURCE_EXHAUSTED. quota exceeded for publishers/anthropic/models/claude-sonnet-4-5"
    )
    assert _is_vertex_model_access_error(exc) is True


def test_vertex_model_access_error_ignores_unrelated_errors() -> None:
    exc = RuntimeError("rate limit exceeded 429")
    assert _is_vertex_model_access_error(exc) is False


def test_gemini_quota_error_detects_resource_exhausted() -> None:
    exc = RuntimeError(
        "google.genai.errors.ClientError: 429 RESOURCE_EXHAUSTED. "
        "Resource exhausted. Please try again later."
    )
    assert _is_gemini_quota_error(exc) is True


def test_gemini_quota_error_ignores_non_quota_errors() -> None:
    exc = RuntimeError("invalid_argument: malformed function call schema")
    assert _is_gemini_quota_error(exc) is False


def test_apply_tool_schema_metadata_prefers_registry_description() -> None:
    def original_tool(ticker: str) -> dict[str, str]:
        """Original docstring that should not leak to the model."""
        return {"ticker": ticker}

    entry = ToolEntry(
        tool_id="screen_market",
        name="screen_market",
        description="Canonical registry description for the model schema.",
        category="quant",
        callable=original_tool,
    )

    wrapped = _apply_tool_schema_metadata(
        original_tool,
        entry=entry,
        sig=inspect.signature(original_tool),
    )

    assert wrapped.__name__ == "screen_market"
    assert wrapped.__doc__ == "Canonical registry description for the model schema."


class _RepoForPrompt:
    def __init__(self, value: str | None):
        self._value = value

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        _ = tenant_id, config_key
        return self._value


def test_system_prompt_uses_db_when_available() -> None:
    repo = _RepoForPrompt("You are a trading agent.")
    out = _system_prompt("test-agent", repo=repo, tenant_id="tenant-a")
    assert "test-agent" in out
    assert "You are a trading agent." in out


def test_system_prompt_uses_agent_config_override() -> None:
    repo = _RepoForPrompt("Global prompt from DB.")
    ac = AgentConfig(
        agent_id="custom",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        system_prompt="Custom per-agent prompt.",
    )
    out = _system_prompt("custom", repo=repo, tenant_id="tenant-a", agent_config=ac)
    assert "Custom per-agent prompt." in out
    # Should NOT include the DB global prompt
    assert "Global prompt from DB." not in out


class _RepoForTools:
    def __init__(self, disabled: str | None):
        self.disabled = disabled

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        _ = tenant_id, config_key
        return self.disabled


def test_load_disabled_tool_ids_uses_tool_id_tokens() -> None:
    repo = _RepoForTools('["fetch_reddit_sentiment","optimize_portfolio"]')
    out = _load_disabled_tool_ids(repo, "tenant-a")
    assert out == {"fetch_reddit_sentiment", "optimize_portfolio"}


def test_runner_identity_uses_agent_scoped_names() -> None:
    identity = runner_identity("gpt")

    assert identity.app_name == "llm_arena_gpt"
    assert identity.user_id == "arena"
    assert identity.session_id == "gpt_react"


def test_resolve_max_tool_events_clamps_invalid_and_high_values() -> None:
    settings = load_settings()
    settings.adk_max_tool_events = "oops"
    assert resolve_max_tool_events(settings) == 120

    settings.adk_max_tool_events = 999
    assert resolve_max_tool_events(settings) == 400


def test_prepare_decision_prompt_resume_reuses_session_and_includes_board_context() -> None:
    session_id, prompt, needs_new_session = prepare_decision_prompt(
        {
            "board_context": "peer conviction is rising",
            "order_budget": {"max_buy_notional_krw": 1_000_000},
            "risk_policy": {"max_position_ratio": 0.2},
            "decision_frame": "Compare opportunities against weakest holding.",
            "candidate_cases": [{"ticker": "MSFT", "case_for": "screened candidate"}],
        },
        default_universe=["AAPL"],
        phase="execution",
        base_session_id="gpt_react",
        max_tool_events=12,
        resume_session_id="resume_1",
        analysis_funnel={"pending_nonheld": 2},
    )

    assert session_id == "resume_1"
    assert needs_new_session is False
    assert "peer conviction is rising" in prompt
    assert "Compare opportunities against weakest holding." in prompt
    assert "screened candidate" in prompt
    assert '"max_tool_calls": 12' in prompt


def test_build_tool_summary_memory_record_keeps_token_usage_even_without_events() -> None:
    record = build_tool_summary_memory_record(
        [],
        registry=ToolRegistry([]),
        phase="draft",
        analysis_funnel={"discovered_nonheld": 0},
        cycle_id="cycle_1",
        token_usage={"llm_calls": 1, "prompt_tokens": 120},
    )

    assert record is not None
    summary, payload = record
    assert "ReAct tools used (draft): 0" in summary
    assert payload["token_usage"]["prompt_tokens"] == 120
    assert payload["analysis_funnel"]["discovered_nonheld"] == 0


def test_parse_board_response_raises_on_plain_text_body() -> None:
    with pytest.raises(Exception):
        parse_board_response("plain board body")


def test_retry_policy_from_env_clamps_extreme_values(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_ADK_RETRY_MAX", "99")
    monkeypatch.setenv("ARENA_ADK_RETRY_BACKOFF_SECONDS", "99")

    retry_limit, retry_delay = retry_policy_from_env()

    assert retry_limit == 4
    assert retry_delay == 10.0


class _RepoForMarketLookup:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def latest_market_features(self, *, tickers, limit, sources=None):
        self.calls.append(
            {
                "tickers": list(tickers),
                "limit": limit,
                "sources": list(sources) if isinstance(sources, list) else sources,
            }
        )
        return list(self.rows)


def test_user_prompt_omits_sleeve_state_payload() -> None:
    prompt = _user_prompt(
        {
            "cycle_phase": "execution",
            "portfolio": {"cash_krw": 0},
            "risk_policy": {},
            "order_budget": {"max_buy_notional_krw": 0.0},
            "sleeve_state": {"buy_blocked": True, "over_target": True},
            "analysis_funnel": {"discovered_nonheld": 3, "analyzed_nonheld": 1, "pending_nonheld": 2},
            "active_thesis_context": "Active Thesis:\n- [AAPL | open] compact thesis",
            "active_theses": [{"ticker": "AAPL", "payload_json": '{"raw": "large"}'}],
            "opportunity_working_set": [{"ticker": "TSLA", "status": "pending"}],
            "decision_frame": "Compare self-discovered opportunities against cash first.",
            "market_context": [{"ticker": "AAPL", "close": 123.45}],
            "research_context": "- [AAPL] New product cycle - Demand watchlist.",
            "relation_context": "Relation Hints:\n- contains ticker AAPL: prior risk lesson.",
            "graph_context": "Decision Paths:\n- AAPL prior entry connects to a later win.",
            "memory_context": "Portfolio Memory:\n- [AAPL | BUY] Keep this compressed lesson.",
            "memory_events": [{"summary": "Do not duplicate this raw memory summary."}],
        },
        default_universe=[],
        max_tool_calls=5,
    )
    marker = "Context payload JSON"
    json_start = prompt.index("{", prompt.index(marker))
    payload = json.loads(prompt[json_start:])
    assert "sleeve_state" not in payload
    assert payload["active_thesis_context"] == "Active Thesis:\n- [AAPL | open] compact thesis"
    assert "active_theses" not in payload
    assert payload["analysis_funnel"]["screened_only_candidates"] == 2
    assert "pending_nonheld" not in payload["analysis_funnel"]
    assert "opportunity_working_set" not in payload
    assert payload["candidate_cases"] == []
    assert payload["decision_frame"] == "Compare self-discovered opportunities against cash first."
    assert payload["market_context"] == [{"ticker": "AAPL", "close": 123.45}]
    assert payload["research_context"] == "- [AAPL] New product cycle - Demand watchlist."
    assert payload["relation_context"] == "Relation Hints:\n- contains ticker AAPL: prior risk lesson."
    assert payload["graph_context"] == "Decision Paths:\n- AAPL prior entry connects to a later win."
    assert payload["memory_context"] == "Portfolio Memory:\n- [AAPL | BUY] Keep this compressed lesson."
    assert "memory_events" not in payload
    assert "recent_memory_summaries" not in payload
    assert payload["tool_budget"]["max_tool_calls"] == 5


def test_prompt_context_sections_collects_prompt_details() -> None:
    sections = _ADKDecisionRunner._prompt_context_sections(
        {
            "portfolio": {"cash_krw": 1000},
            "market_features": [{"ticker": "AAPL", "close": 123.45}],
            "board_posts": [{"post_id": "board_1", "summary": "Hold watchlist"}],
            "research_context": "- [AAPL] New product cycle - Demand watchlist.",
            "relation_context": "Relation Hints:\n- contains ticker AAPL: prior risk lesson.",
            "graph_context": "Decision Paths:\n- AAPL prior entry connects to a later win.",
            "memory_context": "Memory:\n- Prefer staged entries.",
        }
    )

    assert sections["portfolio_context"] == {"cash_krw": 1000}
    assert sections["market_context"] == [{"ticker": "AAPL", "close": 123.45}]
    assert sections["board_context"] == [{"post_id": "board_1", "summary": "Hold watchlist"}]
    assert sections["research_context"] == "- [AAPL] New product cycle - Demand watchlist."
    assert sections["relation_context"] == "Relation Hints:\n- contains ticker AAPL: prior risk lesson."
    assert sections["graph_context"] == "Decision Paths:\n- AAPL prior entry connects to a later win."
    assert sections["memory_context"] == "Memory:\n- Prefer staged entries."


def test_fetch_market_row_from_bq_uses_live_kospi_sources() -> None:
    settings = load_settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "kospi"
    repo = _RepoForMarketLookup(
        [
            {"ticker": "005930", "close_price_krw": 70500.0, "close_price_native": 70500.0},
        ]
    )

    row = fetch_market_row_from_bq(repo, settings, "005930")

    assert row is not None
    assert row["ticker"] == "005930"
    assert repo.calls[0]["sources"] == ["open_trading_kospi_quote", "open_trading_kospi"]


def test_resolve_order_price_prefers_live_fx_for_us_quotes() -> None:
    settings = load_settings()
    settings.kis_target_market = "nasdaq"
    settings.usd_krw_rate = 1300.0

    price_krw, native_price, quote_currency, fx_rate = resolve_order_price(
        settings,
        market_row={
            "close_price_native": 10.0,
            "fx_rate_used": 1250.0,
        },
        portfolio={"usd_krw_rate": 1400.0},
    )

    assert price_krw == pytest.approx(14000.0)
    assert native_price == pytest.approx(10.0)
    assert quote_currency == "USD"
    assert fx_rate == pytest.approx(1400.0)


def test_resolve_order_price_returns_zero_when_us_fx_is_missing() -> None:
    settings = load_settings()
    settings.kis_target_market = "nasdaq"
    settings.usd_krw_rate = 1300.0

    price_krw, native_price, quote_currency, fx_rate = resolve_order_price(
        settings,
        market_row={
            "close_price_native": 10.0,
            "fx_rate_used": 0.0,
        },
        portfolio={"usd_krw_rate": 0.0},
    )

    assert price_krw == pytest.approx(0.0)
    assert native_price == pytest.approx(10.0)
    assert quote_currency == "USD"
    assert fx_rate == pytest.approx(0.0)


def test_format_orders_summary_includes_hold_rows() -> None:
    summary = format_orders_summary(
        [
            OrderIntent(
                agent_id="gpt",
                ticker="AAPL",
                side=Side.BUY,
                quantity=3.0,
                price_krw=10000.0,
                rationale="Breakout continuation with supportive breadth.",
            )
        ],
        [{"ticker": "MSFT", "side": "HOLD", "rationale": "No edge after recent gap."}],
    )

    assert "AAPL BUY 3.0주" in summary
    assert "MSFT HOLD" in summary


def test_format_orders_summary_uses_known_kospi_ticker_name() -> None:
    summary = format_orders_summary(
        [
            OrderIntent(
                agent_id="claude",
                ticker="025860",
                side=Side.BUY,
                quantity=3.0,
                price_krw=8270.0,
                rationale="실적 회복 기대.",
            )
        ],
        [],
        ticker_names={"025860": "남해화학"},
    )

    assert "남해화학(025860) BUY 3.0주" in summary


def test_extract_decision_payload_normalizes_non_list_orders() -> None:
    draft_summary, orders = extract_decision_payload(
        {
            "draft_summary": "  concise draft  ",
            "orders": {"ticker": "AAPL"},
        }
    )

    assert draft_summary == "concise draft"
    assert orders == []


def test_draft_phase_output_uses_distinct_tickers() -> None:
    out = draft_phase_output(
        agent_id="gpt",
        cycle_id="cycle_draft_1",
        decision={
            "board_title": "draft title",
            "board_body": "draft body",
        },
        draft_summary="summary",
        orders=[
            {"ticker": "AAPL"},
            {"ticker": "MSFT"},
            {"ticker": "AAPL"},
        ],
    )

    assert out.intents == []
    assert out.board_post.title == "draft title"
    assert out.board_post.draft_summary == "summary"
    assert out.board_post.tickers == ["AAPL", "MSFT"]


def test_build_order_intents_defaults_single_market_us_exchange() -> None:
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "nasdaq"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0

    intents, tickers_mentioned = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_1",
        context={
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "positions": {},
            },
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        },
        orders=[
            {
                "ticker": "AAPL",
                "side": "BUY",
                "size_ratio": 0.5,
                "rationale": "single-market default exchange",
            }
        ],
        row_map={
            "AAPL": {
                "ticker": "AAPL",
                "exchange_code": "",
                "instrument_id": "",
                "close_price_krw": 130000.0,
                "close_price_native": 100.0,
                "quote_currency": "USD",
                "fx_rate_used": 1300.0,
            }
        },
    )

    assert tickers_mentioned == {"AAPL"}
    assert len(intents) == 1
    assert intents[0].exchange_code == "NASD"
    assert intents[0].instrument_id == "NASD:AAPL"


def test_resolve_order_price_multi_market_infers_usd_from_exchange_identity() -> None:
    settings = load_settings()
    settings.kis_target_market = "us,kospi"

    price_krw, native_price, quote_currency, fx_rate = resolve_order_price(
        settings,
        market_row={
            "ticker": "AAPL",
            "exchange_code": "NAS",
            "instrument_id": "NASD:AAPL",
            "close_price_native": 100.0,
            "fx_rate_used": 1300.0,
        },
        portfolio={},
    )

    assert price_krw == 130000.0
    assert native_price == 100.0
    assert quote_currency == "USD"
    assert fx_rate == 1300.0


def test_build_order_intents_multi_market_defaults_korean_exchange() -> None:
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us,kospi"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0

    intents, tickers_mentioned = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_combo_kr",
        context={
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "positions": {},
            },
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        },
        orders=[
            {
                "ticker": "005930",
                "side": "BUY",
                "size_ratio": 0.5,
                "rationale": "combo-market KRX inference",
            }
        ],
        row_map={
            "005930": {
                "ticker": "005930",
                "exchange_code": "",
                "instrument_id": "",
                "close_price_krw": 70000.0,
                "quote_currency": "KRW",
            }
        },
    )

    assert tickers_mentioned == {"005930"}
    assert len(intents) == 1
    assert intents[0].exchange_code == "KRX"
    assert intents[0].instrument_id == "KRX:005930"


def test_build_order_intents_collects_feedback_events() -> None:
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "nasdaq"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    feedback_events: list[dict[str, object]] = []

    intents, tickers_mentioned = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_feedback",
        context={
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "positions": {},
            },
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        },
        orders=[
            {
                "ticker": "AAPL",
                "side": "BUY",
                "size_ratio": 0.5,
                "rationale": "build intent",
            },
            {
                "ticker": "TSLA",
                "side": "BUY",
                "size_ratio": 0.3,
                "rationale": "missing price",
            },
        ],
        row_map={
            "AAPL": {
                "ticker": "AAPL",
                "exchange_code": "",
                "instrument_id": "",
                "close_price_krw": 130000.0,
                "close_price_native": 100.0,
                "quote_currency": "USD",
                "fx_rate_used": 1300.0,
            }
        },
        feedback_events=feedback_events,
    )

    assert tickers_mentioned == {"AAPL", "TSLA"}
    assert len(intents) == 1
    assert feedback_events == [
        {"ticker": "AAPL", "side": "BUY", "status": "intent_built"},
        {"ticker": "TSLA", "side": "BUY", "status": "skipped", "reason": "no_price"},
    ]


def test_build_order_intents_live_sell_rounds_up_small_position() -> None:
    settings = load_settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "kospi"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0

    intents, tickers_mentioned = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_2",
        context={
            "portfolio": {
                "cash_krw": 100000.0,
                "total_equity_krw": 100000.0,
                "positions": {
                    "005930": {
                        "quantity": 1.0,
                        "avg_price_krw": 70000.0,
                    }
                },
            },
            "order_budget": {},
        },
        orders=[
            {
                "ticker": "005930",
                "side": "SELL",
                "size_ratio": 0.1,
                "rationale": "small live trim",
            }
        ],
        row_map={
            "005930": {
                "ticker": "005930",
                "exchange_code": "",
                "instrument_id": "",
                "close_price_krw": 70000.0,
                "quote_currency": "KRW",
            }
        },
    )

    assert tickers_mentioned == {"005930"}
    assert len(intents) == 1
    assert intents[0].quantity == 1.0
    assert intents[0].exchange_code == "KRX"
    assert intents[0].instrument_id == "KRX:005930"


def test_candidate_ledger_tracks_discovery_and_analysis_from_result_tickers() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {}
    runner._held_tickers_cache = {"AAPL"}
    runner._current_phase = "draft"
    runner._current_context = {}
    runner._tool_events = []

    runner._update_candidate_ledger(
        "screen_market",
        {},
        [
            {"ticker": "AAPL", "bucket": "momentum"},
            {
                "ticker": "MSFT",
                "bucket": "value",
                "score": 1.7,
                "reason_for": "Valuation support: PER 14.0",
                "reason_risk": "Screen-only evidence; confirm first.",
                "ret_20d": 0.04,
            },
            {"ticker": "TSLA", "bucket": "pullback", "reason": "Recent pullback"},
        ],
    )

    assert set(runner._candidate_ledger.keys()) == {"MSFT", "TSLA"}
    assert runner._candidate_ledger["MSFT"]["source_tools"] == {"screen_market:value"}
    assert runner._candidate_ledger["TSLA"]["source_tools"] == {"screen_market:pullback"}
    assert runner._current_context["_candidate_tickers"] == ["MSFT", "TSLA"]
    assert runner._current_context["_discovered_candidate_tickers"] == ["MSFT", "TSLA"]
    assert [row["ticker"] for row in runner._current_context["opportunity_working_set"]] == ["MSFT", "TSLA"]
    assert runner._current_context["opportunity_working_set"][0]["status"] == "screened_only"
    assert runner._current_context["opportunity_working_set"][0]["workflow_status"] == "pending"
    assert runner._current_context["opportunity_working_set"][0]["discovery_buckets"] == ["value"]
    assert runner._current_context["analysis_funnel_prompt"]["screened_only_candidates"] == 2
    assert runner._current_context["candidate_cases"][0]["ticker"] == "MSFT"
    assert runner._current_context["candidate_cases"][0]["case_for"].startswith("Valuation support")
    assert runner._current_context["candidate_cases"][0]["case_risk"] == "Screen-only evidence; confirm first."
    assert runner._current_context["candidate_cases"][0]["evidence_level"] == "screened_only"
    assert "thesis_summary" not in runner._current_context["candidate_cases"][0]

    runner._update_candidate_ledger(
        "forecast_returns",
        {},
        [{"ticker": "MSFT", "exp_return_period": 0.02}],
    )

    assert runner._candidate_ledger["MSFT"]["analyzed_by"] == {"forecast_returns"}
    assert runner._current_context["analysis_funnel"]["analyzed_nonheld"] == 1
    assert runner._current_context["analysis_funnel"]["pending_nonheld"] == 1
    assert runner._current_context["analysis_funnel_prompt"]["fully_analyzed_candidates"] == 1
    assert runner._current_context["analysis_funnel_prompt"]["screened_only_candidates"] == 1
    assert runner._current_context["opportunity_working_set"][0]["ticker"] == "TSLA"


def test_candidate_ledger_records_screen_market_momentum_bucket_source() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {}
    runner._held_tickers_cache = set()
    runner._current_phase = "draft"
    runner._current_context = {}
    runner._tool_events = []

    runner._update_candidate_ledger(
        "screen_market",
        {},
        [{"ticker": "PBR", "bucket": "momentum", "score": 1.25}],
    )

    assert runner._candidate_ledger["PBR"]["source_tools"] == {"screen_market:momentum"}
    assert runner._current_context["opportunity_working_set"][0]["discovery_buckets"] == ["momentum"]
    assert runner._current_context["opportunity_working_set"][0]["status"] == "screened_only"
    assert runner._current_context["candidate_cases"][0]["ticker"] == "PBR"


def test_funnel_metrics_counts_held_analysis_from_result_tickers() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {
        "MSFT": {"analyzed_by": {"forecast_returns"}},
        "TSLA": {"analyzed_by": set()},
    }
    runner._held_tickers_cache = {"AAPL"}
    runner._tool_events = [
        {
            "tool": "forecast_returns",
            "args": {},
            "result": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
        },
        {
            "tool": "technical_signals",
            "args": {"ticker": "AAPL"},
            "result": {"ticker": "AAPL", "trend_state": "uptrend"},
        },
    ]

    metrics = runner._funnel_metrics()

    assert metrics == {
        "discovered_nonheld": 2,
        "analyzed_nonheld": 1,
        "pending_nonheld": 1,
        "analyzed_held": 1,
        "ordered_nonheld": 0,
        "intended_nonheld": 0,
        "executed_nonheld": 0,
        "skipped_nonheld": 0,
        "skip_reasons": {},
    }


def test_candidate_ledger_tracks_order_and_execution_funnel() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {
        "MSFT": {"analyzed_by": {"forecast_returns"}},
        "TSLA": {"analyzed_by": set()},
    }
    runner._held_tickers_cache = {"AAPL"}
    runner._current_phase = "execution"
    runner._current_context = {}
    runner._tool_events = []

    runner.record_candidate_orders(
        [
            {"ticker": "MSFT", "side": "BUY", "size_ratio": 0.2},
            {"ticker": "AAPL", "side": "BUY", "size_ratio": 0.1},
        ]
    )
    runner.record_candidate_order_feedback(
        [
            {"ticker": "MSFT", "side": "BUY", "status": "intent_built"},
            {"ticker": "TSLA", "side": "BUY", "status": "skipped", "reason": "no_price"},
        ]
    )
    runner.record_candidate_executions(
        [
            OrderIntent(
                agent_id="gpt",
                ticker="MSFT",
                side=Side.BUY,
                quantity=1.0,
                price_krw=1000.0,
                rationale="candidate buy",
            )
        ],
        [
            ExecutionReport(
                status=ExecutionStatus.FILLED,
                order_id="ord_1",
                filled_qty=1.0,
                avg_price_krw=1000.0,
                message="filled",
            )
        ],
    )

    metrics = runner._funnel_metrics()

    assert metrics["ordered_nonheld"] == 1
    assert metrics["intended_nonheld"] == 1
    assert metrics["executed_nonheld"] == 1
    assert metrics["skipped_nonheld"] == 1
    assert metrics["skip_reasons"] == {"no_price": 1}


def test_sync_pipeline_context_adds_decision_frame_when_opportunities_have_budget() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {"MSFT": {"source_tools": {"screen_market"}, "analyzed_by": set()}}
    runner._held_tickers_cache = {"AAPL"}
    runner._current_phase = "execution"
    runner._tool_events = []
    runner._current_context = {"order_budget": {"max_buy_notional_krw": 500000.0}}

    runner._sync_pipeline_context()

    assert "Compare any self-discovered opportunities" in runner._current_context["decision_frame"]
    assert runner._current_context["candidate_cases"][0]["ticker"] == "MSFT"
    assert runner._current_context["candidate_cases"][0]["candidate_status"] == "screened_only"


class _RepoForPortfolioDiagnosis:
    def get_daily_closes(self, tickers, lookback_days, sources=None):
        _ = lookback_days, sources
        base = {
            "AAPL": [100.0, 101.0, 103.0, 104.0, 106.0, 108.0, 109.0, 111.0, 112.0, 114.0, 116.0, 118.0],
            "MSFT": [200.0, 199.0, 198.0, 201.0, 202.0, 204.0, 205.0, 207.0, 209.0, 210.0, 212.0, 214.0],
            "QQQ": [300.0, 301.0, 302.0, 304.0, 306.0, 307.0, 309.0, 311.0, 312.0, 314.0, 316.0, 318.0],
        }
        return {ticker: base.get(ticker, []) for ticker in tickers}


class _RepoForPortfolioDiagnosisExact(_RepoForPortfolioDiagnosis):
    def __init__(self) -> None:
        self.frame_calls: list[dict[str, object]] = []

    def get_daily_close_frame(self, *, tickers, start, end, sources=None):  # noqa: ANN001
        self.frame_calls.append(
            {
                "tickers": list(tickers),
                "start": start,
                "end": end,
                "sources": list(sources) if isinstance(sources, list) else sources,
            }
        )
        frame = pd.DataFrame(
            {"QQQ": [90.0, 100.0, 100.0]},
            index=pd.to_datetime(["2026-01-02", "2026-03-03", "2026-03-27"]),
        )
        mask = (frame.index.date >= start) & (frame.index.date <= end)
        return frame.loc[mask]


class _RepoForPortfolioDiagnosisRaises(_RepoForPortfolioDiagnosis):
    def get_daily_closes(self, tickers, lookback_days, sources=None):
        if int(lookback_days) <= 10:
            raise RuntimeError("no closes")
        return super().get_daily_closes(tickers, lookback_days, sources=sources)


class _RepoForPeerLessons:
    def memory_events_by_ids_any_agent(self, *, event_ids, trading_mode="paper", tenant_id=None):
        _ = (trading_mode, tenant_id)
        rows = {
            "mem_peer": {
                "event_id": "mem_peer",
                "agent_id": "gemini",
                "payload_json": json.dumps({"source": "thesis_chain_compaction"}),
            },
            "mem_manual": {
                "event_id": "mem_manual",
                "agent_id": "claude",
                "payload_json": json.dumps({"source": "manual_note"}),
            },
        }
        return [rows[eid] for eid in event_ids if eid in rows]


class _VectorStoreForPeerLessons:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_peer_lessons(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "event_id": "mem_peer",
                "agent_id": "gemini",
                "summary": "Trim single-name exposure after fast gains.",
                "created_date": "2026-03-07",
            },
            {
                "event_id": "mem_manual",
                "agent_id": "claude",
                "summary": "Manual reflection that should be filtered out.",
                "created_date": "2026-03-06",
            },
        ]


class _RepoForResearchBriefingFallback:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_research_briefings(self, *, tickers=None, categories=None, limit=10, trading_mode="paper", tenant_id=None):
        self.calls.append(
            {
                "tickers": list(tickers) if tickers else None,
                "categories": list(categories) if categories else None,
                "limit": limit,
                "trading_mode": trading_mode,
                "tenant_id": tenant_id,
            }
        )
        tenant = str(tenant_id or "").strip().lower()
        if tenant == "tenant-a":
            return []
        if tenant == "midnightnnn":
            rows = [
                {
                    "briefing_id": "pub_global",
                    "category": "global_market",
                    "ticker": "GLOBAL",
                    "headline": "Global",
                    "summary": "global summary",
                    "sources": "[]",
                },
                {
                    "briefing_id": "pub_geo",
                    "category": "geopolitical",
                    "ticker": "GEOPOLITICAL",
                    "headline": "Geo",
                    "summary": "geo summary",
                    "sources": "[]",
                },
                {
                    "briefing_id": "pub_sector",
                    "category": "sector_trends",
                    "ticker": "SECTOR",
                    "headline": "Sector",
                    "summary": "sector summary",
                    "sources": "[]",
                },
            ]
            if categories:
                allowed = {str(token).strip().lower() for token in categories if str(token).strip()}
                rows = [row for row in rows if str(row.get("category") or "").strip().lower() in allowed]
            return rows[:limit]
        return []


def test_portfolio_diagnosis_returns_derived_fields_not_raw_portfolio_echo() -> None:
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForPortfolioDiagnosis()
    tool.settings = load_settings()
    tool.settings.kis_target_market = "nasdaq"
    tool._context = {
        "portfolio": {
            "cash_krw": 1_000.0,
            "positions": {
                "AAPL": {"quantity": 10.0, "avg_price_krw": 100.0},
                "MSFT": {"quantity": 5.0, "avg_price_krw": 200.0},
            },
        },
        "market_features": [
            {"ticker": "AAPL", "close_price_krw": 110.0, "volatility_20d": 0.2, "ret_20d": 0.08, "ret_5d": 0.03},
            {"ticker": "MSFT", "close_price_krw": 208.0, "volatility_20d": 0.1, "ret_20d": 0.04, "ret_5d": 0.01},
        ],
        "performance": {
            "initialized_at": "2026-01-01T00:00:00+00:00",
            "pnl_ratio": 0.12,
        },
        "risk_policy": {
            "min_cash_buffer_ratio": 0.10,
            "max_position_ratio": 0.60,
        },
    }

    out = tool.portfolio_diagnosis(mdd_days=5, top_n=2)

    assert "cash_krw" not in out
    assert "stock_market_value_krw" not in out
    assert "weights" not in out
    assert "performance" not in out
    assert "top_weights" not in out
    assert "cash_weight" not in out
    assert "gross_exposure" not in out
    assert "risk_contribution" in out
    assert "mdd" in out
    assert out["benchmark"]["ticker"] == "QQQ"
    assert "rebalance_plan" not in out
    assert out["hrp_allocation"]["status"] == "ready"
    assert out["hrp_allocation"]["strategy"] == "hrp"
    assert out["hrp_allocation"]["hrp_cash_weight"] == pytest.approx(0.10, abs=1e-6)
    assert sum(row["hrp_weight"] for row in out["hrp_allocation"]["hrp_weights"]) == pytest.approx(0.90, abs=1e-6)
    assert out["hrp_allocation"]["weight_deltas"]
    assert all("side" not in row for row in out["hrp_allocation"]["weight_deltas"])
    assert all("size_ratio" not in row for row in out["hrp_allocation"]["weight_deltas"])


def test_portfolio_diagnosis_aligns_benchmark_period_with_current_sleeve_return(monkeypatch) -> None:
    tool = _ContextTools.__new__(_ContextTools)
    repo = _RepoForPortfolioDiagnosisExact()
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.kis_target_market = "nasdaq"
    tool._context = {
        "portfolio": {
            "cash_krw": 1_000.0,
            "positions": {
                "AAPL": {"quantity": 10.0, "avg_price_krw": 100.0},
                "MSFT": {"quantity": 5.0, "avg_price_krw": 200.0},
            },
        },
        "market_features": [
            {"ticker": "AAPL", "close_price_krw": 110.0, "volatility_20d": 0.2, "ret_20d": 0.08, "ret_5d": 0.03},
            {"ticker": "MSFT", "close_price_krw": 208.0, "volatility_20d": 0.1, "ret_20d": 0.04, "ret_5d": 0.01},
        ],
        "performance": {
            "initialized_at": "2026-01-01T00:00:00+00:00",
            "cumulative_pnl_ratio": 0.30,
            "current_sleeve_initialized_at": "2026-03-01T00:00:00+00:00",
            "current_sleeve_pnl_ratio": 0.05,
            "pnl_ratio": 0.05,
        },
        "risk_policy": {
            "min_cash_buffer_ratio": 0.10,
            "max_position_ratio": 0.60,
        },
    }
    monkeypatch.setattr(
        "arena.agents.adk_context_tools.utc_now",
        lambda: datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc),
    )

    out = tool.portfolio_diagnosis(mdd_days=5, top_n=2)

    assert repo.frame_calls == [
        {
            "tickers": ["QQQ"],
            "start": datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc).date(),
            "end": datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc).date(),
            "sources": tool._sources(),
        },
        {
            "tickers": ["QQQ"],
            "start": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).date(),
            "end": datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc).date(),
            "sources": tool._sources(),
        },
    ]
    assert out["benchmark"]["period_alignment"] == "exact"
    assert out["benchmark"]["portfolio_start_date"] == "2026-03-01"
    assert out["benchmark"]["benchmark_start_date"] == "2026-03-03"
    assert out["benchmark"]["benchmark_end_date"] == "2026-03-27"
    assert out["benchmark"]["agent_return_metric"] == "current_sleeve_pnl_ratio"
    assert out["benchmark"]["comparison_scope"] == "current_sleeve"
    assert out["benchmark"]["currency_basis"] == "KRW"
    assert out["benchmark"]["price_basis"] == "close_price_krw"
    assert out["benchmark"]["source_basis"] == "quote_aware"
    assert out["benchmark"]["agent_return"] == pytest.approx(0.05, abs=1e-6)
    assert out["benchmark"]["excess_return_vs_benchmark"] == pytest.approx(0.05, abs=1e-6)
    assert out["benchmark"]["return"] == pytest.approx(0.0, abs=1e-6)
    assert "alpha_vs_benchmark" not in out["benchmark"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["current_sleeve"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["cumulative"]
    assert "not risk-adjusted alpha" in out["benchmark"]["alpha_definition"]
    assert "quote-aware KRW" in out["benchmark"]["note"]
    assert "2026-03-01 -> 2026-03-03" in out["benchmark"]["note"]
    assert out["benchmark"] == out["benchmarks"]["current_sleeve"]
    assert set(out["benchmarks"]) == {"current_sleeve", "cumulative"}
    cumulative = out["benchmarks"]["cumulative"]
    assert cumulative["comparison_scope"] == "cumulative"
    assert cumulative["portfolio_start_date"] == "2026-01-01"
    assert cumulative["benchmark_start_date"] == "2026-01-02"
    assert cumulative["agent_return_metric"] == "cumulative_pnl_ratio"
    assert cumulative["agent_return"] == pytest.approx(0.30, abs=1e-6)
    assert cumulative["return_krw"] == pytest.approx((100.0 / 90.0) - 1.0, abs=1e-6)
    assert cumulative["excess_return_vs_benchmark"] == pytest.approx(0.30 - ((100.0 / 90.0) - 1.0), abs=1e-6)
    assert "cumulative/TWR" in cumulative["note"]


def test_portfolio_diagnosis_logs_warning_when_mdd_calculation_fails(caplog) -> None:
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForPortfolioDiagnosisRaises()
    tool.settings = load_settings()
    tool.settings.kis_target_market = "nasdaq"
    tool.agent_id = "gpt"
    tool._context = {
        "portfolio": {
            "cash_krw": 1_000.0,
            "positions": {
                "AAPL": {"quantity": 10.0, "avg_price_krw": 100.0},
                "MSFT": {"quantity": 5.0, "avg_price_krw": 200.0},
            },
        },
        "market_features": [
            {"ticker": "AAPL", "close_price_krw": 110.0, "volatility_20d": 0.2, "ret_20d": 0.08, "ret_5d": 0.03},
            {"ticker": "MSFT", "close_price_krw": 208.0, "volatility_20d": 0.1, "ret_20d": 0.04, "ret_5d": 0.01},
        ],
        "risk_policy": {
            "min_cash_buffer_ratio": 0.10,
            "max_position_ratio": 0.60,
        },
    }

    with caplog.at_level("WARNING"):
        out = tool.portfolio_diagnosis(mdd_days=5, top_n=2)

    assert "mdd" not in out
    assert "portfolio diagnosis MDD calculation failed" in caplog.text


def test_compact_portfolio_diagnosis_includes_hrp_allocation_summary() -> None:
    out = _compact_tool_result_for_prompt(
        "portfolio_diagnosis",
        {
            "risk_contribution": [
                {"ticker": "AAPL", "rc": 0.6},
                {"ticker": "MSFT", "rc": 0.4},
            ],
            "concentration_top3": 0.82,
            "hhi": 0.34,
            "momentum_20d_weighted": 0.07,
            "momentum_5d_weighted": 0.02,
            "volatility_20d_weighted": 0.18,
            "mdd": {"days": 60, "value": -0.12},
            "benchmark": {
                "ticker": "SPY",
                "return_krw": 0.05,
                "agent_return": -0.01,
                "alpha_vs_benchmark": -0.06,
                "alpha_definition": "simple excess return: agent_return - benchmark return_krw; not risk-adjusted alpha",
            },
            "benchmarks": {
                "current_sleeve": {
                    "ticker": "SPY",
                    "return_krw": 0.05,
                    "agent_return": -0.01,
                    "alpha_vs_benchmark": -0.06,
                    "alpha_definition": "simple excess return: agent_return - benchmark return_krw; not risk-adjusted alpha",
                },
                "cumulative": {
                    "ticker": "SPY",
                    "return_krw": 0.08,
                    "agent_return": 0.02,
                    "alpha_vs_benchmark": -0.06,
                    "alpha_definition": "simple excess return: agent_return - benchmark return_krw; not risk-adjusted alpha",
                },
            },
            "hrp_allocation": {
                "status": "ready",
                "strategy": "hrp",
                "hrp_cash_weight": 0.10,
                "hrp_concentration_top3": 0.90,
                "hrp_hhi": 0.28,
                "hrp_weights": [
                    {"ticker": "AAPL", "current_weight": 0.52, "hrp_weight": 0.45, "delta_weight": -0.07, "relative_to_current": "lower"},
                    {"ticker": "MSFT", "current_weight": 0.28, "hrp_weight": 0.45, "delta_weight": 0.17, "relative_to_current": "higher"},
                ],
                "weight_deltas": [
                    {"ticker": "AAPL", "relative_to_current": "lower", "delta_weight": -0.07, "current_weight": 0.52, "hrp_weight": 0.45},
                    {"ticker": "MSFT", "relative_to_current": "higher", "delta_weight": 0.17, "current_weight": 0.28, "hrp_weight": 0.45},
                ],
                "projected_mdd": {"days": 252, "value": -0.08},
            },
        },
    )

    assert "rebalance_plan" not in out
    assert out["hrp_allocation"]["strategy"] == "hrp"
    assert out["hrp_allocation"]["hrp_cash_weight"] == 0.10
    assert len(out["hrp_allocation"]["hrp_weights"]) == 2
    assert len(out["hrp_allocation"]["weight_deltas"]) == 2
    assert all("side" not in row for row in out["hrp_allocation"]["weight_deltas"])
    assert all("size_ratio" not in row for row in out["hrp_allocation"]["weight_deltas"])
    assert "alpha_vs_benchmark" not in out["benchmark"]
    assert out["benchmark"]["excess_return_vs_benchmark"] == -0.06
    assert "not risk-adjusted alpha" in out["benchmark"]["alpha_definition"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["current_sleeve"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["cumulative"]
    assert out["benchmarks"]["current_sleeve"]["excess_return_vs_benchmark"] == -0.06


def test_search_peer_lessons_returns_only_compactor_reflections() -> None:
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForPeerLessons()
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.agent_id = "gpt"
    tool.tenant_id = "local"
    tool._vector_store = _VectorStoreForPeerLessons()

    out = tool.search_peer_lessons("concentration risk", limit=5)

    assert len(out) == 1
    assert out[0]["event_id"] == "mem_peer"
    assert out[0]["agent_id"] == "gemini"
    assert out[0]["author_id"] == "gemini"
    assert out[0]["memory_source"] == "thesis_chain_compaction"
    assert tool._vector_store.calls[0]["agent_id"] == "gpt"


def test_get_research_briefing_falls_back_to_public_demo_for_no_key_tenant(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_PUBLIC_DEMO_TENANT", "midnightnnn")
    monkeypatch.delenv("ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)

    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForResearchBriefingFallback()
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.settings.research_enabled = True
    tool.settings.gemini_api_key = ""
    tool.settings.research_gemini_api_key = ""
    tool.settings.research_gemini_source = ""
    tool.settings.research_gemini_source_tenant = ""
    tool.tenant_id = "tenant-a"

    out = tool.get_research_briefing(limit=2)

    assert [row["briefing_id"] for row in out] == ["pub_global", "pub_geo"]
    assert all(row["public_fallback"] is True for row in out)
    assert all(row["source_tenant_id"] == "midnightnnn" for row in out)
    assert tool.repo.calls == [
        {
            "tickers": None,
            "categories": None,
            "limit": 2,
            "trading_mode": "paper",
            "tenant_id": "tenant-a",
        },
        {
            "tickers": None,
            "categories": ["global_market", "geopolitical", "sector_trends"],
            "limit": 2,
            "trading_mode": "paper",
            "tenant_id": "midnightnnn",
        },
    ]


def test_get_research_briefing_does_not_fallback_for_ticker_queries(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_PUBLIC_DEMO_TENANT", "midnightnnn")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)

    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForResearchBriefingFallback()
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.settings.research_enabled = True
    tool.settings.gemini_api_key = ""
    tool.settings.research_gemini_api_key = ""
    tool.settings.research_gemini_source = ""
    tool.settings.research_gemini_source_tenant = ""
    tool.tenant_id = "tenant-a"

    out = tool.get_research_briefing(tickers=["AAPL"], limit=2)

    assert out == []
    assert tool.repo.calls == [
        {
            "tickers": ["AAPL"],
            "categories": None,
            "limit": 2,
            "trading_mode": "paper",
            "tenant_id": "tenant-a",
        }
    ]


def test_compact_tool_result_reddit_drops_url_and_trims_text() -> None:
    out = _compact_tool_result_for_prompt(
        "fetch_reddit_sentiment",
        [
            {
                "title": "AAPL sentiment is ripping higher on wallstreetbets and this title is intentionally very long",
                "subreddit": "wallstreetbets",
                "score": 123,
                "num_comments": 45,
                "created": "2026-03-14T00:00:00+00:00",
                "url": "https://reddit.com/r/x",
                "selftext_snippet": "x" * 400,
            }
        ],
        args={"ticker": "AAPL"},
    )

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["subreddit"] == "wallstreetbets"
    assert "url" not in out[0]
    assert len(out[0]["selftext_snippet"]) <= 140


def test_compact_tool_result_technical_signals_multi_returns_summary_rows() -> None:
    out = _compact_tool_result_for_prompt(
        "technical_signals",
        {
            "tickers": ["AAPL", "MSFT"],
            "count": 2,
            "rows": [
                {
                    "ticker": "AAPL",
                    "price": 100.0,
                    "rsi_14": 61.2,
                    "rsi_state": "neutral",
                    "macd": {"line": 1.0, "signal": 0.8, "hist": 0.2, "state": "bullish"},
                    "moving_averages": {"sma_20": 98.0, "sma_50": 95.0, "price_vs_sma20": 0.0204},
                    "bollinger_20_2": {"upper": 102.0, "mid": 98.0, "lower": 94.0, "state": "inside_bands"},
                    "trend_state": "uptrend",
                }
            ],
        },
    )

    assert out["count"] == 1
    assert out["rows"][0]["ticker"] == "AAPL"
    assert out["rows"][0]["macd_state"] == "bullish"
    assert "macd" not in out["rows"][0]


def test_compact_tool_result_screen_market_keeps_bucket_reason_and_value_fields() -> None:
    out = _compact_tool_result_for_prompt(
        "screen_market",
        [
            {
                "ticker": "PBR",
                "bucket": "value",
                "bucket_rank": 1,
                "score": 2.14,
                "reason": "Valuation support: PER 6.2, PBR 1.1",
                "reason_for": "Valuation support: PER 6.2, PBR 1.1",
                "reason_risk": "Screen-only evidence; confirm first.",
                "ret_20d": 0.11,
                "ret_5d": -0.02,
                "volatility_20d": 0.21,
                "sentiment_score": 0.08,
                "per": 6.2,
                "pbr": 1.1,
                "roe": 18.0,
                "debt_ratio": 72.0,
                "close_price_krw": 18340.0,
            }
        ],
    )

    assert out[0]["ticker"] == "PBR"
    assert out[0]["bucket"] == "value"
    assert out[0]["reason"].startswith("Valuation support")
    assert out[0]["reason_for"].startswith("Valuation support")
    assert out[0]["reason_risk"] == "Screen-only evidence; confirm first."
    assert out[0]["per"] == 6.2
    assert out[0]["pbr"] == 1.1


def test_build_memory_query_screen_market_mentions_buckets_and_tickers() -> None:
    query = build_memory_query(
        "screen_market",
        {},
        [
            {"ticker": "PBR", "bucket": "value"},
            {"ticker": "MRVL", "bucket": "momentum"},
            {"ticker": "DUK", "bucket": "defensive"},
        ],
    )

    assert query == "market screening value momentum defensive PBR MRVL DUK"


def test_compact_tool_result_get_fundamentals_reduces_meta_lists() -> None:
    out = _compact_tool_result_for_prompt(
        "get_fundamentals",
        {
            "requested": ["AAPL", "MSFT", "XYZ"],
            "eligible": ["AAPL", "MSFT"],
            "excluded": ["XYZ"],
            "rows": [
                {"ticker": "AAPL", "market": "us", "per": 31.5, "pbr": 45.2, "eps": 6.38, "currency": "USD", "exchange": "NAS"},
                {"ticker": "MSFT", "market": "us", "per": 34.0, "pbr": 12.1, "eps": 12.5, "currency": "USD", "exchange": "NAS"},
            ],
            "errors": [{"ticker": "XYZ", "error": "ticker not found in upstream fundamentals payload"}],
        },
    )

    assert out["requested_count"] == 3
    assert out["eligible_count"] == 2
    assert out["excluded_count"] == 1
    assert out["excluded"] == ["XYZ"]
    assert out["rows"][0]["ticker"] == "AAPL"
    assert out["errors"][0]["ticker"] == "XYZ"


class _RepoForAdkGenerate:
    def latest_market_features(self, tickers, limit, sources=None):
        _ = (tickers, limit, sources)
        return []


class _FakeRunner:
    def __init__(self) -> None:
        self.board_calls: list[tuple[str, str, str]] = []

    def decide_orders(self, *, context, default_universe, resume_session_id=None):
        _ = (context, default_universe, resume_session_id)
        return (
            {
                "orders": [
                    {
                        "ticker": "AAPL",
                        "side": "BUY",
                        "size_ratio": 0.5,
                        "rationale": "fx repricing",
                    }
                ],
                "board_title": "draft",
                "board_body": "draft",
            },
            "sid_1",
        )

    def decide_board(self, session_id, orders_summary, *, cycle_id=""):
        self.board_calls.append((session_id, orders_summary, cycle_id))
        return {"board_title": "confirmed", "board_body": orders_summary}


class _FakeKospiRunner(_FakeRunner):
    def decide_orders(self, *, context, default_universe, resume_session_id=None):
        _ = (context, default_universe, resume_session_id)
        return (
            {
                "orders": [
                    {
                        "ticker": "025860",
                        "side": "BUY",
                        "size_ratio": 0.2,
                        "rationale": "momentum continuation",
                    }
                ],
                "board_title": "draft",
                "board_body": "draft",
            },
            "sid_kospi_1",
        )

    def decide_board(self, session_id, orders_summary, *, cycle_id=""):
        self.board_calls.append((session_id, orders_summary, cycle_id))
        return {
            "board_title": "이녹스첨단소재를 다시 담다",
            "board_body": "**이녹스첨단소재(025860)** BUY 48주 체결\n전날 27주에 이어 오늘 48주.",
        }


class _FailRunner:
    def decide_orders(self, *, context, default_universe, resume_session_id=None):
        _ = (context, default_universe, resume_session_id)
        raise RuntimeError("runner boom")


def test_generate_reprices_us_order_with_live_fx(monkeypatch) -> None:
    runner = _FakeRunner()
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: runner)
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    out = agent.generate(
        {
            "cycle_phase": "execution",
            "cycle_id": "cycle_fx_1",
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "usd_krw_rate": 1450.0,
                "positions": {},
            },
            "market_features": [
                {
                    "ticker": "AAPL",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:AAPL",
                    "close_price_krw": 130000.0,
                    "close_price_native": 100.0,
                    "quote_currency": "USD",
                    "fx_rate_used": 1300.0,
                }
            ],
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        }
    )

    assert len(out.intents) == 1
    intent = out.intents[0]
    assert intent.price_krw == pytest.approx(145000.0)
    assert intent.price_native == pytest.approx(100.0)
    assert intent.quote_currency == "USD"
    assert intent.fx_rate == pytest.approx(1450.0)
    assert runner.board_calls == []


def test_generate_raises_when_market_features_missing(monkeypatch) -> None:
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: _FakeRunner())
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    with pytest.raises(RuntimeError, match="market_features missing"):
        agent.generate({"cycle_phase": "execution", "cycle_id": "cycle_missing_rows", "market_features": []})


def test_generate_raises_when_decision_fails(monkeypatch) -> None:
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: _FailRunner())
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    with pytest.raises(RuntimeError, match="ADK decision failed"):
        agent.generate(
            {
                "cycle_phase": "execution",
                "cycle_id": "cycle_decision_fail",
                "market_features": [
                    {
                        "ticker": "AAPL",
                        "exchange_code": "NASD",
                        "instrument_id": "NASD:AAPL",
                        "close_price_krw": 130000.0,
                        "close_price_native": 100.0,
                        "quote_currency": "USD",
                        "fx_rate_used": 1300.0,
                    }
                ],
            }
        )


def test_finalize_board_post_uses_execution_summary(monkeypatch) -> None:
    runner = _FakeRunner()
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: runner)
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    out = agent.generate(
        {
            "cycle_phase": "execution",
            "cycle_id": "cycle_fx_2",
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "usd_krw_rate": 1450.0,
                "positions": {},
            },
            "market_features": [
                {
                    "ticker": "AAPL",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:AAPL",
                    "close_price_krw": 130000.0,
                    "close_price_native": 100.0,
                    "quote_currency": "USD",
                    "fx_rate_used": 1300.0,
                }
            ],
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        }
    )

    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_1",
        filled_qty=1.0,
        avg_price_krw=145145.0,
        avg_price_native=100.1,
        quote_currency="USD",
        fx_rate=1450.0,
        message="confirmed",
        created_at=utc_now(),
    )

    post = agent.finalize_board_post(
        cycle_id="cycle_fx_2",
        initial_post=BoardPost(
            agent_id="gpt",
            title="placeholder",
            body="pending",
            tickers=["AAPL"],
            cycle_id="cycle_fx_2",
        ),
        intents=out.intents,
        reports=[report],
    )

    assert len(runner.board_calls) == 1
    _, summary, board_cycle_id = runner.board_calls[0]
    assert "실제 실행 결과" in summary
    assert "AAPL BUY 1주 FILLED" in summary
    assert board_cycle_id == "cycle_fx_2"
    assert post.body == summary


def test_finalize_board_post_keeps_freeform_board_text(monkeypatch) -> None:
    runner = _FakeKospiRunner()
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: runner)
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "kospi"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    settings.default_universe = ["025860"]

    agent = AdkTradingAgent(
        agent_id="claude",
        provider="claude",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    out = agent.generate(
        {
            "cycle_phase": "execution",
            "cycle_id": "cycle_kospi_1",
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_500_000.0,
                "positions": {
                    "025860": {
                        "quantity": 27.0,
                        "avg_price_krw": 8290.0,
                        "market_price_krw": 8270.0,
                        "ticker_name": "남해화학",
                    }
                },
            },
            "market_features": [
                {
                    "ticker": "025860",
                    "exchange_code": "KRX",
                    "instrument_id": "KRX:025860",
                    "close_price_krw": 8270.0,
                    "close_price_native": 8270.0,
                    "quote_currency": "KRW",
                    "fx_rate_used": 1.0,
                }
            ],
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        }
    )

    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_kospi_1",
        filled_qty=48.0,
        avg_price_krw=8290.0,
        quote_currency="KRW",
        fx_rate=1.0,
        message="confirmed",
        created_at=utc_now(),
    )

    post = agent.finalize_board_post(
        cycle_id="cycle_kospi_1",
        initial_post=out.board_post,
        intents=out.intents,
        reports=[report],
    )

    assert len(runner.board_calls) == 1
    assert runner.board_calls[0][2] == "cycle_kospi_1"
    assert post.title == "이녹스첨단소재를 다시 담다"
    assert "**이녹스첨단소재(025860)** BUY 48주 체결" in post.body
    assert "전날 27주에 이어 오늘 48주." in post.body


def test_generate_skips_mixed_us_order_when_exchange_is_unresolved(monkeypatch) -> None:
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: _FakeRunner())
    settings = load_settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "us"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    out = agent.generate(
        {
            "cycle_phase": "execution",
            "cycle_id": "cycle_fx_2",
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "usd_krw_rate": 1450.0,
                "positions": {},
            },
            "market_features": [
                {
                    "ticker": "AAPL",
                    "exchange_code": "",
                    "instrument_id": "",
                    "close_price_krw": 130000.0,
                    "close_price_native": 100.0,
                    "quote_currency": "USD",
                    "fx_rate_used": 1300.0,
                }
            ],
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        }
    )

    assert out.intents == []


class _MemoryStoreForToolSummary:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_memory(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _MemoryStoreForCandidateMemory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_candidate_memories(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return 1


class _MemoryStoreForManualNote:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_manual_note(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _VectorStoreForToolMemory:
    def search_similar_memories(self, **kwargs):
        _ = kwargs
        from datetime import datetime

        return [
            {
                "summary": "Macro-sensitive trim discipline mattered.",
                "importance_score": 0.8,
                "created_at": datetime.fromisoformat("2026-03-05T00:00:00+00:00"),
                "outcome_score": 0.8,
            }
        ]


class _VectorStoreForDedupedToolMemory:
    def search_similar_memories(self, **kwargs):
        _ = kwargs
        from datetime import datetime

        return [
            {
                "event_id": "mem_seen",
                "summary": "Already injected lesson.",
                "importance_score": 0.9,
                "created_at": datetime.fromisoformat("2026-03-05T00:00:00+00:00"),
                "outcome_score": 0.8,
            },
            {
                "event_id": "mem_new",
                "summary": "Fresh trim discipline lesson.",
                "importance_score": 0.7,
                "created_at": datetime.fromisoformat("2026-03-04T00:00:00+00:00"),
                "outcome_score": 0.2,
            },
        ]


class _MemoryStoreForToolMemory:
    def __init__(self) -> None:
        self.vector_store = _VectorStoreForToolMemory()

    def _tenant(self) -> str:
        return "local"


class _MemoryStoreForDedupedToolMemory:
    def __init__(self) -> None:
        self.vector_store = _VectorStoreForDedupedToolMemory()

    def _tenant(self) -> str:
        return "local"


class _RepoForToolSummary:
    def __init__(self) -> None:
        self.events = []

    def write_memory_event(self, event) -> None:
        self.events.append(event)


def test_persist_tool_summary_memory_prefers_memory_store() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForToolSummary()
    runner.repo = _RepoForToolSummary()

    runner._persist_tool_summary_memory(
        summary="ReAct tools used (draft): 2",
        payload={
            "tool_events": [{"tool": "technical_signals"}],
            "phase": "draft",
            "token_usage": {"llm_calls": 2, "prompt_tokens": 1200, "completion_tokens": 180, "total_tokens": 1380},
        },
    )

    assert len(runner._memory_store.calls) == 1
    call = runner._memory_store.calls[0]
    assert call["agent_id"] == "gpt"
    assert call["event_type"] == "react_tools_summary"
    assert call["score"] == pytest.approx(0.6)
    assert call["payload"]["token_usage"]["total_tokens"] == 1380
    assert runner.repo.events == []


def test_persist_candidate_memories_uses_candidate_ledger() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner._memory_store = _MemoryStoreForCandidateMemory()
    runner._candidate_ledger = {
        "MSFT": {
            "source_tools": {"screen_market:value"},
            "discovery_count": 1,
            "last_seen_rank": 2,
            "discovery_evidence": {"reason_for": "Valuation support"},
        }
    }
    runner._held_tickers_cache = {"AAPL"}
    runner._current_phase = "execution"

    written = runner._persist_candidate_memories(cycle_id="cycle_candidate")

    assert written == 1
    call = runner._memory_store.calls[0]
    assert call["agent_id"] == "gpt"
    assert call["held_tickers"] == {"AAPL"}
    assert call["cycle_id"] == "cycle_candidate"
    assert call["phase"] == "execution"
    assert "MSFT" in call["candidate_ledger"]


def test_decide_orders_keeps_tool_events_reference_for_wrapped_tools(monkeypatch) -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    shared_tool_events = [{"tool": "stale_event"}]
    runner._tool_events = shared_tool_events
    runner._seen_memory_ids = set()
    runner._candidate_ledger = {}
    runner._current_phase = "unknown"
    runner._current_context = None
    runner._held_tickers_cache = set()
    runner._session_id = "sid_base"
    runner._max_tool_events = 5
    runner._run_config = object()
    runner._runner = object()
    runner._user_id = "arena"
    runner.agent_id = "gpt"
    runner._registry = SimpleNamespace(set_context=lambda context: None)
    runner._toolbox = SimpleNamespace(set_context=lambda context: None)
    runner._memory_store = None
    runner._seed_seen_memory_ids = lambda context: None
    runner._extract_held_tickers = lambda context: set()
    runner._sync_pipeline_context = lambda: None
    runner._funnel_metrics = lambda: {}
    runner._persist_tool_summary_memory = lambda *, summary, payload: None
    runner._run_on_loop = lambda value: value

    def _fake_run_async(_runner, session_id, prompt):
        _ = (_runner, session_id, prompt)
        shared_tool_events.append(
            {
                "tool": "technical_signals",
                "args": {"ticker": "AAPL"},
                "result": {"ticker": "AAPL", "trend_state": "uptrend"},
            }
        )
        return '{"orders": []}'

    runner._run_async = _fake_run_async

    monkeypatch.setattr(
        "arena.agents.adk_agents.prepare_decision_prompt",
        lambda *args, **kwargs: ("sid_test", "prompt", False),
    )
    monkeypatch.setattr("arena.agents.adk_agents.parse_decision_response", lambda text: {"orders": []})
    monkeypatch.setattr("arena.agents.adk_agents.tag_phase_tool_events", lambda *args, **kwargs: None)

    captured: dict[str, object] = {}

    def _capture_summary(tool_events, **kwargs):
        _ = kwargs
        captured["tool_names"] = [str(event.get("tool") or "") for event in tool_events]
        captured["tool_events_id"] = id(tool_events)
        return None

    monkeypatch.setattr("arena.agents.adk_agents.build_tool_summary_memory_record", _capture_summary)

    decision, session_id = runner.decide_orders({"cycle_phase": "execution"}, [])

    assert decision == {"orders": []}
    assert session_id == "sid_test"
    assert runner._tool_events is shared_tool_events
    assert captured["tool_events_id"] == id(shared_tool_events)
    assert captured["tool_names"] == ["technical_signals"]


class _AsyncRunnerForResponseCollection:
    async def run_async(self, *, user_id, session_id, new_message, run_config):
        _ = (user_id, session_id, new_message, run_config)
        yield SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=120,
                candidates_token_count=25,
                cached_content_token_count=60,
                thoughts_token_count=5,
            ),
            content=SimpleNamespace(
                parts=[
                    SimpleNamespace(
                        function_call=SimpleNamespace(name="remote_macro_tool", args={"ticker": "AAPL"}),
                        text=None,
                    )
                ]
            ),
        )
        yield SimpleNamespace(
            usage_metadata=None,
            content=SimpleNamespace(
                parts=[
                    SimpleNamespace(function_call=None, text='{"orders": []}'),
                ]
            ),
        )


def test_collect_response_text_records_mcp_calls_and_token_usage() -> None:
    tool_events: list[dict] = []

    text, token_usage = asyncio.run(
        collect_response_text(
            runner=_AsyncRunnerForResponseCollection(),
            user_id="arena",
            session_id="sid_1",
            prompt="cycle_phase: execution",
            run_config=object(),
            max_tool_events=5,
            wrapped_tool_names={"search_past_experiences"},
            tool_events=tool_events,
            agent_id="gpt",
        )
    )

    assert text == '{"orders": []}'
    assert token_usage["llm_calls"] == 1
    assert token_usage["prompt_tokens"] == 120
    assert token_usage["completion_tokens"] == 25
    assert token_usage["cached_tokens"] == 60
    assert token_usage["thinking_tokens"] == 5
    assert tool_events == [
        {
            "tool": "remote_macro_tool",
            "args": {"ticker": "AAPL"},
            "elapsed_ms": 0,
            "result_preview": None,
            "error": None,
            "source": "mcp",
        }
    ]


def test_save_memory_tool_creates_manual_note() -> None:
    tool = _ContextTools.__new__(_ContextTools)
    tool.agent_id = "gpt"
    tool._memory_store = _MemoryStoreForManualNote()

    out = tool.save_memory("Opening auction looked disorderly.", score=0.65)

    assert out["status"] == "saved"
    assert out["event_type"] == "manual_note"
    assert len(tool._memory_store.calls) == 1
    call = tool._memory_store.calls[0]
    assert call["agent_id"] == "gpt"
    assert call["summary"] == "Opening auction looked disorderly."
    assert call["score"] == pytest.approx(0.65)


def test_search_tool_memories_includes_created_date() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForToolMemory()
    runner._seen_memory_ids = set()

    rows = runner._search_tool_memories("macro regime trim discipline")

    assert rows is not None
    assert rows[0]["created_date"] == "2026-03-05"
    assert rows[0]["created_at"].startswith("2026-03-05T00:00:00")
    assert rows[0]["outcome_label"] == "win"


def test_search_tool_memories_skips_initially_injected_event_ids() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForDedupedToolMemory()
    runner._seen_memory_ids = {"mem_seen"}

    rows = runner._search_tool_memories("trim discipline")

    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["summary"] == "Fresh trim discipline lesson."
    assert "mem_new" in runner._seen_memory_ids


def test_seed_seen_memory_ids_uses_initial_context_memory_rows() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._seen_memory_ids = set()

    runner._seed_seen_memory_ids(
        {
            "memory_events": [
                {"event_id": "mem_a"},
                {"event_id": "mem_b"},
                {"summary": "no id"},
            ]
        }
    )

    assert runner._seen_memory_ids == {"mem_a", "mem_b"}


# ──────────────────────────────────────────────────
# Per-agent: _resolve_disabled_tool_ids
# ──────────────────────────────────────────────────

def test_resolve_disabled_tool_ids_uses_agent_config_override() -> None:
    repo = _RepoForTools('["tool_a","tool_b"]')
    ac = AgentConfig(
        agent_id="custom",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        disabled_tools=["tool_x"],
    )
    result = _resolve_disabled_tool_ids(repo, "tenant-a", ac)
    assert result == {"tool_x"}


def test_resolve_disabled_tool_ids_falls_back_to_global() -> None:
    repo = _RepoForTools('["tool_a"]')
    ac = AgentConfig(
        agent_id="custom",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        disabled_tools=None,
    )
    result = _resolve_disabled_tool_ids(repo, "tenant-a", ac)
    assert result == {"tool_a"}


def test_resolve_disabled_tool_ids_without_agent_config() -> None:
    repo = _RepoForTools('["tool_a"]')
    result = _resolve_disabled_tool_ids(repo, "tenant-a", None)
    assert result == {"tool_a"}


# ──────────────────────────────────────────────────
# Per-agent: _has_credentials
# ──────────────────────────────────────────────────

def test_has_credentials_gpt() -> None:
    s = load_settings()
    s.openai_api_key = "sk-test"
    assert _has_credentials("gpt", s) is True

    s.openai_api_key = ""
    assert _has_credentials("gpt", s) is False


def test_has_credentials_claude() -> None:
    s = load_settings()
    s.anthropic_api_key = "ak-test"
    s.anthropic_use_vertexai = False
    assert _has_credentials("claude", s) is True

    s.anthropic_api_key = ""
    s.anthropic_use_vertexai = True
    assert _has_credentials("claude", s) is True

    s.anthropic_use_vertexai = False
    assert _has_credentials("claude", s) is False


def test_has_credentials_unknown() -> None:
    s = load_settings()
    assert _has_credentials("unknown", s) is False


def test_resolve_model_openai_uses_instance_scoped_api_key() -> None:
    settings = load_settings()
    settings.openai_api_key = "tenant-openai"

    model = _resolve_model("gpt", settings, model_override="gpt-5.4")

    assert model.model == "openai/gpt-5.4"
    assert model._additional_args["api_key"] == "tenant-openai"


def test_resolve_model_claude_direct_uses_instance_scoped_api_key() -> None:
    settings = load_settings()
    settings.anthropic_api_key = "tenant-anthropic"
    settings.anthropic_use_vertexai = False
    settings.anthropic_model = "claude-sonnet-4-6"

    model = _resolve_model("claude", settings)

    assert model.model == "anthropic/claude-sonnet-4-6"
    assert model._additional_args["api_key"] == "tenant-anthropic"
    assert model._additional_args["cache_control_injection_points"] == [
        {"location": "message", "role": "system"},
    ]


def test_resolve_model_deepseek_uses_provider_payload_api_key_and_base_url() -> None:
    settings = load_settings()
    settings.provider_secrets = {
        "deepseek": {
            "api_key": "tenant-deepseek",
            "model": "deepseek-chat",
            "base_url": "https://custom.deepseek/v1",
        }
    }

    model = _resolve_model("deepseek", settings)

    assert model.model == "deepseek/deepseek-chat"
    assert model._additional_args["api_key"] == "tenant-deepseek"
    assert model._additional_args["base_url"] == "https://custom.deepseek/v1"
