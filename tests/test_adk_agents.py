from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from arena.agents.adk_agents import (
    _ADKDecisionRunner,
    _apply_tool_schema_metadata,
    _ContextTools,
    _is_retryable_adk_error,
    _load_disabled_tool_ids,
    _system_prompt,
    _user_prompt,
)
from arena.agents.adk_agent_flow import (
    explore_phase_output,
    extract_decision_payload,
    retry_policy_from_env,
)
from arena.agents.adk_decision_flow import (
    build_tool_summary_memory_record,
    parse_board_response,
    prepare_decision_prompt,
)
from arena.agents.adk_runner_bootstrap import build_tool_wrapper, resolve_max_tool_events, runner_identity
from arena.agents.adk_runner_runtime import AdkToolBudgetExceeded
from arena.config import AgentConfig, load_settings
from arena.models import OrderIntent, Side
from arena.tools.default_registry import build_default_registry
from arena.tools.registry import ToolEntry, ToolRegistry
from arena.tools.scratch_workspace import ScratchWorkspace


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


def test_batch_default_tool_schema_preserves_required_fields_and_enums() -> None:
    from google.adk.tools.function_tool import FunctionTool

    from arena.agents.adk_tool_helpers import noop_search_tool_memories, noop_update_candidate_ledger

    settings = load_settings()
    repo = SimpleNamespace(get_config=lambda *args, **kwargs: "")
    registry = build_default_registry(repo, settings, tenant_id="local")
    scratch = ScratchWorkspace(agent_id="gpt", tenant_id="local", tool_events=[])
    registry.bind("scratch_run_python", scratch.run_python)

    def declaration(tool_id: str):
        entry = registry.get(tool_id)
        assert entry is not None
        assert entry.callable is not None
        wrapped = build_tool_wrapper(
            entry,
            settings=settings,
            agent_id="gpt",
            tool_events=[],
            update_candidate_ledger=noop_update_candidate_ledger,
            search_tool_memories=noop_search_tool_memories,
            apply_tool_schema_metadata=_apply_tool_schema_metadata,
        )
        return FunctionTool(wrapped)._get_declaration()

    screen_params = declaration("screen_market").parameters.model_dump(mode="json", exclude_none=True)
    assert screen_params["properties"]["bucket"]["enum"] == [
        "auto",
        "balanced",
        "momentum",
        "pullback",
        "recovery",
        "defensive",
        "value",
    ]
    assert screen_params["properties"]["sort_by"]["enum"] == [
        "none",
        "as_of_ts",
        "ret_20d",
        "ret_5d",
        "volatility_20d",
        "sentiment_score",
        "close_price_krw",
    ]
    assert screen_params["properties"]["order"]["enum"] == ["asc", "desc"]

    optimize_params = declaration("optimize_portfolio").parameters.model_dump(mode="json", exclude_none=True)
    assert "tickers" in optimize_params["required"]
    assert optimize_params["properties"]["strategy"]["enum"] == ["sharpe", "risk_parity", "forecast"]
    assert optimize_params["properties"]["forecast_mode"]["enum"] == [
        "default",
        "all",
        "stacked",
        "base",
        "balanced",
        "lgbm",
        "ridge",
        "avg",
    ]

    opportunity_params = declaration("recommend_opportunities").parameters.model_dump(mode="json", exclude_none=True)
    assert opportunity_params["properties"]["buckets"]["items"]["enum"] == ["momentum", "pullback", "recovery"]
    assert opportunity_params["properties"]["profiles"]["items"]["enum"] == [
        "aggressive",
        "balanced",
        "defensive",
        "value",
        "tactical",
        "tactical_leverage",
        "tactical_inverse",
        "tactical_hedge",
    ]

    scratch_params = declaration("scratch_run_python").parameters.model_dump(mode="json", exclude_none=True)
    assert "code" in scratch_params["required"]
    assert scratch_params["properties"]["inputs"]["type"] == "OBJECT"
    assert scratch_params["properties"]["inputs"]["nullable"] is True


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
        phase="explore",
        analysis_funnel={"discovered_nonheld": 0},
        cycle_id="cycle_1",
        token_usage={"llm_calls": 1, "prompt_tokens": 120},
    )

    assert record is not None
    summary, payload = record
    assert "ReAct tools used (explore): 0" in summary
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


def test_adk_own_timeout_is_not_retryable() -> None:
    assert _is_retryable_adk_error(TimeoutError("ADK coroutine timed out after 1530s")) is False
    assert _is_retryable_adk_error(TimeoutError("ADK tool-budget finalization timed out after 60s")) is False
    assert _is_retryable_adk_error(AdkToolBudgetExceeded("ADK tool budget exceeded after 121 tool calls")) is False
    assert _is_retryable_adk_error(RuntimeError("429 RESOURCE_EXHAUSTED")) is True
    assert _is_retryable_adk_error(RuntimeError("litellm.BadGatewayError: 502 Bad gateway")) is True


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


def test_extract_decision_payload_normalizes_non_list_orders() -> None:
    explore_summary, orders = extract_decision_payload(
        {
            "explore_summary": "  concise explore  ",
            "orders": {"ticker": "AAPL"},
        }
    )

    assert explore_summary == "concise explore"
    assert orders == []


def test_explore_phase_output_uses_distinct_tickers() -> None:
    out = explore_phase_output(
        agent_id="gpt",
        cycle_id="cycle_explore_1",
        explore_summary="summary",
        orders=[
            {"ticker": "AAPL"},
            {"ticker": "MSFT"},
            {"ticker": "AAPL"},
        ],
        share_summary=True,
    )

    assert out.intents == []
    assert out.board_post.title == "탐색 요약"
    assert out.board_post.body == "summary"
    assert out.board_post.explore_summary == "summary"
    assert out.board_post.tickers == ["AAPL", "MSFT"]

