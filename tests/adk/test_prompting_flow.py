from __future__ import annotations

import json

import pytest

from arena.agents.adk_agent_flow import (
    explore_phase_output,
    extract_decision_payload,
    retry_policy_from_env,
)
from arena.agents.adk_agents import _ADKDecisionRunner, _system_prompt, _user_prompt
from arena.agents.adk_decision_flow import (
    build_tool_summary_memory_record,
    parse_board_response,
    prepare_decision_prompt,
)
from arena.config import AgentConfig
from arena.tools.registry import ToolRegistry


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
    assert "Global prompt from DB." not in out


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


def test_retry_policy_from_env_clamps_extreme_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_ADK_RETRY_MAX", "99")
    monkeypatch.setenv("ARENA_ADK_RETRY_BACKOFF_SECONDS", "99")

    retry_limit, retry_delay = retry_policy_from_env()

    assert retry_limit == 4
    assert retry_delay == 10.0


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
