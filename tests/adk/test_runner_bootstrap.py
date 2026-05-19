from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from arena.agents.adk_agents import _ADKDecisionRunner
from arena.agents.adk_runner_bootstrap import build_tool_wrapper, resolve_max_tool_events, runner_identity
from arena.config import load_settings
from arena.tools.registry import ToolEntry


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


def test_tool_wrapper_returns_runtime_clock_without_polluting_tool_event() -> None:
    def sample_tool(*, ticker: str) -> dict:
        return {"ticker": ticker, "score": 1}

    tool_events: list[dict] = []
    wrapper = build_tool_wrapper(
        ToolEntry(
            tool_id="sample_tool",
            name="sample_tool",
            description="Sample tool",
            category="test",
            callable=sample_tool,
        ),
        settings=load_settings(),
        agent_id="gpt",
        tool_events=tool_events,
        update_candidate_ledger=lambda name, args, result: None,
        search_tool_memories=lambda query: None,
        apply_tool_schema_metadata=lambda fn, **kwargs: fn,
    )

    result = asyncio.run(wrapper(ticker="005930"))

    assert result["ticker"] == "005930"
    assert "_runtime_clock" in result
    assert "now_kst" in result["_runtime_clock"]
    assert "_runtime_clock" not in tool_events[-1]["result"]
    assert tool_events[-1]["model_visible_result"]["_runtime_clock"] == result["_runtime_clock"]


def test_tool_event_audit_persists_model_visible_result_when_present() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    clock = {"now_kst": "2026-05-15T15:26:18+09:00"}

    rows = runner._tool_event_rows_for_audit(
        llm_call_id="llm_1",
        events=[
            {
                "tool": "sample_tool",
                "args": {"ticker": "005930"},
                "result": {"ticker": "005930"},
                "model_visible_result": {"ticker": "005930", "_runtime_clock": clock},
            }
        ],
        phase="explore",
        cycle_id="cycle_1",
        default_created_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
    )

    assert rows[0]["model_visible_result_json"] == {
        "ticker": "005930",
        "_runtime_clock": clock,
    }
