from __future__ import annotations

from arena.agents.adk_runner_bootstrap import resolve_max_tool_events, runner_identity
from arena.config import load_settings


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
