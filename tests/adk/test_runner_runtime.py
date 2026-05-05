from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from arena.agents.adk_agents import _ADKDecisionRunner
from arena.agents.adk_runner_runtime import AdkToolBudgetExceeded, collect_response_text


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


class _AsyncRunnerExceedsToolBudget:
    async def run_async(self, *, user_id, session_id, new_message, run_config):
        _ = (user_id, session_id, new_message, run_config)
        for ticker in ("AAPL", "MSFT"):
            yield SimpleNamespace(
                usage_metadata=None,
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(name="remote_macro_tool", args={"ticker": ticker}),
                            text=None,
                        )
                    ]
                ),
            )


class _AsyncRunnerBudgetThenFinal:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def run_async(self, *, user_id, session_id, new_message, run_config):
        _ = (user_id, session_id, run_config)
        prompt = str(getattr(new_message.parts[0], "text", "") or "")
        self.prompts.append(prompt)
        if "도구 호출 예산이 끝났습니다" in prompt:
            yield SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=40,
                    candidates_token_count=12,
                    cached_content_token_count=0,
                    thoughts_token_count=0,
                ),
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(function_call=None, text='{"explore_summary":"budget closed"}'),
                    ]
                ),
            )
            return
        for ticker in ("AAPL", "MSFT"):
            yield SimpleNamespace(
                usage_metadata=None,
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(name="remote_macro_tool", args={"ticker": ticker}),
                            text=None,
                        )
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
    assert token_usage["tool_calls"] == 1
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


def test_collect_response_text_aborts_when_tool_budget_exceeded() -> None:
    tool_events: list[dict] = []

    with pytest.raises(AdkToolBudgetExceeded):
        asyncio.run(
            collect_response_text(
                runner=_AsyncRunnerExceedsToolBudget(),
                user_id="arena",
                session_id="sid_1",
                prompt="cycle_phase: execution",
                run_config=object(),
                max_tool_events=1,
                wrapped_tool_names=set(),
                tool_events=tool_events,
                agent_id="gpt",
            )
        )

    assert [event["args"]["ticker"] for event in tool_events] == ["AAPL", "MSFT"]


def test_run_async_requests_final_json_when_tool_budget_exceeded() -> None:
    decision_runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    decision_runner._user_id = "arena"
    decision_runner._run_config = object()
    decision_runner._max_tool_events = 1
    decision_runner._wrapped_tool_names = set()
    decision_runner._tool_events = []
    decision_runner.agent_id = "gpt"
    decision_runner.provider = "gpt"
    decision_runner._current_phase = "explore"
    decision_runner.settings = SimpleNamespace(timeout_for=lambda role: 10)
    fake_runner = _AsyncRunnerBudgetThenFinal()

    text = asyncio.run(decision_runner._run_async(fake_runner, "sid_1", "initial prompt"))

    assert text == '{"explore_summary":"budget closed"}'
    assert len(fake_runner.prompts) == 2
    assert "더 이상 도구를 호출하지 마십시오" in fake_runner.prompts[1]
