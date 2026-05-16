from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from arena.agents.adk_agents import _ADKDecisionRunner, _is_retryable_adk_error
from arena.agents.cycle_supervisor import AgentCycleSupervisor
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


def _callback_decision_runner() -> _ADKDecisionRunner:
    decision_runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    decision_runner.tenant_id = "local"
    decision_runner.agent_id = "gpt"
    decision_runner.provider = "gpt"
    decision_runner._current_phase = "explore"
    decision_runner._current_context = {"cycle_id": "cycle_1"}
    decision_runner._latest_llm_call_id = "call_1"
    return decision_runner


def test_model_callbacks_accept_adk_keyword_arguments() -> None:
    decision_runner = _callback_decision_runner()
    request = SimpleNamespace(contents=[], config=SimpleNamespace(tools=[]))
    response = SimpleNamespace(content=SimpleNamespace(parts=[]), usage_metadata=None)

    decision_runner._before_model_callback(callback_context=object(), llm_request=request)
    decision_runner._after_model_callback(callback_context=object(), llm_response=response)
    decision_runner._on_model_error_callback(
        callback_context=object(),
        llm_request=request,
        exception=RuntimeError("boom"),
    )


def test_model_callbacks_accept_legacy_positional_arguments() -> None:
    decision_runner = _callback_decision_runner()
    request = SimpleNamespace(contents=[], config=SimpleNamespace(tools=[]))
    response = SimpleNamespace(content=SimpleNamespace(parts=[]), usage_metadata=None)

    decision_runner._before_model_callback(object(), request)
    decision_runner._after_model_callback(object(), response)
    decision_runner._on_model_error_callback(object(), request, RuntimeError("boom"))


def test_model_timeout_policy_logs_supervisor_decision(caplog: pytest.LogCaptureFixture) -> None:
    decision_runner = _callback_decision_runner()
    decision_runner._cycle_supervisor = AgentCycleSupervisor(cycle_id="cycle_1")

    with caplog.at_level(logging.INFO, logger="arena.agents.adk_agents"):
        timeout = decision_runner._model_call_timeout_seconds("anthropic/claude-opus-4-7")

    assert timeout == 300
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "agent_cycle_supervisor_model_timeout_policy"
    ]
    assert len(records) == 1
    assert records[0].cycle_id == "cycle_1"
    assert records[0].agent_id == "gpt"
    assert records[0].provider == "gpt"
    assert records[0].phase == "explore"
    assert records[0].llm_call_id == "call_1"
    assert records[0].model == "anthropic/claude-opus-4-7"
    assert records[0].timeout_seconds == 300


def test_model_callbacks_log_supervisor_operation_for_native_models(caplog: pytest.LogCaptureFixture) -> None:
    decision_runner = _callback_decision_runner()
    decision_runner.provider = "gemini"
    decision_runner._cycle_supervisor = AgentCycleSupervisor(cycle_id="cycle_1")
    decision_runner._supervisor_model_operations_by_llm_call_id = {}
    request = SimpleNamespace(contents=[], config=SimpleNamespace(tools=[]))
    response = SimpleNamespace(content=SimpleNamespace(parts=[]), usage_metadata=None)

    with caplog.at_level(logging.INFO, logger="arena.agents.cycle_supervisor"):
        decision_runner._before_model_callback(callback_context=object(), llm_request=request)
        decision_runner._after_model_callback(callback_context=object(), llm_response=response)

    records = [
        record
        for record in caplog.records
        if getattr(record, "event", "").startswith("agent_cycle_supervisor_operation_")
    ]
    assert [record.event for record in records] == [
        "agent_cycle_supervisor_operation_start",
        "agent_cycle_supervisor_operation_finish",
    ]
    assert records[0].cycle_id == "cycle_1"
    assert records[0].agent_id == "gpt"
    assert records[0].provider == "gemini"
    assert records[0].phase == "explore"
    assert records[0].llm_call_id == "call_1"
    assert records[1].status == "success"
    assert records[1].llm_call_id == "call_1"


def test_model_error_callback_finishes_supervisor_operation_as_error(caplog: pytest.LogCaptureFixture) -> None:
    decision_runner = _callback_decision_runner()
    decision_runner.provider = "gemini"
    decision_runner._cycle_supervisor = AgentCycleSupervisor(cycle_id="cycle_1")
    decision_runner._supervisor_model_operations_by_llm_call_id = {}
    request = SimpleNamespace(contents=[], config=SimpleNamespace(tools=[]))

    with caplog.at_level(logging.INFO, logger="arena.agents.cycle_supervisor"):
        decision_runner._before_model_callback(callback_context=object(), llm_request=request)
        decision_runner._on_model_error_callback(
            callback_context=object(),
            llm_request=request,
            exception=RuntimeError("boom"),
        )

    finish_records = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "agent_cycle_supervisor_operation_finish"
    ]
    assert len(finish_records) == 1
    assert finish_records[0].status == "error"
    assert finish_records[0].llm_call_id == "call_1"


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


def test_adk_own_timeout_is_not_retryable() -> None:
    assert _is_retryable_adk_error(TimeoutError("ADK coroutine timed out after 1530s")) is False
    assert _is_retryable_adk_error(TimeoutError("ADK tool-budget finalization timed out after 60s")) is False
    assert _is_retryable_adk_error(AdkToolBudgetExceeded("ADK tool budget exceeded after 121 tool calls")) is False
    assert _is_retryable_adk_error(RuntimeError("429 RESOURCE_EXHAUSTED")) is True
    assert _is_retryable_adk_error(RuntimeError("litellm.BadGatewayError: 502 Bad gateway")) is True
