from __future__ import annotations

import asyncio
import logging

import pytest

from arena.agents.adk_agents import _has_credentials, _resolve_disabled_tool_ids, _resolve_model
from arena.agents.adk_models import _InstrumentedLiteLLMClient
from arena.config import AgentConfig, load_settings


class _RepoForTools:
    def __init__(self, disabled: str | None):
        self.disabled = disabled

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        _ = tenant_id, config_key
        return self.disabled


def _agent_config(*, disabled_tools: list[str] | None) -> AgentConfig:
    return AgentConfig(
        agent_id="custom",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        disabled_tools=disabled_tools,
    )


@pytest.mark.parametrize(
    ("global_disabled", "agent_disabled", "expected"),
    [
        ('["tool_a","tool_b"]', ["tool_x"], {"tool_x"}),
        ('["tool_a"]', None, {"tool_a"}),
        ('["tool_a"]', [], set()),
    ],
    ids=["agent-override", "global-fallback", "empty-agent-override"],
)
def test_resolve_disabled_tool_ids_prefers_agent_config_when_set(
    global_disabled: str,
    agent_disabled: list[str] | None,
    expected: set[str],
) -> None:
    result = _resolve_disabled_tool_ids(
        _RepoForTools(global_disabled),
        "tenant-a",
        _agent_config(disabled_tools=agent_disabled),
    )

    assert result == expected


def test_resolve_disabled_tool_ids_without_agent_config_uses_global_config() -> None:
    result = _resolve_disabled_tool_ids(_RepoForTools('["tool_a"]'), "tenant-a", None)

    assert result == {"tool_a"}


@pytest.mark.parametrize(
    ("provider", "settings_updates", "expected"),
    [
        ("gpt", {"openai_api_key": "sk-test"}, True),
        ("gpt", {"openai_api_key": ""}, False),
        ("claude", {"anthropic_api_key": "ak-test", "anthropic_use_vertexai": False}, True),
        ("claude", {"anthropic_api_key": "", "anthropic_use_vertexai": True}, True),
        ("claude", {"anthropic_api_key": "", "anthropic_use_vertexai": False}, False),
        ("unknown", {}, False),
    ],
    ids=[
        "gpt-openai-key",
        "gpt-missing-key",
        "claude-direct-key",
        "claude-vertex",
        "claude-missing-direct-key",
        "unknown-provider",
    ],
)
def test_has_credentials_matches_provider_requirements(provider: str, settings_updates: dict, expected: bool) -> None:
    settings = load_settings()
    for key, value in settings_updates.items():
        setattr(settings, key, value)

    assert _has_credentials(provider, settings) is expected


def test_resolve_model_openai_uses_instance_scoped_api_key() -> None:
    settings = load_settings()
    settings.openai_api_key = "tenant-openai"
    settings.llm_timeout_seconds = 1500

    model = _resolve_model("gpt", settings, model_override="gpt-5.4")

    assert model.model == "openai/gpt-5.4"
    assert model._additional_args["api_key"] == "tenant-openai"
    assert model._additional_args["timeout"] == 1500


def test_resolve_model_claude_direct_uses_instance_scoped_api_key() -> None:
    settings = load_settings()
    settings.anthropic_api_key = "tenant-anthropic"
    settings.anthropic_use_vertexai = False
    settings.anthropic_model = "claude-sonnet-4-6"
    settings.llm_timeout_seconds = 1500

    model = _resolve_model("claude", settings)

    assert model.model == "anthropic/claude-sonnet-4-6"
    assert model._additional_args["api_key"] == "tenant-anthropic"
    assert model._additional_args["timeout"] == 1500
    assert model._additional_args["cache_control_injection_points"] == [
        {"location": "message", "role": "system"},
    ]


def test_resolve_model_wires_model_call_timeout_getter() -> None:
    settings = load_settings()
    settings.anthropic_api_key = "tenant-anthropic"
    settings.anthropic_use_vertexai = False
    settings.anthropic_model = "claude-sonnet-4-6"

    model = _resolve_model(
        "claude",
        settings,
        model_call_timeout_seconds_getter=lambda model_id: 7 if "claude" in model_id else None,
    )

    client = model.llm_client
    assert client._watchdog_timeout_seconds("anthropic/claude-sonnet-4-6") == 7.0


def test_instrumented_litellm_client_logs_completion_boundaries(caplog: pytest.LogCaptureFixture) -> None:
    class _Delegate:
        async def acompletion(self, *, model, messages, tools, **kwargs):
            _ = model, messages, tools, kwargs
            return {"ok": True}

    client = _InstrumentedLiteLLMClient(
        agent_id="claude",
        provider="claude",
        metadata_getter=lambda: {
            "tenant_id": "tenant-a",
            "cycle_id": "cycle-a",
            "phase": "explore",
            "llm_call_id": "llm-a",
        },
        delegate=_Delegate(),
    )

    with caplog.at_level(logging.INFO, logger="arena.agents.adk_models"):
        result = asyncio.run(
            client.acompletion(
                model="anthropic/claude-opus-4-7",
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
                timeout=30,
            )
        )

    assert result == {"ok": True}
    records = [record for record in caplog.records if getattr(record, "event", "").startswith("adk_model_acompletion_")]
    assert [record.event for record in records] == [
        "adk_model_acompletion_start",
        "adk_model_acompletion_end",
    ]
    assert records[0].llm_call_id == "llm-a"
    assert records[0].phase == "explore"
    assert records[0].message_count == 1
    assert records[0].tool_count == 0
    assert records[0].timeout_seconds == 30
    assert records[1].elapsed_ms >= 0


def test_instrumented_litellm_client_logs_completion_errors(caplog: pytest.LogCaptureFixture) -> None:
    class _Delegate:
        async def acompletion(self, *, model, messages, tools, **kwargs):
            _ = model, messages, tools, kwargs
            raise RuntimeError("provider stalled")

    client = _InstrumentedLiteLLMClient(
        agent_id="claude",
        provider="claude",
        metadata_getter=lambda: {"llm_call_id": "llm-err", "phase": "explore"},
        delegate=_Delegate(),
    )

    with caplog.at_level(logging.INFO, logger="arena.agents.adk_models"):
        with pytest.raises(RuntimeError, match="provider stalled"):
            asyncio.run(
                client.acompletion(
                    model="anthropic/claude-opus-4-7",
                    messages=[],
                    tools=None,
                )
            )

    error_records = [
        record for record in caplog.records if getattr(record, "event", "") == "adk_model_acompletion_error"
    ]
    assert len(error_records) == 1
    assert error_records[0].llm_call_id == "llm-err"
    assert error_records[0].err_type == "RuntimeError"


def test_instrumented_litellm_client_times_out_slow_delegate(caplog: pytest.LogCaptureFixture) -> None:
    class _Delegate:
        async def acompletion(self, *, model, messages, tools, **kwargs):
            _ = model, messages, tools, kwargs
            await asyncio.sleep(10)

    client = _InstrumentedLiteLLMClient(
        agent_id="claude",
        provider="claude",
        metadata_getter=lambda: {"llm_call_id": "llm-timeout", "phase": "explore"},
        delegate=_Delegate(),
        model_call_timeout_seconds_getter=lambda model: 0.01,
    )

    with caplog.at_level(logging.INFO, logger="arena.agents.adk_models"):
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(
                client.acompletion(
                    model="anthropic/claude-opus-4-7",
                    messages=[],
                    tools=None,
                )
            )

    timeout_records = [
        record for record in caplog.records if getattr(record, "event", "") == "adk_model_acompletion_timeout"
    ]
    assert len(timeout_records) == 1
    assert timeout_records[0].llm_call_id == "llm-timeout"
    assert timeout_records[0].timeout_seconds == 0.01


def test_instrumented_litellm_client_logs_completion_cancellation(caplog: pytest.LogCaptureFixture) -> None:
    class _Delegate:
        async def acompletion(self, *, model, messages, tools, **kwargs):
            _ = model, messages, tools, kwargs
            raise asyncio.CancelledError()

    client = _InstrumentedLiteLLMClient(
        agent_id="claude",
        provider="claude",
        metadata_getter=lambda: {"llm_call_id": "llm-cancel", "phase": "explore"},
        delegate=_Delegate(),
    )

    with caplog.at_level(logging.INFO, logger="arena.agents.adk_models"):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                client.acompletion(
                    model="anthropic/claude-opus-4-7",
                    messages=[],
                    tools=None,
                )
            )

    cancel_records = [
        record for record in caplog.records if getattr(record, "event", "") == "adk_model_acompletion_cancelled"
    ]
    assert len(cancel_records) == 1
    assert cancel_records[0].llm_call_id == "llm-cancel"


def test_resolve_model_deepseek_is_disabled_for_adk_until_implemented() -> None:
    settings = load_settings()
    settings.provider_secrets = {
        "deepseek": {
            "api_key": "tenant-deepseek",
            "model": "deepseek-chat",
            "base_url": "https://custom.deepseek/v1",
        }
    }

    with pytest.raises(ValueError, match="Unsupported ADK provider: deepseek"):
        _resolve_model("deepseek", settings)
