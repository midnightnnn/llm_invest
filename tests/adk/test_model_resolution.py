from __future__ import annotations

import pytest

from arena.agents.adk_agents import _has_credentials, _resolve_disabled_tool_ids, _resolve_model
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
