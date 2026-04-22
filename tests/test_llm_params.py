"""Tests for per-agent LLM parameter plumbing.

Covers:
  - arena.agents.llm_params sanitization (provider whitelists + clamps)
  - _resolve_model injects provider-native kwargs from llm_params
  - _build_generate_content_config wires Gemini ThinkingConfig + sampling
  - AgentConfig round-trip through agents_config JSON parser
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from arena.agents.llm_params import sanitize_llm_params, supported_fields


# ---------------------------------------------------------------------------
# Sanitizer
# ---------------------------------------------------------------------------


class TestSanitizeLlmParams:
    def test_claude_keeps_supported_fields_and_clamps_ranges(self) -> None:
        cleaned = sanitize_llm_params(
            "claude",
            {
                "effort": "XHIGH",
                "temperature": 1.5,  # Anthropic max is 1.0 → clamp
                "top_p": 0.9,
                "max_tokens": 20_000,
                "reasoning_effort": "high",  # not a Claude key → drop
                "thinking_level": "high",    # not a Claude key → drop
            },
        )
        assert cleaned == {
            "effort": "xhigh",
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 20_000,
        }

    def test_claude_invalid_effort_is_dropped(self) -> None:
        cleaned = sanitize_llm_params("claude", {"effort": "turbo"})
        assert cleaned == {}

    def test_openai_filters_and_keeps_native_keys(self) -> None:
        cleaned = sanitize_llm_params(
            "gpt",
            {
                "reasoning_effort": "HIGH",
                "verbosity": "low",
                "max_completion_tokens": 8192,
                "temperature": 0.7,        # OpenAI reasoning: drop
                "effort": "xhigh",         # Anthropic key: drop
            },
        )
        assert cleaned == {
            "reasoning_effort": "high",
            "verbosity": "low",
            "max_completion_tokens": 8192,
        }

    def test_gemini_thinking_level_wins_over_budget(self) -> None:
        cleaned = sanitize_llm_params(
            "gemini",
            {
                "thinking_level": "medium",
                "thinking_budget": 1024,    # dropped when level present
                "temperature": 1.8,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            },
        )
        assert cleaned == {
            "thinking_level": "medium",
            "temperature": 1.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

    def test_gemini_thinking_budget_alone_is_kept(self) -> None:
        cleaned = sanitize_llm_params(
            "gemini",
            {"thinking_budget": -1},
        )
        assert cleaned == {"thinking_budget": -1}

    def test_gemini_top_k_is_clamped_to_max(self) -> None:
        cleaned = sanitize_llm_params("gemini", {"top_k": 500})
        assert cleaned == {"top_k": 40}

    def test_unknown_provider_returns_empty(self) -> None:
        assert sanitize_llm_params("mistral", {"effort": "high"}) == {}

    def test_empty_or_invalid_input_returns_empty(self) -> None:
        assert sanitize_llm_params("claude", {}) == {}
        assert sanitize_llm_params("claude", None) == {}
        assert sanitize_llm_params("claude", "not-a-dict") == {}


# ---------------------------------------------------------------------------
# supported_fields helper
# ---------------------------------------------------------------------------


def test_supported_fields_per_provider() -> None:
    assert supported_fields("claude") == ["effort", "temperature", "top_p", "max_tokens"]
    assert supported_fields("anthropic") == ["effort", "temperature", "top_p", "max_tokens"]
    assert supported_fields("gpt") == ["reasoning_effort", "verbosity", "max_completion_tokens"]
    assert supported_fields("gemini") == [
        "thinking_level",
        "thinking_budget",
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
    ]


# ---------------------------------------------------------------------------
# Anthropic kwargs from llm_params
# ---------------------------------------------------------------------------


class TestAnthropicRuntimeKwargs:
    def test_opus_4_7_defaults_without_overrides_keep_xhigh_plus_adaptive(self) -> None:
        from arena.agents.adk_models import _anthropic_runtime_kwargs

        extra = _anthropic_runtime_kwargs("claude-opus-4-7")
        assert extra["output_config"] == {"effort": "xhigh"}
        assert extra["thinking"] == {"type": "adaptive"}

    def test_opus_4_7_user_effort_override_respected(self) -> None:
        from arena.agents.adk_models import _anthropic_runtime_kwargs

        extra = _anthropic_runtime_kwargs("claude-opus-4-7", {"effort": "medium"})
        assert extra["output_config"] == {"effort": "medium"}
        # adaptive thinking stays regardless of effort
        assert extra["thinking"] == {"type": "adaptive"}

    def test_sonnet_4_6_xhigh_override_clamped_to_high(self) -> None:
        from arena.agents.adk_models import _anthropic_runtime_kwargs

        extra = _anthropic_runtime_kwargs("claude-sonnet-4-6", {"effort": "xhigh"})
        assert extra["output_config"] == {"effort": "high"}
        assert "thinking" not in extra  # non-4.7 models don't force adaptive

    def test_opus_4_5_max_effort_clamped_to_high(self) -> None:
        from arena.agents.adk_models import _anthropic_runtime_kwargs

        extra = _anthropic_runtime_kwargs("claude-opus-4-5", {"effort": "max"})
        # 4.5 doesn't support max → clamp to high (xhigh is 4.7-only)
        assert extra["output_config"] == {"effort": "high"}

    def test_sampling_params_passed_through(self) -> None:
        from arena.agents.adk_models import _anthropic_runtime_kwargs

        extra = _anthropic_runtime_kwargs(
            "claude-opus-4-7",
            {"temperature": 0.5, "top_p": 0.9, "max_tokens": 16000},
        )
        assert extra["temperature"] == 0.5
        assert extra["top_p"] == 0.9
        assert extra["max_tokens"] == 16000


# ---------------------------------------------------------------------------
# _resolve_model with llm_params
# ---------------------------------------------------------------------------


class TestResolveModelLlmParams:
    def test_openai_injects_reasoning_effort_and_max_completion_tokens(self) -> None:
        from arena.agents.adk_models import _resolve_model
        from arena.config import Settings

        settings = Settings.__new__(Settings)  # skip validation — minimal fields
        # Fallback: use load_settings flavor used in the existing tests
        from tests.test_adk_agents import load_settings

        settings = load_settings()
        settings.openai_api_key = "tenant-openai"

        model = _resolve_model(
            "gpt",
            settings,
            model_override="gpt-5.4",
            llm_params={
                "reasoning_effort": "high",
                "verbosity": "medium",
                "max_completion_tokens": 8192,
            },
        )
        args = model._additional_args
        assert args["reasoning_effort"] == "high"
        assert args["verbosity"] == "medium"
        assert args["max_completion_tokens"] == 8192

    def test_openai_without_llm_params_does_not_inject_reasoning(self) -> None:
        from arena.agents.adk_models import _resolve_model
        from tests.test_adk_agents import load_settings

        settings = load_settings()
        settings.openai_api_key = "tenant-openai"

        model = _resolve_model("gpt", settings, model_override="gpt-5.4")
        assert "reasoning_effort" not in model._additional_args
        assert "verbosity" not in model._additional_args

    def test_claude_direct_injects_user_effort_override(self) -> None:
        from arena.agents.adk_models import _resolve_model
        from tests.test_adk_agents import load_settings

        settings = load_settings()
        settings.anthropic_api_key = "tenant-anthropic"
        settings.anthropic_use_vertexai = False

        model = _resolve_model(
            "claude",
            settings,
            model_override="claude-opus-4-7",
            llm_params={"effort": "medium", "temperature": 0.3},
        )
        args = model._additional_args
        assert args["output_config"] == {"effort": "medium"}
        assert args["thinking"] == {"type": "adaptive"}
        assert args["temperature"] == 0.3


# ---------------------------------------------------------------------------
# Gemini GenerateContentConfig
# ---------------------------------------------------------------------------


class TestGeminiGenerateContentConfig:
    def test_thinking_level_sets_enum_variant(self) -> None:
        from arena.agents.adk_runner_bootstrap import _build_generate_content_config

        cfg = _build_generate_content_config(
            provider="gemini",
            llm_params={"thinking_level": "high", "temperature": 1.0, "top_p": 0.9, "top_k": 20, "max_output_tokens": 8192},
            max_tool_events=50,
        )
        assert cfg.thinking_config is not None
        assert str(cfg.thinking_config.thinking_level).endswith("HIGH")
        assert cfg.temperature == 1.0
        assert cfg.top_p == 0.9
        assert cfg.top_k == 20
        assert cfg.max_output_tokens == 8192

    def test_thinking_budget_used_when_no_level(self) -> None:
        from arena.agents.adk_runner_bootstrap import _build_generate_content_config

        cfg = _build_generate_content_config(
            provider="gemini",
            llm_params={"thinking_budget": -1},
            max_tool_events=50,
        )
        assert cfg.thinking_config is not None
        assert cfg.thinking_config.thinking_budget == -1

    def test_non_gemini_provider_produces_minimal_config(self) -> None:
        from arena.agents.adk_runner_bootstrap import _build_generate_content_config

        cfg = _build_generate_content_config(
            provider="claude",
            llm_params={"effort": "xhigh", "temperature": 0.5},  # should be ignored here
            max_tool_events=50,
        )
        assert cfg.thinking_config is None
        assert cfg.temperature is None

    def test_empty_llm_params_keeps_provider_defaults(self) -> None:
        from arena.agents.adk_runner_bootstrap import _build_generate_content_config

        cfg = _build_generate_content_config(
            provider="gemini",
            llm_params={},
            max_tool_events=50,
        )
        assert cfg.thinking_config is None
        assert cfg.temperature is None


# ---------------------------------------------------------------------------
# Round-trip via agents_config JSON
# ---------------------------------------------------------------------------


class _FakeConfigRepo:
    def __init__(self, values: dict[str, str]):
        self._values = values

    def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]:
        _ = tenant_id, config_keys
        return dict(self._values)


class TestAgentsConfigRoundTrip:
    def test_llm_params_survive_json_parse(self) -> None:
        """agents_config JSON → Settings.agent_configs → AgentConfig.llm_params."""
        from arena.config import apply_runtime_overrides, load_settings

        settings = load_settings()
        agents_payload = [
            {
                "id": "claude",
                "provider": "claude",
                "model": "claude-opus-4-7",
                "capital_krw": 1_000_000,
                "llm_params": {"effort": "medium", "temperature": 0.4},
            },
            {
                "id": "gpt",
                "provider": "gpt",
                "model": "gpt-5.4",
                "capital_krw": 1_000_000,
                "llm_params": {"reasoning_effort": "high", "max_completion_tokens": 8192},
            },
            {
                "id": "gemini",
                "provider": "gemini",
                "model": "gemini-3-pro",
                "capital_krw": 1_000_000,
                "llm_params": {"thinking_level": "high", "temperature": 1.2},
            },
        ]
        repo = _FakeConfigRepo({"agents_config": json.dumps(agents_payload)})
        out = apply_runtime_overrides(settings, repo, tenant_id="t")

        assert set(out.agent_configs) == {"claude", "gpt", "gemini"}
        assert out.agent_configs["claude"].llm_params == {
            "effort": "medium",
            "temperature": 0.4,
        }
        assert out.agent_configs["gpt"].llm_params == {
            "reasoning_effort": "high",
            "max_completion_tokens": 8192,
        }
        assert out.agent_configs["gemini"].llm_params == {
            "thinking_level": "high",
            "temperature": 1.2,
        }

    def test_llm_params_invalid_keys_are_dropped_at_load(self) -> None:
        from arena.config import apply_runtime_overrides, load_settings

        settings = load_settings()
        payload = [{
            "id": "gpt",
            "provider": "gpt",
            "model": "gpt-5.4",
            "capital_krw": 1_000_000,
            "llm_params": {
                "reasoning_effort": "high",
                "effort": "xhigh",  # wrong provider key — should be dropped
                "temperature": 0.8,  # not supported on OpenAI reasoning — drop
            },
        }]
        repo = _FakeConfigRepo({"agents_config": json.dumps(payload)})
        out = apply_runtime_overrides(settings, repo, tenant_id="t")
        assert out.agent_configs["gpt"].llm_params == {"reasoning_effort": "high"}
