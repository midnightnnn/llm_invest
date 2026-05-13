from __future__ import annotations

import json

from arena.config import AgentConfig, apply_runtime_overrides, load_settings, merge_agent_risk_settings, validate_settings
from tests.config.agents_config_helpers import _FakeConfigRepo


def test_agents_config_overrides_agent_ids_and_capitals() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 500_000

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gemini", "model": "gemini-3-flash", "capital_krw": 1_000_000},
                {"id": "claude", "model": "claude-sonnet-4-6", "capital_krw": 2_000_000},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gemini", "claude"]
    assert out.agent_capitals["gemini"] == 1_000_000
    assert out.agent_capitals["claude"] == 2_000_000
    assert out.gemini_model == "gemini-3-flash"
    assert out.anthropic_model == "claude-sonnet-4-6"


def test_agents_config_fallback_to_sleeve_capital_when_no_capital() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 750_000

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gpt", "model": "gpt-4.1"},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gpt"]
    # capital_krw not specified → fallback to sleeve_capital_krw
    assert out.agent_capitals["gpt"] == 750_000


def test_agents_config_missing_ignores_legacy_agent_keys() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 100_000

    repo = _FakeConfigRepo(
        {
            "agent_ids": json.dumps(["gemini", "claude"]),
            "agent_models": json.dumps({"gemini": "gemini-3-flash"}),
            "sleeve_capital_krw": "200000",
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gpt"]
    assert out.sleeve_capital_krw == 200_000
    assert out.openai_model == settings.openai_model
    assert out.agent_capitals["gpt"] == 200_000


def test_agents_config_skips_invalid_entries() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 500_000

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gemini", "model": "m1", "capital_krw": 1_000_000},
                "not-a-dict",
                {"id": "", "model": "m2"},
                {"model": "m3"},
                {"id": "claude", "capital_krw": "not-a-number"},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gemini", "claude"]
    assert out.agent_capitals["gemini"] == 1_000_000
    # claude had invalid capital → fallback to sleeve_capital_krw
    assert out.agent_capitals["claude"] == 500_000


def test_agents_config_per_provider_model_override() -> None:
    settings = load_settings()
    settings.openai_model = "gpt-old"
    settings.gemini_model = "gemini-old"
    settings.anthropic_model = "claude-old"

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gpt", "model": "gpt-5"},
                {"id": "gemini", "model": "gemini-3-ultra"},
                {"id": "claude", "model": "claude-opus-4-6"},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.openai_model == "gpt-5"
    assert out.gemini_model == "gemini-3-ultra"
    assert out.anthropic_model == "claude-opus-4-6"


def test_agents_config_parses_per_agent_provider_and_fields() -> None:
    """agents_config with explicit provider + per-agent prompt/risk/tools."""
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 500_000

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {
                    "id": "aggressive-gpt",
                    "provider": "gpt",
                    "model": "gpt-5.2",
                    "capital_krw": 2_000_000,
                    "system_prompt": "Be aggressive.",
                    "risk_policy": {"max_order_krw": 50_000_000, "max_daily_orders": 20},
                    "disabled_tools": ["screen_market"],
                    "memory_compaction_model": "gpt-5.4",
                },
                {
                    "id": "safe-gpt",
                    "provider": "gpt",
                    "model": "gpt-4.1",
                    "capital_krw": 1_000_000,
                },
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["aggressive-gpt", "safe-gpt"]
    assert "aggressive-gpt" in out.agent_configs
    assert "safe-gpt" in out.agent_configs

    ac_agg = out.agent_configs["aggressive-gpt"]
    assert ac_agg.provider == "gpt"
    assert ac_agg.model == "gpt-5.2"
    assert ac_agg.capital_krw == 2_000_000
    assert ac_agg.system_prompt == "Be aggressive."
    assert ac_agg.risk_overrides == {"max_order_krw": 50_000_000, "max_daily_orders": 20}
    assert ac_agg.disabled_tools == ["screen_market"]
    assert ac_agg.memory_compaction_model == "gpt-5.4"

    ac_safe = out.agent_configs["safe-gpt"]
    assert ac_safe.provider == "gpt"
    assert ac_safe.system_prompt is None
    assert ac_safe.risk_overrides is None
    assert ac_safe.disabled_tools is None


def test_agents_config_infers_provider_from_id() -> None:
    """Legacy agents_config without explicit provider infers from id."""
    settings = load_settings()
    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gpt", "model": "gpt-5.2"},
                {"id": "gemini", "model": "gemini-3-flash"},
                {"id": "claude", "model": "claude-sonnet-4-6"},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_configs["gpt"].provider == "gpt"
    assert out.agent_configs["gemini"].provider == "gemini"
    assert out.agent_configs["claude"].provider == "claude"


def test_agents_config_canonicalizes_provider_aliases_for_validation() -> None:
    settings = load_settings()
    settings.openai_api_key = "test-openai-key"
    settings.gemini_api_key = "test-gemini-key"
    settings.anthropic_api_key = "test-anthropic-key"
    settings.anthropic_use_vertexai = False
    settings.research_enabled = False
    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gpt", "provider": "openai", "model": "gpt-5.2"},
                {"id": "gemini", "provider": "google", "model": "gemini-3-flash-preview"},
                {"id": "claude", "provider": "anthropic", "model": "claude-sonnet-4-6"},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")
    validate_settings(out, require_llm=True)

    assert out.agent_configs["gpt"].provider == "gpt"
    assert out.agent_configs["gemini"].provider == "gemini"
    assert out.agent_configs["claude"].provider == "claude"


def test_agents_config_same_provider_duplicate() -> None:
    """Two agents with same provider (gpt) but different ids."""
    settings = load_settings()
    settings.openai_api_key = "sk-test"

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gpt-a", "provider": "gpt", "model": "gpt-5.2", "capital_krw": 1_000_000},
                {"id": "gpt-b", "provider": "gpt", "model": "gpt-4.1", "capital_krw": 500_000},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gpt-a", "gpt-b"]
    assert out.agent_configs["gpt-a"].model == "gpt-5.2"
    assert out.agent_configs["gpt-b"].model == "gpt-4.1"
    assert out.agent_capitals["gpt-a"] == 1_000_000
    assert out.agent_capitals["gpt-b"] == 500_000


def test_agents_config_empty_keeps_env_agents_normalized() -> None:
    """When agents_config is absent, env/default agents remain normalized."""
    settings = load_settings()
    settings.agent_ids = ["gpt", "gemini"]

    repo = _FakeConfigRepo({})

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gpt", "gemini"]
    assert out.agent_configs["gpt"].provider == "gpt"
    assert out.agent_configs["gemini"].provider == "gemini"


def test_merge_agent_risk_settings_applies_overrides() -> None:
    settings = load_settings()
    settings.max_order_krw = 100_000
    settings.max_daily_orders = 5

    ac = AgentConfig(
        agent_id="agg",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        risk_overrides={"max_order_krw": 500_000, "max_daily_orders": 20},
    )

    merged = merge_agent_risk_settings(settings, ac)

    assert merged.max_order_krw == 500_000
    assert merged.max_daily_orders == 20
    # Original unchanged
    assert settings.max_order_krw == 100_000
    assert settings.max_daily_orders == 5


def test_merge_agent_risk_settings_returns_original_when_none() -> None:
    settings = load_settings()
    settings.max_order_krw = 100_000

    result = merge_agent_risk_settings(settings, None)
    assert result is settings

    ac = AgentConfig(
        agent_id="x",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        risk_overrides=None,
    )
    result = merge_agent_risk_settings(settings, ac)
    assert result is settings
