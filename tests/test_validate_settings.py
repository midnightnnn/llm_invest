from __future__ import annotations

import pytest

from arena.config import SettingsError, load_settings, research_generation_status, validate_settings


def test_validate_settings_allows_missing_gemini_for_research_when_llm_enabled(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.openai_api_key = "test-openai-key"
    settings.gemini_api_key = ""
    settings.research_gemini_model = "gemini-2.5-flash"
    settings.research_enabled = True

    validate_settings(settings, require_llm=True)

    status = research_generation_status(settings)
    assert status["code"] == "missing_gemini_key"
    assert status["can_generate"] is False


def test_validate_settings_allows_single_gpt_trader_with_gemini_research() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.openai_api_key = "test-openai-key"
    settings.gemini_api_key = "test-gemini-key"
    settings.research_gemini_model = "gemini-2.5-flash"
    settings.research_enabled = True

    validate_settings(settings, require_llm=True)


def test_research_generation_status_reports_shared_live_tenant(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    settings = load_settings()
    settings.gemini_api_key = ""
    settings.research_gemini_api_key = "shared-research-gemini"
    settings.research_gemini_source = "shared_live_tenant"
    settings.research_gemini_source_tenant = "midnightnnn"
    settings.research_enabled = True

    status = research_generation_status(settings)

    assert status["code"] == "shared_live_tenant"
    assert status["can_generate"] is True
    assert status["research_source_tenant"] == "midnightnnn"


def test_validate_settings_allows_single_deepseek_trader_without_research() -> None:
    settings = load_settings()
    settings.agent_ids = ["deepseek"]
    settings.agent_configs = {}
    settings.provider_secrets = {"deepseek": {"api_key": "tenant-deepseek", "model": "deepseek-chat"}}
    settings.research_enabled = False

    validate_settings(settings, require_llm=True)


def test_validate_settings_requires_api_key_for_registry_backed_adk_provider() -> None:
    settings = load_settings()
    settings.agent_ids = ["deepseek"]
    settings.agent_configs = {}
    settings.provider_secrets = {}
    settings.research_enabled = False

    with pytest.raises(SettingsError) as exc_info:
        validate_settings(settings, require_llm=True)

    assert "No agents have usable credentials" in str(exc_info.value)


def test_validate_settings_skips_missing_claude_credentials_gracefully(caplog) -> None:
    """Tenants that inherit a global ARENA_AGENT_IDS but lack a provider's key
    should have that agent silently dropped (with a warning), not fail the cycle."""
    import logging

    settings = load_settings()
    settings.agent_ids = ["gpt", "claude"]
    settings.agent_configs = {}
    settings.openai_api_key = "test-openai-key"
    settings.gemini_api_key = ""
    settings.anthropic_api_key = ""
    settings.anthropic_use_vertexai = False
    settings.research_enabled = False

    with caplog.at_level(logging.WARNING, logger="arena.config"):
        validate_settings(settings, require_llm=True)

    assert "claude" not in settings.agent_ids
    assert settings.agent_ids == ["gpt"]
    assert any("Agent skipped: no credentials" in r.message and "claude" in r.message
               for r in caplog.records)


def test_load_settings_paper_only_forces_demo_and_live(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_DISTRIBUTION_MODE", "paper_only")
    monkeypatch.setenv("KIS_ENV", "real")
    monkeypatch.setenv("ARENA_ALLOW_LIVE_TRADING", "false")

    settings = load_settings()

    assert settings.distribution_mode == "paper_only"
    assert settings.kis_env == "demo"
    assert settings.allow_live_trading is True


def test_validate_settings_rejects_live_in_simulated_only(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_DISTRIBUTION_MODE", "simulated_only")

    settings = load_settings()

    with pytest.raises(SettingsError) as exc_info:
        validate_settings(settings, live=True)

    assert "simulated_only" in str(exc_info.value)


def test_validate_settings_allows_multi_market_combo_for_kis(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_DISTRIBUTION_MODE", raising=False)

    settings = load_settings()
    settings.kis_target_market = "us,kospi"
    settings.kis_env = "demo"
    settings.kis_secret_name = "KISAPI"
    settings.kis_account_no = "12345678-01"

    validate_settings(settings, require_kis=True)


def test_timeout_for_falls_back_when_role_override_unset(monkeypatch) -> None:
    for env in (
        "ARENA_LLM_TIMEOUT_TRADING_SECONDS",
        "ARENA_LLM_TIMEOUT_RESEARCH_SECONDS",
        "ARENA_LLM_TIMEOUT_COMPACTION_SECONDS",
    ):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("ARENA_LLM_TIMEOUT_SECONDS", "120")

    settings = load_settings()

    assert settings.llm_timeout_seconds == 120
    assert settings.llm_timeout_trading_seconds is None
    assert settings.timeout_for("trading") == 120
    assert settings.timeout_for("research") == 120
    assert settings.timeout_for("compaction") == 120
    assert settings.timeout_for("unknown-role") == 120


def test_timeout_for_uses_role_override_when_set(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_LLM_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("ARENA_LLM_TIMEOUT_TRADING_SECONDS", "900")
    monkeypatch.setenv("ARENA_LLM_TIMEOUT_RESEARCH_SECONDS", "60")
    monkeypatch.setenv("ARENA_LLM_TIMEOUT_COMPACTION_SECONDS", "300")

    settings = load_settings()

    assert settings.timeout_for("trading") == 900
    assert settings.timeout_for("research") == 60
    assert settings.timeout_for("compaction") == 300
    assert settings.timeout_for("other") == 90


def test_timeout_for_runtime_override_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_LLM_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("ARENA_LLM_TIMEOUT_RESEARCH_SECONDS", "60")
    monkeypatch.setenv("ARENA_LLM_TIMEOUT_TRADING_SECONDS", "900")

    settings = load_settings()

    assert settings.timeout_for("research") == 60
    assert settings.timeout_for("trading") == 900

    settings.llm_timeout_runtime_override_seconds = 15

    assert settings.timeout_for("research") == 15, "runtime override must beat role env"
    assert settings.timeout_for("trading") == 15
    assert settings.timeout_for("unknown") == 15


def test_validate_settings_rejects_non_positive_role_timeout(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_LLM_TIMEOUT_RESEARCH_SECONDS", "0")

    settings = load_settings()
    settings.kis_env = "demo"
    settings.kis_secret_name = "KISAPI"
    settings.kis_account_no = "12345678-01"

    with pytest.raises(SettingsError) as exc_info:
        validate_settings(settings, require_kis=True)

    assert "ARENA_LLM_TIMEOUT_RESEARCH_SECONDS must be positive" in str(exc_info.value)
