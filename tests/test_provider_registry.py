from __future__ import annotations

from arena.config import load_settings
from arena.providers.registry import (
    canonical_provider,
    default_model_for_provider,
    get_provider_spec,
    provider_base_url_from_settings,
    provider_alias_map,
    provider_has_credentials,
    provider_has_direct_api_key,
)


def test_provider_alias_map_exposes_builtin_aliases() -> None:
    aliases = provider_alias_map()

    assert aliases["gpt"] == "gpt"
    assert aliases["openai"] == "gpt"
    assert aliases["gemini"] == "gemini"
    assert aliases["google"] == "gemini"
    assert aliases["claude"] == "claude"
    assert aliases["anthropic"] == "claude"
    assert aliases["deepseek"] == "deepseek"


def test_canonical_provider_normalizes_aliases() -> None:
    assert canonical_provider("openai") == "gpt"
    assert canonical_provider("google") == "gemini"
    assert canonical_provider("anthropic") == "claude"
    assert canonical_provider("deepseek") == "deepseek"
    assert canonical_provider("unknown") == ""


def test_default_model_for_provider_reads_bound_setting_field() -> None:
    settings = load_settings()
    settings.openai_model = "gpt-5.2"
    settings.gemini_model = "gemini-2.5-flash"
    settings.anthropic_model = "claude-sonnet-4-6"
    settings.provider_secrets = {"deepseek": {"model": "deepseek-reasoner"}}

    assert default_model_for_provider(settings, "gpt") == "gpt-5.2"
    assert default_model_for_provider(settings, "gemini") == "gemini-2.5-flash"
    assert default_model_for_provider(settings, "claude") == "claude-sonnet-4-6"
    assert default_model_for_provider(settings, "deepseek") == "deepseek-reasoner"


def test_provider_has_direct_api_key_reads_expected_field() -> None:
    settings = load_settings()
    settings.openai_api_key = "tenant-openai"
    settings.gemini_api_key = ""
    settings.anthropic_api_key = ""
    settings.provider_secrets = {"deepseek": {"api_key": "tenant-deepseek"}}

    assert provider_has_direct_api_key(settings, "gpt") is True
    assert provider_has_direct_api_key(settings, "gemini") is False
    assert provider_has_direct_api_key(settings, "deepseek") is True


def test_provider_has_credentials_includes_vertex_modes() -> None:
    settings = load_settings()
    settings.gemini_api_key = ""
    settings.anthropic_api_key = ""
    settings.anthropic_use_vertexai = True

    assert provider_has_credentials(settings, "claude") is True


def test_get_provider_spec_exposes_capabilities() -> None:
    gemini = get_provider_spec("gemini")
    claude = get_provider_spec("claude")
    deepseek = get_provider_spec("deepseek")

    assert gemini is not None
    assert gemini.transport == "gemini_native"
    assert gemini.supports_grounded_search is True
    assert claude is not None
    assert claude.supports_compaction is True
    assert deepseek is not None
    assert deepseek.transport == "openai_compatible"
    assert deepseek.supports_adk is True


def test_provider_base_url_uses_secret_override_then_default() -> None:
    settings = load_settings()
    settings.provider_secrets = {"deepseek": {"base_url": "https://custom.deepseek/v1"}}

    assert provider_base_url_from_settings(settings, "deepseek") == "https://custom.deepseek/v1"

    settings.provider_secrets = {}
    assert provider_base_url_from_settings(settings, "deepseek") == "https://api.deepseek.com/v1"
