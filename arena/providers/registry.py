from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Built-in provider metadata for routing and capability checks."""

    provider_id: str
    label: str
    aliases: tuple[str, ...]
    transport: str
    api_key_setting: str | None
    model_setting: str | None
    litellm_provider: str | None = None
    default_model: str | None = None
    default_base_url: str | None = None
    supports_adk: bool = False
    supports_direct_text: bool = False
    supports_grounded_search: bool = False
    supports_compaction: bool = False
    supports_vertex_env: bool = False
    supports_vertex_setting: bool = False


_BUILTIN_PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        provider_id="gpt",
        label="OpenAI",
        aliases=("gpt", "openai"),
        transport="openai_compatible",
        api_key_setting="openai_api_key",
        model_setting="openai_model",
        litellm_provider="openai",
        supports_adk=True,
        supports_direct_text=True,
        supports_compaction=True,
    ),
    ProviderSpec(
        provider_id="gemini",
        label="Google Gemini",
        aliases=("gemini", "google"),
        transport="gemini_native",
        api_key_setting="gemini_api_key",
        model_setting="gemini_model",
        litellm_provider="gemini",
        supports_adk=True,
        supports_direct_text=True,
        supports_grounded_search=True,
        supports_compaction=True,
        supports_vertex_env=True,
    ),
    ProviderSpec(
        provider_id="claude",
        label="Anthropic Claude",
        aliases=("claude", "anthropic"),
        transport="anthropic_native",
        api_key_setting="anthropic_api_key",
        model_setting="anthropic_model",
        litellm_provider="anthropic",
        supports_adk=True,
        supports_direct_text=True,
        supports_compaction=True,
        supports_vertex_setting=True,
    ),
    ProviderSpec(
        provider_id="deepseek",
        label="DeepSeek",
        aliases=("deepseek",),
        transport="openai_compatible",
        api_key_setting=None,
        model_setting=None,
        litellm_provider="deepseek",
        default_model="deepseek-chat",
        default_base_url="https://api.deepseek.com/v1",
        supports_adk=True,
        supports_direct_text=True,
        supports_compaction=True,
    ),
)

_PROVIDER_BY_ID: dict[str, ProviderSpec] = {
    spec.provider_id: spec for spec in _BUILTIN_PROVIDERS
}

_ALIAS_MAP: dict[str, str] = {}
for _spec in _BUILTIN_PROVIDERS:
    for _alias in _spec.aliases:
        _ALIAS_MAP[_alias] = _spec.provider_id


def provider_alias_map() -> dict[str, str]:
    """Returns alias -> canonical provider map."""
    return dict(_ALIAS_MAP)


def list_provider_specs() -> tuple[ProviderSpec, ...]:
    """Returns the built-in provider specs in stable display order."""
    return _BUILTIN_PROVIDERS


def list_adk_provider_specs() -> tuple[ProviderSpec, ...]:
    """Returns providers supported by the ADK trading-agent path."""
    return tuple(spec for spec in _BUILTIN_PROVIDERS if spec.supports_adk)


def list_helper_provider_specs() -> tuple[ProviderSpec, ...]:
    """Returns providers supported by direct helper paths like compaction."""
    return tuple(spec for spec in _BUILTIN_PROVIDERS if spec.supports_direct_text or spec.supports_compaction)


def canonical_provider(value: str | None) -> str:
    """Normalizes a provider token into a canonical built-in provider id."""
    token = str(value or "").strip().lower()
    if not token:
        return ""
    return _ALIAS_MAP.get(token, "")


def get_provider_spec(provider: str | None) -> ProviderSpec | None:
    """Returns the built-in provider spec for a canonical token or alias."""
    key = canonical_provider(provider)
    if not key:
        return None
    return _PROVIDER_BY_ID.get(key)


def _provider_settings_entry(settings: Any, provider: str | None) -> dict[str, str]:
    token = canonical_provider(provider) or str(provider or "").strip().lower()
    if not token:
        return {}
    raw = getattr(settings, "provider_secrets", {})
    if not isinstance(raw, Mapping):
        return {}
    entry = raw.get(token)
    if not isinstance(entry, Mapping):
        return {}
    clean: dict[str, str] = {}
    for key, value in entry.items():
        field = str(key or "").strip().lower()
        text = str(value or "").strip()
        if field and text:
            clean[field] = text
    return clean


def provider_api_key_from_settings(settings: Any, provider: str | None) -> str:
    """Returns the provider API key from dedicated settings or generic provider payload."""
    spec = get_provider_spec(provider)
    if spec is not None and spec.api_key_setting:
        value = str(getattr(settings, spec.api_key_setting, "") or "").strip()
        if value:
            return value
    return str(_provider_settings_entry(settings, provider).get("api_key") or "").strip()


def default_model_for_provider(settings: Any, provider: str | None) -> str:
    """Returns the default configured model token for one provider."""
    spec = get_provider_spec(provider)
    if spec is not None and spec.model_setting:
        value = str(getattr(settings, spec.model_setting, "") or "").strip()
        if value:
            return value
    entry_value = str(_provider_settings_entry(settings, provider).get("model") or "").strip()
    if entry_value:
        return entry_value
    if spec is not None and spec.default_model:
        return str(spec.default_model).strip()
    return ""


def provider_base_url_from_settings(settings: Any, provider: str | None) -> str:
    """Returns the provider base URL from generic provider payload or defaults."""
    spec = get_provider_spec(provider)
    entry_value = str(_provider_settings_entry(settings, provider).get("base_url") or "").strip()
    if entry_value:
        return entry_value
    if spec is not None and spec.default_base_url:
        return str(spec.default_base_url).strip()
    return ""


def provider_has_direct_api_key(settings: Any, provider: str | None) -> bool:
    """Returns True when the provider has a direct tenant API key configured."""
    return bool(provider_api_key_from_settings(settings, provider))


def provider_has_credentials(settings: Any, provider: str | None) -> bool:
    """Returns True when the provider can run under current runtime settings."""
    spec = get_provider_spec(provider)
    if spec is None:
        return False

    if provider_has_direct_api_key(settings, spec.provider_id):
        return True

    if spec.supports_vertex_env:
        use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower()
        if use_vertex in {"1", "true", "yes", "y", "on"}:
            return True

    if spec.supports_vertex_setting and bool(getattr(settings, "anthropic_use_vertexai", False)):
        return True

    return False
