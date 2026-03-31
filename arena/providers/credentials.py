from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from arena.providers.registry import canonical_provider, list_provider_specs


def normalize_provider_token(value: str | None) -> str:
    """Normalizes a provider token while allowing future non-built-in providers."""
    token = str(value or "").strip().lower()
    if not token:
        return ""
    return canonical_provider(token) or token


def _clean_provider_entry(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    clean: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip().lower()
        if not key:
            continue
        text = str(raw_value or "").strip()
        if text:
            clean[key] = text
    return clean


def parse_model_secret_providers(payload: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """Parses provider-scoped model credentials from the secret payload."""
    parsed: dict[str, dict[str, str]] = {}
    raw_payload = payload if isinstance(payload, Mapping) else {}

    raw_providers = raw_payload.get("providers")
    if isinstance(raw_providers, Mapping):
        for raw_provider, raw_entry in raw_providers.items():
            provider = normalize_provider_token(str(raw_provider or ""))
            entry = _clean_provider_entry(raw_entry)
            if provider and entry:
                parsed[provider] = entry

    for spec in list_provider_specs():
        if not spec.api_key_setting:
            continue
        api_key = str(raw_payload.get(spec.api_key_setting) or "").strip()
        if not api_key:
            continue
        current = dict(parsed.get(spec.provider_id) or {})
        current["api_key"] = api_key
        parsed[spec.provider_id] = current

    return parsed


def build_model_secret_payload(
    *,
    previous_payload: Mapping[str, Any] | None = None,
    provider_updates: Mapping[str, Mapping[str, Any]] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Builds the canonical Secret Manager payload for model credentials."""
    merged = parse_model_secret_providers(previous_payload)

    for raw_provider, raw_entry in dict(provider_updates or {}).items():
        provider = normalize_provider_token(str(raw_provider or ""))
        entry = _clean_provider_entry(raw_entry)
        if not provider or not entry:
            continue
        current = dict(merged.get(provider) or {})
        current.update(entry)
        merged[provider] = current

    payload: dict[str, Any] = {
        "providers": merged,
    }
    for spec in list_provider_specs():
        if not spec.api_key_setting:
            continue
        payload[spec.api_key_setting] = str((merged.get(spec.provider_id) or {}).get("api_key") or "").strip()
    if updated_at:
        payload["updated_at"] = str(updated_at).strip()
    return payload


def apply_model_secret_payload(settings: Any, payload: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    """Applies model-secret provider credentials onto the runtime settings object."""
    providers = parse_model_secret_providers(payload)
    setattr(
        settings,
        "provider_secrets",
        {provider: dict(entry) for provider, entry in providers.items()},
    )
    for spec in list_provider_specs():
        if not spec.api_key_setting:
            continue
        api_key = str((providers.get(spec.provider_id) or {}).get("api_key") or "").strip()
        setattr(settings, spec.api_key_setting, api_key)
    return providers


def runtime_credential_flags(providers: Mapping[str, Mapping[str, Any]] | None) -> dict[str, bool]:
    """Returns built-in runtime metadata flags derived from provider credentials."""
    parsed = {
        normalize_provider_token(str(provider or "")): _clean_provider_entry(entry)
        for provider, entry in dict(providers or {}).items()
        if normalize_provider_token(str(provider or ""))
    }
    return {
        "has_openai": bool(str((parsed.get("gpt") or {}).get("api_key") or "").strip()),
        "has_gemini": bool(str((parsed.get("gemini") or {}).get("api_key") or "").strip()),
        "has_anthropic": bool(str((parsed.get("claude") or {}).get("api_key") or "").strip()),
    }


def runtime_row_api_key_status(row: Mapping[str, Any] | None) -> dict[str, bool]:
    """Returns provider/alias -> API-key status map for UI use."""
    raw = row if isinstance(row, Mapping) else {}
    has_openai = bool(raw.get("has_openai"))
    has_gemini = bool(raw.get("has_gemini"))
    has_anthropic = bool(raw.get("has_anthropic"))
    return {
        "gpt": has_openai,
        "openai": has_openai,
        "gemini": has_gemini,
        "google": has_gemini,
        "claude": has_anthropic,
        "anthropic": has_anthropic,
    }
