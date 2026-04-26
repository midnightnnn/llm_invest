from __future__ import annotations

from arena.agents.adk_agents import _normalize_gemini_model, _resolve_model
from arena.config import Settings
from arena.providers.registry import (
    canonical_provider,
    default_model_for_provider,
    get_provider_spec,
    provider_api_key_from_settings,
    provider_base_url_from_settings,
    provider_has_credentials,
    provider_has_direct_api_key,
    list_helper_provider_specs,
)


def _provider_for_agent_id(settings: Settings, agent_id: str) -> str:
    ac = settings.agent_configs.get(agent_id)
    if ac and str(ac.provider or "").strip():
        return canonical_provider(str(ac.provider or "").strip())
    return canonical_provider(str(agent_id or "").strip().lower())


def _configured_model_for_provider(settings: Settings, provider: str) -> str:
    canonical = canonical_provider(provider) or str(provider or "").strip().lower()
    if not canonical:
        return ""
    for agent_id in settings.agent_ids:
        ac = settings.agent_configs.get(agent_id)
        if ac is None:
            continue
        if canonical_provider(str(ac.provider or "").strip()) != canonical:
            continue
        model_id = str(ac.model or "").strip()
        if model_id:
            return model_id
    return ""


def _model_override_map(settings: Settings, attr: str) -> dict[str, str]:
    raw = getattr(settings, attr, {})
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw.items():
        provider = canonical_provider(str(key or "").strip().lower())
        model = str(value or "").strip()
        if provider and model:
            out[provider] = model
    return out


def _economical_memory_model(provider: str, model_id: str) -> str:
    """Derives a cheaper memory-maintenance model from a trading model."""
    canonical = canonical_provider(provider) or str(provider or "").strip().lower()
    token = str(model_id or "").strip()
    if "/" in token:
        prefix, suffix = token.split("/", 1)
        if prefix in {"openai", "anthropic", "gemini"}:
            token = suffix
    if not token:
        return ""

    if canonical == "gpt":
        if token.endswith("-pro"):
            return token[: -len("-pro")]
        return token

    if canonical == "claude":
        if "opus" in token:
            derived = token.replace("opus", "sonnet", 1)
            # Keep derived defaults conservative. Some forward-looking Opus
            # aliases appear in tenant configs before the matching Sonnet
            # helper model is available on the direct API.
            if derived.startswith("claude-sonnet-4-7"):
                return "claude-sonnet-4-6"
            return derived
        return token

    if canonical == "gemini":
        if token.startswith("gemini-3.1-"):
            token = token.replace("gemini-3.1-", "gemini-3-", 1)
        if "-pro-" in token:
            return token.replace("-pro-", "-flash-", 1)
        if token.endswith("-pro"):
            return token[: -len("-pro")] + "-flash"
        return token

    return token


def suggest_memory_compaction_model_for_agent(settings: Settings, agent_id: str) -> str:
    """Returns the economical memory model suggested for one agent's provider/model."""
    clean_agent_id = str(agent_id or "").strip().lower()
    ac = settings.agent_configs.get(clean_agent_id)
    if ac is not None:
        model_id = str(ac.model or "").strip()
        provider = canonical_provider(str(ac.provider or "").strip())
        if provider and model_id:
            return _economical_memory_model(provider, model_id)
    provider = _provider_for_agent_id(settings, clean_agent_id)
    return suggest_memory_compaction_model(settings, provider)


def suggest_memory_compaction_model(settings: Settings, provider: str) -> str:
    """Returns the default memory-compaction model before explicit overrides."""
    spec = get_provider_spec(provider)
    if spec is None:
        return ""
    model_id = (
        str(_configured_model_for_provider(settings, spec.provider_id) or "").strip()
        or str(default_model_for_provider(settings, spec.provider_id) or "").strip()
    )
    return _economical_memory_model(spec.provider_id, model_id)


def _format_helper_model_token(
    settings: Settings,
    provider: str,
    model_id: str,
    *,
    direct_only: bool = False,
) -> str:
    spec = get_provider_spec(provider)
    if spec is None:
        raise ValueError(f"Unsupported helper provider: {provider}")

    clean_model_id = str(model_id or "").strip()
    if spec.transport == "gemini_native":
        clean_model_id = _normalize_gemini_model(clean_model_id)
        if not clean_model_id:
            raise ValueError(f"model is required for helper provider={spec.provider_id}")
        return clean_model_id if "/" in clean_model_id else f"gemini/{clean_model_id}"

    if spec.transport == "anthropic_native":
        if not clean_model_id:
            raise ValueError(f"model is required for helper provider={spec.provider_id}")
        if direct_only or not settings.anthropic_use_vertexai:
            return clean_model_id if "/" in clean_model_id else f"anthropic/{clean_model_id}"
        return clean_model_id

    if spec.transport == "openai_compatible":
        if not clean_model_id:
            raise ValueError(f"model is required for helper provider={spec.provider_id}")
        prefix = str(spec.litellm_provider or "openai").strip()
        return clean_model_id if "/" in clean_model_id else f"{prefix}/{clean_model_id}"

    raise ValueError(f"Unsupported helper transport: provider={provider} transport={spec.transport}")


def select_helper_provider(settings: Settings, *, direct_only: bool = False) -> str:
    """Selects one provider for non-trading helper agents.

    Preference order follows configured trading agents so single-agent tenants keep
    the same provider family for research/compaction.
    """
    checker = (
        (lambda provider: provider_has_direct_api_key(settings, provider))
        if direct_only
        else (lambda provider: provider_has_credentials(settings, provider))
    )
    for agent_id in settings.agent_ids:
        provider = _provider_for_agent_id(settings, agent_id)
        if provider and checker(provider):
            return provider

    for spec in list_helper_provider_specs():
        if checker(spec.provider_id):
            return spec.provider_id

    raise ValueError("No configured LLM provider available for helper agents")


def select_helper_provider_for_agent(settings: Settings, agent_id: str, *, direct_only: bool = False) -> str:
    """Selects the agent's provider for helper work when its direct credentials exist."""
    provider = _provider_for_agent_id(settings, str(agent_id or "").strip().lower())
    checker = (
        (lambda provider_id: provider_has_direct_api_key(settings, provider_id))
        if direct_only
        else (lambda provider_id: provider_has_credentials(settings, provider_id))
    )
    if provider and checker(provider):
        return provider
    return select_helper_provider(settings, direct_only=direct_only)


def resolve_helper_model_token(settings: Settings, provider: str, *, direct_only: bool = False) -> str:
    spec = get_provider_spec(provider)
    if spec is None:
        raise ValueError(f"Unsupported helper provider: {provider}")

    model_id = (
        str(_configured_model_for_provider(settings, spec.provider_id) or "").strip()
        or str(default_model_for_provider(settings, spec.provider_id) or "").strip()
    )

    return _format_helper_model_token(settings, spec.provider_id, model_id, direct_only=direct_only)


def resolve_memory_compaction_model_token(
    settings: Settings,
    provider: str,
    *,
    agent_id: str | None = None,
    direct_only: bool = False,
) -> str:
    spec = get_provider_spec(provider)
    if spec is None:
        raise ValueError(f"Unsupported helper provider: {provider}")
    if agent_id:
        clean_agent_id = str(agent_id or "").strip().lower()
        agent_provider = _provider_for_agent_id(settings, clean_agent_id)
        ac = settings.agent_configs.get(clean_agent_id)
        agent_model = str(getattr(ac, "memory_compaction_model", "") or "").strip() if ac is not None else ""
        if agent_provider == spec.provider_id and agent_model:
            return _format_helper_model_token(settings, spec.provider_id, agent_model, direct_only=direct_only)
    overrides = _model_override_map(settings, "memory_compaction_models")
    if overrides.get(spec.provider_id):
        return _format_helper_model_token(
            settings,
            spec.provider_id,
            overrides[spec.provider_id],
            direct_only=direct_only,
        )
    if agent_id:
        suggested = (
            suggest_memory_compaction_model_for_agent(settings, agent_id)
            if _provider_for_agent_id(settings, str(agent_id or "").strip().lower()) == spec.provider_id
            else ""
        )
        if suggested:
            return _format_helper_model_token(settings, spec.provider_id, suggested, direct_only=direct_only)
    model_id = overrides.get(spec.provider_id) or suggest_memory_compaction_model(settings, spec.provider_id)
    return _format_helper_model_token(settings, spec.provider_id, model_id, direct_only=direct_only)


def resolve_helper_api_key(settings: Settings, provider: str) -> str:
    return provider_api_key_from_settings(settings, provider)


def resolve_helper_base_url(settings: Settings, provider: str) -> str:
    return provider_base_url_from_settings(settings, provider)


def build_helper_model(settings: Settings):
    provider = select_helper_provider(settings)
    return provider, _resolve_model(provider, settings)
