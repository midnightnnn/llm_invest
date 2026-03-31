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


def resolve_helper_model_token(settings: Settings, provider: str, *, direct_only: bool = False) -> str:
    spec = get_provider_spec(provider)
    if spec is None:
        raise ValueError(f"Unsupported helper provider: {provider}")

    model_id = (
        str(_configured_model_for_provider(settings, spec.provider_id) or "").strip()
        or str(default_model_for_provider(settings, spec.provider_id) or "").strip()
    )

    if spec.transport == "gemini_native":
        model_id = _normalize_gemini_model(model_id)
        if not model_id:
            raise ValueError(f"model is required for helper provider={spec.provider_id}")
        return model_id if "/" in model_id else f"gemini/{model_id}"

    if spec.transport == "anthropic_native":
        if not model_id:
            raise ValueError(f"model is required for helper provider={spec.provider_id}")
        if direct_only or not settings.anthropic_use_vertexai:
            return model_id if "/" in model_id else f"anthropic/{model_id}"
        return model_id

    if spec.transport == "openai_compatible":
        if not model_id:
            raise ValueError(f"model is required for helper provider={spec.provider_id}")
        prefix = str(spec.litellm_provider or "openai").strip()
        return model_id if "/" in model_id else f"{prefix}/{model_id}"

    raise ValueError(f"Unsupported helper transport: provider={provider} transport={spec.transport}")


def resolve_helper_api_key(settings: Settings, provider: str) -> str:
    return provider_api_key_from_settings(settings, provider)


def resolve_helper_base_url(settings: Settings, provider: str) -> str:
    return provider_base_url_from_settings(settings, provider)


def build_helper_model(settings: Settings):
    provider = select_helper_provider(settings)
    return provider, _resolve_model(provider, settings)
