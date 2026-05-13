from __future__ import annotations

from typing import Any, Iterable, Mapping

from arena.config import Settings
from arena.providers.registry import canonical_provider, default_model_for_provider


_CHAT_MODEL_ALIASES: dict[tuple[str, str], str] = {
    ("gemini", "gemini-3.1-flash-preview"): "gemini-3-flash-preview",
}

DEFAULT_CHEAP_CHAT_MODEL_BY_PROVIDER: dict[str, str] = {
    "gpt": "gpt-5.4-mini",
    "gemini": "gemini-3-flash-preview",
    "claude": "claude-haiku-4-5-20251001",
}


def normalize_chat_model_selection(provider: str | None, model: str | None) -> str:
    """Normalizes chat model tokens that providers have renamed or removed."""
    provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
    model_token = str(model or "").strip()
    if model_token.startswith("models/"):
        model_token = model_token.split("/", 1)[1].strip()
    if not provider_token or not model_token:
        return model_token
    return _CHAT_MODEL_ALIASES.get((provider_token, model_token), model_token)


def chat_model_routing_config(chat_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Returns the optional model-routing config for investment chat."""
    routing = (chat_config or {}).get("model_routing") if isinstance(chat_config, Mapping) else None
    return dict(routing) if isinstance(routing, Mapping) else {}


def cheap_chat_model_for_provider(
    provider: str | None,
    *,
    chat_config: Mapping[str, Any] | None = None,
) -> str:
    """Returns the low-cost router/utility model for the same provider as the advisor."""
    provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
    if not provider_token:
        return ""
    routing = chat_model_routing_config(chat_config)
    configured_by_provider = routing.get("cheap_model_by_provider")
    configured = ""
    if isinstance(configured_by_provider, Mapping):
        configured = str(configured_by_provider.get(provider_token) or "").strip()
    if not configured:
        configured = str(routing.get("cheap_model") or "").strip()
    if not configured:
        configured = DEFAULT_CHEAP_CHAT_MODEL_BY_PROVIDER.get(provider_token, "")
    return normalize_chat_model_selection(provider_token, configured)


def normalize_stored_advisor_model_selection(
    provider: str | None,
    model: str | None,
    *,
    advisor_default_model: str = "",
    chat_config: Mapping[str, Any] | None = None,
) -> str:
    """Normalizes stored/session advisor models without overriding the tenant agent model."""
    provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
    model_token = normalize_chat_model_selection(provider_token, model)
    advisor_default = normalize_chat_model_selection(provider_token, advisor_default_model)
    if model_token and advisor_default and model_token != advisor_default:
        return ""
    return model_token


def tenant_default_chat_selection(
    settings: Settings,
    *,
    allowed_providers: Iterable[str] | None = None,
) -> tuple[str, str]:
    """Returns the tenant's default investment-chat provider/model from active agents."""
    allowed = {
        canonical_provider(provider) or str(provider or "").strip().lower()
        for provider in (allowed_providers or [])
        if str(provider or "").strip()
    }

    def _selection_for_agent(agent_id: str) -> tuple[str, str] | None:
        agent = str(agent_id or "").strip().lower()
        if not agent:
            return None
        config = (getattr(settings, "agent_configs", {}) or {}).get(agent)
        provider = canonical_provider(getattr(config, "provider", "") if config is not None else "") or canonical_provider(agent)
        if not provider:
            return None
        if allowed and provider not in allowed:
            return None
        model = str(getattr(config, "model", "") if config is not None else "").strip()
        if not model:
            model = default_model_for_provider(settings, provider)
        return provider, normalize_chat_model_selection(provider, model)

    for agent_id in getattr(settings, "agent_ids", []) or []:
        selected = _selection_for_agent(str(agent_id))
        if selected is not None:
            return selected

    for agent_id in (getattr(settings, "agent_configs", {}) or {}).keys():
        selected = _selection_for_agent(str(agent_id))
        if selected is not None:
            return selected

    return "", ""
