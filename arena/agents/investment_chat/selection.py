from __future__ import annotations

from typing import Any, Iterable, Mapping

from arena.config import Settings
from arena.providers.registry import canonical_provider, default_model_for_provider


_CHAT_MODEL_ALIASES: dict[tuple[str, str], str] = {
    ("gemini", "gemini-3.1-flash-preview"): "gemini-3-flash-preview",
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
    """Returns the legacy shared router/utility model override, when configured."""
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
    return normalize_chat_model_selection(provider_token, configured)


def router_chat_model_for_provider(
    provider: str | None,
    *,
    advisor_model: str,
    chat_config: Mapping[str, Any] | None = None,
) -> str:
    """Returns the configured router model, falling back to the advisor model."""
    provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
    routing = chat_model_routing_config(chat_config)
    configured = str(routing.get("router_model") or "").strip() or cheap_chat_model_for_provider(
        provider_token,
        chat_config=chat_config,
    )
    return normalize_chat_model_selection(provider_token, configured or advisor_model)


def utility_chat_model_for_provider(
    provider: str | None,
    *,
    advisor_model: str,
    chat_config: Mapping[str, Any] | None = None,
) -> str:
    """Returns the configured utility model, falling back to router/advisor model."""
    provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
    routing = chat_model_routing_config(chat_config)
    configured = str(routing.get("utility_model") or "").strip() or cheap_chat_model_for_provider(
        provider_token,
        chat_config=chat_config,
    )
    if not configured:
        configured = router_chat_model_for_provider(provider_token, advisor_model=advisor_model, chat_config=chat_config)
    return normalize_chat_model_selection(provider_token, configured or advisor_model)


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
