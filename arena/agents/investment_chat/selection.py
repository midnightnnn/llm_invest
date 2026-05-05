from __future__ import annotations

from typing import Iterable

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
