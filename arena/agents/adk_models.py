from __future__ import annotations

import logging
import os
from typing import Any

from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm

from arena.config import Settings
from arena.providers.registry import (
    default_model_for_provider,
    get_provider_spec,
    provider_api_key_from_settings,
    provider_base_url_from_settings,
    provider_has_credentials,
)

logger = logging.getLogger(__name__)


def _normalize_vertex_anthropic_model(model_id: str) -> str:
    """Normalizes Anthropic model ids for Vertex AI transport."""
    raw = model_id.strip()
    token = raw
    if token.startswith("vertex_ai/"):
        token = token.split("/", 1)[1]
    elif "/" in token:
        token = token.split("/", 1)[1]

    alias_map = {
        "claude-sonnet-4-6": "claude-sonnet-4-5",
    }
    normalized = alias_map.get(token, token).strip()
    if normalized != token:
        logger.warning(
            "[yellow]Anthropic Vertex model alias remapped[/yellow] from=%s to=%s",
            token,
            normalized,
        )
    if not normalized:
        raise ValueError("ANTHROPIC_MODEL is empty after normalization")
    return f"vertex_ai/{normalized}"


def _is_vertex_model_access_error(exc: Exception) -> bool:
    """Returns True for Vertex partner-model unavailable errors (access/quota)."""
    text = f"{type(exc).__name__}: {exc}".strip().lower()
    if not text:
        return False
    if "vertex_aiexception" not in text and "publishers/anthropic/models" not in text:
        return False
    markers = [
        "publisher model",
        "was not found",
        "does not have access",
        "permission denied",
        "not_found",
        "404",
        "resource exhausted",
        "resource_exhausted",
        "quota",
        "429",
    ]
    return any(marker in text for marker in markers)


def _normalize_gemini_model(model_id: str) -> str:
    """Normalizes Gemini model id across SDK/LiteLLM paths."""
    token = str(model_id or "").strip()
    if token.startswith("models/"):
        token = token.split("/", 1)[1]
    if token.startswith("gemini/"):
        token = token.split("/", 1)[1]
    return token


def _is_gemini_quota_error(exc: Exception) -> bool:
    """Returns True for Gemini quota/rate-limit style transient errors."""
    text = f"{type(exc).__name__}: {exc}".strip().lower()
    if not text:
        return False
    markers = [
        "resource_exhausted",
        "resource exhausted",
        "429",
        "rate limit",
        "too many requests",
        "quota",
    ]
    return any(marker in text for marker in markers)


def _resolve_model(provider: str, settings: Settings, *, model_override: str = "") -> Any:
    """Builds ADK model objects per provider configuration."""
    key = str(provider or "").strip().lower()
    if key == "gemini_direct":
        model_id = _normalize_gemini_model(
            (model_override.strip() if model_override else "")
            or default_model_for_provider(settings, "gemini")
        )
        if not model_id:
            raise ValueError("GEMINI_MODEL is required for gemini agent")
        litellm_model = model_id if "/" in model_id else f"gemini/{model_id}"
        kwargs: dict[str, Any] = {}
        api_key = provider_api_key_from_settings(settings, "gemini")
        if api_key:
            kwargs["api_key"] = api_key
        return LiteLlm(litellm_model, **kwargs)

    spec = get_provider_spec(key)
    if spec is None or not spec.supports_adk:
        raise ValueError(f"Unsupported ADK provider: {provider}")

    configured_model = (model_override.strip() if model_override else "") or default_model_for_provider(
        settings,
        spec.provider_id,
    )

    if spec.transport == "gemini_native":
        model_id = _normalize_gemini_model(configured_model)
        if not model_id:
            raise ValueError("GEMINI_MODEL is required for gemini agent")
        if settings.gemini_api_key and not os.getenv("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
        return Gemini(model=model_id)

    if spec.transport == "anthropic_native":
        model_id = configured_model.strip()
        if not model_id:
            raise ValueError("ANTHROPIC_MODEL is required for claude agent")
        kwargs: dict[str, Any] = {
            "cache_control_injection_points": [
                {"location": "message", "role": "system"},
            ],
        }
        if settings.anthropic_use_vertexai:
            model_id = _normalize_vertex_anthropic_model(model_id)
            if settings.google_cloud_project and not os.getenv("VERTEX_PROJECT"):
                os.environ["VERTEX_PROJECT"] = settings.google_cloud_project
            vertex_location = (
                os.getenv("VERTEX_LOCATION", "").strip()
                or os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
                or "global"
            )
            if not os.getenv("VERTEX_LOCATION"):
                os.environ["VERTEX_LOCATION"] = vertex_location
        else:
            if "/" not in model_id:
                model_id = f"{spec.litellm_provider or 'anthropic'}/{model_id}"
            api_key = provider_api_key_from_settings(settings, spec.provider_id)
            if api_key:
                kwargs["api_key"] = api_key
        return LiteLlm(model_id, **kwargs)

    if spec.transport == "openai_compatible":
        model_id = configured_model.strip()
        if not model_id:
            raise ValueError(f"model is required for provider '{spec.provider_id}'")
        litellm_provider = str(spec.litellm_provider or spec.provider_id).strip() or spec.provider_id
        if "/" not in model_id:
            model_id = f"{litellm_provider}/{model_id}"
        kwargs: dict[str, Any] = {}
        api_key = provider_api_key_from_settings(settings, spec.provider_id)
        if api_key:
            kwargs["api_key"] = api_key
        base_url = provider_base_url_from_settings(settings, spec.provider_id)
        if base_url:
            kwargs["base_url"] = base_url
        return LiteLlm(model_id, **kwargs)

    raise ValueError(
        f"Unsupported ADK provider transport: provider={spec.provider_id} transport={spec.transport}"
    )


def _has_credentials(provider: str, settings: Settings) -> bool:
    """Returns whether the provider has usable credentials in current settings."""
    return provider_has_credentials(settings, provider)
