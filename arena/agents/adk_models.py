from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from google.adk.models import Gemini
from google.adk.models.lite_llm import LiteLlm, LiteLLMClient

from arena.config import Settings
from arena.logging_utils import event_extra, failure_extra
from arena.providers.registry import (
    default_model_for_provider,
    get_provider_spec,
    provider_api_key_from_settings,
    provider_base_url_from_settings,
    provider_has_credentials,
)

logger = logging.getLogger(__name__)


class _InstrumentedLiteLLMClient(LiteLLMClient):
    """Logs the exact LiteLLM acompletion boundary without logging prompts."""

    def __init__(
        self,
        *,
        agent_id: str = "",
        provider: str = "",
        metadata_getter: Callable[[], dict[str, Any]] | None = None,
        delegate: LiteLLMClient | None = None,
    ) -> None:
        self._agent_id = str(agent_id or "").strip()
        self._provider = str(provider or "").strip()
        self._metadata_getter = metadata_getter
        self._delegate = delegate

    def _metadata(self) -> dict[str, Any]:
        if not callable(self._metadata_getter):
            return {}
        try:
            data = self._metadata_getter() or {}
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    async def acompletion(self, model, messages, tools, **kwargs):
        started = time.perf_counter()
        metadata = self._metadata()
        message_count = len(messages) if isinstance(messages, list) else None
        tool_count = len(tools) if isinstance(tools, list) else None
        timeout = kwargs.get("timeout")
        base_fields = {
            **metadata,
            "agent_id": metadata.get("agent_id") or self._agent_id,
            "provider": metadata.get("provider") or self._provider,
            "model": str(model or "").strip(),
            "message_count": message_count,
            "tool_count": tool_count,
            "timeout_seconds": timeout,
            "stream": bool(kwargs.get("stream", False)),
        }
        logger.info(
            "[blue]ADK model acompletion start[/blue] agent=%s provider=%s model=%s",
            base_fields.get("agent_id") or "-",
            base_fields.get("provider") or "-",
            base_fields.get("model") or "-",
            extra=event_extra("adk_model_acompletion_start", **base_fields),
        )
        try:
            if self._delegate is not None:
                response = await self._delegate.acompletion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    **kwargs,
                )
            else:
                response = await super().acompletion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    **kwargs,
                )
        except asyncio.CancelledError as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "[yellow]ADK model acompletion failed[/yellow] agent=%s provider=%s model=%s elapsed=%dms err=%s",
                base_fields.get("agent_id") or "-",
                base_fields.get("provider") or "-",
                base_fields.get("model") or "-",
                elapsed_ms,
                str(exc),
                extra=failure_extra(
                    "adk_model_acompletion_cancelled",
                    exc,
                    **base_fields,
                    elapsed_ms=elapsed_ms,
                ),
            )
            raise
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.warning(
                "[yellow]ADK model acompletion failed[/yellow] agent=%s provider=%s model=%s elapsed=%dms err=%s",
                base_fields.get("agent_id") or "-",
                base_fields.get("provider") or "-",
                base_fields.get("model") or "-",
                elapsed_ms,
                str(exc),
                extra=failure_extra("adk_model_acompletion_error", exc, **base_fields, elapsed_ms=elapsed_ms),
            )
            raise

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "[blue]ADK model acompletion end[/blue] agent=%s provider=%s model=%s elapsed=%dms response_type=%s",
            base_fields.get("agent_id") or "-",
            base_fields.get("provider") or "-",
            base_fields.get("model") or "-",
            elapsed_ms,
            type(response).__name__,
            extra=event_extra(
                "adk_model_acompletion_end",
                **base_fields,
                elapsed_ms=elapsed_ms,
                response_type=type(response).__name__,
            ),
        )
        return response


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


def _anthropic_model_tier(model_id: str) -> tuple[bool, bool, bool]:
    """Returns (is_opus_4_7, supports_effort_param, supports_max_effort) for an Anthropic model id.

    - supports_effort: Opus 4.5+, Sonnet 4.6+
    - supports_max_effort: Opus 4.7/4.6 + Sonnet 4.6 (NOT Opus 4.5)
    - xhigh: Opus 4.7 only (checked via is_opus_4_7)
    """
    token = str(model_id or "").strip().lower()
    if token.startswith("anthropic/") or token.startswith("vertex_ai/"):
        token = token.split("/", 1)[1]
    is_opus_4_7 = token.startswith("claude-opus-4-7")
    supports_effort = (
        token.startswith("claude-opus-4-")
        or token.startswith("claude-sonnet-4-6")
    )
    is_opus_4_5 = token.startswith("claude-opus-4-5")
    supports_max_effort = supports_effort and not is_opus_4_5
    return is_opus_4_7, supports_effort, supports_max_effort


def _clamp_anthropic_effort(effort: str, *, is_opus_4_7: bool, supports_max: bool) -> str:
    """Clamps user-supplied effort to the closest value the model actually supports."""
    value = str(effort or "").strip().lower()
    if value == "xhigh" and not is_opus_4_7:
        return "high"
    if value == "max" and not supports_max:
        return "xhigh" if is_opus_4_7 else "high"
    if value not in {"low", "medium", "high", "xhigh", "max"}:
        return ""
    return value


def _anthropic_runtime_kwargs(model_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Builds per-model LiteLLM runtime kwargs for Anthropic (effort, thinking, sampling).

    Without overrides: Opus 4.7 → effort=xhigh + adaptive thinking; others → effort=high.
    With overrides: user-supplied effort replaces the default (clamped per-tier);
    Opus 4.7 keeps thinking=adaptive always (budget_tokens is removed per API docs).
    """
    is_opus_4_7, supports_effort, supports_max = _anthropic_model_tier(model_id)
    extra: dict[str, Any] = {}
    ov = overrides or {}

    user_effort = _clamp_anthropic_effort(
        str(ov.get("effort") or ""),
        is_opus_4_7=is_opus_4_7,
        supports_max=supports_max,
    )
    if supports_effort:
        default_effort = "xhigh" if is_opus_4_7 else "high"
        extra["output_config"] = {"effort": user_effort or default_effort}
        extra["allowed_openai_params"] = ["output_config"]

    # Opus 4.7 mandates adaptive thinking; budget_tokens is rejected by the API.
    if is_opus_4_7:
        extra["thinking"] = {"type": "adaptive"}

    # Sampling / length — Anthropic has no top_k.
    if (t := ov.get("temperature")) is not None:
        extra["temperature"] = float(t)
    if (p := ov.get("top_p")) is not None:
        extra["top_p"] = float(p)
    if (mx := ov.get("max_tokens")) is not None:
        extra["max_tokens"] = int(mx)
    return extra


def _resolve_model(
    provider: str,
    settings: Settings,
    *,
    model_override: str = "",
    llm_params: dict[str, Any] | None = None,
    model_call_metadata_getter: Callable[[], dict[str, Any]] | None = None,
) -> Any:
    """Builds ADK model objects per provider configuration.

    llm_params is provider-native (already sanitized by arena.agents.llm_params).
    For Gemini the per-agent sampling/thinking flows through GenerateContentConfig
    in build_agent(), not here — we only keep the model object construction here.
    """
    key = str(provider or "").strip().lower()
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
        llm_timeout = max(1, int(settings.timeout_for("trading")))
        kwargs: dict[str, Any] = {
            "cache_control_injection_points": [
                {"location": "message", "role": "system"},
            ],
            "timeout": llm_timeout,
            "llm_client": _InstrumentedLiteLLMClient(
                agent_id=spec.provider_id,
                provider=spec.provider_id,
                metadata_getter=model_call_metadata_getter,
            ),
        }
        kwargs.update(_anthropic_runtime_kwargs(model_id, llm_params))
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
        llm_timeout = max(1, int(settings.timeout_for("trading")))
        litellm_provider = str(spec.litellm_provider or spec.provider_id).strip() or spec.provider_id
        if "/" not in model_id:
            model_id = f"{litellm_provider}/{model_id}"
        kwargs: dict[str, Any] = {
            "timeout": llm_timeout,
            "llm_client": _InstrumentedLiteLLMClient(
                agent_id=spec.provider_id,
                provider=spec.provider_id,
                metadata_getter=model_call_metadata_getter,
            ),
        }
        api_key = provider_api_key_from_settings(settings, spec.provider_id)
        if api_key:
            kwargs["api_key"] = api_key
        base_url = provider_base_url_from_settings(settings, spec.provider_id)
        if base_url:
            kwargs["base_url"] = base_url
        ov = llm_params or {}
        if (eff := ov.get("reasoning_effort")):
            kwargs["reasoning_effort"] = str(eff)
        if (vb := ov.get("verbosity")):
            kwargs["verbosity"] = str(vb)
        if (mx := ov.get("max_completion_tokens")) is not None:
            kwargs["max_completion_tokens"] = int(mx)
        return LiteLlm(model_id, **kwargs)

    raise ValueError(
        f"Unsupported ADK provider transport: provider={spec.provider_id} transport={spec.transport}"
    )


def _has_credentials(provider: str, settings: Settings) -> bool:
    """Returns whether the provider has usable credentials in current settings."""
    return provider_has_credentials(settings, provider)
