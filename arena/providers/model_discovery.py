from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any

import requests

from arena.providers.registry import canonical_provider


class ModelDiscoveryError(RuntimeError):
    """Raised when provider model discovery cannot produce usable chat models."""


@dataclass(frozen=True)
class ProviderModelOptions:
    provider: str
    advisor_models: list[str]
    router_models: list[str]
    utility_models: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "advisor_models": list(self.advisor_models),
            "router_models": list(self.router_models),
            "utility_models": list(self.utility_models),
        }


MODEL_OPTIONS_CONFIG_KEY = "investment_chat_model_options"

_NON_CHAT_MARKERS = (
    "audio",
    "bidi",
    "codex-mini-latest",
    "dall",
    "edit",
    "embed",
    "embedding",
    "image",
    "imagen",
    "moderation",
    "rerank",
    "realtime",
    "speech",
    "tts",
    "transcribe",
    "veo",
    "whisper",
)

_CHEAP_MARKERS = {
    "gpt": ("mini", "nano", "small", "fast", "lite"),
    "gemini": ("flash", "lite"),
    "claude": ("haiku",),
}

_MODEL_ALIASES: dict[tuple[str, str], str] = {
    ("gemini", "gemini-3.1-flash-preview"): "gemini-3-flash-preview",
}


def _provider_token(provider: str) -> str:
    return canonical_provider(provider) or str(provider or "").strip().lower()


def _clean_model_id(provider: str, model_id: str) -> str:
    token = str(model_id or "").strip()
    if token.startswith("models/"):
        token = token.split("/", 1)[1]
    return _MODEL_ALIASES.get((provider, token), token)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        token = str(value or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _looks_like_provider_chat_model(provider: str, model_id: str) -> bool:
    token = str(model_id or "").strip().lower()
    if not token or any(marker in token for marker in _NON_CHAT_MARKERS):
        return False
    if provider == "gpt":
        return token.startswith(("gpt-", "o", "chatgpt-"))
    if provider == "gemini":
        return token.startswith("gemini-")
    if provider == "claude":
        return token.startswith("claude-")
    return False


def _cheap_models(provider: str, models: list[str]) -> list[str]:
    markers = _CHEAP_MARKERS.get(provider, ())
    cheap = [model for model in models if any(marker in model.lower() for marker in markers)]
    return cheap or list(models)


def _response_json(response: Any) -> dict[str, Any]:
    try:
        response.raise_for_status()
    except Exception as exc:
        status = getattr(response, "status_code", "")
        raise ModelDiscoveryError(f"model list request failed: status={status}") from exc
    try:
        payload = response.json()
    except Exception as exc:
        raise ModelDiscoveryError("model list response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ModelDiscoveryError("model list response must be a JSON object")
    return payload


def _openai_model_ids(session: Any, api_key: str, timeout: float) -> list[str]:
    response = session.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    payload = _response_json(response)
    return [str(item.get("id") or "") for item in payload.get("data") or [] if isinstance(item, dict)]


def _gemini_model_ids(session: Any, api_key: str, timeout: float) -> list[str]:
    ids: list[str] = []
    page_token = ""
    for _ in range(20):
        params = {"key": api_key, "pageSize": 1000}
        if page_token:
            params["pageToken"] = page_token
        response = session.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params=params,
            timeout=timeout,
        )
        payload = _response_json(response)
        for item in payload.get("models") or []:
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods")
            if isinstance(methods, list) and "generateContent" not in {str(method) for method in methods}:
                continue
            ids.append(str(item.get("name") or item.get("baseModelId") or ""))
        page_token = str(payload.get("nextPageToken") or "").strip()
        if not page_token:
            break
    return ids


def _claude_model_ids(session: Any, api_key: str, timeout: float) -> list[str]:
    ids: list[str] = []
    after_id = ""
    for _ in range(20):
        params = {"limit": 1000}
        if after_id:
            params["after_id"] = after_id
        response = session.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            params=params,
            timeout=timeout,
        )
        payload = _response_json(response)
        ids.extend([str(item.get("id") or "") for item in payload.get("data") or [] if isinstance(item, dict)])
        if not bool(payload.get("has_more")):
            break
        after_id = str(payload.get("last_id") or "").strip()
        if not after_id:
            break
    return ids


def discover_model_options_with_api_key(
    provider: str,
    api_key: str,
    *,
    timeout: float = 10.0,
    session: Any | None = None,
) -> dict[str, Any]:
    provider_token = _provider_token(provider)
    if provider_token not in {"gpt", "gemini", "claude"}:
        raise ModelDiscoveryError("unsupported provider")
    api_key_token = str(api_key or "").strip()
    if not api_key_token:
        raise ModelDiscoveryError("api_key is required")
    http = session or requests.Session()
    try:
        if provider_token == "gpt":
            raw_ids = _openai_model_ids(http, api_key_token, timeout)
        elif provider_token == "gemini":
            raw_ids = _gemini_model_ids(http, api_key_token, timeout)
        else:
            raw_ids = _claude_model_ids(http, api_key_token, timeout)
    except ModelDiscoveryError:
        raise
    except Exception as exc:
        raise ModelDiscoveryError(f"model list request failed: {exc}") from exc

    advisor_models = _unique(
        [
            model
            for raw_id in raw_ids
            if (model := _clean_model_id(provider_token, raw_id))
            and _looks_like_provider_chat_model(provider_token, model)
        ]
    )
    if not advisor_models:
        raise ModelDiscoveryError("no supported chat models were returned by the provider")
    cheap_models = _cheap_models(provider_token, advisor_models)
    return ProviderModelOptions(
        provider=provider_token,
        advisor_models=advisor_models,
        router_models=cheap_models,
        utility_models=cheap_models,
    ).as_dict()


def discover_saved_model_options(credential_store: Any, *, tenant_id: str, provider: str) -> dict[str, Any]:
    provider_token = _provider_token(provider)
    loader = getattr(credential_store, "model_api_key", None)
    if not callable(loader):
        raise ModelDiscoveryError("credential store cannot read provider API keys")
    api_key = str(loader(tenant_id=tenant_id, provider=provider_token) or "").strip()
    if not api_key:
        raise ModelDiscoveryError("stored provider API key is missing")
    return discover_model_options_with_api_key(provider_token, api_key)


def load_model_options_catalog(repo: Any, *, tenant_id: str) -> dict[str, Any]:
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return {}
    try:
        raw = getter(tenant_id, MODEL_OPTIONS_CONFIG_KEY)
    except TypeError:
        raw = getter(tenant_id=tenant_id, config_key=MODEL_OPTIONS_CONFIG_KEY)
    except Exception:
        return {}
    try:
        parsed = json.loads(str(raw or ""))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def save_model_options_catalog(
    repo: Any,
    *,
    tenant_id: str,
    options: dict[str, Any],
    updated_by: str = "model_discovery",
) -> None:
    setter = getattr(repo, "set_config", None)
    if not callable(setter):
        return
    provider = _provider_token(str(options.get("provider") or ""))
    if not provider:
        return
    catalog = load_model_options_catalog(repo, tenant_id=tenant_id)
    providers = catalog.get("providers") if isinstance(catalog.get("providers"), dict) else {}
    providers = dict(providers or {})
    providers[provider] = {
        "provider": provider,
        "advisor_models": [str(item) for item in options.get("advisor_models") or [] if str(item).strip()],
        "router_models": [str(item) for item in options.get("router_models") or [] if str(item).strip()],
        "utility_models": [str(item) for item in options.get("utility_models") or [] if str(item).strip()],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = {"providers": providers}
    setter(tenant_id, MODEL_OPTIONS_CONFIG_KEY, json.dumps(payload, ensure_ascii=False), updated_by=updated_by)


def model_option_sets(options: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    return (
        {str(item or "").strip() for item in options.get("advisor_models") or [] if str(item or "").strip()},
        {str(item or "").strip() for item in options.get("router_models") or [] if str(item or "").strip()},
        {str(item or "").strip() for item in options.get("utility_models") or [] if str(item or "").strip()},
    )
