from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LlmPricing:
    provider: str
    pricing_model: str
    input_usd_per_mtok: float
    cache_read_usd_per_mtok: float | None
    cache_write_usd_per_mtok: float | None
    output_usd_per_mtok: float
    pricing_status: str = "estimated"


@dataclass(frozen=True)
class PricingRule:
    provider: str
    model_id: str | None
    model_pattern: re.Pattern[str] | None
    pricing: LlmPricing


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _normalize_provider(provider: str | None, model: str | None) -> str:
    text = f"{provider or ''} {model or ''}".strip().lower()
    if any(token in text for token in ("anthropic", "claude")):
        return "anthropic"
    if any(token in text for token in ("gemini", "google", "vertex")):
        return "gemini"
    if any(token in text for token in ("openai", "gpt", "o1", "o3", "o4")):
        return "openai"
    return str(provider or "").strip().lower()


def _normalize_model(model: str | None) -> str:
    normalized = str(model or "").strip().lower()
    normalized = re.sub(r"^(models/|openai/|anthropic/|google/|vertex_ai/)", "", normalized)
    return normalized


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _catalog_payload_from_env() -> dict[str, Any] | None:
    raw = str(os.getenv("ARENA_LLM_PRICING_CATALOG_JSON") or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    path = str(os.getenv("ARENA_LLM_PRICING_CATALOG_PATH") or "").strip()
    if path:
        try:
            parsed = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _default_catalog_payload() -> dict[str, Any]:
    with resources.files("arena.providers").joinpath("llm_pricing_catalog.json").open(
        "r",
        encoding="utf-8",
    ) as handle:
        parsed = json.load(handle)
    return parsed if isinstance(parsed, dict) else {"rules": []}


def _rules_from_payload(payload: dict[str, Any], *, status_default: str = "estimated") -> tuple[PricingRule, ...]:
    rules: list[PricingRule] = []
    for item in payload.get("rules") or []:
        if not isinstance(item, dict):
            continue
        model_id = _normalize_model(str(item.get("model_id") or item.get("model") or ""))
        pattern_text = str(item.get("model_pattern") or "").strip()
        match_key = model_id or pattern_text
        provider = _normalize_provider(str(item.get("provider") or ""), match_key)
        pricing_model = str(item.get("pricing_model") or model_id or pattern_text or "").strip()
        input_rate = _safe_float(item.get("input_usd_per_mtok"))
        output_rate = _safe_float(item.get("output_usd_per_mtok"))
        if not provider or not match_key or input_rate is None or output_rate is None:
            continue
        pattern: re.Pattern[str] | None = None
        if pattern_text:
            try:
                pattern = re.compile(pattern_text, re.IGNORECASE)
            except re.error:
                continue
        rules.append(
            PricingRule(
                provider=provider,
                model_id=model_id or None,
                model_pattern=pattern,
                pricing=LlmPricing(
                    provider=provider,
                    pricing_model=pricing_model,
                    input_usd_per_mtok=input_rate,
                    cache_read_usd_per_mtok=_safe_float(item.get("cache_read_usd_per_mtok")),
                    cache_write_usd_per_mtok=_safe_float(item.get("cache_write_usd_per_mtok")),
                    output_usd_per_mtok=output_rate,
                    pricing_status=str(item.get("pricing_status") or status_default).strip().lower() or status_default,
                ),
            )
        )
    return tuple(rules)


@lru_cache(maxsize=1)
def pricing_catalog_rules() -> tuple[PricingRule, ...]:
    payloads: list[tuple[dict[str, Any], str]] = []
    env_payload = _catalog_payload_from_env()
    if env_payload is not None:
        payloads.append((env_payload, "configured"))
    payloads.append((_default_catalog_payload(), "estimated"))

    rules: list[PricingRule] = []
    for payload, status_default in payloads:
        rules.extend(_rules_from_payload(payload, status_default=status_default))
    return tuple(rules)


def reset_pricing_catalog_cache() -> None:
    pricing_catalog_rules.cache_clear()


def pricing_for_model(provider: str | None, model: str | None) -> LlmPricing | None:
    normalized_provider = _normalize_provider(provider, model)
    normalized_model = _normalize_model(model)
    for rule in pricing_catalog_rules():
        if rule.provider != normalized_provider:
            continue
        if rule.model_id is not None and rule.model_id == normalized_model:
            return rule.pricing
        if rule.model_pattern is not None and rule.model_pattern.search(normalized_model):
            return rule.pricing
    return None


def _completion_tokens(usage: dict[str, Any]) -> int:
    if "completion_tokens" in usage:
        return _safe_int(usage.get("completion_tokens"))
    if "candidates_token_count" in usage:
        return _safe_int(usage.get("candidates_token_count"))
    if "candidates_tokens" in usage:
        return _safe_int(usage.get("candidates_tokens"))
    if "output_tokens" in usage:
        return _safe_int(usage.get("output_tokens"))
    return 0


def estimate_llm_cost(provider: str | None, model: str | None, usage: dict[str, Any]) -> dict[str, Any]:
    prompt_tokens = _safe_int(usage.get("prompt_tokens") or usage.get("prompt_token_count"))
    official_input_tokens = _safe_int(usage.get("input_tokens"))
    cached_tokens = _safe_int(
        usage.get("cache_read_input_tokens")
        or usage.get("cached_input_tokens")
        or usage.get("cached_tokens")
        or usage.get("cached_content_token_count")
    )
    cache_write_tokens = _safe_int(
        usage.get("cache_write_input_tokens")
        or usage.get("cache_creation_input_tokens")
    )
    prompt_style_usage = prompt_tokens > 0

    if prompt_style_usage:
        input_tokens = prompt_tokens
        cache_read_tokens = min(cached_tokens, input_tokens)
        cache_write_tokens = min(cache_write_tokens, max(input_tokens - cache_read_tokens, 0))
        uncached_input_tokens = max(input_tokens - cache_read_tokens - cache_write_tokens, 0)
    else:
        cache_read_tokens = cached_tokens
        input_tokens = official_input_tokens + cache_read_tokens + cache_write_tokens
        uncached_input_tokens = official_input_tokens

    thinking_tokens = _safe_int(
        usage.get("thinking_tokens")
        or usage.get("thoughts_token_count")
        or usage.get("reasoning_tokens")
    )
    completion_tokens = _completion_tokens(usage)
    output_tokens = completion_tokens + thinking_tokens
    raw_total_tokens = input_tokens + output_tokens

    pricing = pricing_for_model(provider, model)
    input_cost_usd = 0.0
    cache_read_cost_usd = 0.0
    cache_write_cost_usd = 0.0
    output_cost_usd = 0.0
    pricing_status = "unknown"
    pricing_model = _normalize_model(model)

    if pricing is not None:
        pricing_status = pricing.pricing_status
        pricing_model = pricing.pricing_model
        input_cost_usd = uncached_input_tokens * pricing.input_usd_per_mtok / 1_000_000.0
        cache_read_rate = pricing.cache_read_usd_per_mtok
        if cache_read_rate is None:
            cache_read_rate = pricing.input_usd_per_mtok
            if cache_read_tokens:
                pricing_status = "partial"
        cache_read_cost_usd = cache_read_tokens * cache_read_rate / 1_000_000.0
        if pricing.cache_write_usd_per_mtok is None:
            if cache_write_tokens:
                pricing_status = "partial"
        else:
            cache_write_cost_usd = cache_write_tokens * pricing.cache_write_usd_per_mtok / 1_000_000.0
        output_cost_usd = output_tokens * pricing.output_usd_per_mtok / 1_000_000.0

    estimated_cost_usd = input_cost_usd + cache_read_cost_usd + cache_write_cost_usd + output_cost_usd
    cache_ratio = round(cache_read_tokens / input_tokens * 100.0, 1) if input_tokens > 0 else 0.0
    return {
        "provider": _normalize_provider(provider, model),
        "pricing_model": pricing_model,
        "input_tokens": input_tokens,
        "prompt_tokens": input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "cached_input_tokens": cache_read_tokens,
        "cached_tokens": cache_read_tokens,
        "cache_write_input_tokens": cache_write_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
        "output_tokens": output_tokens,
        "raw_total_tokens": raw_total_tokens,
        "total_tokens": raw_total_tokens,
        "cache_ratio": cache_ratio,
        "input_cost_usd": input_cost_usd,
        "cache_read_cost_usd": cache_read_cost_usd,
        "cached_input_cost_usd": cache_read_cost_usd,
        "cache_write_cost_usd": cache_write_cost_usd,
        "output_cost_usd": output_cost_usd,
        "estimated_cost_usd": estimated_cost_usd,
        "pricing_status": pricing_status,
    }
