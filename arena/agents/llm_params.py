"""Per-provider LLM parameter validation and schema.

Each provider has a distinct set of supported knobs. Rather than forcing a
synthetic unified vocabulary, we store provider-native SDK field names in a
flat dict and let the validator filter/clamp per provider. This keeps the
mapping to the SDK layer straightforward (no translation table rot).

Supported fields (by provider):
  - claude (Anthropic): effort, temperature, top_p, max_tokens
  - gpt    (OpenAI):    reasoning_effort, verbosity, max_completion_tokens
  - gemini (Google):    thinking_level, thinking_budget, temperature, top_p,
                        top_k, max_output_tokens
"""
from __future__ import annotations

from typing import Any


# --- enums (source of truth — keep in sync with UI dropdowns) -----------------

ANTHROPIC_EFFORTS = ("low", "medium", "high", "xhigh", "max")
OPENAI_REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
OPENAI_VERBOSITIES = ("low", "medium", "high")
GEMINI_THINKING_LEVELS = ("low", "medium", "high")


# --- helpers ------------------------------------------------------------------


def _clean_str(value: Any) -> str:
    return str(value or "").strip().lower()


def _coerce_float(value: Any, *, low: float, high: float) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN guard
        return None
    return max(low, min(high, num))


def _coerce_int(value: Any, *, low: int, high: int) -> int | None:
    try:
        num = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(low, min(high, num))


def _canonical_provider(provider: str) -> str:
    key = _clean_str(provider)
    if key in {"anthropic", "claude"}:
        return "claude"
    if key in {"openai", "gpt"}:
        return "gpt"
    if key in {"google", "gemini"}:
        return "gemini"
    return key


# --- per-provider sanitizers --------------------------------------------------


def _sanitize_claude(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    effort = _clean_str(raw.get("effort"))
    if effort in ANTHROPIC_EFFORTS:
        out["effort"] = effort
    temp = _coerce_float(raw.get("temperature"), low=0.0, high=1.0)
    if temp is not None:
        out["temperature"] = temp
    top_p = _coerce_float(raw.get("top_p"), low=0.0, high=1.0)
    if top_p is not None:
        out["top_p"] = top_p
    max_tokens = _coerce_int(raw.get("max_tokens"), low=1, high=200_000)
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    return out


def _sanitize_openai(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    effort = _clean_str(raw.get("reasoning_effort"))
    if effort in OPENAI_REASONING_EFFORTS:
        out["reasoning_effort"] = effort
    verbosity = _clean_str(raw.get("verbosity"))
    if verbosity in OPENAI_VERBOSITIES:
        out["verbosity"] = verbosity
    max_ct = _coerce_int(raw.get("max_completion_tokens"), low=1, high=200_000)
    if max_ct is not None:
        out["max_completion_tokens"] = max_ct
    return out


def _sanitize_gemini(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    level = _clean_str(raw.get("thinking_level"))
    budget_raw = raw.get("thinking_budget")
    # thinking_level and thinking_budget are mutually exclusive (Gemini 3 constraint)
    if level in GEMINI_THINKING_LEVELS:
        out["thinking_level"] = level
    elif budget_raw is not None:
        # -1 = dynamic; otherwise positive int up to 32k
        try:
            bud = int(float(budget_raw))
            if bud == -1 or 0 <= bud <= 32_768:
                out["thinking_budget"] = bud
        except (TypeError, ValueError):
            pass
    temp = _coerce_float(raw.get("temperature"), low=0.0, high=2.0)
    if temp is not None:
        out["temperature"] = temp
    top_p = _coerce_float(raw.get("top_p"), low=0.0, high=1.0)
    if top_p is not None:
        out["top_p"] = top_p
    top_k = _coerce_int(raw.get("top_k"), low=1, high=40)
    if top_k is not None:
        out["top_k"] = top_k
    max_out = _coerce_int(raw.get("max_output_tokens"), low=1, high=65_536)
    if max_out is not None:
        out["max_output_tokens"] = max_out
    return out


_SANITIZERS = {
    "claude": _sanitize_claude,
    "gpt": _sanitize_openai,
    "gemini": _sanitize_gemini,
}


def sanitize_llm_params(provider: str, raw: Any) -> dict[str, Any]:
    """Returns a provider-filtered, range-clamped copy of llm_params.

    Unknown keys are dropped; out-of-range values are clamped. Empty dict
    signals "use provider defaults" downstream.
    """
    if not isinstance(raw, dict) or not raw:
        return {}
    canonical = _canonical_provider(provider)
    sanitizer = _SANITIZERS.get(canonical)
    if sanitizer is None:
        return {}
    return sanitizer(raw)


def supported_fields(provider: str) -> list[str]:
    """UI helper — returns field names this provider accepts."""
    canonical = _canonical_provider(provider)
    if canonical == "claude":
        return ["effort", "temperature", "top_p", "max_tokens"]
    if canonical == "gpt":
        return ["reasoning_effort", "verbosity", "max_completion_tokens"]
    if canonical == "gemini":
        return [
            "thinking_level",
            "thinking_budget",
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
        ]
    return []
