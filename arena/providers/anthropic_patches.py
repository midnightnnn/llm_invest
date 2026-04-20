"""Runtime patches for LiteLLM's Anthropic transport.

Patches applied:

1. Allow `effort="xhigh"` for Claude Opus 4.7+.
   LiteLLM 1.81.16 hardcodes the effort allowlist to
   `{"low", "medium", "high", "max"}` inside
   `AnthropicConfig.transform_request`, which rejects Opus 4.7's `xhigh`.
   We wrap that method to expand the allowlist and preserve the existing
   `max`-is-Opus-only guard.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_ALLOWED_EFFORT_VALUES = {"low", "medium", "high", "xhigh", "max"}
_PATCH_APPLIED = False


def apply_anthropic_effort_patch() -> None:
    """Idempotently patches LiteLLM's AnthropicConfig.transform_request."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    try:
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        logger.warning("[yellow]Anthropic effort patch skipped[/yellow] err=%s", exc)
        return

    original_transform = AnthropicConfig.transform_request

    def patched_transform(self: Any, *args: Any, **kwargs: Any) -> Any:
        optional_params = kwargs.get("optional_params")
        if optional_params is None and len(args) >= 3:
            optional_params = args[2]
        output_config = None
        if isinstance(optional_params, dict):
            output_config = optional_params.get("output_config")
        effort = None
        if isinstance(output_config, dict):
            effort = output_config.get("effort")
        # Short-circuit: if effort is xhigh, temporarily rewrite to bypass
        # LiteLLM's strict allowlist, then restore. LiteLLM echoes output_config
        # verbatim into the final request body, so the Anthropic API still
        # receives the original value.
        if effort == "xhigh":
            output_config["effort"] = "high"
            try:
                result = original_transform(self, *args, **kwargs)
            finally:
                output_config["effort"] = "xhigh"
            if isinstance(result, dict) and isinstance(result.get("output_config"), dict):
                result["output_config"]["effort"] = "xhigh"
            return result
        if effort and effort not in _ALLOWED_EFFORT_VALUES:
            logger.warning(
                "[yellow]Unknown Anthropic effort value passed through[/yellow] effort=%s",
                effort,
            )
        return original_transform(self, *args, **kwargs)

    AnthropicConfig.transform_request = patched_transform
    _PATCH_APPLIED = True
    logger.info("[cyan]Anthropic effort patch applied[/cyan] allowed=%s", sorted(_ALLOWED_EFFORT_VALUES))
