from __future__ import annotations

from arena.agents.adk_models import (
    _is_vertex_model_access_error,
    _normalize_vertex_anthropic_model,
)


def test_normalize_vertex_anthropic_alias_sonnet_46() -> None:
    out = _normalize_vertex_anthropic_model("claude-sonnet-4-6")
    assert out == "vertex_ai/claude-sonnet-4-5"


def test_normalize_vertex_anthropic_keeps_versioned_model() -> None:
    out = _normalize_vertex_anthropic_model("vertex_ai/claude-sonnet-4-5@20250929")
    assert out == "vertex_ai/claude-sonnet-4-5@20250929"


def test_vertex_model_access_error_detects_not_found_access() -> None:
    exc = RuntimeError(
        "litellm.NotFoundError: Vertex_aiException - "
        "Publisher Model `projects/x/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-6` "
        "was not found or your project does not have access to it."
    )
    assert _is_vertex_model_access_error(exc) is True


def test_vertex_model_access_error_detects_quota_exhausted() -> None:
    exc = RuntimeError(
        "litellm.RateLimitError: Vertex_aiException - "
        "429 RESOURCE_EXHAUSTED. quota exceeded for publishers/anthropic/models/claude-sonnet-4-5"
    )
    assert _is_vertex_model_access_error(exc) is True


def test_vertex_model_access_error_ignores_unrelated_errors() -> None:
    exc = RuntimeError("rate limit exceeded 429")
    assert _is_vertex_model_access_error(exc) is False
