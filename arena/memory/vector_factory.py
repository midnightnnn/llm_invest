"""Vector-store factory preserving GCP default behaviour."""

from __future__ import annotations

import os
from importlib.util import find_spec
from typing import Any

from arena.memory.policy import memory_embed_cache_max


def _mode(repo: Any) -> str:
    settings = getattr(repo, "settings", None)
    candidate = getattr(settings, "arena_mode", None) if settings is not None else None
    if not candidate:
        candidate = os.getenv("ARENA_MODE")
    return str(candidate or "gcp").strip().lower()


def build_vector_store(repo: Any, memory_policy: dict[str, Any] | None = None) -> Any:
    """Constructs the vector store for the active storage backend.

    GCP mode imports and returns the existing Firestore/Vertex ``VectorStore``.
    Local mode imports only local optional components and falls back to a no-op
    store when ``[local-vector]`` dependencies are absent.
    """
    cache_max = memory_embed_cache_max(memory_policy or {})
    if _mode(repo) == "local":
        from arena.memory.vector_local import LocalChromaVectorStore, NullVectorStore

        if find_spec("chromadb") is None or find_spec("sentence_transformers") is None:
            return NullVectorStore()
        try:
            return LocalChromaVectorStore(embed_cache_max=cache_max)
        except Exception:
            return NullVectorStore()

    from arena.memory.vector import VectorStore

    return VectorStore(
        project=repo.project,
        location=repo.location,
        embed_cache_max=cache_max,
    )
