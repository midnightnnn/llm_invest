"""Local vector store implementations.

``LocalChromaVectorStore`` lazily imports chromadb/sentence-transformers.  If
those optional dependencies are unavailable, callers should use
``NullVectorStore`` so local quickstart still runs with recency-only memory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def default_vector_dir() -> Path:
    raw = os.getenv("ARENA_LOCAL_VECTOR_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / "data" / "chroma").resolve()


class NullVectorStore:
    """No-op vector store used when local-vector extras are not installed."""

    db = None

    def embed_text(self, text: str) -> list[float]:
        _ = text
        return []

    def clear_embed_cache(self) -> None:
        return None

    def save_memory_vector(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        return None

    def search_similar_memories(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        _ = (args, kwargs)
        return []

    def search_peer_lessons(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        _ = (args, kwargs)
        return []


class LocalChromaVectorStore:
    """ChromaDB-backed local memory vector store."""

    db = None

    def __init__(
        self,
        *,
        persist_dir: str | Path | None = None,
        collection_name: str = "agent_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
        embed_cache_max: int | None = None,
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir is not None else default_vector_dir()
        self.collection_name = str(collection_name or "agent_memories")
        self.embedding_model = str(embedding_model or "all-MiniLM-L6-v2")
        self._embed_cache: dict[str, list[float]] = {}
        self._embed_cache_max = max(16, min(int(embed_cache_max or 128), 4096))
        self._client: Any | None = None
        self._collection: Any | None = None
        self._model: Any | None = None

    def _ensure_client(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed. Install local vector extras with pip install -e \".[local-vector]\".") from exc
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.db = self._client
        self._collection = self._client.get_or_create_collection(self.collection_name)
        return self._collection

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install local vector extras with pip install -e \".[local-vector]\"."
            ) from exc
        self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        token = str(text or "").strip()
        if not token:
            return []
        cache_key = token[:500]
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]
        try:
            model = self._ensure_model()
            vector = [float(v) for v in model.encode(token[:2000], normalize_embeddings=True).tolist()]
        except Exception as exc:
            logger.warning("[yellow]Local embedding unavailable[/yellow] err=%s", str(exc))
            return []
        if len(self._embed_cache) >= self._embed_cache_max:
            self._embed_cache.pop(next(iter(self._embed_cache)))
        self._embed_cache[cache_key] = vector
        return vector

    def clear_embed_cache(self) -> None:
        self._embed_cache.clear()

    def save_memory_vector(
        self,
        event_id: str,
        agent_id: str,
        summary: str,
        score: float = 1.0,
        importance_score: float | None = None,
        outcome_score: float | None = None,
        trading_mode: str = "paper",
        created_at: Any | None = None,
        tenant_id: str = "local",
        event_type: str = "",
        memory_source: str = "",
        memory_tier: str = "",
        primary_regime: str = "",
        primary_strategy_tag: str = "",
        primary_sector: str = "",
        context_tags: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        graph_node_id: str = "",
        causal_chain_id: str = "",
    ) -> None:
        vector = self.embed_text(summary)
        if not vector:
            return
        try:
            collection = self._ensure_client()
            metadata = {
                "tenant_id": str(tenant_id or "").strip().lower() or "local",
                "agent_id": str(agent_id or ""),
                "score": float(score),
                "trading_mode": str(trading_mode or "paper"),
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at or ""),
                "event_type": str(event_type or ""),
                "memory_source": str(memory_source or ""),
                "memory_tier": str(memory_tier or ""),
                "primary_regime": str(primary_regime or ""),
                "primary_strategy_tag": str(primary_strategy_tag or ""),
                "primary_sector": str(primary_sector or ""),
                "graph_node_id": str(graph_node_id or ""),
                "causal_chain_id": str(causal_chain_id or ""),
            }
            if importance_score is not None:
                metadata["importance_score"] = float(importance_score)
            if outcome_score is not None:
                metadata["outcome_score"] = float(outcome_score)
            if context_tags:
                for key, value in context_tags.items():
                    metadata[f"context_{key}"] = ",".join(str(v) for v in value) if isinstance(value, list) else str(value)
            if payload:
                metadata["payload_json"] = json.dumps(payload, ensure_ascii=False, default=str)
            collection.upsert(ids=[event_id], documents=[summary], embeddings=[vector], metadatas=[metadata])
        except Exception as exc:
            logger.warning("[yellow]Local vector save skipped[/yellow] event=%s err=%s", str(event_id)[:8], str(exc))

    @staticmethod
    def _row(event_id: str, doc: str | None, meta: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "event_id": event_id,
            "agent_id": str(meta.get("agent_id") or ""),
            "summary": doc or "",
            "score": float(meta.get("score") or 0.0),
            "created_at": meta.get("created_at") or "",
            "created_date": str(meta.get("created_at") or "")[:10],
        }
        for key in (
            "importance_score",
            "outcome_score",
            "event_type",
            "memory_source",
            "memory_tier",
            "primary_regime",
            "primary_strategy_tag",
            "primary_sector",
            "graph_node_id",
            "causal_chain_id",
        ):
            if meta.get(key) is not None:
                row[key] = meta.get(key)
        payload_json = meta.get("payload_json")
        if isinstance(payload_json, str) and payload_json.strip():
            try:
                parsed = json.loads(payload_json)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                row["payload"] = parsed
                row["payload_json"] = payload_json
        return row

    def search_similar_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        trading_mode: str = "paper",
        tenant_id: str = "local",
    ) -> list[dict[str, Any]]:
        vector = self.embed_text(query)
        if not vector:
            return []
        try:
            collection = self._ensure_client()
            result = collection.query(
                query_embeddings=[vector],
                n_results=max(1, int(limit)),
                where={
                    "$and": [
                        {"tenant_id": str(tenant_id or "").strip().lower() or "local"},
                        {"agent_id": str(agent_id or "")},
                        {"trading_mode": str(trading_mode or "paper")},
                    ]
                },
            )
        except Exception as exc:
            logger.warning("[yellow]Local vector search skipped[/yellow] err=%s", str(exc))
            return []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        return [self._row(str(event_id), doc, dict(meta or {})) for event_id, doc, meta in zip(ids, docs, metas)]

    def search_peer_lessons(
        self,
        *,
        agent_id: str,
        query: str,
        limit: int = 5,
        trading_mode: str = "paper",
        tenant_id: str = "local",
    ) -> list[dict[str, Any]]:
        vector = self.embed_text(query)
        if not vector:
            return []
        try:
            collection = self._ensure_client()
            result = collection.query(
                query_embeddings=[vector],
                n_results=max(max(1, int(limit)) * 4, 12),
                where={
                    "$and": [
                        {"tenant_id": str(tenant_id or "").strip().lower() or "local"},
                        {"trading_mode": str(trading_mode or "paper")},
                        {"event_type": "strategy_reflection"},
                    ]
                },
            )
        except Exception as exc:
            logger.warning("[yellow]Local peer lesson search skipped[/yellow] err=%s", str(exc))
            return []
        rows: list[dict[str, Any]] = []
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        self_agent = str(agent_id or "").strip()
        for event_id, doc, meta_raw in zip(ids, docs, metas):
            meta = dict(meta_raw or {})
            row_agent = str(meta.get("agent_id") or "").strip()
            if not row_agent or row_agent == self_agent:
                continue
            row = self._row(str(event_id), doc, meta)
            row["author_id"] = row_agent
            rows.append(row)
            if len(rows) >= max(1, int(limit)):
                break
        return rows
