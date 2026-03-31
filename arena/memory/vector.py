from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from google import genai
from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

logger = logging.getLogger(__name__)

# Fallback embedding dimension for text-embedding-004
EMBEDDING_DIMENSION = 768


class VectorStore:
    """Handles generating embeddings via Vertex AI and storing/searching in Firestore."""

    def __init__(self, project: str, location: str = "asia-northeast3", *, embed_cache_max: int | None = None) -> None:
        self.project = project
        self.location = location
        self._embed_cache: dict[str, list[float]] = {}
        cache_max = 128 if embed_cache_max is None else int(embed_cache_max)
        self._embed_cache_max: int = max(16, min(cache_max, 4096))

        # Embedding Client (Vertex AI)
        # Note: Depending on the region, text-embedding-004 might require a specific location.
        embed_location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        if embed_location == "global":
            embed_location = "asia-northeast3"

        try:
            self.embed_client = genai.Client(
                vertexai=True,
                project=self.project,
                location=embed_location,
            )
        except Exception as e:
            logger.warning("Failed to initialize Vertex AI GenAI client for embeddings: %s", e)
            self.embed_client = None

        # Firestore Client
        try:
            self.db = firestore.Client(project=self.project)
        except Exception as e:
            logger.warning("Failed to initialize Firestore client: %s", e)
            self.db = None

    def embed_text(self, text: str) -> list[float]:
        """Generates a 768-dimensional float vector for the given text."""
        if not self.embed_client:
            return []

        if not text or not text.strip():
            return []

        cache_key = text[:500].strip()
        if cache_key in self._embed_cache:
            logger.debug("[embed_cache] HIT key=%s...", cache_key[:60])
            return self._embed_cache[cache_key]

        try:
            response = self.embed_client.models.embed_content(
                model="text-embedding-004",
                contents=text[:2000],  # safety truncation
            )
            embeddings = response.embeddings
            if embeddings and len(embeddings) > 0:
                values = [float(v) for v in embeddings[0].values]
                if len(self._embed_cache) >= self._embed_cache_max:
                    self._embed_cache.pop(next(iter(self._embed_cache)))
                self._embed_cache[cache_key] = values
                return values
            return []
        except Exception as e:
            logger.error("Error generating embedding: %s", e)
            return []

    def clear_embed_cache(self) -> None:
        """Clears the session-level embedding cache."""
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
        created_at: datetime | None = None,
        tenant_id: str = "local",
        event_type: str = "",
        memory_source: str = "",
        memory_tier: str = "",
        primary_regime: str = "",
        primary_strategy_tag: str = "",
        primary_sector: str = "",
        context_tags: dict[str, Any] | None = None,
        graph_node_id: str = "",
        causal_chain_id: str = "",
    ) -> None:
        """Embeds the summary and saves it to the agent_memories Firestore collection."""
        if not self.db:
            return

        vector = self.embed_text(summary)
        if not vector:
            return

        from arena.models import utc_now
        from google.cloud.firestore_v1.vector import Vector

        created_ts = created_at or utc_now()
        doc_data: dict[str, Any] = {
            "tenant_id": str(tenant_id or "").strip().lower() or "local",
            "agent_id": agent_id,
            "summary": summary,
            "embedding": Vector(vector),
            "score": float(score),
            "trading_mode": trading_mode,
            "created_at": created_ts,
            "created_date": created_ts.date().isoformat(),
        }
        if importance_score is not None:
            doc_data["importance_score"] = float(importance_score)
        if outcome_score is not None:
            doc_data["outcome_score"] = float(outcome_score)
        if event_type:
            doc_data["event_type"] = event_type
        if memory_source:
            doc_data["memory_source"] = str(memory_source).strip()
        if memory_tier:
            doc_data["memory_tier"] = str(memory_tier).strip().lower()
        if primary_regime:
            doc_data["primary_regime"] = str(primary_regime).strip().lower()
        if primary_strategy_tag:
            doc_data["primary_strategy_tag"] = str(primary_strategy_tag).strip().lower()
        if primary_sector:
            doc_data["primary_sector"] = str(primary_sector).strip().lower()
        if context_tags:
            doc_data["context_tags"] = dict(context_tags)
        if graph_node_id:
            doc_data["graph_node_id"] = str(graph_node_id).strip()
        if causal_chain_id:
            doc_data["causal_chain_id"] = str(causal_chain_id).strip()

        doc_ref = self.db.collection("agent_memories").document(event_id)
        try:
            doc_ref.set(doc_data, merge=True)
        except Exception as e:
            logger.error("Failed to save memory vector to Firestore: %s", e)

    @staticmethod
    def _memory_row_from_doc(doc) -> dict[str, Any]:
        data = doc.to_dict()
        row: dict[str, Any] = {
            "event_id": doc.id,
            "agent_id": data.get("agent_id", ""),
            "summary": data.get("summary", ""),
            "score": data.get("score", 0.0),
            "created_at": data.get("created_at"),
            "created_date": data.get("created_date", ""),
        }
        if data.get("importance_score") is not None:
            row["importance_score"] = data.get("importance_score")
        if data.get("outcome_score") is not None:
            row["outcome_score"] = data.get("outcome_score")
        et = data.get("event_type")
        if et:
            row["event_type"] = str(et)
        memory_source = data.get("memory_source")
        if memory_source:
            row["memory_source"] = str(memory_source)
        memory_tier = data.get("memory_tier")
        if memory_tier:
            row["memory_tier"] = str(memory_tier)
        primary_regime = data.get("primary_regime")
        if primary_regime:
            row["primary_regime"] = str(primary_regime)
        primary_strategy_tag = data.get("primary_strategy_tag")
        if primary_strategy_tag:
            row["primary_strategy_tag"] = str(primary_strategy_tag)
        primary_sector = data.get("primary_sector")
        if primary_sector:
            row["primary_sector"] = str(primary_sector)
        context_tags = data.get("context_tags")
        if isinstance(context_tags, dict):
            row["context_tags"] = dict(context_tags)
        graph_node_id = data.get("graph_node_id")
        if graph_node_id:
            row["graph_node_id"] = str(graph_node_id)
        causal_chain_id = data.get("causal_chain_id")
        if causal_chain_id:
            row["causal_chain_id"] = str(causal_chain_id)
        return row

    def search_similar_memories(
        self,
        agent_id: str,
        query: str,
        limit: int = 5,
        trading_mode: str = "paper",
        tenant_id: str = "local",
    ) -> list[dict[str, Any]]:
        """Searches for similar memories using Firestore Vector Search."""
        if not self.db:
            return []
            
        vector = self.embed_text(query)
        if not vector:
            return []

        try:
            collection = self.db.collection("agent_memories")
            # Apply pre-filter: only memories belonging to this agent and matching trading_mode
            vector_query = (
                collection.where(filter=firestore.FieldFilter("tenant_id", "==", str(tenant_id or "").strip().lower() or "local"))
                .where(filter=firestore.FieldFilter("agent_id", "==", agent_id))
                .where(filter=firestore.FieldFilter("trading_mode", "==", trading_mode))
                .find_nearest(
                    vector_field="embedding",
                    query_vector=vector,
                    distance_measure=DistanceMeasure.COSINE,
                    limit=limit,
                )
            )
            
            results = []
            for doc in vector_query.stream():
                results.append(self._memory_row_from_doc(doc))
            return results
        except Exception as e:
            logger.error("Vector search failed (may require index creation): %s", e)
            return []

    def search_peer_lessons(
        self,
        *,
        agent_id: str,
        query: str,
        limit: int = 5,
        trading_mode: str = "paper",
        tenant_id: str = "local",
    ) -> list[dict[str, Any]]:
        """Searches other agents' compacted lesson memories within the same tenant/mode."""
        if not self.db:
            return []

        vector = self.embed_text(query)
        if not vector:
            return []

        fetch_limit = max(min(int(limit), 10) * 4, 12)
        tenant = str(tenant_id or "").strip().lower() or "local"
        try:
            collection = self.db.collection("agent_memories")
            vector_query = (
                collection.where(filter=firestore.FieldFilter("tenant_id", "==", tenant))
                .where(filter=firestore.FieldFilter("trading_mode", "==", trading_mode))
                .where(filter=firestore.FieldFilter("event_type", "==", "strategy_reflection"))
                .find_nearest(
                    vector_field="embedding",
                    query_vector=vector,
                    distance_measure=DistanceMeasure.COSINE,
                    limit=fetch_limit,
                )
            )

            results: list[dict[str, Any]] = []
            self_agent = str(agent_id or "").strip()
            for doc in vector_query.stream():
                row = self._memory_row_from_doc(doc)
                row_agent = str(row.get("agent_id") or "").strip()
                if not row_agent or row_agent == self_agent:
                    continue
                row["author_id"] = row_agent
                results.append(row)
                if len(results) >= max(1, min(int(limit), 10)):
                    break
            return results
        except Exception as e:
            logger.error("Peer lesson vector search failed (may require index creation): %s", e)
            return []

