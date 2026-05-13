"""Local DuckDB-backed memory, board, graph, and relation store."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from arena.memory.graph import ensure_memory_event_graph_ids
from arena.data.local.session import DuckDBSession
from arena.models import BoardPost, MemoryEvent, utc_now


_MEMORY_SELECT_COLUMNS = (
    "event_id, created_at, agent_id, event_type, summary, trading_mode, payload_json, "
    "importance_score, outcome_score, score, memory_tier, expires_at, promoted_at, "
    "semantic_key, context_tags_json, primary_regime, primary_strategy_tag, primary_sector, "
    "access_count, last_accessed_at, decay_score, effective_score, graph_node_id, causal_chain_id, "
    "cycle_id, llm_call_id"
)


def _json_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


class LocalMemoryStore:
    """DuckDB-backed memory, board, and graph store."""

    def __init__(self, session: DuckDBSession) -> None:
        self.session = session

    def _has_column(self, table: str, column: str) -> bool:
        rows = self.session.fetch_rows(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name = $table
              AND column_name = $column
            """,
            {"table": table, "column": column},
        )
        return bool(rows)

    def _ensure_research_detail_json_column(self) -> bool:
        if self._has_column("research_briefings", "detail_json"):
            return True
        try:
            self.session.execute("ALTER TABLE research_briefings ADD COLUMN detail_json JSON")
            return True
        except Exception:
            return self._has_column("research_briefings", "detail_json")

    def recent_memory_events(
        self,
        agent_id: str,
        limit: int,
        trading_mode: str = "paper",
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        return self.session.fetch_rows(
            f"""
            SELECT {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory_events
            WHERE tenant_id = $tenant_id
              AND agent_id = $agent_id
              AND trading_mode = $trading_mode
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            {
                "tenant_id": tenant,
                "agent_id": str(agent_id or "").strip(),
                "trading_mode": trading_mode,
                "limit": max(1, int(limit)),
            },
        )

    def memory_event_by_id(
        self,
        *,
        event_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        tenant = self.session.resolve_tenant_id(tenant_id)
        token = str(event_id or "").strip()
        if not token:
            return None
        rows = self.session.fetch_rows(
            f"""
            SELECT {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory_events
            WHERE tenant_id = $tenant_id
              AND event_id = $event_id
            LIMIT 1
            """,
            {"tenant_id": tenant, "event_id": token},
        )
        return rows[0] if rows else None

    def memory_events_by_ids(
        self,
        *,
        agent_id: str,
        event_ids: list[str],
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        clean_ids = [str(eid or "").strip() for eid in (event_ids or []) if str(eid or "").strip()]
        if not clean_ids:
            return []
        return self.session.fetch_rows(
            f"""
            SELECT {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory_events
            WHERE tenant_id = $tenant_id
              AND agent_id = $agent_id
              AND trading_mode = $trading_mode
              AND event_id IN (SELECT unnest($event_ids))
            """,
            {
                "tenant_id": tenant,
                "agent_id": str(agent_id or "").strip(),
                "trading_mode": trading_mode,
                "event_ids": clean_ids,
            },
        )

    def memory_events_for_cycle(
        self,
        *,
        agent_id: str,
        cycle_id: str,
        event_types: list[str] | None = None,
        limit: int = 20,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "agent_id": str(agent_id or "").strip(),
            "trading_mode": trading_mode,
            "cycle_id": str(cycle_id or "").strip(),
            "limit": max(1, int(limit)),
        }
        filters = [
            "tenant_id = $tenant_id",
            "agent_id = $agent_id",
            "trading_mode = $trading_mode",
            "("
            "COALESCE(cycle_id, json_extract_string(payload_json, '$.cycle_id'), "
            "json_extract_string(payload_json, '$.intent.cycle_id'), '') = $cycle_id"
            ")",
        ]
        clean_types = [str(token or "").strip() for token in (event_types or []) if str(token or "").strip()]
        if clean_types:
            filters.append("event_type IN (SELECT unnest($event_types))")
            params["event_types"] = clean_types
        return self.session.fetch_rows(
            f"""
            SELECT {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory_events
            WHERE {' AND '.join(filters)}
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            params,
        )

    def latest_memory_compaction_cycle_id(
        self,
        *,
        agent_ids: list[str],
        event_types: list[str],
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> str:
        tenant = self.session.resolve_tenant_id(tenant_id)
        clean_agents = [str(token or "").strip() for token in (agent_ids or []) if str(token or "").strip()]
        clean_types = [str(token or "").strip() for token in (event_types or []) if str(token or "").strip()]
        if not clean_agents or not clean_types:
            return ""
        rows = self.session.fetch_rows(
            """
            SELECT cycle_id_key AS cycle_id
            FROM (
              SELECT
                COALESCE(
                  cycle_id,
                  json_extract_string(payload_json, '$.cycle_id'),
                  json_extract_string(payload_json, '$.intent.cycle_id'),
                  ''
                ) AS cycle_id_key,
                created_at
              FROM agent_memory_events
              WHERE tenant_id = $tenant_id
                AND agent_id IN (SELECT unnest($agent_ids))
                AND trading_mode = $trading_mode
                AND event_type IN (SELECT unnest($event_types))
            )
            WHERE cycle_id_key <> ''
            GROUP BY cycle_id_key
            ORDER BY MAX(created_at) DESC
            LIMIT 1
            """,
            {
                "tenant_id": tenant,
                "agent_ids": clean_agents,
                "event_types": clean_types,
                "trading_mode": trading_mode,
            },
        )
        return str((rows[0] if rows else {}).get("cycle_id") or "").strip()

    def compaction_reflections_for_cycle(
        self,
        *,
        agent_id: str,
        cycle_id: str,
        trading_mode: str = "paper",
        limit: int = 10,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        return self.session.fetch_rows(
            f"""
            SELECT {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory_events
            WHERE tenant_id = $tenant_id
              AND agent_id = $agent_id
              AND trading_mode = $trading_mode
              AND event_type = 'strategy_reflection'
              AND COALESCE(cycle_id, json_extract_string(payload_json, '$.cycle_id'), '') = $cycle_id
              AND json_extract_string(payload_json, '$.source') IN (SELECT unnest($sources))
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            {
                "tenant_id": tenant,
                "agent_id": str(agent_id or "").strip(),
                "cycle_id": str(cycle_id or "").strip(),
                "trading_mode": trading_mode,
                "sources": ["memory_compaction", "thesis_chain_compaction"],
                "limit": max(1, int(limit)),
            },
        )

    # ------------------------------------------------------------------
    # Board posts
    # ------------------------------------------------------------------

    def write_board_post(self, post: BoardPost, *, tenant_id: str | None = None) -> None:
        tenant = self.session.resolve_tenant_id(tenant_id)
        self.session.insert_dict(
            "board_posts",
            {
                "tenant_id": tenant,
                "post_id": post.post_id,
                "cycle_id": str(post.cycle_id or "").strip() or None,
                "llm_call_id": str(post.llm_call_id or "").strip() or None,
                "created_at": post.created_at,
                "agent_id": post.agent_id,
                "title": post.title,
                "body": post.body,
                "explore_summary": post.explore_summary or None,
                "trading_mode": post.trading_mode,
                "tickers": list(post.tickers or []),
            },
        )

    def recent_board_posts(
        self,
        limit: int,
        trading_mode: str = "paper",
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        return self.session.fetch_rows(
            """
            SELECT post_id, cycle_id, llm_call_id, created_at, agent_id, title, body, trading_mode, tickers
            FROM board_posts
            WHERE tenant_id = $tenant_id
              AND trading_mode = $trading_mode
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            {"tenant_id": tenant, "trading_mode": trading_mode, "limit": max(1, int(limit))},
        )

    def board_posts_for_cycle(
        self,
        *,
        cycle_id: str,
        agent_id: str | None = None,
        limit: int = 10,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "cycle_id": str(cycle_id or "").strip(),
            "trading_mode": trading_mode,
            "limit": max(1, int(limit)),
        }
        filters = ["tenant_id = $tenant_id", "cycle_id = $cycle_id", "trading_mode = $trading_mode"]
        if agent_id:
            filters.append("agent_id = $agent_id")
            params["agent_id"] = str(agent_id or "").strip()
        return self.session.fetch_rows(
            f"""
            SELECT post_id, cycle_id, llm_call_id, created_at, agent_id, title, body, explore_summary, trading_mode, tickers
            FROM board_posts
            WHERE {' AND '.join(filters)}
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            params,
        )

    # ------------------------------------------------------------------
    # Research briefings
    # ------------------------------------------------------------------

    def insert_research_briefings(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        has_detail_json = self._ensure_research_detail_json_column()
        payload: list[dict[str, Any]] = []
        for row in rows:
            briefing_id = str(row.get("briefing_id") or "").strip()
            ticker = str(row.get("ticker") or "").strip().upper()
            summary = str(row.get("summary") or "").strip()
            if not briefing_id or not ticker or not summary:
                continue
            payload.append(
                {
                    "tenant_id": tenant,
                    "briefing_id": briefing_id,
                    "created_at": row.get("created_at") or utc_now(),
                    "ticker": ticker,
                    "category": str(row.get("category") or "").strip() or "general",
                    "headline": str(row.get("headline") or "").strip() or f"{ticker} briefing",
                    "summary": summary,
                    "sources": str(row.get("sources") or "").strip() or None,
                    "trading_mode": str(row.get("trading_mode") or "").strip().lower() or "paper",
                }
            )
            if has_detail_json:
                payload[-1]["detail_json"] = json.dumps(
                    _json_or_none(row.get("detail_json")) or {},
                    ensure_ascii=False,
                    default=str,
                )
        self.session.insert_dicts("research_briefings", payload)

    def get_research_briefings(
        self,
        *,
        tickers: list[str] | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns stored research briefings with the same filter semantics as BigQuery."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        clean_tickers = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
        clean_categories = [str(c).strip().lower() for c in (categories or []) if str(c).strip()]
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "trading_mode": str(trading_mode or "").strip().lower() or "paper",
            "limit": max(1, int(limit)),
        }
        conditions = ["tenant_id = $tenant_id", "trading_mode = $trading_mode"]
        filters: list[str] = []
        if clean_tickers:
            filters.append("ticker IN (SELECT unnest($tickers))")
            params["tickers"] = clean_tickers
        if clean_categories:
            filters.append("category IN (SELECT unnest($categories))")
            params["categories"] = clean_categories
        if filters:
            conditions.append(f"({' OR '.join(filters)})")

        detail_expr = "detail_json" if self._has_column("research_briefings", "detail_json") else "NULL AS detail_json"
        rows = self.session.fetch_rows(
            f"""
            SELECT briefing_id, created_at, ticker, category, headline, summary, {detail_expr}, sources
            FROM research_briefings
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            params,
        )
        for row in rows:
            row["detail_json"] = _json_or_none(row.get("detail_json"))
        return rows

    # ------------------------------------------------------------------
    # Memory write/update
    # ------------------------------------------------------------------

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)

    def write_memory_event(self, event: MemoryEvent, *, tenant_id: str | None = None) -> None:
        tenant = self.session.resolve_tenant_id(tenant_id)
        ensure_memory_event_graph_ids(event)
        importance_score = event.importance_score if event.importance_score is not None else event.score
        payload_cycle_id = ""
        payload_llm_call_id = ""
        if isinstance(event.payload, dict):
            payload_cycle_id = str(event.payload.get("cycle_id") or "").strip()
            payload_llm_call_id = str(event.payload.get("llm_call_id") or "").strip()
            intent_payload = event.payload.get("intent")
            if not payload_cycle_id and isinstance(intent_payload, dict):
                payload_cycle_id = str(intent_payload.get("cycle_id") or "").strip()
            if not payload_llm_call_id and isinstance(intent_payload, dict):
                payload_llm_call_id = str(intent_payload.get("llm_call_id") or "").strip()
        self.session.insert_dict(
            "agent_memory_events",
            {
                "tenant_id": tenant,
                "event_id": event.event_id,
                "created_at": event.created_at,
                "agent_id": event.agent_id,
                "event_type": event.event_type,
                "summary": event.summary,
                "trading_mode": event.trading_mode,
                "cycle_id": str(event.cycle_id or "").strip() or payload_cycle_id or None,
                "llm_call_id": str(event.llm_call_id or "").strip() or payload_llm_call_id or None,
                "payload_json": self._json_dumps(event.payload),
                "importance_score": importance_score,
                "outcome_score": event.outcome_score,
                "score": event.score,
                "memory_tier": str(event.memory_tier or "").strip().lower() or None,
                "expires_at": event.expires_at,
                "promoted_at": event.promoted_at,
                "semantic_key": str(event.semantic_key or "").strip() or None,
                "context_tags_json": self._json_dumps(event.context_tags) if event.context_tags else None,
                "primary_regime": str(event.primary_regime or "").strip().lower() or None,
                "primary_strategy_tag": str(event.primary_strategy_tag or "").strip().lower() or None,
                "primary_sector": str(event.primary_sector or "").strip().lower() or None,
                "access_count": event.access_count,
                "last_accessed_at": event.last_accessed_at,
                "decay_score": event.decay_score,
                "effective_score": event.effective_score,
                "graph_node_id": str(event.graph_node_id or "").strip() or None,
                "causal_chain_id": str(event.causal_chain_id or "").strip() or None,
            },
        )

    def update_memory_event(
        self,
        *,
        event_id: str,
        summary: str,
        payload: dict[str, Any],
        score: float,
        importance_score: float | None = None,
        outcome_score: float | None = None,
        memory_tier: str | None = None,
        expires_at: datetime | None = None,
        context_tags: dict[str, Any] | None = None,
        primary_regime: str | None = None,
        primary_strategy_tag: str | None = None,
        primary_sector: str | None = None,
        graph_node_id: str | None = None,
        causal_chain_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        tenant = self.session.resolve_tenant_id(tenant_id)
        self.session.execute(
            """
            UPDATE agent_memory_events
            SET summary = $summary,
                payload_json = $payload_json,
                importance_score = $importance_score,
                outcome_score = $outcome_score,
                score = $score,
                memory_tier = COALESCE($memory_tier, memory_tier),
                expires_at = COALESCE($expires_at, expires_at),
                context_tags_json = COALESCE($context_tags_json, context_tags_json),
                primary_regime = COALESCE($primary_regime, primary_regime),
                primary_strategy_tag = COALESCE($primary_strategy_tag, primary_strategy_tag),
                primary_sector = COALESCE($primary_sector, primary_sector),
                graph_node_id = COALESCE($graph_node_id, graph_node_id),
                causal_chain_id = COALESCE($causal_chain_id, causal_chain_id)
            WHERE tenant_id = $tenant_id
              AND event_id = $event_id
            """,
            {
                "tenant_id": tenant,
                "event_id": str(event_id or "").strip(),
                "summary": str(summary or "").strip(),
                "payload_json": self._json_dumps(payload or {}),
                "importance_score": importance_score,
                "outcome_score": outcome_score,
                "score": max(0.0, min(float(score), 1.0)),
                "memory_tier": str(memory_tier or "").strip().lower() or None,
                "expires_at": expires_at,
                "context_tags_json": self._json_dumps(context_tags) if context_tags else None,
                "primary_regime": str(primary_regime or "").strip().lower() or None,
                "primary_strategy_tag": str(primary_strategy_tag or "").strip().lower() or None,
                "primary_sector": str(primary_sector or "").strip().lower() or None,
                "graph_node_id": str(graph_node_id or "").strip() or None,
                "causal_chain_id": str(causal_chain_id or "").strip() or None,
            },
        )

    def update_memory_score(self, event_id: str, new_score: float, *, tenant_id: str | None = None) -> None:
        tenant = self.session.resolve_tenant_id(tenant_id)
        self.session.execute(
            """
            UPDATE agent_memory_events
            SET outcome_score = $new_score
            WHERE tenant_id = $tenant_id
              AND event_id = $event_id
            """,
            {
                "tenant_id": tenant,
                "event_id": str(event_id or "").strip(),
                "new_score": max(0.0, min(float(new_score), 1.0)),
            },
        )

    def append_memory_access_events(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        payload_rows: list[dict[str, Any]] = []
        for row in rows:
            payload_rows.append(
                {
                    "tenant_id": tenant,
                    "access_id": str(row.get("access_id") or "").strip(),
                    "accessed_at": row.get("accessed_at"),
                    "event_id": str(row.get("event_id") or "").strip(),
                    "agent_id": str(row.get("agent_id") or "").strip() or None,
                    "source_agent_id": str(row.get("source_agent_id") or "").strip() or None,
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "access_type": str(row.get("access_type") or "retrieval").strip().lower() or "retrieval",
                    "query_text": str(row.get("query_text") or "").strip() or None,
                    "retrieval_score": row.get("retrieval_score"),
                    "used_in_prompt": row.get("used_in_prompt"),
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "detail_json": self._json_dumps(row.get("detail_json")) if row.get("detail_json") is not None else None,
                }
            )
        self.session.insert_dicts("memory_access_events", payload_rows)

    # ------------------------------------------------------------------
    # Memory query helpers used by agent/runtime paths
    # ------------------------------------------------------------------

    def candidate_memory_events(
        self,
        *,
        agent_id: str,
        exclude_tickers: list[str] | None = None,
        limit: int = 12,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        excluded = [str(t or "").strip().upper() for t in (exclude_tickers or []) if str(t or "").strip()]
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "agent_id": str(agent_id or "").strip(),
            "trading_mode": trading_mode,
            "event_types": [
                "candidate_screen_hit",
                "candidate_discovery",
                "candidate_opportunity",
            ],
            "limit": max(1, int(limit)),
        }
        exclude_clause = ""
        if excluded:
            exclude_clause = "AND UPPER(COALESCE(json_extract_string(payload_json, '$.ticker'), '')) NOT IN (SELECT unnest($exclude_tickers))"
            params["exclude_tickers"] = excluded
        return self.session.fetch_rows(
            f"""
            SELECT {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory_events
            WHERE tenant_id = $tenant_id
              AND agent_id = $agent_id
              AND trading_mode = $trading_mode
              AND event_type IN (SELECT unnest($event_types))
              {exclude_clause}
            ORDER BY COALESCE(effective_score, score, importance_score, 0.0) DESC, created_at DESC
            LIMIT $limit
            """,
            params,
        )

    def find_buy_memories_for_ticker(
        self,
        agent_id: str,
        ticker: str,
        limit: int = 5,
        trading_mode: str = "paper",
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        return self.session.fetch_rows(
            f"""
            SELECT {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory_events
            WHERE tenant_id = $tenant_id
              AND agent_id = $agent_id
              AND trading_mode = $trading_mode
              AND event_type = 'trade_execution'
              AND (
                UPPER(COALESCE(json_extract_string(payload_json, '$.intent.ticker'), '')) = $ticker
                AND UPPER(COALESCE(json_extract_string(payload_json, '$.intent.side'), '')) = 'BUY'
              )
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            {
                "tenant_id": tenant,
                "agent_id": str(agent_id or "").strip(),
                "ticker": str(ticker or "").strip().upper(),
                "trading_mode": trading_mode,
                "limit": max(1, int(limit)),
            },
        )

    def memory_events_by_semantic_keys(
        self,
        *,
        agent_id: str,
        semantic_keys: list[str],
        event_types: list[str] | None = None,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        keys = [str(key or "").strip() for key in semantic_keys if str(key or "").strip()]
        if not keys:
            return []
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "agent_id": str(agent_id or "").strip(),
            "semantic_keys": keys,
            "trading_mode": trading_mode,
        }
        event_clause = ""
        if event_types:
            event_clause = "AND event_type IN (SELECT unnest($event_types))"
            params["event_types"] = [str(item or "").strip() for item in event_types if str(item or "").strip()]
        return self.session.fetch_rows(
            f"""
            SELECT {_MEMORY_SELECT_COLUMNS}
            FROM agent_memory_events
            WHERE tenant_id = $tenant_id
              AND agent_id = $agent_id
              AND trading_mode = $trading_mode
              AND semantic_key IN (SELECT unnest($semantic_keys))
              {event_clause}
            ORDER BY semantic_key ASC, created_at ASC
            """,
            params,
        )

    def memory_event_exists_by_semantic_key(
        self,
        *,
        agent_id: str,
        event_type: str,
        semantic_key: str,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> bool:
        rows = self.memory_events_by_semantic_keys(
            agent_id=agent_id,
            semantic_keys=[semantic_key],
            event_types=[event_type],
            trading_mode=trading_mode,
            tenant_id=tenant_id,
        )
        return bool(rows)

    def active_thesis_events(
        self,
        *,
        agent_id: str,
        tickers: list[str],
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        clean_tickers = [str(t or "").strip().upper() for t in (tickers or []) if str(t or "").strip()]
        if not clean_tickers:
            return []
        return self.session.fetch_rows(
            f"""
            WITH ranked AS (
              SELECT {_MEMORY_SELECT_COLUMNS},
                     ROW_NUMBER() OVER (
                       PARTITION BY UPPER(COALESCE(json_extract_string(payload_json, '$.ticker'), ''))
                       ORDER BY created_at DESC
                     ) AS rn
              FROM agent_memory_events
              WHERE tenant_id = $tenant_id
                AND agent_id = $agent_id
                AND trading_mode = $trading_mode
                AND event_type IN ('thesis_open', 'thesis_update')
                AND UPPER(COALESCE(json_extract_string(payload_json, '$.ticker'), '')) IN (SELECT unnest($tickers))
            )
            SELECT {_MEMORY_SELECT_COLUMNS}
            FROM ranked
            WHERE rn = 1
            """,
            {
                "tenant_id": tenant,
                "agent_id": str(agent_id or "").strip(),
                "trading_mode": trading_mode,
                "tickers": clean_tickers,
            },
        )

    # ------------------------------------------------------------------
    # Graph/relation write and read helpers
    # ------------------------------------------------------------------

    def upsert_memory_graph_nodes(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        for row in rows:
            node_id = str(row.get("node_id") or "").strip()
            if not node_id:
                continue
            self.session.execute(
                "DELETE FROM memory_graph_nodes WHERE tenant_id = $tenant_id AND node_id = $node_id",
                {"tenant_id": tenant, "node_id": node_id},
            )
            self.session.insert_dict(
                "memory_graph_nodes",
                {
                    "tenant_id": tenant,
                    "node_id": node_id,
                    "created_at": row.get("created_at"),
                    "node_kind": str(row.get("node_kind") or "memory").strip(),
                    "source_table": str(row.get("source_table") or "").strip(),
                    "source_id": str(row.get("source_id") or "").strip(),
                    "agent_id": str(row.get("agent_id") or "").strip() or None,
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "summary": str(row.get("summary") or "").strip() or None,
                    "ticker": str(row.get("ticker") or "").strip().upper() or None,
                    "memory_tier": str(row.get("memory_tier") or "").strip().lower() or None,
                    "primary_regime": str(row.get("primary_regime") or "").strip().lower() or None,
                    "context_tags_json": self._json_dumps(row.get("context_tags_json") or row.get("context_tags"))
                    if (row.get("context_tags_json") is not None or row.get("context_tags") is not None)
                    else None,
                    "payload_json": self._json_dumps(row.get("payload_json") or row.get("payload"))
                    if (row.get("payload_json") is not None or row.get("payload") is not None)
                    else None,
                },
            )

    def upsert_memory_graph_edges(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        for row in rows:
            edge_id = str(row.get("edge_id") or "").strip()
            if not edge_id:
                continue
            self.session.execute(
                "DELETE FROM memory_graph_edges WHERE tenant_id = $tenant_id AND edge_id = $edge_id",
                {"tenant_id": tenant, "edge_id": edge_id},
            )
            self.session.insert_dict(
                "memory_graph_edges",
                {
                    "tenant_id": tenant,
                    "edge_id": edge_id,
                    "created_at": row.get("created_at"),
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "from_node_id": str(row.get("from_node_id") or "").strip(),
                    "to_node_id": str(row.get("to_node_id") or "").strip(),
                    "edge_type": str(row.get("edge_type") or "").strip(),
                    "edge_strength": row.get("edge_strength"),
                    "confidence": row.get("confidence"),
                    "causal_chain_id": str(row.get("causal_chain_id") or "").strip() or None,
                    "detail_json": self._json_dumps(row.get("detail_json")) if row.get("detail_json") is not None else None,
                },
            )

    def memory_graph_neighbors(
        self,
        *,
        seed_node_ids: list[str],
        trading_mode: str = "paper",
        min_confidence: float = 0.0,
        limit: int = 24,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        seeds = [str(seed or "").strip() for seed in seed_node_ids if str(seed or "").strip()]
        if not seeds:
            return []
        return self.session.fetch_rows(
            """
            SELECT e.edge_id, e.from_node_id, e.to_node_id, e.edge_type, e.edge_strength,
                   e.confidence, e.causal_chain_id, e.detail_json,
                   n.node_id, n.node_kind, n.source_table, n.source_id, n.agent_id,
                   n.summary, n.ticker, n.memory_tier, n.primary_regime, n.context_tags_json, n.payload_json
            FROM memory_graph_edges e
            JOIN memory_graph_nodes n
              ON n.tenant_id = e.tenant_id
             AND n.node_id = CASE
               WHEN e.from_node_id IN (SELECT unnest($seed_node_ids)) THEN e.to_node_id
               ELSE e.from_node_id
             END
            WHERE e.tenant_id = $tenant_id
              AND e.trading_mode = $trading_mode
              AND (e.from_node_id IN (SELECT unnest($seed_node_ids)) OR e.to_node_id IN (SELECT unnest($seed_node_ids)))
              AND COALESCE(e.confidence, 1.0) >= $min_confidence
            ORDER BY COALESCE(e.confidence, 1.0) DESC, e.created_at DESC
            LIMIT $limit
            """,
            {
                "tenant_id": tenant,
                "seed_node_ids": seeds,
                "trading_mode": trading_mode,
                "min_confidence": float(min_confidence),
                "limit": max(1, int(limit)),
            },
        )

    def upsert_memory_relation_triples(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        for row in rows:
            triple_id = str(row.get("triple_id") or "").strip()
            if not triple_id:
                continue
            self.session.execute(
                "DELETE FROM memory_relation_triples WHERE tenant_id = $tenant_id AND triple_id = $triple_id",
                {"tenant_id": tenant, "triple_id": triple_id},
            )
            self.session.insert_dict(
                "memory_relation_triples",
                {
                    "tenant_id": tenant,
                    "triple_id": triple_id,
                    "created_at": row.get("created_at"),
                    "source_table": str(row.get("source_table") or "").strip(),
                    "source_id": str(row.get("source_id") or "").strip(),
                    "source_node_id": str(row.get("source_node_id") or "").strip() or None,
                    "source_created_at": row.get("source_created_at"),
                    "agent_id": str(row.get("agent_id") or "").strip() or None,
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "subject_node_id": str(row.get("subject_node_id") or "").strip(),
                    "subject_label": str(row.get("subject_label") or "").strip(),
                    "subject_type": str(row.get("subject_type") or "").strip(),
                    "predicate": str(row.get("predicate") or "").strip(),
                    "object_node_id": str(row.get("object_node_id") or "").strip(),
                    "object_label": str(row.get("object_label") or "").strip(),
                    "object_type": str(row.get("object_type") or "").strip(),
                    "confidence": row.get("confidence"),
                    "evidence_text": str(row.get("evidence_text") or "").strip() or None,
                    "extraction_method": str(row.get("extraction_method") or "").strip() or None,
                    "extraction_version": str(row.get("extraction_version") or "").strip() or None,
                    "status": str(row.get("status") or "").strip() or None,
                    "detail_json": self._json_dumps(row.get("detail_json")) if row.get("detail_json") is not None else None,
                },
            )

    def upsert_memory_relation_triples_with_graph(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        self.upsert_memory_relation_triples(rows, tenant_id=tenant_id)

    def memory_relation_triples_for_source(
        self,
        *,
        source_table: str,
        source_id: str,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        return self.session.fetch_rows(
            """
            SELECT *
            FROM memory_relation_triples
            WHERE tenant_id = $tenant_id
              AND source_table = $source_table
              AND source_id = $source_id
            ORDER BY created_at DESC
            """,
            {
                "tenant_id": tenant,
                "source_table": str(source_table or "").strip(),
                "source_id": str(source_id or "").strip(),
            },
        )

    def memory_relation_memory_candidates(
        self,
        *,
        agent_id: str,
        seed_node_ids: list[str],
        trading_mode: str = "paper",
        min_confidence: float = 0.75,
        limit: int = 8,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        neighbors = self.memory_graph_neighbors(
            seed_node_ids=seed_node_ids,
            trading_mode=trading_mode,
            min_confidence=min_confidence,
            limit=limit * 3,
            tenant_id=tenant_id,
        )
        event_ids = [
            str(row.get("source_id") or "").strip()
            for row in neighbors
            if str(row.get("source_table") or "").strip() == "agent_memory_events"
        ]
        return self.memory_events_by_ids(
            agent_id=agent_id,
            event_ids=event_ids[: max(1, int(limit))],
            trading_mode=trading_mode,
            tenant_id=tenant_id,
        )

    def relation_extraction_pending_sources(
        self,
        *,
        limit: int = 25,
        source_table: str | None = None,
        event_types: list[str] | None = None,
        trading_mode: str = "paper",
        extractor_version: str,
        prompt_version: str,
        ontology_version: str,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        _ = (extractor_version, prompt_version, ontology_version)
        tenant = self.session.resolve_tenant_id(tenant_id)
        table = str(source_table or "").strip()
        if table and table not in {"agent_memory_events", "board_posts", "research_briefings"}:
            return []
        if table in {"", "agent_memory_events"}:
            params: dict[str, Any] = {
                "tenant_id": tenant,
                "trading_mode": trading_mode,
                "limit": max(1, int(limit)),
            }
            event_clause = ""
            if event_types:
                event_clause = "AND event_type IN (SELECT unnest($event_types))"
                params["event_types"] = [str(e or "").strip() for e in event_types if str(e or "").strip()]
            return self.session.fetch_rows(
                f"""
                SELECT
                  'agent_memory_events' AS source_table,
                  event_id AS source_id,
                  created_at AS source_created_at,
                  agent_id,
                  trading_mode,
                  cycle_id,
                  summary AS source_text
                FROM agent_memory_events
                WHERE tenant_id = $tenant_id
                  AND trading_mode = $trading_mode
                  {event_clause}
                  AND NOT EXISTS (
                    SELECT 1 FROM memory_relation_extraction_runs run
                    WHERE run.tenant_id = $tenant_id
                      AND run.source_table = 'agent_memory_events'
                      AND run.source_id = agent_memory_events.event_id
                  )
                ORDER BY created_at ASC
                LIMIT $limit
                """,
                params,
            )
        if table == "board_posts":
            return self.session.fetch_rows(
                """
                SELECT 'board_posts' AS source_table, post_id AS source_id, created_at AS source_created_at,
                       agent_id, trading_mode, cycle_id, title || '\n' || body AS source_text
                FROM board_posts
                WHERE tenant_id = $tenant_id
                  AND trading_mode = $trading_mode
                ORDER BY created_at ASC
                LIMIT $limit
                """,
                {"tenant_id": tenant, "trading_mode": trading_mode, "limit": max(1, int(limit))},
            )
        return []

    def append_memory_relation_extraction_runs(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        payload_rows = []
        for row in rows:
            payload_rows.append(
                {
                    "tenant_id": tenant,
                    "run_id": str(row.get("run_id") or "").strip(),
                    "started_at": row.get("started_at"),
                    "finished_at": row.get("finished_at"),
                    "source_table": str(row.get("source_table") or "").strip(),
                    "source_id": str(row.get("source_id") or "").strip(),
                    "source_hash": str(row.get("source_hash") or "").strip(),
                    "source_created_at": row.get("source_created_at"),
                    "agent_id": str(row.get("agent_id") or "").strip() or None,
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "extractor_version": str(row.get("extractor_version") or "").strip(),
                    "prompt_version": str(row.get("prompt_version") or "").strip(),
                    "ontology_version": str(row.get("ontology_version") or "").strip(),
                    "provider": str(row.get("provider") or "").strip() or None,
                    "model": str(row.get("model") or "").strip() or None,
                    "status": str(row.get("status") or "").strip(),
                    "accepted_count": row.get("accepted_count"),
                    "rejected_count": row.get("rejected_count"),
                    "raw_output_json": self._json_dumps(row.get("raw_output_json")) if row.get("raw_output_json") is not None else None,
                    "error_message": str(row.get("error_message") or "").strip() or None,
                    "detail_json": self._json_dumps(row.get("detail_json")) if row.get("detail_json") is not None else None,
                }
            )
        self.session.insert_dicts("memory_relation_extraction_runs", payload_rows)

    def append_memory_relation_tuning_runs(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        payload_rows = []
        for row in rows:
            next_row = dict(row)
            next_row["tenant_id"] = tenant
            for key in ("detail_json",):
                if next_row.get(key) is not None:
                    next_row[key] = self._json_dumps(next_row[key])
            payload_rows.append(next_row)
        self.session.insert_dicts("memory_relation_tuning_runs", payload_rows)
