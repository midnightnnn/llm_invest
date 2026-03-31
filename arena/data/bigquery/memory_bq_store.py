"""Memory and board BQ store — memory events, graph nodes/edges, board posts, research briefings."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from arena.memory.graph import (
    build_board_post_graph_node,
    build_memory_event_graph_edges,
    build_memory_event_graph_node,
    build_research_briefing_graph_node,
    ensure_memory_event_graph_ids,
)
from arena.memory.thesis import ACTIVE_THESIS_EVENT_TYPES, CLOSED_THESIS_EVENT_TYPES, THESIS_EVENT_TYPES
from arena.models import BoardPost, MemoryEvent

if TYPE_CHECKING:
    from arena.data.bigquery.session import BigQuerySession

logger = logging.getLogger(__name__)

_MEMORY_SELECT_COLUMNS = (
    "event_id, created_at, agent_id, event_type, summary, trading_mode, payload_json, "
    "importance_score, outcome_score, score, memory_tier, expires_at, promoted_at, "
    "semantic_key, context_tags_json, primary_regime, primary_strategy_tag, primary_sector, "
    "access_count, last_accessed_at, decay_score, effective_score, graph_node_id, causal_chain_id"
)


def _json_safe(value: Any) -> Any:
    """Converts nested values to JSON-serializable primitives for insert_rows_json."""
    if isinstance(value, datetime):
        # BQ streaming insert expects "YYYY-MM-DD HH:MM:SS.SSSSSS" (no T, no tz offset)
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class MemoryBQStore:
    """Memory events, board posts, graph nodes/edges, and research briefings."""

    def __init__(self, session: BigQuerySession) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Board posts
    # ------------------------------------------------------------------

    def write_board_post(self, post: BoardPost, *, tenant_id: str | None = None) -> None:
        """Stores inter-agent board posts."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        sql = f"""
        INSERT INTO `{self.session.dataset_fqn}.board_posts`
        (tenant_id, post_id, cycle_id, created_at, agent_id, title, body, draft_summary, trading_mode, tickers)
        VALUES (@tenant_id, @post_id, @cycle_id, @created_at, @agent_id, @title, @body, @draft_summary, @trading_mode, @tickers)
        """
        self.session.execute(
            sql,
            {
                "tenant_id": tenant,
                "post_id": post.post_id,
                "cycle_id": post.cycle_id or None,
                "created_at": post.created_at,
                "agent_id": post.agent_id,
                "title": post.title,
                "body": post.body,
                "draft_summary": post.draft_summary or None,
                "trading_mode": post.trading_mode,
                "tickers": post.tickers,
            },
        )
        self.upsert_memory_graph_nodes([build_board_post_graph_node(post)], tenant_id=tenant)

    def recent_board_posts(
        self,
        limit: int,
        trading_mode: str = "paper",
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns recent board posts for context building."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        sql = f"""
        SELECT post_id, created_at, agent_id, title, body, trading_mode, tickers
        FROM `{self.session.dataset_fqn}.board_posts`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        ORDER BY created_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, {"tenant_id": tenant, "limit": limit, "trading_mode": trading_mode})

    def board_posts_for_cycle(
        self,
        *,
        cycle_id: str,
        agent_id: str | None = None,
        limit: int = 10,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns board posts created within a specific cycle."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        filters = [
            "tenant_id = @tenant_id",
            "cycle_id = @cycle_id",
            "trading_mode = @trading_mode",
        ]
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "cycle_id": str(cycle_id or "").strip(),
            "trading_mode": trading_mode,
            "limit": max(1, int(limit)),
        }
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = str(agent_id or "").strip()
        sql = f"""
        SELECT post_id, cycle_id, created_at, agent_id, title, body, draft_summary, trading_mode, tickers
        FROM `{self.session.dataset_fqn}.board_posts`
        WHERE {' AND '.join(filters)}
        ORDER BY created_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    # ------------------------------------------------------------------
    # Memory events
    # ------------------------------------------------------------------

    def write_memory_event(self, event: MemoryEvent, *, tenant_id: str | None = None) -> None:
        """Stores memory events used by long-term retrieval."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        ensure_memory_event_graph_ids(event)
        sql = f"""
        INSERT INTO `{self.session.dataset_fqn}.agent_memory_events`
        (
          tenant_id, event_id, created_at, agent_id, event_type, summary, trading_mode, payload_json,
          importance_score, outcome_score, score, memory_tier, expires_at, promoted_at, semantic_key,
          context_tags_json, primary_regime, primary_strategy_tag, primary_sector, access_count,
          last_accessed_at, decay_score, effective_score, graph_node_id, causal_chain_id
        )
        VALUES (
          @tenant_id, @event_id, @created_at, @agent_id, @event_type, @summary, @trading_mode, @payload_json,
          @importance_score, @outcome_score, @score, @memory_tier, @expires_at, @promoted_at, @semantic_key,
          @context_tags_json, @primary_regime, @primary_strategy_tag, @primary_sector, @access_count,
          @last_accessed_at, @decay_score, @effective_score, @graph_node_id, @causal_chain_id
        )
        """
        importance_score = event.importance_score
        if importance_score is None:
            importance_score = event.score
        self.session.execute(
            sql,
            {
                "tenant_id": tenant,
                "event_id": event.event_id,
                "created_at": event.created_at,
                "agent_id": event.agent_id,
                "event_type": event.event_type,
                "summary": event.summary,
                "trading_mode": event.trading_mode,
                "payload_json": json.dumps(event.payload, ensure_ascii=False, default=str),
                "importance_score": importance_score,
                "outcome_score": ("FLOAT64", event.outcome_score),
                "score": event.score,
                "memory_tier": str(event.memory_tier or "").strip().lower() or None,
                "expires_at": ("TIMESTAMP", event.expires_at),
                "promoted_at": ("TIMESTAMP", event.promoted_at),
                "semantic_key": str(event.semantic_key or "").strip() or None,
                "context_tags_json": (
                    "JSON",
                    json.dumps(event.context_tags, ensure_ascii=False, default=str) if event.context_tags else None,
                ),
                "primary_regime": str(event.primary_regime or "").strip().lower() or None,
                "primary_strategy_tag": str(event.primary_strategy_tag or "").strip().lower() or None,
                "primary_sector": str(event.primary_sector or "").strip().lower() or None,
                "access_count": ("INT64", event.access_count),
                "last_accessed_at": ("TIMESTAMP", event.last_accessed_at),
                "decay_score": ("FLOAT64", event.decay_score),
                "effective_score": ("FLOAT64", event.effective_score),
                "graph_node_id": str(event.graph_node_id or "").strip() or None,
                "causal_chain_id": str(event.causal_chain_id or "").strip() or None,
            },
        )
        self.upsert_memory_graph_nodes([build_memory_event_graph_node(event)], tenant_id=tenant)
        self.upsert_memory_graph_edges(build_memory_event_graph_edges(event), tenant_id=tenant)

    def recent_memory_events(
        self,
        agent_id: str,
        limit: int,
        trading_mode: str = "paper",
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns recent memory events for a specific agent."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        sql = f"""
        SELECT {_MEMORY_SELECT_COLUMNS}
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND trading_mode = @trading_mode
        ORDER BY created_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(
            sql,
            {"tenant_id": tenant, "agent_id": agent_id, "limit": limit, "trading_mode": trading_mode},
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
        """Returns memory events linked to a specific cycle via payload.cycle_id or payload.intent.cycle_id."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        filters = [
            "tenant_id = @tenant_id",
            "agent_id = @agent_id",
            "trading_mode = @trading_mode",
            "("
            "COALESCE(JSON_VALUE(payload_json, '$.cycle_id'), '') = @cycle_id "
            "OR COALESCE(JSON_VALUE(payload_json, '$.intent.cycle_id'), '') = @cycle_id"
            ")",
        ]
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "agent_id": agent_id,
            "trading_mode": trading_mode,
            "cycle_id": str(cycle_id or "").strip(),
            "limit": max(1, int(limit)),
        }
        clean_types = [str(token or "").strip() for token in (event_types or []) if str(token or "").strip()]
        if clean_types:
            filters.append("event_type IN UNNEST(@event_types)")
            params["event_types"] = clean_types
        sql = f"""
        SELECT {_MEMORY_SELECT_COLUMNS}
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE {' AND '.join(filters)}
        ORDER BY created_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    def active_thesis_events(
        self,
        *,
        agent_id: str,
        tickers: list[str],
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns the latest active thesis event for each requested ticker."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        clean_tickers = [str(token or "").strip().upper() for token in tickers if str(token or "").strip()]
        if not clean_tickers:
            return []
        sql = f"""
        SELECT {_MEMORY_SELECT_COLUMNS}
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND trading_mode = @trading_mode
          AND event_type IN UNNEST(@event_types)
          AND UPPER(COALESCE(JSON_VALUE(payload_json, '$.ticker'), '')) IN UNNEST(@tickers)
        ORDER BY created_at DESC
        LIMIT @limit
        """
        rows = self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "agent_id": agent_id,
                "trading_mode": trading_mode,
                "event_types": sorted(THESIS_EVENT_TYPES),
                "tickers": clean_tickers,
                "limit": max(24, len(clean_tickers) * 12),
            },
        )
        latest_by_semantic_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload_raw = row.get("payload_json")
            payload = {}
            if isinstance(payload_raw, str) and payload_raw.strip():
                try:
                    payload = json.loads(payload_raw)
                except Exception:
                    payload = {}
            elif isinstance(payload_raw, dict):
                payload = payload_raw
            semantic_key = str(row.get("semantic_key") or payload.get("thesis_id") or "").strip()
            if not semantic_key or semantic_key in latest_by_semantic_key:
                continue
            latest_by_semantic_key[semantic_key] = row

        active_by_ticker: dict[str, dict[str, Any]] = {}
        for row in latest_by_semantic_key.values():
            event_type = str(row.get("event_type") or "").strip().lower()
            if event_type not in ACTIVE_THESIS_EVENT_TYPES:
                continue
            payload_raw = row.get("payload_json")
            payload = {}
            if isinstance(payload_raw, str) and payload_raw.strip():
                try:
                    payload = json.loads(payload_raw)
                except Exception:
                    payload = {}
            elif isinstance(payload_raw, dict):
                payload = payload_raw
            ticker = str(payload.get("ticker") or "").strip().upper()
            if ticker and ticker not in active_by_ticker:
                active_by_ticker[ticker] = row
        return [active_by_ticker[ticker] for ticker in clean_tickers if ticker in active_by_ticker]

    def closed_thesis_keys_for_cycle(
        self,
        *,
        agent_id: str,
        cycle_id: str,
        limit: int = 4,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[str]:
        """Returns recently closed thesis semantic keys for the given cycle."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        sql = f"""
        SELECT
          COALESCE(semantic_key, JSON_VALUE(payload_json, '$.thesis_id')) AS semantic_key,
          MAX(created_at) AS last_created_at
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND trading_mode = @trading_mode
          AND event_type IN UNNEST(@event_types)
          AND (
            COALESCE(JSON_VALUE(payload_json, '$.cycle_id'), '') = @cycle_id
            OR COALESCE(JSON_VALUE(payload_json, '$.intent.cycle_id'), '') = @cycle_id
          )
          AND COALESCE(semantic_key, JSON_VALUE(payload_json, '$.thesis_id'), '') != ''
        GROUP BY semantic_key
        ORDER BY last_created_at DESC
        LIMIT @limit
        """
        rows = self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "agent_id": agent_id,
                "trading_mode": trading_mode,
                "event_types": sorted(CLOSED_THESIS_EVENT_TYPES),
                "cycle_id": str(cycle_id or "").strip(),
                "limit": max(1, int(limit)),
            },
        )
        out: list[str] = []
        for row in rows:
            token = str(row.get("semantic_key") or "").strip()
            if token and token not in out:
                out.append(token)
        return out

    def memory_events_by_semantic_keys(
        self,
        *,
        agent_id: str,
        semantic_keys: list[str],
        event_types: list[str] | None = None,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Loads memory rows for one or more semantic keys ordered by thesis timeline."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        clean_keys = [str(token or "").strip() for token in semantic_keys if str(token or "").strip()]
        if not clean_keys:
            return []
        filters = [
            "tenant_id = @tenant_id",
            "agent_id = @agent_id",
            "trading_mode = @trading_mode",
            "COALESCE(semantic_key, JSON_VALUE(payload_json, '$.thesis_id')) IN UNNEST(@semantic_keys)",
        ]
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "agent_id": agent_id,
            "trading_mode": trading_mode,
            "semantic_keys": clean_keys,
        }
        clean_types = [str(token or "").strip() for token in (event_types or []) if str(token or "").strip()]
        if clean_types:
            filters.append("event_type IN UNNEST(@event_types)")
            params["event_types"] = clean_types
        sql = f"""
        SELECT {_MEMORY_SELECT_COLUMNS}
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE {' AND '.join(filters)}
        ORDER BY COALESCE(semantic_key, JSON_VALUE(payload_json, '$.thesis_id')) ASC, created_at ASC
        """
        return self.session.fetch_rows(sql, params)

    def memory_event_exists_by_semantic_key(
        self,
        *,
        agent_id: str,
        event_type: str,
        semantic_key: str,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> bool:
        """Returns True when a memory event already exists for the given semantic key."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        key = str(semantic_key or "").strip()
        if not key:
            return False
        sql = f"""
        SELECT event_id
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND trading_mode = @trading_mode
          AND event_type = @event_type
          AND semantic_key = @semantic_key
        LIMIT 1
        """
        rows = self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "agent_id": agent_id,
                "trading_mode": trading_mode,
                "event_type": str(event_type or "").strip(),
                "semantic_key": key,
            },
        )
        return bool(rows)

    def memory_events_by_ids(
        self,
        *,
        agent_id: str,
        event_ids: list[str],
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hydrates memory rows for a given list of event ids."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        clean_ids = [str(eid or "").strip() for eid in event_ids if str(eid or "").strip()]
        if not clean_ids:
            return []
        sql = f"""
        SELECT {_MEMORY_SELECT_COLUMNS}
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND trading_mode = @trading_mode
          AND event_id IN UNNEST(@event_ids)
        """
        return self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "agent_id": agent_id,
                "trading_mode": trading_mode,
                "event_ids": clean_ids,
            },
        )

    def memory_events_by_ids_any_agent(
        self,
        *,
        event_ids: list[str],
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hydrates memory rows for given ids across all agents in the tenant."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        clean_ids = [str(eid or "").strip() for eid in event_ids if str(eid or "").strip()]
        if not clean_ids:
            return []
        sql = f"""
        SELECT {_MEMORY_SELECT_COLUMNS}
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
          AND event_id IN UNNEST(@event_ids)
        """
        return self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "trading_mode": trading_mode,
                "event_ids": clean_ids,
            },
        )

    def memory_event_by_id(
        self,
        *,
        event_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Loads one memory event regardless of agent for graph projection refreshes."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        token = str(event_id or "").strip()
        if not token:
            return None
        sql = f"""
        SELECT {_MEMORY_SELECT_COLUMNS}
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND event_id = @event_id
        LIMIT 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant, "event_id": token})
        return rows[0] if rows else None

    def find_trade_execution_memory_event(
        self,
        *,
        agent_id: str,
        order_id: str,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Returns one trade_execution memory row matched by report.order_id."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        token = str(order_id or "").strip()
        if not token:
            return None
        sql = f"""
        SELECT {_MEMORY_SELECT_COLUMNS}
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND event_type = 'trade_execution'
          AND trading_mode = @trading_mode
          AND JSON_VALUE(payload_json, '$.report.order_id') = @order_id
        ORDER BY created_at DESC
        LIMIT 1
        """
        rows = self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "agent_id": agent_id,
                "trading_mode": trading_mode,
                "order_id": token,
            },
        )
        return rows[0] if rows else None

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
        """Updates one existing memory event payload/summary/score."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        sql = f"""
        UPDATE `{self.session.dataset_fqn}.agent_memory_events`
        SET summary = @summary,
            payload_json = @payload_json,
            importance_score = @importance_score,
            outcome_score = @outcome_score,
            score = @score,
            memory_tier = COALESCE(@memory_tier, memory_tier),
            expires_at = COALESCE(@expires_at, expires_at),
            context_tags_json = COALESCE(@context_tags_json, context_tags_json),
            primary_regime = COALESCE(@primary_regime, primary_regime),
            primary_strategy_tag = COALESCE(@primary_strategy_tag, primary_strategy_tag),
            primary_sector = COALESCE(@primary_sector, primary_sector),
            graph_node_id = COALESCE(@graph_node_id, graph_node_id),
            causal_chain_id = COALESCE(@causal_chain_id, causal_chain_id)
        WHERE tenant_id = @tenant_id
          AND event_id = @event_id
        """
        self.session.execute(
            sql,
            {
                "tenant_id": tenant,
                "event_id": str(event_id or "").strip(),
                "summary": str(summary or "").strip(),
                "payload_json": json.dumps(payload or {}, ensure_ascii=False, default=str),
                "importance_score": ("FLOAT64", importance_score),
                "outcome_score": ("FLOAT64", outcome_score),
                "score": max(0.0, min(float(score), 1.0)),
                "memory_tier": str(memory_tier or "").strip().lower() or None,
                "expires_at": ("TIMESTAMP", expires_at),
                "context_tags_json": (
                    "JSON",
                    json.dumps(context_tags, ensure_ascii=False, default=str) if context_tags else None,
                ),
                "primary_regime": str(primary_regime or "").strip().lower() or None,
                "primary_strategy_tag": str(primary_strategy_tag or "").strip().lower() or None,
                "primary_sector": str(primary_sector or "").strip().lower() or None,
                "graph_node_id": str(graph_node_id or "").strip() or None,
                "causal_chain_id": str(causal_chain_id or "").strip() or None,
            },
        )
        row = self.memory_event_by_id(event_id=event_id, tenant_id=tenant)
        if row:
            self.upsert_memory_graph_nodes([build_memory_event_graph_node(row)], tenant_id=tenant)
            self.upsert_memory_graph_edges(build_memory_event_graph_edges(row), tenant_id=tenant)

    def update_memory_score(self, event_id: str, new_score: float, *, tenant_id: str | None = None) -> None:
        """Updates the outcome_score of an existing memory event."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        sql = f"""
        UPDATE `{self.session.dataset_fqn}.agent_memory_events`
        SET outcome_score = @new_score
        WHERE tenant_id = @tenant_id
          AND event_id = @event_id
        """
        self.session.execute(
            sql,
            {
                "tenant_id": tenant,
                "event_id": event_id,
                "new_score": max(0.0, min(float(new_score), 1.0)),
            },
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
        """Finds recent BUY execution memories for a specific ticker to enable score feedback on SELL."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        sql = f"""
        SELECT {_MEMORY_SELECT_COLUMNS}
        FROM `{self.session.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND event_type = 'trade_execution'
          AND summary LIKE CONCAT(@ticker, ' BUY%')
          AND trading_mode = @trading_mode
        ORDER BY created_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "agent_id": agent_id,
                "ticker": ticker,
                "limit": limit,
                "trading_mode": trading_mode,
            },
        )

    # ------------------------------------------------------------------
    # Memory access events
    # ------------------------------------------------------------------

    def append_memory_access_events(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends memory access log rows for future adaptive forgetting."""
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        table_id = f"{self.session.dataset_fqn}.memory_access_events"
        payload_rows: list[dict[str, Any]] = []
        for row in rows:
            payload_rows.append(
                {
                    "tenant_id": tenant,
                    "access_id": str(row.get("access_id") or "").strip(),
                    "accessed_at": _json_safe(row.get("accessed_at")),
                    "event_id": str(row.get("event_id") or "").strip(),
                    "agent_id": str(row.get("agent_id") or "").strip() or None,
                    "source_agent_id": str(row.get("source_agent_id") or "").strip() or None,
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "access_type": str(row.get("access_type") or "retrieval").strip().lower() or "retrieval",
                    "query_text": str(row.get("query_text") or "").strip() or None,
                    "retrieval_score": (
                        float(row.get("retrieval_score"))
                        if row.get("retrieval_score") is not None
                        else None
                    ),
                    "used_in_prompt": (
                        bool(row.get("used_in_prompt"))
                        if row.get("used_in_prompt") is not None
                        else None
                    ),
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "detail_json": (
                        json.dumps(row.get("detail_json"), ensure_ascii=False, separators=(",", ":"))
                        if row.get("detail_json") is not None
                        else None
                    ),
                }
            )
        errors = self.session.client.insert_rows_json(table_id, payload_rows)
        if errors:
            raise RuntimeError(f"memory_access_events insert failed: {errors}")

    # ------------------------------------------------------------------
    # Memory graph nodes
    # ------------------------------------------------------------------

    def append_memory_graph_nodes(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends projected graph nodes for phase-2 full graph migration."""
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        table_id = f"{self.session.dataset_fqn}.memory_graph_nodes"
        payload_rows: list[dict[str, Any]] = []
        for row in rows:
            payload_rows.append(
                {
                    "tenant_id": tenant,
                    "node_id": str(row.get("node_id") or "").strip(),
                    "created_at": _json_safe(row.get("created_at")),
                    "node_kind": str(row.get("node_kind") or "").strip(),
                    "source_table": str(row.get("source_table") or "").strip(),
                    "source_id": str(row.get("source_id") or "").strip(),
                    "agent_id": str(row.get("agent_id") or "").strip() or None,
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "summary": str(row.get("summary") or "").strip() or None,
                    "ticker": str(row.get("ticker") or "").strip().upper() or None,
                    "memory_tier": str(row.get("memory_tier") or "").strip().lower() or None,
                    "primary_regime": str(row.get("primary_regime") or "").strip().lower() or None,
                    "context_tags_json": (
                        json.dumps(_json_safe(row.get("context_tags_json")), ensure_ascii=False, separators=(",", ":"))
                        if row.get("context_tags_json") is not None
                        else None
                    ),
                    "payload_json": (
                        json.dumps(_json_safe(row.get("payload_json")), ensure_ascii=False, separators=(",", ":"))
                        if row.get("payload_json") is not None
                        else None
                    ),
                }
            )
        errors = self.session.client.insert_rows_json(table_id, payload_rows)
        if errors:
            raise RuntimeError(f"memory_graph_nodes insert failed: {errors}")

    def upsert_memory_graph_nodes(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Merges graph nodes by node_id to keep runtime projection idempotent."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        for row in rows:
            node_id = str(row.get("node_id") or "").strip()
            if not node_id:
                continue
            sql = f"""
            MERGE `{self.session.dataset_fqn}.memory_graph_nodes` AS t
            USING (
              SELECT
                @tenant_id AS tenant_id,
                @node_id AS node_id,
                @created_at AS created_at,
                @node_kind AS node_kind,
                @source_table AS source_table,
                @source_id AS source_id,
                @agent_id AS agent_id,
                @trading_mode AS trading_mode,
                @cycle_id AS cycle_id,
                @summary AS summary,
                @ticker AS ticker,
                @memory_tier AS memory_tier,
                @primary_regime AS primary_regime,
                @context_tags_json AS context_tags_json,
                @payload_json AS payload_json
            ) AS s
            ON t.tenant_id = s.tenant_id AND t.node_id = s.node_id
            WHEN MATCHED THEN UPDATE SET
              created_at = s.created_at,
              node_kind = s.node_kind,
              source_table = s.source_table,
              source_id = s.source_id,
              agent_id = s.agent_id,
              trading_mode = s.trading_mode,
              cycle_id = s.cycle_id,
              summary = s.summary,
              ticker = s.ticker,
              memory_tier = s.memory_tier,
              primary_regime = s.primary_regime,
              context_tags_json = s.context_tags_json,
              payload_json = s.payload_json
            WHEN NOT MATCHED THEN INSERT
              (tenant_id, node_id, created_at, node_kind, source_table, source_id, agent_id, trading_mode, cycle_id, summary, ticker, memory_tier, primary_regime, context_tags_json, payload_json)
              VALUES
              (s.tenant_id, s.node_id, s.created_at, s.node_kind, s.source_table, s.source_id, s.agent_id, s.trading_mode, s.cycle_id, s.summary, s.ticker, s.memory_tier, s.primary_regime, s.context_tags_json, s.payload_json)
            """
            self.session.execute(
                sql,
                {
                    "tenant_id": tenant,
                    "node_id": node_id,
                    "created_at": row.get("created_at"),
                    "node_kind": str(row.get("node_kind") or "").strip(),
                    "source_table": str(row.get("source_table") or "").strip(),
                    "source_id": str(row.get("source_id") or "").strip(),
                    "agent_id": str(row.get("agent_id") or "").strip() or None,
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "summary": str(row.get("summary") or "").strip() or None,
                    "ticker": str(row.get("ticker") or "").strip().upper() or None,
                    "memory_tier": str(row.get("memory_tier") or "").strip().lower() or None,
                    "primary_regime": str(row.get("primary_regime") or "").strip().lower() or None,
                    "context_tags_json": (
                        "JSON",
                        json.dumps(_json_safe(row.get("context_tags_json")), ensure_ascii=False, separators=(",", ":"))
                        if row.get("context_tags_json") is not None
                        else None,
                    ),
                    "payload_json": (
                        "JSON",
                        json.dumps(_json_safe(row.get("payload_json")), ensure_ascii=False, separators=(",", ":"))
                        if row.get("payload_json") is not None
                        else None,
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Memory graph edges
    # ------------------------------------------------------------------

    def append_memory_graph_edges(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends projected graph edges for phase-2 full graph migration."""
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        table_id = f"{self.session.dataset_fqn}.memory_graph_edges"
        payload_rows: list[dict[str, Any]] = []
        for row in rows:
            payload_rows.append(
                {
                    "tenant_id": tenant,
                    "edge_id": str(row.get("edge_id") or "").strip(),
                    "created_at": _json_safe(row.get("created_at")),
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "from_node_id": str(row.get("from_node_id") or "").strip(),
                    "to_node_id": str(row.get("to_node_id") or "").strip(),
                    "edge_type": str(row.get("edge_type") or "").strip().upper(),
                    "edge_strength": (
                        float(row.get("edge_strength"))
                        if row.get("edge_strength") is not None
                        else None
                    ),
                    "confidence": (
                        float(row.get("confidence"))
                        if row.get("confidence") is not None
                        else None
                    ),
                    "causal_chain_id": str(row.get("causal_chain_id") or "").strip() or None,
                    "detail_json": (
                        json.dumps(_json_safe(row.get("detail_json")), ensure_ascii=False, separators=(",", ":"))
                        if row.get("detail_json") is not None
                        else None
                    ),
                }
            )
        errors = self.session.client.insert_rows_json(table_id, payload_rows)
        if errors:
            raise RuntimeError(f"memory_graph_edges insert failed: {errors}")

    def upsert_memory_graph_edges(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Merges graph edges by edge_id to keep runtime projection idempotent."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        for row in rows:
            edge_id = str(row.get("edge_id") or "").strip()
            if not edge_id:
                continue
            sql = f"""
            MERGE `{self.session.dataset_fqn}.memory_graph_edges` AS t
            USING (
              SELECT
                @tenant_id AS tenant_id,
                @edge_id AS edge_id,
                @created_at AS created_at,
                @trading_mode AS trading_mode,
                @cycle_id AS cycle_id,
                @from_node_id AS from_node_id,
                @to_node_id AS to_node_id,
                @edge_type AS edge_type,
                @edge_strength AS edge_strength,
                @confidence AS confidence,
                @causal_chain_id AS causal_chain_id,
                @detail_json AS detail_json
            ) AS s
            ON t.tenant_id = s.tenant_id AND t.edge_id = s.edge_id
            WHEN MATCHED THEN UPDATE SET
              created_at = s.created_at,
              trading_mode = s.trading_mode,
              cycle_id = s.cycle_id,
              from_node_id = s.from_node_id,
              to_node_id = s.to_node_id,
              edge_type = s.edge_type,
              edge_strength = s.edge_strength,
              confidence = s.confidence,
              causal_chain_id = s.causal_chain_id,
              detail_json = s.detail_json
            WHEN NOT MATCHED THEN INSERT
              (tenant_id, edge_id, created_at, trading_mode, cycle_id, from_node_id, to_node_id, edge_type, edge_strength, confidence, causal_chain_id, detail_json)
              VALUES
              (s.tenant_id, s.edge_id, s.created_at, s.trading_mode, s.cycle_id, s.from_node_id, s.to_node_id, s.edge_type, s.edge_strength, s.confidence, s.causal_chain_id, s.detail_json)
            """
            self.session.execute(
                sql,
                {
                    "tenant_id": tenant,
                    "edge_id": edge_id,
                    "created_at": row.get("created_at"),
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "cycle_id": str(row.get("cycle_id") or "").strip() or None,
                    "from_node_id": str(row.get("from_node_id") or "").strip(),
                    "to_node_id": str(row.get("to_node_id") or "").strip(),
                    "edge_type": str(row.get("edge_type") or "").strip().upper(),
                    "edge_strength": ("FLOAT64", row.get("edge_strength")),
                    "confidence": ("FLOAT64", row.get("confidence")),
                    "causal_chain_id": str(row.get("causal_chain_id") or "").strip() or None,
                    "detail_json": (
                        "JSON",
                        json.dumps(_json_safe(row.get("detail_json")), ensure_ascii=False, separators=(",", ":"))
                        if row.get("detail_json") is not None
                        else None,
                    ),
                },
            )

    # ------------------------------------------------------------------
    # Graph neighbor query
    # ------------------------------------------------------------------

    def memory_graph_neighbors(
        self,
        *,
        seed_node_ids: list[str],
        trading_mode: str = "paper",
        min_confidence: float = 0.0,
        limit: int = 24,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns graph neighbors for the provided seed nodes with joined node metadata."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        clean_ids = [str(node_id or "").strip() for node_id in seed_node_ids if str(node_id or "").strip()]
        if not clean_ids:
            return []
        sql = f"""
        WITH seed AS (
          SELECT DISTINCT seed_node_id
          FROM UNNEST(@seed_node_ids) AS seed_node_id
        ),
        matched_edges AS (
          SELECT
            seed.seed_node_id,
            CASE
              WHEN e.from_node_id = seed.seed_node_id THEN 'outgoing'
              ELSE 'incoming'
            END AS direction,
            CASE
              WHEN e.from_node_id = seed.seed_node_id THEN e.to_node_id
              ELSE e.from_node_id
            END AS neighbor_node_id,
            e.edge_id,
            e.created_at AS edge_created_at,
            e.cycle_id AS edge_cycle_id,
            e.edge_type,
            e.edge_strength,
            e.confidence,
            e.causal_chain_id,
            e.detail_json
          FROM seed
          JOIN `{self.session.dataset_fqn}.memory_graph_edges` AS e
            ON e.tenant_id = @tenant_id
           AND e.trading_mode = @trading_mode
           AND (e.from_node_id = seed.seed_node_id OR e.to_node_id = seed.seed_node_id)
          WHERE COALESCE(e.confidence, 1.0) >= @min_confidence
        ),
        deduped AS (
          SELECT * EXCEPT(rn)
          FROM (
            SELECT
              matched_edges.*,
              ROW_NUMBER() OVER (
                PARTITION BY seed_node_id, neighbor_node_id, edge_type
                ORDER BY COALESCE(confidence, 1.0) DESC, COALESCE(edge_strength, 0.0) DESC, edge_created_at DESC
              ) AS rn
            FROM matched_edges
          )
          WHERE rn = 1
        )
        SELECT
          d.seed_node_id,
          d.direction,
          d.neighbor_node_id,
          d.edge_id,
          d.edge_created_at,
          d.edge_cycle_id,
          d.edge_type,
          d.edge_strength,
          d.confidence,
          d.causal_chain_id,
          d.detail_json,
          n.created_at AS node_created_at,
          n.node_kind,
          n.source_table,
          n.source_id,
          n.agent_id,
          n.trading_mode AS node_trading_mode,
          n.cycle_id,
          n.summary,
          n.ticker,
          n.memory_tier,
          n.primary_regime,
          n.context_tags_json,
          n.payload_json
        FROM deduped AS d
        JOIN `{self.session.dataset_fqn}.memory_graph_nodes` AS n
          ON n.tenant_id = @tenant_id
         AND n.node_id = d.neighbor_node_id
        WHERE n.trading_mode = @trading_mode
        ORDER BY COALESCE(d.confidence, 1.0) DESC, COALESCE(d.edge_strength, 0.0) DESC, d.edge_created_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "seed_node_ids": clean_ids,
                "trading_mode": str(trading_mode or "paper").strip().lower() or "paper",
                "min_confidence": max(0.0, min(float(min_confidence), 1.0)),
                "limit": max(1, int(limit)),
            },
        )

    # ------------------------------------------------------------------
    # Board post existence check
    # ------------------------------------------------------------------

    def board_post_exists(
        self,
        *,
        title: str,
        agent_id: str | None = None,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> bool:
        """Checks whether a board post with the given title already exists."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        filters = ["tenant_id = @tenant_id", "title = @title", "trading_mode = @trading_mode"]
        params: dict[str, Any] = {"tenant_id": tenant, "title": title, "trading_mode": trading_mode}
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = agent_id

        where = " AND ".join(filters)
        sql = f"""
        SELECT COUNT(1) AS cnt
        FROM `{self.session.dataset_fqn}.board_posts`
        WHERE {where}
        """
        rows = self.session.fetch_rows(sql, params)
        return bool(rows and int(rows[0].get("cnt") or 0) > 0)

    # ------------------------------------------------------------------
    # Research briefings
    # ------------------------------------------------------------------

    def insert_research_briefings(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        """리서치 브리핑 배치 삽입"""
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        # BQ streaming insert
        table_id = f"{self.session.dataset_fqn}.research_briefings"
        safe_rows = []
        graph_rows = []
        for row in rows:
            payload = dict(row)
            payload["tenant_id"] = tenant
            graph_rows.append(dict(payload))
            safe_rows.append(_json_safe(payload))
        errors = self.session.client.insert_rows_json(table_id, safe_rows)
        if errors:
            logger.error("[red]BigQuery research_briefings insert failed[/red] errors=%s", errors)
            return
        graph_nodes = [build_research_briefing_graph_node(row) for row in graph_rows]
        self.upsert_memory_graph_nodes(graph_nodes, tenant_id=tenant)

    def get_research_briefings(
        self,
        *,
        tickers: list[str] | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """리서치 브리핑 조회 (tickers/categories 복수 필터, 둘 다 비우면 최근 전체)."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        conditions = ["tenant_id = @tenant_id", "trading_mode = @trading_mode"]
        params: dict[str, Any] = {"tenant_id": tenant, "trading_mode": trading_mode, "limit": limit}

        filters: list[str] = []
        if tickers:
            filters.append("ticker IN UNNEST(@tickers)")
            params["tickers"] = tickers
        if categories:
            filters.append("category IN UNNEST(@categories)")
            params["categories"] = categories
        if filters:
            conditions.append(f"({' OR '.join(filters)})")

        sql = f"""
        SELECT briefing_id, created_at, ticker, category, headline, summary, sources
        FROM `{self.session.dataset_fqn}.research_briefings`
        WHERE {' AND '.join(conditions)}
        ORDER BY created_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)
