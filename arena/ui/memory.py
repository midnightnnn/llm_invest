from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.memory.cleanup import run_memory_cleanup
from arena.memory.policy import (
    GLOBAL_MEMORY_PROMPT_CONFIG_KEY,
    MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY,
    MEMORY_POLICY_CONFIG_KEY,
    build_memory_graph,
    default_memory_policy,
    load_global_compaction_prompt,
    load_tenant_compaction_prompt,
    load_memory_policy,
    normalize_memory_policy,
    serialize_memory_policy,
)
from arena.ui.http import html_response, json_response
from arena.ui.templating import render_ui_template
_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_THREE_JS_PATH = _VENDOR_DIR / "three.min.js"
_FORCE_GRAPH_JS_PATH = _VENDOR_DIR / "3d-force-graph.min.js"
logger = logging.getLogger(__name__)

EVENT_TYPE_TO_POLICY_GROUP: dict[str, str] = {
    "trade_execution": "event_types",
    "strategy_reflection": "event_types",
    "manual_note": "event_types",
    "react_tools_summary": "react_injection",
    "thesis_open": "event_types",
    "thesis_update": "event_types",
    "thesis_invalidated": "event_types",
    "thesis_realized": "event_types",
}
NODE_KIND_TO_POLICY_GROUP: dict[str, str] = {
    "memory_event": "event_types",
    "order_intent": "event_types",
    "execution_report": "event_types",
    "board_post": "graph",
    "research_briefing": "retrieval",
}
_DEFAULT_BRIDGE_GROUP = "event_types"


def _load_json_config_with_state(
    repo: BigQueryRepository,
    *,
    tenant_id: str,
    config_key: str,
) -> tuple[dict[str, Any], bool]:
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return {}, False
    try:
        raw = getter(tenant_id, config_key)
    except Exception as exc:
        logger.warning(
            "[yellow]memory config load failed[/yellow] tenant=%s key=%s err=%s",
            tenant_id,
            config_key,
            str(exc),
        )
        return {}, True
    text = str(raw or "").strip()
    if not text:
        return {}, False
    try:
        parsed = json.loads(text)
    except Exception as exc:
        logger.warning(
            "[yellow]memory config JSON parse failed[/yellow] tenant=%s key=%s err=%s",
            tenant_id,
            config_key,
            str(exc),
        )
        return {}, True
    if isinstance(parsed, dict):
        return parsed, False
    logger.warning(
        "[yellow]memory config JSON type mismatch[/yellow] tenant=%s key=%s expected=object actual=%s",
        tenant_id,
        config_key,
        type(parsed).__name__,
    )
    return {}, True


def _load_json_config(repo: BigQueryRepository, *, tenant_id: str, config_key: str) -> dict[str, Any]:
    parsed, _invalid = _load_json_config_with_state(repo, tenant_id=tenant_id, config_key=config_key)
    return parsed


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _memory_payload_ticker(row: dict[str, Any]) -> str:
    payload = _json_dict(row.get("payload_json"))
    intent = payload.get("intent")
    if isinstance(intent, dict):
        ticker = str(intent.get("ticker") or "").strip().upper()
        if ticker:
            return ticker
    return str(payload.get("ticker") or "").strip().upper()


def _memory_context_badges(row: dict[str, Any]) -> list[str]:
    tags = _json_dict(row.get("context_tags_json"))
    out: list[str] = []
    for key in ("regime_tags", "strategy_tags", "sector_tags", "tickers"):
        value = tags.get(key)
        if isinstance(value, list):
            for item in value:
                token = str(item or "").strip()
                if token and token not in out:
                    out.append(token)
    return out[:6]


def _activity_payload(repo: BigQueryRepository, *, tenant_id: str, trading_mode: str) -> dict[str, Any]:
    stats = _stats_payload(repo, tenant_id=tenant_id, trading_mode=trading_mode)
    rows = repo.fetch_rows(
        f"""
        WITH access_summary AS (
          SELECT
            event_id,
            COUNT(1) AS access_events,
            COUNTIF(COALESCE(used_in_prompt, FALSE)) AS prompt_uses,
            MAX(IF(COALESCE(used_in_prompt, FALSE), accessed_at, NULL)) AS last_prompt_at
          FROM `{repo.dataset_fqn}.memory_access_events`
          WHERE tenant_id = @tenant_id
            AND trading_mode = @trading_mode
          GROUP BY event_id
        )
        SELECT
          m.event_id,
          m.created_at,
          m.agent_id,
          m.event_type,
          m.summary,
          m.memory_tier,
          m.primary_regime,
          m.primary_strategy_tag,
          m.primary_sector,
          m.access_count,
          m.last_accessed_at,
          m.effective_score,
          m.context_tags_json,
          m.payload_json,
          COALESCE(a.access_events, 0) AS access_events,
          COALESCE(a.prompt_uses, 0) AS prompt_uses,
          a.last_prompt_at
        FROM `{repo.dataset_fqn}.agent_memory_events` AS m
        LEFT JOIN access_summary AS a
          ON a.event_id = m.event_id
        WHERE m.tenant_id = @tenant_id
          AND m.trading_mode = @trading_mode
        ORDER BY
          COALESCE(a.prompt_uses, 0) DESC,
          COALESCE(m.last_accessed_at, m.created_at) DESC,
          m.created_at DESC
        LIMIT 8
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )

    examples: list[dict[str, Any]] = []
    for row in rows:
        examples.append(
            {
                "event_id": str(row.get("event_id") or "").strip(),
                "created_at": str(row.get("created_at") or "").strip(),
                "agent_id": str(row.get("agent_id") or "").strip().lower(),
                "event_type": str(row.get("event_type") or "").strip().lower(),
                "summary": str(row.get("summary") or "").strip(),
                "memory_tier": str(row.get("memory_tier") or "").strip().lower(),
                "primary_regime": str(row.get("primary_regime") or "").strip().lower(),
                "primary_strategy_tag": str(row.get("primary_strategy_tag") or "").strip().lower(),
                "primary_sector": str(row.get("primary_sector") or "").strip(),
                "ticker": _memory_payload_ticker(row),
                "access_count": int(row.get("access_count") or 0),
                "access_events": int(row.get("access_events") or 0),
                "prompt_uses": int(row.get("prompt_uses") or 0),
                "last_accessed_at": str(row.get("last_accessed_at") or "").strip(),
                "last_prompt_at": str(row.get("last_prompt_at") or "").strip(),
                "effective_score": row.get("effective_score"),
                "badges": _memory_context_badges(row),
            }
        )

    return {
        "tenant_id": tenant_id,
        "trading_mode": trading_mode,
        "stats": stats,
        "examples": examples,
    }


def _network_payload(
    repo: BigQueryRepository,
    *,
    tenant_id: str,
    trading_mode: str,
    days: int = 30,
    node_limit: int = 60,
    edge_limit: int = 160,
    agent_id: str = "",
    event_type: str = "",
    ticker: str = "",
    prompt_only: bool = False,
    min_confidence: float = 0.55,
) -> dict[str, Any]:
    clean_agent = str(agent_id or "").strip().lower()
    clean_event_type = str(event_type or "").strip().lower()
    clean_ticker = str(ticker or "").strip().upper()
    clean_days = max(1, min(int(days), 365))
    clean_node_limit = max(8, min(int(node_limit), 160))
    clean_edge_limit = max(16, min(int(edge_limit), 320))
    clean_confidence = max(0.0, min(float(min_confidence), 1.0))

    node_rows = repo.fetch_rows(
        f"""
        WITH access_summary AS (
          SELECT
            event_id,
            COUNT(1) AS access_events,
            COUNTIF(COALESCE(used_in_prompt, FALSE)) AS prompt_uses
          FROM `{repo.dataset_fqn}.memory_access_events`
          WHERE tenant_id = @tenant_id
            AND trading_mode = @trading_mode
          GROUP BY event_id
        )
        SELECT
          n.node_id,
          n.created_at,
          n.node_kind,
          n.source_table,
          n.source_id,
          n.agent_id,
          n.cycle_id,
          n.summary,
          n.ticker,
          n.memory_tier,
          n.primary_regime,
          n.context_tags_json,
          n.payload_json,
          m.event_type,
          m.access_count,
          m.last_accessed_at,
          m.effective_score,
          COALESCE(a.access_events, 0) AS access_events,
          COALESCE(a.prompt_uses, 0) AS prompt_uses
        FROM `{repo.dataset_fqn}.memory_graph_nodes` AS n
        LEFT JOIN `{repo.dataset_fqn}.agent_memory_events` AS m
          ON m.tenant_id = @tenant_id
         AND m.trading_mode = @trading_mode
         AND n.source_table = 'agent_memory_events'
         AND m.event_id = n.source_id
        LEFT JOIN access_summary AS a
          ON a.event_id = m.event_id
        WHERE n.tenant_id = @tenant_id
          AND n.trading_mode = @trading_mode
          AND n.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
          AND (@agent_id = '' OR LOWER(COALESCE(n.agent_id, m.agent_id, '')) = @agent_id)
          AND (@event_type = '' OR LOWER(COALESCE(m.event_type, '')) = @event_type)
          AND (@prompt_only = FALSE OR COALESCE(a.prompt_uses, 0) > 0)
        ORDER BY
          COALESCE(a.prompt_uses, 0) DESC,
          COALESCE(m.access_count, 0) DESC,
          n.created_at DESC
        LIMIT @node_limit
        """,
        {
            "tenant_id": tenant_id,
            "trading_mode": trading_mode,
            "days": clean_days,
            "agent_id": clean_agent,
            "event_type": clean_event_type,
            "prompt_only": bool(prompt_only),
            "node_limit": clean_node_limit,
        },
    )

    nodes: list[dict[str, Any]] = []
    node_ids: list[str] = []
    for row in node_rows:
        row_ticker = str(row.get("ticker") or "").strip().upper() or _memory_payload_ticker(row)
        if clean_ticker and row_ticker != clean_ticker:
            continue
        node_id = str(row.get("node_id") or "").strip()
        if not node_id:
            continue
        node_kind = str(row.get("node_kind") or "").strip().lower() or "memory_event"
        row_event_type = str(row.get("event_type") or "").strip().lower()
        group = row_event_type or node_kind
        prompt_uses = int(row.get("prompt_uses") or 0)
        access_count = int(row.get("access_count") or 0)
        access_events = int(row.get("access_events") or 0)
        effective_score = row.get("effective_score")
        try:
            score_bonus = max(float(effective_score or 0.0), 0.0)
        except (TypeError, ValueError):
            score_bonus = 0.0
        size = min(16.0, 5.0 + (prompt_uses * 0.9) + (access_events * 0.15) + (score_bonus * 2.4))
        nodes.append(
            {
                "id": node_id,
                "label": str(row.get("summary") or row_event_type or node_kind or node_id).strip()[:72] or node_id,
                "group": group,
                "size": round(size, 2),
                "node_kind": node_kind,
                "source_table": str(row.get("source_table") or "").strip(),
                "source_id": str(row.get("source_id") or "").strip(),
                "created_at": str(row.get("created_at") or "").strip(),
                "agent_id": str(row.get("agent_id") or "").strip().lower(),
                "cycle_id": str(row.get("cycle_id") or "").strip(),
                "summary": str(row.get("summary") or "").strip(),
                "ticker": row_ticker,
                "memory_tier": str(row.get("memory_tier") or "").strip().lower(),
                "primary_regime": str(row.get("primary_regime") or "").strip().lower(),
                "event_type": row_event_type,
                "access_count": access_count,
                "access_events": access_events,
                "prompt_uses": prompt_uses,
                "last_accessed_at": str(row.get("last_accessed_at") or "").strip(),
                "effective_score": effective_score,
                "used_in_prompt": prompt_uses > 0,
                "badges": _memory_context_badges(row),
                "bridge_target": EVENT_TYPE_TO_POLICY_GROUP.get(
                    row_event_type,
                    NODE_KIND_TO_POLICY_GROUP.get(node_kind, _DEFAULT_BRIDGE_GROUP),
                ),
            }
        )
        node_ids.append(node_id)

    edge_rows: list[dict[str, Any]] = []
    if node_ids:
        edge_rows = repo.fetch_rows(
            f"""
            SELECT
              edge_id,
              created_at,
              from_node_id,
              to_node_id,
              edge_type,
              edge_strength,
              confidence,
              causal_chain_id
            FROM `{repo.dataset_fqn}.memory_graph_edges`
            WHERE tenant_id = @tenant_id
              AND trading_mode = @trading_mode
              AND from_node_id IN UNNEST(@node_ids)
              AND to_node_id IN UNNEST(@node_ids)
              AND COALESCE(confidence, 1.0) >= @min_confidence
            ORDER BY
              COALESCE(confidence, 1.0) DESC,
              COALESCE(edge_strength, 0.0) DESC,
              created_at DESC
            LIMIT @edge_limit
            """,
            {
                "tenant_id": tenant_id,
                "trading_mode": trading_mode,
                "node_ids": node_ids,
                "min_confidence": clean_confidence,
                "edge_limit": clean_edge_limit,
            },
        )

    links: list[dict[str, Any]] = []
    for row in edge_rows:
        source = str(row.get("from_node_id") or "").strip()
        target = str(row.get("to_node_id") or "").strip()
        if not source or not target:
            continue
        links.append(
            {
                "id": str(row.get("edge_id") or "").strip(),
                "source": source,
                "target": target,
                "edge_type": str(row.get("edge_type") or "").strip().upper(),
                "edge_strength": row.get("edge_strength"),
                "confidence": row.get("confidence"),
                "causal_chain_id": str(row.get("causal_chain_id") or "").strip(),
                "created_at": str(row.get("created_at") or "").strip(),
            }
        )

    available_agents = sorted({str(node.get("agent_id") or "").strip() for node in nodes if str(node.get("agent_id") or "").strip()})
    available_event_types = sorted({str(node.get("event_type") or "").strip() for node in nodes if str(node.get("event_type") or "").strip()})

    return {
        "tenant_id": tenant_id,
        "trading_mode": trading_mode,
        "nodes": nodes,
        "links": links,
        "meta": {
            "days": clean_days,
            "node_limit": clean_node_limit,
            "edge_limit": clean_edge_limit,
            "node_count": len(nodes),
            "edge_count": len(links),
            "available_agents": available_agents,
            "available_event_types": available_event_types,
            "filters": {
                "agent_id": clean_agent,
                "event_type": clean_event_type,
                "ticker": clean_ticker,
                "prompt_only": bool(prompt_only),
                "min_confidence": clean_confidence,
            },
        },
    }


def _stats_payload(repo: BigQueryRepository, *, tenant_id: str, trading_mode: str) -> dict[str, Any]:
    sql = f"""
    SELECT event_type, agent_id, COUNT(1) AS cnt, MAX(created_at) AS last_created_at
    FROM `{repo.dataset_fqn}.agent_memory_events`
    WHERE tenant_id = @tenant_id
      AND trading_mode = @trading_mode
    GROUP BY event_type, agent_id
    """
    rows = repo.fetch_rows(sql, {"tenant_id": tenant_id, "trading_mode": trading_mode})
    counts_by_event_type: dict[str, int] = {}
    counts_by_agent: dict[str, int] = {}
    counts_by_agent_event_type: dict[str, dict[str, int]] = {}
    last_created_at = ""
    total_events = 0
    for row in rows:
        event_type = str(row.get("event_type") or "").strip().lower()
        agent_id = str(row.get("agent_id") or "").strip().lower()
        count = int(row.get("cnt") or 0)
        if not event_type or count <= 0:
            continue
        total_events += count
        counts_by_event_type[event_type] = counts_by_event_type.get(event_type, 0) + count
        if agent_id:
            counts_by_agent[agent_id] = counts_by_agent.get(agent_id, 0) + count
            bucket = counts_by_agent_event_type.setdefault(agent_id, {})
            bucket[event_type] = bucket.get(event_type, 0) + count
        created_at = str(row.get("last_created_at") or "").strip()
        if created_at and created_at > last_created_at:
            last_created_at = created_at
    coverage_rows = repo.fetch_rows(
        f"""
        SELECT
          COUNT(1) AS total_memory_events,
          COUNTIF(TRIM(COALESCE(graph_node_id, '')) != '') AS with_graph_node_id,
          COUNTIF(TRIM(COALESCE(causal_chain_id, '')) != '') AS with_causal_chain_id,
          COUNTIF(last_accessed_at IS NOT NULL) AS with_last_accessed_at,
          COUNTIF(effective_score IS NOT NULL) AS with_effective_score,
          MAX(last_accessed_at) AS last_accessed_at
        FROM `{repo.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    coverage_row = coverage_rows[0] if coverage_rows else {}
    total_memory_events = int(coverage_row.get("total_memory_events") or total_events or 0)
    with_graph_node_id = int(coverage_row.get("with_graph_node_id") or 0)
    with_causal_chain_id = int(coverage_row.get("with_causal_chain_id") or 0)
    with_last_accessed_at = int(coverage_row.get("with_last_accessed_at") or 0)
    with_effective_score = int(coverage_row.get("with_effective_score") or 0)

    tier_rows = repo.fetch_rows(
        f"""
        SELECT memory_tier, COUNT(1) AS cnt
        FROM `{repo.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        GROUP BY memory_tier
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    counts_by_memory_tier: dict[str, int] = {}
    for row in tier_rows:
        memory_tier = str(row.get("memory_tier") or "").strip().lower()
        count = int(row.get("cnt") or 0)
        if memory_tier and count > 0:
            counts_by_memory_tier[memory_tier] = count

    access_rows = repo.fetch_rows(
        f"""
        SELECT
          COUNT(1) AS access_event_count,
          COUNTIF(COALESCE(used_in_prompt, FALSE)) AS prompt_use_count,
          MAX(accessed_at) AS last_accessed_at
        FROM `{repo.dataset_fqn}.memory_access_events`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    access_row = access_rows[0] if access_rows else {}

    graph_node_rows = repo.fetch_rows(
        f"""
        SELECT node_kind, COUNT(1) AS cnt, MAX(created_at) AS last_created_at
        FROM `{repo.dataset_fqn}.memory_graph_nodes`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        GROUP BY node_kind
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    counts_by_node_kind: dict[str, int] = {}
    last_graph_node_at = ""
    total_graph_nodes = 0
    for row in graph_node_rows:
        node_kind = str(row.get("node_kind") or "").strip().lower()
        count = int(row.get("cnt") or 0)
        if not node_kind or count <= 0:
            continue
        total_graph_nodes += count
        counts_by_node_kind[node_kind] = count
        created_at = str(row.get("last_created_at") or "").strip()
        if created_at and created_at > last_graph_node_at:
            last_graph_node_at = created_at

    graph_edge_rows = repo.fetch_rows(
        f"""
        SELECT edge_type, COUNT(1) AS cnt, MAX(created_at) AS last_created_at
        FROM `{repo.dataset_fqn}.memory_graph_edges`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        GROUP BY edge_type
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    counts_by_edge_type: dict[str, int] = {}
    last_graph_edge_at = ""
    total_graph_edges = 0
    for row in graph_edge_rows:
        edge_type = str(row.get("edge_type") or "").strip().upper()
        count = int(row.get("cnt") or 0)
        if not edge_type or count <= 0:
            continue
        total_graph_edges += count
        counts_by_edge_type[edge_type] = count
        created_at = str(row.get("last_created_at") or "").strip()
        if created_at and created_at > last_graph_edge_at:
            last_graph_edge_at = created_at

    return {
        "tenant_id": tenant_id,
        "total_events": total_events,
        "counts_by_event_type": counts_by_event_type,
        "counts_by_memory_tier": counts_by_memory_tier,
        "counts_by_agent": counts_by_agent,
        "counts_by_agent_event_type": counts_by_agent_event_type,
        "last_created_at": last_created_at,
        "memory_runtime": {
            "total_memory_events": total_memory_events,
            "with_graph_node_id": with_graph_node_id,
            "with_causal_chain_id": with_causal_chain_id,
            "with_last_accessed_at": with_last_accessed_at,
            "with_effective_score": with_effective_score,
            "graph_node_coverage": _ratio(with_graph_node_id, total_memory_events),
            "causal_chain_coverage": _ratio(with_causal_chain_id, total_memory_events),
            "last_accessed_at": str(coverage_row.get("last_accessed_at") or "").strip(),
        },
        "access_runtime": {
            "access_event_count": int(access_row.get("access_event_count") or 0),
            "prompt_use_count": int(access_row.get("prompt_use_count") or 0),
            "last_accessed_at": str(access_row.get("last_accessed_at") or "").strip(),
        },
        "graph_runtime": {
            "total_nodes": total_graph_nodes,
            "total_edges": total_graph_edges,
            "counts_by_node_kind": counts_by_node_kind,
            "counts_by_edge_type": counts_by_edge_type,
            "last_node_created_at": last_graph_node_at,
            "last_edge_created_at": last_graph_edge_at,
        },
    }


def _memory_defaults(settings: Settings) -> dict[str, Any]:
    defaults = default_memory_policy(
        context_limit=settings.context_max_memory_events,
        compaction_enabled=settings.memory_compaction_enabled,
        cycle_event_limit=settings.memory_compaction_cycle_event_limit,
        recent_lessons_limit=settings.memory_compaction_recent_lessons_limit,
        max_reflections=settings.memory_compaction_max_reflections,
    )
    current = getattr(settings, "memory_policy", None)
    if isinstance(current, dict) and current:
        return normalize_memory_policy(current, defaults=defaults)
    return defaults


def _graph_payload(
    repo: BigQueryRepository,
    settings: Settings,
    *,
    tenant_id: str,
    cached_fetch: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    tenant = str(tenant_id or "").strip().lower() or "local"
    defaults = _memory_defaults(settings)
    policy = load_memory_policy(repo, tenant, defaults=defaults)
    tenant_prompt = load_tenant_compaction_prompt(repo, tenant)
    global_prompt = load_global_compaction_prompt(repo)
    effective_prompt = tenant_prompt or global_prompt
    prompt_source = "tenant" if tenant_prompt else "global"
    stats_key = f"memory_stats:{tenant}:{settings.trading_mode}"
    if callable(cached_fetch):
        stats = cached_fetch(stats_key, _stats_payload, repo, tenant_id=tenant, trading_mode=settings.trading_mode)
    else:
        stats = _stats_payload(repo, tenant_id=tenant, trading_mode=settings.trading_mode)
    tuning_state, tuning_state_invalid = _load_json_config_with_state(
        repo,
        tenant_id=tenant,
        config_key=MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY,
    )
    graph = build_memory_graph(
        policy,
        tenant_id=tenant,
        stats=stats,
        tenant_compaction_prompt=tenant_prompt,
        global_compaction_prompt=global_prompt,
        effective_compaction_prompt=effective_prompt,
        prompt_source=prompt_source,
        compaction_prompt_editable=True,
    )
    meta = graph.setdefault("meta", {})
    meta["stats"] = stats
    meta["runtime"] = {
        "trading_mode": str(settings.trading_mode or "").strip().lower() or "paper",
        "memory": stats.get("memory_runtime") or {},
        "access": stats.get("access_runtime") or {},
        "graph": stats.get("graph_runtime") or {},
        "event_types": stats.get("counts_by_event_type") or {},
        "memory_tiers": stats.get("counts_by_memory_tier") or {},
        "forgetting_tuning_state": tuning_state,
        "invalid_config_keys": [MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY] if tuning_state_invalid else [],
    }
    return graph


def build_memory_settings_panel(
    repo: BigQueryRepository,
    settings: Settings,
    *,
    tenant_id: str,
    cached_fetch: Callable[..., Any] | None = None,
) -> str:
    graph = _graph_payload(repo, settings, tenant_id=tenant_id, cached_fetch=cached_fetch)
    return '<div data-settings-panel="memory" class="settings-panel hidden">' + _memory_page_html(graph) + "</div>"


def _memory_page_html(graph_payload: dict[str, Any]) -> str:
    initial_json = json.dumps(graph_payload, ensure_ascii=False).replace("<", "\\u003c")
    initial_node = next(
        (node for node in (graph_payload.get("nodes") or []) if str(node.get("id") or "") == "root"),
        ((graph_payload.get("nodes") or [None])[0] if (graph_payload.get("nodes") or []) else None),
    )
    initial_title = "메모리 동작 개요" if str((initial_node or {}).get("id") or "") == "root" else html.escape(str((initial_node or {}).get("label") or "Memory System"))
    return render_ui_template(
        "memory_panel.jinja2",
        graph_payload_json=initial_json,
        initial_title=initial_title,
    )



def register_memory_routes(
    app: FastAPI,
    *,
    repo: BigQueryRepository,
    settings: Settings,
    settings_enabled: bool,
    resolve_admin_context: Callable[..., Any],
    cached_fetch: Callable[..., Any] | None = None,
    invalidate_tenant_cache: Callable[..., Any] | None = None,
) -> None:
    @app.get("/assets/vendor/three.min.js")
    def memory_vendor_three() -> FileResponse:
        return FileResponse(_THREE_JS_PATH, media_type="application/javascript")

    @app.get("/assets/vendor/3d-force-graph.min.js")
    def memory_vendor_force_graph() -> FileResponse:
        return FileResponse(_FORCE_GRAPH_JS_PATH, media_type="application/javascript")

    @app.get("/admin/memory", response_class=HTMLResponse)
    def admin_memory_page(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> RedirectResponse:
        if not settings_enabled:
            return HTMLResponse("settings disabled", status_code=403)
        return RedirectResponse(url=f"/settings?tenant_id={tenant_id}&tab=memory", status_code=302)

    @app.get("/api/memory/graph")
    def api_memory_graph(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, _user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/graph?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        return json_response(_graph_payload(repo, settings, tenant_id=tenant, cached_fetch=cached_fetch), max_age=0)

    @app.get("/api/memory/config")
    def api_memory_config(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, _user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/config?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        graph = _graph_payload(repo, settings, tenant_id=tenant, cached_fetch=cached_fetch)
        meta = graph.get("meta") or {}
        return json_response(
            {
                "tenant_id": meta.get("tenant_id"),
                "policy": meta.get("policy") or {},
                "compaction_prompt": meta.get("effective_compaction_prompt") or "",
                "tenant_compaction_prompt": meta.get("tenant_compaction_prompt") or "",
                "global_compaction_prompt": meta.get("global_compaction_prompt") or "",
                "effective_compaction_prompt": meta.get("effective_compaction_prompt") or "",
                "prompt_source": meta.get("prompt_source") or "global",
                "compaction_prompt_editable": bool(meta.get("compaction_prompt_editable", True)),
                "runtime": meta.get("runtime") or {},
            },
            max_age=0,
        )

    @app.get("/api/memory/stats")
    def api_memory_stats(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, _user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/stats?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        stats_key = f"memory_stats:{tenant}:{settings.trading_mode}"
        if callable(cached_fetch):
            stats = cached_fetch(stats_key, _stats_payload, repo, tenant_id=tenant, trading_mode=settings.trading_mode)
        else:
            stats = _stats_payload(repo, tenant_id=tenant, trading_mode=settings.trading_mode)
        return json_response(stats, max_age=0)

    @app.get("/api/memory/activity")
    def api_memory_activity(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, _user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/activity?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        return json_response(
            _activity_payload(repo, tenant_id=tenant, trading_mode=settings.trading_mode),
            max_age=0,
        )

    @app.get("/api/memory/network")
    def api_memory_network(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
        days: int = Query(default=30, ge=1, le=365),
        node_limit: int = Query(default=60, ge=8, le=160),
        edge_limit: int = Query(default=160, ge=16, le=320),
        agent_id: str = Query(default="", description="agent filter"),
        event_type: str = Query(default="", description="event type filter"),
        ticker: str = Query(default="", description="ticker filter"),
        prompt_only: bool = Query(default=False, description="prompt-used memories only"),
        min_confidence: float = Query(default=0.55, ge=0.0, le=1.0),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, _user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/network?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        return json_response(
            _network_payload(
                repo,
                tenant_id=tenant,
                trading_mode=settings.trading_mode,
                days=days,
                node_limit=node_limit,
                edge_limit=edge_limit,
                agent_id=agent_id,
                event_type=event_type,
                ticker=ticker,
                prompt_only=prompt_only,
                min_confidence=min_confidence,
            ),
            max_age=0,
        )

    @app.post("/api/memory/cleanup")
    def api_memory_cleanup(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/cleanup?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        try:
            result = run_memory_cleanup(
                repo,
                settings,
                tenant_id=tenant,
                require_enabled=True,
            )
            if not bool(result.get("enabled")):
                return JSONResponse({"error": "cleanup disabled", **result}, status_code=400)
            try:
                repo.append_runtime_audit_log(
                    action="memory_cleanup_run",
                    status="ok",
                    user_email=user_email or "ui-admin",
                    tenant_id=tenant,
                    detail={
                        "candidate_count": result.get("candidate_count"),
                        "deleted_bigquery": result.get("deleted_bigquery"),
                        "deleted_firestore": result.get("deleted_firestore"),
                        "firestore_error": result.get("firestore_error") or "",
                    },
                )
            except Exception:
                pass
            if callable(invalidate_tenant_cache):
                invalidate_tenant_cache(tenant, "memory")
            return json_response(result, max_age=0)
        except Exception as exc:
            try:
                repo.append_runtime_audit_log(
                    action="memory_cleanup_run",
                    status="error",
                    user_email=user_email or "ui-admin",
                    tenant_id=tenant,
                    detail={"error": str(exc)},
                )
            except Exception:
                pass
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/memory/config")
    async def api_memory_config_save(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/config?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)

        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            return JSONResponse({"error": "invalid json body"}, status_code=400)

        defaults = _memory_defaults(settings)
        policy = normalize_memory_policy(payload.get("policy"), defaults=defaults)
        try:
            repo.set_config(
                tenant,
                MEMORY_POLICY_CONFIG_KEY,
                serialize_memory_policy(policy),
                updated_by=user_email or "ui-admin",
            )
            if "compaction_prompt" in payload or "global_compaction_prompt" in payload:
                prompt_value = payload.get("compaction_prompt", payload.get("global_compaction_prompt"))
                compaction_prompt = str(prompt_value or "").strip()
                if not compaction_prompt:
                    return JSONResponse({"error": f"{GLOBAL_MEMORY_PROMPT_CONFIG_KEY} cannot be empty"}, status_code=400)
                repo.set_config(
                    tenant,
                    GLOBAL_MEMORY_PROMPT_CONFIG_KEY,
                    compaction_prompt,
                    updated_by=user_email or "ui-admin",
                )
            try:
                repo.append_runtime_audit_log(
                    action="memory_policy_save",
                    status="ok",
                    user_email=user_email or "ui-admin",
                    tenant_id=tenant,
                    detail={"config_key": MEMORY_POLICY_CONFIG_KEY},
                )
            except Exception:
                pass
        except Exception as exc:
            try:
                repo.append_runtime_audit_log(
                    action="memory_policy_save",
                    status="error",
                    user_email=user_email or "ui-admin",
                    tenant_id=tenant,
                    detail={"error": str(exc)},
                )
            except Exception:
                pass
            return JSONResponse({"error": str(exc)}, status_code=500)

        if callable(invalidate_tenant_cache):
            invalidate_tenant_cache(tenant, "runtime", "memory")
        return json_response(_graph_payload(repo, settings, tenant_id=tenant, cached_fetch=cached_fetch), max_age=0)
