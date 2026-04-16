from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from arena.memory.forgetting import recompute_memory_forgetting_fields
from arena.memory.policy import (
    default_memory_policy,
    memory_forgetting_access_curve,
    load_memory_policy,
    memory_forgetting_default_decay_factor,
    memory_forgetting_enabled,
    memory_forgetting_min_effective_score,
    memory_forgetting_tier_weight,
    memory_hierarchy_enabled,
    normalize_memory_policy,
)
from arena.models import utc_now

logger = logging.getLogger(__name__)


def _policy_defaults(settings: Any) -> dict[str, Any]:
    defaults = default_memory_policy(
        context_limit=getattr(settings, "context_max_memory_events", 32),
        compaction_enabled=getattr(settings, "memory_compaction_enabled", True),
        cycle_event_limit=getattr(settings, "memory_compaction_cycle_event_limit", 12),
        recent_lessons_limit=getattr(settings, "memory_compaction_recent_lessons_limit", 4),
        max_reflections=getattr(settings, "memory_compaction_max_reflections", 3),
    )
    current = getattr(settings, "memory_policy", None)
    if isinstance(current, dict) and current:
        return normalize_memory_policy(current, defaults=defaults)
    return defaults


def cleanup_candidates(
    repo: Any,
    *,
    tenant_id: str,
    trading_mode: str,
    max_age_days: int,
    min_score: float,
    limit: int = 500,
    use_effective_score: bool = False,
    use_last_accessed: bool = False,
    preserve_semantic: bool = False,
) -> list[dict[str, Any]]:
    tenant = str(tenant_id or "").strip().lower() or "local"
    cutoff_ts = utc_now() - timedelta(days=max(1, int(max_age_days)))
    stale_expr = "COALESCE(last_accessed_at, created_at)" if use_last_accessed else "created_at"
    score_expr = (
        "COALESCE(effective_score, outcome_score, importance_score, score, 0.0)"
        if use_effective_score
        else "COALESCE(outcome_score, score, importance_score, 0.0)"
    )
    semantic_filter = "AND LOWER(COALESCE(memory_tier, '')) != 'semantic'" if preserve_semantic else ""
    sql = f"""
    SELECT
      event_id,
      agent_id,
      event_type,
      created_at,
      last_accessed_at,
      score,
      importance_score,
      outcome_score,
      {score_expr} AS effective_score,
      summary
    FROM `{repo.dataset_fqn}.agent_memory_events`
    WHERE tenant_id = @tenant_id
      AND trading_mode = @trading_mode
      AND {stale_expr} < @cutoff_ts
      AND {score_expr} < @min_score
      {semantic_filter}
    ORDER BY created_at ASC
    LIMIT @limit
    """
    return repo.fetch_rows(
        sql,
        {
            "tenant_id": tenant,
            "trading_mode": str(trading_mode or "").strip().lower() or "paper",
            "cutoff_ts": cutoff_ts,
            "min_score": float(min_score),
            "limit": max(1, min(int(limit), 5000)),
        },
    )


def recompute_forgetting_scores(
    repo: Any,
    settings: Any,
    *,
    tenant_id: str,
    policy: dict[str, Any] | None = None,
) -> bool:
    defaults = _policy_defaults(settings)
    active_policy = normalize_memory_policy(
        policy if isinstance(policy, dict) else load_memory_policy(repo, tenant_id, defaults=defaults),
        defaults=defaults,
    )
    if not memory_forgetting_enabled(active_policy):
        return False
    recompute_memory_forgetting_fields(
        repo,
        tenant_id=tenant_id,
        trading_mode=getattr(settings, "trading_mode", "paper").strip().lower(),
        default_decay_factor=memory_forgetting_default_decay_factor(active_policy),
        min_decay_multiplier=memory_forgetting_min_effective_score(active_policy),
        access_curve=memory_forgetting_access_curve(active_policy),
        working_weight=memory_forgetting_tier_weight(active_policy, "working"),
        episodic_weight=memory_forgetting_tier_weight(active_policy, "episodic"),
        semantic_weight=memory_forgetting_tier_weight(active_policy, "semantic"),
    )
    return True


def delete_cleanup_candidates(repo: Any, *, tenant_id: str, event_ids: list[str]) -> int:
    clean_ids = [str(event_id or "").strip() for event_id in event_ids if str(event_id or "").strip()]
    if not clean_ids:
        return 0
    tenant = str(tenant_id or "").strip().lower() or "local"
    sql = f"""
    DELETE FROM `{repo.dataset_fqn}.agent_memory_events`
    WHERE tenant_id = @tenant_id
      AND event_id IN UNNEST(@event_ids)
    """
    repo.execute(sql, {"tenant_id": tenant, "event_ids": clean_ids})
    return len(clean_ids)


def delete_firestore_vectors(*, project: str, event_ids: list[str]) -> tuple[int, str]:
    clean_ids = [str(event_id or "").strip() for event_id in event_ids if str(event_id or "").strip()]
    if not clean_ids:
        return 0, ""
    try:
        from google.cloud import firestore as _firestore
    except Exception as exc:
        return 0, str(exc)

    try:
        db = _firestore.Client(project=project)
    except Exception as exc:
        return 0, str(exc)

    deleted = 0
    try:
        for start in range(0, len(clean_ids), 250):
            chunk = clean_ids[start:start + 250]
            batch = db.batch()
            for event_id in chunk:
                batch.delete(db.collection("agent_memories").document(event_id))
            batch.commit()
            deleted += len(chunk)
    except Exception as exc:
        return deleted, str(exc)
    return deleted, ""


def run_memory_cleanup(
    repo: Any,
    settings: Any,
    *,
    tenant_id: str,
    policy: dict[str, Any] | None = None,
    limit: int = 500,
    dry_run: bool = False,
    require_enabled: bool = True,
) -> dict[str, Any]:
    tenant = str(tenant_id or "").strip().lower() or "local"
    defaults = _policy_defaults(settings)
    active_policy = normalize_memory_policy(
        policy if isinstance(policy, dict) else load_memory_policy(repo, tenant, defaults=defaults),
        defaults=defaults,
    )
    cleanup_policy = dict(active_policy.get("cleanup") or {})
    enabled = bool(cleanup_policy.get("enabled"))
    forgetting_enabled = memory_forgetting_enabled(active_policy)
    preserve_semantic = memory_hierarchy_enabled(active_policy)
    max_age_days = max(1, int(cleanup_policy.get("max_age_days") or 180))
    min_score = max(0.0, min(float(cleanup_policy.get("min_score") or 0.3), 1.0))
    cutoff_ts = utc_now() - timedelta(days=max_age_days)
    summary = {
        "tenant_id": tenant,
        "enabled": enabled,
        "forgetting_enabled": forgetting_enabled,
        "trading_mode": getattr(settings, "trading_mode", "paper").strip().lower(),
        "max_age_days": max_age_days,
        "min_score": min_score,
        "cutoff_ts": cutoff_ts.isoformat(),
        "candidate_count": 0,
        "deleted_bigquery": 0,
        "deleted_firestore": 0,
        "limit": max(1, min(int(limit), 5000)),
        "dry_run": bool(dry_run),
        "firestore_error": "",
        "reason": "",
        "preview": [],
    }
    if require_enabled and not enabled:
        summary["reason"] = "cleanup disabled"
        return summary

    if forgetting_enabled:
        recompute_forgetting_scores(repo, settings, tenant_id=tenant, policy=active_policy)

    candidates = cleanup_candidates(
        repo,
        tenant_id=tenant,
        trading_mode=summary["trading_mode"],
        max_age_days=max_age_days,
        min_score=min_score,
        limit=summary["limit"],
        use_effective_score=forgetting_enabled,
        use_last_accessed=forgetting_enabled,
        preserve_semantic=preserve_semantic,
    )
    summary["candidate_count"] = len(candidates)
    summary["preview"] = [
        {
            "event_id": str(row.get("event_id") or ""),
            "agent_id": str(row.get("agent_id") or ""),
            "event_type": str(row.get("event_type") or ""),
            "created_at": str(row.get("created_at") or ""),
            "effective_score": float(row.get("effective_score") or 0.0),
            "summary": str(row.get("summary") or "")[:160],
        }
        for row in candidates[:8]
    ]
    if dry_run or not candidates:
        return summary

    event_ids = [str(row.get("event_id") or "").strip() for row in candidates if str(row.get("event_id") or "").strip()]
    summary["deleted_bigquery"] = delete_cleanup_candidates(repo, tenant_id=tenant, event_ids=event_ids)
    deleted_firestore, firestore_error = delete_firestore_vectors(
        project=str(getattr(settings, "google_cloud_project", "") or "").strip(),
        event_ids=event_ids,
    )
    summary["deleted_firestore"] = deleted_firestore
    summary["firestore_error"] = firestore_error
    return summary


def run_memory_cleanup_for_all_tenants(
    repo: Any,
    settings: Any,
    *,
    tenant_ids: list[str] | None = None,
    limit_per_tenant: int = 500,
    require_enabled: bool = True,
) -> dict[str, Any]:
    resolved = [str(token or "").strip().lower() for token in (tenant_ids or []) if str(token or "").strip()]
    if not resolved:
        lister = getattr(repo, "list_runtime_tenants", None)
        if callable(lister):
            try:
                resolved = [str(token or "").strip().lower() for token in lister(limit=500) if str(token or "").strip()]
            except Exception as exc:
                logger.warning("[yellow]runtime tenant discovery failed[/yellow] err=%s", str(exc))
                resolved = []
    fallback_tenant = str(getattr(repo, "tenant_id", None) or "local").strip().lower() or "local"
    if fallback_tenant not in resolved:
        resolved.insert(0, fallback_tenant)

    totals = {
        "tenants": [],
        "total_candidates": 0,
        "total_deleted_bigquery": 0,
        "total_deleted_firestore": 0,
    }
    seen: set[str] = set()
    for tenant in resolved:
        if tenant in seen:
            continue
        seen.add(tenant)
        result = run_memory_cleanup(
            repo,
            settings,
            tenant_id=tenant,
            limit=limit_per_tenant,
            require_enabled=require_enabled,
        )
        totals["tenants"].append(result)
        totals["total_candidates"] += int(result.get("candidate_count") or 0)
        totals["total_deleted_bigquery"] += int(result.get("deleted_bigquery") or 0)
        totals["total_deleted_firestore"] += int(result.get("deleted_firestore") or 0)
    return totals
