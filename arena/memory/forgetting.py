from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from arena.models import utc_now

ACCESS_CURVE_TOKENS = ("sqrt", "log", "capped_linear")
DEFAULT_ACCESS_CURVE = "sqrt"
DEFAULT_WORKING_TIER_WEIGHT = 2.0
DEFAULT_EPISODIC_TIER_WEIGHT = 1.0
DEFAULT_SEMANTIC_TIER_WEIGHT = 0.35
DEFAULT_CAPPED_LINEAR_STEP = 0.25
DEFAULT_CAPPED_LINEAR_CAP = 4.0


def base_memory_score(row: dict[str, Any]) -> float:
    for key in ("outcome_score", "importance_score", "score"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            continue
    return 0.0


def normalize_access_curve(access_curve: str | None) -> str:
    token = str(access_curve or "").strip().lower()
    return token if token in ACCESS_CURVE_TOKENS else DEFAULT_ACCESS_CURVE


def tier_decay_weight(
    memory_tier: str | None,
    *,
    working_weight: float = DEFAULT_WORKING_TIER_WEIGHT,
    episodic_weight: float = DEFAULT_EPISODIC_TIER_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_TIER_WEIGHT,
) -> float:
    token = str(memory_tier or "").strip().lower()
    if token == "working":
        return max(0.1, float(working_weight))
    if token == "semantic":
        return max(0.05, float(semantic_weight))
    return max(0.1, float(episodic_weight))


def access_boost(
    access_count: int,
    *,
    access_curve: str | None = DEFAULT_ACCESS_CURVE,
    capped_linear_step: float = DEFAULT_CAPPED_LINEAR_STEP,
    capped_linear_cap: float = DEFAULT_CAPPED_LINEAR_CAP,
) -> float:
    safe_access_count = max(0, int(access_count))
    curve = normalize_access_curve(access_curve)
    if curve == "log":
        return max(1.0, math.log1p(float(safe_access_count)) + 1.0)
    if curve == "capped_linear":
        step = max(0.01, float(capped_linear_step))
        cap = max(1.0, float(capped_linear_cap))
        return max(1.0, min(1.0 + (float(safe_access_count) * step), cap))
    return max(1.0, (float(safe_access_count) + 1.0) ** 0.5)


def staleness_days(
    *,
    created_at: datetime | None,
    last_accessed_at: datetime | None,
    now: datetime | None = None,
) -> int:
    reference = last_accessed_at or created_at
    if not isinstance(reference, datetime):
        return 0
    current = now or utc_now()
    return max(0, int((current - reference).days))


def decay_multiplier(
    *,
    memory_tier: str | None,
    age_days: int,
    access_count: int,
    default_decay_factor: float,
    min_decay_multiplier: float,
    access_curve: str | None = DEFAULT_ACCESS_CURVE,
    working_weight: float = DEFAULT_WORKING_TIER_WEIGHT,
    episodic_weight: float = DEFAULT_EPISODIC_TIER_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_TIER_WEIGHT,
) -> float:
    safe_age = max(0, int(age_days))
    safe_access_count = max(0, int(access_count))
    base = max(0.90, min(float(default_decay_factor), 1.0))
    floor = max(0.0, min(float(min_decay_multiplier), 1.0))
    exponent = (
        float(safe_age)
        * tier_decay_weight(
            memory_tier,
            working_weight=working_weight,
            episodic_weight=episodic_weight,
            semantic_weight=semantic_weight,
        )
    ) / access_boost(safe_access_count, access_curve=access_curve)
    return max(pow(base, exponent), floor)


def effective_memory_score(
    row: dict[str, Any],
    *,
    default_decay_factor: float,
    min_decay_multiplier: float,
    access_curve: str | None = DEFAULT_ACCESS_CURVE,
    working_weight: float = DEFAULT_WORKING_TIER_WEIGHT,
    episodic_weight: float = DEFAULT_EPISODIC_TIER_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_TIER_WEIGHT,
    now: datetime | None = None,
) -> tuple[float, float]:
    created_at = row.get("created_at") if isinstance(row.get("created_at"), datetime) else None
    last_accessed_at = row.get("last_accessed_at") if isinstance(row.get("last_accessed_at"), datetime) else None
    age_days = staleness_days(created_at=created_at, last_accessed_at=last_accessed_at, now=now)
    try:
        access_count = int(row.get("access_count") or 0)
    except (TypeError, ValueError):
        access_count = 0
    decay_score = decay_multiplier(
        memory_tier=str(row.get("memory_tier") or ""),
        age_days=age_days,
        access_count=access_count,
        default_decay_factor=default_decay_factor,
        min_decay_multiplier=min_decay_multiplier,
        access_curve=access_curve,
        working_weight=working_weight,
        episodic_weight=episodic_weight,
        semantic_weight=semantic_weight,
    )
    effective_score = max(0.0, min(base_memory_score(row) * decay_score, 1.0))
    return decay_score, effective_score


def recompute_memory_forgetting_fields(
    repo: Any,
    *,
    tenant_id: str,
    trading_mode: str,
    default_decay_factor: float,
    min_decay_multiplier: float,
    access_curve: str = DEFAULT_ACCESS_CURVE,
    working_weight: float = DEFAULT_WORKING_TIER_WEIGHT,
    episodic_weight: float = DEFAULT_EPISODIC_TIER_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_TIER_WEIGHT,
) -> None:
    tenant = str(tenant_id or "").strip().lower() or "local"
    mode = str(trading_mode or "").strip().lower() or "paper"
    curve = normalize_access_curve(access_curve)
    access_expr = "SQRT(CAST(access_count AS FLOAT64) + 1.0)"
    if curve == "log":
        access_expr = "GREATEST(1.0, LN(CAST(access_count AS FLOAT64) + 1.0) + 1.0)"
    elif curve == "capped_linear":
        access_expr = f"LEAST(1.0 + (CAST(access_count AS FLOAT64) * {DEFAULT_CAPPED_LINEAR_STEP}), {DEFAULT_CAPPED_LINEAR_CAP})"
    sql = f"""
    UPDATE `{repo.dataset_fqn}.agent_memory_events` AS mem
    SET
      access_count = src.access_count,
      last_accessed_at = src.last_accessed_at,
      decay_score = src.decay_score,
      effective_score = src.effective_score
    FROM (
      WITH access_stats AS (
        SELECT
          event_id,
          COUNT(1) AS access_count,
          MAX(accessed_at) AS last_accessed_at
        FROM `{repo.dataset_fqn}.memory_access_events`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        GROUP BY event_id
      ),
      row_stats AS (
        SELECT
          src_mem.event_id,
          COALESCE(acc.access_count, 0) AS access_count,
          acc.last_accessed_at AS last_accessed_at,
          COALESCE(src_mem.outcome_score, src_mem.importance_score, src_mem.score, 0.0) AS base_score,
          SAFE_CAST(
            DATE_DIFF(CURRENT_DATE(), DATE(COALESCE(acc.last_accessed_at, src_mem.created_at)), DAY) AS FLOAT64
          ) AS staleness_days,
          CASE
            WHEN LOWER(COALESCE(src_mem.memory_tier, '')) = 'working' THEN @working_weight
            WHEN LOWER(COALESCE(src_mem.memory_tier, '')) = 'semantic' THEN @semantic_weight
            ELSE @episodic_weight
          END AS tier_weight
        FROM `{repo.dataset_fqn}.agent_memory_events` AS src_mem
        LEFT JOIN access_stats AS acc
          ON src_mem.event_id = acc.event_id
        WHERE src_mem.tenant_id = @tenant_id
          AND src_mem.trading_mode = @trading_mode
      ),
      computed AS (
        SELECT
          event_id,
          access_count,
          last_accessed_at,
          GREATEST(
            POW(
              @default_decay_factor,
              staleness_days * tier_weight / {access_expr}
            ),
            @min_decay_multiplier
          ) AS decay_score,
          LEAST(
            1.0,
            GREATEST(
              0.0,
              base_score * GREATEST(
                POW(
                  @default_decay_factor,
                  staleness_days * tier_weight / {access_expr}
                ),
                @min_decay_multiplier
              )
            )
          ) AS effective_score
        FROM row_stats
      )
      SELECT event_id, access_count, last_accessed_at, decay_score, effective_score
      FROM computed
    ) AS src
    WHERE mem.tenant_id = @tenant_id
      AND mem.trading_mode = @trading_mode
      AND mem.event_id = src.event_id
    """
    repo.execute(
        sql,
        {
            "tenant_id": tenant,
            "trading_mode": mode,
            "default_decay_factor": max(0.90, min(float(default_decay_factor), 1.0)),
            "min_decay_multiplier": max(0.0, min(float(min_decay_multiplier), 1.0)),
            "working_weight": max(0.1, float(working_weight)),
            "episodic_weight": max(0.1, float(episodic_weight)),
            "semantic_weight": max(0.05, float(semantic_weight)),
        },
    )
