from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from statistics import mean
from typing import Any

from arena.memory.forgetting import base_memory_score, effective_memory_score
from arena.memory.policy import (
    MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY,
    MEMORY_POLICY_CONFIG_KEY,
    default_memory_policy,
    load_memory_policy,
    memory_forgetting_access_curve,
    memory_forgetting_default_decay_factor,
    memory_forgetting_enabled,
    memory_forgetting_min_effective_score,
    memory_forgetting_tier_weight,
    memory_forgetting_tuning_ema_alpha,
    memory_forgetting_tuning_auto_demote_enabled,
    memory_forgetting_tuning_auto_demote_unhealthy_runs,
    memory_forgetting_tuning_auto_promote_enabled,
    memory_forgetting_tuning_auto_promote_max_recommendation_drift,
    memory_forgetting_tuning_auto_promote_min_shadow_days,
    memory_forgetting_tuning_auto_promote_min_shadow_runs,
    memory_forgetting_tuning_auto_promote_required_improving_runs,
    memory_forgetting_tuning_auto_promote_required_stable_runs,
    memory_forgetting_tuning_enabled,
    memory_forgetting_tuning_lookback_days,
    memory_forgetting_tuning_max_decay_factor_delta,
    memory_forgetting_tuning_max_min_effective_score_delta,
    memory_forgetting_tuning_max_tier_weight_delta,
    memory_forgetting_tuning_min_access_events,
    memory_forgetting_tuning_min_prompt_uses,
    memory_forgetting_tuning_min_unique_memories,
    memory_forgetting_tuning_mode,
    memory_forgetting_tuning_objective_topk,
    memory_forgetting_tuning_stability_window_days,
    normalize_memory_policy,
    serialize_memory_policy,
)
from arena.models import utc_now


@dataclass(frozen=True, slots=True)
class ForgettingCandidate:
    default_decay_factor: float
    min_effective_score: float
    access_curve: str
    tier_weight_working: float
    tier_weight_episodic: float
    tier_weight_semantic: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "default_decay_factor": round(self.default_decay_factor, 6),
            "min_effective_score": round(self.min_effective_score, 6),
            "access_curve": self.access_curve,
            "tier_weight_working": round(self.tier_weight_working, 6),
            "tier_weight_episodic": round(self.tier_weight_episodic, 6),
            "tier_weight_semantic": round(self.tier_weight_semantic, 6),
        }


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


def forgetting_policy_snapshot(policy: dict[str, Any]) -> ForgettingCandidate:
    return ForgettingCandidate(
        default_decay_factor=memory_forgetting_default_decay_factor(policy),
        min_effective_score=memory_forgetting_min_effective_score(policy),
        access_curve=memory_forgetting_access_curve(policy),
        tier_weight_working=memory_forgetting_tier_weight(policy, "working"),
        tier_weight_episodic=memory_forgetting_tier_weight(policy, "episodic"),
        tier_weight_semantic=memory_forgetting_tier_weight(policy, "semantic"),
    )


def _load_previous_tuning_state(repo: Any, tenant_id: str) -> dict[str, Any]:
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return {}
    try:
        raw = getter(str(tenant_id or "").strip().lower() or "local", MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY)
    except Exception:
        return {}
    try:
        parsed = json.loads(str(raw or "").strip() or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_state_datetime(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _policy_with_tuning_mode(policy: dict[str, Any], mode: str) -> dict[str, Any]:
    updated = copy.deepcopy(policy)
    updated.setdefault("forgetting", {}).setdefault("tuning", {})["mode"] = str(mode or "shadow").strip().lower() or "shadow"
    return normalize_memory_policy(updated, defaults=normalize_memory_policy(policy))


def _recommendation_drift(previous: dict[str, Any] | None, current: dict[str, Any] | None) -> float:
    if not isinstance(previous, dict) or not isinstance(current, dict):
        return 1.0
    parts = [
        min(abs(_safe_float(previous.get("default_decay_factor")) - _safe_float(current.get("default_decay_factor"))) / 0.01, 1.0),
        min(abs(_safe_float(previous.get("min_effective_score")) - _safe_float(current.get("min_effective_score"))) / 0.05, 1.0),
        min(abs(_safe_float(previous.get("tier_weight_working")) - _safe_float(current.get("tier_weight_working"))) / 0.5, 1.0),
        min(abs(_safe_float(previous.get("tier_weight_episodic")) - _safe_float(current.get("tier_weight_episodic"))) / 0.5, 1.0),
        min(abs(_safe_float(previous.get("tier_weight_semantic")) - _safe_float(current.get("tier_weight_semantic"))) / 0.25, 1.0),
        1.0 if str(previous.get("access_curve") or "").strip().lower() != str(current.get("access_curve") or "").strip().lower() else 0.0,
    ]
    return round(_mean(parts), 6)


def fetch_forgetting_tuning_rows(
    repo: Any,
    *,
    tenant_id: str,
    trading_mode: str,
    lookback_days: int,
    stability_window_days: int,
) -> list[dict[str, Any]]:
    sql = f"""
    WITH long_access AS (
      SELECT
        event_id,
        COUNT(1) AS access_count,
        COUNTIF(COALESCE(used_in_prompt, FALSE)) AS prompt_use_count,
        COUNT(DISTINCT NULLIF(TRIM(COALESCE(cycle_id, '')), '')) AS distinct_cycle_count,
        MAX(accessed_at) AS last_accessed_at
      FROM `{repo.dataset_fqn}.memory_access_events`
      WHERE tenant_id = @tenant_id
        AND trading_mode = @trading_mode
        AND accessed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
      GROUP BY event_id
    ),
    short_access AS (
      SELECT
        event_id,
        COUNT(1) AS short_access_count,
        COUNTIF(COALESCE(used_in_prompt, FALSE)) AS short_prompt_use_count,
        COUNT(DISTINCT NULLIF(TRIM(COALESCE(cycle_id, '')), '')) AS short_distinct_cycle_count,
        MAX(accessed_at) AS short_last_accessed_at
      FROM `{repo.dataset_fqn}.memory_access_events`
      WHERE tenant_id = @tenant_id
        AND trading_mode = @trading_mode
        AND accessed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @stability_window_days DAY)
      GROUP BY event_id
    )
    SELECT
      mem.event_id,
      mem.created_at,
      mem.memory_tier,
      mem.score,
      mem.importance_score,
      mem.outcome_score,
      COALESCE(long_access.access_count, 0) AS access_count,
      COALESCE(long_access.prompt_use_count, 0) AS prompt_use_count,
      COALESCE(long_access.distinct_cycle_count, 0) AS distinct_cycle_count,
      long_access.last_accessed_at AS last_accessed_at,
      COALESCE(short_access.short_access_count, 0) AS short_access_count,
      COALESCE(short_access.short_prompt_use_count, 0) AS short_prompt_use_count,
      COALESCE(short_access.short_distinct_cycle_count, 0) AS short_distinct_cycle_count,
      short_access.short_last_accessed_at AS short_last_accessed_at
    FROM `{repo.dataset_fqn}.agent_memory_events` AS mem
    LEFT JOIN long_access
      ON mem.event_id = long_access.event_id
    LEFT JOIN short_access
      ON mem.event_id = short_access.event_id
    WHERE mem.tenant_id = @tenant_id
      AND mem.trading_mode = @trading_mode
      AND (
        mem.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
        OR long_access.event_id IS NOT NULL
      )
    """
    return repo.fetch_rows(
        sql,
        {
            "tenant_id": str(tenant_id or "").strip().lower() or "local",
            "trading_mode": str(trading_mode or "").strip().lower() or "paper",
            "lookback_days": max(7, int(lookback_days)),
            "stability_window_days": max(3, int(stability_window_days)),
        },
    )


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalizer(rows: list[dict[str, Any]], access_key: str, prompt_key: str, cycle_key: str) -> dict[str, float]:
    max_prompt = max(1, max((_safe_int(row.get(prompt_key)) for row in rows), default=0))
    max_cycles = max(1, max((_safe_int(row.get(cycle_key)) for row in rows), default=0))
    max_access = max(1, max((_safe_int(row.get(access_key)) for row in rows), default=0))
    return {
        "max_prompt": float(max_prompt),
        "max_cycles": float(max_cycles),
        "max_access": float(max_access),
    }


def _quality_label(
    row: dict[str, Any],
    *,
    access_key: str,
    prompt_key: str,
    cycle_key: str,
    normalizer: dict[str, float],
) -> float:
    access_count = _safe_int(row.get(access_key))
    prompt_use_count = _safe_int(row.get(prompt_key))
    distinct_cycle_count = _safe_int(row.get(cycle_key))
    base_score = base_memory_score(row)
    prompt_norm = min(float(prompt_use_count) / max(normalizer["max_prompt"], 1.0), 1.0)
    cycle_norm = min(float(distinct_cycle_count) / max(normalizer["max_cycles"], 1.0), 1.0)
    access_norm = min(float(access_count) / max(normalizer["max_access"], 1.0), 1.0)
    prompt_use_rate = min(float(prompt_use_count) / max(float(access_count), 1.0), 1.0)
    reuse_label = (0.45 * prompt_norm) + (0.25 * cycle_norm) + (0.15 * prompt_use_rate) + (0.15 * access_norm)
    return max(0.0, min((0.70 * reuse_label) + (0.30 * base_score), 1.0))


def _window_row(
    row: dict[str, Any],
    *,
    short_window: bool,
) -> dict[str, Any]:
    if not short_window:
        return {
            "created_at": row.get("created_at"),
            "last_accessed_at": row.get("last_accessed_at"),
            "access_count": _safe_int(row.get("access_count")),
            "memory_tier": str(row.get("memory_tier") or ""),
            "score": row.get("score"),
            "importance_score": row.get("importance_score"),
            "outcome_score": row.get("outcome_score"),
        }
    return {
        "created_at": row.get("created_at"),
        "last_accessed_at": row.get("short_last_accessed_at"),
        "access_count": _safe_int(row.get("short_access_count")),
        "memory_tier": str(row.get("memory_tier") or ""),
        "score": row.get("score"),
        "importance_score": row.get("importance_score"),
        "outcome_score": row.get("outcome_score"),
    }


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _evaluate_window(
    rows: list[dict[str, Any]],
    candidate: ForgettingCandidate,
    *,
    cleanup_min_score: float,
    objective_topk: int,
    short_window: bool,
) -> dict[str, float]:
    access_key = "short_access_count" if short_window else "access_count"
    prompt_key = "short_prompt_use_count" if short_window else "prompt_use_count"
    cycle_key = "short_distinct_cycle_count" if short_window else "distinct_cycle_count"
    norm = _normalizer(rows, access_key, prompt_key, cycle_key)

    evaluated: list[tuple[float, float]] = []
    for row in rows:
        quality = _quality_label(row, access_key=access_key, prompt_key=prompt_key, cycle_key=cycle_key, normalizer=norm)
        _, effective = effective_memory_score(
            _window_row(row, short_window=short_window),
            default_decay_factor=candidate.default_decay_factor,
            min_decay_multiplier=candidate.min_effective_score,
            access_curve=candidate.access_curve,
            working_weight=candidate.tier_weight_working,
            episodic_weight=candidate.tier_weight_episodic,
            semantic_weight=candidate.tier_weight_semantic,
            now=utc_now(),
        )
        evaluated.append((quality, effective))

    if not evaluated:
        return {
            "separation_score": 0.0,
            "topk_precision": 0.0,
            "cleanup_safety": 0.0,
        }

    ranked_by_quality = sorted(evaluated, key=lambda item: item[0])
    bucket = max(1, len(ranked_by_quality) // 4)
    low_bucket = ranked_by_quality[:bucket]
    high_bucket = ranked_by_quality[-bucket:]
    separation_score = max(0.0, _mean([value for _, value in high_bucket]) - _mean([value for _, value in low_bucket]))

    topk = min(max(1, int(objective_topk)), len(evaluated))
    ranked_by_effective = sorted(evaluated, key=lambda item: item[1], reverse=True)[:topk]
    topk_precision = max(0.0, min(_mean([quality for quality, _ in ranked_by_effective]), 1.0))

    quality_mass = sum(quality for quality, _ in evaluated)
    at_risk_mass = sum(quality for quality, effective in evaluated if effective < cleanup_min_score)
    cleanup_safety = 1.0 if quality_mass <= 0.0 else max(0.0, min(1.0, 1.0 - (at_risk_mass / quality_mass)))
    return {
        "separation_score": round(separation_score, 6),
        "topk_precision": round(topk_precision, 6),
        "cleanup_safety": round(cleanup_safety, 6),
    }


def _evaluate_candidate(
    rows: list[dict[str, Any]],
    candidate: ForgettingCandidate,
    *,
    cleanup_min_score: float,
    objective_topk: int,
) -> dict[str, Any]:
    long_metrics = _evaluate_window(
        rows,
        candidate,
        cleanup_min_score=cleanup_min_score,
        objective_topk=objective_topk,
        short_window=False,
    )
    has_short_history = any(_safe_int(row.get("short_access_count")) > 0 for row in rows)
    short_metrics = long_metrics if not has_short_history else _evaluate_window(
        rows,
        candidate,
        cleanup_min_score=cleanup_min_score,
        objective_topk=objective_topk,
        short_window=True,
    )
    instability_penalty = min(
        1.0,
        abs(long_metrics["topk_precision"] - short_metrics["topk_precision"])
        + abs(long_metrics["separation_score"] - short_metrics["separation_score"]),
    )
    objective = (
        0.40 * long_metrics["separation_score"]
        + 0.30 * long_metrics["topk_precision"]
        + 0.20 * long_metrics["cleanup_safety"]
        - 0.10 * instability_penalty
    )
    return {
        "candidate": candidate.as_dict(),
        "metrics": {
            **long_metrics,
            "short_topk_precision": short_metrics["topk_precision"],
            "short_separation_score": short_metrics["separation_score"],
            "instability_penalty": round(instability_penalty, 6),
            "objective": round(objective, 6),
        },
    }


def _candidate_values(current: float, *, lower: float, upper: float, deltas: tuple[float, ...], digits: int = 6) -> list[float]:
    values = {round(max(lower, min(upper, current)), digits)}
    for delta in deltas:
        values.add(round(max(lower, min(upper, current - delta)), digits))
        values.add(round(max(lower, min(upper, current + delta)), digits))
    return sorted(values)


def build_forgetting_candidate_grid(policy: dict[str, Any]) -> list[ForgettingCandidate]:
    current = forgetting_policy_snapshot(policy)
    decay_values = sorted(
        {
            *_candidate_values(current.default_decay_factor, lower=0.90, upper=1.0, deltas=(0.002, 0.005)),
            0.985,
            0.99,
            0.992,
            0.995,
        }
    )
    min_effective_values = sorted(
        {
            *_candidate_values(current.min_effective_score, lower=0.0, upper=1.0, deltas=(0.02, 0.05)),
            0.05,
            0.10,
            0.15,
            0.20,
        }
    )
    working_values = sorted(
        {
            *_candidate_values(current.tier_weight_working, lower=0.1, upper=4.0, deltas=(0.25, 0.50)),
            1.5,
            2.0,
            2.5,
        }
    )
    episodic_values = sorted(
        {
            *_candidate_values(current.tier_weight_episodic, lower=0.1, upper=4.0, deltas=(0.10, 0.20)),
            0.8,
            1.0,
            1.2,
        }
    )
    semantic_values = sorted(
        {
            *_candidate_values(current.tier_weight_semantic, lower=0.05, upper=2.0, deltas=(0.10, 0.15)),
            0.20,
            0.35,
            0.50,
        }
    )
    curves = []
    for token in [current.access_curve, "sqrt", "log", "capped_linear"]:
        if token not in curves:
            curves.append(token)
    candidates: list[ForgettingCandidate] = []
    for decay_factor in decay_values:
        for min_effective in min_effective_values:
            for access_curve in curves:
                for working_weight in working_values:
                    for episodic_weight in episodic_values:
                        for semantic_weight in semantic_values:
                            candidates.append(
                                ForgettingCandidate(
                                    default_decay_factor=decay_factor,
                                    min_effective_score=min_effective,
                                    access_curve=access_curve,
                                    tier_weight_working=working_weight,
                                    tier_weight_episodic=episodic_weight,
                                    tier_weight_semantic=semantic_weight,
                                )
                            )
    return candidates


def _sample_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    accessed_rows = [row for row in rows if _safe_int(row.get("access_count")) > 0]
    return {
        "total_rows": len(rows),
        "unique_memories": len(accessed_rows),
        "access_events": sum(_safe_int(row.get("access_count")) for row in rows),
        "prompt_uses": sum(_safe_int(row.get("prompt_use_count")) for row in rows),
        "short_access_events": sum(_safe_int(row.get("short_access_count")) for row in rows),
        "short_prompt_uses": sum(_safe_int(row.get("short_prompt_use_count")) for row in rows),
    }


def _bounded_ema(current: float, recommended: float, *, alpha: float, max_delta: float, lower: float, upper: float) -> float:
    bounded_target = max(lower, min(upper, max(current - max_delta, min(current + max_delta, recommended))))
    return round(max(lower, min(upper, current + ((bounded_target - current) * alpha))), 6)


def _apply_recommendation(
    policy: dict[str, Any],
    recommendation: ForgettingCandidate,
    *,
    ema_alpha: float,
    max_decay_factor_delta: float,
    max_min_effective_score_delta: float,
    max_tier_weight_delta: float,
) -> dict[str, Any]:
    current = forgetting_policy_snapshot(policy)
    updated = copy.deepcopy(policy)
    forgetting = updated.setdefault("forgetting", {})
    forgetting["default_decay_factor"] = _bounded_ema(
        current.default_decay_factor,
        recommendation.default_decay_factor,
        alpha=ema_alpha,
        max_delta=max_decay_factor_delta,
        lower=0.90,
        upper=1.0,
    )
    forgetting["min_effective_score"] = _bounded_ema(
        current.min_effective_score,
        recommendation.min_effective_score,
        alpha=ema_alpha,
        max_delta=max_min_effective_score_delta,
        lower=0.0,
        upper=1.0,
    )
    forgetting["tier_weight_working"] = _bounded_ema(
        current.tier_weight_working,
        recommendation.tier_weight_working,
        alpha=ema_alpha,
        max_delta=max_tier_weight_delta,
        lower=0.1,
        upper=4.0,
    )
    forgetting["tier_weight_episodic"] = _bounded_ema(
        current.tier_weight_episodic,
        recommendation.tier_weight_episodic,
        alpha=ema_alpha,
        max_delta=max_tier_weight_delta,
        lower=0.1,
        upper=4.0,
    )
    forgetting["tier_weight_semantic"] = _bounded_ema(
        current.tier_weight_semantic,
        recommendation.tier_weight_semantic,
        alpha=ema_alpha,
        max_delta=max_tier_weight_delta,
        lower=0.05,
        upper=2.0,
    )
    # Categorical curve changes stay shadow-only until recommendations are stable enough to review manually.
    forgetting["access_curve"] = current.access_curve
    return normalize_memory_policy(updated, defaults=normalize_memory_policy(policy))


def run_memory_forgetting_tuner(
    repo: Any,
    settings: Any,
    *,
    tenant_id: str,
    policy: dict[str, Any] | None = None,
    updated_by: str | None = None,
    persist_state: bool = True,
) -> dict[str, Any]:
    tenant = str(tenant_id or "").strip().lower() or "local"
    defaults = _policy_defaults(settings)
    active_policy = normalize_memory_policy(
        policy if isinstance(policy, dict) else load_memory_policy(repo, tenant, defaults=defaults),
        defaults=defaults,
    )
    previous_state = _load_previous_tuning_state(repo, tenant)
    previous_history = previous_state.get("history") if isinstance(previous_state.get("history"), dict) else {}
    previous_effective_mode = str(previous_state.get("effective_mode") or previous_state.get("mode") or "").strip().lower()
    trading_mode = getattr(settings, "trading_mode", "paper").strip().lower()
    tuning_enabled = memory_forgetting_tuning_enabled(active_policy)
    mode = memory_forgetting_tuning_mode(active_policy)
    lookback_days = memory_forgetting_tuning_lookback_days(active_policy)
    stability_window_days = memory_forgetting_tuning_stability_window_days(active_policy)
    objective_topk = memory_forgetting_tuning_objective_topk(active_policy)
    sample_thresholds = {
        "min_access_events": memory_forgetting_tuning_min_access_events(active_policy),
        "min_prompt_uses": memory_forgetting_tuning_min_prompt_uses(active_policy),
        "min_unique_memories": memory_forgetting_tuning_min_unique_memories(active_policy),
    }
    auto_promote_enabled = memory_forgetting_tuning_auto_promote_enabled(active_policy)
    auto_promote_min_shadow_days = memory_forgetting_tuning_auto_promote_min_shadow_days(active_policy)
    auto_promote_min_shadow_runs = memory_forgetting_tuning_auto_promote_min_shadow_runs(active_policy)
    auto_promote_required_stable_runs = memory_forgetting_tuning_auto_promote_required_stable_runs(active_policy)
    auto_promote_required_improving_runs = memory_forgetting_tuning_auto_promote_required_improving_runs(active_policy)
    auto_promote_max_recommendation_drift = memory_forgetting_tuning_auto_promote_max_recommendation_drift(active_policy)
    auto_demote_enabled = memory_forgetting_tuning_auto_demote_enabled(active_policy)
    auto_demote_unhealthy_runs = memory_forgetting_tuning_auto_demote_unhealthy_runs(active_policy)
    evaluation_now = utc_now()
    current_candidate = forgetting_policy_snapshot(active_policy)

    state: dict[str, Any] = {
        "tenant_id": tenant,
        "trading_mode": trading_mode,
        "evaluated_at": evaluation_now.isoformat(),
        "enabled": tuning_enabled,
        "mode": mode,
        "configured_mode": mode,
        "effective_mode": mode,
        "reason": "",
        "current": current_candidate.as_dict(),
        "recommended": None,
        "applied": None,
        "sample": {},
        "gates": {},
        "top_candidates": [],
        "history": {},
        "transition": {"action": "", "reason": ""},
        "drift": {},
    }
    if not memory_forgetting_enabled(active_policy):
        state["reason"] = "forgetting disabled"
        if persist_state and callable(getattr(repo, "set_config", None)):
            repo.set_config(tenant, MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY, json.dumps(state, ensure_ascii=False), updated_by)
        return state

    rows = fetch_forgetting_tuning_rows(
        repo,
        tenant_id=tenant,
        trading_mode=trading_mode,
        lookback_days=lookback_days,
        stability_window_days=stability_window_days,
    )
    sample = _sample_summary(rows)
    state["sample"] = sample
    sample_ok = (
        sample["access_events"] >= sample_thresholds["min_access_events"]
        and sample["prompt_uses"] >= sample_thresholds["min_prompt_uses"]
        and sample["unique_memories"] >= sample_thresholds["min_unique_memories"]
    )
    cleanup_min_score = max(0.0, min(float(((active_policy.get("cleanup") or {}).get("min_score") or 0.30)), 1.0))
    recommendation: ForgettingCandidate | None = None
    improvement = 0.0
    stability_ok = False
    recommendation_drift = 1.0
    ranked: list[dict[str, Any]] = []
    if rows:
        current_eval = _evaluate_candidate(
            rows,
            current_candidate,
            cleanup_min_score=cleanup_min_score,
            objective_topk=objective_topk,
        )
        ranked = sorted(
            (
                _evaluate_candidate(
                    rows,
                    candidate,
                    cleanup_min_score=cleanup_min_score,
                    objective_topk=objective_topk,
                )
                for candidate in build_forgetting_candidate_grid(active_policy)
            ),
            key=lambda item: (
                float(item["metrics"]["objective"]),
                -float(item["metrics"]["instability_penalty"]),
                float(item["metrics"]["topk_precision"]),
            ),
            reverse=True,
        )
        best = ranked[0] if ranked else current_eval
        improvement = round(float(best["metrics"]["objective"]) - float(current_eval["metrics"]["objective"]), 6)
        stability_ok = float(best["metrics"]["instability_penalty"]) <= 0.20
        recommendation = ForgettingCandidate(**best["candidate"])
        recommendation_drift = _recommendation_drift(
            previous_state.get("recommended") if isinstance(previous_state.get("recommended"), dict) else None,
            best["candidate"],
        )
        state["recommended"] = {
            **best["candidate"],
            "metrics": best["metrics"],
            "improvement_vs_current": improvement,
        }
        state["top_candidates"] = [
            {
                **item["candidate"],
                "metrics": item["metrics"],
            }
            for item in ranked[:5]
        ]
    improvement_ok = improvement >= 0.01
    health_ok = bool(rows) and sample_ok and stability_ok
    prev_total_runs = _safe_int(previous_history.get("total_runs"))
    prev_shadow_runs = _safe_int(previous_history.get("shadow_runs_since_transition"))
    prev_bounded_ema_runs = _safe_int(previous_history.get("bounded_ema_runs_since_transition"))
    consecutive_sample_ok = (_safe_int(previous_history.get("consecutive_sample_ok")) + 1) if sample_ok else 0
    consecutive_stability_ok = (_safe_int(previous_history.get("consecutive_stability_ok")) + 1) if stability_ok else 0
    consecutive_improvement_ok = (_safe_int(previous_history.get("consecutive_improvement_ok")) + 1) if improvement_ok else 0
    consecutive_apply_eligible = (_safe_int(previous_history.get("consecutive_apply_eligible")) + 1) if (sample_ok and stability_ok and improvement_ok) else 0
    consecutive_unhealthy_runs = (_safe_int(previous_history.get("consecutive_unhealthy_runs")) + 1) if not health_ok else 0

    shadow_runs_since_transition = 0
    shadow_started_at_text = ""
    shadow_days_since_transition = 0
    if mode == "shadow":
        shadow_runs_since_transition = (prev_shadow_runs + 1) if previous_effective_mode == "shadow" else 1
        shadow_started_at_text = (
            str(previous_history.get("shadow_started_at") or "").strip()
            if previous_effective_mode == "shadow" and str(previous_history.get("shadow_started_at") or "").strip()
            else evaluation_now.isoformat()
        )
        parsed_shadow_started_at = _parse_state_datetime(shadow_started_at_text)
        if parsed_shadow_started_at is not None:
            shadow_days_since_transition = max(0, int((evaluation_now - parsed_shadow_started_at).days))

    effective_mode = mode
    transition_action = ""
    transition_reason = ""
    if (
        tuning_enabled
        and mode == "shadow"
        and auto_promote_enabled
        and sample_ok
        and stability_ok
        and improvement_ok
        and shadow_runs_since_transition >= auto_promote_min_shadow_runs
        and shadow_days_since_transition >= auto_promote_min_shadow_days
        and consecutive_stability_ok >= auto_promote_required_stable_runs
        and consecutive_improvement_ok >= auto_promote_required_improving_runs
        and recommendation_drift <= auto_promote_max_recommendation_drift
    ):
        effective_mode = "bounded_ema"
        transition_action = "auto_promote"
        transition_reason = "stable shadow recommendation reached bounded EMA criteria"
    elif tuning_enabled and mode == "bounded_ema" and auto_demote_enabled and consecutive_unhealthy_runs >= auto_demote_unhealthy_runs:
        effective_mode = "shadow"
        transition_action = "auto_demote"
        transition_reason = "bounded EMA health degraded below stability requirements"

    state["effective_mode"] = effective_mode
    state["gates"] = {
        "sample_ok": sample_ok,
        **sample_thresholds,
        "stability_ok": stability_ok,
        "improvement_ok": improvement_ok,
        "apply_allowed": False,
        "auto_promote_ready": bool(
            mode == "shadow"
            and sample_ok
            and stability_ok
            and improvement_ok
            and shadow_runs_since_transition >= auto_promote_min_shadow_runs
            and shadow_days_since_transition >= auto_promote_min_shadow_days
            and consecutive_stability_ok >= auto_promote_required_stable_runs
            and consecutive_improvement_ok >= auto_promote_required_improving_runs
            and recommendation_drift <= auto_promote_max_recommendation_drift
        ),
        "auto_demote_ready": bool(mode == "bounded_ema" and consecutive_unhealthy_runs >= auto_demote_unhealthy_runs),
    }
    state["drift"] = {
        "recommendation_drift": round(recommendation_drift, 6),
        "auto_promote_max_recommendation_drift": auto_promote_max_recommendation_drift,
    }

    if effective_mode == "shadow":
        post_shadow_runs_since_transition = 1 if transition_action == "auto_demote" else shadow_runs_since_transition
        post_shadow_started_at_text = evaluation_now.isoformat() if transition_action == "auto_demote" else shadow_started_at_text
        post_bounded_ema_runs_since_transition = 0
        post_bounded_ema_started_at_text = ""
    else:
        post_shadow_runs_since_transition = 0
        post_shadow_started_at_text = ""
        post_bounded_ema_runs_since_transition = 1 if transition_action == "auto_promote" else ((prev_bounded_ema_runs + 1) if previous_effective_mode == "bounded_ema" else 1)
        post_bounded_ema_started_at_text = (
            evaluation_now.isoformat()
            if transition_action == "auto_promote"
            else (
                str(previous_history.get("bounded_ema_started_at") or "").strip()
                if previous_effective_mode == "bounded_ema" and str(previous_history.get("bounded_ema_started_at") or "").strip()
                else evaluation_now.isoformat()
            )
        )
    state["history"] = {
        "total_runs": prev_total_runs + 1,
        "consecutive_sample_ok": consecutive_sample_ok,
        "consecutive_stability_ok": consecutive_stability_ok,
        "consecutive_improvement_ok": consecutive_improvement_ok,
        "consecutive_apply_eligible": consecutive_apply_eligible,
        "consecutive_unhealthy_runs": consecutive_unhealthy_runs,
        "shadow_runs_since_transition": post_shadow_runs_since_transition,
        "shadow_started_at": post_shadow_started_at_text,
        "shadow_days_since_transition": shadow_days_since_transition if effective_mode == "shadow" else 0,
        "bounded_ema_runs_since_transition": post_bounded_ema_runs_since_transition,
        "bounded_ema_started_at": post_bounded_ema_started_at_text,
        "last_transition_at": evaluation_now.isoformat() if transition_action else str(previous_history.get("last_transition_at") or ""),
        "last_transition_action": transition_action or str(previous_history.get("last_transition_action") or ""),
        "auto_promotions": _safe_int(previous_history.get("auto_promotions")) + (1 if transition_action == "auto_promote" else 0),
        "auto_demotions": _safe_int(previous_history.get("auto_demotions")) + (1 if transition_action == "auto_demote" else 0),
    }
    state["transition"] = {"action": transition_action, "reason": transition_reason}

    policy_to_persist = _policy_with_tuning_mode(active_policy, effective_mode) if transition_action else active_policy
    if tuning_enabled and effective_mode == "bounded_ema" and sample_ok and stability_ok and improvement_ok and recommendation is not None:
        next_policy = _apply_recommendation(
            policy_to_persist,
            recommendation,
            ema_alpha=memory_forgetting_tuning_ema_alpha(active_policy),
            max_decay_factor_delta=memory_forgetting_tuning_max_decay_factor_delta(active_policy),
            max_min_effective_score_delta=memory_forgetting_tuning_max_min_effective_score_delta(active_policy),
            max_tier_weight_delta=memory_forgetting_tuning_max_tier_weight_delta(active_policy),
        )
        state["applied"] = {
            "policy": forgetting_policy_snapshot(next_policy).as_dict(),
            "access_curve_pending": recommendation.access_curve if recommendation.access_curve != current_candidate.access_curve else "",
        }
        state["gates"]["apply_allowed"] = True
        policy_to_persist = next_policy
    else:
        state["applied"] = {
            "policy": forgetting_policy_snapshot(policy_to_persist).as_dict(),
            "access_curve_pending": recommendation.access_curve if recommendation is not None and recommendation.access_curve != current_candidate.access_curve else "",
        }

    if not tuning_enabled:
        state["reason"] = "tuning disabled"
    elif not rows:
        state["reason"] = "no access history"
    elif transition_action == "auto_promote":
        state["reason"] = "auto promoted and applied bounded ema" if state["gates"]["apply_allowed"] else "auto promoted to bounded ema"
    elif transition_action == "auto_demote":
        state["reason"] = "auto demoted to shadow"
    elif effective_mode == "shadow":
        state["reason"] = "shadow only"
    elif not sample_ok:
        state["reason"] = "insufficient sample"
    elif not stability_ok:
        state["reason"] = "stability gate failed"
    elif not improvement_ok:
        state["reason"] = "recommendation improvement too small"
    else:
        state["reason"] = "applied bounded ema"

    setter = getattr(repo, "set_config", None)
    if callable(setter) and (transition_action or state["gates"]["apply_allowed"]):
        setter(tenant, MEMORY_POLICY_CONFIG_KEY, serialize_memory_policy(policy_to_persist), updated_by)
    if persist_state and callable(setter):
        setter(tenant, MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY, json.dumps(state, ensure_ascii=False), updated_by)
    return state
