from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from arena.memory.policy import (
    MEMORY_POLICY_CONFIG_KEY,
    MEMORY_RELATION_TUNING_STATE_CONFIG_KEY,
    apply_memory_policy_to_settings,
    default_memory_policy,
    load_memory_policy,
    memory_graph_semantic_triples_enabled,
    memory_graph_semantic_triples_max_candidates,
    memory_graph_semantic_triples_max_relation_context_items,
    memory_graph_semantic_triples_mode,
    memory_graph_semantic_triples_tuning_auto_transition_enabled,
    memory_graph_semantic_triples_tuning_demote_on_version_change,
    memory_graph_semantic_triples_tuning_demote_unhealthy_evaluations,
    memory_graph_semantic_triples_tuning_enabled,
    memory_graph_semantic_triples_tuning_lookback_days,
    memory_graph_semantic_triples_tuning_min_accepted_triples,
    memory_graph_semantic_triples_tuning_min_sources,
    memory_graph_semantic_triples_tuning_post_demote_cooldown_evaluations,
    memory_graph_semantic_triples_tuning_required_healthy_evaluations,
    memory_graph_semantic_triples_tuning_stability_window_days,
    normalize_memory_policy,
    serialize_memory_policy,
)
from arena.memory.relation_ontology import ONTOLOGY_VERSION
from arena.memory.semantic_extractor import EXTRACTOR_VERSION, PROMPT_VERSION
from arena.models import utc_now

_SUPPORTIVE_PREDICATES = frozenset({"supports", "outcome_of", "similar_setup"})
_ADVERSE_PREDICATES = frozenset({"invalidates", "contradicts", "risk_to"})
_UNSAFE_REJECT_REASONS = frozenset(
    {
        "evidence_not_found",
        "unsupported_predicate",
        "unsupported_entity_type",
        "predicate_type_mismatch",
        "invalid_entity_node",
        "self_relation",
        "missing_triples_array",
        "not_object",
    }
)


@dataclass(frozen=True, slots=True)
class RateInterval:
    value: float
    low: float
    high: float


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


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _window_cutoff(now: datetime, days: int) -> datetime:
    return now - timedelta(days=max(1, int(days)))


def _in_window(row: dict[str, Any], key: str, cutoff: datetime) -> bool:
    stamp = _parse_datetime(row.get(key))
    if stamp is None:
        return True
    if stamp.tzinfo is None and cutoff.tzinfo is not None:
        stamp = stamp.replace(tzinfo=cutoff.tzinfo)
    return stamp >= cutoff


def _wilson_interval(successes: int, total: int) -> RateInterval:
    if total <= 0:
        return RateInterval(value=0.0, low=0.0, high=1.0)
    # 1.96 is the standard normal quantile for a two-sided 95% Wilson interval.
    z = 1.96
    n = float(total)
    p = max(0.0, min(float(successes) / n, 1.0))
    denom = 1.0 + (z * z / n)
    center = (p + (z * z / (2.0 * n))) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))) / denom
    return RateInterval(value=p, low=max(0.0, center - margin), high=min(1.0, center + margin))


def _intervals_overlap(left: RateInterval, right: RateInterval) -> bool:
    return left.low <= right.high and right.low <= left.high


def _concentration(values: list[str]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    shares = [count / total for count in counts.values()]
    hhi = sum(share * share for share in shares)
    effective_count = 0.0 if hhi <= 0.0 else 1.0 / hhi
    return max(shares), effective_count


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


def _load_previous_state(repo: Any, tenant_id: str) -> dict[str, Any]:
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return {}
    try:
        raw = getter(str(tenant_id or "").strip().lower() or "local", MEMORY_RELATION_TUNING_STATE_CONFIG_KEY)
    except Exception:
        return {}
    try:
        parsed = json.loads(str(raw or "").strip() or "{}")
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _policy_with_semantic_mode(policy: dict[str, Any], mode: str) -> dict[str, Any]:
    updated = copy.deepcopy(policy)
    updated.setdefault("graph", {}).setdefault("semantic_triples", {})["mode"] = (
        str(mode or "shadow").strip().lower() or "shadow"
    )
    return normalize_memory_policy(updated, defaults=normalize_memory_policy(policy))


def _derived_min_sources(policy: dict[str, Any], settings: Any) -> int:
    configured = memory_graph_semantic_triples_tuning_min_sources(policy)
    if configured > 0:
        return configured
    context_limit = max(1, int(getattr(settings, "context_max_memory_events", 32) or 32))
    max_hints = max(1, memory_graph_semantic_triples_max_relation_context_items(policy))
    max_candidates = max(1, memory_graph_semantic_triples_max_candidates(policy))
    return max(context_limit * max_hints, max_candidates * max_hints * 2)


def _derived_min_accepted_triples(policy: dict[str, Any]) -> int:
    configured = memory_graph_semantic_triples_tuning_min_accepted_triples(policy)
    if configured > 0:
        return configured
    max_hints = max(1, memory_graph_semantic_triples_max_relation_context_items(policy))
    required_runs = memory_graph_semantic_triples_tuning_required_healthy_evaluations(policy)
    return max_hints * required_runs


def fetch_relation_tuning_extraction_runs(
    repo: Any,
    *,
    tenant_id: str,
    trading_mode: str,
    lookback_days: int,
) -> list[dict[str, Any]]:
    sql = f"""
    SELECT
      run_id, started_at, finished_at, source_table, source_id, source_hash,
      source_created_at, agent_id, trading_mode, cycle_id, extractor_version,
      prompt_version, ontology_version, provider, model, status,
      accepted_count, rejected_count, detail_json
    FROM `{repo.dataset_fqn}.memory_relation_extraction_runs`
    WHERE tenant_id = @tenant_id
      AND trading_mode = @trading_mode
      AND started_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
      AND extractor_version = @extractor_version
      AND prompt_version = @prompt_version
      AND ontology_version = @ontology_version
    ORDER BY started_at DESC
    """
    return repo.fetch_rows(
        sql,
        {
            "tenant_id": str(tenant_id or "").strip().lower() or "local",
            "trading_mode": str(trading_mode or "paper").strip().lower() or "paper",
            "lookback_days": max(3, int(lookback_days)),
            "extractor_version": EXTRACTOR_VERSION,
            "prompt_version": PROMPT_VERSION,
            "ontology_version": ONTOLOGY_VERSION,
        },
    )


def fetch_relation_tuning_triples(
    repo: Any,
    *,
    tenant_id: str,
    trading_mode: str,
    lookback_days: int,
) -> list[dict[str, Any]]:
    sql = f"""
    SELECT
      triple_id, created_at, source_table, source_id, source_node_id,
      subject_node_id, subject_type, predicate, object_node_id, object_type,
      confidence, extraction_method, extraction_version, detail_json
    FROM `{repo.dataset_fqn}.memory_relation_triples`
    WHERE tenant_id = @tenant_id
      AND trading_mode = @trading_mode
      AND status = 'accepted'
      AND extraction_method = 'llm'
      AND extraction_version = @extractor_version
      AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY)
    """
    return repo.fetch_rows(
        sql,
        {
            "tenant_id": str(tenant_id or "").strip().lower() or "local",
            "trading_mode": str(trading_mode or "paper").strip().lower() or "paper",
            "lookback_days": max(3, int(lookback_days)),
            "extractor_version": EXTRACTOR_VERSION,
        },
    )


def _reject_reason_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        detail = _json_object(row.get("detail_json"))
        rejected = detail.get("rejected")
        if not isinstance(rejected, list):
            continue
        for item in rejected:
            if not isinstance(item, dict):
                continue
            reason = str(item.get("reason") or "").strip().lower()
            if reason:
                counts[reason] += 1
    return counts


def _ticker_key(row: dict[str, Any]) -> str:
    subject_type = str(row.get("subject_type") or "").strip().lower()
    object_type = str(row.get("object_type") or "").strip().lower()
    if subject_type == "ticker":
        return str(row.get("subject_node_id") or "").strip()
    if object_type == "ticker":
        return str(row.get("object_node_id") or "").strip()
    return ""


def _metrics_for_window(run_rows: list[dict[str, Any]], triples: list[dict[str, Any]]) -> dict[str, Any]:
    source_count = len(run_rows)
    accepted_count = sum(_safe_int(row.get("accepted_count")) for row in run_rows)
    rejected_count = sum(_safe_int(row.get("rejected_count")) for row in run_rows)
    failed_run_count = sum(1 for row in run_rows if str(row.get("status") or "").strip().lower() == "failed")
    invalid_output_count = sum(1 for row in run_rows if str(row.get("status") or "").strip().lower() == "invalid_output")
    reason_counts = _reject_reason_counts(run_rows)
    unsafe_reject_count = sum(reason_counts.get(reason, 0) for reason in _UNSAFE_REJECT_REASONS) + failed_run_count + invalid_output_count
    denominator = max(0, accepted_count + rejected_count + failed_run_count + invalid_output_count)

    predicate_counts: Counter[str] = Counter(str(row.get("predicate") or "").strip().lower() for row in triples)
    strong_count = sum(predicate_counts.get(predicate, 0) for predicate in _ADVERSE_PREDICATES)
    accepted_triples = len(triples)
    strong_interval = _wilson_interval(strong_count, accepted_triples)

    object_predicates: dict[str, set[str]] = defaultdict(set)
    for row in triples:
        obj = str(row.get("object_node_id") or "").strip()
        predicate = str(row.get("predicate") or "").strip().lower()
        if obj and predicate:
            object_predicates[obj].add(predicate)
    conflict_objects = [
        obj
        for obj, predicates in object_predicates.items()
        if predicates & _SUPPORTIVE_PREDICATES and predicates & _ADVERSE_PREDICATES
    ]
    conflict_interval = _wilson_interval(len(conflict_objects), len(object_predicates))

    accepted_source_ids = [
        str(row.get("source_table") or "").strip() + ":" + str(row.get("source_id") or "").strip()
        for row in triples
        if str(row.get("source_id") or "").strip()
    ]
    ticker_ids = [ticker for ticker in (_ticker_key(row) for row in triples) if ticker]
    source_concentration, effective_source_count = _concentration(accepted_source_ids)
    ticker_concentration, effective_ticker_count = _concentration(ticker_ids)
    accepted_interval = _wilson_interval(accepted_count, denominator)
    unsafe_interval = _wilson_interval(unsafe_reject_count, denominator)

    return {
        "source_count": source_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "failed_run_count": failed_run_count,
        "invalid_output_count": invalid_output_count,
        "unsafe_reject_count": unsafe_reject_count,
        "denominator": denominator,
        "accepted_rate": accepted_interval.value,
        "unsafe_reject_rate": unsafe_interval.value,
        "accepted_interval": accepted_interval,
        "unsafe_interval": unsafe_interval,
        "predicate_counts": dict(predicate_counts),
        "strong_predicate_count": strong_count,
        "strong_predicate_ratio": strong_interval.value,
        "strong_interval": strong_interval,
        "conflict_count": len(conflict_objects),
        "unique_object_count": len(object_predicates),
        "conflict_ratio": conflict_interval.value,
        "conflict_interval": conflict_interval,
        "source_concentration": source_concentration,
        "effective_source_count": effective_source_count,
        "ticker_concentration": ticker_concentration,
        "effective_ticker_count": effective_ticker_count,
        "reject_reason_counts": dict(reason_counts),
        "models": sorted({str(row.get("model") or "").strip() for row in run_rows if str(row.get("model") or "").strip()}),
        "providers": sorted({str(row.get("provider") or "").strip() for row in run_rows if str(row.get("provider") or "").strip()}),
    }


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    for key in ("accepted_interval", "unsafe_interval", "strong_interval", "conflict_interval"):
        interval = out.get(key)
        if isinstance(interval, RateInterval):
            out[key] = {"value": interval.value, "low": interval.low, "high": interval.high}
    return out


def _source_diversity_ok(metrics: dict[str, Any]) -> bool:
    accepted_count = _safe_int(metrics.get("accepted_count"))
    if accepted_count <= 0:
        return False
    effective = _safe_float(metrics.get("effective_source_count"))
    source_count = max(1, _safe_int(metrics.get("source_count")))
    # Effective source count is compared with sqrt(source_count), so the required
    # diversity grows with observed sample size rather than a fixed cutoff.
    return effective >= math.sqrt(source_count)


def _health_gates(metrics: dict[str, Any], *, min_sources: int, min_accepted_triples: int) -> dict[str, bool]:
    sample_ok = (
        _safe_int(metrics.get("source_count")) >= max(1, int(min_sources))
        and _safe_int(metrics.get("accepted_count")) >= max(1, int(min_accepted_triples))
    )
    accepted_interval = metrics.get("accepted_interval")
    unsafe_interval = metrics.get("unsafe_interval")
    strong_count = _safe_int(metrics.get("strong_predicate_count"))
    accepted_count = _safe_int(metrics.get("accepted_count"))
    conflict_count = _safe_int(metrics.get("conflict_count"))
    object_count = _safe_int(metrics.get("unique_object_count"))
    accepted_signal_ok = (
        isinstance(accepted_interval, RateInterval)
        and isinstance(unsafe_interval, RateInterval)
        and accepted_interval.low > unsafe_interval.high
    )
    strong_not_dominant = accepted_count > 0 and strong_count <= (accepted_count - strong_count)
    conflict_not_dominant = object_count == 0 or conflict_count <= (object_count - conflict_count)
    diversity_ok = _source_diversity_ok(metrics)
    return {
        "sample_ok": sample_ok,
        "accepted_signal_ok": accepted_signal_ok,
        "strong_not_dominant": strong_not_dominant,
        "conflict_not_dominant": conflict_not_dominant,
        "source_diversity_ok": diversity_ok,
        "health_ok": sample_ok and accepted_signal_ok and strong_not_dominant and conflict_not_dominant and diversity_ok,
    }


def _stability_ok(long_metrics: dict[str, Any], short_metrics: dict[str, Any], *, short_min_sources: int) -> bool:
    if _safe_int(short_metrics.get("source_count")) < max(1, int(short_min_sources)):
        return True
    checks = [
        _intervals_overlap(short_metrics["unsafe_interval"], long_metrics["unsafe_interval"])
        or short_metrics["unsafe_reject_rate"] <= long_metrics["unsafe_reject_rate"],
        _intervals_overlap(short_metrics["strong_interval"], long_metrics["strong_interval"])
        or short_metrics["strong_predicate_ratio"] <= long_metrics["strong_predicate_ratio"],
        _intervals_overlap(short_metrics["conflict_interval"], long_metrics["conflict_interval"])
        or short_metrics["conflict_ratio"] <= long_metrics["conflict_ratio"],
    ]
    return all(checks)


def _version_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "extractor_version": EXTRACTOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "models": sorted({str(token or "").strip() for token in (metrics.get("models") or []) if str(token or "").strip()}),
        "providers": sorted({str(token or "").strip() for token in (metrics.get("providers") or []) if str(token or "").strip()}),
    }


def _canonical_version_snapshot(raw: Any) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    return {
        "extractor_version": str(data.get("extractor_version") or "").strip(),
        "prompt_version": str(data.get("prompt_version") or "").strip(),
        "ontology_version": str(data.get("ontology_version") or "").strip(),
        "models": sorted({str(token or "").strip() for token in (data.get("models") or []) if str(token or "").strip()}),
        "providers": sorted({str(token or "").strip() for token in (data.get("providers") or []) if str(token or "").strip()}),
    }


def run_memory_relation_tuner(
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
    previous_state = _load_previous_state(repo, tenant)
    previous_history = previous_state.get("history") if isinstance(previous_state.get("history"), dict) else {}
    previous_versions = previous_state.get("versions") if isinstance(previous_state.get("versions"), dict) else {}
    trading_mode = getattr(settings, "trading_mode", "paper").strip().lower()
    evaluation_now = utc_now()

    mode = memory_graph_semantic_triples_mode(active_policy)
    tuning_enabled = memory_graph_semantic_triples_tuning_enabled(active_policy)
    auto_transition_enabled = memory_graph_semantic_triples_tuning_auto_transition_enabled(active_policy)
    lookback_days = memory_graph_semantic_triples_tuning_lookback_days(active_policy)
    stability_window_days = min(
        memory_graph_semantic_triples_tuning_stability_window_days(active_policy),
        lookback_days,
    )
    min_sources = _derived_min_sources(active_policy, settings)
    min_accepted = _derived_min_accepted_triples(active_policy)
    required_healthy = memory_graph_semantic_triples_tuning_required_healthy_evaluations(active_policy)
    demote_unhealthy = memory_graph_semantic_triples_tuning_demote_unhealthy_evaluations(active_policy)
    configured_cooldown = memory_graph_semantic_triples_tuning_post_demote_cooldown_evaluations(active_policy)
    post_demote_cooldown = configured_cooldown if configured_cooldown > 0 else required_healthy

    run_rows = fetch_relation_tuning_extraction_runs(
        repo,
        tenant_id=tenant,
        trading_mode=trading_mode,
        lookback_days=lookback_days,
    )
    triples = fetch_relation_tuning_triples(
        repo,
        tenant_id=tenant,
        trading_mode=trading_mode,
        lookback_days=lookback_days,
    )
    long_metrics = _metrics_for_window(run_rows, triples)
    short_cutoff = _window_cutoff(evaluation_now, stability_window_days)
    short_run_rows = [row for row in run_rows if _in_window(row, "started_at", short_cutoff)]
    short_triples = [row for row in triples if _in_window(row, "created_at", short_cutoff)]
    short_metrics = _metrics_for_window(short_run_rows, short_triples)
    short_min_sources = max(1, math.ceil(min_sources * (float(stability_window_days) / max(float(lookback_days), 1.0))))
    gates = _health_gates(long_metrics, min_sources=min_sources, min_accepted_triples=min_accepted)
    stability_ok = _stability_ok(long_metrics, short_metrics, short_min_sources=short_min_sources)
    versions = _version_snapshot(long_metrics)
    previous_versions_canonical = _canonical_version_snapshot(previous_versions)
    versions_canonical = _canonical_version_snapshot(versions)
    version_changed = bool(previous_versions and previous_versions_canonical != versions_canonical)
    previous_last_transition = str(previous_history.get("last_transition_action") or "").strip()
    previous_evaluations_since_demotion = _safe_int(previous_history.get("evaluations_since_demotion"))
    if previous_last_transition == "auto_demote_to_shadow":
        evaluations_since_demotion = previous_evaluations_since_demotion + 1
        demotion_cooldown_ok = evaluations_since_demotion >= post_demote_cooldown
    else:
        evaluations_since_demotion = post_demote_cooldown
        demotion_cooldown_ok = True

    if not memory_graph_semantic_triples_enabled(active_policy):
        effective_mode = mode
        transition_action = ""
        reason = "semantic triples disabled"
    else:
        previous_consecutive_healthy = 0 if version_changed else _safe_int(previous_history.get("consecutive_healthy"))
        previous_consecutive_unhealthy = 0 if version_changed else _safe_int(previous_history.get("consecutive_unhealthy"))
        healthy_now = bool(gates["health_ok"] and stability_ok)
        consecutive_healthy = previous_consecutive_healthy + 1 if healthy_now else 0
        consecutive_unhealthy = previous_consecutive_unhealthy + 1 if not healthy_now else 0

        effective_mode = mode
        transition_action = ""
        reason = "shadow only" if mode == "shadow" else "inject healthy" if mode == "inject" else "boost evaluated"
        if tuning_enabled and auto_transition_enabled:
            if mode in {"shadow", "boost"} and consecutive_healthy >= required_healthy and demotion_cooldown_ok:
                effective_mode = "inject"
                transition_action = "auto_promote_to_inject"
                reason = "relation quality and stability gates passed consecutively"
            elif mode == "inject" and version_changed and memory_graph_semantic_triples_tuning_demote_on_version_change(active_policy):
                # Version changes invalidate the observed safety window, so demotion wins
                # even if the current sample is otherwise healthy.
                effective_mode = "shadow"
                transition_action = "auto_demote_to_shadow"
                reason = "extractor, prompt, ontology, or model version changed"
            elif mode == "inject" and consecutive_unhealthy >= demote_unhealthy:
                effective_mode = "shadow"
                transition_action = "auto_demote_to_shadow"
                reason = "relation quality or stability gates failed consecutively"
        elif not tuning_enabled:
            reason = "relation tuning disabled"
        elif not auto_transition_enabled:
            reason = "auto transition disabled"

    if "consecutive_healthy" not in locals():
        healthy_now = bool(gates["health_ok"] and stability_ok)
        consecutive_healthy = _safe_int(previous_history.get("consecutive_healthy")) + 1 if healthy_now else 0
        consecutive_unhealthy = _safe_int(previous_history.get("consecutive_unhealthy")) + 1 if not healthy_now else 0

    state = {
        "tenant_id": tenant,
        "trading_mode": trading_mode,
        "evaluated_at": evaluation_now.isoformat(),
        "enabled": tuning_enabled,
        "configured_mode": mode,
        "effective_mode": effective_mode,
        "recommended_mode": "inject" if gates["health_ok"] and stability_ok else "shadow",
        "reason": reason,
        "transition": {"action": transition_action, "reason": reason},
        "gates": {
            **gates,
            "stability_ok": stability_ok,
            "healthy_now": bool(gates["health_ok"] and stability_ok),
            "version_changed": version_changed,
            "demotion_cooldown_ok": demotion_cooldown_ok,
            "auto_transition_enabled": auto_transition_enabled,
            "required_healthy_evaluations": required_healthy,
            "demote_unhealthy_evaluations": demote_unhealthy,
            "post_demote_cooldown_evaluations": post_demote_cooldown,
            "evaluations_since_demotion": evaluations_since_demotion,
            "min_sources": min_sources,
            "min_accepted_triples": min_accepted,
            "short_min_sources": short_min_sources,
        },
        "metrics": _public_metrics(long_metrics),
        "short_metrics": _public_metrics(short_metrics),
        "versions": versions_canonical,
        "history": {
            "total_runs": _safe_int(previous_history.get("total_runs")) + 1,
            "consecutive_healthy": consecutive_healthy,
            "consecutive_unhealthy": consecutive_unhealthy,
            "last_transition_at": evaluation_now.isoformat()
            if transition_action
            else str(previous_history.get("last_transition_at") or ""),
            "last_transition_action": transition_action or str(previous_history.get("last_transition_action") or ""),
            "auto_promotions": _safe_int(previous_history.get("auto_promotions"))
            + (1 if transition_action == "auto_promote_to_inject" else 0),
            "auto_demotions": _safe_int(previous_history.get("auto_demotions"))
            + (1 if transition_action == "auto_demote_to_shadow" else 0),
            "evaluations_since_demotion": 0
            if transition_action == "auto_demote_to_shadow"
            else evaluations_since_demotion,
        },
    }

    policy_to_persist = _policy_with_semantic_mode(active_policy, effective_mode) if transition_action else active_policy
    setter = getattr(repo, "set_config", None)
    if callable(setter) and transition_action:
        setter(tenant, MEMORY_POLICY_CONFIG_KEY, serialize_memory_policy(policy_to_persist), updated_by)
        apply_memory_policy_to_settings(settings, policy_to_persist)
    if persist_state and callable(setter):
        setter(tenant, MEMORY_RELATION_TUNING_STATE_CONFIG_KEY, json.dumps(state, ensure_ascii=False), updated_by)

    append_metrics = getattr(repo, "append_memory_relation_tuning_runs", None)
    if callable(append_metrics):
        append_metrics(
            [
                {
                    "run_id": f"rel_tune_{uuid4().hex[:12]}",
                    "evaluated_at": evaluation_now,
                    "trading_mode": trading_mode,
                    "configured_mode": mode,
                    "effective_mode": effective_mode,
                    "recommended_mode": state["recommended_mode"],
                    "transition_action": transition_action,
                    "reason": reason,
                    "source_count": long_metrics["source_count"],
                    "accepted_count": long_metrics["accepted_count"],
                    "rejected_count": long_metrics["rejected_count"],
                    "unsafe_reject_count": long_metrics["unsafe_reject_count"],
                    "failed_run_count": long_metrics["failed_run_count"],
                    "invalid_output_count": long_metrics["invalid_output_count"],
                    "accepted_rate": long_metrics["accepted_rate"],
                    "unsafe_reject_rate": long_metrics["unsafe_reject_rate"],
                    "strong_predicate_ratio": long_metrics["strong_predicate_ratio"],
                    "conflict_ratio": long_metrics["conflict_ratio"],
                    "source_concentration": long_metrics["source_concentration"],
                    "ticker_concentration": long_metrics["ticker_concentration"],
                    "sample_ok": gates["sample_ok"],
                    "health_ok": gates["health_ok"],
                    "stability_ok": stability_ok,
                    "detail_json": state,
                }
            ],
            tenant_id=tenant,
        )
    return state
