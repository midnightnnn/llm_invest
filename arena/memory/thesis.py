from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from arena.memory.relation_ontology import canonical_entity_type, predicate_allows
from arena.models import ExecutionReport, OrderIntent, RiskDecision, utc_now

THESIS_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "thesis_open",
        "thesis_update",
        "thesis_invalidated",
        "thesis_realized",
    }
)
ACTIVE_THESIS_EVENT_TYPES: frozenset[str] = frozenset({"thesis_open", "thesis_update"})
CLOSED_THESIS_EVENT_TYPES: frozenset[str] = frozenset({"thesis_invalidated", "thesis_realized"})

_THESIS_BREAK_TOKENS: frozenset[str] = frozenset(
    {
        "thesis_broken",
        "thesis_invalidated",
        "thesis_failed",
        "invalidated",
    }
)

_THESIS_FACTOR_LIMIT = 5
_SUPPORT_DEFAULT_TYPE = "catalyst"
_RISK_DEFAULT_TYPE = "risk"
_INVALIDATION_DEFAULT_TYPE = "risk"


def _trim_text(value: Any, *, max_len: int = 220) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _single_line_text(value: Any, *, max_len: int | None = None) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if max_len is None or len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "..."


def _normalize_text_key(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def normalize_strategy_refs(value: Any) -> list[str]:
    refs = value if isinstance(value, list) else []
    out: list[str] = []
    for item in refs:
        token = str(item or "").strip().lower()
        if token and token not in out:
            out.append(token)
    return out


def _normalize_factor_items(
    value: Any,
    *,
    predicate: str,
    default_type: str,
    limit: int = _THESIS_FACTOR_LIMIT,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            label = _single_line_text(item.get("label") or item.get("name") or item.get("text"), max_len=180)
            entity_type = canonical_entity_type(str(item.get("type") or item.get("entity_type") or default_type))
            evidence = _single_line_text(item.get("evidence") or item.get("evidence_text") or "", max_len=360)
        else:
            label = _single_line_text(item, max_len=180)
            entity_type = canonical_entity_type(default_type)
            evidence = ""
        if not label or not predicate_allows(predicate, entity_type, "thesis"):
            continue
        factor = {"label": label, "type": entity_type}
        if evidence:
            factor["evidence"] = evidence
        if factor not in out:
            out.append(factor)
        if len(out) >= limit:
            break
    return out


def _previous_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return list(value) if isinstance(value, list) else []


def _optional_positive_int(value: Any) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _optional_unit_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(number, 1.0))


def build_thesis_id(
    *,
    agent_id: str,
    ticker: str,
    trading_mode: str,
    intent_id: str,
    created_at: datetime | None = None,
) -> str:
    stamp = (created_at or utc_now()).date().isoformat()
    return (
        f"thesis:{str(agent_id or '').strip().lower()}:{str(ticker or '').strip().upper()}:"
        f"{str(trading_mode or '').strip().lower() or 'paper'}:{stamp}:{str(intent_id or '').strip()}"
    )


def thesis_state_for_event_type(event_type: str) -> str:
    token = str(event_type or "").strip().lower()
    mapping = {
        "thesis_open": "open",
        "thesis_update": "active",
        "thesis_invalidated": "invalidated",
        "thesis_realized": "realized",
    }
    return mapping.get(token, "")


def is_thesis_broken(strategy_refs: list[str] | Any) -> bool:
    refs = normalize_strategy_refs(strategy_refs)
    return any(ref in _THESIS_BREAK_TOKENS for ref in refs)


def is_material_thesis_update(
    previous_payload: dict[str, Any] | None,
    *,
    rationale: str,
    strategy_refs: list[str] | Any,
    thesis_core: str = "",
    supporting_factors: list[dict[str, Any]] | None = None,
    risk_factors: list[dict[str, Any]] | None = None,
    invalidation_conditions: list[dict[str, Any]] | None = None,
    expected_outcome: str = "",
    sizing_reason: str = "",
    time_horizon_days: int | None = None,
) -> bool:
    previous = previous_payload if isinstance(previous_payload, dict) else {}
    previous_summary = _normalize_text_key(previous.get("thesis_core") or previous.get("thesis_summary") or "")
    next_summary = _normalize_text_key(thesis_core or rationale)
    previous_refs = set(normalize_strategy_refs(previous.get("strategy_refs") or []))
    next_refs = set(normalize_strategy_refs(strategy_refs))
    next_supporting = _normalize_factor_items(
        supporting_factors or [],
        predicate="supports",
        default_type=_SUPPORT_DEFAULT_TYPE,
    )
    next_risks = _normalize_factor_items(
        risk_factors or [],
        predicate="risk_to",
        default_type=_RISK_DEFAULT_TYPE,
    )
    next_invalidations = _normalize_factor_items(
        invalidation_conditions or [],
        predicate="invalidates",
        default_type=_INVALIDATION_DEFAULT_TYPE,
    )
    next_structured = {
        "supporting_factors": next_supporting,
        "risk_factors": next_risks,
        "invalidation_conditions": next_invalidations,
        "expected_outcome": _single_line_text(expected_outcome),
        "sizing_reason": _single_line_text(sizing_reason),
        "time_horizon_days": _optional_positive_int(time_horizon_days),
    }
    if any(bool(value) for value in next_structured.values()):
        previous_structured = {
            "supporting_factors": previous.get("supporting_factors") if isinstance(previous.get("supporting_factors"), list) else [],
            "risk_factors": previous.get("risk_factors") if isinstance(previous.get("risk_factors"), list) else [],
            "invalidation_conditions": previous.get("invalidation_conditions") if isinstance(previous.get("invalidation_conditions"), list) else [],
            "expected_outcome": _single_line_text(previous.get("expected_outcome")),
            "sizing_reason": _single_line_text(previous.get("sizing_reason")),
            "time_horizon_days": _optional_positive_int(previous.get("time_horizon_days")),
        }
        if previous_structured != next_structured:
            return True

    if next_refs and next_refs != previous_refs:
        return True
    if not next_summary:
        return False
    if not previous_summary:
        return True
    return next_summary != previous_summary and next_summary not in previous_summary and previous_summary not in next_summary


def build_thesis_payload(
    *,
    event_type: str,
    thesis_id: str,
    intent: OrderIntent,
    decision: RiskDecision,
    report: ExecutionReport,
    previous_payload: dict[str, Any] | None = None,
    position_action: str = "",
    position_qty_before: float | None = None,
    position_qty_after: float | None = None,
) -> dict[str, Any]:
    previous = previous_payload if isinstance(previous_payload, dict) else {}
    thesis_core = _single_line_text(getattr(intent, "thesis_core", ""))
    summary = thesis_core or str(intent.rationale or previous.get("thesis_summary") or "").replace("\n", " ").strip()
    rationale = _single_line_text(intent.rationale)
    cycle_id = str(intent.cycle_id or "").strip()
    strategy_refs = normalize_strategy_refs(intent.strategy_refs)
    state = thesis_state_for_event_type(event_type)
    supporting_factors = _normalize_factor_items(
        getattr(intent, "supporting_factors", []),
        predicate="supports",
        default_type=_SUPPORT_DEFAULT_TYPE,
    )
    risk_factors = _normalize_factor_items(
        getattr(intent, "risk_factors", []),
        predicate="risk_to",
        default_type=_RISK_DEFAULT_TYPE,
    )
    invalidation_conditions = _normalize_factor_items(
        getattr(intent, "invalidation_conditions", []),
        predicate="invalidates",
        default_type=_INVALIDATION_DEFAULT_TYPE,
    )
    expected_outcome = _single_line_text(getattr(intent, "expected_outcome", ""))
    sizing_reason = _single_line_text(getattr(intent, "sizing_reason", ""))
    time_horizon_days = _optional_positive_int(getattr(intent, "time_horizon_days", None))
    thesis_confidence = _optional_unit_float(getattr(intent, "thesis_confidence", None))
    payload: dict[str, Any] = {
        "source": "thesis_lifecycle",
        "thesis_id": thesis_id,
        "ticker": str(intent.ticker or "").strip().upper(),
        "side": str(getattr(intent.side, "value", intent.side) or "").strip().upper(),
        "state": state,
        "thesis_summary": summary,
        "rationale": rationale,
        "strategy_refs": strategy_refs,
        "entry_cycle_id": str(previous.get("entry_cycle_id") or cycle_id or "").strip() or None,
        "last_cycle_id": cycle_id or None,
        "intent": intent.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
    }
    if thesis_core:
        payload["thesis_core"] = thesis_core
    if supporting_factors:
        payload["supporting_factors"] = supporting_factors
        payload["key_claims"] = [item["label"] for item in supporting_factors]
    elif _previous_list(previous, "supporting_factors"):
        payload["supporting_factors"] = _previous_list(previous, "supporting_factors")
    elif _previous_list(previous, "key_claims"):
        payload["key_claims"] = _previous_list(previous, "key_claims")
    if risk_factors:
        payload["risk_factors"] = risk_factors
    elif _previous_list(previous, "risk_factors"):
        payload["risk_factors"] = _previous_list(previous, "risk_factors")
    if invalidation_conditions:
        payload["invalidation_conditions"] = invalidation_conditions
    elif _previous_list(previous, "invalidation_conditions"):
        payload["invalidation_conditions"] = _previous_list(previous, "invalidation_conditions")
    if expected_outcome:
        payload["expected_outcome"] = expected_outcome
    elif previous.get("expected_outcome"):
        payload["expected_outcome"] = previous.get("expected_outcome")
    if sizing_reason:
        payload["sizing_reason"] = sizing_reason
    elif previous.get("sizing_reason"):
        payload["sizing_reason"] = previous.get("sizing_reason")
    if time_horizon_days is not None:
        payload["time_horizon_days"] = time_horizon_days
    elif previous.get("time_horizon_days") is not None:
        payload["time_horizon_days"] = previous.get("time_horizon_days")
    if thesis_confidence is not None:
        payload["thesis_confidence"] = thesis_confidence
        payload["confidence"] = thesis_confidence
    elif previous.get("thesis_confidence") is not None:
        payload["thesis_confidence"] = previous.get("thesis_confidence")
    elif previous.get("confidence") is not None:
        payload["confidence"] = previous.get("confidence")
    if position_action:
        payload["position_action"] = str(position_action).strip().lower()
    if position_qty_before is not None:
        payload["position_qty_before"] = float(position_qty_before)
    if position_qty_after is not None:
        payload["position_qty_after"] = float(position_qty_after)
    for key in ("source_post_ids", "source_briefing_ids", "source_event_ids"):
        value = previous.get(key)
        if isinstance(value, list) and value:
            payload[key] = list(value)
    return payload


def thesis_event_summary(
    *,
    event_type: str,
    payload: dict[str, Any],
    report: ExecutionReport,
) -> str:
    ticker = str(payload.get("ticker") or "").strip().upper()
    thesis_summary = _trim_text(payload.get("thesis_summary"), max_len=140)
    position_action = str(payload.get("position_action") or "").strip().lower()
    status = str(getattr(report.status, "value", report.status) or "").strip().upper()
    if event_type == "thesis_open":
        return f"{ticker} thesis open status={status} thesis={thesis_summary}"
    if event_type == "thesis_update":
        action = position_action or "update"
        return f"{ticker} thesis update action={action} status={status} thesis={thesis_summary}"
    if event_type == "thesis_invalidated":
        return f"{ticker} thesis invalidated status={status} thesis={thesis_summary}"
    if event_type == "thesis_realized":
        return f"{ticker} thesis realized status={status} thesis={thesis_summary}"
    return f"{ticker} thesis event status={status} thesis={thesis_summary}"
