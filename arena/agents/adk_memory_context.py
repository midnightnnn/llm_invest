from __future__ import annotations

import json
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _clean_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _drop_empty(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value not in (None, "", [])}


def _candidate_type(row: dict[str, Any], payload: dict[str, Any]) -> str:
    event_type = str(row.get("event_type") or "").strip()
    if event_type:
        return event_type
    status = str(payload.get("candidate_status") or "").strip()
    return f"candidate_{status}" if status else "candidate"


def _candidate_memory(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    evidence = _as_dict(payload.get("discovery_evidence"))
    source_tools = _clean_list(payload.get("source_tools"))
    out: dict[str, Any] = {
        "event_id": row.get("event_id"),
        "d": row.get("created_date"),
        "type": _candidate_type(row, payload),
        "t": payload.get("ticker"),
        "src": source_tools[0] if source_tools else None,
        "rank": payload.get("last_seen_rank"),
        "score": _first_present(evidence.get("score"), row.get("score"), row.get("importance_score")),
        "checked": _clean_list(payload.get("analyzed_by")),
        "why": evidence.get("reason_for") or evidence.get("reason"),
        "risk": evidence.get("reason_risk"),
        "outcome": row.get("outcome_label"),
    }
    return _drop_empty(out)


def _generic_memory(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    intent = _as_dict(payload.get("intent"))
    out: dict[str, Any] = {
        "event_id": row.get("event_id"),
        "d": row.get("created_date"),
        "type": row.get("event_type"),
        "summary": row.get("summary"),
        "score": _first_present(row.get("importance_score"), row.get("score")),
        "outcome": row.get("outcome_label"),
        "src": row.get("memory_source") or payload.get("source"),
        "tier": row.get("memory_tier"),
        "t": payload.get("ticker") or intent.get("ticker"),
    }
    return _drop_empty(out)


def model_memory_context_rows(rows: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in rows[: max(1, int(limit))]:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        payload = _as_dict(row.get("payload") or row.get("payload_json"))
        event_type = str(row.get("event_type") or "").strip().lower()
        source = str(payload.get("source") or "").strip().lower()
        if event_type.startswith("candidate_") or source == "candidate_discovery":
            item = _candidate_memory(row, payload)
        else:
            item = _generic_memory(row, payload)
        if item:
            out.append(item)
    return out
