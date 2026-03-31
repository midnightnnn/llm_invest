from __future__ import annotations

import re
from datetime import datetime
from typing import Any

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


def _trim_text(value: Any, *, max_len: int = 220) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


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
) -> bool:
    previous = previous_payload if isinstance(previous_payload, dict) else {}
    previous_summary = _normalize_text_key(previous.get("thesis_summary") or "")
    next_summary = _normalize_text_key(rationale)
    previous_refs = set(normalize_strategy_refs(previous.get("strategy_refs") or []))
    next_refs = set(normalize_strategy_refs(strategy_refs))

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
    summary = _trim_text(intent.rationale or previous.get("thesis_summary") or "", max_len=220)
    cycle_id = str(intent.cycle_id or "").strip()
    strategy_refs = normalize_strategy_refs(intent.strategy_refs)
    state = thesis_state_for_event_type(event_type)
    payload: dict[str, Any] = {
        "source": "thesis_lifecycle",
        "thesis_id": thesis_id,
        "ticker": str(intent.ticker or "").strip().upper(),
        "side": str(getattr(intent.side, "value", intent.side) or "").strip().upper(),
        "state": state,
        "thesis_summary": summary,
        "strategy_refs": strategy_refs,
        "entry_cycle_id": str(previous.get("entry_cycle_id") or cycle_id or "").strip() or None,
        "last_cycle_id": cycle_id or None,
        "intent": intent.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
    }
    if position_action:
        payload["position_action"] = str(position_action).strip().lower()
    if position_qty_before is not None:
        payload["position_qty_before"] = float(position_qty_before)
    if position_qty_after is not None:
        payload["position_qty_after"] = float(position_qty_after)
    for key in ("key_claims", "invalidation_conditions", "source_post_ids", "source_briefing_ids", "source_event_ids"):
        value = previous.get(key)
        if isinstance(value, list) and value:
            payload[key] = list(value)
    for key in ("confidence", "time_horizon_days"):
        value = previous.get(key)
        if value is not None:
            payload[key] = value
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
