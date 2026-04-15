"""Builds a ticker-indexed view over the per-cycle tool evidence log.

The evidence index does not synthesise new data; it re-sorts the outputs of
evidence-producing tools (see :mod:`arena.agents.tool_evidence`) so that held
positions and candidate tickers appear in the same schema and the model can
compare them without re-assembling scattered tool-result blocks.

Scope taxonomy:

- ``ticker`` events group under :data:`security_cases` by ticker.
- Non-ticker events (``portfolio``, ``market``, ``macro``, ``sector``) are
  collected into :data:`cycle_evidence`.

Same ``(ticker, tool)`` calls replace (latest wins). The log itself remains
append-only so order is preserved for the non-ticker stream.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from arena.agents.tool_evidence import extract_evidence


def record_tool_evidence(
    log: list[dict[str, Any]],
    *,
    tool_name: str,
    result: Any,
    phase: str | None = None,
    called_at: str | None = None,
) -> None:
    """Appends evidence events extracted from one tool call onto the cycle log."""
    for event in extract_evidence(tool_name, result):
        entry: dict[str, Any] = dict(event)
        entry["tool"] = str(tool_name or "").strip()
        if phase:
            entry["phase"] = str(phase)
        if called_at:
            entry["called_at"] = str(called_at)
        log.append(entry)


def build_evidence_index(
    evidence_log: Iterable[dict[str, Any]],
    *,
    held_tickers: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Projects an append-only evidence log into the model-facing index shape."""
    held = {
        str(t).strip().upper()
        for t in (held_tickers or ())
        if str(t or "").strip()
    }

    cycle_evidence: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    event_slot: dict[tuple[str, str], int] = {}

    for raw in evidence_log:
        if not isinstance(raw, dict):
            continue
        tool = str(raw.get("tool") or "").strip()
        scope = str(raw.get("scope") or "").strip()
        if not tool or not scope:
            continue
        summary = raw.get("summary") if isinstance(raw.get("summary"), dict) else {}
        phase = str(raw.get("phase") or "").strip() or None
        called_at = str(raw.get("called_at") or "").strip() or None

        if scope == "ticker":
            ticker = str(raw.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            case = cases.get(ticker)
            if case is None:
                case = {
                    "ticker": ticker,
                    "role": "held" if ticker in held else "candidate",
                    "sources": [],
                    "latest_evidence": [],
                }
                cases[ticker] = case
                order.append(ticker)
            event: dict[str, Any] = {"tool": tool, "summary": dict(summary)}
            if phase:
                event["phase"] = phase
            if called_at:
                event["called_at"] = called_at
            slot_key = (ticker, tool)
            slot = event_slot.get(slot_key)
            if slot is None:
                case["latest_evidence"].append(event)
                event_slot[slot_key] = len(case["latest_evidence"]) - 1
                if tool not in case["sources"]:
                    case["sources"].append(tool)
            else:
                case["latest_evidence"][slot] = event
            continue

        cycle_event: dict[str, Any] = {
            "tool": tool,
            "scope": scope,
            "summary": dict(summary),
        }
        if phase:
            cycle_event["phase"] = phase
        if called_at:
            cycle_event["called_at"] = called_at
        if raw.get("sector"):
            cycle_event["sector"] = str(raw["sector"])
        cycle_evidence.append(cycle_event)

    security_cases = sorted(
        (cases[t] for t in order),
        key=lambda case: (0 if case["role"] == "held" else 1, case["ticker"]),
    )
    return {
        "cycle_evidence": cycle_evidence,
        "security_cases": security_cases,
    }
