from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


CANDIDATE_STRUCTURED_VERSION = "candidate_memory_v1"
CANDIDATE_MEMORY_EVENT_TYPES: tuple[str, ...] = (
    "candidate_screen_hit",
    "candidate_watchlist",
    "candidate_rejected",
    "candidate_thesis",
)

_TICKER_RE = re.compile(r"\b([A-Z]{1,5}|\d{6})\b")
_SOURCE_RE = re.compile(
    r"(?:surfaced by|previously surfaced by|promoted candidate from)\s+"
    r"(?P<value>.*?)(?=\s+rank=|\s+repeat=|\s*\(|;|\.)",
    re.IGNORECASE,
)
_CHECKED_RE = re.compile(r"follow-up seen via\s+(?P<value>.*?)(?:\.|$)", re.IGNORECASE)
_REASON_RE = re.compile(r"Reason:\s*(?P<value>.*?)(?=\s+Risk:|\s+Next checks:|$)", re.IGNORECASE)
_RISK_RE = re.compile(r"Risk:\s*(?P<value>.*?)(?=\s+Next checks:|$)", re.IGNORECASE)
_RANK_RE = re.compile(r"\brank=(\d+)\b", re.IGNORECASE)
_REPEAT_RE = re.compile(r"\brepeat=(\d+)\b", re.IGNORECASE)
_SCORE_RE = re.compile(r"\bscore=([-+]?\d+(?:\.\d+)?)\b", re.IGNORECASE)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _payload_from_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    if isinstance(payload, dict):
        return dict(payload)
    return _json_dict(row.get("payload_json"))


def _non_empty_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        out = [str(item).strip() for item in value if str(item).strip()]
        return list(dict.fromkeys(out))
    token = str(value or "").strip()
    return [token] if token else []


def _split_tokens(value: str) -> list[str]:
    return [token.strip() for token in re.split(r"\s*,\s*", value or "") if token.strip()]


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None and str(value).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _summary_match(pattern: re.Pattern[str], summary: str) -> str:
    match = pattern.search(summary or "")
    if not match:
        return ""
    return str(match.group("value") or "").strip()


def _summary_int(pattern: re.Pattern[str], summary: str) -> int | None:
    match = pattern.search(summary or "")
    if not match:
        return None
    return _int_or_none(match.group(1))


def _summary_float(pattern: re.Pattern[str], summary: str) -> float | None:
    match = pattern.search(summary or "")
    if not match:
        return None
    return _float_or_none(match.group(1))


def _candidate_type(event_type: str, payload: dict[str, Any]) -> str:
    raw = str(payload.get("candidate_status") or "").strip().lower()
    if raw:
        return raw
    token = str(event_type or "").strip().lower()
    return token.removeprefix("candidate_") or "candidate"


def _candidate_date(row: dict[str, Any]) -> str:
    created_date = str(row.get("created_date") or "").strip()
    if created_date:
        return created_date
    raw = str(row.get("created_at") or "").strip()
    if "T" in raw:
        return raw.split("T", 1)[0]
    if " " in raw:
        return raw.split(" ", 1)[0]
    return raw[:10] if len(raw) >= 10 else raw


def _summary_ticker(summary: str) -> str:
    match = _TICKER_RE.search(summary or "")
    return str(match.group(1) or "").strip().upper() if match else ""


def _summary_source(summary: str) -> list[str]:
    source = _summary_match(_SOURCE_RE, summary)
    return _split_tokens(source)


def _summary_checked(summary: str) -> list[str]:
    checked = _summary_match(_CHECKED_RE, summary)
    return _split_tokens(checked)


def _quality(*, payload: dict[str, Any], parsed_from_summary: bool, structured: dict[str, Any]) -> str:
    if payload and not parsed_from_summary:
        useful = any(
            key in structured
            for key in ("t", "src", "rank", "score", "checked", "why", "risk", "workflow", "evidence")
        )
        return "payload_full" if useful else "payload_partial"
    if parsed_from_summary:
        useful = any(key in structured for key in ("t", "src", "rank", "score", "checked", "why", "risk"))
        return "summary_parsed" if useful else "summary_only"
    return "summary_only"


def build_structured_candidate_memory(row: dict[str, Any]) -> dict[str, Any]:
    """Builds a compact, loss-minimizing candidate memory view for model prompts."""
    payload = _payload_from_row(row)
    summary = str(row.get("summary") or "").strip()
    event_type = str(row.get("event_type") or "").strip().lower()
    evidence = payload.get("discovery_evidence") if isinstance(payload.get("discovery_evidence"), dict) else {}
    parsed_from_summary = not payload

    ticker = str(payload.get("ticker") or "").strip().upper() or _summary_ticker(summary)
    source_tools = _non_empty_list(payload.get("source_tools")) or _summary_source(summary)
    checked = _non_empty_list(payload.get("analyzed_by")) or _summary_checked(summary)
    rank = _int_or_none(payload.get("last_seen_rank"))
    if rank is None:
        rank = _summary_int(_RANK_RE, summary)
    discovery_count = _int_or_none(payload.get("discovery_count"))
    if discovery_count is None:
        discovery_count = _summary_int(_REPEAT_RE, summary)
    score = _float_or_none(evidence.get("score"))
    if score is None:
        score = _summary_float(_SCORE_RE, summary)
    raw_why = evidence.get("reason_for") if evidence.get("reason_for") is not None else evidence.get("reason")
    why = str(raw_why) if raw_why is not None else ""
    if not why.strip():
        why = _summary_match(_REASON_RE, summary)
    raw_risk = evidence.get("reason_risk")
    risk = str(raw_risk) if raw_risk is not None else ""
    if not risk.strip():
        risk = _summary_match(_RISK_RE, summary)

    structured: dict[str, Any] = {
        "v": CANDIDATE_STRUCTURED_VERSION,
        "event_id": str(row.get("event_id") or "").strip(),
        "type": _candidate_type(event_type, payload),
    }
    date_value = _candidate_date(row)
    if date_value:
        structured["d"] = date_value
    if ticker:
        structured["t"] = ticker
    if source_tools:
        structured["src"] = source_tools
    if rank is not None:
        structured["rank"] = rank
    if discovery_count is not None:
        structured["seen"] = discovery_count
    if score is not None:
        structured["score"] = score
    if checked:
        structured["checked"] = checked
    workflow_status = str(payload.get("workflow_status") or "").strip()
    if workflow_status:
        structured["workflow"] = workflow_status
    evidence_level = str(payload.get("evidence_level") or "").strip()
    if evidence_level:
        structured["evidence"] = evidence_level
    next_checks = _non_empty_list(payload.get("suggested_next_checks"))
    if next_checks:
        structured["next"] = next_checks
    skip_reasons = payload.get("skip_reasons")
    if isinstance(skip_reasons, dict) and skip_reasons:
        structured["skip"] = {str(key): value for key, value in skip_reasons.items()}
    if why:
        structured["why"] = why
    if risk:
        structured["risk"] = risk

    structured["quality"] = _quality(
        payload=payload,
        parsed_from_summary=parsed_from_summary,
        structured=structured,
    )
    return {key: value for key, value in structured.items() if value not in ("", [], {}, None)}


def enrich_candidate_memory_payload(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns payload_json with a structured prompt view attached."""
    payload = _payload_from_row(row)
    structured = build_structured_candidate_memory(row)
    enriched = dict(payload)
    enriched["structured_memory"] = structured
    return enriched, structured


def format_candidate_memory_prompt_line(row: dict[str, Any]) -> str:
    """Formats one candidate memory row as compact JSON for prompt injection."""
    payload = _payload_from_row(row)
    structured = payload.get("structured_memory") if isinstance(payload.get("structured_memory"), dict) else None
    if not structured:
        structured = build_structured_candidate_memory(row)
    else:
        structured = dict(structured)
        if row.get("event_id") and not structured.get("event_id"):
            structured["event_id"] = str(row.get("event_id") or "").strip()
        if row.get("created_date") and not structured.get("d"):
            structured["d"] = str(row.get("created_date") or "").strip()
    return "- " + json.dumps(structured, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class CandidateStructuredBackfillResult:
    scanned: int = 0
    updated: int = 0
    would_update: int = 0
    skipped: int = 0
    quality_counts: dict[str, int] = field(default_factory=dict)


def _candidate_backfill_rows(
    repo: Any,
    *,
    agent_id: str,
    trading_mode: str,
    tenant_id: str | None,
    limit: int,
    include_existing: bool,
) -> list[dict[str, Any]]:
    loader = getattr(repo, "candidate_memory_events_for_structured_backfill", None)
    if callable(loader):
        return list(
            loader(
                agent_id=agent_id,
                trading_mode=trading_mode,
                tenant_id=tenant_id,
                limit=limit,
                include_existing=include_existing,
            )
        )
    loader = getattr(repo, "candidate_memory_events", None)
    if callable(loader):
        return list(
            loader(
                agent_id=agent_id,
                trading_mode=trading_mode,
                tenant_id=tenant_id,
                limit=limit,
            )
        )
    return []


def _supports_bigquery_payload_batch(repo: Any) -> bool:
    return bool(getattr(repo, "dataset_fqn", "") and callable(getattr(repo, "execute", None)) and getattr(repo, "client", None) is not None)


def _batch_update_candidate_payload_json(
    repo: Any,
    *,
    tenant_id: str | None,
    updates: list[tuple[str, dict[str, Any]]],
    batch_size: int = 100,
) -> int:
    if not updates:
        return 0
    dataset_fqn = str(getattr(repo, "dataset_fqn") or "").strip()
    execute = getattr(repo, "execute", None)
    if not dataset_fqn or not callable(execute):
        return 0
    tenant = str(tenant_id or "").strip().lower() or "local"
    updated = 0
    clean_batch_size = max(1, min(int(batch_size or 100), 250))
    for offset in range(0, len(updates), clean_batch_size):
        chunk = updates[offset : offset + clean_batch_size]
        params: dict[str, Any] = {"tenant_id": tenant, "event_ids": [event_id for event_id, _ in chunk]}
        cases: list[str] = []
        for idx, (event_id, payload) in enumerate(chunk):
            event_key = f"event_id_{idx}"
            payload_key = f"payload_json_{idx}"
            params[event_key] = event_id
            params[payload_key] = json.dumps(payload, ensure_ascii=False, default=str)
            cases.append(f"WHEN @{event_key} THEN @{payload_key}")
        sql = f"""
        UPDATE `{dataset_fqn}.agent_memory_events`
        SET payload_json = CASE event_id
          {' '.join(cases)}
          ELSE payload_json
        END
        WHERE tenant_id = @tenant_id
          AND event_id IN UNNEST(@event_ids)
        """
        execute(sql, params)
        updated += len(chunk)
    return updated


def backfill_candidate_memory_structures(
    repo: Any,
    *,
    agent_ids: list[str],
    trading_mode: str = "paper",
    tenant_id: str | None = None,
    limit_per_agent: int = 1000,
    dry_run: bool = False,
    include_existing: bool = False,
) -> CandidateStructuredBackfillResult:
    """Backfills payload_json.structured_memory for existing candidate memories."""
    scanned = 0
    updated = 0
    would_update = 0
    skipped = 0
    quality_counts: dict[str, int] = {}
    update_fn = getattr(repo, "update_memory_event", None)
    for raw_agent_id in agent_ids:
        agent_id = str(raw_agent_id or "").strip()
        if not agent_id:
            continue
        rows = _candidate_backfill_rows(
            repo,
            agent_id=agent_id,
            trading_mode=str(trading_mode or "paper").strip().lower() or "paper",
            tenant_id=tenant_id,
                limit=max(1, int(limit_per_agent or 1000)),
                include_existing=include_existing,
            )
        batch_updates: list[tuple[str, dict[str, Any]]] = []
        can_batch_update = _supports_bigquery_payload_batch(repo) and not dry_run
        for row in rows:
            if not isinstance(row, dict):
                continue
            scanned += 1
            enriched, structured = enrich_candidate_memory_payload(row)
            quality = str(structured.get("quality") or "unknown")
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            current_payload = _payload_from_row(row)
            if current_payload.get("structured_memory") == structured and not include_existing:
                skipped += 1
                continue
            would_update += 1
            if dry_run:
                continue
            if can_batch_update:
                event_id = str(row.get("event_id") or "").strip()
                if event_id:
                    batch_updates.append((event_id, enriched))
                    continue
                skipped += 1
                continue
            if not callable(update_fn):
                skipped += 1
                continue
            try:
                score = float(row.get("score") if row.get("score") is not None else row.get("importance_score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            update_fn(
                event_id=str(row.get("event_id") or "").strip(),
                summary=str(row.get("summary") or "").strip(),
                payload=enriched,
                score=score,
                importance_score=row.get("importance_score"),
                outcome_score=row.get("outcome_score"),
                memory_tier=row.get("memory_tier"),
                expires_at=row.get("expires_at"),
                tenant_id=tenant_id,
            )
            updated += 1
        if batch_updates:
            updated += _batch_update_candidate_payload_json(
                repo,
                tenant_id=tenant_id,
                updates=batch_updates,
            )
    return CandidateStructuredBackfillResult(
        scanned=scanned,
        updated=updated,
        would_update=would_update,
        skipped=skipped,
        quality_counts=quality_counts,
    )
