from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any

from arena.models import utc_now

WATCH_ITEM_KINDS: tuple[str, ...] = ("macro_takeaway", "candidate", "post_exit")
WATCH_ITEM_STATUSES: tuple[str, ...] = ("active", "resolved", "archived")


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def _json_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return json.dumps(_json_value(value), ensure_ascii=False, separators=(",", ":"))


def _clean_text(value: Any, *, max_len: int = 180) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    out: list[str] = []
    for item in value:
        token = str(item or "").strip()
        if token and token not in out:
            out.append(token)
    return out


def normalize_watch_kind(value: Any) -> str:
    token = str(value or "").strip().lower()
    aliases = {
        "macro_research_takeaway": "macro_takeaway",
        "macro_research": "macro_takeaway",
        "research_takeaway": "macro_takeaway",
        "macro": "macro_takeaway",
        "candidate_watchlist": "candidate",
        "candidate_watch": "candidate",
        "watchlist_candidate": "candidate",
        "post_exit_watch": "post_exit",
        "exit_watch": "post_exit",
    }
    token = aliases.get(token, token)
    return token if token in WATCH_ITEM_KINDS else "candidate"


def normalize_watch_status(value: Any) -> str:
    token = str(value or "").strip().lower()
    aliases = {
        "open": "active",
        "monitoring": "active",
        "pending": "active",
        "closed": "resolved",
        "done": "resolved",
        "promoted": "resolved",
        "dismissed": "resolved",
        "expired": "archived",
    }
    token = aliases.get(token, token)
    return token if token in WATCH_ITEM_STATUSES else "active"


def watch_item_key(
    *,
    agent_id: str,
    watch_kind: str,
    ticker: str | None = None,
    source_doc_id: str | None = None,
    source_doc_ids: list[str] | None = None,
    title: str | None = None,
    summary: str | None = None,
    cycle_id: str | None = None,
    source_event: str | None = None,
    source_phase: str | None = None,
    intent_id: str | None = None,
    watch_key: str | None = None,
) -> str:
    explicit = str(watch_key or "").strip()
    if explicit:
        return explicit

    kind = normalize_watch_kind(watch_kind)
    agent = str(agent_id or "").strip().lower() or "agent"
    ticker_token = str(ticker or "").strip().upper()
    source_doc_token = str(source_doc_id or "").strip()
    doc_tokens = _clean_list(source_doc_ids)
    phase_token = str(source_phase or "").strip().lower()
    event_token = str(source_event or "").strip().lower()
    cycle_token = str(cycle_id or "").strip()
    intent_token = str(intent_id or "").strip()
    text_source = " | ".join(
        token
        for token in (
            source_doc_token,
            title or "",
            summary or "",
            cycle_token,
            phase_token,
            event_token,
            intent_token,
            ticker_token,
            ",".join(doc_tokens[:3]),
        )
        if str(token or "").strip()
    )
    digest = hashlib.sha1(text_source.encode("utf-8")).hexdigest()[:12]

    if kind == "macro_takeaway":
        doc_part = source_doc_token or (doc_tokens[0] if doc_tokens else "doc")
        return f"macro:{agent}:{doc_part}:{digest}"
    if kind == "post_exit":
        base = intent_token or ticker_token or source_doc_token or digest
        return f"post_exit:{agent}:{base}:{digest}"
    base = ticker_token or source_doc_token or digest
    return f"candidate:{agent}:{base}:{digest}"


def build_watch_record(
    *,
    agent_id: str,
    watch_kind: str,
    summary: str,
    title: str | None = None,
    status: str = "active",
    ticker: str | None = None,
    source_doc_id: str | None = None,
    source_doc_ids: list[str] | None = None,
    payload: dict[str, Any] | None = None,
    cycle_id: str | None = None,
    llm_call_id: str | None = None,
    source_phase: str | None = None,
    source_event: str | None = None,
    priority_score: float | None = None,
    time_horizon_days: int | None = None,
    next_review_at: datetime | None = None,
    expires_at: datetime | None = None,
    resolved_at: datetime | None = None,
    resolution: str | None = None,
    observed_return_krw: float | None = None,
    observed_return_ratio: float | None = None,
    observed_price_krw: float | None = None,
    observed_note: str | None = None,
    context_tags: dict[str, Any] | None = None,
    watch_key: str | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> dict[str, Any]:
    now = utc_now()
    watch_kind_token = normalize_watch_kind(watch_kind)
    status_token = normalize_watch_status(status)
    summary_text = _clean_text(summary, max_len=800)
    title_text = _clean_text(title or summary_text, max_len=160)
    record: dict[str, Any] = {
        "watch_key": watch_item_key(
            agent_id=agent_id,
            watch_kind=watch_kind_token,
            ticker=ticker,
            source_doc_id=source_doc_id,
            source_doc_ids=source_doc_ids,
            title=title_text,
            summary=summary_text,
            cycle_id=cycle_id,
            source_event=source_event,
            source_phase=source_phase,
            watch_key=watch_key,
        ),
        "tenant_id": None,
        "created_at": created_at or now,
        "updated_at": updated_at or now,
        "agent_id": str(agent_id or "").strip(),
        "watch_kind": watch_kind_token,
        "watch_status": status_token,
        "ticker": str(ticker or "").strip().upper() or None,
        "source_doc_id": str(source_doc_id or "").strip() or None,
        "source_doc_ids_json": _json_text(_clean_list(source_doc_ids) or None),
        "title": title_text or None,
        "summary": summary_text,
        "payload_json": _json_text(payload or {}),
        "cycle_id": str(cycle_id or "").strip() or None,
        "llm_call_id": str(llm_call_id or "").strip() or None,
        "source_phase": str(source_phase or "").strip() or None,
        "source_event": str(source_event or "").strip() or None,
        "priority_score": priority_score,
        "time_horizon_days": time_horizon_days,
        "next_review_at": next_review_at,
        "expires_at": expires_at,
        "resolved_at": resolved_at,
        "resolution": str(resolution or "").strip() or None,
        "observed_return_krw": observed_return_krw,
        "observed_return_ratio": observed_return_ratio,
        "observed_price_krw": observed_price_krw,
        "observed_note": _clean_text(observed_note, max_len=300) or None,
        "context_tags_json": _json_text(context_tags or {}),
    }
    return record


def parse_watch_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for field in ("payload_json", "source_doc_ids_json", "context_tags_json"):
        value = normalized.get(field)
        if isinstance(value, (dict, list)):
            continue
        if value is None:
            continue
        text = str(value or "").strip()
        if not text:
            normalized[field] = None
            continue
        try:
            normalized[field] = json.loads(text)
        except Exception:
            normalized[field] = text
    normalized["watch_kind"] = normalize_watch_kind(normalized.get("watch_kind"))
    normalized["watch_status"] = normalize_watch_status(normalized.get("watch_status"))
    normalized["summary"] = _clean_text(normalized.get("summary"), max_len=800)
    normalized["title"] = _clean_text(normalized.get("title"), max_len=160) or normalized["summary"]
    normalized["ticker"] = str(normalized.get("ticker") or "").strip().upper() or None
    normalized["source_doc_id"] = str(normalized.get("source_doc_id") or "").strip() or None
    return normalized


def watch_item_prompt_line(row: dict[str, Any]) -> str:
    item = parse_watch_row(row)
    kind = str(item.get("watch_kind") or "").strip().lower()
    status = str(item.get("watch_status") or "").strip().lower()
    title = str(item.get("title") or item.get("summary") or "").strip()
    ticker = str(item.get("ticker") or "").strip().upper()
    source_doc_id = str(item.get("source_doc_id") or "").strip()
    payload = item.get("payload_json") if isinstance(item.get("payload_json"), dict) else {}

    if kind == "macro_takeaway":
        indicators = payload.get("watch_indicators") if isinstance(payload, dict) else []
        indicator_text = ", ".join(_clean_list(indicators)[:4])
        doc_text = source_doc_id or ",".join(_clean_list(item.get("source_doc_ids_json"))[:3]) or "-"
        suffix = f" watch={indicator_text}" if indicator_text else ""
        horizon = payload.get("horizon_days") if isinstance(payload, dict) else None
        horizon_text = f" horizon={horizon}d" if horizon is not None else ""
        return f"[macro/{status}] {title} | doc={doc_text}{suffix}{horizon_text}"

    if kind == "post_exit":
        resolution = str(item.get("resolution") or "").strip()
        return f"[post_exit/{status}] {ticker or '-'} {title}" + (f" | resolution={resolution}" if resolution else "")

    reason = ""
    if isinstance(payload, dict):
        reason = str(payload.get("summary") or payload.get("reason") or "").strip()
    source_docs = _clean_list(item.get("source_doc_ids_json"))
    src_text = source_doc_id or (source_docs[0] if source_docs else "")
    bits = [f"[candidate/{status}] {ticker or '-'} {title}"]
    if src_text:
        bits.append(f"src={src_text}")
    if reason:
        bits.append(reason)
    return " | ".join(bits)


def compress_watch_rows(rows: list[dict[str, Any]], *, limit: int = 6) -> str:
    cleaned = [row for row in rows if isinstance(row, dict)]
    if not cleaned:
        return ""
    lines = [watch_item_prompt_line(row) for row in cleaned[: max(1, min(int(limit), 12))]]
    return "\n".join(f"- {line}" for line in lines if str(line or "").strip())
