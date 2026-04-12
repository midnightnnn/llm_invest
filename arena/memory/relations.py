from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from arena.memory.graph import board_post_node_id, memory_event_node_id, research_briefing_node_id
from arena.models import BoardPost, MemoryEvent

RELATION_EXTRACTION_VERSION = "deterministic_v1"

ALLOWED_RELATION_PREDICATES = frozenset(
    {
        "mentions",
        "contains",
        "supports",
        "contradicts",
        "risk_to",
        "caused_by",
        "leads_to",
        "similar_setup",
        "invalidates",
        "outcome_of",
    }
)

_SLUG_RE = re.compile(r"[^a-z0-9_.:-]+")
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,14}$")
_SOURCE_NODE_PREFIXES = ("mem:", "post:", "brief:", "intent:", "exec:")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _lower(value: Any) -> str:
    return _text(value).lower()


def _upper(value: Any) -> str:
    return _text(value).upper()


def _trim(value: Any, *, max_len: int = 240) -> str:
    text = " ".join(_text(value).split())
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "..."


def _json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    text = _text(raw)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _list_tokens(value: Any) -> list[str]:
    if isinstance(value, str):
        values = re.split(r"[,\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    out: list[str] = []
    for item in values:
        token = _text(item)
        if token and token not in out:
            out.append(token)
    return out


def _slug(value: Any) -> str:
    text = _lower(value)
    text = _SLUG_RE.sub("_", text).strip("_")
    return text[:96]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ticker_node_id(ticker: str) -> str:
    token = _upper(ticker)
    return f"ticker:{token}" if token else ""


def semantic_entity_node_id(entity_type: str, label: str) -> str:
    typ = _slug(entity_type)
    slug = _slug(label)
    return f"entity:{typ}:{slug}" if typ and slug else ""


def relation_triple_id(
    *,
    source_table: str,
    source_id: str,
    subject_node_id: str,
    predicate: str,
    object_node_id: str,
) -> str:
    raw = "|".join(
        [
            _lower(source_table),
            _text(source_id),
            _text(subject_node_id),
            _lower(predicate),
            _text(object_node_id),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]
    return f"triple:{digest}"


def _entity_node_id(entity_type: str, label: str) -> str:
    if _lower(entity_type) == "ticker":
        return ticker_node_id(label)
    return semantic_entity_node_id(entity_type, label)


def _entity_type_for_node(node_id: str, fallback: str) -> str:
    if node_id.startswith("ticker:"):
        return "ticker"
    return _lower(fallback) or "entity"


def _make_relation_triple(
    *,
    source_table: str,
    source_id: str,
    source_node_id: str,
    source_label: str,
    source_created_at: Any,
    agent_id: str | None,
    trading_mode: str,
    cycle_id: str | None,
    subject_node_id: str,
    subject_label: str,
    subject_type: str,
    predicate: str,
    object_node_id: str,
    object_label: str,
    object_type: str,
    evidence_text: str,
    confidence: float = 1.0,
    extraction_method: str = "deterministic",
    extraction_version: str = RELATION_EXTRACTION_VERSION,
    status: str = "accepted",
    detail_json: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    clean_predicate = _lower(predicate)
    if clean_predicate not in ALLOWED_RELATION_PREDICATES:
        return None
    if not source_id or not subject_node_id or not object_node_id:
        return None
    if subject_node_id == object_node_id:
        return None
    created_at = source_created_at or _utc_now()
    triple_id = relation_triple_id(
        source_table=source_table,
        source_id=source_id,
        subject_node_id=subject_node_id,
        predicate=clean_predicate,
        object_node_id=object_node_id,
    )
    detail = dict(detail_json or {})
    detail.setdefault("source_label", _trim(source_label, max_len=180))
    return {
        "triple_id": triple_id,
        "created_at": created_at,
        "source_table": _text(source_table),
        "source_id": _text(source_id),
        "source_node_id": _text(source_node_id),
        "source_created_at": source_created_at,
        "agent_id": _text(agent_id) or None,
        "trading_mode": _lower(trading_mode) or "paper",
        "cycle_id": _text(cycle_id) or None,
        "subject_node_id": _text(subject_node_id),
        "subject_label": _trim(subject_label, max_len=180),
        "subject_type": _lower(subject_type) or "entity",
        "predicate": clean_predicate,
        "object_node_id": _text(object_node_id),
        "object_label": _trim(object_label, max_len=180),
        "object_type": _lower(object_type) or "entity",
        "confidence": max(0.0, min(float(confidence), 1.0)),
        "evidence_text": _trim(evidence_text, max_len=360),
        "extraction_method": _text(extraction_method) or "deterministic",
        "extraction_version": _text(extraction_version) or RELATION_EXTRACTION_VERSION,
        "status": _lower(status) or "accepted",
        "detail_json": detail,
    }


def make_relation_triple(
    *,
    source_table: str,
    source_id: str,
    source_node_id: str,
    source_label: str,
    source_created_at: Any,
    agent_id: str | None,
    trading_mode: str,
    cycle_id: str | None,
    subject_node_id: str,
    subject_label: str,
    subject_type: str,
    predicate: str,
    object_node_id: str,
    object_label: str,
    object_type: str,
    evidence_text: str,
    confidence: float = 1.0,
    extraction_method: str = "deterministic",
    extraction_version: str = RELATION_EXTRACTION_VERSION,
    status: str = "accepted",
    detail_json: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Builds one source-grounded relation triple row for storage."""
    return _make_relation_triple(
        source_table=source_table,
        source_id=source_id,
        source_node_id=source_node_id,
        source_label=source_label,
        source_created_at=source_created_at,
        agent_id=agent_id,
        trading_mode=trading_mode,
        cycle_id=cycle_id,
        subject_node_id=subject_node_id,
        subject_label=subject_label,
        subject_type=subject_type,
        predicate=predicate,
        object_node_id=object_node_id,
        object_label=object_label,
        object_type=object_type,
        evidence_text=evidence_text,
        confidence=confidence,
        extraction_method=extraction_method,
        extraction_version=extraction_version,
        status=status,
        detail_json=detail_json,
    )


def _dedupe_triples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        triple_id = _text(row.get("triple_id"))
        if not triple_id or triple_id in seen:
            continue
        seen.add(triple_id)
        out.append(row)
    return out


def _memory_payload(value: MemoryEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, MemoryEvent):
        return dict(value.payload or {})
    return _json_object(value.get("payload_json")) if isinstance(value, dict) else {}


def _memory_context_tags(value: MemoryEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, MemoryEvent):
        return dict(value.context_tags or {})
    if not isinstance(value, dict):
        return {}
    tags = value.get("context_tags")
    if isinstance(tags, dict):
        return dict(tags)
    return _json_object(value.get("context_tags_json"))


def _memory_value(value: MemoryEvent | dict[str, Any], field: str) -> Any:
    return getattr(value, field, None) if isinstance(value, MemoryEvent) else value.get(field)


def _extract_structured_tickers(*values: Any) -> list[str]:
    tickers: list[str] = []
    for value in values:
        for token in _list_tokens(value):
            clean = _upper(token)
            if _TICKER_RE.match(clean) and clean not in tickers:
                tickers.append(clean)
    return tickers


def build_memory_event_relation_triples(value: MemoryEvent | dict[str, Any]) -> list[dict[str, Any]]:
    event_id = _text(_memory_value(value, "event_id"))
    if not event_id:
        return []
    payload = _memory_payload(value)
    tags = _memory_context_tags(value)
    source_node_id = _text(_memory_value(value, "graph_node_id")) or memory_event_node_id(event_id)
    summary = _text(_memory_value(value, "summary"))
    created_at = _memory_value(value, "created_at")
    agent_id = _text(_memory_value(value, "agent_id")) or None
    trading_mode = _lower(_memory_value(value, "trading_mode")) or "paper"
    cycle_id = _text(payload.get("cycle_id") or payload.get("intent", {}).get("cycle_id"))
    subject_label = f"memory_event:{event_id}"
    evidence = summary or subject_label
    triples: list[dict[str, Any]] = []

    tickers = _extract_structured_tickers(
        payload.get("ticker"),
        payload.get("tickers"),
        payload.get("intent", {}).get("ticker"),
        tags.get("tickers"),
        tags.get("canonical_tickers"),
        tags.get("derived_tickers"),
        _memory_value(value, "ticker"),
        _memory_value(value, "tickers"),
    )
    for ticker in tickers:
        row = _make_relation_triple(
            source_table="agent_memory_events",
            source_id=event_id,
            source_node_id=source_node_id,
            source_label=summary or subject_label,
            source_created_at=created_at,
            agent_id=agent_id,
            trading_mode=trading_mode,
            cycle_id=cycle_id,
            subject_node_id=source_node_id,
            subject_label=subject_label,
            subject_type="passage",
            predicate="contains",
            object_node_id=ticker_node_id(ticker),
            object_label=ticker,
            object_type="ticker",
            evidence_text=evidence,
            detail_json={"source": "memory_event", "field": "ticker"},
        )
        if row:
            triples.append(row)

    for object_type, label in [
        ("strategy_tag", _memory_value(value, "primary_strategy_tag") or tags.get("strategy_tag")),
        ("sector", _memory_value(value, "primary_sector") or tags.get("sector")),
        ("regime", _memory_value(value, "primary_regime") or tags.get("regime")),
    ]:
        object_label = _text(label)
        object_node_id = _entity_node_id(object_type, object_label)
        row = _make_relation_triple(
            source_table="agent_memory_events",
            source_id=event_id,
            source_node_id=source_node_id,
            source_label=summary or subject_label,
            source_created_at=created_at,
            agent_id=agent_id,
            trading_mode=trading_mode,
            cycle_id=cycle_id,
            subject_node_id=source_node_id,
            subject_label=subject_label,
            subject_type="passage",
            predicate="contains",
            object_node_id=object_node_id,
            object_label=object_label,
            object_type=object_type,
            evidence_text=evidence,
            detail_json={"source": "memory_event", "field": object_type},
        )
        if row:
            triples.append(row)

    return _dedupe_triples(triples)


def build_board_post_relation_triples(value: BoardPost | dict[str, Any]) -> list[dict[str, Any]]:
    post_id = _text(value.post_id) if isinstance(value, BoardPost) else _text(value.get("post_id"))
    if not post_id:
        return []
    source_node_id = board_post_node_id(post_id)
    created_at = value.created_at if isinstance(value, BoardPost) else value.get("created_at")
    agent_id = _text(value.agent_id) if isinstance(value, BoardPost) else _text(value.get("agent_id")) or None
    trading_mode = _lower(value.trading_mode) if isinstance(value, BoardPost) else _lower(value.get("trading_mode"))
    cycle_id = _text(value.cycle_id) if isinstance(value, BoardPost) else _text(value.get("cycle_id"))
    title = _text(value.title) if isinstance(value, BoardPost) else _text(value.get("title"))
    body = _text(value.body) if isinstance(value, BoardPost) else _text(value.get("body"))
    draft_summary = _text(value.draft_summary) if isinstance(value, BoardPost) else _text(value.get("draft_summary"))
    tickers = value.tickers if isinstance(value, BoardPost) else value.get("tickers")
    evidence = _trim(" - ".join(part for part in [title, draft_summary, body] if part), max_len=360)
    triples: list[dict[str, Any]] = []
    for ticker in _extract_structured_tickers(tickers):
        row = _make_relation_triple(
            source_table="board_posts",
            source_id=post_id,
            source_node_id=source_node_id,
            source_label=title or f"board_post:{post_id}",
            source_created_at=created_at,
            agent_id=agent_id,
            trading_mode=trading_mode or "paper",
            cycle_id=cycle_id,
            subject_node_id=source_node_id,
            subject_label=f"board_post:{post_id}",
            subject_type="passage",
            predicate="mentions",
            object_node_id=ticker_node_id(ticker),
            object_label=ticker,
            object_type="ticker",
            evidence_text=evidence or title or f"board_post:{post_id}",
            detail_json={"source": "board_post", "field": "tickers"},
        )
        if row:
            triples.append(row)
    return _dedupe_triples(triples)


def build_research_briefing_relation_triples(row: dict[str, Any]) -> list[dict[str, Any]]:
    briefing_id = _text(row.get("briefing_id"))
    if not briefing_id:
        return []
    source_node_id = research_briefing_node_id(briefing_id)
    created_at = row.get("created_at")
    ticker = _upper(row.get("ticker"))
    category = _lower(row.get("category"))
    headline = _text(row.get("headline"))
    summary = _text(row.get("summary"))
    evidence = _trim(" - ".join(part for part in [headline, summary] if part), max_len=360)
    triples: list[dict[str, Any]] = []
    for object_type, predicate, label in [
        ("ticker", "mentions", ticker),
        ("research_category", "contains", category),
    ]:
        object_label = _text(label)
        object_node_id = _entity_node_id(object_type, object_label)
        row_out = _make_relation_triple(
            source_table="research_briefings",
            source_id=briefing_id,
            source_node_id=source_node_id,
            source_label=headline or f"research_briefing:{briefing_id}",
            source_created_at=created_at,
            agent_id=None,
            trading_mode=_lower(row.get("trading_mode")) or "paper",
            cycle_id=_text(row.get("cycle_id")),
            subject_node_id=source_node_id,
            subject_label=f"research_briefing:{briefing_id}",
            subject_type="passage",
            predicate=predicate,
            object_node_id=object_node_id,
            object_label=object_label,
            object_type=object_type,
            evidence_text=evidence or headline or f"research_briefing:{briefing_id}",
            detail_json={"source": "research_briefing", "field": object_type},
        )
        if row_out:
            triples.append(row_out)
    return _dedupe_triples(triples)


def _should_project_entity_node(node_id: str, entity_type: str) -> bool:
    if not node_id or node_id.startswith(_SOURCE_NODE_PREFIXES):
        return False
    return _lower(entity_type) not in {"passage", "memory_event", "board_post", "research_briefing"}


def build_relation_triple_graph_nodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        for side in ("subject", "object"):
            node_id = _text(row.get(f"{side}_node_id"))
            entity_type = _entity_type_for_node(node_id, _text(row.get(f"{side}_type")))
            label = _text(row.get(f"{side}_label"))
            if not _should_project_entity_node(node_id, entity_type) or node_id in seen:
                continue
            seen.add(node_id)
            nodes.append(
                {
                    "node_id": node_id,
                    "created_at": row.get("created_at") or row.get("source_created_at") or _utc_now(),
                    "node_kind": "semantic_entity",
                    "source_table": "memory_relation_triples",
                    "source_id": _text(row.get("triple_id")),
                    "agent_id": _text(row.get("agent_id")) or None,
                    "trading_mode": _lower(row.get("trading_mode")) or "paper",
                    "cycle_id": _text(row.get("cycle_id")) or None,
                    "summary": label or node_id,
                    "ticker": _upper(label) if entity_type == "ticker" else None,
                    "memory_tier": None,
                    "primary_regime": None,
                    "context_tags_json": {"entity_type": entity_type},
                    "payload_json": {"label": label or node_id, "entity_type": entity_type},
                }
            )
    return nodes


def build_relation_triple_graph_edges(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        triple_id = _text(row.get("triple_id"))
        subject_node_id = _text(row.get("subject_node_id"))
        object_node_id = _text(row.get("object_node_id"))
        if not triple_id or not subject_node_id or not object_node_id or subject_node_id == object_node_id:
            continue
        confidence = row.get("confidence")
        try:
            strength = max(0.0, min(float(confidence), 1.0))
        except (TypeError, ValueError):
            strength = 1.0
        base = {
            "created_at": row.get("created_at") or row.get("source_created_at") or _utc_now(),
            "trading_mode": _lower(row.get("trading_mode")) or "paper",
            "cycle_id": _text(row.get("cycle_id")) or None,
            "edge_strength": strength,
            "confidence": strength,
            "causal_chain_id": None,
            "detail_json": {
                "triple_id": triple_id,
                "predicate": _lower(row.get("predicate")),
                "source_table": _text(row.get("source_table")),
                "source_id": _text(row.get("source_id")),
                "evidence_text": _text(row.get("evidence_text")),
                "extraction_method": _text(row.get("extraction_method")),
                "extraction_version": _text(row.get("extraction_version")),
            },
        }
        edge_id = f"edge:relation:{triple_id}"
        if edge_id not in seen:
            seen.add(edge_id)
            edges.append(
                {
                    **base,
                    "edge_id": edge_id,
                    "from_node_id": subject_node_id,
                    "to_node_id": object_node_id,
                    "edge_type": _lower(row.get("predicate")).upper(),
                }
            )
        source_node_id = _text(row.get("source_node_id"))
        for target_role, target_node_id in [("subject", subject_node_id), ("object", object_node_id)]:
            if not source_node_id or source_node_id == target_node_id:
                continue
            context_edge_id = f"edge:relation_context:{target_role}:{triple_id}"
            if context_edge_id in seen:
                continue
            seen.add(context_edge_id)
            edges.append(
                {
                    **base,
                    "edge_id": context_edge_id,
                    "from_node_id": source_node_id,
                    "to_node_id": target_node_id,
                    "edge_type": "CONTAINS",
                    "edge_strength": min(strength, 0.9),
                }
            )
    return edges


def relation_triples_to_graph_projection(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = [
        row
        for row in rows
        if _lower(row.get("status") or "accepted") == "accepted"
    ]
    return build_relation_triple_graph_nodes(accepted), build_relation_triple_graph_edges(accepted)
