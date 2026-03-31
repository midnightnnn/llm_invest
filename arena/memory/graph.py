from __future__ import annotations

import json
from typing import Any

from arena.models import BoardPost, ExecutionReport, MemoryEvent, OrderIntent


def _text(value: Any) -> str:
    return str(value or "").strip()


def _upper(value: Any) -> str:
    return _text(value).upper()


def _lower(value: Any) -> str:
    return _text(value).lower()


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
    if not isinstance(value, list):
        return []
    return [_text(item) for item in value if _text(item)]


def memory_event_node_id(event_id: str) -> str:
    token = _text(event_id)
    return f"mem:{token}" if token else ""


def order_intent_node_id(intent_id: str) -> str:
    token = _text(intent_id)
    return f"intent:{token}" if token else ""


def execution_report_node_id(order_id: str) -> str:
    token = _text(order_id)
    return f"exec:{token}" if token else ""


def board_post_node_id(post_id: str) -> str:
    token = _text(post_id)
    return f"post:{token}" if token else ""


def research_briefing_node_id(briefing_id: str) -> str:
    token = _text(briefing_id)
    return f"brief:{token}" if token else ""


def _memory_payload(value: MemoryEvent | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, MemoryEvent):
        return dict(value.payload or {})
    return _json_object(value.get("payload_json")) if isinstance(value, dict) else {}


def _memory_context_tags(value: MemoryEvent | dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(value, MemoryEvent):
        return dict(value.context_tags or {}) or None
    if not isinstance(value, dict):
        return None
    tags = value.get("context_tags")
    if isinstance(tags, dict):
        return dict(tags) or None
    parsed = _json_object(value.get("context_tags_json"))
    return parsed or None


def _memory_cycle_id(value: MemoryEvent | dict[str, Any]) -> str:
    payload = _memory_payload(value)
    return _text(payload.get("cycle_id") or payload.get("intent", {}).get("cycle_id"))


def _memory_intent_id(value: MemoryEvent | dict[str, Any]) -> str:
    payload = _memory_payload(value)
    return _text(payload.get("intent", {}).get("intent_id"))


def _memory_order_id(value: MemoryEvent | dict[str, Any]) -> str:
    payload = _memory_payload(value)
    return _text(payload.get("report", {}).get("order_id"))


def _memory_ticker(value: MemoryEvent | dict[str, Any]) -> str:
    payload = _memory_payload(value)
    ticker = _text(payload.get("intent", {}).get("ticker"))
    if ticker:
        return ticker.upper()
    if isinstance(value, dict):
        return _upper(value.get("ticker"))
    return ""


def infer_memory_event_causal_chain_id(value: MemoryEvent | dict[str, Any]) -> str:
    explicit = _text(value.causal_chain_id) if isinstance(value, MemoryEvent) else _text(value.get("causal_chain_id"))
    if explicit:
        return explicit
    intent_id = _memory_intent_id(value)
    if intent_id:
        return f"chain:intent:{intent_id}"
    cycle_id = _memory_cycle_id(value)
    agent_id = _text(value.agent_id) if isinstance(value, MemoryEvent) else _text(value.get("agent_id"))
    if cycle_id and agent_id:
        return f"chain:cycle:{agent_id}:{cycle_id}"
    return ""


def ensure_memory_event_graph_ids(event: MemoryEvent) -> MemoryEvent:
    if not _text(event.graph_node_id):
        event.graph_node_id = memory_event_node_id(event.event_id)
    if not _text(event.causal_chain_id):
        inferred = infer_memory_event_causal_chain_id(event)
        event.causal_chain_id = inferred or None
    return event


def build_memory_event_graph_node(value: MemoryEvent | dict[str, Any]) -> dict[str, Any]:
    payload = _memory_payload(value)
    context_tags = _memory_context_tags(value)
    event_id = _text(value.event_id) if isinstance(value, MemoryEvent) else _text(value.get("event_id"))
    created_at = value.created_at if isinstance(value, MemoryEvent) else value.get("created_at")
    agent_id = _text(value.agent_id) if isinstance(value, MemoryEvent) else _text(value.get("agent_id")) or None
    trading_mode = (
        _lower(value.trading_mode) if isinstance(value, MemoryEvent) else _lower(value.get("trading_mode"))
    ) or "paper"
    summary = _text(value.summary) if isinstance(value, MemoryEvent) else _text(value.get("summary")) or None
    memory_tier = (
        _lower(value.memory_tier) if isinstance(value, MemoryEvent) else _lower(value.get("memory_tier"))
    ) or None
    primary_regime = (
        _lower(value.primary_regime) if isinstance(value, MemoryEvent) else _lower(value.get("primary_regime"))
    ) or None
    node_id = (
        _text(value.graph_node_id) if isinstance(value, MemoryEvent) else _text(value.get("graph_node_id"))
    ) or memory_event_node_id(event_id)
    return {
        "node_id": node_id,
        "created_at": created_at,
        "node_kind": "memory_event",
        "source_table": "agent_memory_events",
        "source_id": event_id,
        "agent_id": agent_id,
        "trading_mode": trading_mode,
        "cycle_id": _memory_cycle_id(value) or None,
        "summary": summary,
        "ticker": _memory_ticker(value) or None,
        "memory_tier": memory_tier,
        "primary_regime": primary_regime,
        "context_tags_json": context_tags,
        "payload_json": payload or None,
    }


def build_memory_event_graph_edges(value: MemoryEvent | dict[str, Any]) -> list[dict[str, Any]]:
    payload = _memory_payload(value)
    event_type = _lower(value.event_type) if isinstance(value, MemoryEvent) else _lower(value.get("event_type"))
    created_at = value.created_at if isinstance(value, MemoryEvent) else value.get("created_at")
    trading_mode = (
        _lower(value.trading_mode) if isinstance(value, MemoryEvent) else _lower(value.get("trading_mode"))
    ) or "paper"
    target_event_id = _text(value.event_id) if isinstance(value, MemoryEvent) else _text(value.get("event_id"))
    target_node_id = (
        _text(value.graph_node_id) if isinstance(value, MemoryEvent) else _text(value.get("graph_node_id"))
    ) or memory_event_node_id(target_event_id)
    causal_chain_id = infer_memory_event_causal_chain_id(value) or None
    cycle_id = _memory_cycle_id(value) or None
    edges: list[dict[str, Any]] = []

    source_edge_type = "ABSTRACTED_TO" if event_type == "strategy_reflection" else "REFERENCES"
    for source_event_id in _list_tokens(payload.get("source_event_ids")):
        edges.append(
            {
                "edge_id": f"edge:{source_edge_type.lower()}:{source_event_id}:{target_event_id}",
                "created_at": created_at,
                "trading_mode": trading_mode,
                "cycle_id": cycle_id,
                "from_node_id": memory_event_node_id(source_event_id),
                "to_node_id": target_node_id,
                "edge_type": source_edge_type,
                "edge_strength": 1.0,
                "confidence": 1.0,
                "causal_chain_id": causal_chain_id,
                "detail_json": {"source": event_type or "memory_event", "via": "source_event_ids"},
            }
        )

    for source_post_id in _list_tokens(payload.get("source_post_ids")):
        edges.append(
            {
                "edge_id": f"edge:informed_by:post:{source_post_id}:{target_event_id}",
                "created_at": created_at,
                "trading_mode": trading_mode,
                "cycle_id": cycle_id,
                "from_node_id": board_post_node_id(source_post_id),
                "to_node_id": target_node_id,
                "edge_type": "INFORMED_BY",
                "edge_strength": 0.95,
                "confidence": 1.0,
                "causal_chain_id": causal_chain_id,
                "detail_json": {"source": event_type or "memory_event", "via": "source_post_ids"},
            }
        )

    for source_briefing_id in _list_tokens(payload.get("source_briefing_ids")):
        edges.append(
            {
                "edge_id": f"edge:informed_by:brief:{source_briefing_id}:{target_event_id}",
                "created_at": created_at,
                "trading_mode": trading_mode,
                "cycle_id": cycle_id,
                "from_node_id": research_briefing_node_id(source_briefing_id),
                "to_node_id": target_node_id,
                "edge_type": "INFORMED_BY",
                "edge_strength": 1.0,
                "confidence": 1.0,
                "causal_chain_id": causal_chain_id,
                "detail_json": {"source": event_type or "memory_event", "via": "source_briefing_ids"},
            }
        )

    intent_id = _memory_intent_id(value)
    if intent_id:
        edges.append(
            {
                "edge_id": f"edge:precedes:intent:{intent_id}:{target_event_id}",
                "created_at": created_at,
                "trading_mode": trading_mode,
                "cycle_id": cycle_id,
                "from_node_id": order_intent_node_id(intent_id),
                "to_node_id": target_node_id,
                "edge_type": "PRECEDES",
                "edge_strength": 0.9,
                "confidence": 1.0,
                "causal_chain_id": causal_chain_id,
                "detail_json": {"source": event_type or "memory_event", "via": "payload.intent.intent_id"},
            }
        )

    order_id = _memory_order_id(value)
    if order_id:
        edges.append(
            {
                "edge_id": f"edge:resulted_in:exec:{order_id}:{target_event_id}",
                "created_at": created_at,
                "trading_mode": trading_mode,
                "cycle_id": cycle_id,
                "from_node_id": execution_report_node_id(order_id),
                "to_node_id": target_node_id,
                "edge_type": "RESULTED_IN",
                "edge_strength": 1.0,
                "confidence": 1.0,
                "causal_chain_id": causal_chain_id,
                "detail_json": {"source": event_type or "memory_event", "via": "payload.report.order_id"},
            }
        )

    return edges


def build_order_intent_graph_node(intent: OrderIntent, decision: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = intent.model_dump(mode="json")
    if isinstance(decision, dict) and decision:
        payload["risk_decision"] = dict(decision)
    rationale = _text(intent.rationale)
    summary = f"{intent.side.value} {intent.ticker} qty={intent.quantity:.4f}"
    if rationale:
        summary += f" rationale={rationale[:160]}"
    return {
        "node_id": order_intent_node_id(intent.intent_id),
        "created_at": intent.created_at,
        "node_kind": "order_intent",
        "source_table": "agent_order_intents",
        "source_id": intent.intent_id,
        "agent_id": _text(intent.agent_id) or None,
        "trading_mode": _lower(intent.trading_mode) or "paper",
        "cycle_id": _text(intent.cycle_id) or None,
        "summary": summary,
        "ticker": _upper(intent.ticker) or None,
        "memory_tier": None,
        "primary_regime": None,
        "context_tags_json": None,
        "payload_json": payload,
    }


def build_execution_report_graph_node(intent: OrderIntent, report: ExecutionReport) -> dict[str, Any]:
    message = _text(report.message)
    summary = (
        f"{report.status.value} {intent.side.value} {intent.ticker} "
        f"filled={float(report.filled_qty or 0.0):.4f} avg={float(report.avg_price_krw or 0.0):.0f}"
    )
    if message:
        summary += f" message={message[:140]}"
    payload = {
        "intent": intent.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
    }
    return {
        "node_id": execution_report_node_id(report.order_id),
        "created_at": report.created_at,
        "node_kind": "execution_report",
        "source_table": "execution_reports",
        "source_id": _text(report.order_id),
        "agent_id": _text(intent.agent_id) or None,
        "trading_mode": _lower(intent.trading_mode) or "paper",
        "cycle_id": _text(intent.cycle_id) or None,
        "summary": summary,
        "ticker": _upper(intent.ticker) or None,
        "memory_tier": None,
        "primary_regime": None,
        "context_tags_json": None,
        "payload_json": payload,
    }


def build_intent_execution_edge(intent: OrderIntent, report: ExecutionReport) -> dict[str, Any]:
    causal_chain_id = f"chain:intent:{intent.intent_id}" if _text(intent.intent_id) else None
    return {
        "edge_id": f"edge:executed_as:{intent.intent_id}:{report.order_id}",
        "created_at": report.created_at,
        "trading_mode": _lower(intent.trading_mode) or "paper",
        "cycle_id": _text(intent.cycle_id) or None,
        "from_node_id": order_intent_node_id(intent.intent_id),
        "to_node_id": execution_report_node_id(report.order_id),
        "edge_type": "EXECUTED_AS",
        "edge_strength": 1.0,
        "confidence": 1.0,
        "causal_chain_id": causal_chain_id,
        "detail_json": {"status": report.status.value},
    }


def build_board_post_graph_node(value: BoardPost | dict[str, Any]) -> dict[str, Any]:
    post_id = _text(value.post_id) if isinstance(value, BoardPost) else _text(value.get("post_id"))
    title = _text(value.title) if isinstance(value, BoardPost) else _text(value.get("title"))
    draft_summary = _text(value.draft_summary) if isinstance(value, BoardPost) else _text(value.get("draft_summary"))
    body = _text(value.body) if isinstance(value, BoardPost) else _text(value.get("body"))
    summary = draft_summary or title or body[:180] or None
    tickers = list(value.tickers) if isinstance(value, BoardPost) else list(value.get("tickers") or [])
    payload = {
        "title": title,
        "body": body,
        "draft_summary": draft_summary or None,
        "tickers": [_upper(token) for token in tickers if _text(token)],
    }
    return {
        "node_id": board_post_node_id(post_id),
        "created_at": value.created_at if isinstance(value, BoardPost) else value.get("created_at"),
        "node_kind": "board_post",
        "source_table": "board_posts",
        "source_id": post_id,
        "agent_id": (_text(value.agent_id) if isinstance(value, BoardPost) else _text(value.get("agent_id"))) or None,
        "trading_mode": (
            _lower(value.trading_mode) if isinstance(value, BoardPost) else _lower(value.get("trading_mode"))
        ) or "paper",
        "cycle_id": (_text(value.cycle_id) if isinstance(value, BoardPost) else _text(value.get("cycle_id"))) or None,
        "summary": summary,
        "ticker": (_upper(tickers[0]) if tickers else "") or None,
        "memory_tier": None,
        "primary_regime": None,
        "context_tags_json": None,
        "payload_json": payload,
    }


def build_research_briefing_graph_node(row: dict[str, Any]) -> dict[str, Any]:
    summary = _text(row.get("summary"))
    headline = _text(row.get("headline"))
    node_summary = headline or summary[:180] or None
    return {
        "node_id": research_briefing_node_id(_text(row.get("briefing_id"))),
        "created_at": row.get("created_at"),
        "node_kind": "research_briefing",
        "source_table": "research_briefings",
        "source_id": _text(row.get("briefing_id")),
        "agent_id": None,
        "trading_mode": _lower(row.get("trading_mode")) or "paper",
        "cycle_id": None,
        "summary": node_summary,
        "ticker": _upper(row.get("ticker")) or None,
        "memory_tier": None,
        "primary_regime": None,
        "context_tags_json": None,
        "payload_json": {
            "category": _lower(row.get("category")) or None,
            "headline": headline or None,
            "summary": summary or None,
            "sources": row.get("sources"),
        },
    }


def build_execution_report_row_graph_node(row: dict[str, Any], *, trading_mode: str = "paper") -> dict[str, Any]:
    ticker = _upper(row.get("ticker"))
    side = _upper(row.get("side"))
    status = _upper(row.get("status"))
    summary = (
        f"{status or 'EXECUTION'} {side or '?'} {ticker or '?'} "
        f"filled={float(row.get('filled_qty') or 0.0):.4f}"
    )
    return {
        "node_id": execution_report_node_id(_text(row.get("order_id"))),
        "created_at": row.get("created_at"),
        "node_kind": "execution_report",
        "source_table": "execution_reports",
        "source_id": _text(row.get("order_id")),
        "agent_id": _text(row.get("agent_id")) or None,
        "trading_mode": _lower(row.get("trading_mode")) or _lower(trading_mode) or "paper",
        "cycle_id": _text(row.get("cycle_id")) or None,
        "summary": summary,
        "ticker": ticker or None,
        "memory_tier": None,
        "primary_regime": None,
        "context_tags_json": None,
        "payload_json": dict(row),
    }


def build_order_intent_row_graph_node(row: dict[str, Any], *, trading_mode: str = "paper") -> dict[str, Any]:
    ticker = _upper(row.get("ticker"))
    side = _upper(row.get("side"))
    rationale = _text(row.get("rationale"))
    summary = f"{side or '?'} {ticker or '?'} qty={float(row.get('quantity') or 0.0):.4f}"
    if rationale:
        summary += f" rationale={rationale[:160]}"
    return {
        "node_id": order_intent_node_id(_text(row.get("intent_id"))),
        "created_at": row.get("created_at"),
        "node_kind": "order_intent",
        "source_table": "agent_order_intents",
        "source_id": _text(row.get("intent_id")),
        "agent_id": _text(row.get("agent_id")) or None,
        "trading_mode": _lower(row.get("trading_mode")) or _lower(trading_mode) or "paper",
        "cycle_id": _text(row.get("cycle_id")) or None,
        "summary": summary,
        "ticker": ticker or None,
        "memory_tier": None,
        "primary_regime": None,
        "context_tags_json": None,
        "payload_json": dict(row),
    }


def build_intent_execution_edge_from_rows(
    intent_row: dict[str, Any],
    execution_row: dict[str, Any],
    *,
    trading_mode: str = "paper",
) -> dict[str, Any]:
    intent_id = _text(intent_row.get("intent_id"))
    order_id = _text(execution_row.get("order_id"))
    return {
        "edge_id": f"edge:executed_as:{intent_id}:{order_id}",
        "created_at": execution_row.get("created_at") or intent_row.get("created_at"),
        "trading_mode": _lower(execution_row.get("trading_mode")) or _lower(trading_mode) or "paper",
        "cycle_id": _text(execution_row.get("cycle_id") or intent_row.get("cycle_id")) or None,
        "from_node_id": order_intent_node_id(intent_id),
        "to_node_id": execution_report_node_id(order_id),
        "edge_type": "EXECUTED_AS",
        "edge_strength": 1.0,
        "confidence": 1.0,
        "causal_chain_id": f"chain:intent:{intent_id}" if intent_id else None,
        "detail_json": {"status": _upper(execution_row.get("status")) or None},
    }
