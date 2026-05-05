from __future__ import annotations

from arena.memory.graph import (
    build_execution_report_graph_node,
    build_intent_execution_edge,
    build_memory_event_graph_edges,
    build_memory_event_graph_node,
    ensure_memory_event_graph_ids,
)
from arena.models import ExecutionReport, ExecutionStatus, MemoryEvent, OrderIntent, Side


def test_memory_event_graph_projection_uses_trade_payload_links() -> None:
    event = MemoryEvent(
        agent_id="gpt",
        event_type="trade_execution",
        summary="AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
        trading_mode="live",
        payload={
            "intent": {"intent_id": "intent_1", "ticker": "AAPL", "side": "BUY", "cycle_id": "cycle_1"},
            "report": {"order_id": "ord_1", "status": "FILLED"},
        },
    )
    ensure_memory_event_graph_ids(event)

    node = build_memory_event_graph_node(event)
    edges = build_memory_event_graph_edges(event)

    assert node["node_id"] == f"mem:{event.event_id}"
    assert node["ticker"] == "AAPL"
    assert node["trading_mode"] == "live"
    assert node["cycle_id"] == "cycle_1"
    assert {edge["edge_type"] for edge in edges} == {"PRECEDES", "RESULTED_IN"}
    assert {edge["from_node_id"] for edge in edges} == {"intent:intent_1", "exec:ord_1"}


def test_reflection_graph_projection_links_sources() -> None:
    event = MemoryEvent(
        agent_id="gpt",
        event_type="strategy_reflection",
        summary="Trim concentration when fragile macro breadth appears.",
        payload={
            "cycle_id": "cycle_2",
            "source_event_ids": ["mem_a", "mem_b"],
            "source_post_ids": ["post_1"],
            "source_briefing_ids": ["brf_1"],
        },
    )
    ensure_memory_event_graph_ids(event)

    edges = build_memory_event_graph_edges(event)

    assert {edge["edge_type"] for edge in edges} == {"ABSTRACTED_TO", "INFORMED_BY"}
    assert "post:post_1" in {edge["from_node_id"] for edge in edges}
    assert "brief:brf_1" in {edge["from_node_id"] for edge in edges}


def test_orphan_memory_event_does_not_get_synthetic_chain_id() -> None:
    event = MemoryEvent(
        agent_id="gpt",
        event_type="manual_note",
        summary="Watch liquidity behavior near the open.",
        payload={"source": "manual_note"},
    )

    ensure_memory_event_graph_ids(event)

    assert event.graph_node_id == f"mem:{event.event_id}"
    assert event.causal_chain_id is None


def test_order_and_execution_graph_projection_form_exact_chain() -> None:
    intent = OrderIntent(
        agent_id="gpt",
        ticker="MSFT",
        trading_mode="paper",
        side=Side.BUY,
        quantity=1.5,
        price_krw=120_000,
        rationale="breakout",
        intent_id="intent_exact",
        cycle_id="cycle_x",
    )
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_exact",
        filled_qty=1.5,
        avg_price_krw=121_000.0,
        message="filled",
    )

    node = build_execution_report_graph_node(intent, report)
    edge = build_intent_execution_edge(intent, report)

    assert node["node_id"] == "exec:ord_exact"
    assert node["trading_mode"] == "paper"
    assert edge["from_node_id"] == "intent:intent_exact"
    assert edge["to_node_id"] == "exec:ord_exact"
    assert edge["edge_type"] == "EXECUTED_AS"
