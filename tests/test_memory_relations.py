from __future__ import annotations

from datetime import datetime, timezone

from arena.memory.relations import (
    build_board_post_relation_triples,
    build_memory_event_relation_triples,
    build_research_briefing_relation_triples,
    relation_triples_to_graph_projection,
    ticker_node_id,
)
from arena.models import BoardPost, MemoryEvent


def test_memory_event_relation_triples_extract_structured_passage_links() -> None:
    event = MemoryEvent(
        agent_id="gpt",
        event_type="trade_execution",
        summary="AAPL BUY worked after earnings guide stabilized.",
        trading_mode="live",
        payload={"intent": {"ticker": "AAPL", "cycle_id": "cycle_1"}},
        primary_strategy_tag="earnings",
        primary_sector="technology",
        created_at=datetime(2026, 3, 29, 1, 2, tzinfo=timezone.utc),
        event_id="evt_1",
        graph_node_id="mem:evt_1",
    )

    triples = build_memory_event_relation_triples(event)

    by_object = {row["object_node_id"]: row for row in triples}
    assert ticker_node_id("AAPL") in by_object
    assert by_object[ticker_node_id("AAPL")]["predicate"] == "contains"
    assert by_object[ticker_node_id("AAPL")]["subject_node_id"] == "mem:evt_1"
    assert by_object["entity:strategy_tag:earnings"]["object_type"] == "strategy_tag"
    assert by_object["entity:sector:technology"]["object_type"] == "sector"


def test_relation_triples_project_entity_nodes_and_edges() -> None:
    event = MemoryEvent(
        agent_id="gpt",
        event_type="strategy_reflection",
        summary="Avoid oversized NVDA entries when export restriction risk resurfaces.",
        payload={"ticker": "NVDA", "cycle_id": "cycle_2"},
        event_id="evt_2",
        graph_node_id="mem:evt_2",
    )
    triples = build_memory_event_relation_triples(event)

    nodes, edges = relation_triples_to_graph_projection(triples)

    assert any(node["node_id"] == "ticker:NVDA" and node["node_kind"] == "semantic_entity" for node in nodes)
    assert any(
        edge["from_node_id"] == "mem:evt_2"
        and edge["to_node_id"] == "ticker:NVDA"
        and edge["edge_type"] == "CONTAINS"
        for edge in edges
    )
    assert all(edge["detail_json"]["triple_id"].startswith("triple:") for edge in edges)


def test_board_and_research_relation_triples_use_passage_sources() -> None:
    post = BoardPost(
        agent_id="claude",
        title="Semis watch",
        body="NVDA demand is still strong.",
        tickers=["NVDA"],
        post_id="post_1",
        cycle_id="cycle_3",
    )
    briefing = {
        "briefing_id": "brief_1",
        "created_at": datetime(2026, 3, 29, 2, 0, tzinfo=timezone.utc),
        "ticker": "MSFT",
        "category": "company",
        "headline": "Cloud demand stabilizes",
        "summary": "Azure checks improved.",
        "trading_mode": "paper",
    }

    post_triples = build_board_post_relation_triples(post)
    briefing_triples = build_research_briefing_relation_triples(briefing)

    assert post_triples[0]["source_table"] == "board_posts"
    assert post_triples[0]["subject_node_id"] == "post:post_1"
    assert post_triples[0]["object_node_id"] == "ticker:NVDA"
    assert {row["object_node_id"] for row in briefing_triples} == {
        "ticker:MSFT",
        "entity:research_category:company",
    }
