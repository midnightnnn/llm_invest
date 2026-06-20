from __future__ import annotations

from arena.memory.policy import normalize_memory_policy
from arena.memory.store import MemoryStore
from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, RiskDecision, Side
from tests.memory.memory_store_helpers import _FakeRepo, _FakeVectorStore


def test_record_reflection_extracts_structured_tags_when_tagging_enabled() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    policy = normalize_memory_policy({"tagging": {"enabled": True, "max_tags": 8}})
    store = MemoryStore(repo=repo, vector_store=vector_store, memory_policy=policy)

    store.record_reflection(
        "gpt",
        "Momentum breakouts work better in bull markets for AAPL.",
        score=0.72,
        payload={"source": "memory_compaction", "tags": ["bull", "momentum", "breakout", "technology"]},
    )

    event = repo.events[0]
    assert event.primary_regime == "bull"
    assert event.primary_strategy_tag == "momentum"
    assert event.primary_sector == "tech"
    assert event.context_tags["regimes"] == ["bull"]
    assert "breakout" in event.context_tags["strategies"]
    assert "AAPL" in event.context_tags["tickers"]
    assert vector_store.saved[0]["primary_regime"] == "bull"
    assert vector_store.saved[0]["primary_sector"] == "tech"


def test_record_memory_assigns_temporal_tiers_when_hierarchy_enabled() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    policy = normalize_memory_policy({"hierarchy": {"enabled": True, "working_ttl_hours": 24, "episodic_ttl_days": 60}})
    store = MemoryStore(repo=repo, vector_store=vector_store, memory_policy=policy)

    store.record_reflection("gpt", "Protect capital first when macro breadth deteriorates.", score=0.7)
    store.record_manual_note("gpt", "AAPL broke below weekly support.", score=0.55)

    reflection = repo.events[0]
    note = repo.events[1]
    assert reflection.memory_tier == "semantic"
    assert reflection.expires_at is None
    assert note.memory_tier == "episodic"
    assert note.expires_at is not None
    assert (note.expires_at - note.created_at).days >= 59
    assert vector_store.saved[0]["memory_tier"] == "semantic"
    assert vector_store.saved[1]["memory_tier"] == "episodic"


def test_record_execution_assigns_episodic_tier_when_hierarchy_enabled() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    policy = normalize_memory_policy({"hierarchy": {"enabled": True, "episodic_ttl_days": 45}})
    store = MemoryStore(repo=repo, vector_store=vector_store, memory_policy=policy)

    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2.0,
        price_krw=100_000,
        rationale="setup",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_tier",
        filled_qty=2.0,
        avg_price_krw=100_000.0,
        message="filled",
    )

    store.record_execution(intent=intent, decision=decision, report=report)

    event = repo.events[0]
    assert event.memory_tier == "episodic"
    assert event.expires_at is not None
    assert (event.expires_at - event.created_at).days >= 44
    assert vector_store.saved[0]["memory_tier"] == "episodic"


def test_record_execution_attaches_graph_metadata_to_memory_and_vector() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        trading_mode="live",
        side=Side.BUY,
        quantity=2.0,
        price_krw=100_000,
        rationale="setup",
        intent_id="intent_graph",
        cycle_id="cycle_graph",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_graph",
        filled_qty=2.0,
        avg_price_krw=100_000.0,
        message="filled",
    )

    store.record_execution(intent=intent, decision=decision, report=report)

    event = repo.events[0]
    assert event.graph_node_id == f"mem:{event.event_id}"
    assert event.causal_chain_id == "chain:intent:intent_graph"
    assert vector_store.saved[0]["graph_node_id"] == f"mem:{event.event_id}"
    assert vector_store.saved[0]["causal_chain_id"] == "chain:intent:intent_graph"


def test_react_tool_summary_index_policy_filters_low_signal_explore() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    store.record_memory(
        "gpt",
        "Explore tools used: 1",
        event_type="react_tools_summary",
        score=0.6,
        payload={"phase": "explore", "tool_events": [{"tool": "technical_signals"}]},
    )
    store.record_memory(
        "gpt",
        "Execution tools used: 3",
        event_type="react_tools_summary",
        score=0.6,
        payload={
            "phase": "execution",
            "cycle_id": "cycle_123",
            "tool_events": [
                {"tool": "technical_signals"},
                {"tool": "screen_market"},
                {"tool": "forecast_returns"},
            ],
        },
    )

    assert len(repo.events) == 2
    assert vector_store.saved == []
