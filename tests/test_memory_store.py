from __future__ import annotations

import math
import json
from datetime import datetime, timedelta, timezone

import pytest

from arena.memory.policy import normalize_memory_policy
from arena.memory.store import MemoryStore
from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, RiskDecision, Side


class _FakeVectorStore:
    db = None
    def __init__(self) -> None:
        self.saved = []
    def save_memory_vector(self, **kwargs) -> None:
        self.saved.append(kwargs)
    def search_similar_memories(self, **kwargs) -> list:
        return []
    def search_peer_lessons(self, **kwargs) -> list:
        return []


class _FakeRepo:
    def __init__(self) -> None:
        self.events = []
        self.event_updates = []
        self.score_updates = {}
        self.buy_memories = []
        self.trade_memory_by_order_id = {}
        self.recent_rows = []
        self.active_thesis_rows: dict[str, dict] = {}

    def write_memory_event(self, event) -> None:
        self.events.append(event)
        self.recent_rows.insert(
            0,
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "summary": event.summary,
                "created_at": event.created_at,
            },
        )

    def find_trade_execution_memory_event(self, *, agent_id: str, order_id: str, trading_mode: str = "paper"):
        _ = (agent_id, trading_mode)
        return self.trade_memory_by_order_id.get(order_id)

    def update_memory_event(
        self,
        *,
        event_id: str,
        summary: str,
        payload: dict,
        score: float,
        importance_score: float | None = None,
        outcome_score: float | None = None,
        memory_tier: str | None = None,
        expires_at=None,
        context_tags: dict | None = None,
        primary_regime: str | None = None,
        primary_strategy_tag: str | None = None,
        primary_sector: str | None = None,
        graph_node_id: str | None = None,
        causal_chain_id: str | None = None,
    ) -> None:
        self.event_updates.append(
            {
                "event_id": event_id,
                "summary": summary,
                "payload": payload,
                "score": score,
                "importance_score": importance_score,
                "outcome_score": outcome_score,
                "memory_tier": memory_tier,
                "expires_at": expires_at,
                "context_tags": context_tags,
                "primary_regime": primary_regime,
                "primary_strategy_tag": primary_strategy_tag,
                "primary_sector": primary_sector,
                "graph_node_id": graph_node_id,
                "causal_chain_id": causal_chain_id,
            }
        )

    def recent_memory_events(self, agent_id: str, limit: int, trading_mode: str = "paper") -> list[dict]:
        _ = (agent_id, limit, trading_mode)
        return list(self.recent_rows[:limit])

    def find_buy_memories_for_ticker(self, agent_id: str, ticker: str, limit: int = 5, trading_mode: str = "paper") -> list[dict]:
        return self.buy_memories

    def update_memory_score(self, event_id: str, new_score: float) -> None:
        self.score_updates[event_id] = new_score

    def active_thesis_events(self, *, agent_id: str, tickers: list[str], trading_mode: str = "paper"):
        _ = (agent_id, trading_mode)
        return [self.active_thesis_rows[ticker] for ticker in tickers if ticker in self.active_thesis_rows]


def test_record_execution_summary_includes_policy_and_broker_reason() -> None:
    repo = _FakeRepo()
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())

    intent = OrderIntent(
        agent_id="gpt",
        ticker="WMT",
        side=Side.BUY,
        quantity=1.149,
        price_krw=120_000,
        rationale="test",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.ERROR,
        order_id="err_123",
        filled_qty=0.0,
        avg_price_krw=0.0,
        message="market is closed",
    )

    store.record_execution(intent=intent, decision=decision, report=report)

    assert len(repo.events) == 1
    summary = str(repo.events[0].summary)
    assert "status=ERROR" in summary
    assert "policy=approved" in summary
    assert "broker=market is closed" in summary


def test_record_execution_indexes_only_filled_or_simulated() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    intent = OrderIntent(
        agent_id="gpt",
        ticker="WMT",
        side=Side.BUY,
        quantity=1.0,
        price_krw=120_000,
        rationale="test",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])

    store.record_execution(
        intent=intent,
        decision=decision,
        report=ExecutionReport(
            status=ExecutionStatus.REJECTED,
            order_id="rej_1",
            filled_qty=0.0,
            avg_price_krw=0.0,
            message="policy reject",
        ),
    )
    store.record_execution(
        intent=intent,
        decision=decision,
        report=ExecutionReport(
            status=ExecutionStatus.FILLED,
            order_id="fill_1",
            filled_qty=1.0,
            avg_price_krw=120_000.0,
            message="filled",
        ),
    )

    assert [row["event_type"] for row in vector_store.saved] == ["trade_execution"]


def test_record_candidate_memories_persists_bounded_nonheld_screen_hits() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(
        repo=repo,
        vector_store=vector_store,
        trading_mode="paper",
        memory_policy=normalize_memory_policy({}),
    )

    written = store.record_candidate_memories(
        agent_id="gpt",
        candidate_ledger={
            "AAPL": {"source_tools": {"screen_market:momentum"}, "discovery_count": 1},
            "MSFT": {
                "source_tools": {"screen_market:value"},
                "discovery_count": 1,
                "last_seen_rank": 2,
                "discovery_evidence": {"reason_for": "Valuation support", "score": 1.2},
            },
        },
        held_tickers={"AAPL"},
        cycle_id="cycle_candidate",
        phase="execution",
    )

    assert written == 1
    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.event_type == "candidate_screen_hit"
    assert event.payload["ticker"] == "MSFT"
    assert event.payload["cycle_id"] == "cycle_candidate"
    assert event.semantic_key.startswith("candidate:gpt:paper:MSFT:")
    assert event.expires_at is not None
    assert vector_store.saved[0]["event_type"] == "candidate_screen_hit"


def test_record_execution_updates_existing_order_memory() -> None:
    repo = _FakeRepo()
    repo.trade_memory_by_order_id["ord_keep"] = {
        "event_id": "mem_existing",
        "created_at": None,
    }
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())

    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2.0,
        price_krw=100_000,
        rationale="reconcile",
        intent_id="intent_x",
    )
    decision = RiskDecision(allowed=True, reason="reconciled", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_keep",
        filled_qty=2.0,
        avg_price_krw=100_000.0,
        message="reconciled",
    )

    store.record_execution(intent=intent, decision=decision, report=report)

    assert len(repo.events) == 0
    assert len(repo.event_updates) == 1
    updated = repo.event_updates[0]
    assert updated["event_id"] == "mem_existing"
    assert "status=FILLED" in str(updated["summary"])
    assert float(updated["score"]) == pytest.approx(0.75)
    assert float(updated["importance_score"]) == pytest.approx(0.75)
    assert float(updated["outcome_score"]) == pytest.approx(0.5)


def test_record_thesis_lifecycle_opens_new_thesis_on_filled_buy() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2.0,
        price_krw=100_000,
        rationale="AI demand and margin recovery",
        strategy_refs=["momentum"],
        intent_id="intent_open",
        cycle_id="cycle_open",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_open",
        filled_qty=2.0,
        avg_price_krw=100_000.0,
        message="filled",
    )

    store.record_thesis_lifecycle(intent=intent, decision=decision, report=report, snapshot_before=None)

    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.event_type == "thesis_open"
    assert event.semantic_key
    assert event.payload["thesis_id"] == event.semantic_key
    assert event.payload["position_action"] == "entry"
    assert vector_store.saved == []


def test_record_thesis_lifecycle_skips_non_material_active_buy() -> None:
    repo = _FakeRepo()
    repo.active_thesis_rows["AAPL"] = {
        "event_id": "mem_active",
        "event_type": "thesis_open",
        "semantic_key": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
        "payload_json": json.dumps(
            {
                "thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
                "ticker": "AAPL",
                "thesis_summary": "AI demand and margin recovery",
                "strategy_refs": ["momentum"],
            }
        ),
    }
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=101_000,
        rationale="AI demand and margin recovery",
        strategy_refs=["momentum"],
        intent_id="intent_add",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_add",
        filled_qty=1.0,
        avg_price_krw=101_000.0,
        message="filled",
    )

    store.record_thesis_lifecycle(intent=intent, decision=decision, report=report, snapshot_before=None)

    assert repo.events == []


def test_record_thesis_lifecycle_invalidates_on_thesis_broken_sell() -> None:
    repo = _FakeRepo()
    repo.active_thesis_rows["AAPL"] = {
        "event_id": "mem_active",
        "event_type": "thesis_update",
        "semantic_key": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
        "payload_json": json.dumps(
            {
                "thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
                "ticker": "AAPL",
                "state": "active",
                "thesis_summary": "AI demand and margin recovery",
                "strategy_refs": ["momentum"],
                "entry_cycle_id": "cycle_old",
            }
        ),
    }
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.SELL,
        quantity=1.0,
        price_krw=99_000,
        rationale="Guidance cut broke the thesis",
        strategy_refs=["thesis_broken"],
        intent_id="intent_sell",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_sell",
        filled_qty=1.0,
        avg_price_krw=99_000.0,
        message="filled",
    )

    snapshot = {
        "positions": {
            "AAPL": {"quantity": 2.0},
        }
    }
    store.record_thesis_lifecycle(intent=intent, decision=decision, report=report, snapshot_before=snapshot)

    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.event_type == "thesis_invalidated"
    assert event.payload["position_action"] == "trim"
    assert vector_store.saved[0]["event_type"] == "thesis_invalidated"


def test_record_thesis_lifecycle_realizes_full_exit() -> None:
    repo = _FakeRepo()
    repo.active_thesis_rows["AAPL"] = {
        "event_id": "mem_active",
        "event_type": "thesis_open",
        "semantic_key": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
        "payload_json": json.dumps(
            {
                "thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_old",
                "ticker": "AAPL",
                "state": "open",
                "thesis_summary": "AI demand and margin recovery",
                "strategy_refs": ["momentum"],
                "entry_cycle_id": "cycle_old",
            }
        ),
    }
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.SELL,
        quantity=2.0,
        price_krw=110_000,
        rationale="Target multiple reached",
        strategy_refs=["profit_taking"],
        intent_id="intent_exit",
    )
    decision = RiskDecision(allowed=True, reason="approved", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_exit",
        filled_qty=2.0,
        avg_price_krw=110_000.0,
        message="filled",
    )

    snapshot = {
        "positions": {
            "AAPL": {"quantity": 2.0},
        }
    }
    store.record_thesis_lifecycle(intent=intent, decision=decision, report=report, snapshot_before=snapshot)

    assert len(repo.events) == 1
    event = repo.events[0]
    assert event.event_type == "thesis_realized"
    assert event.payload["position_action"] == "exit"


def _expected_tanh_score(pnl_ratio: float) -> float:
    """tanh 기반 score 공식의 기대값을 계산한다."""
    return max(0.1, min(0.5 + 0.5 * math.tanh(pnl_ratio * 3), 1.0))


def test_record_execution_sells_trigger_buy_score_feedback() -> None:
    repo = _FakeRepo()
    repo.buy_memories = [
        {"event_id": "buy1", "payload_json": '{"intent": {"price_krw": 100.0}}', "score": 1.0}
    ]
    store = MemoryStore(repo=repo, vector_store=_FakeVectorStore())

    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.SELL,
        quantity=1.0,
        price_krw=150.0,
        rationale="take profit",
    )
    decision = RiskDecision(allowed=True, reason="ok", policy_hits=[])
    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ok_1",
        filled_qty=1.0,
        avg_price_krw=150.0,
        message="filled",
    )

    # +50% profit
    store.record_execution(intent=intent, decision=decision, report=report)
    expected = _expected_tanh_score(0.5)
    assert abs(repo.score_updates.get("buy1", 0) - expected) < 0.01

    # -10% loss
    repo.score_updates.clear()
    report.avg_price_krw = 90.0
    store.record_execution(intent=intent, decision=decision, report=report)
    expected = _expected_tanh_score(-0.1)
    assert abs(repo.score_updates.get("buy1", 0) - expected) < 0.01


def test_score_formula_boundary_values() -> None:
    """Score 공식의 변별력을 검증한다."""
    s_50_profit = _expected_tanh_score(0.5)
    s_10_profit = _expected_tanh_score(0.1)
    s_zero = _expected_tanh_score(0.0)
    s_10_loss = _expected_tanh_score(-0.1)
    s_50_loss = _expected_tanh_score(-0.5)

    assert s_50_profit > s_10_profit, "+50% > +10%"
    assert s_10_profit > s_zero, "+10% > 0%"
    assert s_zero > s_10_loss, "0% > -10%"
    assert s_10_loss > s_50_loss, "-10% > -50%"
    assert s_zero == 0.5, "0% pnl = 0.5 score"
    assert s_50_loss >= 0.1, "min score >= 0.1"
    assert s_50_profit <= 1.0, "max score <= 1.0"


def test_record_reflection_only_indexes_reflections() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    store.record_reflection("gpt", "I traded too frequently", score=0.4)
    assert repo.events[-1].event_type == "strategy_reflection"
    assert repo.events[-1].score == 0.4
    assert repo.events[-1].importance_score == 0.4
    assert repo.events[-1].outcome_score is None
    assert repo.events[-1].summary == "I traded too frequently"
    assert [row["event_type"] for row in vector_store.saved] == ["strategy_reflection"]


def test_record_manual_note_indexes_as_manual_note() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    store.record_manual_note("gpt", "Liquidity looked fake around the open.", score=0.55)

    assert repo.events[-1].event_type == "manual_note"
    assert repo.events[-1].payload["source"] == "manual_note"
    assert vector_store.saved[-1]["event_type"] == "manual_note"
    assert vector_store.saved[-1]["memory_source"] == "manual_note"


def test_record_manual_note_write_through_repo_does_not_self_dedup() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    store.record_manual_note("gpt", "AAPL trim near resistance", score=0.55)

    assert repo.events[-1].event_type == "manual_note"
    assert len(vector_store.saved) == 1
    assert vector_store.saved[0]["summary"] == "AAPL trim near resistance"


def test_record_manual_note_skips_recent_duplicate_indexing() -> None:
    repo = _FakeRepo()
    repo.recent_rows = [
        {
            "event_type": "manual_note",
            "summary": "AAPL trim near resistance",
            "created_at": datetime.now(timezone.utc) - timedelta(days=1),
        }
    ]
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    store.record_manual_note("gpt", "AAPL trim near resistance.", score=0.55)

    assert repo.events[-1].event_type == "manual_note"
    assert vector_store.saved == []


def test_record_manual_note_requires_contentful_signal() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    store.record_manual_note("gpt", "watch", score=0.4)
    store.record_manual_note("gpt", "overbought", score=0.4)

    assert [row["event_type"] for row in vector_store.saved] == ["manual_note"]
    assert vector_store.saved[0]["summary"] == "overbought"


def test_record_reflection_carries_memory_source_to_vector_metadata() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    store = MemoryStore(repo=repo, vector_store=vector_store)

    store.record_reflection(
        "gemini",
        "Trim winners when concentration grows under fragile macro conditions.",
        score=0.7,
        payload={"source": "memory_compaction", "cycle_id": "cycle_1"},
    )

    assert len(vector_store.saved) == 1
    saved = vector_store.saved[0]
    assert saved["event_type"] == "strategy_reflection"
    assert saved["memory_source"] == "memory_compaction"


def test_record_reflection_extracts_structured_tags_when_tagging_enabled() -> None:
    repo = _FakeRepo()
    vector_store = _FakeVectorStore()
    policy = normalize_memory_policy({"tagging": {"enabled": True, "max_tags": 8}})
    store = MemoryStore(repo=repo, vector_store=vector_store, memory_policy=policy)

    store.record_reflection(
        "gpt",
        "Momentum breakouts work better in bull markets for AAPL.",
        score=0.72,
        payload={"source": "memory_compaction", "tags": ["bull", "momentum", "breakout"]},
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
