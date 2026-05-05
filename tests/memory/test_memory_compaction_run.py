from __future__ import annotations

import asyncio

from arena.agents.memory_compaction_agent import MemoryCompactionAgent
from tests.memory.memory_compaction_helpers import _FakeMemoryStore, _FakeRepo, _settings, _thesis_rows


def test_memory_compaction_agent_saves_reflections_from_cycle_outputs(monkeypatch) -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = (
        "최대 {max_reflections}개의 reflection만 생성하라.\n{payload_json}"
    )
    repo.cycle_rows = [
        {
            "event_id": "evt_trade",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.75,
            "outcome_score": 0.8,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY","rationale":"breakout"},"report":{"status":"FILLED"},"decision":{"policy_hits":[]}}',
        },
        {
            "event_id": "evt_tools",
            "event_type": "react_tools_summary",
            "summary": "ReAct tools used (execution): 3",
            "importance_score": 0.6,
            "payload_json": '{"phase":"execution","tool_mix":{"quant":3},"tool_events":[{"tool":"technical_signals"},{"tool":"screen_market"},{"tool":"forecast_returns"}]}',
        },
    ]
    repo.board_rows = [
        {
            "post_id": "post_cycle",
            "title": "거래 아이디어",
            "body": "AAPL breakout 재진입",
            "explore_summary": "AAPL momentum continuation",
            "tickers": ["AAPL"],
        }
    ]
    repo.research_rows = [
        {
            "briefing_id": "brf_global",
            "category": "global_market",
            "headline": "Higher-for-longer rates keep growth multiples under pressure",
            "summary": "Macro backdrop remains selective for crowded momentum trades.",
        },
        {
            "briefing_id": "brf_geo",
            "category": "geopolitical",
            "headline": "Shipping disruptions add energy and logistics volatility",
            "summary": "Geopolitical stress is reinforcing a defensive bias.",
        },
    ]
    memory_store = _FakeMemoryStore()
    memory_store.recent_rows = [
        {
            "event_id": "evt_old",
            "event_type": "strategy_reflection",
            "summary": "Avoid chasing late-stage semis after euphoric gaps.",
        }
    ]
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    async def _fake_compact_one(*, agent_id, cycle_id, inputs):
        assert agent_id == "gpt"
        assert cycle_id == "cycle_123"
        assert inputs["cycle_memories"][0]["event_id"] == "evt_trade"
        assert inputs["board_posts"][0]["post_id"] == "post_cycle"
        assert inputs["environment_research"][0]["briefing_id"] == "brf_global"
        return [
            {
                "summary": "AAPL breakout trades worked best when momentum confirmation, size discipline, and a favorable macro regime aligned.",
                "importance_score": 0.82,
                "tags": ["momentum", "sizing", "macro"],
                "source_event_ids": ["evt_trade", "evt_tools"],
                "source_post_ids": ["post_cycle"],
                "source_briefing_ids": ["brf_global"],
            }
        ]

    monkeypatch.setattr(agent, "_compact_one", _fake_compact_one)

    saved = asyncio.run(agent.run(cycle_id="cycle_123", agent_ids=["gpt", "gpt"]))

    assert len(saved) == 1
    assert len(memory_store.saved) == 1
    assert memory_store.saved[0]["agent_id"] == "gpt"
    assert memory_store.saved[0]["summary"].startswith("AAPL breakout trades worked best")
    assert memory_store.saved[0]["score"] == 0.82
    assert memory_store.saved[0]["payload"]["source"] == "memory_compaction"
    assert memory_store.saved[0]["payload"]["cycle_id"] == "cycle_123"
    assert memory_store.saved[0]["payload"]["source_event_ids"] == ["evt_trade", "evt_tools"]
    assert memory_store.saved[0]["payload"]["source_post_ids"] == ["post_cycle"]
    assert memory_store.saved[0]["payload"]["source_briefing_ids"] == ["brf_global"]
    assert memory_store.saved[0]["semantic_key"] is None


def test_memory_compaction_agent_loads_closed_thesis_chains() -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = "{payload_json}"
    thesis_id = "thesis:gpt:AAPL:paper:2026-03-29:intent_1"
    repo.closed_thesis_keys = [thesis_id]
    repo.semantic_rows_by_key[thesis_id] = _thesis_rows(thesis_id)
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    inputs = agent._load_agent_inputs("gpt", "cycle_123")

    assert len(inputs["closed_thesis_chains"]) == 1
    chain = inputs["closed_thesis_chains"][0]
    assert chain["thesis_id"] == thesis_id
    assert chain["terminal_event_type"] == "thesis_invalidated"
    assert chain["event_ids"] == ["evt_thesis_open", "evt_thesis_update", "evt_thesis_close"]
    assert chain["events"][-1]["event_type"] == "thesis_invalidated"


def test_memory_compaction_agent_uses_policy_sized_board_and_cycle_payloads() -> None:
    repo = _FakeRepo()
    long_body = "AAPL board detail " * 120
    long_summary = "tool evidence " * 120
    repo.cycle_rows = [
        {
            "event_id": "evt_tools",
            "event_type": "react_tools_summary",
            "summary": long_summary,
            "payload_json": {
                "phase": "execution",
                "tool_mix": {"quant": 1},
                "tool_events": [{"tool": "technical_signals", "phase": "execution", "result": {"signal": long_summary}}],
            },
        }
    ]
    repo.board_rows = [
        {"post_id": "post_1", "title": "T1", "body": long_body, "explore_summary": "summary", "tickers": ["AAPL"]},
        {"post_id": "post_2", "title": "T2", "body": "second", "explore_summary": "summary", "tickers": ["MSFT"]},
    ]
    settings = _settings()
    settings.memory_policy = {
        "compaction": {
            "board_post_limit": 1,
            "board_body_chars": 600,
            "cycle_summary_chars": 700,
        }
    }
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=_FakeMemoryStore())

    inputs = agent._load_agent_inputs("gpt", "cycle_123")

    assert repo.last_board_limit == 1
    assert len(inputs["board_posts"]) == 1
    assert len(inputs["board_posts"][0]["body"]) > 240
    assert inputs["board_posts"][0]["body_truncated"] is True
    assert len(inputs["cycle_memories"][0]["summary"]) > 220
    assert inputs["cycle_memories"][0]["tool_previews"][0]["tool"] == "technical_signals"


def test_memory_compaction_agent_saves_thesis_chain_reflection(monkeypatch) -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = "{payload_json}"
    thesis_id = "thesis:gpt:AAPL:paper:2026-03-29:intent_1"
    repo.closed_thesis_keys = [thesis_id]
    repo.semantic_rows_by_key[thesis_id] = _thesis_rows(thesis_id)
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    async def _fake_compact_one(*, agent_id, cycle_id, inputs):
        assert agent_id == "gpt"
        assert cycle_id == "cycle_123"
        assert inputs["closed_thesis_chains"][0]["thesis_id"] == thesis_id
        return [
            {
                "summary": "AAPL thesis drifted from AI demand to services mix and should have been de-risked before the guidance cut invalidated it.",
                "importance_score": 0.84,
                "tags": ["thesis", "risk"],
                "source_event_ids": ["evt_thesis_open", "evt_thesis_close"],
                "thesis_id": thesis_id,
                "terminal_event_type": "thesis_invalidated",
            }
        ]

    monkeypatch.setattr(agent, "_compact_one", _fake_compact_one)

    saved = asyncio.run(agent.run(cycle_id="cycle_123", agent_ids=["gpt"]))

    assert len(saved) == 1
    assert saved[0]["thesis_id"] == thesis_id
    assert len(memory_store.saved) == 1
    assert memory_store.saved[0]["payload"]["source"] == "thesis_chain_compaction"
    assert memory_store.saved[0]["payload"]["thesis_id"] == thesis_id
    assert memory_store.saved[0]["payload"]["terminal_event_type"] == "thesis_invalidated"
    assert memory_store.saved[0]["semantic_key"] == f"reflection:{thesis_id}"


def test_memory_compaction_agent_skips_already_compacted_thesis_chain(monkeypatch) -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = "{payload_json}"
    thesis_id = "thesis:gpt:AAPL:paper:2026-03-29:intent_1"
    repo.closed_thesis_keys = [thesis_id]
    repo.semantic_rows_by_key[thesis_id] = _thesis_rows(thesis_id)
    repo.existing_reflection_keys.add(("strategy_reflection", f"reflection:{thesis_id}"))
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    async def _should_not_run(*, agent_id, cycle_id, inputs):
        raise AssertionError("duplicate thesis chain should not trigger compaction")

    monkeypatch.setattr(agent, "_compact_one", _should_not_run)

    saved = asyncio.run(agent.run(cycle_id="cycle_123", agent_ids=["gpt"]))

    assert saved == []
    assert memory_store.saved == []


def test_memory_compaction_agent_skips_existing_cycle_compaction_reflections(monkeypatch) -> None:
    repo = _FakeRepo()
    repo.existing_compaction_rows = [{"event_id": "mem_existing", "summary": "already compacted"}]
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    async def _should_not_run(*, agent_id, cycle_id, inputs):
        raise AssertionError("existing compaction reflection should make rerun idempotent")

    monkeypatch.setattr(agent, "_compact_one", _should_not_run)

    saved = asyncio.run(agent.run(cycle_id="cycle_123", agent_ids=["gpt"]))

    assert saved == []
    assert memory_store.saved == []
