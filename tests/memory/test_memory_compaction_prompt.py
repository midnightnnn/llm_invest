from __future__ import annotations

from arena.agents.memory_compaction_agent import MemoryCompactionAgent
from tests.memory.memory_compaction_helpers import _FakeMemoryStore, _FakeRepo, _settings


def test_memory_compaction_prompt_must_exist_in_db() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    try:
        agent._build_prompt(agent_id="gpt", cycle_id="cycle_123", inputs={})
    except RuntimeError as exc:
        assert "memory_compactor_prompt" in str(exc)
    else:
        raise AssertionError("expected missing DB prompt to raise")


def test_memory_compaction_prompt_uses_db_template() -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = (
        "AGENT={agent_id}\nCYCLE={cycle_id}\nMAX={max_reflections}\n{payload_json}"
    )
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    prompt = agent._build_prompt(
        agent_id="gpt",
        cycle_id="cycle_123",
        inputs={
            "closed_thesis_chains": [{"thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_1"}],
            "cycle_memories": [{"event_id": "evt_1"}],
            "board_posts": [],
            "environment_research": [{"briefing_id": "brf_1", "headline": "Rates steady"}],
            "prior_lessons": [],
        },
    )

    assert "AGENT=gpt" in prompt
    assert "CYCLE=cycle_123" in prompt
    assert '"closed_thesis_chains"' in prompt
    assert '"event_id": "evt_1"' in prompt
    assert '"briefing_id": "brf_1"' in prompt


def test_memory_compaction_prompt_prefers_tenant_override_over_global() -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = "GLOBAL {agent_id}"
    repo.configs[("local", "memory_compactor_prompt")] = "LOCAL {agent_id}"
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    prompt = agent._build_prompt(
        agent_id="gpt",
        cycle_id="cycle_123",
        inputs={
            "closed_thesis_chains": [],
            "cycle_memories": [],
            "board_posts": [],
            "environment_research": [],
            "prior_lessons": [],
        },
    )

    assert "LOCAL gpt" in prompt
    assert "GLOBAL gpt" not in prompt
