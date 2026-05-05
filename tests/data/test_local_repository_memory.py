from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.models import MemoryEvent
from tests.data.local_repository_helpers import _now, _seed_memory_event, repo


def test_recent_memory_events_returns_per_agent_in_order(repo):
    t1 = datetime(2026, 4, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 2, tzinfo=timezone.utc)
    _seed_memory_event(repo, event_id="e1", agent_id="gpt", summary="old", ts=t1)
    _seed_memory_event(repo, event_id="e2", agent_id="gpt", summary="new", ts=t2)
    _seed_memory_event(repo, event_id="e3", agent_id="claude", summary="other", ts=t2)

    rows = repo.recent_memory_events("gpt", limit=10)
    summaries = [r["summary"] for r in rows]
    assert summaries == ["new", "old"]


def test_memory_event_by_id_round_trips(repo):
    t = _now()
    _seed_memory_event(repo, event_id="abc", agent_id="gemini", summary="hello", ts=t)
    row = repo.memory_event_by_id(event_id="abc")
    assert row and row["summary"] == "hello"
    assert repo.memory_event_by_id(event_id="missing") is None


def test_memory_events_by_ids_filters_to_known(repo):
    t = _now()
    for eid in ("a", "b", "c"):
        _seed_memory_event(repo, event_id=eid, agent_id="gpt", summary=eid, ts=t)
    rows = repo.memory_events_by_ids(agent_id="gpt", event_ids=["a", "c", "ghost"])
    summaries = sorted(r["summary"] for r in rows)
    assert summaries == ["a", "c"]


def test_memory_events_for_cycle_matches_column_and_payload_cycle(repo):
    t = _now()
    repo.execute(
        """
        INSERT INTO agent_memory_events
          (tenant_id, event_id, created_at, agent_id, event_type, summary, trading_mode, cycle_id, payload_json)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "tenant-a", "cycle-col", t, "gemini", "trade_outcome", "column match", "paper", "cycle-x", "{}",
            "tenant-a", "cycle-payload", t, "gemini", "lesson", "payload match", "paper", None, '{"intent":{"cycle_id":"cycle-x"}}',
            "tenant-a", "other-cycle", t, "gemini", "lesson", "miss", "paper", "cycle-y", "{}",
        ],
    )

    rows = repo.memory_events_for_cycle(
        agent_id="gemini",
        cycle_id="cycle-x",
        event_types=["lesson"],
        limit=10,
    )

    assert [row["event_id"] for row in rows] == ["cycle-payload"]


def test_latest_memory_compaction_cycle_id_uses_latest_matching_cycle(repo):
    older = datetime(2026, 4, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 4, 2, tzinfo=timezone.utc)
    repo.execute(
        """
        INSERT INTO agent_memory_events
          (tenant_id, event_id, created_at, agent_id, event_type, summary, trading_mode, cycle_id, payload_json)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "tenant-a", "old-cycle", older, "gemini", "trade_execution", "old", "paper", "cycle-old", "{}",
            "tenant-a", "new-cycle", newer, "gemini", "thesis_open", "new", "paper", None, '{"cycle_id":"cycle-new"}',
            "tenant-a", "ignored-agent", newer, "gpt", "trade_execution", "ignored", "paper", "cycle-gpt", "{}",
        ],
    )

    cycle_id = repo.latest_memory_compaction_cycle_id(
        agent_ids=["gemini"],
        event_types=["trade_execution", "thesis_open"],
        trading_mode="paper",
    )

    assert cycle_id == "cycle-new"


def test_relation_extraction_pending_sources_returns_source_text(repo):
    t = _now()
    _seed_memory_event(
        repo,
        event_id="rel-source",
        agent_id="gpt",
        summary="AI demand supports NVDA margin recovery.",
        ts=t,
    )

    rows = repo.relation_extraction_pending_sources(
        limit=10,
        source_table="agent_memory_events",
        event_types=["lesson"],
        trading_mode="paper",
        extractor_version="semantic_relation_extractor_v1",
        prompt_version="semantic_relation_prompt_v2",
        ontology_version="semantic_relation_ontology_v1",
        tenant_id="tenant-a",
    )

    assert len(rows) == 1
    assert rows[0]["source_text"] == "AI demand supports NVDA margin recovery."
    assert "text" not in rows[0]


def test_compaction_reflections_for_cycle_returns_existing_reflections(repo):
    t = _now()
    repo.execute(
        """
        INSERT INTO agent_memory_events
          (tenant_id, event_id, created_at, agent_id, event_type, summary, trading_mode, cycle_id, payload_json)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "tenant-a", "reflection", t, "gemini", "strategy_reflection", "keep", "paper", "cycle-x", '{"source":"memory_compaction"}',
            "tenant-a", "manual", t, "gemini", "strategy_reflection", "skip", "paper", "cycle-x", '{"source":"manual"}',
        ],
    )

    rows = repo.compaction_reflections_for_cycle(agent_id="gemini", cycle_id="cycle-x", limit=10)

    assert [row["event_id"] for row in rows] == ["reflection"]


def test_write_memory_event_round_trips(repo):
    event = MemoryEvent(
        agent_id="gpt",
        event_type="manual_note",
        summary="local memory write",
        payload={"ticker": "AAPL"},
        score=0.7,
        trading_mode="paper",
    )

    repo.write_memory_event(event)

    row = repo.memory_event_by_id(event_id=event.event_id)
    assert row is not None
    assert row["summary"] == "local memory write"
    assert row["payload_json"]
    assert row["score"] == pytest.approx(0.7)


def test_update_memory_score_updates_outcome(repo):
    event = MemoryEvent(agent_id="gpt", event_type="manual_note", summary="score me", score=0.4)
    repo.write_memory_event(event)

    repo.update_memory_score(event.event_id, 0.9)

    row = repo.memory_event_by_id(event_id=event.event_id)
    assert row and row["outcome_score"] == pytest.approx(0.9)
