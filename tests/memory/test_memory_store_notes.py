from __future__ import annotations

from datetime import datetime, timedelta, timezone

from arena.memory.store import MemoryStore
from tests.memory.memory_store_helpers import _FakeRepo, _FakeVectorStore


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


def test_record_manual_note_skips_recent_duplicate_storage_and_indexing() -> None:
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

    assert repo.events == []
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
