from __future__ import annotations

from arena.memory.vector import VectorStore


class _FakeDoc:
    def __init__(self, doc_id: str, data: dict) -> None:
        self.id = doc_id
        self._data = data

    def to_dict(self) -> dict:
        return dict(self._data)


class _FakeVectorQuery:
    def __init__(self, docs: list[_FakeDoc]) -> None:
        self._docs = docs

    def where(self, filter=None):
        _ = filter
        return self

    def find_nearest(self, **kwargs):
        _ = kwargs
        return self

    def stream(self):
        return iter(self._docs)


class _FakeDb:
    def __init__(self, docs: list[_FakeDoc]) -> None:
        self._docs = docs

    def collection(self, name: str):
        assert name == "agent_memories"
        return _FakeVectorQuery(self._docs)


def test_search_peer_lessons_excludes_self_and_sets_author_id() -> None:
    store = VectorStore.__new__(VectorStore)
    store.db = _FakeDb(
        [
            _FakeDoc(
                "mem_self",
                {
                    "agent_id": "gpt",
                    "summary": "My own compacted lesson",
                    "score": 0.7,
                    "created_date": "2026-03-07",
                    "event_type": "strategy_reflection",
                },
            ),
            _FakeDoc(
                "mem_peer",
                {
                    "agent_id": "gemini",
                    "summary": "Peer compacted lesson",
                    "score": 0.8,
                    "created_date": "2026-03-06",
                    "event_type": "strategy_reflection",
                    "memory_source": "memory_compaction",
                },
            ),
        ]
    )
    store.embed_text = lambda text: [0.1, 0.2] if text else []

    rows = store.search_peer_lessons(
        agent_id="gpt",
        query="concentration lesson",
        limit=3,
        trading_mode="paper",
        tenant_id="local",
    )

    assert len(rows) == 1
    assert rows[0]["event_id"] == "mem_peer"
    assert rows[0]["agent_id"] == "gemini"
    assert rows[0]["author_id"] == "gemini"
    assert rows[0]["memory_source"] == "memory_compaction"
