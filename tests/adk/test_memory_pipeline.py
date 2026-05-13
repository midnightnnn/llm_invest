from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from arena.agents.adk_agents import _ADKDecisionRunner, _ContextTools


class _MemoryStoreForToolSummary:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_memory(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _MemoryStoreForCandidateMemory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_candidate_memories(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return 1


class _VectorStoreForToolMemory:
    def search_similar_memories(self, **kwargs):
        _ = kwargs
        return [
            {
                "summary": "Macro-sensitive trim discipline mattered.",
                "importance_score": 0.8,
                "created_at": datetime.fromisoformat("2026-03-05T00:00:00+00:00"),
                "outcome_score": 0.8,
            }
        ]


class _VectorStoreForDedupedToolMemory:
    def search_similar_memories(self, **kwargs):
        _ = kwargs
        return [
            {
                "event_id": "mem_seen",
                "summary": "Already injected lesson.",
                "importance_score": 0.9,
                "created_at": datetime.fromisoformat("2026-03-05T00:00:00+00:00"),
                "outcome_score": 0.8,
            },
            {
                "event_id": "mem_new",
                "summary": "Fresh trim discipline lesson.",
                "importance_score": 0.7,
                "created_at": datetime.fromisoformat("2026-03-04T00:00:00+00:00"),
                "outcome_score": 0.2,
            },
        ]


class _VectorStoreForContextToolSearch:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_similar_memories(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {"event_id": "mem_seen", "summary": "Already prompt-injected lesson."},
            {"event_id": "mem_new", "summary": "Fresh trim discipline lesson."},
            {"event_id": "mem_extra", "summary": "Second fresh lesson."},
        ]


class _MemoryStoreForToolMemory:
    def __init__(self) -> None:
        self.vector_store = _VectorStoreForToolMemory()

    def _tenant(self) -> str:
        return "local"


class _MemoryStoreForDedupedToolMemory:
    def __init__(self) -> None:
        self.vector_store = _VectorStoreForDedupedToolMemory()

    def _tenant(self) -> str:
        return "local"


class _VectorStoreForLegacyCandidateMemory:
    def search_similar_memories(self, **kwargs):
        _ = kwargs
        return [
            {
                "event_id": "mem_candidate",
                "summary": "007610 candidate_watchlist: surfaced by recommend_opportunities:aggressive rank=1. Reas...",
                "importance_score": 0.38,
                "created_at": datetime.fromisoformat("2026-05-07T00:00:00+00:00"),
            }
        ]


class _RepoForLegacyCandidateMemory:
    def memory_event_by_id(self, *, event_id: str, tenant_id: str | None = None):
        assert event_id == "mem_candidate"
        assert tenant_id == "local"
        return {
            "event_id": "mem_candidate",
            "event_type": "candidate_watchlist",
            "summary": "007610 candidate_watchlist: full stored memory summary",
            "payload_json": {
                "source": "candidate_discovery",
                "ticker": "007610",
                "candidate_status": "watchlist",
                "source_tools": ["recommend_opportunities:aggressive"],
                "analyzed_by": ["forecast_returns", "get_fundamentals", "technical_signals"],
                "last_seen_rank": 1,
                "discovery_evidence": {
                    "score": 0.86605,
                    "reason_for": "Learned IC ranker score=+0.8661; contribs: momentum_20d(+0.2992)",
                    "reason_risk": "model_confidence=low",
                },
            },
        }


class _MemoryStoreForLegacyCandidateMemory:
    def __init__(self) -> None:
        self.vector_store = _VectorStoreForLegacyCandidateMemory()
        self.repo = _RepoForLegacyCandidateMemory()

    def _tenant(self) -> str:
        return "local"


class _RepoForToolSummary:
    def __init__(self) -> None:
        self.events = []

    def write_memory_event(self, event) -> None:
        self.events.append(event)


def test_persist_tool_summary_memory_prefers_memory_store() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForToolSummary()
    runner.repo = _RepoForToolSummary()

    runner._persist_tool_summary_memory(
        summary="ReAct tools used (explore): 2",
        payload={
            "tool_events": [{"tool": "technical_signals"}],
            "phase": "explore",
            "token_usage": {"llm_calls": 2, "prompt_tokens": 1200, "completion_tokens": 180, "total_tokens": 1380},
        },
    )

    assert len(runner._memory_store.calls) == 1
    call = runner._memory_store.calls[0]
    assert call["agent_id"] == "gpt"
    assert call["event_type"] == "react_tools_summary"
    assert call["score"] == pytest.approx(0.6)
    assert call["payload"]["token_usage"]["total_tokens"] == 1380
    assert runner.repo.events == []


def test_persist_candidate_memories_uses_candidate_ledger() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner._memory_store = _MemoryStoreForCandidateMemory()
    runner._candidate_ledger = {
        "MSFT": {
            "source_tools": {"screen_market:value"},
            "discovery_count": 1,
            "last_seen_rank": 2,
            "discovery_evidence": {"reason_for": "Valuation support"},
        }
    }
    runner._held_tickers_cache = {"AAPL"}
    runner._current_phase = "execution"

    written = runner._persist_candidate_memories(cycle_id="cycle_candidate")

    assert written == 1
    call = runner._memory_store.calls[0]
    assert call["agent_id"] == "gpt"
    assert call["held_tickers"] == {"AAPL"}
    assert call["cycle_id"] == "cycle_candidate"
    assert call["phase"] == "execution"
    assert "MSFT" in call["candidate_ledger"]


def test_record_memory_passes_payload_to_vector_store() -> None:
    calls: list[dict] = []

    class Repo:
        def write_memory_event(self, event) -> None:
            self.event = event

    class VectorStore:
        def save_memory_vector(self, **kwargs) -> None:
            calls.append(kwargs)

    from arena.memory.store import MemoryStore

    store = MemoryStore(repo=Repo(), vector_store=VectorStore(), trading_mode="paper", memory_policy=None)
    payload = {
        "source": "candidate_discovery",
        "ticker": "007610",
        "discovery_evidence": {"reason_for": "reason survives"},
    }

    store.record_memory(
        agent_id="claude",
        summary="007610 candidate_watchlist: reason survives",
        event_type="candidate_watchlist",
        score=0.38,
        payload=payload,
    )

    assert calls
    assert calls[0]["payload"] == payload


def test_decide_orders_keeps_tool_events_reference_for_wrapped_tools(monkeypatch) -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    shared_tool_events = [{"tool": "stale_event"}]
    runner._tool_events = shared_tool_events
    runner._seen_memory_ids = set()
    runner._candidate_ledger = {}
    runner._current_phase = "unknown"
    runner._current_context = None
    runner._held_tickers_cache = set()
    runner._session_id = "sid_base"
    runner._max_tool_events = 5
    runner._run_config = object()
    runner._runner = object()
    runner._user_id = "arena"
    runner.agent_id = "gpt"
    runner._registry = SimpleNamespace(
        set_context=lambda context: None,
        list_entries=lambda **kwargs: [],
    )
    runner._toolbox = SimpleNamespace(set_context=lambda context: None)
    runner._memory_store = None
    runner._seed_seen_memory_ids = lambda context: None
    runner._extract_held_tickers = lambda context: set()
    runner._sync_pipeline_context = lambda: None
    runner._funnel_metrics = lambda: {}
    runner._persist_tool_summary_memory = lambda *, summary, payload: None
    runner._run_on_loop = lambda value: value
    runner._disabled_tool_ids = set()
    runner._mcp_toolset_count = 0
    runner._system_prompt_snapshot = ""
    runner._agent_config = None
    runner._prompt_snapshots = []
    runner._llm_call_ids_by_phase = {}
    runner._latest_llm_call_id = ""
    runner.provider = "gpt"
    runner.settings = SimpleNamespace(trading_mode="paper", kis_target_market="", memory_policy=None)
    runner.tenant_id = "local"
    runner.repo = SimpleNamespace()

    def _fake_run_async(_runner, session_id, prompt):
        _ = (_runner, session_id, prompt)
        shared_tool_events.append(
            {
                "tool": "technical_signals",
                "args": {"ticker": "AAPL"},
                "result": {"ticker": "AAPL", "trend_state": "uptrend"},
            }
        )
        return '{"orders": []}'

    runner._run_async = _fake_run_async

    monkeypatch.setattr(
        "arena.agents.adk_agents.prepare_decision_prompt",
        lambda *args, **kwargs: ("sid_test", "prompt", False),
    )
    monkeypatch.setattr("arena.agents.adk_agents.parse_decision_response", lambda text: {"orders": []})
    monkeypatch.setattr("arena.agents.adk_agents.tag_phase_tool_events", lambda *args, **kwargs: None)

    captured: dict[str, object] = {}

    def _capture_summary(tool_events, **kwargs):
        _ = kwargs
        captured["tool_names"] = [str(event.get("tool") or "") for event in tool_events]
        captured["tool_events_id"] = id(tool_events)
        return None

    monkeypatch.setattr("arena.agents.adk_agents.build_tool_summary_memory_record", _capture_summary)

    decision, session_id = runner.decide_orders({"cycle_phase": "execution"}, [])

    assert decision == {"orders": []}
    assert session_id == "sid_test"
    assert runner._tool_events is shared_tool_events
    assert captured["tool_events_id"] == id(shared_tool_events)
    assert captured["tool_names"] == ["technical_signals"]


def test_search_past_experiences_skips_cycle_seen_memory_ids() -> None:
    vector_store = _VectorStoreForContextToolSearch()
    tool = _ContextTools.__new__(_ContextTools)
    tool.agent_id = "gpt"
    tool.tenant_id = "local"
    tool.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    tool._vector_store = vector_store
    tool._seen_memory_ids = set()
    tool._seen_memory_ids_shared = False

    tool.set_context({"memory_events": [{"event_id": "mem_seen"}]})
    rows = tool.search_past_experiences("trim discipline", limit=2)

    assert [row["event_id"] for row in rows] == ["mem_new", "mem_extra"]
    assert vector_store.calls[0]["limit"] == 5
    assert tool._seen_memory_ids == {"mem_seen", "mem_new", "mem_extra"}


def test_search_tool_memories_includes_created_date() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForToolMemory()
    runner._seen_memory_ids = set()

    rows = runner._search_tool_memories("macro regime trim discipline")

    assert rows is not None
    assert rows[0]["created_date"] == "2026-03-05"
    assert rows[0]["created_at"].startswith("2026-03-05T00:00:00")
    assert rows[0]["outcome_label"] == "win"


def test_search_tool_memories_keeps_full_summary_without_slice() -> None:
    long_summary = "Macro-sensitive trim discipline mattered. " * 20

    class VectorStore:
        def search_similar_memories(self, **kwargs):
            _ = kwargs
            return [{"event_id": "mem_long", "summary": long_summary, "importance_score": 0.8}]

    class MemoryStore:
        vector_store = VectorStore()

        def _tenant(self) -> str:
            return "local"

    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = MemoryStore()
    runner._seen_memory_ids = set()

    rows = runner._search_tool_memories("macro regime trim discipline")

    assert rows is not None
    assert rows[0]["summary"] == long_summary
    assert not rows[0]["summary"].endswith("...")


def test_search_tool_memories_enriches_legacy_vector_hit_from_repo_payload() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "claude"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForLegacyCandidateMemory()
    runner._seen_memory_ids = set()

    rows = runner._search_tool_memories("007610 opportunity")

    assert rows is not None
    assert rows[0]["event_id"] == "mem_candidate"
    assert rows[0]["event_type"] == "candidate_watchlist"
    assert rows[0]["summary"] == "007610 candidate_watchlist: full stored memory summary"
    assert rows[0]["payload"]["ticker"] == "007610"
    assert rows[0]["payload"]["discovery_evidence"]["score"] == 0.86605


def test_search_tool_memories_skips_initially_injected_event_ids() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForDedupedToolMemory()
    runner._seen_memory_ids = {"mem_seen"}

    rows = runner._search_tool_memories("trim discipline")

    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["summary"] == "Fresh trim discipline lesson."
    assert "mem_new" in runner._seen_memory_ids


def test_seed_seen_memory_ids_uses_initial_context_memory_rows() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._seen_memory_ids = set()

    runner._seed_seen_memory_ids(
        {
            "memory_events": [
                {"event_id": "mem_a"},
                {"event_id": "mem_b"},
                {"summary": "no id"},
            ]
        }
    )

    assert runner._seen_memory_ids == {"mem_a", "mem_b"}
