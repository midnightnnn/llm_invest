# Structured Memory Context Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace blind `_memory_context.summary` truncation with lean structured, model-visible memory context that preserves candidate reason/risk/follow-up facts for every tool that receives REACT memory injection.

**Architecture:** Tool memory injection already has one shared path: `build_tool_wrapper()` attaches `_memory_context`, then `_compact_tool_result_for_prompt()` makes the model-visible payload. Add a focused memory-context formatter and enrich vector hits with stored memory payloads so all injectable tools benefit without changing each tool implementation.

**Tech Stack:** Python, pytest, BigQuery-backed and local memory stores, Firestore/Chroma vector metadata.

---

## File Structure

- Create `arena/agents/adk_memory_context.py`
  - Owns model-visible memory context formatting.
  - Converts candidate memories into compact structured fields (`d`, `type`, `t`, `src`, `rank`, `score`, `checked`, `why`, `risk`) instead of prose snippets.
  - Falls back to full stored summary for unknown memory types without arbitrary substring truncation.
- Modify `arena/agents/adk_tool_compaction.py`
  - Replace `_compact_memory_context_rows()` implementation with the new formatter.
  - Keep the existing attachment point so every injected tool uses the improved context automatically.
- Modify `arena/agents/adk_runner_runtime.py`
  - Stop slicing `summary[:220]`.
  - Add memory-hit enrichment from `memory_store.repo.memory_event_by_id()` when vector search returns only `event_id`/`summary`, preserving support for existing indexed memories.
  - Pass payload/event type fields through to the compactor.
- Modify `arena/memory/vector.py`
  - Store memory payload metadata in Firestore vector docs for newly indexed memories.
  - Return that payload in vector search rows.
- Modify `arena/memory/vector_local.py`
  - Store payload JSON in local vector metadata.
  - Return parsed payload in local search rows.
- Modify `arena/memory/store.py`
  - Pass `event.payload` into `save_memory_vector()`.
- Test `tests/adk/test_tool_compaction.py`
  - Verify candidate memory context is structured and not blindly truncated.
  - Verify wrapper output for any injectable tool still receives the improved `_memory_context`.
- Test `tests/adk/test_memory_pipeline.py`
  - Verify `search_tool_memories()` preserves full summaries.
  - Verify legacy vector hits are enriched from `memory_event_by_id()`.
- Test `tests/memory/test_memory_store_execution.py` or a focused vector-store test if existing helpers are lighter
  - Verify vector save receives payload from `MemoryStore.record_memory()`.

---

### Task 1: Add Structured Memory Context Formatter

**Files:**
- Create: `arena/agents/adk_memory_context.py`
- Test: `tests/adk/test_tool_compaction.py`

- [ ] **Step 1: Write failing tests for candidate memory context**

Add these tests to `tests/adk/test_tool_compaction.py` near the existing memory compaction tests:

```python
def test_compact_memory_context_candidate_uses_structured_payload_without_summary_truncation() -> None:
    long_reason = (
        "Learned IC ranker score=+0.8661; contribs: momentum_20d(+0.2992) "
        "meanrev_5d(+0.2325) pullback(+0.1566) lowvol(+0.1165); prob_up=50.0%"
    )
    out = _compact_tool_result_for_prompt(
        "macro_snapshot",
        {
            "status": "ok",
            "_memory_context": [
                {
                    "event_id": "mem_007610",
                    "created_date": "2026-05-07",
                    "event_type": "candidate_watchlist",
                    "summary": "007610 candidate_watchlist: " + ("x" * 400),
                    "importance_score": 0.38,
                    "payload": {
                        "source": "candidate_discovery",
                        "ticker": "007610",
                        "candidate_status": "watchlist",
                        "workflow_status": "analyzed",
                        "evidence_level": "validated",
                        "source_tools": ["recommend_opportunities:aggressive"],
                        "analyzed_by": ["forecast_returns", "get_fundamentals", "technical_signals"],
                        "discovery_count": 1,
                        "last_seen_rank": 1,
                        "discovery_evidence": {
                            "score": 0.86605,
                            "reason_for": long_reason,
                            "reason_risk": "blended_oos_ic=-0.024; signals_scored=17; model_confidence=low",
                        },
                        "suggested_next_checks": [],
                    },
                }
            ],
        },
    )

    memory = out["_memory_context"][0]
    assert memory["event_id"] == "mem_007610"
    assert memory["type"] == "candidate_watchlist"
    assert memory["t"] == "007610"
    assert memory["src"] == "recommend_opportunities:aggressive"
    assert memory["checked"] == ["forecast_returns", "get_fundamentals", "technical_signals"]
    assert memory["rank"] == 1
    assert memory["score"] == 0.86605
    assert memory["why"] == long_reason
    assert memory["risk"] == "blended_oos_ic=-0.024; signals_scored=17; model_confidence=low"
    assert "xxx" not in str(memory)
    assert "..." not in memory["why"]


def test_compact_memory_context_generic_keeps_full_summary() -> None:
    summary = "Risk lesson: " + ("position sizing discipline matters. " * 20)
    out = _compact_tool_result_for_prompt(
        "macro_snapshot",
        {
            "status": "ok",
            "_memory_context": [
                {
                    "event_id": "mem_lesson",
                    "created_date": "2026-05-01",
                    "event_type": "strategy_reflection",
                    "summary": summary,
                    "importance_score": 0.7,
                    "outcome_label": "win",
                }
            ],
        },
    )

    memory = out["_memory_context"][0]
    assert memory["summary"] == summary
    assert not memory["summary"].endswith("...")
    assert memory["outcome_label"] == "win"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py -q -p no:cacheprovider
```

Expected: the new candidate test fails because `_memory_context` only contains compacted `summary`; the generic test fails because long summaries are clipped to 180 chars.

- [ ] **Step 3: Implement `arena/agents/adk_memory_context.py`**

Create the file with these functions:

```python
from __future__ import annotations

import json
from typing import Any


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _clean_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _list(value) if str(item).strip()]


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _candidate_memory(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    evidence = _dict(payload.get("discovery_evidence"))
    out: dict[str, Any] = {
        "event_id": row.get("event_id"),
        "d": row.get("created_date"),
        "type": row.get("event_type") or payload.get("candidate_status"),
        "t": payload.get("ticker"),
        "src": (_clean_list(payload.get("source_tools")) or [None])[0],
        "checked": _clean_list(payload.get("analyzed_by")),
        "rank": payload.get("last_seen_rank"),
        "score": _first_present(evidence.get("score"), row.get("score"), row.get("importance_score")),
        "why": evidence.get("reason_for") or evidence.get("reason"),
        "risk": evidence.get("reason_risk"),
        "outcome_label": row.get("outcome_label"),
    }
    return {key: value for key, value in out.items() if value not in (None, "", [])}


def _generic_memory(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "event_id": row.get("event_id"),
        "created_date": row.get("created_date"),
        "event_type": row.get("event_type"),
        "summary": row.get("summary"),
        "importance_score": _first_present(row.get("importance_score"), row.get("score")),
        "outcome_label": row.get("outcome_label"),
        "memory_source": row.get("memory_source") or payload.get("source"),
        "memory_tier": row.get("memory_tier"),
        "ticker": payload.get("ticker") or _dict(payload.get("intent")).get("ticker"),
    }
    return {key: value for key, value in out.items() if value not in (None, "", [])}


def model_memory_context_rows(rows: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for raw in rows[: max(1, int(limit))]:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        payload = _dict(row.get("payload") or row.get("payload_json"))
        event_type = str(row.get("event_type") or "").strip().lower()
        source = str(payload.get("source") or "").strip().lower()
        if event_type.startswith("candidate_") or source == "candidate_discovery":
            item = _candidate_memory(row, payload)
        else:
            item = _generic_memory(row, payload)
        if item:
            out.append(item)
    return out
```

- [ ] **Step 4: Wire formatter into compaction**

Modify `arena/agents/adk_tool_compaction.py`:

```python
from arena.agents.adk_memory_context import model_memory_context_rows
```

Replace `_compact_memory_context_rows()` with:

```python
def _compact_memory_context_rows(rows: Any) -> list[dict[str, Any]]:
    return model_memory_context_rows(rows, limit=3)
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py -q -p no:cacheprovider
```

Expected: all tests in `tests/adk/test_tool_compaction.py` pass.

- [ ] **Step 6: Commit**

```bash
git add arena/agents/adk_memory_context.py arena/agents/adk_tool_compaction.py tests/adk/test_tool_compaction.py
git commit -m "fix: structure model memory context"
```

---

### Task 2: Preserve and Enrich Memory Payloads Before Compaction

**Files:**
- Modify: `arena/agents/adk_runner_runtime.py`
- Test: `tests/adk/test_memory_pipeline.py`

- [ ] **Step 1: Write failing tests for full summary and legacy enrichment**

Add these helpers and tests to `tests/adk/test_memory_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_memory_pipeline.py -q -p no:cacheprovider
```

Expected: full-summary test fails because summary is sliced; enrichment test fails because `payload` is not loaded from repo.

- [ ] **Step 3: Add enrichment helpers in `adk_runner_runtime.py`**

Add helper functions above `search_tool_memories()`:

```python
def _payload_from_memory_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload")
    if isinstance(payload, dict):
        return dict(payload)
    payload_json = row.get("payload_json")
    if isinstance(payload_json, dict):
        return dict(payload_json)
    if isinstance(payload_json, str) and payload_json.strip():
        try:
            parsed = json.loads(payload_json)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _enrich_tool_memory_hit(memory_store: Any, memory: dict[str, Any]) -> dict[str, Any]:
    row = dict(memory)
    if row.get("payload") or row.get("payload_json") or not row.get("event_id"):
        return row
    repo = getattr(memory_store, "repo", None)
    loader = getattr(repo, "memory_event_by_id", None)
    if not callable(loader):
        return row
    try:
        full = loader(event_id=str(row.get("event_id")), tenant_id=memory_store._tenant())
    except Exception:
        return row
    if not isinstance(full, dict):
        return row
    for key in ("summary", "event_type", "payload_json", "score", "importance_score", "memory_source", "memory_tier"):
        if full.get(key) is not None:
            row[key] = full.get(key)
    payload = _payload_from_memory_row(row)
    if payload:
        row["payload"] = payload
    return row
```

Add `import json` at the top of `arena/agents/adk_runner_runtime.py`.

- [ ] **Step 4: Preserve full summary and payload in `search_tool_memories()`**

In the loop inside `search_tool_memories()`, call enrichment before building the row:

```python
memory = _enrich_tool_memory_hit(memory_store, memory)
payload = _payload_from_memory_row(memory)
```

Replace the row construction with:

```python
row: dict[str, Any] = {
    "event_id": str(memory.get("event_id") or "").strip() or None,
    "summary": str(memory.get("summary") or ""),
    "importance_score": (
        memory.get("importance_score")
        if memory.get("importance_score") is not None
        else memory.get("score", 0.5)
    ),
}
for key in ("event_type", "score", "memory_source", "memory_tier", "agent_id"):
    if memory.get(key) is not None:
        row[key] = memory.get(key)
if payload:
    row["payload"] = payload
```

Keep the existing created-date and outcome-label logic.

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_memory_pipeline.py -q -p no:cacheprovider
```

Expected: all memory pipeline tests pass.

- [ ] **Step 6: Commit**

```bash
git add arena/agents/adk_runner_runtime.py tests/adk/test_memory_pipeline.py
git commit -m "fix: enrich tool memory hits"
```

---

### Task 3: Store Payloads in Vector Metadata for New Memories

**Files:**
- Modify: `arena/memory/vector.py`
- Modify: `arena/memory/vector_local.py`
- Modify: `arena/memory/store.py`
- Test: `tests/adk/test_memory_pipeline.py`

- [ ] **Step 1: Write failing test for `record_memory()` passing payload to vector store**

Add this test to `tests/adk/test_memory_pipeline.py`:

```python
def test_record_memory_passes_payload_to_vector_store() -> None:
    calls: list[dict] = []

    class Repo:
        def write_memory_event(self, event):
            self.event = event

    class VectorStore:
        def save_memory_vector(self, **kwargs):
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
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_memory_pipeline.py::test_record_memory_passes_payload_to_vector_store -q -p no:cacheprovider
```

Expected: fails because `payload` is not passed to `save_memory_vector()`.

- [ ] **Step 3: Update vector store signatures and return rows**

In `arena/memory/vector.py`, add `payload: dict[str, Any] | None = None` to `save_memory_vector()`. When building `doc_data`, store payload:

```python
if payload:
    doc_data["payload"] = dict(payload)
```

In `_memory_row_from_doc()`, return it:

```python
payload = data.get("payload")
if isinstance(payload, dict):
    row["payload"] = dict(payload)
```

In `arena/memory/vector_local.py`, add `import json`, add `payload: dict[str, Any] | None = None` to `save_memory_vector()`, and store compact JSON metadata:

```python
if payload:
    metadata["payload_json"] = json.dumps(payload, ensure_ascii=False, default=str)
```

In `_row()`, parse it:

```python
payload_json = meta.get("payload_json")
if isinstance(payload_json, str) and payload_json.strip():
    try:
        parsed = json.loads(payload_json)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        row["payload"] = parsed
        row["payload_json"] = payload_json
```

- [ ] **Step 4: Pass payload from `MemoryStore.record_memory()`**

In `arena/memory/store.py`, update the `save_memory_vector()` call:

```python
payload=event.payload,
```

Place it with the other semantic metadata arguments.

- [ ] **Step 5: Run focused tests**

Run:

```bash
TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_memory_pipeline.py -q -p no:cacheprovider
```

Expected: all memory pipeline tests pass.

- [ ] **Step 6: Commit**

```bash
git add arena/memory/vector.py arena/memory/vector_local.py arena/memory/store.py tests/adk/test_memory_pipeline.py
git commit -m "fix: store memory payloads in vector index"
```

---

### Task 4: Verify Shared Injection Path Across Tool Types

**Files:**
- Test: `tests/adk/test_tool_compaction.py`

- [ ] **Step 1: Extend wrapper test to prove all injectable tools use the same improved path**

Update `test_tool_wrapper_injects_memory_for_macro_tools_with_typed_query()` in `tests/adk/test_tool_compaction.py` so `search_tool_memories()` returns a candidate-style payload:

```python
def search_tool_memories(query):
    captured["query"] = query
    return [
        {
            "event_id": "mem_macro",
            "event_type": "candidate_watchlist",
            "created_date": "2026-05-07",
            "payload": {
                "source": "candidate_discovery",
                "ticker": "007610",
                "candidate_status": "watchlist",
                "source_tools": ["recommend_opportunities:aggressive"],
                "analyzed_by": ["forecast_returns"],
                "last_seen_rank": 1,
                "discovery_evidence": {
                    "score": 0.86605,
                    "reason_for": "Full reason survives in macro tool memory context.",
                    "reason_risk": "Full risk survives too.",
                },
            },
        }
    ]
```

Replace the assertion:

```python
assert out["_memory_context"][0]["summary"] == "High-rate regimes require smaller gross exposure."
```

with:

```python
memory = out["_memory_context"][0]
assert memory["t"] == "007610"
assert memory["why"] == "Full reason survives in macro tool memory context."
assert memory["risk"] == "Full risk survives too."
```

- [ ] **Step 2: Run wrapper/compaction tests**

Run:

```bash
TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py -q -p no:cacheprovider
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add tests/adk/test_tool_compaction.py
git commit -m "test: cover structured memory context injection"
```

---

### Task 5: Run Integration-Level Verification

**Files:**
- No code changes unless verification exposes a regression.

- [ ] **Step 1: Run focused ADK tests**

Run:

```bash
TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_tool_compaction.py tests/adk/test_memory_pipeline.py tests/adk/test_candidate_ledger.py -q -p no:cacheprovider
```

Expected: pass.

- [ ] **Step 2: Run memory store execution tests**

Run:

```bash
TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/memory/test_memory_store_execution.py -q -p no:cacheprovider
```

Expected: pass.

- [ ] **Step 3: Optionally inspect one live audit row after next agent cycle**

After deployment and one Cloud Run job, verify the model-visible result:

```bash
bq --project_id=rising-parser-464807-f6 query --use_legacy_sql=false --format=prettyjson '
SELECT created_at, agent_id, tool_name, model_visible_result_json
FROM `rising-parser-464807-f6.llm_arena.agent_llm_tool_events`
WHERE created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
  AND tool_name = "recommend_opportunities"
  AND JSON_QUERY(model_visible_result_json, "$._memory_context") IS NOT NULL
ORDER BY created_at DESC
LIMIT 3'
```

Expected: `_memory_context` entries contain lean structured fields such as `t`, `why`, `risk`, `src`, and do not contain `Reas...` style mid-word truncation.

- [ ] **Step 4: Final commit if verification required fixes**

If any verification changes were needed:

```bash
git add arena tests
git commit -m "fix: stabilize structured memory context"
```

If no changes were needed, do not create an empty commit.

---

## Self-Review

- Spec coverage: The plan removes blind memory summary truncation, keeps context compact through structure, and applies to all injected tools through the shared compaction path.
- Placeholder scan: No TBD/TODO placeholders remain.
- Type consistency: `payload` is represented as `dict[str, Any]` in runtime rows and Firestore, with `payload_json` used only for local vector metadata and legacy BigQuery rows.
- Scope check: This is one subsystem: REACT tool memory context. It does not change the recommendation ranker or main tool-result compaction for non-memory fields.
