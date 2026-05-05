# Test Quality Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable test infrastructure and migrate two small BigQuery-backed tests to prove the pattern.

**Architecture:** Create a test-only helper package under `tests/helpers/` with reusable BigQuery session/client fakes. Keep `tests/conftest.py` small and focused on common fixtures. Update pytest markers, then migrate `test_execution_store.py` and `test_memory_bq_store.py` without changing production behavior.

**Tech Stack:** Python 3.12, pytest, existing `arena.data.bigquery` stores.

---

## File Structure

- Create `tests/helpers/__init__.py`
  - Marks `tests.helpers` as a test helper package.
- Create `tests/helpers/bigquery.py`
  - Provides reusable BigQuery-style fake client/session classes and small call records.
- Modify `tests/conftest.py`
  - Keeps root path setup and adds conservative shared fixtures.
- Modify `pyproject.toml`
  - Adds explicit pytest marker definitions.
- Modify `tests/test_execution_store.py`
  - Replaces local `_FakeSession` with `FakeBigQuerySession`.
- Modify `tests/test_memory_bq_store.py`
  - Replaces local `_FakeClient` and `_FakeSession` with `FakeBigQuerySession`.

---

### Task 1: Add BigQuery Test Helpers

**Files:**
- Create: `tests/helpers/__init__.py`
- Create: `tests/helpers/bigquery.py`
- Test: `tests/test_execution_store.py`

- [ ] **Step 1: Write the helper module**

Create `tests/helpers/__init__.py`:

```python
"""Shared test helpers for the LLM Arena test suite."""
```

Create `tests/helpers/bigquery.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BigQueryCall:
    """Recorded SQL call made through a fake BigQuery session."""

    sql: str
    params: dict[str, Any] | None


class FakeInsertClient:
    """Minimal BigQuery client fake that records streaming inserts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    @property
    def inserts(self) -> list[tuple[str, list[dict[str, Any]]]]:
        return self.calls

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [row for _, rows in self.calls for row in rows]

    def insert_rows_json(
        self,
        table_id: str,
        rows: list[dict[str, Any]],
        row_ids: object | None = None,
    ) -> list[dict[str, Any]]:
        _ = row_ids
        self.calls.append((table_id, list(rows)))
        return []


class FakeLoadJob:
    """Minimal load job fake returned by fake table-load clients."""

    def result(self) -> None:
        return None


class FakeBigQuerySession:
    """Small BigQuerySession test double with call recording."""

    def __init__(
        self,
        *,
        project: str = "proj",
        dataset: str = "ds",
        tenant_id: str = "tenant-a",
        client: object | None = None,
        fetch_result: list[dict[str, Any]] | None = None,
        fetch_results: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.project = project
        self.dataset = dataset
        self.dataset_fqn = f"{project}.{dataset}"
        self.tenant_id = tenant_id
        self.client = client or FakeInsertClient()
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.fetched: list[tuple[str, dict[str, Any]]] = []
        self.execute_calls: list[BigQueryCall] = []
        self.fetch_calls: list[BigQueryCall] = []
        self.fetch_result = list(fetch_result or [])
        self.fetch_results = [list(rows) for rows in (fetch_results or [])]

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return str(tenant_id or self.tenant_id)

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        copied = dict(params or {})
        self.executed.append((sql, copied))
        self.execute_calls.append(BigQueryCall(sql=sql, params=copied))

    def fetch_rows(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        copied = dict(params or {})
        self.fetched.append((sql, copied))
        self.fetch_calls.append(BigQueryCall(sql=sql, params=copied))
        if self.fetch_results:
            return list(self.fetch_results.pop(0))
        return list(self.fetch_result)
```

- [ ] **Step 2: Run a focused import check**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_execution_store.py -q -p no:cacheprovider
```

Expected: PASS. The new helper is not imported yet, so this confirms no package/import side effects.

- [ ] **Step 3: Commit**

```bash
git add tests/helpers/__init__.py tests/helpers/bigquery.py
git commit -m "test: add shared BigQuery fakes"
```

---

### Task 2: Add Conservative Shared Pytest Fixtures

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/test_execution_store.py`

- [ ] **Step 1: Replace `tests/conftest.py` with the expanded version**

Use this complete file content:

```python
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from tests.helpers.bigquery import FakeBigQuerySession

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tenant_id() -> str:
    return "tenant-a"


@pytest.fixture
def fixed_utc_now() -> datetime:
    return datetime(2026, 3, 29, 12, 30, tzinfo=timezone.utc)


@pytest.fixture
def fake_bq_session_factory() -> Callable[..., FakeBigQuerySession]:
    return FakeBigQuerySession
```

- [ ] **Step 2: Run a focused test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_execution_store.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared pytest fixtures"
```

---

### Task 3: Register Pytest Markers

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/integration/test_embed.py`

- [ ] **Step 1: Update marker definitions**

Replace the existing marker block in `pyproject.toml`:

```toml
markers = [
    "integration: opt-in tests that require external services such as BigQuery or Vertex AI",
]
```

with:

```toml
markers = [
    "unit: fast isolated tests with no external service dependency",
    "integration: local integration tests or opt-in service-bound tests",
    "live: tests requiring real external credentials or networked services",
    "slow: deterministic tests that are too slow for default focused runs",
]
```

- [ ] **Step 2: Verify marker parsing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/integration/test_embed.py --collect-only -q -p no:cacheprovider
```

Expected: collection succeeds and reports the integration test node without unknown-marker warnings.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "test: register pytest marker taxonomy"
```

---

### Task 4: Migrate Execution Store Test To Shared Session Fake

**Files:**
- Modify: `tests/test_execution_store.py`
- Test: `tests/test_execution_store.py`

- [ ] **Step 1: Replace local fake session with shared helper import**

Replace the top of `tests/test_execution_store.py` with:

```python
from __future__ import annotations

from arena.data.bigquery.execution_store import ExecutionStore
from arena.models import OrderIntent, RiskDecision, Side
from tests.helpers.bigquery import FakeBigQuerySession
```

Delete the local `_FakeSession` class.

- [ ] **Step 2: Update session construction**

In `test_write_order_intent_persists_cycle_and_llm_call_ids`, replace:

```python
session = _FakeSession()
```

with:

```python
session = FakeBigQuerySession()
```

In `test_recent_trade_history_joins_execution_reports_to_order_intents`, replace:

```python
session = _FakeSession()
session.fetch_result = [{"order_id": "order-1", "rationale": "why"}]
```

with:

```python
session = FakeBigQuerySession(fetch_result=[{"order_id": "order-1", "rationale": "why"}])
```

- [ ] **Step 3: Run the migrated test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_execution_store.py -q -p no:cacheprovider
```

Expected: `2 passed`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_execution_store.py
git commit -m "test: reuse BigQuery fake in execution store tests"
```

---

### Task 5: Migrate Memory BigQuery Store Test To Shared Session Fake

**Files:**
- Modify: `tests/test_memory_bq_store.py`
- Test: `tests/test_memory_bq_store.py`

- [ ] **Step 1: Replace local fake classes with shared helper import**

Replace the import block in `tests/test_memory_bq_store.py` with:

```python
from __future__ import annotations

from datetime import datetime, timezone

from arena.data.bigquery.memory_bq_store import MemoryBQStore
from arena.models import MemoryEvent
from tests.helpers.bigquery import FakeBigQuerySession
```

Delete the local `_FakeClient` and `_FakeSession` classes.

- [ ] **Step 2: Replace session construction**

Replace every occurrence of:

```python
session = _FakeSession()
```

with:

```python
session = FakeBigQuerySession()
```

The assertions using `session.client.inserts`, `session.executed`, and `session.fetched` should remain valid because `FakeInsertClient.inserts` aliases recorded insert calls.

- [ ] **Step 3: Run the migrated test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_memory_bq_store.py -q -p no:cacheprovider
```

Expected: all tests in `tests/test_memory_bq_store.py` pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_memory_bq_store.py
git commit -m "test: reuse BigQuery fake in memory store tests"
```

---

### Task 6: Run Foundation Verification

**Files:**
- Verify only.

- [ ] **Step 1: Run focused foundation tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_execution_store.py tests/test_memory_bq_store.py tests/test_ledger_repo.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 2: Check final diff scope**

Run:

```bash
git status --short
git diff -- tests/helpers tests/conftest.py tests/test_execution_store.py tests/test_memory_bq_store.py pyproject.toml
```

Expected: only the intended test infrastructure and migrated test files are changed. Existing unrelated user changes may still appear in `git status`, but should not be modified by this plan.

- [ ] **Step 3: Commit any remaining verification-only adjustments**

If the focused tests required small corrections, commit only those touched files:

```bash
git add tests/helpers tests/conftest.py tests/test_execution_store.py tests/test_memory_bq_store.py pyproject.toml
git commit -m "test: verify test quality foundation"
```

If there are no remaining changes, skip this commit.
