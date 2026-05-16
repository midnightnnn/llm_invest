# Runtime Clock Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hidden rule-based cycle supervision while exposing only current wall-clock time to LLM agents.

**Architecture:** ADK remains the execution engine. The LLM sees `_runtime_clock.now_kst` in phase prompts and tool responses, but never sees Supervisor names, policy names, deadline instructions, or allowed/blocked action lists. Deterministic Supervisor logic stays in Python and later enforces model-call watchdogs, phase timing, and order cutoff behavior.

**Tech Stack:** Python 3.12, Google ADK Runner/Agent callbacks, LiteLLM client wrapper, pytest.

---

## File Structure

- Create `arena/agents/runtime_clock.py`: small, deterministic helper for KST clock payloads.
- Modify `arena/prompts/prompt_pack.py`: include `_runtime_clock` in decision and resume payloads when present in context.
- Modify `arena/agents/adk_agents.py`: attach fresh `_runtime_clock` to context before each phase prompt is built.
- Modify `arena/agents/adk_runner_bootstrap.py`: attach fresh `_runtime_clock` to compacted tool responses returned to the model.
- Later create `arena/agents/cycle_supervisor.py`: hidden rule-based operation timing and watchdog policy. This file must not generate prompt text.
- Test `tests/adk/test_runtime_clock.py`: unit tests for helper behavior.
- Test `tests/test_prompt_pack.py` or `tests/adk/test_runner_runtime.py`: prompt/context integration.
- Test `tests/adk/test_model_resolution.py`: later watchdog behavior in LiteLLM wrapper.

---

### Task 1: Runtime Clock Helper

**Files:**
- Create: `arena/agents/runtime_clock.py`
- Test: `tests/adk/test_runtime_clock.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from arena.agents.runtime_clock import build_runtime_clock, with_runtime_clock


def test_build_runtime_clock_uses_kst_iso_timestamp() -> None:
    now = datetime(2026, 5, 15, 6, 25, 14, tzinfo=timezone.utc)

    payload = build_runtime_clock(now=now)

    assert payload == {
        "now_kst": "2026-05-15T15:25:14+09:00",
    }


def test_with_runtime_clock_returns_copy_without_mutating_original() -> None:
    context = {"cycle_phase": "execution"}
    now = datetime(2026, 5, 15, 15, 26, 0, tzinfo=ZoneInfo("Asia/Seoul"))

    updated = with_runtime_clock(context, now=now)

    assert context == {"cycle_phase": "execution"}
    assert updated["cycle_phase"] == "execution"
    assert updated["_runtime_clock"] == {
        "now_kst": "2026-05-15T15:26:00+09:00",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_runtime_clock.py -q -p no:cacheprovider`

Expected: FAIL with `ModuleNotFoundError: No module named 'arena.agents.runtime_clock'`.

- [ ] **Step 3: Implement the helper**

```python
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

KST = ZoneInfo("Asia/Seoul")


def build_runtime_clock(*, now: datetime | None = None) -> dict[str, str]:
    instant = now or datetime.now(tz=KST)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=KST)
    return {"now_kst": instant.astimezone(KST).replace(microsecond=0).isoformat()}


def with_runtime_clock(context: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    updated = dict(context)
    updated["_runtime_clock"] = build_runtime_clock(now=now)
    return updated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_runtime_clock.py -q -p no:cacheprovider`

Expected: `2 passed`.

---

### Task 2: Phase Prompt Clock Injection

**Files:**
- Modify: `arena/agents/adk_agents.py`
- Modify: `arena/prompts/prompt_pack.py`
- Test: `tests/test_prompt_pack.py`

- [ ] **Step 1: Write prompt payload test**

Add this test to `tests/test_prompt_pack.py`:

```python
def test_decision_payload_includes_runtime_clock_when_present() -> None:
    context = {
        "cycle_phase": "execution",
        "_runtime_clock": {"now_kst": "2026-05-15T15:25:14+09:00"},
    }

    payload = PromptPack.decision_payload(context, max_tool_calls=10)

    assert payload["_runtime_clock"] == {
        "now_kst": "2026-05-15T15:25:14+09:00",
    }
    assert "supervisor" not in str(payload).lower()
    assert "deadline" not in str(payload).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_prompt_pack.py::test_decision_payload_includes_runtime_clock_when_present -q -p no:cacheprovider`

Expected: FAIL with `KeyError: '_runtime_clock'`.

- [ ] **Step 3: Include `_runtime_clock` in prompt payload**

In `PromptPack.decision_payload`, after `payload` is built and before relation/graph context handling:

```python
        runtime_clock = context.get("_runtime_clock")
        if isinstance(runtime_clock, dict) and runtime_clock:
            payload["_runtime_clock"] = runtime_clock
```

- [ ] **Step 4: Attach fresh runtime clock before prompt generation**

In `arena/agents/adk_agents.py`, import:

```python
from arena.agents.runtime_clock import with_runtime_clock
```

In `_ADKDecisionRunner.runner.decide_orders`, before `self._current_context = context`, replace the incoming context with a copied clock-bearing context:

```python
        context = with_runtime_clock(context)
        self._current_context = context
```

Make sure subsequent calls in the method use the new local `context`.

- [ ] **Step 5: Run focused tests**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_prompt_pack.py tests/adk/test_runner_runtime.py -q -p no:cacheprovider`

Expected: all selected tests pass.

---

### Task 3: Tool Response Clock Injection

**Files:**
- Modify: `arena/agents/adk_runner_bootstrap.py`
- Test: `tests/adk/test_runtime_clock.py`

- [ ] **Step 1: Add tool response helper tests**

Extend `tests/adk/test_runtime_clock.py`:

```python
from arena.agents.runtime_clock import attach_runtime_clock


def test_attach_runtime_clock_adds_reserved_key_to_dict_result() -> None:
    result = {"ticker": "005930", "price": 70000}
    clock = {"now_kst": "2026-05-15T15:26:18+09:00"}

    updated = attach_runtime_clock(result, clock=clock)

    assert updated == {
        "ticker": "005930",
        "price": 70000,
        "_runtime_clock": clock,
    }
    assert "_runtime_clock" not in result


def test_attach_runtime_clock_wraps_non_dict_result() -> None:
    clock = {"now_kst": "2026-05-15T15:26:18+09:00"}

    updated = attach_runtime_clock(["005930"], clock=clock)

    assert updated == {
        "result": ["005930"],
        "_runtime_clock": clock,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_runtime_clock.py -q -p no:cacheprovider`

Expected: FAIL with `ImportError` or `AttributeError` for `attach_runtime_clock`.

- [ ] **Step 3: Implement tool response helper**

Add to `arena/agents/runtime_clock.py`:

```python
def attach_runtime_clock(value: Any, *, clock: dict[str, str] | None = None) -> dict[str, Any]:
    runtime_clock = clock or build_runtime_clock()
    if isinstance(value, dict):
        updated = dict(value)
        updated["_runtime_clock"] = runtime_clock
        return updated
    return {"result": value, "_runtime_clock": runtime_clock}
```

- [ ] **Step 4: Attach clock after compaction**

In `arena/agents/adk_runner_bootstrap.py`, import:

```python
from arena.agents.runtime_clock import attach_runtime_clock
```

After:

```python
        compact_res = _compact_tool_result_for_prompt(name, res, args=args_preview)
```

add:

```python
        compact_res = attach_runtime_clock(compact_res)
```

Keep `append_builtin_tool_event(...)` using the original raw `res` so audit/debug data is not changed by clock decoration.

- [ ] **Step 5: Run focused tests**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_runtime_clock.py tests/adk/test_runner_runtime.py -q -p no:cacheprovider`

Expected: all selected tests pass.

---

### Task 4: Hidden Supervisor Skeleton

**Files:**
- Create: `arena/agents/cycle_supervisor.py`
- Test: `tests/adk/test_cycle_supervisor.py`

- [ ] **Step 1: Write behavior tests**

```python
from __future__ import annotations

from arena.agents.cycle_supervisor import AgentCycleSupervisor


def test_supervisor_records_operations_without_prompt_text() -> None:
    supervisor = AgentCycleSupervisor(cycle_id="cycle_1")

    op_id = supervisor.start_operation(kind="model_call", phase="explore", agent_id="claude")
    supervisor.finish_operation(op_id)
    summary = supervisor.summary()

    assert summary["cycle_id"] == "cycle_1"
    assert summary["operation_count"] == 1
    assert "prompt" not in str(summary).lower()
    assert "supervisor" not in summary.get("llm_visible_text", "")


def test_supervisor_model_call_timeout_policy_defaults_to_300_seconds() -> None:
    supervisor = AgentCycleSupervisor(cycle_id="cycle_1")

    assert supervisor.model_call_timeout_seconds(provider="claude", model="claude-opus-4-7") == 300
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_cycle_supervisor.py -q -p no:cacheprovider`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement minimal hidden supervisor**

```python
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OperationRecord:
    operation_id: str
    kind: str
    phase: str
    agent_id: str
    started_monotonic: float
    finished_monotonic: float | None = None


@dataclass
class AgentCycleSupervisor:
    cycle_id: str
    default_model_call_timeout_seconds: int = 300
    _operations: dict[str, OperationRecord] = field(default_factory=dict)

    def start_operation(self, *, kind: str, phase: str, agent_id: str) -> str:
        operation_id = f"{kind}_{phase}_{agent_id}_{len(self._operations) + 1}"
        self._operations[operation_id] = OperationRecord(
            operation_id=operation_id,
            kind=kind,
            phase=phase,
            agent_id=agent_id,
            started_monotonic=time.monotonic(),
        )
        return operation_id

    def finish_operation(self, operation_id: str) -> None:
        self._operations[operation_id].finished_monotonic = time.monotonic()

    def model_call_timeout_seconds(self, *, provider: str, model: str) -> int:
        _ = provider, model
        return self.default_model_call_timeout_seconds

    def summary(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "operation_count": len(self._operations),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_cycle_supervisor.py -q -p no:cacheprovider`

Expected: `2 passed`.

---

### Task 5: Model-Call Watchdog Uses Hidden Policy

**Files:**
- Modify: `arena/agents/adk_models.py`
- Test: `tests/adk/test_model_resolution.py`

- [ ] **Step 1: Add watchdog regression test**

Add to `tests/adk/test_model_resolution.py`:

```python
def test_instrumented_litellm_client_times_out_slow_delegate() -> None:
    class _Delegate:
        async def acompletion(self, *, model, messages, tools, **kwargs):
            await asyncio.sleep(10)

    client = _InstrumentedLiteLLMClient(
        agent_id="claude",
        provider="claude",
        metadata_getter=lambda: {"llm_call_id": "llm-timeout", "phase": "explore"},
        delegate=_Delegate(),
        model_call_timeout_seconds_getter=lambda model: 0.01,
    )

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(
            client.acompletion(
                model="anthropic/claude-opus-4-7",
                messages=[],
                tools=None,
            )
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_model_resolution.py::test_instrumented_litellm_client_times_out_slow_delegate -q -p no:cacheprovider`

Expected: FAIL because `_InstrumentedLiteLLMClient` does not accept `model_call_timeout_seconds_getter`.

- [ ] **Step 3: Implement watchdog inside `_InstrumentedLiteLLMClient`**

Add an optional constructor argument:

```python
model_call_timeout_seconds_getter: Callable[[str], float | int | None] | None = None
```

Wrap the delegate/super call:

```python
            call_coro = (
                self._delegate.acompletion(model=model, messages=messages, tools=tools, **kwargs)
                if self._delegate is not None
                else super().acompletion(model=model, messages=messages, tools=tools, **kwargs)
            )
            watchdog_timeout = (
                self._model_call_timeout_seconds_getter(str(model or ""))
                if callable(self._model_call_timeout_seconds_getter)
                else None
            )
            if watchdog_timeout and float(watchdog_timeout) > 0:
                response = await asyncio.wait_for(call_coro, timeout=float(watchdog_timeout))
            else:
                response = await call_coro
```

On `asyncio.TimeoutError`, emit `adk_model_acompletion_timeout` with the same metadata fields as start/error logs.

- [ ] **Step 4: Run focused tests**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk/test_model_resolution.py -q -p no:cacheprovider`

Expected: all selected tests pass.

---

### Task 6: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run ADK suite**

Run: `TMPDIR=/tmp/codex-pytest PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/adk -q -p no:cacheprovider`

Expected: all tests pass.

- [ ] **Step 2: Compile touched modules**

Run: `PYTHONDONTWRITEBYTECODE=1 python -m py_compile arena/agents/runtime_clock.py arena/agents/cycle_supervisor.py arena/agents/adk_agents.py arena/agents/adk_models.py arena/agents/adk_runner_bootstrap.py arena/prompts/prompt_pack.py`

Expected: command exits with status 0 and no output.

- [ ] **Step 3: Check prompt exposure**

Run: `rg -n "Supervisor|supervisor|deadline|new_research_allowed|blocked_actions|allowed_actions" arena/agents/runtime_clock.py arena/prompts/prompt_pack.py arena/agents/adk_agents.py arena/agents/adk_runner_bootstrap.py`

Expected: no matches in prompt/clock injection code except tests or hidden supervisor implementation if explicitly searched.

