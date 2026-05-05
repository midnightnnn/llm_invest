from __future__ import annotations

import json
from datetime import datetime, timezone

from arena.data.bigquery.llm_audit_store import LlmAuditStore


class _FakeClient:
    def __init__(self) -> None:
        self.inserts: list[tuple[str, list[dict], list[str] | None]] = []

    def insert_rows_json(self, table_id: str, rows: list[dict], row_ids=None):
        self.inserts.append((table_id, list(rows), list(row_ids) if row_ids is not None else None))
        return []


class _FakeSession:
    def __init__(self) -> None:
        self.dataset_fqn = "proj.ds"
        self.client = _FakeClient()

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return str(tenant_id or "tenant-a")


def test_append_llm_interactions_serializes_json_and_uses_call_id_row_id() -> None:
    session = _FakeSession()
    store = LlmAuditStore(session)
    created_at = datetime(2026, 4, 1, 1, 2, 3, tzinfo=timezone.utc)

    store.append_llm_interactions(
        [
            {
                "llm_call_id": "llm_execution_abc",
                "cycle_id": "cycle_1",
                "created_at": created_at,
                "completed_at": created_at,
                "agent_id": "gpt",
                "provider": "gpt",
                "model": "gpt-5.2",
                "phase": "execution",
                "session_id": "sid_1",
                "resume_session": True,
                "trading_mode": "paper",
                "status": "ok",
                "system_prompt": "system",
                "user_prompt": "prompt",
                "context_payload_json": {"memory_context": "lesson"},
                "available_tools_json": [{"tool_id": "technical_signals"}],
                "response_json": {"orders": []},
                "token_usage_json": {"prompt_tokens": 10},
            }
        ],
        tenant_id="tenant-a",
    )

    table_id, rows, row_ids = session.client.inserts[0]
    assert table_id == "proj.ds.agent_llm_interactions"
    assert row_ids == ["llm_execution_abc"]
    assert rows[0]["created_at"] == "2026-04-01 01:02:03.000000"
    assert json.loads(rows[0]["context_payload_json"]) == {"memory_context": "lesson"}
    assert json.loads(rows[0]["available_tools_json"]) == [{"tool_id": "technical_signals"}]


def test_append_llm_tool_events_keeps_model_visible_result() -> None:
    session = _FakeSession()
    store = LlmAuditStore(session)

    store.append_llm_tool_events(
        [
            {
                "llm_call_id": "llm_execution_abc",
                "tool_event_id": "tool_1",
                "cycle_id": "cycle_1",
                "created_at": datetime(2026, 4, 1, 1, 2, 3, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "phase": "execution",
                "tool_name": "technical_signals",
                "args_json": {"ticker": "AAPL"},
                "model_visible_result_json": {"trend": "up"},
                "elapsed_ms": 12,
            }
        ],
        tenant_id="tenant-a",
    )

    table_id, rows, row_ids = session.client.inserts[0]
    assert table_id == "proj.ds.agent_llm_tool_events"
    assert row_ids == ["tool_1"]
    assert json.loads(rows[0]["args_json"]) == {"ticker": "AAPL"}
    assert json.loads(rows[0]["model_visible_result_json"]) == {"trend": "up"}
