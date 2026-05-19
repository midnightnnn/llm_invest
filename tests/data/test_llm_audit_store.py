from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

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


def test_local_repository_appends_llm_audit_rows(tmp_path) -> None:
    pytest.importorskip("duckdb")

    from arena.data.local.repository import LocalRepository

    repo = LocalRepository(tenant_id="tenant-a", db_path=str(tmp_path / "arena.duckdb"))
    repo.ensure_tables()
    created_at = datetime(2026, 4, 1, 1, 2, 3, tzinfo=timezone.utc)

    repo.append_llm_interactions(
        [
            {
                "llm_call_id": "llm_execution_local",
                "cycle_id": "cycle_1",
                "created_at": created_at,
                "completed_at": created_at,
                "agent_id": "gemini",
                "provider": "gemini",
                "model": "gemini-3-flash-preview",
                "phase": "execution",
                "session_id": "sid_1",
                "resume_session": True,
                "trading_mode": "paper",
                "status": "error",
                "system_prompt": "system",
                "user_prompt": "prompt",
                "context_payload_json": {"_runtime_clock": {"timezone": "Asia/Seoul"}},
                "context_sections_json": {"payload_keys": ["_runtime_clock"]},
                "available_tools_json": [{"name": "technical_signals", "parameters": {"type": "object"}}],
                "token_usage_json": {},
                "error_message": "permission denied",
            }
        ],
        tenant_id="tenant-a",
    )
    repo.append_llm_tool_events(
        [
            {
                "llm_call_id": "llm_execution_local",
                "tool_event_id": "tool_local_1",
                "cycle_id": "cycle_1",
                "created_at": created_at,
                "agent_id": "gemini",
                "phase": "execution",
                "tool_name": "technical_signals",
                "args_json": {"ticker": "AAPL"},
                "model_visible_result_json": {"trend": "up"},
                "elapsed_ms": 12,
            }
        ],
        tenant_id="tenant-a",
    )
    repo.append_llm_context_refs(
        [
            {
                "llm_call_id": "llm_execution_local",
                "context_ref_id": "ctx_1",
                "cycle_id": "cycle_1",
                "created_at": created_at,
                "agent_id": "gemini",
                "phase": "execution",
                "source_table": "market_features",
                "source_id": "AAPL|seed_demo",
                "context_role": "market",
                "prompt_section": "market_context",
                "rank": 1,
                "used_in_prompt": True,
                "detail_json": {"ticker": "AAPL"},
            }
        ],
        tenant_id="tenant-a",
    )
    repo.append_llm_artifact_links(
        [
            {
                "llm_call_id": "llm_execution_local",
                "artifact_link_id": "alink_1",
                "cycle_id": "cycle_1",
                "created_at": created_at,
                "agent_id": "gemini",
                "phase": "execution",
                "artifact_table": "agent_memory_events",
                "artifact_id": "mem_1",
                "artifact_role": "tool_summary_memory",
                "detail_json": {"summary": "saved"},
            }
        ],
        tenant_id="tenant-a",
    )

    interaction = repo.fetch_rows(
        """
        SELECT tenant_id, llm_call_id, model, status, context_payload_json, available_tools_json
        FROM agent_llm_interactions
        WHERE llm_call_id = 'llm_execution_local'
        """
    )[0]
    tool_event = repo.fetch_rows(
        """
        SELECT tool_event_id, args_json, model_visible_result_json
        FROM agent_llm_tool_events
        WHERE tool_event_id = 'tool_local_1'
        """
    )[0]
    context_ref = repo.fetch_rows(
        """
        SELECT context_ref_id, detail_json
        FROM agent_llm_context_refs
        WHERE context_ref_id = 'ctx_1'
        """
    )[0]
    artifact_link = repo.fetch_rows(
        """
        SELECT artifact_link_id, detail_json
        FROM agent_llm_artifact_links
        WHERE artifact_link_id = 'alink_1'
        """
    )[0]

    assert interaction["tenant_id"] == "tenant-a"
    assert interaction["model"] == "gemini-3-flash-preview"
    assert interaction["status"] == "error"
    assert json.loads(interaction["context_payload_json"]) == {"_runtime_clock": {"timezone": "Asia/Seoul"}}
    assert json.loads(interaction["available_tools_json"])[0]["name"] == "technical_signals"
    assert json.loads(tool_event["args_json"]) == {"ticker": "AAPL"}
    assert json.loads(tool_event["model_visible_result_json"]) == {"trend": "up"}
    assert json.loads(context_ref["detail_json"]) == {"ticker": "AAPL"}
    assert json.loads(artifact_link["detail_json"]) == {"summary": "saved"}
