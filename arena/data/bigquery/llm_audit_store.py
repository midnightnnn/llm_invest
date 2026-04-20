"""LLM prompt, tool, and artifact audit store."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arena.data.bigquery.session import BigQuerySession


def _json_safe(value: Any) -> Any:
    """Converts nested values to BigQuery streaming-safe primitives."""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_cell(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))


def _text(value: Any) -> str | None:
    token = str(value or "").strip()
    return token or None


class LlmAuditStore:
    """Append-only audit rows for model-visible prompt and tool context."""

    def __init__(self, session: BigQuerySession) -> None:
        self.session = session

    def append_llm_interactions(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        """Appends one row per LLM call."""
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        table_id = f"{self.session.dataset_fqn}.agent_llm_interactions"
        payload_rows: list[dict[str, Any]] = []
        row_ids: list[str] = []
        for row in rows:
            llm_call_id = str(row.get("llm_call_id") or "").strip()
            if not llm_call_id:
                continue
            payload_rows.append(
                {
                    "tenant_id": tenant,
                    "llm_call_id": llm_call_id,
                    "cycle_id": _text(row.get("cycle_id")),
                    "created_at": _json_safe(row.get("created_at")),
                    "completed_at": _json_safe(row.get("completed_at")),
                    "agent_id": str(row.get("agent_id") or "").strip(),
                    "provider": _text(row.get("provider")),
                    "model": _text(row.get("model")),
                    "phase": str(row.get("phase") or "unknown").strip().lower() or "unknown",
                    "session_id": _text(row.get("session_id")),
                    "resume_session": bool(row.get("resume_session")),
                    "trading_mode": str(row.get("trading_mode") or "paper").strip().lower() or "paper",
                    "status": str(row.get("status") or "unknown").strip().lower() or "unknown",
                    "system_prompt": row.get("system_prompt"),
                    "user_prompt": row.get("user_prompt"),
                    "context_payload_json": _json_cell(row.get("context_payload_json")),
                    "context_sections_json": _json_cell(row.get("context_sections_json")),
                    "available_tools_json": _json_cell(row.get("available_tools_json")),
                    "response_text": row.get("response_text"),
                    "response_json": _json_cell(row.get("response_json")),
                    "token_usage_json": _json_cell(row.get("token_usage_json")),
                    "request_hash": _text(row.get("request_hash")),
                    "prompt_version": _text(row.get("prompt_version")),
                    "context_builder_version": _text(row.get("context_builder_version")),
                    "settings_hash": _text(row.get("settings_hash")),
                    "latency_ms": int(row.get("latency_ms") or 0) if row.get("latency_ms") is not None else None,
                    "error_message": _text(row.get("error_message")),
                }
            )
            row_ids.append(llm_call_id)
        if not payload_rows:
            return
        errors = self.session.client.insert_rows_json(table_id, payload_rows, row_ids=row_ids)
        if errors:
            raise RuntimeError(f"agent_llm_interactions insert failed: {errors}")

    def append_llm_tool_events(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        """Appends model-visible tool transcript rows."""
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        table_id = f"{self.session.dataset_fqn}.agent_llm_tool_events"
        payload_rows: list[dict[str, Any]] = []
        row_ids: list[str] = []
        for row in rows:
            tool_event_id = str(row.get("tool_event_id") or "").strip()
            llm_call_id = str(row.get("llm_call_id") or "").strip()
            tool_name = str(row.get("tool_name") or row.get("tool") or "").strip()
            if not tool_event_id or not llm_call_id or not tool_name:
                continue
            payload_rows.append(
                {
                    "tenant_id": tenant,
                    "llm_call_id": llm_call_id,
                    "tool_event_id": tool_event_id,
                    "cycle_id": _text(row.get("cycle_id")),
                    "created_at": _json_safe(row.get("created_at")),
                    "agent_id": str(row.get("agent_id") or "").strip(),
                    "phase": _text(row.get("phase")),
                    "tool_name": tool_name,
                    "source": _text(row.get("source")),
                    "args_json": _json_cell(row.get("args_json")),
                    "model_visible_result_json": _json_cell(row.get("model_visible_result_json")),
                    "raw_result_hash": _text(row.get("raw_result_hash")),
                    "elapsed_ms": int(row.get("elapsed_ms") or 0) if row.get("elapsed_ms") is not None else None,
                    "error": _text(row.get("error")),
                }
            )
            row_ids.append(tool_event_id)
        if not payload_rows:
            return
        errors = self.session.client.insert_rows_json(table_id, payload_rows, row_ids=row_ids)
        if errors:
            raise RuntimeError(f"agent_llm_tool_events insert failed: {errors}")

    def append_llm_context_refs(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        """Appends references to source rows represented in model-visible context."""
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        table_id = f"{self.session.dataset_fqn}.agent_llm_context_refs"
        payload_rows: list[dict[str, Any]] = []
        row_ids: list[str] = []
        for row in rows:
            context_ref_id = str(row.get("context_ref_id") or "").strip()
            llm_call_id = str(row.get("llm_call_id") or "").strip()
            source_table = str(row.get("source_table") or "").strip()
            source_id = str(row.get("source_id") or "").strip()
            if not context_ref_id or not llm_call_id or not source_table or not source_id:
                continue
            payload_rows.append(
                {
                    "tenant_id": tenant,
                    "llm_call_id": llm_call_id,
                    "context_ref_id": context_ref_id,
                    "cycle_id": _text(row.get("cycle_id")),
                    "created_at": _json_safe(row.get("created_at")),
                    "agent_id": str(row.get("agent_id") or "").strip(),
                    "phase": _text(row.get("phase")),
                    "source_table": source_table,
                    "source_id": source_id,
                    "source_ts": _json_safe(row.get("source_ts")),
                    "source_hash": _text(row.get("source_hash")),
                    "context_role": _text(row.get("context_role")),
                    "prompt_section": _text(row.get("prompt_section")),
                    "rank": int(row.get("rank") or 0) if row.get("rank") is not None else None,
                    "used_in_prompt": bool(row.get("used_in_prompt")) if row.get("used_in_prompt") is not None else None,
                    "detail_json": _json_cell(row.get("detail_json")),
                }
            )
            row_ids.append(context_ref_id)
        if not payload_rows:
            return
        errors = self.session.client.insert_rows_json(table_id, payload_rows, row_ids=row_ids)
        if errors:
            raise RuntimeError(f"agent_llm_context_refs insert failed: {errors}")

    def append_llm_artifact_links(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None:
        """Appends links from LLM calls to produced DB artifacts."""
        if not rows:
            return
        tenant = self.session.resolve_tenant_id(tenant_id)
        table_id = f"{self.session.dataset_fqn}.agent_llm_artifact_links"
        payload_rows: list[dict[str, Any]] = []
        row_ids: list[str] = []
        for row in rows:
            artifact_link_id = str(row.get("artifact_link_id") or "").strip()
            llm_call_id = str(row.get("llm_call_id") or "").strip()
            artifact_table = str(row.get("artifact_table") or "").strip()
            artifact_id = str(row.get("artifact_id") or "").strip()
            if not artifact_link_id or not llm_call_id or not artifact_table or not artifact_id:
                continue
            payload_rows.append(
                {
                    "tenant_id": tenant,
                    "llm_call_id": llm_call_id,
                    "artifact_link_id": artifact_link_id,
                    "cycle_id": _text(row.get("cycle_id")),
                    "created_at": _json_safe(row.get("created_at")),
                    "agent_id": str(row.get("agent_id") or "").strip(),
                    "phase": _text(row.get("phase")),
                    "artifact_table": artifact_table,
                    "artifact_id": artifact_id,
                    "artifact_role": _text(row.get("artifact_role")),
                    "detail_json": _json_cell(row.get("detail_json")),
                }
            )
            row_ids.append(artifact_link_id)
        if not payload_rows:
            return
        errors = self.session.client.insert_rows_json(table_id, payload_rows, row_ids=row_ids)
        if errors:
            raise RuntimeError(f"agent_llm_artifact_links insert failed: {errors}")
