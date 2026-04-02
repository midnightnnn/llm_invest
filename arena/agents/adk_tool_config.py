from __future__ import annotations

import json
import logging
import re
from typing import Any

from arena.config import AgentConfig
from arena.data.bq import BigQueryRepository

logger = logging.getLogger(__name__)

try:
    from google.adk.tools.mcp_tool import (
        McpToolset,
        SseConnectionParams,
        StreamableHTTPConnectionParams,
    )
except Exception:  # pragma: no cover - optional dependency/runtime environment
    McpToolset = None  # type: ignore[assignment]
    SseConnectionParams = None  # type: ignore[assignment]
    StreamableHTTPConnectionParams = None  # type: ignore[assignment]


def _parse_json_list(raw: str | None, *, field_name: str = "json_list") -> list[Any]:
    """Safely parses a JSON array; returns [] on parse/type mismatch."""
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception as exc:
        logger.warning(
            "[yellow]tool config JSON parse failed[/yellow] field=%s err=%s",
            field_name,
            str(exc),
        )
        return []
    if isinstance(parsed, list):
        return parsed
    logger.warning(
        "[yellow]tool config JSON type mismatch[/yellow] field=%s expected=array actual=%s",
        field_name,
        type(parsed).__name__,
    )
    return []


def _load_disabled_tool_ids(repo: BigQueryRepository, tenant_id: str) -> set[str]:
    """Loads disabled optional tools by tool_id from arena_config."""
    try:
        raw = repo.get_config(tenant_id, "disabled_tools")
    except Exception as exc:
        logger.warning(
            "[yellow]disabled_tools load failed[/yellow] tenant=%s err=%s",
            tenant_id,
            str(exc),
        )
        return set()

    out: set[str] = set()
    for item in _parse_json_list(raw, field_name="disabled_tools"):
        token = str(item or "").strip()
        if token:
            out.add(token)
    return out


def _resolve_disabled_tool_ids(
    repo: BigQueryRepository,
    tenant_id: str,
    agent_config: AgentConfig | None = None,
) -> set[str]:
    """Returns disabled tool ids, using per-agent override when present."""
    if agent_config and agent_config.disabled_tools is not None:
        return set(agent_config.disabled_tools)
    return _load_disabled_tool_ids(repo, tenant_id)


def _load_mcp_toolsets(repo: BigQueryRepository, tenant_id: str) -> list[Any]:
    """Builds MCP toolsets from tenant config."""
    if McpToolset is None or SseConnectionParams is None or StreamableHTTPConnectionParams is None:
        return []
    try:
        raw = repo.get_config(tenant_id, "mcp_servers")
    except Exception as exc:
        logger.warning(
            "[yellow]mcp_servers load failed[/yellow] tenant=%s err=%s",
            tenant_id,
            str(exc),
        )
        return []

    rows = _parse_json_list(raw, field_name="mcp_servers")
    toolsets: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        enabled_raw = row.get("enabled", True)
        if isinstance(enabled_raw, str):
            enabled = enabled_raw.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            enabled = bool(enabled_raw)
        if not enabled:
            continue
        url = str(row.get("url") or "").strip()
        if not url:
            continue

        transport = str(row.get("transport") or "sse").strip().lower()
        name = str(row.get("name") or "mcp").strip() or "mcp"
        tool_name_prefix = re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_") or "mcp"
        try:
            if transport == "streamable_http":
                params = StreamableHTTPConnectionParams(url=url)
            elif transport == "sse":
                params = SseConnectionParams(url=url)
            else:
                logger.warning(
                    "[yellow]Unsupported MCP transport skipped[/yellow] tenant=%s name=%s transport=%s",
                    tenant_id,
                    name,
                    transport,
                )
                continue
            toolsets.append(
                McpToolset(
                    connection_params=params,
                    tool_name_prefix=tool_name_prefix,
                )
            )
        except Exception as exc:
            logger.warning(
                "[yellow]MCP toolset init failed[/yellow] tenant=%s name=%s url=%s err=%s",
                tenant_id,
                name,
                url,
                str(exc),
            )
    return toolsets
