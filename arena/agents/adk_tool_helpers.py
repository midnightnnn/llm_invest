from __future__ import annotations

import inspect
from typing import Any

from arena.tools.registry import ToolEntry


def apply_tool_schema_metadata(fn: Any, *, entry: ToolEntry, sig: inspect.Signature) -> Any:
    name = str(entry.name or entry.tool_id or getattr(fn, "__name__", "tool")).strip() or "tool"
    description = str(entry.description or "").strip()
    fn.__name__ = name
    fn.__qualname__ = name
    if description:
        fn.__doc__ = description
    fn.__signature__ = sig
    return fn


def noop_update_candidate_ledger(tool_name: str, args_preview: dict[str, Any], result: Any) -> None:
    _ = tool_name, args_preview, result


def noop_search_tool_memories(query_spec: Any) -> list[dict[str, Any]] | None:
    _ = query_spec
    return None
