from __future__ import annotations

import inspect
import typing
from typing import Any

from arena.tools.registry import ToolEntry


def _schema_safe_signature(sig: inspect.Signature, annotations: dict[str, Any] | None = None) -> inspect.Signature:
    parameters: list[inspect.Parameter] = []
    changed = False
    for param in sig.parameters.values():
        if param.name == "tool_context" and param.annotation is not inspect.Parameter.empty:
            parameters.append(param.replace(annotation=inspect.Parameter.empty))
            changed = True
        elif annotations is not None and param.name in annotations and param.annotation != annotations[param.name]:
            parameters.append(param.replace(annotation=annotations[param.name]))
            changed = True
        else:
            parameters.append(param)
    return sig.replace(parameters=parameters) if changed else sig


def _schema_safe_annotations(fn: Any) -> dict[str, Any] | None:
    source = getattr(fn, "__wrapped__", fn)
    try:
        annotations = typing.get_type_hints(source)
    except Exception:
        annotations = getattr(fn, "__annotations__", None)
    if not isinstance(annotations, dict) or "tool_context" not in annotations:
        return annotations if isinstance(annotations, dict) else None
    safe = dict(annotations)
    safe.pop("tool_context", None)
    return safe


def apply_tool_schema_metadata(fn: Any, *, entry: ToolEntry, sig: inspect.Signature) -> Any:
    name = str(entry.name or entry.tool_id or getattr(fn, "__name__", "tool")).strip() or "tool"
    description = str(entry.description or "").strip()
    fn.__name__ = name
    fn.__qualname__ = name
    if description:
        fn.__doc__ = description
    safe_annotations = _schema_safe_annotations(fn)
    if safe_annotations is not None:
        fn.__annotations__ = safe_annotations
    fn.__signature__ = _schema_safe_signature(sig, safe_annotations)
    return fn


def noop_update_candidate_ledger(tool_name: str, args_preview: dict[str, Any], result: Any) -> None:
    _ = tool_name, args_preview, result


def noop_search_tool_memories(query_spec: Any) -> list[dict[str, Any]] | None:
    _ = query_spec
    return None
