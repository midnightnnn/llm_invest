from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional


_ALLOWED_IMPORTS = (
    "collections",
    "datetime",
    "decimal",
    "functools",
    "itertools",
    "json",
    "math",
    "numpy",
    "operator",
    "pandas",
    "re",
    "statistics",
)

_BLOCKED_IMPORTS = (
    "builtins",
    "duckdb",
    "google",
    "http",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "sqlite3",
    "subprocess",
    "sys",
    "urllib",
)

_BLOCKED_CAPABILITIES = (
    "database",
    "filesystem",
    "network",
    "order_execution",
    "subprocess",
)


def _safe_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _clip_text(value: Any, *, max_len: int) -> str:
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


_SANDBOX_RUNNER = r"""
import builtins
import ast
import contextlib
import io
import json
import math
import sys
import traceback
from datetime import date, datetime

import numpy as np
import pandas as pd

ALLOWED_IMPORTS = set(__payload__["allowed_imports"])
BLOCKED_IMPORTS = set(__payload__["blocked_imports"])
BLOCKED_CAPABILITIES = list(__payload__["blocked_capabilities"])


def _jsonable(value):
    if isinstance(value, pd.DataFrame):
        clean = value.replace([np.inf, -np.inf], np.nan)
        return _jsonable(clean.where(pd.notna(clean), None).to_dict("records"))
    if isinstance(value, pd.Series):
        clean = value.replace([np.inf, -np.inf], np.nan)
        if value.index.dtype == "object":
            return _jsonable(clean.where(pd.notna(clean), None).to_dict())
        return _jsonable(clean.where(pd.notna(clean), None).tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name or "").split(".", 1)[0]
    if root in BLOCKED_IMPORTS:
        raise ImportError(f"blocked import: {root}")
    if root and root not in ALLOWED_IMPORTS and root not in sys.builtin_module_names and name not in sys.modules:
        raise ImportError(f"blocked import: {root}")
    return __orig_import__(name, globals, locals, fromlist, level)


def _save_table(name, rows):
    key = str(name or "").strip()
    if not key:
        raise ValueError("table name is required")
    value = _jsonable(rows)
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise TypeError("save_table expects a DataFrame, Series, dict, or list of rows")
    tables[key] = value[:500]
    saved_artifacts.append({"type": "table", "name": key, "rows": len(tables[key])})
    return {"type": "table", "name": key, "rows": len(tables[key])}


def _load_table(name):
    key = str(name or "").strip()
    return pd.DataFrame(tables.get(key, []))


def _list_tables():
    return sorted(tables.keys())


def _save_note(name, text):
    key = str(name or "").strip()
    if not key:
        raise ValueError("note name is required")
    notes[key] = str(text or "")[:8000]
    saved_artifacts.append({"type": "note", "name": key, "chars": len(notes[key])})
    return {"type": "note", "name": key, "chars": len(notes[key])}


def _load_note(name):
    key = str(name or "").strip()
    return notes.get(key, "")


def _list_notes():
    return sorted(notes.keys())


def _latest_tool_result(tool_name):
    target = str(tool_name or "").strip()
    for event in reversed(tool_results):
        if str(event.get("tool") or "").strip() == target:
            return event.get("result")
    return None


payload = __payload__
context = payload.get("context") or {}
inputs = payload.get("inputs") or {}
tool_results = payload.get("tool_results") or []
tables = dict(payload.get("tables") or {})
notes = dict(payload.get("notes") or {})
saved_artifacts = []
stdout_buffer = io.StringIO()

__orig_import__ = builtins.__import__
safe_builtins = {
    "ArithmeticError": ArithmeticError,
    "AssertionError": AssertionError,
    "Exception": Exception,
    "ImportError": ImportError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "__import__": _guarded_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "getattr": getattr,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
globals_dict = {
    "__builtins__": safe_builtins,
    "context": context,
    "inputs": inputs,
    "tool_results": tool_results,
    "tables": tables,
    "notes": notes,
    "pd": pd,
    "np": np,
    "json": json,
    "math": math,
    "save_table": _save_table,
    "load_table": _load_table,
    "list_tables": _list_tables,
    "save_note": _save_note,
    "load_note": _load_note,
    "list_notes": _list_notes,
    "latest_tool_result": _latest_tool_result,
}

try:
    with contextlib.redirect_stdout(stdout_buffer):
        raw_code = payload.get("code") or ""
        parsed = ast.parse(raw_code, mode="exec")
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            final_expr = ast.Expression(parsed.body.pop().value)
            ast.fix_missing_locations(parsed)
            ast.fix_missing_locations(final_expr)
            if parsed.body:
                exec(compile(parsed, "<scratch>", "exec"), globals_dict, globals_dict)
            globals_dict["result"] = eval(compile(final_expr, "<scratch>", "eval"), globals_dict, globals_dict)
        else:
            exec(compile(parsed, "<scratch>", "exec"), globals_dict, globals_dict)
    response = {
        "status": "ok",
        "stdout": stdout_buffer.getvalue()[:8000],
        "result": _jsonable(globals_dict.get("result")),
        "saved_artifacts": saved_artifacts,
        "sandbox": {
            "allowed_imports": sorted(ALLOWED_IMPORTS),
            "blocked": BLOCKED_CAPABILITIES,
        },
        "_state": {
            "tables": tables,
            "notes": notes,
        },
    }
except Exception as exc:
    response = {
        "status": "error",
        "stdout": stdout_buffer.getvalue()[:8000],
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(limit=3),
        "saved_artifacts": saved_artifacts,
        "sandbox": {
            "allowed_imports": sorted(ALLOWED_IMPORTS),
            "blocked": BLOCKED_CAPABILITIES,
        },
        "_state": {
            "tables": tables,
            "notes": notes,
        },
    }

print(json.dumps(response, ensure_ascii=False, default=str))
"""


@dataclass
class ScratchWorkspace:
    """Cycle-local Python scratch workspace exposed as one ReAct tool."""

    agent_id: str
    tenant_id: str = "local"
    tool_events: list[dict[str, Any]] | None = None
    timeout_seconds: float = 3.0
    _cycle_id: str = ""
    _phase: str = ""
    _context: dict[str, Any] = field(default_factory=dict)
    _tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _notes: dict[str, str] = field(default_factory=dict)

    def set_context(self, context: dict[str, Any]) -> None:
        cycle_id = str((context or {}).get("cycle_id") or "").strip()
        if cycle_id and cycle_id != self._cycle_id:
            self._tables.clear()
            self._notes.clear()
        self._cycle_id = cycle_id
        self._phase = str((context or {}).get("cycle_phase") or "").strip().lower()
        self._context = _safe_json(context or {})

    def run_python(self, code: str, inputs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        payload = {
            "code": str(code or ""),
            "inputs": _safe_json(inputs or {}),
            "context": self._context,
            "tool_results": self._recent_tool_results(),
            "tables": self._tables,
            "notes": self._notes,
            "allowed_imports": list(_ALLOWED_IMPORTS),
            "blocked_imports": list(_BLOCKED_IMPORTS),
            "blocked_capabilities": list(_BLOCKED_CAPABILITIES),
        }
        runner = "__payload__ = " + repr(payload) + "\n" + _SANDBOX_RUNNER
        try:
            completed = subprocess.run(
                [sys.executable, "-c", runner],
                cwd=tempfile.gettempdir(),
                env={
                    **os.environ,
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                capture_output=True,
                text=True,
                timeout=max(0.5, float(self.timeout_seconds)),
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "stdout": "",
                "error": f"scratch execution timed out after {self.timeout_seconds:.1f}s",
                "saved_artifacts": [],
                "sandbox": self._sandbox_metadata(),
            }

        if completed.returncode != 0:
            return {
                "status": "error",
                "stdout": _clip_text(completed.stdout, max_len=8000),
                "error": _clip_text(completed.stderr or f"scratch process exited {completed.returncode}", max_len=2000),
                "saved_artifacts": [],
                "sandbox": self._sandbox_metadata(),
            }

        try:
            response = json.loads(completed.stdout.strip().splitlines()[-1])
        except Exception:
            return {
                "status": "error",
                "stdout": _clip_text(completed.stdout, max_len=8000),
                "error": "scratch process returned invalid JSON",
                "saved_artifacts": [],
                "sandbox": self._sandbox_metadata(),
            }

        state = response.pop("_state", {})
        if isinstance(state, dict):
            tables = state.get("tables")
            notes = state.get("notes")
            if isinstance(tables, dict):
                self._tables = _safe_json(tables)
            if isinstance(notes, dict):
                self._notes = {str(k): str(v) for k, v in notes.items()}
        return response

    def _recent_tool_results(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        events = self.tool_events or []
        for event in events[-24:]:
            if not isinstance(event, dict):
                continue
            tool_name = str(event.get("tool") or "").strip()
            if not tool_name or tool_name == "scratch_run_python":
                continue
            rows.append(
                {
                    "tool": tool_name,
                    "phase": event.get("phase"),
                    "args": _safe_json(event.get("args") or {}),
                    "result": _safe_json(event.get("result") if event.get("result") is not None else event.get("result_preview")),
                }
            )
        return rows[-12:]

    @staticmethod
    def _sandbox_metadata() -> dict[str, Any]:
        return {
            "allowed_imports": list(_ALLOWED_IMPORTS),
            "blocked": list(_BLOCKED_CAPABILITIES),
        }
