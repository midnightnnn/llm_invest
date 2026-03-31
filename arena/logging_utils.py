from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

_RICH_TAG_RE = re.compile(r"\[[^\]]+\]")

_STANDARD_RECORD_ATTRS: set[str] = {
    "args",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


def _json_safe(value: Any) -> Any:
    """Normalizes values so they can be embedded in JSON logs."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


class _JsonFormatter(logging.Formatter):
    """Formats logs as JSON for Cloud Logging-friendly ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        # Best-effort: strip Rich markup tags from our own log messages.
        message = _RICH_TAG_RE.sub("", message).strip()

        payload: dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # Include structured fields provided via logger.*(..., extra={...}).
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", log_format: str = "") -> None:
    """Configures console logging; prefers JSON on Cloud Run."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    format_choice = (log_format or os.getenv("ARENA_LOG_FORMAT", "")).strip().lower()
    if not format_choice:
        if os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN_JOB"):
            format_choice = "json"
        else:
            format_choice = "rich"

    if format_choice == "json":
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_JsonFormatter())
        root.addHandler(handler)
    else:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
                markup=True,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            root.addHandler(handler)
        except Exception:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(handler)

    # Keep third-party model client logs concise in runtime output.
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
