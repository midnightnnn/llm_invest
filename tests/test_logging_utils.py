from __future__ import annotations

import json
import logging

from arena.logging_utils import _JsonFormatter, configure_logging, event_extra, failure_extra


def test_event_extra_omits_empty_fields() -> None:
    payload = event_extra(
        "adk_decision_failed",
        tenant_id=" cxznms ",
        cycle_id="",
        phase=None,
        tool_calls=0,
        tickers=["AAPL", "", None],
    )

    assert payload == {
        "event": "adk_decision_failed",
        "tenant_id": "cxznms",
        "tool_calls": 0,
        "tickers": ["AAPL"],
    }


def test_failure_extra_adds_exception_metadata() -> None:
    exc = RuntimeError("boom")

    payload = failure_extra("runtime_failed", exc, tenant_id="tenant-a")

    assert payload == {
        "event": "runtime_failed",
        "tenant_id": "tenant-a",
        "err_type": "RuntimeError",
        "err": "boom",
    }


def test_json_formatter_strips_rich_markup_and_keeps_extra_fields() -> None:
    formatter = _JsonFormatter()
    record = logging.LogRecord(
        name="arena.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=12,
        msg="[red]Failure[/red] tenant=%s",
        args=("cxznms",),
        exc_info=None,
    )
    record.event = "tenant_failed"
    record.tenant_id = "cxznms"

    payload = json.loads(formatter.format(record))

    assert payload["severity"] == "ERROR"
    assert payload["message"] == "Failure tenant=cxznms"
    assert payload["event"] == "tenant_failed"
    assert payload["tenant_id"] == "cxznms"


def test_configure_logging_writes_explicit_file(monkeypatch, tmp_path) -> None:
    log_file = tmp_path / "arena.log"
    monkeypatch.setenv("ARENA_LOG_FILE", str(log_file))
    monkeypatch.delenv("ARENA_LOG_FILE_FORMAT", raising=False)

    configure_logging("INFO", "json")
    logging.getLogger("arena.test").info("[cyan]Local[/cyan] log %s", "ok")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "Local log ok" in log_file.read_text(encoding="utf-8")


def test_configure_logging_defaults_local_mode_file(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ARENA_LOG_FILE", raising=False)
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("CLOUD_RUN_JOB", raising=False)
    monkeypatch.setenv("ARENA_MODE", "local")

    configure_logging("INFO", "json")
    logging.getLogger("arena.test").warning("default local file")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "default local file" in (tmp_path / "logs" / "arena-local.log").read_text(encoding="utf-8")
