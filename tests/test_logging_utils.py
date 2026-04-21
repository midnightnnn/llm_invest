from __future__ import annotations

import json
import logging

from arena.logging_utils import _JsonFormatter, event_extra, failure_extra


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
