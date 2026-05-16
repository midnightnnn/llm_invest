from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


def build_runtime_clock(*, now: datetime | None = None) -> dict[str, str]:
    instant = now or datetime.now(tz=KST)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=KST)
    return {
        "now_kst": instant.astimezone(KST).replace(microsecond=0).isoformat(),
    }


def with_runtime_clock(context: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    updated = dict(context)
    updated["_runtime_clock"] = build_runtime_clock(now=now)
    return updated


def attach_runtime_clock(value: Any, *, clock: dict[str, str] | None = None) -> dict[str, Any]:
    runtime_clock = clock or build_runtime_clock()
    if isinstance(value, dict):
        updated = dict(value)
        updated["_runtime_clock"] = runtime_clock
        return updated
    return {
        "result": value,
        "_runtime_clock": runtime_clock,
    }
