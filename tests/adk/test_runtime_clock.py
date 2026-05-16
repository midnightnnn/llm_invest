from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from arena.agents.runtime_clock import attach_runtime_clock, build_runtime_clock, with_runtime_clock


def test_build_runtime_clock_uses_kst_iso_timestamp() -> None:
    now = datetime(2026, 5, 15, 6, 25, 14, tzinfo=timezone.utc)

    payload = build_runtime_clock(now=now)

    assert payload == {
        "now_kst": "2026-05-15T15:25:14+09:00",
    }


def test_with_runtime_clock_returns_copy_without_mutating_original() -> None:
    context = {"cycle_phase": "execution"}
    now = datetime(2026, 5, 15, 15, 26, 0, tzinfo=ZoneInfo("Asia/Seoul"))

    updated = with_runtime_clock(context, now=now)

    assert context == {"cycle_phase": "execution"}
    assert updated["cycle_phase"] == "execution"
    assert updated["_runtime_clock"] == {
        "now_kst": "2026-05-15T15:26:00+09:00",
    }


def test_attach_runtime_clock_adds_reserved_key_to_dict_result() -> None:
    result = {"ticker": "005930", "price": 70000}
    clock = {"now_kst": "2026-05-15T15:26:18+09:00"}

    updated = attach_runtime_clock(result, clock=clock)

    assert updated == {
        "ticker": "005930",
        "price": 70000,
        "_runtime_clock": clock,
    }
    assert "_runtime_clock" not in result


def test_attach_runtime_clock_wraps_non_dict_result() -> None:
    clock = {"now_kst": "2026-05-15T15:26:18+09:00"}

    updated = attach_runtime_clock(["005930"], clock=clock)

    assert updated == {
        "result": ["005930"],
        "_runtime_clock": clock,
    }
