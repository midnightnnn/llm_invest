from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone

from arena.market_hours import is_kospi_holiday, is_nasdaq_holiday, kospi_window, nasdaq_window


def test_nasdaq_window_is_closed_on_presidents_day_2026() -> None:
    # 2026-02-16 12:00 ET (US market holiday: Presidents' Day).
    now_utc = datetime(2026, 2, 16, 17, 0, tzinfo=timezone.utc)
    window = nasdaq_window(now_utc)

    assert window.trading_date == date(2026, 2, 16)
    assert is_nasdaq_holiday(window.trading_date) is True
    assert window.phase == "CLOSED"


def test_nasdaq_window_is_open_on_normal_weekday_session_time() -> None:
    # 2026-02-17 12:00 ET (next normal trading day).
    now_utc = datetime(2026, 2, 17, 17, 0, tzinfo=timezone.utc)
    window = nasdaq_window(now_utc)

    assert window.trading_date == date(2026, 2, 17)
    assert is_nasdaq_holiday(window.trading_date) is False
    assert window.phase == "OPEN"


def test_observed_holiday_is_detected() -> None:
    # 2026-07-04 is Saturday, so market holiday is observed on Friday 2026-07-03.
    assert is_nasdaq_holiday(date(2026, 7, 3)) is True


# ── KOSPI tests ──────────────────────────────────────────────────────

def test_kospi_holiday_lunar_new_year_2026() -> None:
    # 2026 설날 연휴: Feb 16 (Mon), 17 (Tue), 18 (Wed)
    assert is_kospi_holiday(date(2026, 2, 16)) is True
    assert is_kospi_holiday(date(2026, 2, 17)) is True
    assert is_kospi_holiday(date(2026, 2, 18)) is True
    # Feb 19 (Thu) is normal trading day
    assert is_kospi_holiday(date(2026, 2, 19)) is False


def test_kospi_holiday_chuseok_2026() -> None:
    # 2026 추석: Sep 24 (Thu), 25 (Fri), 28 (Mon substitute)
    assert is_kospi_holiday(date(2026, 9, 24)) is True
    assert is_kospi_holiday(date(2026, 9, 25)) is True
    assert is_kospi_holiday(date(2026, 9, 28)) is True


def test_kospi_holiday_fixed_holidays() -> None:
    # 삼일절 (Mar 1), 어린이날 (May 5), 현충일 (Jun 6), 광복절 (Aug 15)
    assert is_kospi_holiday(date(2026, 3, 1)) is True   # 삼일절 (Sun → still True via weekend)
    assert is_kospi_holiday(date(2026, 5, 5)) is True   # 어린이날 (Tue)
    assert is_kospi_holiday(date(2026, 6, 6)) is True   # 현충일 (Sat → True via weekend)
    assert is_kospi_holiday(date(2026, 10, 3)) is True  # 개천절 (Sat → True via weekend)
    assert is_kospi_holiday(date(2026, 10, 9)) is True  # 한글날 (Fri)
    assert is_kospi_holiday(date(2026, 12, 25)) is True # 성탄절 (Fri)


def test_kospi_holiday_weekend() -> None:
    assert is_kospi_holiday(date(2026, 3, 7)) is True   # Saturday
    assert is_kospi_holiday(date(2026, 3, 8)) is True   # Sunday
    assert is_kospi_holiday(date(2026, 3, 9)) is False  # Monday


def test_kospi_window_open_during_session() -> None:
    # 2026-03-09 12:00 KST (Mon, normal trading day, within 09:00-15:30 KST)
    now_utc = datetime(2026, 3, 9, 3, 0, tzinfo=timezone.utc)  # 12:00 KST
    window = kospi_window(now_utc)
    assert window.phase == "OPEN"
    assert window.trading_date == date(2026, 3, 9)


def test_kospi_window_closed_on_holiday() -> None:
    # 2026-05-05 어린이날 (Tuesday)
    now_utc = datetime(2026, 5, 5, 3, 0, tzinfo=timezone.utc)
    window = kospi_window(now_utc)
    assert window.phase == "CLOSED"


def test_kospi_workers_day_is_holiday() -> None:
    # 근로자의 날 (May 1) — KRX is closed
    assert is_kospi_holiday(date(2026, 5, 1)) is True  # Friday


def test_kospi_lunar_holidays_computed_for_any_year() -> None:
    """Verify dynamic lunar holiday computation works beyond hardcoded range."""
    # 2035 — no hardcoded data, must be computed from korean_lunar_calendar
    from arena.market_hours import _krx_holidays

    holidays = _krx_holidays(2035)
    # Should contain fixed holidays
    assert date(2035, 1, 1) in holidays   # 신정
    assert date(2035, 12, 25) in holidays # 성탄절
    # Should contain some lunar holidays (exact dates computed dynamically)
    # At minimum, 설날/추석/석가탄신일 should produce weekday closures
    non_fixed = holidays - {date(2035, m, d) for m, d in [
        (1,1),(3,1),(5,1),(5,5),(6,6),(8,15),(10,3),(10,9),(12,25),
    ]}
    assert len(non_fixed) >= 5  # at least 3+1+3 base dates minus weekends


def test_kospi_buddha_birthday_substitute_2025() -> None:
    # 2025 석가탄신일 = 음력 4/8 = 양력 5/5 (어린이날과 겹침) → 대체 5/6
    assert is_kospi_holiday(date(2025, 5, 5)) is True   # 어린이날 (fixed)
    assert is_kospi_holiday(date(2025, 5, 6)) is True   # 석가탄신일 대체


def test_kospi_chuseok_substitute_2025() -> None:
    # 2025 추석: 10/5(Sun), 10/6(Mon), 10/7(Tue) + 대체 10/8(Wed)
    assert is_kospi_holiday(date(2025, 10, 5)) is True   # Sunday
    assert is_kospi_holiday(date(2025, 10, 6)) is True
    assert is_kospi_holiday(date(2025, 10, 7)) is True
    assert is_kospi_holiday(date(2025, 10, 8)) is True   # 대체휴일


# ── Market filter tests ─────────────────────────────────────────────

@dataclass
class _FakeSettings:
    kis_target_market: str


class TestMarketFilterMatches:
    """Tests for _market_filter_matches used in multi-tenant scheduling."""

    def _call(self, tenant_market: str, market_filter: str) -> bool:
        from arena.cli import _market_filter_matches
        settings = _FakeSettings(kis_target_market=tenant_market)
        return _market_filter_matches(settings, market_filter)

    def test_no_filter_always_matches(self) -> None:
        assert self._call("us", "") is True
        assert self._call("kospi", "") is True

    def test_us_filter_matches_us_tenant(self) -> None:
        assert self._call("us", "us") is True
        assert self._call("nasdaq", "us") is True
        assert self._call("nyse", "us") is True

    def test_us_filter_skips_kospi_tenant(self) -> None:
        assert self._call("kospi", "us") is False

    def test_kospi_filter_matches_kospi_tenant(self) -> None:
        assert self._call("kospi", "kospi") is True
        assert self._call("kospi", "kr") is True

    def test_kospi_filter_skips_us_tenant(self) -> None:
        assert self._call("us", "kospi") is False
        assert self._call("nasdaq", "kospi") is False

    def test_empty_tenant_market_never_matches(self) -> None:
        assert self._call("", "us") is False
        assert self._call("", "kospi") is False
