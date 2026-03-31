from __future__ import annotations

import re

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

from korean_lunar_calendar import KoreanLunarCalendar


_NY = ZoneInfo("America/New_York")
_KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True, slots=True)
class MarketWindow:
    """Represents today's NASDAQ regular session window."""

    trading_date: date
    now_utc: datetime
    now_local: datetime
    open_utc: datetime
    close_utc: datetime
    phase: str  # PRE_OPEN | OPEN | POST_CLOSE | CLOSED


def _observed_fixed_holiday(day: date) -> date:
    """Returns observed date for fixed-date US holidays."""
    if day.weekday() == 5:  # Saturday
        return day - timedelta(days=1)
    if day.weekday() == 6:  # Sunday
        return day + timedelta(days=1)
    return day


def _nth_weekday_of_month(year: int, month: int, weekday: int, nth: int) -> date:
    """Returns the nth weekday(0=Mon) for a given month."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + (nth - 1) * 7)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Returns the last weekday(0=Mon) for a given month."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last = next_month - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _easter_sunday(year: int) -> date:
    """Computes Gregorian Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nasdaq_holidays(year: int) -> set[date]:
    """Returns major NASDAQ/NYSE full-day holidays for a given year."""
    holidays = {
        _observed_fixed_holiday(date(year, 1, 1)),   # New Year's Day
        _nth_weekday_of_month(year, 1, 0, 3),        # Martin Luther King Jr. Day
        _nth_weekday_of_month(year, 2, 0, 3),        # Presidents' Day
        _easter_sunday(year) - timedelta(days=2),    # Good Friday
        _last_weekday_of_month(year, 5, 0),          # Memorial Day
        _observed_fixed_holiday(date(year, 7, 4)),   # Independence Day
        _nth_weekday_of_month(year, 9, 0, 1),        # Labor Day
        _nth_weekday_of_month(year, 11, 3, 4),       # Thanksgiving Day
        _observed_fixed_holiday(date(year, 12, 25)), # Christmas Day
    }

    # NYSE/NASDAQ started closing for Juneteenth in 2022.
    if year >= 2022:
        holidays.add(_observed_fixed_holiday(date(year, 6, 19)))

    return holidays


def is_nasdaq_holiday(trading_date: date) -> bool:
    """Returns True when `trading_date` is a US equity market holiday."""
    y = trading_date.year
    for year in (y - 1, y, y + 1):
        if trading_date in _nasdaq_holidays(year):
            return True
    return False


# ── KRX (KOSPI/KOSDAQ) holidays ──────────────────────────────────────
# Fixed holidays observed by KRX every year.
_KRX_FIXED_HOLIDAYS: list[tuple[int, int]] = [
    (1, 1),   # 신정
    (3, 1),   # 삼일절
    (5, 1),   # 근로자의 날 (KRX 휴장)
    (5, 5),   # 어린이날
    (6, 6),   # 현충일
    (8, 15),  # 광복절
    (10, 3),  # 개천절
    (10, 9),  # 한글날
    (12, 25), # 성탄절
]

_klc = KoreanLunarCalendar()


def _lunar_to_solar(year: int, month: int, day: int) -> date:
    """Convert Korean lunar date to Gregorian solar date."""
    _klc.setLunarDate(year, month, day, False)
    iso = _klc.SolarIsoFormat()
    parts = iso.split("-")
    return date(int(parts[0]), int(parts[1]), int(parts[2]))


def _with_substitute(base_dates: list[date], fixed_holidays: set[date]) -> list[date]:
    """Return weekday closure dates with 대체휴일 (substitute holiday).

    If any base date falls on a weekend or overlaps with a fixed public
    holiday, the next available weekday after the period is added as a
    substitute (max 1 per holiday group, per current KRX practice).
    """
    closures: set[date] = set()
    needs_sub = False
    for d in base_dates:
        if d.weekday() < 5:
            closures.add(d)
        if d.weekday() >= 5 or d in fixed_holidays:
            needs_sub = True
    if needs_sub:
        candidate = max(base_dates) + timedelta(days=1)
        while candidate.weekday() >= 5 or candidate in closures or candidate in fixed_holidays:
            candidate += timedelta(days=1)
        closures.add(candidate)
    return sorted(closures)


@lru_cache(maxsize=32)
def _krx_holidays(year: int) -> frozenset[date]:
    """Returns KRX market closure dates for a given year."""
    fixed = {date(year, m, d) for m, d in _KRX_FIXED_HOLIDAYS}
    holidays: set[date] = set(fixed)

    # 설날 (음력 1/1) — 전날·당일·다음날 3일 연휴
    seol = _lunar_to_solar(year, 1, 1)
    seol_days = [seol - timedelta(days=1), seol, seol + timedelta(days=1)]
    holidays.update(_with_substitute(seol_days, fixed))

    # 석가탄신일 (음력 4/8)
    buddha = _lunar_to_solar(year, 4, 8)
    holidays.update(_with_substitute([buddha], fixed))

    # 추석 (음력 8/15) — 전날·당일·다음날 3일 연휴
    chuseok = _lunar_to_solar(year, 8, 15)
    chuseok_days = [chuseok - timedelta(days=1), chuseok, chuseok + timedelta(days=1)]
    holidays.update(_with_substitute(chuseok_days, fixed))

    return frozenset(holidays)


def is_kospi_holiday(trading_date: date) -> bool:
    """Returns True when ``trading_date`` is a KRX market holiday."""
    if trading_date.weekday() >= 5:
        return True
    return trading_date in _krx_holidays(trading_date.year)


def kospi_window(now_utc: datetime | None = None) -> MarketWindow:
    """Computes KOSPI/KOSDAQ regular-hours window using Asia/Seoul timezone.

    Regular session: 09:00 – 15:30 KST.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    now_local = now_utc.astimezone(_KST)
    trading_date = now_local.date()

    open_local = datetime.combine(trading_date, time(9, 0), tzinfo=_KST)
    close_local = datetime.combine(trading_date, time(15, 30), tzinfo=_KST)

    open_utc = open_local.astimezone(timezone.utc)
    close_utc = close_local.astimezone(timezone.utc)

    if is_kospi_holiday(trading_date):
        phase = "CLOSED"
    elif now_utc < open_utc:
        phase = "PRE_OPEN"
    elif now_utc < close_utc:
        phase = "OPEN"
    else:
        phase = "POST_CLOSE"

    return MarketWindow(
        trading_date=trading_date,
        now_utc=now_utc,
        now_local=now_local,
        open_utc=open_utc,
        close_utc=close_utc,
        phase=phase,
    )


def nasdaq_window(now_utc: datetime | None = None) -> MarketWindow:
    """Computes NASDAQ regular-hours window using America/New_York timezone."""
    now_utc = now_utc or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    now_local = now_utc.astimezone(_NY)
    trading_date = now_local.date()

    open_local = datetime.combine(trading_date, time(9, 30), tzinfo=_NY)
    close_local = datetime.combine(trading_date, time(16, 0), tzinfo=_NY)

    open_utc = open_local.astimezone(timezone.utc)
    close_utc = close_local.astimezone(timezone.utc)

    if now_local.weekday() >= 5 or is_nasdaq_holiday(trading_date):
        phase = "CLOSED"
    elif now_utc < open_utc:
        phase = "PRE_OPEN"
    elif now_utc < close_utc:
        phase = "OPEN"
    else:
        phase = "POST_CLOSE"

    return MarketWindow(
        trading_date=trading_date,
        now_utc=now_utc,
        now_local=now_local,
        open_utc=open_utc,
        close_utc=close_utc,
        phase=phase,
    )


def is_report_window(window: MarketWindow, *, delay_minutes: int = 10, cutoff_hours: int = 8) -> bool:
    """Returns True when we should post EOD report (post-close, same local day)."""
    if window.phase != "POST_CLOSE":
        return False

    if window.now_local.weekday() >= 5:
        return False

    if window.now_utc < window.close_utc + timedelta(minutes=max(0, int(delay_minutes))):
        return False

    cutoff = datetime.combine(window.trading_date, time(23, 59), tzinfo=_NY) + timedelta(hours=max(0, int(cutoff_hours)))
    return window.now_local <= cutoff


def parse_local_times(value: str | None, *, default: list[str]) -> list[time]:
    """Parses a CSV of HH:MM values into local-session times."""
    raw = (value or "").strip()
    tokens = [t.strip() for t in re.split(r"[,|;]", raw) if t.strip()] if raw else list(default)
    out: list[time] = []
    for tok in tokens:
        try:
            hh, mm = tok.split(":", 1)
            h = int(hh)
            m = int(mm)
            if 0 <= h <= 23 and 0 <= m <= 59:
                out.append(time(h, m))
        except Exception:
            continue
    # Dedup while preserving order
    seen = set()
    uniq: list[time] = []
    for t in out:
        key = (t.hour, t.minute)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    return uniq


def should_run_scheduled_cycle(
    window: MarketWindow,
    *,
    times_local: list[time],
    tolerance_minutes: int = 10,
) -> bool:
    """Returns True if now_local is within tolerance of any scheduled time."""
    if window.phase != "OPEN":
        return False
    if window.now_local.weekday() >= 5:
        return False
    tol = max(0, int(tolerance_minutes))

    now_t = window.now_local.timetz().replace(tzinfo=None)
    now_min = now_t.hour * 60 + now_t.minute

    for t in times_local:
        target = t.hour * 60 + t.minute
        if abs(now_min - target) <= tol:
            return True
    return False


def format_local_times(times_local: list[time]) -> str:
    return ",".join(f"{t.hour:02d}:{t.minute:02d}" for t in times_local)
