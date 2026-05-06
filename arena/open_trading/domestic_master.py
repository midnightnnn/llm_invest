from __future__ import annotations

import io
import math
import zipfile
from typing import Any

import requests

KOSPI_MASTER_URL = "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip"

_KOSPI_PART2_WIDTHS = [
    2,
    1,
    4,
    4,
    4,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    9,
    5,
    5,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    2,
    2,
    2,
    3,
    1,
    3,
    12,
    12,
    8,
    15,
    21,
    2,
    7,
    1,
    1,
    1,
    1,
    1,
    9,
    9,
    9,
    5,
    9,
    8,
    9,
    3,
    1,
    1,
    1,
]
_KOSPI_PART2_WIDTH = sum(_KOSPI_PART2_WIDTHS)

_PREVIOUS_VOLUME_IDX = 47
_ETP_IDX = 12
_SPAC_IDX = 19
_HALTED_IDX = 34
_LIQUIDATION_IDX = 35
_ADMIN_IDX = 36
_PREFERRED_IDX = 54
_KOSPI_IDX = 58
_MARKET_CAP_IDX = 65


def _finite_float(value: object) -> float | None:
    try:
        text = str(value or "").strip().replace(",", "")
        if not text:
            return None
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _split_fixed_width(text: str, widths: list[int]) -> list[str]:
    out: list[str] = []
    pos = 0
    for width in widths:
        out.append(text[pos : pos + width].strip())
        pos += width
    return out


def _flag_enabled(value: object) -> bool:
    return str(value or "").strip().upper() in {"Y", "1"}


def _eligible_kospi_master_row(fields: list[str]) -> bool:
    if len(fields) <= _MARKET_CAP_IDX:
        return False
    if _flag_enabled(fields[_ETP_IDX]):
        return False
    if _flag_enabled(fields[_SPAC_IDX]):
        return False
    if _flag_enabled(fields[_HALTED_IDX]):
        return False
    if _flag_enabled(fields[_LIQUIDATION_IDX]):
        return False
    if _flag_enabled(fields[_ADMIN_IDX]):
        return False
    preferred_flag = str(fields[_PREFERRED_IDX] or "").strip()
    if preferred_flag and preferred_flag != "0":
        return False
    kospi_flag = str(fields[_KOSPI_IDX] or "").strip().upper()
    if kospi_flag and kospi_flag not in {"Y", "1"}:
        return False
    return _finite_float(fields[_MARKET_CAP_IDX]) is not None


def parse_kospi_master_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r\n")
        if len(line) <= _KOSPI_PART2_WIDTH:
            continue
        ticker = line[0:9].strip()
        name = line[21 : len(line) - _KOSPI_PART2_WIDTH].strip()
        if not (ticker.isdigit() and len(ticker) == 6):
            continue
        fields = _split_fixed_width(line[-_KOSPI_PART2_WIDTH:], _KOSPI_PART2_WIDTHS)
        if not _eligible_kospi_master_row(fields):
            continue
        market_cap = _finite_float(fields[_MARKET_CAP_IDX])
        volume = _finite_float(fields[_PREVIOUS_VOLUME_IDX])
        rows.append(
            {
                "ticker": ticker,
                "name": name,
                "market_cap": market_cap,
                "volume": volume,
            }
        )
    rows.sort(key=lambda row: (float(row.get("market_cap") or 0.0), float(row.get("volume") or 0.0)), reverse=True)
    return rows


def parse_kospi_master_zip(payload: bytes) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".mst")]
        if not names:
            return []
        data = archive.read(names[0])
    return parse_kospi_master_text(data.decode("cp949", errors="replace"))


def fetch_kospi_master_rows(*, session: requests.Session | None = None, timeout: int = 30) -> list[dict[str, Any]]:
    http = session or requests.Session()
    response = http.get(KOSPI_MASTER_URL, timeout=max(1, int(timeout)))
    response.raise_for_status()
    return parse_kospi_master_zip(response.content)
