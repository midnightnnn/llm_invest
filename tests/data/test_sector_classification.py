from __future__ import annotations

from arena.open_trading.domestic_master import parse_kosdaq_master_text, parse_kospi_master_text
from arena.open_trading.sector_classification import sector_from_sec_owner_org


_KOSPI_PART2_WIDTHS = [
    2, 1, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 9, 5, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3,
    1, 3, 12, 12, 8, 15, 21, 2, 7, 1, 1, 1, 1, 1, 9, 9, 9, 5, 9, 8, 9,
    3, 1, 1, 1,
]


def _kospi_master_line(ticker: str, name: str, industry_code: str) -> str:
    fields = ["" for _ in _KOSPI_PART2_WIDTHS]
    fields[3] = industry_code
    fields[12] = "N"
    fields[19] = "N"
    fields[34] = "N"
    fields[35] = "N"
    fields[36] = "N"
    fields[47] = "000000021500"
    fields[54] = "0"
    fields[58] = "Y"
    fields[65] = "019701958"
    suffix = "".join(str(value)[:width].ljust(width) for value, width in zip(fields, _KOSPI_PART2_WIDTHS))
    return f"{ticker:<9}{'':12}{name}{suffix}"


def _kosdaq_master_line(ticker: str, name: str, industry_code: str) -> str:
    fields = ["" for _ in _KOSPI_PART2_WIDTHS]
    fields[0] = "ST"
    fields[1] = "1"
    fields[2] = "1009"
    fields[3] = industry_code
    suffix = "".join(str(value)[:width].ljust(width) for value, width in zip(fields, _KOSPI_PART2_WIDTHS))
    return f"{ticker:<9}{'':12}{name:<35}{suffix}"


def test_kis_master_parser_returns_dynamic_sector_metadata() -> None:
    rows = parse_kospi_master_text(_kospi_master_line("005930", "삼성전자", "0013"))

    assert rows[0]["ticker"] == "005930"
    assert rows[0]["sector"] == "Technology"
    assert rows[0]["industry_code"] == "0013"
    assert rows[0]["industry_name"] == "KIS industry 0013"
    assert rows[0]["classification_source"] == "kis_master"


def test_kis_master_parser_leaves_unmapped_industry_unknown() -> None:
    rows = parse_kospi_master_text(_kospi_master_line("035420", "NAVER", "0000"))

    assert rows[0]["ticker"] == "035420"
    assert rows[0]["sector"] is None
    assert rows[0]["industry_code"] == "0000"
    assert rows[0]["classification_source"] == "kis_master"


def test_kosdaq_master_parser_persists_raw_industry_metadata_without_sector_lookup() -> None:
    rows = parse_kosdaq_master_text(_kosdaq_master_line("000250", "삼천당제약", "1024"))

    assert rows[0]["ticker"] == "000250"
    assert rows[0]["sector"] is None
    assert rows[0]["industry_code"] == "1024"
    assert rows[0]["industry_name"] == "KIS industry 1024"
    assert rows[0]["classification_source"] == "kis_master"


def test_sec_owner_org_maps_to_sector() -> None:
    assert sector_from_sec_owner_org("06 Technology") == "Technology"
    assert sector_from_sec_owner_org("04 Finance") == "Financials"
    assert sector_from_sec_owner_org("") is None
