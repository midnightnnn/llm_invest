from __future__ import annotations

from arena.tools.sector_map import SECTOR_BY_TICKER


def test_sector_map_has_core_entries() -> None:
    for ticker in ("AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"):
        assert ticker in SECTOR_BY_TICKER, f"missing sector mapping for {ticker}"
