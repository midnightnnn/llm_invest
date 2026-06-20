from __future__ import annotations

import pytest

from arena.cli_commands.local_demo import cmd_seed_local_demo
from arena.data.local.repository import LocalRepository
from arena.prompts.memory_defaults import default_memory_compaction_prompt


pytest.importorskip("duckdb")


def test_seed_local_demo_populates_market_tables(tmp_path, monkeypatch):
    db_path = tmp_path / "arena.duckdb"
    monkeypatch.setenv("ARENA_MODE", "local")
    monkeypatch.setenv("ARENA_LOCAL_DB_PATH", str(db_path))

    summary = cmd_seed_local_demo(days=10)

    assert summary["market_rows"] == 60
    assert summary["latest_rows"] == 6
    assert summary["instruments"] == 6

    repo = LocalRepository(tenant_id="local", db_path=str(db_path))
    try:
        prices = repo.latest_close_prices(tickers=["AAPL", "MSFT"], sources=["local_demo"])
        names = repo.ticker_name_map(tickers=["AAPL", "MSFT"])
    finally:
        repo.session.close()

    assert set(prices) == {"AAPL", "MSFT"}
    assert names["AAPL"] == "Apple Inc."

    repo = LocalRepository(tenant_id="local", db_path=str(db_path))
    try:
        assert repo.get_config("global", "memory_compactor_prompt") == default_memory_compaction_prompt("global")
        assert repo.get_config("local", "memory_compactor_prompt") == default_memory_compaction_prompt("local")
    finally:
        repo.session.close()
