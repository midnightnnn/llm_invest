"""Tests for ``llm-arena init-local``.

Verifies the CLI entry point creates a DuckDB file with every arena table
present, idempotently.
"""

from __future__ import annotations

import pytest

from arena.cli_commands.init_local import cmd_init_local
from arena.data.local.schema import duckdb_table_names


@pytest.fixture(scope="module")
def duckdb_module():
    return pytest.importorskip("duckdb")


def test_init_local_creates_db_and_all_tables(duckdb_module, tmp_path):
    target = tmp_path / "subdir" / "arena.duckdb"
    cmd_init_local(db_path=str(target))

    assert target.exists(), "init-local must create the DuckDB file"

    con = duckdb_module.connect(str(target), read_only=True)
    try:
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' ORDER BY table_name"
        ).fetchall()
    finally:
        con.close()

    actual = sorted(r[0] for r in rows)
    expected = sorted(duckdb_table_names())
    assert actual == expected


def test_init_local_is_idempotent(duckdb_module, tmp_path):
    target = tmp_path / "arena.duckdb"
    cmd_init_local(db_path=str(target))
    cmd_init_local(db_path=str(target))  # Must not raise.

    con = duckdb_module.connect(str(target), read_only=True)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main'"
        ).fetchone()[0]
    finally:
        con.close()
    assert n == len(duckdb_table_names())


def test_init_local_uses_default_db_path_env(duckdb_module, tmp_path, monkeypatch):
    """ARENA_LOCAL_DB_PATH overrides the default output location."""
    target = tmp_path / "from_env.duckdb"
    monkeypatch.setenv("ARENA_LOCAL_DB_PATH", str(target))

    cmd_init_local()  # No --db-path; falls back to env.

    assert target.exists()
