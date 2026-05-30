"""Tests for the local DuckDB schema renderer.

Two layers of confidence:
  * Pure rendering rules (run without duckdb installed).
  * Round-trip — every rendered DDL must execute against a real DuckDB
    in-memory connection (skipped when duckdb is not installed).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.data.local.schema import (
    duckdb_table_names,
    render_duckdb_ddls,
    table_specs,
)
from arena.data.schema import TABLE_DDLS, render_table_ddls


# --- Pure rendering rules -----------------------------------------------------


def test_table_specs_extract_column_metadata():
    specs = {spec.name: spec for spec in table_specs()}
    agent_order = specs["agent_order_intents"]
    columns = {col.name: col for col in agent_order.columns}

    assert columns["tenant_id"].type_name == "VARCHAR"
    assert columns["tenant_id"].required is True
    assert columns["quantity"].type_name == "DOUBLE"
    assert columns["allowed"].type_name == "BOOLEAN"
    assert columns["strategy_refs"].type_name == "VARCHAR[]"


def test_rendered_ddls_are_fresh_duckdb_create_statements():
    ddls = render_duckdb_ddls()
    assert all(sql.startswith("CREATE TABLE IF NOT EXISTS ") for sql in ddls)
    assert all("`" not in sql for sql in ddls)
    assert all("{project}" not in sql and "{dataset}" not in sql for sql in ddls)
    assert all("PARTITION BY" not in sql.upper() for sql in ddls)
    assert all("CLUSTER BY" not in sql.upper() for sql in ddls)


def test_table_count_matches_source_of_truth():
    assert len(render_duckdb_ddls()) == len(TABLE_DDLS)
    assert len(duckdb_table_names()) == len(TABLE_DDLS)


def test_table_names_are_unique():
    names = duckdb_table_names()
    assert len(names) == len(set(names))


def test_bq_path_unaffected_by_duckdb_translation():
    """Sanity: translating to DuckDB must not mutate TABLE_DDLS in place."""
    snapshot = list(TABLE_DDLS)
    render_duckdb_ddls()
    assert list(TABLE_DDLS) == snapshot
    # The BQ render path must still produce backticked FQNs.
    rendered = list(render_table_ddls("p", "d"))
    assert all("`p.d." in sql for sql in rendered)


# --- Round-trip against real DuckDB ------------------------------------------


@pytest.fixture(scope="module")
def duckdb_module():
    return pytest.importorskip("duckdb")


def test_every_ddl_executes_in_duckdb(duckdb_module):
    """All 50+ arena tables must materialise in a fresh in-memory DuckDB."""
    con = duckdb_module.connect(":memory:")
    try:
        names = duckdb_table_names()
        for ddl in render_duckdb_ddls():
            con.execute(ddl)
        rows = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main' ORDER BY table_name"
        ).fetchall()
        created = sorted(r[0] for r in rows)
        assert created == sorted(names)
    finally:
        con.close()


def test_ddls_are_idempotent(duckdb_module):
    con = duckdb_module.connect(":memory:")
    try:
        ddls = render_duckdb_ddls()
        for ddl in ddls:
            con.execute(ddl)
        # Re-running must not raise.
        for ddl in ddls:
            con.execute(ddl)
        n = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='main'"
        ).fetchone()[0]
        assert n == len(ddls)
    finally:
        con.close()


def test_duckdb_session_ensure_tables_adds_missing_columns(tmp_path, duckdb_module):
    from arena.data.local.session import DuckDBSession

    session = DuckDBSession(tmp_path / "arena.duckdb")
    try:
        session.execute(
            """
            CREATE TABLE account_snapshots (
              tenant_id VARCHAR NOT NULL,
              snapshot_at TIMESTAMP NOT NULL,
              cash_krw DOUBLE NOT NULL,
              total_equity_krw DOUBLE NOT NULL
            )
            """
        )

        session.ensure_tables()

        cols = {
            row["column_name"]
            for row in session.fetch_rows(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'account_snapshots'
                """
            )
        }
        assert "market_scope" in cols
        assert "usd_krw_rate" in cols
    finally:
        session.close()


def test_local_latest_position_tickers_prefers_matching_market_scope(tmp_path, duckdb_module):
    from arena.data.local.session import DuckDBSession
    from arena.data.local.sleeve_store import LocalSleeveStore

    session = DuckDBSession(tmp_path / "arena.duckdb", tenant_id="midnightnnn")
    try:
        session.ensure_tables()
        kr_snapshot_at = datetime(2026, 2, 21, 9, 0, tzinfo=timezone.utc)
        us_snapshot_at = datetime(2026, 2, 21, 10, 0, tzinfo=timezone.utc)
        session.insert_dict(
            "account_snapshots",
            {
                "tenant_id": "midnightnnn",
                "snapshot_at": kr_snapshot_at,
                "market_scope": "kospi,kosdaq",
                "cash_krw": 1_000_000.0,
                "total_equity_krw": 2_000_000.0,
            },
        )
        session.insert_dict(
            "positions_current",
            {
                "tenant_id": "midnightnnn",
                "snapshot_at": kr_snapshot_at,
                "ticker": "053580",
                "quantity": 10.0,
                "avg_price_krw": 1_000.0,
                "market_price_krw": 1_100.0,
            },
        )
        session.insert_dict(
            "account_snapshots",
            {
                "tenant_id": "midnightnnn",
                "snapshot_at": us_snapshot_at,
                "market_scope": "us",
                "cash_krw": 1_000_000.0,
                "total_equity_krw": 3_000_000.0,
            },
        )
        session.insert_dict(
            "positions_current",
            {
                "tenant_id": "midnightnnn",
                "snapshot_at": us_snapshot_at,
                "ticker": "AAPL",
                "quantity": 2.0,
                "avg_price_krw": 150_000.0,
                "market_price_krw": 160_000.0,
            },
        )

        store = LocalSleeveStore(session)

        assert store.get_latest_position_tickers(tenant_id="midnightnnn", market="kosdaq") == ["053580"]
        assert store.get_latest_position_tickers(tenant_id="midnightnnn", market="nasdaq") == ["AAPL"]
    finally:
        session.close()


def test_local_latest_position_tickers_requires_combined_market_scope(tmp_path, duckdb_module):
    from arena.data.local.session import DuckDBSession
    from arena.data.local.sleeve_store import LocalSleeveStore

    session = DuckDBSession(tmp_path / "arena.duckdb", tenant_id="midnightnnn")
    try:
        session.ensure_tables()
        full_snapshot_at = datetime(2026, 2, 21, 9, 0, tzinfo=timezone.utc)
        us_snapshot_at = datetime(2026, 2, 21, 10, 0, tzinfo=timezone.utc)
        session.insert_dict(
            "account_snapshots",
            {
                "tenant_id": "midnightnnn",
                "snapshot_at": full_snapshot_at,
                "market_scope": "us,kospi,kosdaq",
                "cash_krw": 1_000_000.0,
                "total_equity_krw": 2_000_000.0,
            },
        )
        for ticker in ["053580", "AAPL"]:
            session.insert_dict(
                "positions_current",
                {
                    "tenant_id": "midnightnnn",
                    "snapshot_at": full_snapshot_at,
                    "ticker": ticker,
                    "quantity": 1.0,
                    "avg_price_krw": 1_000.0,
                    "market_price_krw": 1_100.0,
                },
            )
        session.insert_dict(
            "account_snapshots",
            {
                "tenant_id": "midnightnnn",
                "snapshot_at": us_snapshot_at,
                "market_scope": "us",
                "cash_krw": 1_000_000.0,
                "total_equity_krw": 3_000_000.0,
            },
        )
        session.insert_dict(
            "positions_current",
            {
                "tenant_id": "midnightnnn",
                "snapshot_at": us_snapshot_at,
                "ticker": "VZ",
                "quantity": 2.0,
                "avg_price_krw": 150_000.0,
                "market_price_krw": 160_000.0,
            },
        )

        store = LocalSleeveStore(session)

        assert store.get_latest_position_tickers(tenant_id="midnightnnn", market="us,kospi,kosdaq") == [
            "053580",
            "AAPL",
        ]
    finally:
        session.close()


def test_market_features_columns_match(duckdb_module):
    """Spot check: ensure column names + types survive translation faithfully."""
    con = duckdb_module.connect(":memory:")
    try:
        for ddl in render_duckdb_ddls():
            con.execute(ddl)
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'agent_order_intents' ORDER BY ordinal_position"
        ).fetchall()
        col_map = dict(cols)
        # Inputs from the BQ source DDL — translated equivalents:
        assert col_map["tenant_id"] == "VARCHAR"
        assert col_map["quantity"] == "DOUBLE"
        assert col_map["allowed"] == "BOOLEAN"
        assert col_map["created_at"] == "TIMESTAMP"
        # ARRAY<STRING> -> VARCHAR[]
        assert col_map["strategy_refs"].upper().startswith("VARCHAR[")
    finally:
        con.close()


def test_macro_indicator_observations_schema_is_available(duckdb_module):
    """Macro history must be a first-class table in both BigQuery and DuckDB."""
    rendered_bq = "\n".join(render_table_ddls("proj", "ds"))
    assert "`proj.ds.macro_indicator_observations`" in rendered_bq
    assert "PARTITION BY observation_date" in rendered_bq
    assert "CLUSTER BY source, indicator_key" in rendered_bq

    con = duckdb_module.connect(":memory:")
    try:
        for ddl in render_duckdb_ddls():
            con.execute(ddl)
        cols = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'macro_indicator_observations' ORDER BY ordinal_position"
        ).fetchall()
        col_map = dict(cols)
        assert col_map["observed_at"] == "TIMESTAMP"
        assert col_map["source"] == "VARCHAR"
        assert col_map["indicator_key"] == "VARCHAR"
        assert col_map["source_series_id"] == "VARCHAR"
        assert col_map["source_item_code"] == "VARCHAR"
        assert col_map["observation_date"] == "DATE"
        assert col_map["value"] == "DOUBLE"
        assert col_map["raw_json"] == "JSON"
    finally:
        con.close()


def test_duckdb_session_insert_dataframe_bulk_path(tmp_path, duckdb_module):
    pd = pytest.importorskip("pandas")
    import numpy as np

    from arena.data.local.session import DuckDBSession

    session = DuckDBSession(tmp_path / "arena.duckdb")
    try:
        session.execute("CREATE TABLE sample_bulk (id BIGINT, score DOUBLE, label VARCHAR)")
        frame = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "score": [0.5, np.nan, 1.5],
                "label": ["a", None, "c"],
            }
        )

        inserted = session.insert_dataframe("sample_bulk", frame, columns=["id", "score", "label"])

        assert inserted == 3
        rows = session.fetch(
            "SELECT id, score, score IS NULL AS score_null, label, label IS NULL AS label_null "
            "FROM sample_bulk ORDER BY id"
        )
        assert rows == [(1, 0.5, False, "a", False), (2, None, True, None, True), (3, 1.5, False, "c", False)]
    finally:
        session.close()


def test_duckdb_session_translates_timestamp_sub_named_interval(tmp_path, duckdb_module):
    from arena.data.local.session import DuckDBSession

    session = DuckDBSession(tmp_path / "arena.duckdb")
    try:
        rows = session.fetch_rows(
            """
            SELECT
              TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_days DAY) AS cutoff_ts,
              TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL @forward_days DAY) AS forward_ts
            """,
            {"lookback_days": 7, "forward_days": 3},
        )
    finally:
        session.close()

    assert len(rows) == 1
    assert rows[0]["cutoff_ts"] is not None
    assert rows[0]["forward_ts"] is not None
