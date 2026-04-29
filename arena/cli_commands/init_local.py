"""CLI command: ``llm-arena init-local``.

Bootstraps the DuckDB-backed local repository: creates the database file
(parent directory included) and runs every arena CREATE TABLE DDL.

This command does not depend on GCP credentials, BigQuery, Firestore, or
Secret Manager — it is the entry point for OSS quickstart users.
"""

from __future__ import annotations

import logging
from pathlib import Path

from arena.data.local.schema import duckdb_table_names
from arena.data.local.session import DuckDBSession, default_db_path

logger = logging.getLogger(__name__)


def cmd_init_local(*, db_path: str | None = None) -> None:
    """Creates ``arena.duckdb`` and runs every arena CREATE TABLE DDL."""
    target = Path(db_path).expanduser().resolve() if db_path else default_db_path()
    session = DuckDBSession(target)
    try:
        count = session.ensure_tables()
    finally:
        session.close()

    table_names = duckdb_table_names()
    logger.info(
        "[green]Local DuckDB initialised[/green] path=%s tables=%d",
        target,
        count,
    )
    head = ", ".join(table_names[:6])
    suffix = " ..." if len(table_names) > 6 else ""
    logger.info("Tables: %s%s", head, suffix)
