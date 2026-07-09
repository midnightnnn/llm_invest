"""CLI command: ``llm-arena init-local``.

Bootstraps the DuckDB-backed local repository: creates the database file
(parent directory included) and runs every arena CREATE TABLE DDL.

This command does not depend on GCP credentials, BigQuery, Firestore, or
Secret Manager — it is the entry point for OSS quickstart users.
"""

from __future__ import annotations

import logging
from pathlib import Path

from arena.config import load_settings
from arena.data.local.repository import LocalRepository
from arena.data.local.schema import duckdb_table_names
from arena.data.local.session import default_db_path
from arena.cli_commands.local_bootstrap import seed_local_memory_compaction_prompts

logger = logging.getLogger(__name__)


def cmd_init_local(*, db_path: str | None = None) -> None:
    """Creates ``arena.duckdb`` and runs every arena CREATE TABLE DDL."""
    target = Path(db_path).expanduser().resolve() if db_path else default_db_path()
    settings = load_settings()
    settings.arena_mode = "local"
    repo = LocalRepository(tenant_id="local", settings=settings, db_path=str(target))
    try:
        count = repo.session.ensure_tables()
        seeded = seed_local_memory_compaction_prompts(repo, tenant_id="local", updated_by="init-local")
    finally:
        repo.session.close()

    table_names = duckdb_table_names()
    logger.info(
        "[green]Local DuckDB initialised[/green] path=%s tables=%d",
        target,
        count,
    )
    head = ", ".join(table_names[:6])
    suffix = " ..." if len(table_names) > 6 else ""
    logger.info("Tables: %s%s", head, suffix)
    logger.info(
        "Memory compaction prompts seeded global=%s tenant=%s",
        "yes" if seeded["global"] else "no",
        "yes" if seeded["tenant"] else "no",
    )
