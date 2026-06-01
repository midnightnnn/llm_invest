#!/usr/bin/env python3
"""Migrates macro research tables from tenant-scoped to public reference data.

The migration keeps a timestamped backup table, then rewrites each target table
without ``tenant_id`` and deduplicates by ``source_doc_id``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os

from google.cloud import bigquery


DOCUMENT_COLUMNS = (
    "source_doc_id",
    "source",
    "feed_id",
    "doc_type",
    "region",
    "market",
    "title",
    "source_url",
    "published_at",
    "fetched_at",
    "content_hash",
    "raw_gcs_uri",
    "content_gcs_uri",
    "pdf_gcs_uri",
    "text_char_count",
    "status",
    "summary_status",
    "error_message",
    "themes",
    "detail_json",
)

BRIEFING_COLUMNS = (
    "source_doc_id",
    "created_at",
    "published_at",
    "source",
    "feed_id",
    "doc_type",
    "region",
    "market",
    "title",
    "source_url",
    "headline",
    "summary",
    "key_points",
    "market_implication",
    "risk_flags",
    "themes",
    "confidence",
    "model",
    "detail_json",
)


def _fq(project: str, dataset: str, table: str) -> str:
    return f"`{project}.{dataset}.{table}`"


def _query(client: bigquery.Client, sql: str) -> None:
    client.query(sql).result()


def _has_tenant_id(client: bigquery.Client, project: str, dataset: str, table: str) -> bool:
    try:
        bq_table = client.get_table(f"{project}.{dataset}.{table}")
    except Exception:
        raise RuntimeError(f"table not found: {project}.{dataset}.{table}")
    return any(field.name == "tenant_id" for field in bq_table.schema)


def _migrate_table(
    client: bigquery.Client,
    *,
    project: str,
    dataset: str,
    table: str,
    backup_suffix: str,
    columns: tuple[str, ...],
    timestamp_column: str,
    partition_expr: str,
    cluster_by: str,
    apply: bool,
) -> None:
    if not _has_tenant_id(client, project, dataset, table):
        print(f"{table}: already public-scope; tenant_id is absent")
        return

    backup_table = f"{table}_tenant_scoped_backup_{backup_suffix}"
    temp_table = f"{table}_public_tmp_{backup_suffix}"
    backup_sql = f"""
    CREATE TABLE {_fq(project, dataset, backup_table)}
    AS SELECT * FROM {_fq(project, dataset, table)}
    """
    def build_public_sql(*, target_table: str, source_table: str, source_has_tenant_id: bool) -> str:
        if source_has_tenant_id:
            select_sql = f"""
            SELECT {", ".join(columns)}
            FROM (
              SELECT
                *,
                ROW_NUMBER() OVER (
                  PARTITION BY source_doc_id
                  ORDER BY {timestamp_column} DESC, IF(tenant_id = 'local', 1, 0) DESC
                ) AS _rn
              FROM {_fq(project, dataset, source_table)}
            )
            WHERE _rn = 1
            """
        else:
            select_sql = f"""
            SELECT {", ".join(columns)}
            FROM {_fq(project, dataset, source_table)}
            """
        return f"""
        CREATE TABLE {_fq(project, dataset, target_table)}
        PARTITION BY {partition_expr}
        CLUSTER BY {cluster_by}
        AS
        {select_sql}
        """

    temp_sql = build_public_sql(target_table=temp_table, source_table=backup_table, source_has_tenant_id=True)
    recreate_sql = build_public_sql(target_table=table, source_table=temp_table, source_has_tenant_id=False)
    drop_target_sql = f"DROP TABLE {_fq(project, dataset, table)}"
    drop_temp_sql = f"DROP TABLE {_fq(project, dataset, temp_table)}"

    print(f"{table}: backup -> {backup_table}")
    print(f"{table}: rewrite without tenant_id, dedupe by source_doc_id")
    if not apply:
        return
    _query(client, backup_sql)
    _query(client, temp_sql)
    _query(client, drop_target_sql)
    _query(client, recreate_sql)
    _query(client, drop_temp_sql)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    parser.add_argument("--dataset", default=os.getenv("BQ_DATASET", "llm_arena"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not args.project:
        raise SystemExit("missing --project or GOOGLE_CLOUD_PROJECT")

    client = bigquery.Client(project=args.project)
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    _migrate_table(
        client,
        project=args.project,
        dataset=args.dataset,
        table="macro_research_documents",
        backup_suffix=suffix,
        columns=DOCUMENT_COLUMNS,
        timestamp_column="fetched_at",
        partition_expr="DATE(fetched_at)",
        cluster_by="source, feed_id, doc_type",
        apply=args.apply,
    )
    _migrate_table(
        client,
        project=args.project,
        dataset=args.dataset,
        table="macro_research_briefings",
        backup_suffix=suffix,
        columns=BRIEFING_COLUMNS,
        timestamp_column="created_at",
        partition_expr="DATE(created_at)",
        cluster_by="source, doc_type, market",
        apply=args.apply,
    )

    if not args.apply:
        print("dry run only; rerun with --apply to migrate")


if __name__ == "__main__":
    main()
