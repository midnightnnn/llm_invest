"""Low-level BigQuery client wrapper shared by all domain stores."""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime
from typing import Any

from google.api_core.exceptions import NotFound
from google.cloud import bigquery

from arena.data.schema import parse_ddl_columns, render_table_ddls

logger = logging.getLogger(__name__)

_VALID_COL_NAME = re.compile(r"^[a-z_][a-z0-9_]{0,127}$")
_VALID_COL_TYPES = frozenset({
    "STRING", "INT64", "FLOAT64", "BOOL", "BOOLEAN",
    "TIMESTAMP", "DATE", "DATETIME", "NUMERIC", "JSON",
})


class BigQuerySession:
    """Thin wrapper around the BigQuery client providing query execution,
    parameter conversion, dataset/table bootstrapping and tenant resolution.

    Every domain store receives a *single* session instance so that project,
    dataset, location and client are shared without inheritance.
    """

    def __init__(
        self,
        project: str,
        dataset: str,
        location: str,
        tenant_id: str | None = None,
    ):
        self.project = project
        self.dataset = dataset
        self.location = location
        self.tenant_id = self._normalize_tenant_id(tenant_id or os.getenv("ARENA_TENANT_ID"))
        self.client = bigquery.Client(project=project, location=location)

    # ------------------------------------------------------------------
    # Dataset / tenant helpers
    # ------------------------------------------------------------------

    @property
    def dataset_fqn(self) -> str:
        """Returns fully qualified dataset name ``project.dataset``."""
        return f"{self.project}.{self.dataset}"

    @staticmethod
    def _normalize_tenant_id(value: str | None) -> str:
        token = str(value or "").strip().lower()
        return token or "local"

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return self._normalize_tenant_id(tenant_id or self.tenant_id)

    def set_tenant_id(self, tenant_id: str | None) -> None:
        self.tenant_id = self._normalize_tenant_id(tenant_id)

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    _FLOAT_COLUMNS = frozenset({
        "avg_price_native", "market_price_native", "price_native",
        "fx_rate", "notional_krw", "close_price_native", "fx_rate_used",
    })

    def _params(self, params: dict[str, Any] | None) -> list[bigquery.QueryParameter]:
        """Converts Python values into BigQuery query parameters."""
        if not params:
            return []
        out: list[bigquery.QueryParameter] = []
        for key, value in params.items():
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], str)
            ):
                out.append(bigquery.ScalarQueryParameter(key, value[0], value[1]))
                continue
            if value is None and key in self._FLOAT_COLUMNS:
                out.append(bigquery.ScalarQueryParameter(key, "FLOAT64", None))
            elif isinstance(value, bool):
                out.append(bigquery.ScalarQueryParameter(key, "BOOL", value))
            elif isinstance(value, int):
                out.append(bigquery.ScalarQueryParameter(key, "INT64", value))
            elif isinstance(value, float):
                out.append(bigquery.ScalarQueryParameter(key, "FLOAT64", value))
            elif isinstance(value, datetime):
                out.append(bigquery.ScalarQueryParameter(key, "TIMESTAMP", value))
            elif isinstance(value, date):
                out.append(bigquery.ScalarQueryParameter(key, "DATE", value))
            elif isinstance(value, (list, tuple)):
                elem_type = "STRING"
                if value:
                    sample = value[0]
                    if isinstance(sample, int):
                        elem_type = "INT64"
                    elif isinstance(sample, float):
                        elem_type = "FLOAT64"
                out.append(bigquery.ArrayQueryParameter(key, elem_type, list(value)))
            else:
                out.append(bigquery.ScalarQueryParameter(key, "STRING", value))
        return out

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        """Executes a DDL or DML statement."""
        cfg = bigquery.QueryJobConfig(query_parameters=self._params(params), use_legacy_sql=False)
        self.client.query(sql, job_config=cfg, location=self.location).result()

    def fetch_rows(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Executes a query and returns plain Python dictionaries."""
        cfg = bigquery.QueryJobConfig(query_parameters=self._params(params), use_legacy_sql=False)
        rows = self.client.query(sql, job_config=cfg, location=self.location).result()
        return [dict(row.items()) for row in rows]

    # ------------------------------------------------------------------
    # Schema bootstrapping
    # ------------------------------------------------------------------

    def ensure_dataset(self) -> None:
        """Creates the dataset when it does not exist."""
        ref = bigquery.DatasetReference(self.project, self.dataset)
        try:
            self.client.get_dataset(ref)
            logger.info("[cyan]BigQuery dataset ready[/cyan] %s", self.dataset_fqn)
            return
        except NotFound:
            ds = bigquery.Dataset(ref)
            ds.location = self.location
            self.client.create_dataset(ds)
            logger.info("[green]BigQuery dataset created[/green] %s", self.dataset_fqn)

    def ensure_tables(self) -> None:
        """Creates all arena runtime tables idempotently.

        Schema migrations are automatic: columns defined in schema.py DDLs
        but missing from live BQ tables are added via ALTER TABLE.  This makes
        schema.py the single source of truth.
        """
        for ddl in render_table_ddls(self.project, self.dataset):
            self.client.query(ddl).result()
        self._ensure_market_features_ingested_at()

        tenant_backfill_targets: list[str] = []
        for table_name, columns in parse_ddl_columns().items():
            added = self._ensure_table_columns(table_name, columns)
            if "tenant_id" in added:
                tenant_backfill_targets.append(table_name)

        self._ensure_runtime_credentials_has_anthropic()
        enable_tenant_backfill = str(os.getenv("ARENA_ENABLE_TENANT_BACKFILL", "")).strip().lower() in {
            "1", "true", "yes", "y", "on",
        }
        if tenant_backfill_targets and enable_tenant_backfill:
            self._backfill_tenant_id_defaults(tenant_backfill_targets)
        logger.info("[green]BigQuery tables ensured[/green] dataset=%s", self.dataset_fqn)

    # ------------------------------------------------------------------
    # Internal schema helpers
    # ------------------------------------------------------------------

    def _ensure_market_features_ingested_at(self) -> None:
        table_id = f"{self.dataset_fqn}.market_features"
        try:
            table = self.client.get_table(table_id)
        except NotFound:
            return
        if any(field.name == "ingested_at" for field in table.schema):
            return
        try:
            self.execute(f"ALTER TABLE `{table_id}` ADD COLUMN ingested_at TIMESTAMP")
            logger.info("[cyan]BigQuery column added[/cyan] table=%s col=ingested_at", table_id)
        except Exception as exc:
            logger.warning("[yellow]BigQuery column add skipped[/yellow] table=%s err=%s", table_id, str(exc))

    def _ensure_table_columns(self, table_name: str, columns: list[tuple[str, str]]) -> set[str]:
        table_id = f"{self.dataset_fqn}.{table_name}"
        try:
            table = self.client.get_table(table_id)
        except NotFound:
            return set()
        existing = {field.name for field in table.schema}
        added: set[str] = set()
        for name, typ in columns:
            if not _VALID_COL_NAME.match(name):
                raise ValueError(f"Invalid column name for DDL: {name!r}")
            if typ.upper() not in _VALID_COL_TYPES:
                raise ValueError(f"Invalid column type for DDL: {typ!r}")
            if name in existing:
                continue
            try:
                self.execute(f"ALTER TABLE `{table_id}` ADD COLUMN `{name}` {typ}")
                logger.info("[cyan]BigQuery column added[/cyan] table=%s col=%s", table_id, name)
                added.add(name)
            except Exception as exc:
                logger.warning(
                    "[yellow]BigQuery column add skipped[/yellow] table=%s col=%s err=%s",
                    table_id, name, str(exc),
                )
        return added

    def _ensure_runtime_credentials_has_anthropic(self) -> None:
        table_id = f"{self.dataset_fqn}.runtime_credentials"
        try:
            table = self.client.get_table(table_id)
        except NotFound:
            return
        if any(field.name == "has_anthropic" for field in table.schema):
            return
        try:
            self.execute(f"ALTER TABLE `{table_id}` ADD COLUMN has_anthropic BOOL")
            logger.info("[cyan]BigQuery column added[/cyan] table=%s col=has_anthropic", table_id)
        except Exception as exc:
            logger.warning("[yellow]BigQuery column add skipped[/yellow] table=%s err=%s", table_id, str(exc))

    def _backfill_tenant_id_defaults(self, table_names: list[str]) -> None:
        for table_name in table_names:
            table_id = f"{self.dataset_fqn}.{table_name}"
            try:
                table = self.client.get_table(table_id)
            except NotFound:
                continue
            if not any(field.name == "tenant_id" for field in table.schema):
                continue
            try:
                self.execute(
                    f"""
                    UPDATE `{table_id}`
                    SET tenant_id = 'local'
                    WHERE tenant_id IS NULL OR TRIM(tenant_id) = ''
                    """
                )
            except Exception as exc:
                logger.warning(
                    "[yellow]BigQuery tenant_id backfill skipped[/yellow] table=%s err=%s",
                    table_id, str(exc),
                )
