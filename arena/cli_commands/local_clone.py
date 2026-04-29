"""Clone BigQuery arena tables into the local DuckDB database."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from arena.config import load_settings
from arena.data.local.schema import TableSpec, table_specs
from arena.data.local.session import DuckDBSession, default_db_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LocalCloneConfig:
    project: str
    dataset: str
    location: str
    db_path: Path
    tables: tuple[str, ...] = ()
    exclude_tables: tuple[str, ...] = ()
    replace: bool = True
    dry_run: bool = False
    continue_on_error: bool = False
    skip_missing_tables: bool = True
    batch_size: int = 5_000
    page_size: int = 10_000
    limit_per_table: int | None = None
    use_arrow: bool = True


@dataclass(slots=True)
class TableCloneResult:
    table_name: str
    status: str
    source_rows: int | None = None
    source_bytes: int | None = None
    rows_written: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class LocalCloneResult:
    db_path: Path
    project: str
    dataset: str
    dry_run: bool
    tables: list[TableCloneResult] = field(default_factory=list)

    @property
    def rows_written(self) -> int:
        return sum(item.rows_written for item in self.tables)

    @property
    def source_bytes(self) -> int:
        return sum(int(item.source_bytes or 0) for item in self.tables)

    @property
    def ok(self) -> bool:
        return all(item.status in {"ok", "dry_run", "missing"} for item in self.tables)


def _import_bigquery():
    try:
        from google.cloud import bigquery  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - covered by user environment.
        raise RuntimeError("google-cloud-bigquery is required for clone-bq-local") from exc
    return bigquery


def _csv_tokens(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_items: Iterable[str]
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = value
    tokens = []
    for item in raw_items:
        for part in str(item or "").split(","):
            token = part.strip()
            if token:
                tokens.append(token)
    return tuple(tokens)


def _quote_name(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_value(value: Any, *, type_name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if str(type_name).upper() == "JSON":
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=True, default=_json_default, sort_keys=True)
    return value


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    getter = getattr(row, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            pass
    try:
        return row[key]
    except Exception:
        return None


def _format_bytes(size: int | None) -> str:
    n = int(size or 0)
    if n < 1024:
        return f"{n} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(n)
    for unit in units:
        value /= 1024.0
        if value < 1024.0:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PiB"


class BigQueryLocalClonePipeline:
    """Copies BigQuery tables into DuckDB using the shared arena schema."""

    def __init__(
        self,
        config: LocalCloneConfig,
        *,
        bigquery_client: Any | None = None,
        duckdb_session: DuckDBSession | None = None,
    ) -> None:
        self.config = config
        self._bigquery = None
        self._client = bigquery_client
        self._session = duckdb_session

    @property
    def client(self) -> Any:
        if self._client is None:
            bigquery = _import_bigquery()
            self._bigquery = bigquery
            self._client = bigquery.Client(
                project=self.config.project,
                location=self.config.location or None,
            )
        return self._client

    @property
    def session(self) -> DuckDBSession:
        if self._session is None:
            self._session = DuckDBSession(self.config.db_path)
        return self._session

    def run(self) -> LocalCloneResult:
        specs = self._selected_specs()
        result = LocalCloneResult(
            db_path=self.config.db_path,
            project=self.config.project,
            dataset=self.config.dataset,
            dry_run=self.config.dry_run,
        )
        if not self.config.dry_run:
            self.session.ensure_tables()

        logger.info(
            "[bold cyan]BigQuery -> DuckDB clone start[/bold cyan] tables=%d db=%s replace=%s dry_run=%s",
            len(specs),
            self.config.db_path,
            self.config.replace,
            self.config.dry_run,
        )
        for spec in specs:
            try:
                table_result = self._clone_or_plan_table(spec)
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                table_result = TableCloneResult(
                    table_name=spec.name,
                    status="failed",
                    error=str(exc)[:1000],
                )
                logger.exception("[red]BigQuery clone table failed[/red] table=%s", spec.name)
            result.tables.append(table_result)

        logger.info(
            "[bold green]BigQuery -> DuckDB clone finished[/bold green] tables=%d rows_written=%d source=%s",
            len(result.tables),
            result.rows_written,
            _format_bytes(result.source_bytes),
        )
        return result

    def _selected_specs(self) -> list[TableSpec]:
        specs_by_name = {spec.name: spec for spec in table_specs()}
        requested = tuple(name.strip() for name in self.config.tables if name.strip())
        excluded = {name.strip() for name in self.config.exclude_tables if name.strip()}
        names = requested or tuple(specs_by_name.keys())
        unknown = [name for name in names if name not in specs_by_name]
        if unknown:
            raise ValueError(f"Unknown local table(s): {', '.join(sorted(unknown))}")
        return [specs_by_name[name] for name in names if name not in excluded]

    def _table_id(self, table_name: str) -> str:
        return f"{self.config.project}.{self.config.dataset}.{table_name}"

    def _get_table_metadata(self, table_name: str) -> Any | None:
        try:
            return self.client.get_table(self._table_id(table_name))
        except Exception as exc:
            if self.config.skip_missing_tables and _looks_missing_table_error(exc):
                return None
            raise

    def _clone_or_plan_table(self, spec: TableSpec) -> TableCloneResult:
        started = time.monotonic()
        metadata = self._get_table_metadata(spec.name)
        if metadata is None:
            logger.warning("[yellow]BigQuery table missing; skipped[/yellow] table=%s", spec.name)
            return TableCloneResult(
                table_name=spec.name,
                status="missing",
                elapsed_seconds=time.monotonic() - started,
                error="source table missing",
            )

        source_rows = _metadata_int(metadata, "num_rows")
        source_bytes = _metadata_int(metadata, "num_bytes")
        if self.config.dry_run:
            logger.info(
                "[cyan]clone-bq-local dry-run[/cyan] table=%s rows=%s bytes=%s",
                spec.name,
                source_rows,
                _format_bytes(source_bytes),
            )
            return TableCloneResult(
                table_name=spec.name,
                status="dry_run",
                source_rows=source_rows,
                source_bytes=source_bytes,
                elapsed_seconds=time.monotonic() - started,
            )

        rows_written = self._copy_table(spec, metadata)
        logger.info(
            "[green]cloned table[/green] table=%s rows=%d source_rows=%s source=%s elapsed=%.1fs",
            spec.name,
            rows_written,
            source_rows,
            _format_bytes(source_bytes),
            time.monotonic() - started,
        )
        return TableCloneResult(
            table_name=spec.name,
            status="ok",
            source_rows=source_rows,
            source_bytes=source_bytes,
            rows_written=rows_written,
            elapsed_seconds=time.monotonic() - started,
        )

    def _copy_table(self, spec: TableSpec, metadata: Any) -> int:
        columns = [col.name for col in spec.columns]
        bq_columns = _metadata_columns(metadata)
        selected_columns = []
        for col in columns:
            if bq_columns is None or col in bq_columns:
                selected_columns.append(_quote_name(col))
            else:
                selected_columns.append(f"NULL AS {_quote_name(col)}")
        column_sql = ", ".join(selected_columns)
        query = f"SELECT {column_sql} FROM `{self._table_id(spec.name)}`"
        if self.config.limit_per_table is not None:
            query += f" LIMIT {max(0, int(self.config.limit_per_table))}"

        job_config = self._query_job_config()
        query_job = self.client.query(
            query,
            job_config=job_config,
            location=self.config.location or None,
        )
        rows_iter = query_job.result(page_size=max(1, int(self.config.page_size)))

        conn = self.session.connect()
        rows_written = 0
        conn.execute("BEGIN TRANSACTION")
        try:
            self._relax_required_columns(conn, spec)
            if self.config.replace:
                conn.execute(f"DELETE FROM {spec.name}")
            if self.config.use_arrow and hasattr(rows_iter, "to_arrow_iterable"):
                try:
                    rows_written = self._copy_arrow_batches(conn, rows_iter, spec)
                except Exception as exc:
                    logger.warning(
                        "[yellow]Arrow clone failed; falling back to row iterator[/yellow] table=%s err=%s",
                        spec.name,
                        str(exc)[:300],
                    )
                    if self.config.replace:
                        conn.execute(f"DELETE FROM {spec.name}")
                    rows_iter = query_job.result(page_size=max(1, int(self.config.page_size)))
                    rows_written = self._copy_rows_from_iterator(conn, rows_iter, spec)
            else:
                rows_written = self._copy_rows_from_iterator(conn, rows_iter, spec)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise
        return rows_written

    def _relax_required_columns(self, conn: Any, spec: TableSpec) -> None:
        """Allow cloning historical BigQuery rows that predate local NOT NULL rules."""
        for col in spec.columns:
            if not col.required:
                continue
            try:
                conn.execute(f"ALTER TABLE {spec.name} ALTER COLUMN {col.name} DROP NOT NULL")
            except Exception:
                logger.debug("DuckDB DROP NOT NULL skipped table=%s column=%s", spec.name, col.name)

    def _copy_arrow_batches(self, conn: Any, rows_iter: Any, spec: TableSpec) -> int:
        try:
            import pyarrow as pa  # type: ignore[import-untyped]
            from google.cloud import bigquery_storage  # type: ignore[import-untyped]
        except ImportError:
            logger.warning("[yellow]Arrow clone unavailable; falling back to row iterator[/yellow] table=%s", spec.name)
            return self._copy_rows_from_iterator(conn, rows_iter, spec)

        bqstorage_client = bigquery_storage.BigQueryReadClient()
        rows_written = 0
        for batch in rows_iter.to_arrow_iterable(
            bqstorage_client=bqstorage_client,
            max_queue_size=2,
            max_stream_count=1,
        ):
            if getattr(batch, "num_rows", 0) <= 0:
                continue
            table = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
            view_name = "_arena_clone_" + uuid.uuid4().hex
            conn.register(view_name, table)
            try:
                select_sql = ", ".join(_duckdb_select_expr(col.name, col.type_name) for col in spec.columns)
                conn.execute(
                    f"INSERT INTO {spec.name} ({', '.join(col.name for col in spec.columns)}) "
                    f"SELECT {select_sql} FROM {view_name}"
                )
            finally:
                conn.unregister(view_name)
            rows_written += int(table.num_rows)
        return rows_written

    def _copy_rows_from_iterator(self, conn: Any, rows_iter: Any, spec: TableSpec) -> int:
        insert_sql = (
            f"INSERT INTO {spec.name} ({', '.join(col.name for col in spec.columns)}) "
            f"VALUES ({', '.join(['?'] * len(spec.columns))})"
        )
        rows_written = 0
        batch: list[list[Any]] = []
        batch_size = max(1, int(self.config.batch_size))
        for row in rows_iter:
            batch.append(
                [
                    _normalize_value(_row_get(row, col.name), type_name=col.type_name)
                    for col in spec.columns
                ]
            )
            if len(batch) >= batch_size:
                rows_written += self._insert_batch(conn, insert_sql, batch)
                batch.clear()
        if batch:
            rows_written += self._insert_batch(conn, insert_sql, batch)
        return rows_written

    def _insert_batch(self, conn: Any, insert_sql: str, rows: list[list[Any]]) -> int:
        if not rows:
            return 0
        conn.executemany(insert_sql, DuckDBSession._normalize_param(rows))
        return len(rows)

    def _query_job_config(self) -> Any | None:
        if self._bigquery is None:
            try:
                self._bigquery = _import_bigquery()
            except RuntimeError:
                return None
        return self._bigquery.QueryJobConfig(use_legacy_sql=False)


def _looks_missing_table_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    return "notfound" in name or "not found" in text or "notfound" in text


def _metadata_int(metadata: Any, attr: str) -> int | None:
    value = getattr(metadata, attr, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _metadata_columns(metadata: Any) -> set[str] | None:
    schema = getattr(metadata, "schema", None)
    if not schema:
        return None
    names = set()
    for field in schema:
        name = str(getattr(field, "name", "") or "").strip()
        if name:
            names.add(name)
    return names or None


def _duckdb_select_expr(name: str, type_name: str) -> str:
    quoted = '"' + str(name).replace('"', '""') + '"'
    if str(type_name).upper() == "JSON":
        return f"CAST({quoted} AS JSON) AS {quoted}"
    return quoted


def cmd_clone_bq_local(args: Any) -> LocalCloneResult:
    settings = load_settings()
    project = str(getattr(args, "project", "") or settings.google_cloud_project or "").strip()
    dataset = str(getattr(args, "dataset", "") or settings.bq_dataset or "").strip()
    location = str(getattr(args, "location", "") or settings.bq_location or "").strip()
    if not project:
        raise SystemExit("Missing BigQuery project. Set GOOGLE_CLOUD_PROJECT or pass --project.")
    if not dataset:
        raise SystemExit("Missing BigQuery dataset. Set BQ_DATASET or pass --dataset.")
    db_path_raw = str(getattr(args, "db_path", "") or "").strip()
    db_path = Path(db_path_raw).expanduser().resolve() if db_path_raw else default_db_path()

    limit_raw = int(getattr(args, "limit_per_table", 0) or 0)
    config = LocalCloneConfig(
        project=project,
        dataset=dataset,
        location=location,
        db_path=db_path,
        tables=_csv_tokens(getattr(args, "tables", "")),
        exclude_tables=_csv_tokens(getattr(args, "exclude_tables", "")),
        replace=not bool(getattr(args, "append", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        continue_on_error=bool(getattr(args, "continue_on_error", False)),
        skip_missing_tables=not bool(getattr(args, "fail_on_missing", False)),
        batch_size=max(1, int(getattr(args, "batch_size", 5_000) or 5_000)),
        page_size=max(1, int(getattr(args, "page_size", 10_000) or 10_000)),
        limit_per_table=limit_raw if limit_raw > 0 else None,
        use_arrow=not bool(getattr(args, "no_arrow", False)),
    )
    result = BigQueryLocalClonePipeline(config).run()
    if not result.ok:
        raise SystemExit(1)
    return result
