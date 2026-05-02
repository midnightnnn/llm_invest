"""DuckDB session — minimal connection wrapper for the local backend.

Lazy ``duckdb`` import: nothing blows up at module load time when the
optional ``local`` extra is missing. Users opt in with
``pip install -e ".[local]"``.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import uuid
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any

from arena.data.local.schema import render_duckdb_ddls

logger = logging.getLogger(__name__)


_DUCKDB_HINT = (
    "duckdb is not installed. Install local extras with "
    'pip install -e ".[local]" before using ARENA_MODE=local.'
)


def _import_duckdb():
    try:
        import duckdb  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(_DUCKDB_HINT) from exc
    return duckdb


def _import_filelock():
    try:
        from filelock import FileLock  # type: ignore[import-untyped]
    except ImportError:
        return None
    return FileLock


def default_db_path() -> Path:
    """Returns the default local DuckDB path; honours ``ARENA_LOCAL_DB_PATH`` env."""
    raw = os.getenv("ARENA_LOCAL_DB_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / "data" / "arena.duckdb").resolve()


class DuckDBSession:
    """Thin DuckDB connection wrapper.

    Owns one connection per session.  PR 2 ships only the bootstrap path; full
    write-path coordination (locking, retries) lands when the local
    repository's write methods do.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        read_only: bool = False,
        tenant_id: str | None = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self.read_only = bool(read_only)
        self.tenant_id = self._normalize_tenant_id(tenant_id or os.getenv("ARENA_TENANT_ID"))
        self._conn: Any | None = None
        self._lock_path = self.db_path.with_suffix(self.db_path.suffix + ".lock")
        self._thread_lock = threading.Lock()

    @staticmethod
    def _normalize_tenant_id(value: str | None) -> str:
        token = str(value or "").strip().lower()
        return token or "local"

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return self._normalize_tenant_id(tenant_id or self.tenant_id)

    def set_tenant_id(self, tenant_id: str | None) -> None:
        self.tenant_id = self._normalize_tenant_id(tenant_id)

    @property
    def dataset_fqn(self) -> str:
        """DuckDB has no project/dataset namespace; returned for facade parity."""
        return ""

    def connect(self):
        if self._conn is not None:
            return self._conn
        duckdb = _import_duckdb()
        if not self.read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path), read_only=self.read_only)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __enter__(self) -> "DuckDBSession":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _is_read_sql(sql: str) -> bool:
        token = str(sql or "").lstrip().split(None, 1)[0].upper() if str(sql or "").strip() else ""
        return token in {"SELECT", "WITH", "SHOW", "DESCRIBE", "EXPLAIN", "PRAGMA"}

    def _write_lock(self):
        if self.read_only:
            return nullcontext()
        FileLock = _import_filelock()
        if FileLock is None:
            logger.warning("[yellow]filelock not installed; DuckDB writes are not cross-process serialized[/yellow]")
            return nullcontext()
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self._lock_path), timeout=60)

    @classmethod
    def _normalize_param(cls, value: Any) -> Any:
        """Normalizes values before passing them to DuckDB.

        DuckDB ``TIMESTAMP`` is timezone-naive. Store aware datetimes as UTC
        naive values so inserts do not get shifted through the host timezone.
        """
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                return value.astimezone(timezone.utc).replace(tzinfo=None)
            return value
        if isinstance(value, dict):
            return {key: cls._normalize_param(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._normalize_param(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._normalize_param(item) for item in value)
        return value

    def execute(self, sql: str, params: list[Any] | dict[str, Any] | None = None) -> Any:
        sql = sql.replace("`.", "`").replace("`", '"')
        sql = re.sub(r'(?i)IN\s+UNNEST\(([^)]+)\)', r'IN (SELECT unnest(\1))', sql)
        sql = re.sub(r'(?i)TIMESTAMP_SUB\(([^,]+),\s*(INTERVAL\s+[^)]+)\)', r'(\1 - \2)', sql)
        sql = re.sub(r'(?i)TIMESTAMP_ADD\(([^,]+),\s*(INTERVAL\s+[^)]+)\)', r'(\1 + \2)', sql)
        sql = re.sub(r"(?i)CURRENT_TIMESTAMP\(\)", "CURRENT_TIMESTAMP", sql)
        sql = re.sub(r'(?i)DATE\(([^,]+),\s*[\'"][^\'"]+[\'"]\)', r'CAST(\1 AS DATE)', sql)
        if isinstance(params, dict):
            for k in params.keys():
                sql = sql.replace(f"@{k}", f"${k}")
            sql = re.sub(
                r"(?i)INTERVAL\s+(\$[A-Za-z_][A-Za-z0-9_]*)\s+(DAY|DAYS|HOUR|HOURS|MINUTE|MINUTES|SECOND|SECONDS)",
                lambda match: f"({match.group(1)} * INTERVAL '1 {match.group(2).lower()}')",
                sql,
            )
        lock = nullcontext() if self._is_read_sql(sql) else self._write_lock()
        with lock:
            cur = self.connect().cursor()
            if params is None:
                return cur.execute(sql)
            return cur.execute(sql, self._normalize_param(params))

    def executemany(self, sql: str, params: list[list[Any]] | list[tuple[Any, ...]]) -> Any:
        sql = sql.replace("`.", "`").replace("`", '"')
        sql = re.sub(r'(?i)IN\s+UNNEST\(([^)]+)\)', r'IN (SELECT unnest(\1))', sql)
        sql = re.sub(r'(?i)TIMESTAMP_SUB\(([^,]+),\s*(INTERVAL\s+[^)]+)\)', r'(\1 - \2)', sql)
        sql = re.sub(r'(?i)TIMESTAMP_ADD\(([^,]+),\s*(INTERVAL\s+[^)]+)\)', r'(\1 + \2)', sql)
        sql = re.sub(r"(?i)CURRENT_TIMESTAMP\(\)", "CURRENT_TIMESTAMP", sql)
        sql = re.sub(r'(?i)DATE\(([^,]+),\s*[\'"][^\'"]+[\'"]\)', r'CAST(\1 AS DATE)', sql)
        lock = nullcontext() if self._is_read_sql(sql) else self._write_lock()
        with lock:
            cur = self.connect().cursor()
            return cur.executemany(sql, self._normalize_param(params))

    def insert_dict(self, table: str, row: dict[str, Any]) -> None:
        """Inserts one row using named values and stable column ordering."""
        cols = list(row.keys())
        placeholders = ", ".join(f"${col}" for col in cols)
        col_sql = ", ".join(cols)
        self.execute(f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})", row)

    def insert_dicts(self, table: str, rows: list[dict[str, Any]]) -> int:
        """Inserts many dict rows; returns number written."""
        if not rows:
            return 0
        cols = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(cols))
        values = [[row.get(col) for col in cols] for row in rows]
        self.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", values)
        return len(rows)

    @staticmethod
    def _quote_identifier(name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def insert_dataframe(self, table: str, frame: Any, *, columns: list[str] | tuple[str, ...] | None = None) -> int:
        """Bulk-inserts a pandas/Arrow-like frame through DuckDB's vectorized scanner."""
        if frame is None or getattr(frame, "empty", False):
            return 0
        cols = list(columns or list(frame.columns))
        if not cols:
            return 0
        missing = [col for col in cols if col not in frame.columns]
        if missing:
            raise ValueError(f"insert_dataframe missing columns for {table}: {missing}")

        view_name = f"_arena_insert_{uuid.uuid4().hex}"
        quoted_cols = ", ".join(self._quote_identifier(col) for col in cols)
        lock = self._write_lock()
        with lock:
            conn = self.connect()
            view_frame = frame.loc[:, cols]
            conn.register(view_name, view_frame)
            try:
                conn.execute(
                    f"INSERT INTO {table} ({quoted_cols}) "
                    f"SELECT {quoted_cols} FROM {self._quote_identifier(view_name)}"
                )
            finally:
                try:
                    conn.unregister(view_name)
                except Exception:
                    logger.debug("DuckDB temporary frame unregister failed", exc_info=True)
        return len(frame)

    def fetch(self, sql: str, params: list[Any] | dict[str, Any] | None = None) -> list[tuple]:
        return self.execute(sql, params).fetchall()

    def fetch_rows(
        self,
        sql: str,
        params: list[Any] | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns query rows as ``[dict[col_name, value], ...]`` like the BQ session."""
        cur = self.execute(sql, params)
        cols = [desc[0] for desc in (cur.description or [])]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def ensure_tables(self) -> int:
        """Idempotently creates every arena table from the rendered DuckDB DDLs.

        Returns the number of DDL statements executed (== arena table count).
        """
        conn = self.connect()
        ddls = render_duckdb_ddls()
        for ddl in ddls:
            conn.execute(ddl)
        return len(ddls)
