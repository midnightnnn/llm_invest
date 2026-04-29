"""DuckDB schema rendering for the local backend.

The local backend does not execute translated BigQuery SQL.  This module only
uses the shared schema declarations to extract table/column metadata, then
renders fresh DuckDB ``CREATE TABLE`` statements from that metadata.

Operational SQL for stores lives in ``arena.data.local.*`` and is written in
DuckDB dialect from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from arena.data.schema import TABLE_DDLS


_TYPE_MAP: dict[str, str] = {
    "STRING": "VARCHAR",
    "INT64": "BIGINT",
    "FLOAT64": "DOUBLE",
    "BOOL": "BOOLEAN",
    "BOOLEAN": "BOOLEAN",
    "TIMESTAMP": "TIMESTAMP",
    "DATE": "DATE",
    "DATETIME": "TIMESTAMP",  # DuckDB has no DATETIME; TIMESTAMP matches semantics.
    "NUMERIC": "DECIMAL(38, 9)",
    "JSON": "JSON",
}

_TABLE_FQN_RE = re.compile(r"`\{project\}\.\{dataset\}\.([A-Za-z_][A-Za-z0-9_]*)`")
_COLUMN_RE = re.compile(
    r"^\s+([A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(ARRAY<\s*[A-Za-z_][A-Za-z0-9_]*\s*>|STRING|INT64|FLOAT64|BOOLEAN|BOOL|TIMESTAMP|DATETIME|DATE|NUMERIC|JSON)"
    r"(\s+NOT\s+NULL)?\s*,?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    type_name: str
    required: bool = False


@dataclass(frozen=True, slots=True)
class TableSpec:
    name: str
    columns: tuple[ColumnSpec, ...]


def _duckdb_type(type_name: str) -> str:
    token = str(type_name or "").strip().upper()
    if token.startswith("ARRAY<") and token.endswith(">"):
        inner = token[len("ARRAY<") : -1].strip().upper()
        return f"{_TYPE_MAP.get(inner, inner)}[]"
    return _TYPE_MAP.get(token, token)


def table_specs() -> list[TableSpec]:
    """Extracts table/column metadata from the shared schema declarations."""
    specs: list[TableSpec] = []
    for ddl in TABLE_DDLS:
        table_match = _TABLE_FQN_RE.search(ddl)
        if not table_match:
            continue
        columns: list[ColumnSpec] = []
        for col_match in _COLUMN_RE.finditer(ddl):
            columns.append(
                ColumnSpec(
                    name=col_match.group(1),
                    type_name=_duckdb_type(col_match.group(2)),
                    required=bool(col_match.group(3)),
                )
            )
        specs.append(TableSpec(name=table_match.group(1), columns=tuple(columns)))
    return specs


def render_duckdb_ddl(spec: TableSpec) -> str:
    """Renders one DuckDB ``CREATE TABLE`` statement from a local table spec."""
    if not spec.columns:
        raise ValueError(f"Table {spec.name!r} has no columns")
    rendered_columns = []
    for col in spec.columns:
        required = " NOT NULL" if col.required else ""
        rendered_columns.append(f"  {col.name} {col.type_name}{required}")
    body = ",\n".join(rendered_columns)
    return f"CREATE TABLE IF NOT EXISTS {spec.name} (\n{body}\n)"


def render_duckdb_ddls() -> list[str]:
    """Returns DuckDB-rendered CREATE TABLE statements for every arena table."""
    return [render_duckdb_ddl(spec) for spec in table_specs()]


def duckdb_table_names() -> list[str]:
    """Returns table names in the order they appear in TABLE_DDLS."""
    return [spec.name for spec in table_specs()]
