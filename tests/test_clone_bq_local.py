from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.cli import build_parser
from arena.cli_commands import local_clone
from arena.cli_commands.local_clone import BigQueryLocalClonePipeline, LocalCloneConfig
from arena.data.local.session import DuckDBSession


pytest.importorskip("duckdb")


class _FakeQueryJobConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeBigQueryModule:
    QueryJobConfig = _FakeQueryJobConfig


class _FakeField:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeTable:
    def __init__(self, *, rows: int, bytes_: int, field_names: list[str]) -> None:
        self.num_rows = rows
        self.num_bytes = bytes_
        self.schema = [_FakeField(name) for name in field_names]


class _FakeQueryJob:
    def __init__(self, rows):
        self._rows = list(rows)

    def result(self, *, page_size: int):
        assert page_size > 0
        return iter(self._rows)


class _FakeBigQueryClient:
    def __init__(self, rows_by_table: dict[str, list[dict]]) -> None:
        self.rows_by_table = rows_by_table
        self.queries: list[str] = []

    def get_table(self, table_id: str):
        name = table_id.rsplit(".", 1)[-1]
        if name not in self.rows_by_table:
            raise RuntimeError(f"Not found: {table_id}")
        field_names = sorted({key for row in self.rows_by_table[name] for key in row.keys()})
        return _FakeTable(rows=len(self.rows_by_table[name]), bytes_=1234, field_names=field_names)

    def query(self, query: str, **kwargs):
        self.queries.append(query)
        table_token = query.split(" FROM `", 1)[1].split("`", 1)[0]
        name = table_token.rsplit(".", 1)[-1]
        rows = self.rows_by_table[name]
        if " LIMIT " in query:
            limit = int(query.rsplit(" LIMIT ", 1)[1])
            rows = rows[:limit]
        return _FakeQueryJob(rows)


def test_parser_exposes_clone_bq_local_command() -> None:
    args = build_parser().parse_args(
        [
            "clone-bq-local",
            "--tables",
            "tenant_run_statuses",
            "--dry-run",
            "--limit-per-table",
            "2",
        ]
    )
    assert args.command == "clone-bq-local"
    assert args.tables == "tenant_run_statuses"
    assert args.dry_run is True
    assert args.limit_per_table == 2


def test_clone_bq_local_replaces_rows_and_normalizes_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(local_clone, "_import_bigquery", lambda: _FakeBigQueryModule)
    db_path = tmp_path / "arena.duckdb"
    session = DuckDBSession(db_path)
    session.ensure_tables()
    session.insert_dicts(
        "tenant_run_statuses",
        [
            {
                "tenant_id": "old",
                "run_id": "old_run",
                "recorded_at": datetime(2026, 1, 1),
                "run_type": "cycle",
                "status": "ok",
            }
        ],
    )

    rows = [
        {
            "tenant_id": "local",
            "run_id": "run_1",
            "recorded_at": datetime(2026, 4, 29, 1, 2, 3, tzinfo=timezone.utc),
            "run_type": "prep",
            "status": "ok",
            "detail_json": {"nested": {"value": 2}},
        },
        {
            "tenant_id": "local",
            "run_id": "run_2",
            "recorded_at": datetime(2026, 4, 29, 1, 3, 3, tzinfo=timezone.utc),
            "run_type": "prep",
            "status": "partial",
            "detail_json": ["a", "b"],
        },
    ]
    client = _FakeBigQueryClient({"tenant_run_statuses": rows})
    config = LocalCloneConfig(
        project="proj",
        dataset="ds",
        location="asia-northeast3",
        db_path=db_path,
        tables=("tenant_run_statuses",),
        batch_size=1,
        page_size=2,
    )

    result = BigQueryLocalClonePipeline(config, bigquery_client=client, duckdb_session=session).run()

    assert result.rows_written == 2
    assert result.tables[0].status == "ok"
    assert "NULL AS `reason_code`" in client.queries[0]
    stored = session.fetch_rows(
        "SELECT tenant_id, run_id, CAST(detail_json AS VARCHAR) AS detail_json "
        "FROM tenant_run_statuses ORDER BY run_id"
    )
    assert [row["run_id"] for row in stored] == ["run_1", "run_2"]
    assert stored[0]["tenant_id"] == "local"
    assert "old_run" not in {row["run_id"] for row in stored}
    assert '"nested"' in stored[0]["detail_json"]
    assert '"a"' in stored[1]["detail_json"]


def test_clone_bq_local_dry_run_does_not_write(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(local_clone, "_import_bigquery", lambda: _FakeBigQueryModule)
    db_path = tmp_path / "arena.duckdb"
    session = DuckDBSession(db_path)
    client = _FakeBigQueryClient({"runtime_user_tenants": []})
    config = LocalCloneConfig(
        project="proj",
        dataset="ds",
        location="asia-northeast3",
        db_path=db_path,
        tables=("runtime_user_tenants",),
        dry_run=True,
    )

    result = BigQueryLocalClonePipeline(config, bigquery_client=client, duckdb_session=session).run()

    assert result.tables[0].status == "dry_run"
    assert result.tables[0].source_bytes == 1234
    assert client.queries == []
    assert not db_path.exists()


def test_clone_bq_local_relaxes_required_columns_for_historical_nulls(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(local_clone, "_import_bigquery", lambda: _FakeBigQueryModule)
    db_path = tmp_path / "arena.duckdb"
    session = DuckDBSession(db_path)
    session.ensure_tables()
    client = _FakeBigQueryClient(
        {
            "runtime_user_tenants": [
                {
                    "user_email": "user@example.com",
                    "tenant_id": "local",
                    "role": None,
                    "created_at": datetime(2026, 4, 29, 1, 2, 3, tzinfo=timezone.utc),
                }
            ]
        }
    )
    config = LocalCloneConfig(
        project="proj",
        dataset="ds",
        location="asia-northeast3",
        db_path=db_path,
        tables=("runtime_user_tenants",),
    )

    result = BigQueryLocalClonePipeline(config, bigquery_client=client, duckdb_session=session).run()

    assert result.rows_written == 1
    stored = session.fetch_rows("SELECT user_email, role FROM runtime_user_tenants")
    assert stored == [{"user_email": "user@example.com", "role": None}]
