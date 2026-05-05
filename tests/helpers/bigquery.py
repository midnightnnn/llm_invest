from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BigQueryCall:
    """Recorded SQL call made through a fake BigQuery session."""

    sql: str
    params: dict[str, Any] | None


class FakeInsertClient:
    """Minimal BigQuery client fake that records streaming inserts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    @property
    def inserts(self) -> list[tuple[str, list[dict[str, Any]]]]:
        return self.calls

    @property
    def payloads(self) -> list[dict[str, Any]]:
        return [row for _, rows in self.calls for row in rows]

    def insert_rows_json(
        self,
        table_id: str,
        rows: list[dict[str, Any]],
        row_ids: object | None = None,
    ) -> list[dict[str, Any]]:
        _ = row_ids
        self.calls.append((table_id, list(rows)))
        return []


class FakeLoadJob:
    """Minimal load job fake returned by fake table-load clients."""

    def result(self) -> None:
        return None


class FakeBigQuerySession:
    """Small BigQuerySession test double with call recording."""

    def __init__(
        self,
        *,
        project: str = "proj",
        dataset: str = "ds",
        tenant_id: str = "tenant-a",
        client: object | None = None,
        fetch_result: list[dict[str, Any]] | None = None,
        fetch_results: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self.project = project
        self.dataset = dataset
        self.dataset_fqn = f"{project}.{dataset}"
        self.tenant_id = tenant_id
        self.client = client or FakeInsertClient()
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.fetched: list[tuple[str, dict[str, Any]]] = []
        self.execute_calls: list[BigQueryCall] = []
        self.fetch_calls: list[BigQueryCall] = []
        self.fetch_result = list(fetch_result or [])
        self.fetch_results = [list(rows) for rows in (fetch_results or [])]

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return str(tenant_id or self.tenant_id)

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        copied = dict(params or {})
        self.executed.append((sql, copied))
        self.execute_calls.append(BigQueryCall(sql=sql, params=copied))

    def fetch_rows(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        copied = dict(params or {})
        self.fetched.append((sql, copied))
        self.fetch_calls.append(BigQueryCall(sql=sql, params=copied))
        if self.fetch_results:
            return list(self.fetch_results.pop(0))
        return list(self.fetch_result)
