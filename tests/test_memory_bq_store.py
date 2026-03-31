from __future__ import annotations

from datetime import datetime, timezone

from arena.data.bigquery.memory_bq_store import MemoryBQStore


class _FakeClient:
    def __init__(self) -> None:
        self.inserts: list[tuple[str, list[dict[str, object]]]] = []

    def insert_rows_json(self, table_id: str, rows: list[dict[str, object]]):
        self.inserts.append((table_id, list(rows)))
        return []


class _FakeSession:
    def __init__(self) -> None:
        self.dataset_fqn = "proj.ds"
        self.client = _FakeClient()
        self.executed: list[tuple[str, dict[str, object]]] = []

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return str(tenant_id or "tenant-a")

    def execute(self, sql: str, params: dict[str, object]) -> None:
        self.executed.append((sql, dict(params)))


def test_insert_research_briefings_preserves_datetime_for_graph_upsert() -> None:
    session = _FakeSession()
    store = MemoryBQStore(session)
    created_at = datetime(2026, 3, 29, 12, 30, tzinfo=timezone.utc)

    store.insert_research_briefings(
        [
            {
                "briefing_id": "brief-1",
                "created_at": created_at,
                "ticker": "AAPL",
                "category": "company",
                "headline": "Headline",
                "summary": "Summary",
                "sources": ["src-1"],
                "trading_mode": "live",
            }
        ],
        tenant_id="tenant-a",
    )

    assert session.client.inserts
    _, inserted_rows = session.client.inserts[0]
    assert inserted_rows[0]["created_at"] == "2026-03-29 12:30:00.000000"

    assert session.executed
    _, params = session.executed[0]
    assert params["created_at"] == created_at
