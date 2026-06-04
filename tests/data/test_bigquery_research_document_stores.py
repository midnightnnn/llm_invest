from __future__ import annotations

from typing import Any

from arena.data.bigquery.macro_research_store import MacroResearchStore
from arena.data.bigquery.research_document_store import ResearchDocumentStore


class _FakeSession:
    dataset_fqn = "project.dataset"

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict[str, Any]]] = []

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return str(tenant_id or "tenant-a")

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self.executed.append((sql, dict(params or {})))


def test_research_document_snapshot_update_casts_nullable_text_count() -> None:
    session = _FakeSession()
    store = ResearchDocumentStore(session)  # type: ignore[arg-type]

    store.update_research_document_snapshot(
        "research:doc:1",
        text_char_count=None,
        status="fetch_failed",
        tenant_id="tenant-a",
    )

    sql, params = session.executed[0]
    assert "COALESCE(CAST(@text_char_count AS INT64), text_char_count)" in sql
    assert params["text_char_count"] is None


def test_macro_research_document_snapshot_update_casts_nullable_text_count() -> None:
    session = _FakeSession()
    store = MacroResearchStore(session)  # type: ignore[arg-type]

    store.update_macro_research_document_snapshot(
        "bok:doc:1",
        text_char_count=None,
        status="fetch_failed",
    )

    sql, params = session.executed[0]
    assert "COALESCE(CAST(@text_char_count AS INT64), text_char_count)" in sql
    assert params["text_char_count"] is None
