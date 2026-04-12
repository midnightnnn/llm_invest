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
        self.fetched: list[tuple[str, dict[str, object]]] = []

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return str(tenant_id or "tenant-a")

    def execute(self, sql: str, params: dict[str, object]) -> None:
        self.executed.append((sql, dict(params)))

    def fetch_rows(self, sql: str, params: dict[str, object]) -> list[dict[str, object]]:
        self.fetched.append((sql, dict(params)))
        return []


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
    assert any("memory_relation_triples" in sql for sql, _ in session.executed)


def test_find_buy_memories_for_ticker_uses_structured_payload_before_summary_fallback() -> None:
    session = _FakeSession()
    store = MemoryBQStore(session)

    rows = store.find_buy_memories_for_ticker(
        agent_id="gpt",
        ticker="aapl",
        limit=3,
        trading_mode="live",
        tenant_id="tenant-a",
    )

    assert rows == []
    assert session.fetched
    sql, params = session.fetched[0]
    assert "JSON_VALUE(payload_json, '$.intent.ticker')" in sql
    assert "JSON_VALUE(payload_json, '$.intent.side')" in sql
    assert "OR summary LIKE CONCAT(@ticker, ' BUY%')" in sql
    assert params["ticker"] == "AAPL"


def test_upsert_memory_relation_triples_merges_with_json_detail() -> None:
    session = _FakeSession()
    store = MemoryBQStore(session)
    created_at = datetime(2026, 3, 29, 12, 30, tzinfo=timezone.utc)

    store.upsert_memory_relation_triples(
        [
            {
                "triple_id": "triple:abc",
                "created_at": created_at,
                "source_table": "agent_memory_events",
                "source_id": "evt_1",
                "source_node_id": "mem:evt_1",
                "source_created_at": created_at,
                "agent_id": "gpt",
                "trading_mode": "paper",
                "cycle_id": "cycle_1",
                "subject_node_id": "mem:evt_1",
                "subject_label": "memory_event:evt_1",
                "subject_type": "passage",
                "predicate": "contains",
                "object_node_id": "ticker:AAPL",
                "object_label": "AAPL",
                "object_type": "ticker",
                "confidence": 1.0,
                "evidence_text": "AAPL BUY worked.",
                "extraction_method": "deterministic",
                "extraction_version": "deterministic_v1",
                "status": "accepted",
                "detail_json": {"field": "ticker"},
            }
        ],
        tenant_id="tenant-a",
    )

    assert session.executed
    sql, params = session.executed[0]
    assert "memory_relation_triples" in sql
    assert params["triple_id"] == "triple:abc"
    assert params["predicate"] == "contains"
    assert params["detail_json"][0] == "JSON"
    assert '"field":"ticker"' in params["detail_json"][1]


def test_upsert_memory_relation_triples_with_graph_projects_only_accepted() -> None:
    session = _FakeSession()
    store = MemoryBQStore(session)
    created_at = datetime(2026, 3, 29, 12, 30, tzinfo=timezone.utc)

    store.upsert_memory_relation_triples_with_graph(
        [
            {
                "triple_id": "triple:rejected",
                "created_at": created_at,
                "source_table": "agent_memory_events",
                "source_id": "evt_1",
                "source_node_id": "mem:evt_1",
                "source_created_at": created_at,
                "agent_id": "gpt",
                "trading_mode": "paper",
                "cycle_id": "cycle_1",
                "subject_node_id": "mem:evt_1",
                "subject_label": "memory_event:evt_1",
                "subject_type": "passage",
                "predicate": "contains",
                "object_node_id": "ticker:AAPL",
                "object_label": "AAPL",
                "object_type": "ticker",
                "confidence": 0.2,
                "evidence_text": "AAPL",
                "extraction_method": "llm",
                "extraction_version": "semantic_relation_extractor_v1",
                "status": "rejected",
                "detail_json": {"reason": "low_confidence"},
            }
        ],
        tenant_id="tenant-a",
    )

    assert len(session.executed) == 1
    assert "memory_relation_triples" in session.executed[0][0]


def test_append_memory_relation_extraction_runs_serializes_json() -> None:
    session = _FakeSession()
    store = MemoryBQStore(session)
    started_at = datetime(2026, 3, 29, 12, 30, tzinfo=timezone.utc)

    store.append_memory_relation_extraction_runs(
        [
            {
                "run_id": "rel_extract_1",
                "started_at": started_at,
                "finished_at": started_at,
                "source_table": "agent_memory_events",
                "source_id": "evt_1",
                "source_hash": "hash_1",
                "source_created_at": started_at,
                "agent_id": "gpt",
                "trading_mode": "paper",
                "cycle_id": "cycle_1",
                "extractor_version": "semantic_relation_extractor_v1",
                "prompt_version": "semantic_relation_prompt_v1",
                "ontology_version": "semantic_relation_ontology_v1",
                "provider": "gpt",
                "model": "openai/gpt-5.2",
                "status": "success",
                "accepted_count": 1,
                "rejected_count": 1,
                "raw_output_json": {"triples": []},
                "detail_json": {"rejected": [{"reason": "low_confidence"}]},
            }
        ],
        tenant_id="tenant-a",
    )

    assert session.client.inserts
    table_id, rows = session.client.inserts[0]
    assert table_id == "proj.ds.memory_relation_extraction_runs"
    assert rows[0]["tenant_id"] == "tenant-a"
    assert rows[0]["started_at"] == "2026-03-29 12:30:00.000000"
    assert '"triples":[]' in rows[0]["raw_output_json"]
    assert '"low_confidence"' in rows[0]["detail_json"]


def test_append_memory_relation_tuning_runs_serializes_metrics() -> None:
    session = _FakeSession()
    store = MemoryBQStore(session)
    evaluated_at = datetime(2026, 3, 29, 12, 30, tzinfo=timezone.utc)

    store.append_memory_relation_tuning_runs(
        [
            {
                "run_id": "rel_tune_1",
                "evaluated_at": evaluated_at,
                "trading_mode": "paper",
                "configured_mode": "shadow",
                "effective_mode": "inject",
                "recommended_mode": "inject",
                "transition_action": "auto_promote_to_inject",
                "reason": "healthy",
                "source_count": 12,
                "accepted_count": 8,
                "rejected_count": 2,
                "unsafe_reject_count": 0,
                "failed_run_count": 0,
                "invalid_output_count": 0,
                "accepted_rate": 0.8,
                "unsafe_reject_rate": 0.0,
                "strong_predicate_ratio": 0.1,
                "conflict_ratio": 0.0,
                "source_concentration": 0.25,
                "ticker_concentration": 0.25,
                "sample_ok": True,
                "health_ok": True,
                "stability_ok": True,
                "detail_json": {"gates": {"health_ok": True}},
            }
        ],
        tenant_id="tenant-a",
    )

    assert session.client.inserts
    table_id, rows = session.client.inserts[0]
    assert table_id == "proj.ds.memory_relation_tuning_runs"
    assert rows[0]["effective_mode"] == "inject"
    assert rows[0]["health_ok"] is True
    assert '"health_ok":true' in rows[0]["detail_json"]


def test_relation_extraction_pending_sources_filters_successful_runs() -> None:
    session = _FakeSession()
    store = MemoryBQStore(session)

    rows = store.relation_extraction_pending_sources(
        tenant_id="tenant-a",
        limit=4,
        source_table="agent_memory_events",
        event_types=["strategy_reflection"],
        trading_mode="paper",
        extractor_version="semantic_relation_extractor_v1",
        prompt_version="semantic_relation_prompt_v1",
        ontology_version="semantic_relation_ontology_v1",
    )

    assert rows == []
    assert session.fetched
    sql, params = session.fetched[0]
    assert "memory_relation_extraction_runs" in sql
    assert "run.status = 'success'" in sql
    assert "TO_HEX(SHA256(source_text)) AS source_hash" in sql
    assert params["source_table"] == "agent_memory_events"
    assert params["event_types"] == ["strategy_reflection"]


def test_memory_graph_neighbors_keeps_relation_triples_shadowed() -> None:
    session = _FakeSession()
    store = MemoryBQStore(session)

    rows = store.memory_graph_neighbors(
        seed_node_ids=["mem:evt_1"],
        trading_mode="paper",
        min_confidence=0.5,
        limit=5,
        tenant_id="tenant-a",
    )

    assert rows == []
    assert session.fetched
    sql, params = session.fetched[0]
    assert "JSON_VALUE(e.detail_json, '$.triple_id')" in sql
    assert params["seed_node_ids"] == ["mem:evt_1"]


def test_memory_relation_memory_candidates_joins_relation_triples_to_memory_events() -> None:
    session = _FakeSession()
    store = MemoryBQStore(session)

    rows = store.memory_relation_memory_candidates(
        agent_id="gpt",
        seed_node_ids=["ticker:AAPL"],
        trading_mode="paper",
        min_confidence=0.8,
        limit=7,
        tenant_id="tenant-a",
    )

    assert rows == []
    assert session.fetched
    sql, params = session.fetched[0]
    assert "memory_relation_triples" in sql
    assert "agent_memory_events" in sql
    assert "rel.source_table = 'agent_memory_events'" in sql
    assert params["seed_node_ids"] == ["ticker:AAPL"]
    assert params["min_confidence"] == 0.8
    assert params["limit"] == 7
