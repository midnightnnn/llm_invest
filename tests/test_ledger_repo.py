from __future__ import annotations

import json
from datetime import datetime, timezone

from arena.data.bigquery.ledger_store import LedgerStore
from arena.data.schema import parse_ddl_columns, render_table_ddls


class _InsertClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def insert_rows_json(self, table_id: str, rows: list[dict[str, object]]):
        self.calls.append((table_id, rows))
        return []


class _FakeSession:
    """Mock BigQuerySession for LedgerStore tests."""

    def __init__(self) -> None:
        self.dataset_fqn = "proj.ds"
        self.tenant_id = "local"
        self.client = _InsertClient()
        self._latest_rows: list[dict[str, object]] = []
        self._existing_event_ids: set[str] = set()
        self.last_fetch_sql = ""
        self.last_fetch_params = None

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return tenant_id or self.tenant_id

    def fetch_rows(self, sql: str, params=None):
        self.last_fetch_sql = sql
        self.last_fetch_params = params
        if "SELECT event_id" in sql:
            tokens = set(((params or {}).get("event_ids") or []))
            return [{"event_id": token} for token in tokens if token in self._existing_event_ids]
        return list(self._latest_rows)

    def execute(self, sql: str, params=None):
        pass


def _make_store() -> tuple[LedgerStore, _FakeSession]:
    session = _FakeSession()
    return LedgerStore(session), session


def test_schema_includes_phase1_ledger_tables() -> None:
    ddls = "\n".join(render_table_ddls("proj", "ds"))
    cols = parse_ddl_columns()

    assert "proj.ds.broker_trade_events" in ddls
    assert "proj.ds.broker_cash_events" in ddls
    assert "proj.ds.capital_events" in ddls
    assert "proj.ds.agent_transfer_events" in ddls
    assert "proj.ds.manual_adjustments" in ddls
    assert "proj.ds.agent_state_checkpoints" in ddls
    assert "proj.ds.reconciliation_runs" in ddls
    assert "proj.ds.reconciliation_issues" in ddls
    assert "proj.ds.tenant_run_statuses" in ddls
    assert "proj.ds.official_nav_daily" in ddls
    assert "proj.ds.fundamentals_snapshot_latest" in ddls
    assert ("summary_json", "JSON") in cols["reconciliation_runs"]
    assert ("detail_json", "JSON") in cols["tenant_run_statuses"]
    assert ("raw_payload_json", "JSON") in cols["broker_trade_events"]
    assert ("positions_json", "JSON") in cols["agent_state_checkpoints"]


def test_append_capital_events_sets_tenant_and_serializes_datetime() -> None:
    store, session = _make_store()
    occurred_at = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    store.append_capital_events(
        [
            {
                "event_id": "cap_1",
                "occurred_at": occurred_at,
                "agent_id": "gpt",
                "amount_krw": 1_000_000.0,
                "event_type": "INJECTION",
            }
        ],
        tenant_id="midnightnnn",
    )

    table_id, rows = session.client.calls[0]
    assert table_id == "proj.ds.capital_events"
    assert rows[0]["tenant_id"] == "midnightnnn"
    assert rows[0]["occurred_at"] == occurred_at.isoformat()


def test_append_broker_trade_events_serializes_json_columns() -> None:
    store, session = _make_store()
    occurred_at = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    store.append_broker_trade_events(
        [
            {
                "event_id": "trade_1",
                "occurred_at": occurred_at,
                "ticker": "AAPL",
                "side": "BUY",
                "quantity": 1.0,
                "price_krw": 150000.0,
                "status": "FILLED",
                "raw_payload_json": {"ODNO": "12345", "nested": {"qty": 1}},
            }
        ],
        tenant_id="midnightnnn",
    )

    table_id, rows = session.client.calls[0]
    assert table_id == "proj.ds.broker_trade_events"
    assert rows[0]["tenant_id"] == "midnightnnn"
    assert rows[0]["occurred_at"] == occurred_at.isoformat()
    assert rows[0]["raw_payload_json"] == json.dumps(
        {"ODNO": "12345", "nested": {"qty": 1}},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_append_agent_state_checkpoints_serializes_positions_and_detail_json() -> None:
    store, session = _make_store()
    checkpoint_at = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    store.append_agent_state_checkpoints(
        [
            {
                "event_id": "ckpt_1",
                "checkpoint_at": checkpoint_at,
                "agent_id": "gpt",
                "cash_krw": 12345.0,
                "positions_json": [{"ticker": "AAPL", "quantity": 1.0}],
                "detail_json": {"seed_source": "agent_sleeves"},
            }
        ],
        tenant_id="midnightnnn",
    )

    table_id, rows = session.client.calls[0]
    assert table_id == "proj.ds.agent_state_checkpoints"
    assert rows[0]["positions_json"] == json.dumps(
        [{"ticker": "AAPL", "quantity": 1.0}],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert rows[0]["detail_json"] == json.dumps(
        {"seed_source": "agent_sleeves"},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_latest_reconciliation_run_returns_first_row() -> None:
    store, session = _make_store()
    session._latest_rows = [
        {
            "run_id": "recon_1",
            "run_at": datetime(2026, 3, 12, 1, 0, tzinfo=timezone.utc),
            "snapshot_at": datetime(2026, 3, 12, 0, 30, tzinfo=timezone.utc),
            "status": "ok",
            "summary_json": {"issues": 0},
        }
    ]

    row = store.latest_reconciliation_run()

    assert row is not None
    assert row["run_id"] == "recon_1"
    assert row["status"] == "ok"


def test_append_capital_events_skips_existing_and_duplicate_event_ids() -> None:
    store, session = _make_store()
    session._existing_event_ids = {"cap_existing"}

    store.append_capital_events(
        [
            {
                "event_id": "cap_existing",
                "occurred_at": datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 100.0,
                "event_type": "INJECTION",
            },
            {
                "event_id": "cap_new",
                "occurred_at": datetime(2026, 3, 12, 0, 1, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 200.0,
                "event_type": "INJECTION",
            },
            {
                "event_id": "cap_new",
                "occurred_at": datetime(2026, 3, 12, 0, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 300.0,
                "event_type": "INJECTION",
            },
        ]
    )

    assert len(session.client.calls) == 1
    table_id, rows = session.client.calls[0]
    assert table_id == "proj.ds.capital_events"
    assert [row["event_id"] for row in rows] == ["cap_new"]


def test_append_agent_state_checkpoints_sets_tenant_and_serializes_positions_json() -> None:
    store, session = _make_store()
    checkpoint_at = datetime(2026, 3, 12, 1, 0, tzinfo=timezone.utc)

    store.append_agent_state_checkpoints(
        [
            {
                "event_id": "chk_1",
                "checkpoint_at": checkpoint_at,
                "agent_id": "gpt",
                "cash_krw": 123_456.0,
                "positions_json": [{"ticker": "AAPL", "quantity": 2.0}],
                "source": "test",
            }
        ],
        tenant_id="midnightnnn",
    )

    table_id, rows = session.client.calls[0]
    assert table_id == "proj.ds.agent_state_checkpoints"
    assert rows[0]["tenant_id"] == "midnightnnn"
    assert rows[0]["checkpoint_at"] == checkpoint_at.isoformat()
    assert rows[0]["positions_json"] == json.dumps(
        [{"ticker": "AAPL", "quantity": 2.0}],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_broker_trade_events_since_uses_since_and_status_filters() -> None:
    store, session = _make_store()
    since = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    store.broker_trade_events_since(since=since, tenant_id="midnightnnn", statuses=["FILLED", "SETTLED"])

    assert "FROM `proj.ds.broker_trade_events`" in session.last_fetch_sql
    assert session.last_fetch_params == {
        "tenant_id": "midnightnnn",
        "since": since,
        "statuses": ["FILLED", "SETTLED"],
    }


def test_manual_position_adjustments_since_uses_since_filter() -> None:
    store, session = _make_store()
    since = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    store.manual_position_adjustments_since(since=since, tenant_id="midnightnnn")

    assert "FROM `proj.ds.manual_adjustments`" in session.last_fetch_sql
    assert session.last_fetch_params == {
        "tenant_id": "midnightnnn",
        "since": since,
    }


def test_broker_cash_events_since_uses_since_filter() -> None:
    store, session = _make_store()
    since = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    store.broker_cash_events_since(since=since, tenant_id="midnightnnn")

    assert "FROM `proj.ds.broker_cash_events`" in session.last_fetch_sql
    assert session.last_fetch_params == {
        "tenant_id": "midnightnnn",
        "since": since,
    }


def test_manual_cash_adjustments_since_uses_agent_and_since_filter() -> None:
    store, session = _make_store()
    since = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    store.manual_cash_adjustments_since(agent_id="gpt", since=since, tenant_id="midnightnnn")

    assert "FROM `proj.ds.manual_adjustments`" in session.last_fetch_sql
    assert session.last_fetch_params == {
        "tenant_id": "midnightnnn",
        "agent_id": "gpt",
        "since": since,
    }


def test_latest_agent_state_checkpoints_uses_partitioned_latest_query() -> None:
    store, session = _make_store()

    store.latest_agent_state_checkpoints(agent_ids=["gpt", "gemini"], tenant_id="midnightnnn")

    assert "FROM `proj.ds.agent_state_checkpoints`" in session.last_fetch_sql
    assert session.last_fetch_params == {
        "tenant_id": "midnightnnn",
        "agent_ids": ["gpt", "gemini"],
    }


def test_capital_events_since_uses_agent_since_and_type_filters() -> None:
    store, session = _make_store()
    since = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    store.capital_events_since(
        agent_id="gpt",
        since=since,
        tenant_id="midnightnnn",
        event_types=["INJECTION", "WITHDRAWAL"],
    )

    assert "FROM `proj.ds.capital_events`" in session.last_fetch_sql
    assert session.last_fetch_params == {
        "tenant_id": "midnightnnn",
        "agent_id": "gpt",
        "since": since,
        "event_types": ["INJECTION", "WITHDRAWAL"],
    }


def test_agent_transfer_events_since_filters_for_agent_scope() -> None:
    store, session = _make_store()
    since = datetime(2026, 3, 12, 0, 0, tzinfo=timezone.utc)

    store.agent_transfer_events_since(agent_id="gpt", since=since, tenant_id="midnightnnn")

    assert "FROM `proj.ds.agent_transfer_events`" in session.last_fetch_sql
    assert session.last_fetch_params == {
        "tenant_id": "midnightnnn",
        "agent_id": "gpt",
        "since": since,
    }
