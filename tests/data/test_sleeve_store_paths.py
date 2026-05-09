from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from arena.data.bigquery.ledger_store import LedgerStore
from arena.data.bigquery.market_store import MarketStore
from arena.data.bigquery.sleeve_store import SleeveStore
from arena.models import AccountSnapshot, Position


# ---------------------------------------------------------------------------
# Shared fake infrastructure
# ---------------------------------------------------------------------------

from tests.data.strict_path_helpers import (
    _InsertClient,
    _FakeSession,
    _ForecastSchemaField,
    _ForecastSchemaClient,
    _LoadJob,
    _MarketWriteClient,
    _make_market_store,
    _make_forecast_query_store,
    _make_market_write_store,
    _SleeveStoreForBuild,
    _SleeveStoreForInit,
    _SleeveStoreForRetarget,
    _LedgerStoreForCapitalReplay,
    _MarketStoreForReplay,
    _SleeveStoreForCapitalReplay,
    _SleeveStoreForCapitalRetarget,
    _LedgerStoreForCapitalRetarget,
    _NavSleeveStore,
    _ActualBasisSleeveStore,
    _make_capital_replay_store,
)

def test_build_agent_sleeve_snapshot_propagates_execution_history_error() -> None:
    session = _FakeSession(responses=[RuntimeError("execution reports timeout")])
    market = _MarketStoreForReplay()
    store = _SleeveStoreForBuild(session, fill_result=RuntimeError("execution reports timeout"), market=market)

    with pytest.raises(RuntimeError, match="execution reports timeout"):
        store.build_agent_sleeve_snapshot(agent_id="agent-1")


def test_build_agent_sleeve_snapshot_rejects_invalid_initial_positions_json() -> None:
    session = _FakeSession(responses=[[]])
    market = _MarketStoreForReplay()
    store = _SleeveStoreForBuild(session, fill_result=[], init_positions_json="{bad json", market=market)

    with pytest.raises(RuntimeError, match="invalid initial_positions_json"):
        store.build_agent_sleeve_snapshot(agent_id="agent-1")


# ===================================================================
# Tests — Sleeve Store: ensure_agent_sleeves
# ===================================================================


def test_ensure_agent_sleeves_uses_virtual_seed_when_bootstrap_disabled(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)
    snapshot = AccountSnapshot(
        cash_krw=900_000.0,
        total_equity_krw=1_200_000.0,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                exchange_code="NASD",
                instrument_id="NASD:AAPL",
                quantity=6.0,
                avg_price_krw=50_000.0,
                market_price_krw=55_000.0,
            )
        },
    )
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=snapshot)

    store.ensure_agent_sleeves(agent_ids=["gpt", "gemini", "claude"], total_cash_krw=3_000_000.0)

    assert len(session.client.payloads) == 3
    first = session.client.payloads[0]
    assert float(first["initial_cash_krw"]) == 1_000_000.0
    assert str(first["initial_positions_json"]) == "[]"


def test_write_account_snapshot_persists_usd_krw_rate() -> None:
    snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=200_000.0,
        positions={},
        usd_krw_rate=1450.0,
    )
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=snapshot)

    store.write_account_snapshot(store._snapshot)

    assert float(session.client.payloads[0]["usd_krw_rate"]) == pytest.approx(1450.0)


def test_write_account_snapshot_persists_market_scope() -> None:
    snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=200_000.0,
        positions={},
    )
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=snapshot)

    store.write_account_snapshot(store._snapshot, market_scope="us,kospi,kosdaq")

    assert session.client.payloads[0]["market_scope"] == "us,kospi,kosdaq"


def test_write_account_snapshot_appends_broker_cash_checkpoint() -> None:
    snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=200_000.0,
        positions={},
        usd_krw_rate=1450.0,
        cash_foreign=50.0,
        cash_foreign_currency="USD",
    )
    # Ledger store needs a session that handles the dedup check
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids returns empty
    ])
    ledger = LedgerStore(ledger_session)
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=snapshot, ledger=ledger)

    store.write_account_snapshot(store._snapshot)

    assert [table_id for table_id, _ in session.client.calls] == [
        "proj.ds.account_snapshots",
    ]
    # The broker cash event is written via ledger store's session client
    assert [table_id for table_id, _ in ledger_session.client.calls] == [
        "proj.ds.broker_cash_events",
    ]
    _, cash_rows = ledger_session.client.calls[0]
    assert cash_rows[0]["event_type"] == "CASH_CHECKPOINT"
    assert float(cash_rows[0]["amount_krw"]) == pytest.approx(100_000.0)
    assert float(cash_rows[0]["amount_native"]) == pytest.approx(50.0)
    assert cash_rows[0]["currency"] == "USD"


def test_latest_account_snapshot_reads_usd_krw_rate() -> None:
    session = _FakeSession(
        responses=[
            [
                {
                    "snapshot_at": datetime(2026, 2, 21, tzinfo=timezone.utc),
                    "cash_krw": 100_000.0,
                    "total_equity_krw": 200_000.0,
                    "usd_krw_rate": 1450.0,
                }
            ],
            [],  # position rows
        ]
    )
    store = SleeveStore(session)

    snapshot = store.latest_account_snapshot()

    assert snapshot is not None
    assert snapshot.usd_krw_rate == pytest.approx(1450.0)


def test_latest_account_snapshot_filters_market_scope() -> None:
    session = _FakeSession(
        responses=[
            [
                {
                    "snapshot_at": datetime(2026, 2, 21, tzinfo=timezone.utc),
                    "cash_krw": 100_000.0,
                    "total_equity_krw": 200_000.0,
                    "usd_krw_rate": 1450.0,
                }
            ],
            [],
        ]
    )
    store = SleeveStore(session)

    snapshot = store.latest_account_snapshot(tenant_id="midnightnnn", market_scope="us,kospi,kosdaq")

    assert snapshot is not None
    assert "market_scope = @market_scope" in session.calls[0]
    assert session.call_pairs[0][1]["market_scope"] == "us,kospi,kosdaq"


def test_get_latest_position_tickers_can_union_latest_snapshots_across_tenants() -> None:
    session = _FakeSession(
        responses=[
            [
                {"ticker": "VZ"},
                {"ticker": "005930"},
                {"ticker": "CSX"},
            ]
        ]
    )
    store = SleeveStore(session)

    tickers = store.get_latest_position_tickers(market="us", all_tenants=True)

    assert tickers == ["VZ", "CSX"]
    assert "PARTITION BY tenant_id" in session.calls[0]
    assert session.call_pairs[0][1] == {"market_scope_like_0": "%,us,%"}


def test_get_latest_position_tickers_scopes_kosdaq_latest_snapshot() -> None:
    session = _FakeSession(
        responses=[
            [
                {"ticker": "053580"},
                {"ticker": "AAPL"},
            ]
        ]
    )
    store = SleeveStore(session)

    tickers = store.get_latest_position_tickers(market="kosdaq", all_tenants=True)

    assert tickers == ["053580"]
    sql, params = session.call_pairs[0]
    assert "market_scope" in sql
    assert "ROW_NUMBER()" in sql
    assert params == {"market_scope_like_0": "%,kosdaq,%"}


def test_get_latest_position_tickers_requires_us_and_kr_for_combined_scope() -> None:
    session = _FakeSession(responses=[[{"ticker": "053580"}, {"ticker": "AAPL"}]])
    store = SleeveStore(session)

    tickers = store.get_latest_position_tickers(market="us,kospi,kosdaq", all_tenants=True)

    assert tickers == ["053580", "AAPL"]
    sql, params = session.call_pairs[0]
    assert "LIKE @market_scope_like_0) AND (" in sql
    assert params == {
        "market_scope_like_0": "%,us,%",
        "market_scope_like_1": "%,kospi,%",
        "market_scope_like_2": "%,kosdaq,%",
    }


def test_account_holdings_at_date_uses_latest_snapshot_before_date() -> None:
    session = _FakeSession(
        responses=[
            [
                {"ticker": "AAPL", "quantity": 2.0},
                {"ticker": "MSFT", "quantity": 1.0},
            ]
        ]
    )
    store = SleeveStore(session)

    holdings = store.account_holdings_at_date(as_of_date=date(2026, 2, 21))

    assert holdings == {"AAPL": pytest.approx(2.0), "MSFT": pytest.approx(1.0)}


def test_account_cash_history_uses_range_filter() -> None:
    start_at = datetime(2026, 2, 21, tzinfo=timezone.utc)
    end_at = datetime(2026, 2, 22, tzinfo=timezone.utc)
    cash_rows = [
        {
            "snapshot_at": start_at,
            "cash_krw": 100_000.0,
            "total_equity_krw": 200_000.0,
            "usd_krw_rate": 1450.0,
            "cash_foreign": 50.0,
            "cash_foreign_currency": "USD",
        }
    ]
    session = _FakeSession(responses=[list(cash_rows)])
    store = SleeveStore(session)

    rows = store.account_cash_history(start_at=start_at, end_at=end_at, tenant_id="midnightnnn")

    assert rows[0]["cash_krw"] == pytest.approx(100_000.0)
    last_sql = session.calls[-1]
    assert "FROM `proj.ds.account_snapshots`" in last_sql
    last_params = session.call_pairs[-1][1]
    assert last_params == {
        "tenant_id": "midnightnnn",
        "start_at": start_at,
        "end_at": end_at,
    }


def test_ensure_agent_sleeves_can_seed_from_account_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", "true")
    snapshot = AccountSnapshot(
        cash_krw=900_000.0,
        total_equity_krw=1_200_000.0,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                exchange_code="NASD",
                instrument_id="NASD:AAPL",
                quantity=6.0,
                avg_price_krw=50_000.0,
                market_price_krw=55_000.0,
            )
        },
    )
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=snapshot)

    store.ensure_agent_sleeves(agent_ids=["gpt", "gemini", "claude"], total_cash_krw=3_000_000.0)

    assert len(session.client.payloads) == 3
    first = session.client.payloads[0]
    assert float(first["initial_cash_krw"]) == 300_000.0
    seeded = json.loads(str(first["initial_positions_json"]))
    assert len(seeded) == 1
    assert float(seeded[0]["quantity"]) == pytest.approx(2.0)
    assert float(seeded[0]["avg_price_krw"]) == pytest.approx(50_000.0)


def test_ensure_agent_sleeves_mirrors_seed_into_agent_state_checkpoints(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)
    # Ledger session needs to handle: existing_event_ids (dedup) returning empty
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids for agent_state_checkpoints
    ])
    ledger = LedgerStore(ledger_session)
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=None, ledger=ledger)

    store.ensure_agent_sleeves(agent_ids=["gpt", "gemini"], total_cash_krw=2_000_000.0)

    # sleeve inserts go to the sleeve session's client
    # checkpoint inserts go to the ledger session's client
    sleeve_tables = [table_id for table_id, _ in session.client.calls]
    ledger_tables = [table_id for table_id, _ in ledger_session.client.calls]
    assert sleeve_tables == ["proj.ds.agent_sleeves"]
    assert ledger_tables == ["proj.ds.agent_state_checkpoints"]
    _, checkpoint_rows = ledger_session.client.calls[0]
    assert {row["agent_id"] for row in checkpoint_rows} == {"gpt", "gemini"}
    assert all(row["source"] == "agent_sleeves.ensure" for row in checkpoint_rows)
    assert all(json.loads(str(row["positions_json"])) == [] for row in checkpoint_rows)


def test_ensure_agent_state_checkpoints_writes_checkpoint_only(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)

    class _LedgerForCheckpointEnsure(LedgerStore):
        def latest_agent_state_checkpoints(self, *, agent_ids, tenant_id=None):
            _ = tenant_id
            return {}

        def existing_event_ids(self, table_name, event_ids, *, tenant_id=None):
            _ = (table_name, event_ids, tenant_id)
            return set()

    ledger_session = _FakeSession()
    ledger = _LedgerForCheckpointEnsure(ledger_session)
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=None, ledger=ledger)

    store.ensure_agent_state_checkpoints(agent_ids=["gpt", "gemini"], total_cash_krw=2_000_000.0)

    assert [table_id for table_id, _ in ledger_session.client.calls] == ["proj.ds.agent_state_checkpoints"]
    _, checkpoint_rows = ledger_session.client.calls[0]
    assert {row["agent_id"] for row in checkpoint_rows} == {"gpt", "gemini"}
    assert all(row["source"] == "agent_state_checkpoints.ensure" for row in checkpoint_rows)
    assert all(float(row["cash_krw"]) == pytest.approx(1_000_000.0) for row in checkpoint_rows)


# ===================================================================
# Tests — Sleeve Store: capital replay via build_agent_sleeve_snapshot
# ===================================================================
