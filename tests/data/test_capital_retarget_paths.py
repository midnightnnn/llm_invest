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

def test_retarget_agent_sleeves_preserve_positions_sets_cash_from_target_gap() -> None:
    session = _FakeSession()
    store = _SleeveStoreForRetarget(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=100_000.0,
                total_equity_krw=400_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000.0,
                        market_price_krw=150_000.0,
                    )
                },
            )
        },
    )

    out = store.retarget_agent_sleeves_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000.0,
    )

    assert len(session.client.payloads) == 1
    first = session.client.payloads[0]
    assert float(first["initial_cash_krw"]) == pytest.approx(200_000.0)
    seeded = json.loads(str(first["initial_positions_json"]))
    assert len(seeded) == 1
    assert float(seeded[0]["quantity"]) == pytest.approx(2.0)
    assert float(seeded[0]["avg_price_krw"]) == pytest.approx(150_000.0)
    assert out["gpt"]["over_target"] is False


def test_retarget_agent_capitals_preserve_positions_appends_capital_events() -> None:
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids for capital_events
        [],  # existing_event_ids for agent_state_checkpoints
    ])
    ledger = _LedgerStoreForCapitalRetarget(ledger_session)
    session = _FakeSession()
    store = _SleeveStoreForCapitalRetarget(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=100_000.0,
                total_equity_krw=400_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000.0,
                        market_price_krw=150_000.0,
                    )
                },
            )
        },
        ledger=ledger,
    )

    out = store.retarget_agent_capitals_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000.0,
        created_by="tester",
    )

    assert len(ledger_session.client.calls) == 2
    table_id, rows = ledger_session.client.calls[0]
    assert table_id == "proj.ds.capital_events"
    assert rows[0]["agent_id"] == "gpt"
    assert float(rows[0]["amount_krw"]) == pytest.approx(100_000.0)
    assert rows[0]["event_type"] == "INJECTION"
    assert out["gpt"]["target_cash_krw"] == pytest.approx(200_000.0)
    assert out["gpt"]["capital_flow_krw"] == pytest.approx(100_000.0)
    assert out["gpt"]["over_target"] is False

    # Checkpoint is also synced after capital event
    cp_table_id, cp_rows = ledger_session.client.calls[1]
    assert cp_table_id == "proj.ds.agent_state_checkpoints"
    assert cp_rows[0]["agent_id"] == "gpt"
    assert float(cp_rows[0]["cash_krw"]) == pytest.approx(200_000.0)
    assert cp_rows[0]["source"] == "capital_events.retarget"


def test_retarget_agent_capitals_preserves_pnl_on_capital_change() -> None:
    """When capital is raised, existing P&L must be preserved on top of new capital."""

    class _PnlAwareStore(_SleeveStoreForCapitalRetarget):
        def __init__(self, session, *, snapshots, baselines, ledger):
            super().__init__(session, snapshots=snapshots, ledger=ledger)
            self._baselines = baselines

        def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True, tenant_id=None, as_of_ts=None):
            _ = (sources, include_simulated, tenant_id, as_of_ts)
            snap = self._snapshots[str(agent_id)]
            baseline = self._baselines[str(agent_id)]
            return snap, baseline, {"agent_id": str(agent_id)}

    # Agent started with 340k capital, now has 60k profit -> equity 400k
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids for capital_events
        [],  # existing_event_ids for agent_state_checkpoints
    ])
    ledger = _LedgerStoreForCapitalRetarget(ledger_session)
    session = _FakeSession()
    store = _PnlAwareStore(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=100_000.0,
                total_equity_krw=400_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000.0,
                        market_price_krw=150_000.0,
                    )
                },
            )
        },
        baselines={"gpt": 340_000.0},
        ledger=ledger,
    )

    out = store.retarget_agent_capitals_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000.0,
        created_by="tester",
    )

    meta = out["gpt"]
    # delta = 500k - 340k = 160k (not 100k like old absolute-target mode)
    assert meta["capital_flow_krw"] == pytest.approx(160_000.0)
    # new cash = 100k + 160k = 260k
    assert meta["target_cash_krw"] == pytest.approx(260_000.0)
    # effective equity = 400k + 160k = 560k = 500k(new capital) + 60k(pnl)
    assert meta["effective_target_equity_krw"] == pytest.approx(560_000.0)
    assert meta["over_target"] is False

    table_id, rows = ledger_session.client.calls[0]
    assert rows[0]["event_type"] == "INJECTION"
    assert float(rows[0]["amount_krw"]) == pytest.approx(160_000.0)


def test_retarget_agent_capitals_clamps_cash_when_withdrawal_exceeds_available() -> None:
    """When capital reduction requires more cash withdrawal than available, clamp to 0."""

    class _PnlAwareStore(_SleeveStoreForCapitalRetarget):
        def __init__(self, session, *, snapshots, baselines, ledger):
            super().__init__(session, snapshots=snapshots, ledger=ledger)
            self._baselines = baselines

        def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True, tenant_id=None, as_of_ts=None):
            _ = (sources, include_simulated, tenant_id, as_of_ts)
            snap = self._snapshots[str(agent_id)]
            baseline = self._baselines[str(agent_id)]
            return snap, baseline, {"agent_id": str(agent_id)}

    # baseline 500k, cash 50k, positions 400k, equity 450k (pnl = -50k)
    # target = 200k -> delta = 200k - 500k = -300k -> new_cash = 50k - 300k = -250k -> clamp
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids for capital_events
        [],  # existing_event_ids for agent_state_checkpoints
    ])
    ledger = _LedgerStoreForCapitalRetarget(ledger_session)
    session = _FakeSession()
    store = _PnlAwareStore(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=50_000.0,
                total_equity_krw=450_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=180_000.0,
                        market_price_krw=200_000.0,
                    )
                },
            )
        },
        baselines={"gpt": 500_000.0},
        ledger=ledger,
    )

    out = store.retarget_agent_capitals_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=200_000.0,
        created_by="tester",
    )

    meta = out["gpt"]
    # Clamped: withdraw only available cash (50k), not full 300k
    assert meta["capital_flow_krw"] == pytest.approx(-50_000.0)
    assert meta["target_cash_krw"] == pytest.approx(0.0)
    assert meta["over_target"] is True


def test_retarget_agent_sleeves_preserve_positions_clamps_cash_when_over_target() -> None:
    session = _FakeSession()
    store = _SleeveStoreForRetarget(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=0.0,
                total_equity_krw=600_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=250_000.0,
                        market_price_krw=300_000.0,
                    )
                },
            )
        },
    )

    out = store.retarget_agent_sleeves_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000.0,
    )

    assert len(session.client.payloads) == 1
    first = session.client.payloads[0]
    assert float(first["initial_cash_krw"]) == 0.0
    assert out["gpt"]["over_target"] is True
