from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from tests.data.strict_path_helpers import (
    _ActualBasisSleeveStore,
    _FakeSession,
    _LedgerStoreForCapitalReplay,
    _NavSleeveStore,
    _SleeveStoreForCapitalReplay,
)


def test_agent_holdings_at_date_replays_agent_transfer_events() -> None:
    # agent_holdings_at_date calls:
    # 1. _load_agent_seed_state -> ledger.latest_agent_state_checkpoints -> sleeve.latest_agent_sleeves -> session.fetch_rows (fills) -> ledger.agent_transfer_events_since
    # We need the sleeve session to handle execution_reports fetch (returns []),
    # plus the agent_holdings_at_date fetch (returns []).
    sleeve_session = _FakeSession(responses=[
        [],  # execution_reports fill query
    ])
    ledger = _LedgerStoreForCapitalReplay(
        _FakeSession(),
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 100_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
        transfer_events=[
            {
                "event_id": "xfer_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "transfer_type": "POSITION_TRANSFER",
                "from_agent_id": "gemini",
                "to_agent_id": "gpt",
                "ticker": "AAPL",
                "quantity": 1.0,
                "price_krw": 50_000.0,
                "amount_krw": 50_000.0,
            }
        ],
    )

    store = _SleeveStoreForCapitalReplay(sleeve_session, ledger=ledger)

    holdings = store.agent_holdings_at_date(agent_id="gpt", as_of_date=date(2026, 3, 3))

    assert holdings == {"AAPL": pytest.approx(1.0)}


def test_trace_agent_actual_capital_basis_replays_real_cash_events_from_origin() -> None:
    origin_state = {
        "source": "legacy_agent_sleeve",
        "since": datetime(2026, 3, 1, tzinfo=timezone.utc),
        "cash_krw": 1_000_000.0,
        "positions_payload": [],
        "positions_error": None,
    }
    ledger = _LedgerStoreForCapitalReplay(
        _FakeSession(),
        capital_events=[
            {
                "event_id": "cap_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 110_000.0,
                "event_type": "INJECTION",
            }
        ],
        manual_cash_adjustments=[
            {
                "event_id": "adj_1",
                "occurred_at": datetime(2026, 3, 3, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "delta_cash_krw": -10_000.0,
            }
        ],
        transfer_events=[
            {
                "event_id": "xfer_cash_1",
                "occurred_at": datetime(2026, 3, 4, tzinfo=timezone.utc),
                "transfer_type": "CASH_TRANSFER",
                "from_agent_id": "gemini",
                "to_agent_id": "gpt",
                "amount_krw": 25_000.0,
            }
        ],
    )
    store = _ActualBasisSleeveStore(_FakeSession(), origin_state=origin_state, ledger=ledger)

    trace = store.trace_agent_actual_capital_basis(agent_id="gpt")

    assert trace["seed_cash_krw"] == pytest.approx(1_000_000.0)
    assert trace["baseline_equity_krw"] == pytest.approx(1_125_000.0)
    assert trace["capital_flow_krw"] == pytest.approx(110_000.0)
    assert trace["manual_cash_adjustment_krw"] == pytest.approx(-10_000.0)
    assert trace["transfer_equity_krw"] == pytest.approx(25_000.0)


def test_fetch_actual_agent_nav_history_overlays_traced_actual_basis() -> None:
    origin_state = {
        "source": "legacy_agent_sleeve",
        "since": datetime(2026, 3, 1, tzinfo=timezone.utc),
        "cash_krw": 1_000_000.0,
        "positions_payload": [],
        "positions_error": None,
    }
    ledger = _LedgerStoreForCapitalReplay(
        _FakeSession(),
        capital_events=[
            {
                "event_id": "cap_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 110_000.0,
                "event_type": "INJECTION",
            }
        ],
    )
    store = _ActualBasisSleeveStore(
        _FakeSession(),
        origin_state=origin_state,
        ledger=ledger,
        nav_rows=[
            {
                "nav_date": date(2026, 3, 3),
                "agent_id": "gpt",
                "nav_krw": 1_090_000.0,
                "pnl_krw": 90_000.0,
                "pnl_ratio": 0.09,
            }
        ],
    )

    rows = store.fetch_actual_agent_nav_history(tenant_id="local", agent_ids=["gpt"], limit=10)

    assert rows[0]["baseline_equity_krw"] == pytest.approx(1_110_000.0)
    assert rows[0]["pnl_krw"] == pytest.approx(-20_000.0)
    assert rows[0]["pnl_ratio"] == pytest.approx(-20_000.0 / 1_110_000.0)


def test_fetch_agent_nav_history_prefers_official_rows() -> None:
    store = _NavSleeveStore.create()
    store.rows = [{"nav_date": date(2026, 3, 12), "agent_id": "gpt", "nav_krw": 1_100_000.0, "pnl_krw": 100_000.0, "pnl_ratio": 0.1}]

    rows = store.fetch_agent_nav_history(tenant_id="midnightnnn", agent_ids=["gpt"], limit=10)

    assert rows[0]["agent_id"] == "gpt"
    sql, params = store.executed[0]
    assert "official_nav_daily" in sql
    assert "agent_nav_daily" in sql
    assert params == {"tenant_id": "midnightnnn", "limit": 10, "agent_ids": ["gpt"]}


def test_upsert_agent_nav_daily_mirrors_into_official_nav_daily() -> None:
    store = _NavSleeveStore.create()

    store.upsert_agent_nav_daily(
        nav_date=date(2026, 3, 12),
        agent_id="gpt",
        nav_krw=1_250_000.0,
        baseline_equity_krw=1_000_000.0,
        cash_krw=200_000.0,
        market_value_krw=1_050_000.0,
        capital_flow_krw=150_000.0,
        fx_source="market_features_latest.fx_rate_used",
        valuation_source="agent_sleeve_snapshot",
        tenant_id="midnightnnn",
    )

    assert len(store.executed) == 4
    _, official_params = store.executed[-1]
    assert official_params is not None
    assert official_params["tenant_id"] == "midnightnnn"
    assert official_params["cash_krw"] == pytest.approx(200_000.0)
    assert official_params["market_value_krw"] == pytest.approx(1_050_000.0)
    assert official_params["capital_flow_krw"] == pytest.approx(150_000.0)
    assert official_params["fx_source"] == "market_features_latest.fx_rate_used"
    assert official_params["valuation_source"] == "agent_sleeve_snapshot"
