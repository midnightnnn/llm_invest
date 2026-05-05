from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tests.data.strict_path_helpers import (
    _FakeSession,
    _LedgerStoreForCapitalReplay,
    _SleeveStoreForCapitalReplay,
    _make_capital_replay_store,
)


def test_build_agent_sleeve_snapshot_replays_capital_events_from_checkpoint_seed() -> None:
    store = _make_capital_replay_store(
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 1_000_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[
            {
                "event_id": "cap_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 250_000.0,
                "event_type": "INJECTION",
            }
        ],
    )

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(1_250_000.0)
    assert snapshot.total_equity_krw == pytest.approx(1_250_000.0)
    assert baseline == pytest.approx(1_250_000.0)
    assert meta["seed_source"] == "checkpoint_test"
    assert meta["capital_event_count"] == 1
    assert meta["capital_flow_krw"] == pytest.approx(250_000.0)


def test_build_agent_sleeve_snapshot_replays_manual_cash_adjustments() -> None:
    store = _make_capital_replay_store(
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 1_000_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
        manual_cash_adjustments=[
            {
                "event_id": "adj_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "delta_cash_krw": -125_000.0,
            }
        ],
    )

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(875_000.0)
    assert baseline == pytest.approx(875_000.0)
    assert meta["manual_cash_adjustment_count"] == 1
    assert meta["manual_cash_adjustment_krw"] == pytest.approx(-125_000.0)


def test_build_agent_sleeve_snapshot_replays_manual_position_adjustments_in_order() -> None:
    store = _make_capital_replay_store(
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 1_000_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
        fills=[
            {
                "created_at": "2026-03-02T00:00:00+00:00",
                "ticker": "001510",
                "exchange_code": "KRX",
                "instrument_id": "KRX:001510",
                "side": "BUY",
                "filled_qty": 56.0,
                "avg_price_krw": 2_077.0,
                "status": "FILLED",
            },
            {
                "created_at": "2026-03-04T00:00:00+00:00",
                "ticker": "001510",
                "exchange_code": "KRX",
                "instrument_id": "KRX:001510",
                "side": "SELL",
                "filled_qty": 20.0,
                "avg_price_krw": 5_370.0,
                "status": "FILLED",
            },
        ],
        manual_position_adjustments=[
            {
                "event_id": "adj_1",
                "occurred_at": datetime(2026, 3, 3, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "ticker": "001510",
                "delta_quantity": -28.0,
                "adjustment_type": "corporate_action",
            }
        ],
    )

    snapshot, _baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.positions["001510"].quantity == pytest.approx(8.0)
    assert snapshot.positions["001510"].avg_price_krw == pytest.approx(4_154.0)
    assert meta["manual_position_adjustment_count"] == 1
    assert meta["manual_position_adjustment_quantity"] == pytest.approx(-28.0)


def test_build_agent_sleeve_snapshot_replays_agent_transfer_events() -> None:
    store = _make_capital_replay_store(
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

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(50_000.0)
    assert snapshot.total_equity_krw == pytest.approx(100_000.0)
    assert snapshot.positions["AAPL"].quantity == pytest.approx(1.0)
    assert snapshot.positions["AAPL"].avg_price_krw == pytest.approx(50_000.0)
    assert baseline == pytest.approx(100_000.0)
    assert meta["transfer_event_count"] == 1
    assert meta["transfer_cash_krw"] == pytest.approx(-50_000.0)
    assert meta["transfer_equity_krw"] == pytest.approx(0.0)


def test_build_agent_sleeve_snapshot_treats_cash_transfer_as_capital_basis() -> None:
    store = _make_capital_replay_store(
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
                "event_id": "xfer_cash_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "transfer_type": "CASH_TRANSFER",
                "from_agent_id": "gemini",
                "to_agent_id": "gpt",
                "amount_krw": 250_000.0,
            }
        ],
    )

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(350_000.0)
    assert snapshot.total_equity_krw == pytest.approx(350_000.0)
    assert baseline == pytest.approx(350_000.0)
    assert meta["transfer_event_count"] == 1
    assert meta["transfer_cash_krw"] == pytest.approx(250_000.0)
    assert meta["transfer_equity_krw"] == pytest.approx(250_000.0)


def test_build_agent_sleeve_snapshot_replays_execution_fills_from_checkpoint_seed() -> None:
    store = _make_capital_replay_store(
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 100_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
        fills=[
            {
                "created_at": "2026-03-02T00:00:00+00:00",
                "ticker": "AAPL",
                "exchange_code": "NAS",
                "instrument_id": "AAPL",
                "side": "BUY",
                "filled_qty": 1.0,
                "avg_price_krw": 50_000.0,
                "avg_price_native": 34.0,
                "quote_currency": "USD",
                "fx_rate": 1470.0,
                "status": "FILLED",
            }
        ],
    )

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(50_000.0)
    assert snapshot.total_equity_krw == pytest.approx(100_000.0)
    assert snapshot.positions["AAPL"].quantity == pytest.approx(1.0)
    assert snapshot.positions["AAPL"].avg_price_krw == pytest.approx(50_000.0)
    assert baseline == pytest.approx(100_000.0)
    assert meta["trade_count_total"] == 1


def test_build_agent_sleeve_snapshot_prefers_latest_instrument_metadata_for_live_positions() -> None:
    class _MarketWithInstrumentMap:
        def latest_close_prices_with_currency(self, *, tickers, sources=None, as_of_date=None):
            _ = sources
            assert tickers == ["CSX"]
            return {
                "CSX": {
                    "close_price_krw": 50_000.0,
                    "close_price_native": 39.3,
                    "quote_currency": "USD",
                    "fx_rate_used": 1272.0,
                }
            }

        def latest_instrument_map(self, tickers):
            assert tickers == ["CSX"]
            return {
                "CSX": {
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:CSX",
                }
            }

    sleeve_session = _FakeSession(responses=[
        [
            {
                "created_at": "2026-03-02T00:00:00+00:00",
                "ticker": "CSX",
                "exchange_code": "NYSE",
                "instrument_id": "NYSE:CSX",
                "side": "BUY",
                "filled_qty": 1.0,
                "avg_price_krw": 40_000.0,
                "avg_price_native": 31.0,
                "quote_currency": "USD",
                "fx_rate": 1290.0,
                "status": "FILLED",
            }
        ],
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
    )
    market = _MarketWithInstrumentMap()
    store = _SleeveStoreForCapitalReplay(sleeve_session, ledger=ledger, market=market)

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.positions["CSX"].exchange_code == "NASD"
    assert snapshot.positions["CSX"].instrument_id == "NASD:CSX"
    assert snapshot.positions["CSX"].market_price_native == pytest.approx(39.3)
    assert baseline == pytest.approx(100_000.0)
    assert meta["trade_count_total"] == 1
