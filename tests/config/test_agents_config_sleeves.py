from __future__ import annotations

import pytest

from arena.models import AccountSnapshot, Position
from tests.config.agents_config_helpers import _RetargetSleeveStore, _make_init_store


def test_ensure_agent_sleeves_uses_capital_per_agent(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)
    store = _make_init_store()

    store.ensure_agent_sleeves(
        agent_ids=["gpt", "gemini"],
        total_cash_krw=3_000_000,
        capital_per_agent={"gpt": 1_000_000, "gemini": 2_000_000},
    )

    assert len(store.session.client.payloads) == 2
    payloads_by_agent = {p["agent_id"]: p for p in store.session.client.payloads}
    assert float(payloads_by_agent["gpt"]["initial_cash_krw"]) == 1_000_000
    assert float(payloads_by_agent["gemini"]["initial_cash_krw"]) == 2_000_000


def test_ensure_agent_sleeves_without_capital_per_agent_uses_equal_split(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)
    store = _make_init_store()

    store.ensure_agent_sleeves(
        agent_ids=["gpt", "gemini"],
        total_cash_krw=2_000_000,
    )

    assert len(store.session.client.payloads) == 2
    for p in store.session.client.payloads:
        assert float(p["initial_cash_krw"]) == 1_000_000


def test_retarget_uses_target_capitals_per_agent() -> None:
    store = _RetargetSleeveStore(
        {
            "gpt": AccountSnapshot(
                cash_krw=100_000,
                total_equity_krw=400_000,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000,
                        market_price_krw=150_000,
                    )
                },
            ),
            "gemini": AccountSnapshot(
                cash_krw=200_000,
                total_equity_krw=500_000,
                positions={
                    "MSFT": Position(
                        ticker="MSFT",
                        exchange_code="NASD",
                        instrument_id="NASD:MSFT",
                        quantity=1.0,
                        avg_price_krw=300_000,
                        market_price_krw=300_000,
                    )
                },
            ),
        }
    )

    out = store.retarget_agent_sleeves_preserve_positions(
        agent_ids=["gpt", "gemini"],
        target_sleeve_capital_krw=500_000,
        target_capitals={"gpt": 600_000, "gemini": 800_000},
    )

    payloads_by_agent = {p["agent_id"]: p for p in store.session.client.payloads}
    # gpt: positions_value=300_000, target=600_000, cash=300_000
    assert float(payloads_by_agent["gpt"]["initial_cash_krw"]) == pytest.approx(300_000)
    assert out["gpt"]["over_target"] is False
    # gemini: positions_value=300_000, target=800_000, cash=500_000
    assert float(payloads_by_agent["gemini"]["initial_cash_krw"]) == pytest.approx(500_000)
    assert out["gemini"]["over_target"] is False


def test_retarget_without_target_capitals_uses_uniform_target() -> None:
    store = _RetargetSleeveStore(
        {
            "gpt": AccountSnapshot(
                cash_krw=100_000,
                total_equity_krw=400_000,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000,
                        market_price_krw=150_000,
                    )
                },
            ),
        }
    )

    out = store.retarget_agent_sleeves_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000,
    )

    first = store.session.client.payloads[0]
    # positions_value=300_000, target=500_000, cash=200_000
    assert float(first["initial_cash_krw"]) == pytest.approx(200_000)
    assert out["gpt"]["over_target"] is False
