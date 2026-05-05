from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from arena.config import Settings
from arena.context import ContextBuilder
from arena.memory.policy import normalize_memory_policy
from arena.models import AccountSnapshot, Position, utc_now

from tests.context.helpers import (
    FakeRepo,
    FakeMemory,
    FakeBoard,
    FakeVectorStore,
    _settings,
)

def test_context_builder_live_mode_raises_when_risk_metrics_unavailable() -> None:
    class FailingRiskRepo(FakeRepo):
        def recent_intent_count(self, day, agent_id=None, include_simulated=True, trading_mode=None):
            _ = (day, agent_id, include_simulated, trading_mode)
            raise RuntimeError("risk metrics unavailable")

    repo = FailingRiskRepo()
    settings = _settings()
    settings.trading_mode = "live"
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_100_000, positions={})

    with pytest.raises(RuntimeError, match="recent_intent_count"):
        builder.build(agent_id="gpt", snapshot=snapshot)


def test_context_builder_no_cap_gap_block_when_nav_exceeds_target() -> None:
    """NAV > target no longer blocks buying — only cash matters."""
    repo = FakeRepo()
    settings = _settings()
    settings.sleeve_capital_krw = 1_000_000
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={},
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["sleeve_state"]["over_target"] is False
    assert context["sleeve_state"]["buy_blocked"] is False
    assert context["order_budget"]["max_buy_notional_by_sleeve_krw"] == 880_000
    assert context["order_budget"]["max_buy_notional_krw"] == 350_000


def test_context_builder_uses_per_agent_capital_for_sleeve_target() -> None:
    repo = FakeRepo()
    settings = _settings()
    settings.sleeve_capital_krw = 1_000_000
    settings.agent_capitals = {"gpt": 500_000, "gemini": 1_500_000}
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(cash_krw=400_000, total_equity_krw=600_000, positions={})

    # gpt target=500_000, equity=600_000 → no longer blocked
    gpt_ctx = builder.build(agent_id="gpt", snapshot=snapshot)
    assert gpt_ctx["sleeve_state"]["target_sleeve_krw"] == 500_000
    assert gpt_ctx["sleeve_state"]["over_target"] is False

    # gemini target=1_500_000, equity=600_000
    gemini_ctx = builder.build(agent_id="gemini", snapshot=snapshot)
    assert gemini_ctx["sleeve_state"]["target_sleeve_krw"] == 1_500_000
    assert gemini_ctx["sleeve_state"]["over_target"] is False

    # unknown agent falls back to sleeve_capital_krw
    unknown_ctx = builder.build(agent_id="unknown_agent", snapshot=snapshot)
    assert unknown_ctx["sleeve_state"]["target_sleeve_krw"] == 1_000_000


def test_context_builder_caps_buy_budget_by_cash_buffer() -> None:
    repo = FakeRepo()
    settings = _settings()
    settings.sleeve_capital_krw = 1_000_000
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=100_000,
        total_equity_krw=950_000,
        positions={},
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["sleeve_state"]["sleeve_remaining_krw"] == 5_000
    assert context["order_budget"]["max_buy_notional_by_sleeve_krw"] == 5_000
    assert context["order_budget"]["max_buy_notional_krw"] == 5_000
