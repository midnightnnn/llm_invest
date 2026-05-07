from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, RiskDecision, Side
from tests.data.local_repository_helpers import repo


def test_research_briefings_round_trip_with_filters(repo):
    old = datetime(2026, 4, 1, tzinfo=timezone.utc)
    new = datetime(2026, 4, 2, tzinfo=timezone.utc)
    repo.insert_research_briefings(
        [
            {
                "briefing_id": "brf_global",
                "created_at": old,
                "ticker": "GLOBAL",
                "category": "global_market",
                "headline": "Global",
                "summary": "Macro update",
                "sources": "[]",
                "trading_mode": "paper",
            },
            {
                "briefing_id": "brf_aapl",
                "created_at": new,
                "ticker": "AAPL",
                "category": "held",
                "headline": "Apple",
                "summary": "Ticker update",
                "sources": "[]",
                "trading_mode": "paper",
            },
            {
                "briefing_id": "brf_live",
                "created_at": new,
                "ticker": "MSFT",
                "category": "held",
                "headline": "Live",
                "summary": "Wrong mode",
                "sources": "[]",
                "trading_mode": "live",
            },
        ]
    )

    all_rows = repo.get_research_briefings(limit=10)
    assert [row["briefing_id"] for row in all_rows] == ["brf_aapl", "brf_global"]

    ticker_rows = repo.get_research_briefings(tickers=["aapl"], limit=10)
    assert [row["briefing_id"] for row in ticker_rows] == ["brf_aapl"]

    category_rows = repo.get_research_briefings(categories=["global_market"], limit=10)
    assert [row["briefing_id"] for row in category_rows] == ["brf_global"]

    live_rows = repo.get_research_briefings(trading_mode="live", limit=10)
    assert [row["briefing_id"] for row in live_rows] == ["brf_live"]


def test_append_runtime_audit_log_uses_bigquery_signature(repo):
    repo.append_runtime_audit_log(
        action="agent_cycle",
        status="warning",
        user_email="User@Example.COM",
        tenant_id="Tenant-A",
        detail={"cycle_id": "cycle_1"},
    )

    rows = repo.recent_runtime_audit_logs(limit=5)
    assert len(rows) == 1
    assert rows[0]["user_email"] == "user@example.com"
    assert rows[0]["tenant_id"] == "tenant-a"
    assert rows[0]["action"] == "agent_cycle"
    assert rows[0]["status"] == "warning"
    assert "cycle_1" in rows[0]["detail_json"]


def test_execution_write_and_daily_risk_readers(repo):
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2,
        price_krw=100.0,
        rationale="test",
        created_at=datetime(2026, 4, 28, 1, 0, tzinfo=timezone.utc),
    )
    decision = RiskDecision(allowed=True, reason="ok")
    report = ExecutionReport(
        status=ExecutionStatus.SIMULATED,
        order_id="order-1",
        filled_qty=2,
        avg_price_krw=110.0,
        message="paper fill",
        created_at=datetime(2026, 4, 28, 1, 1, tzinfo=timezone.utc),
    )

    repo.write_order_intent(intent, decision)
    repo.write_execution_report(intent, report)

    assert repo.recent_intent_count(datetime(2026, 4, 28, tzinfo=timezone.utc).date(), agent_id="gpt") == 1
    assert repo.recent_turnover_krw(datetime(2026, 4, 28, tzinfo=timezone.utc).date(), agent_id="gpt") == pytest.approx(220.0)
    assert repo.last_trade_time("AAPL", agent_id="gpt") == report.created_at.replace(tzinfo=None)


def test_recent_trade_history_joins_execution_with_intent_metadata(repo):
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.SELL,
        quantity=2,
        price_krw=100.0,
        rationale="사용자와 투자챗봇이 일부 익절을 판단함",
        strategy_refs=["scope:agent_sleeve", "judgment:user+investment_chat"],
        created_at=datetime(2026, 4, 28, 1, 0, tzinfo=timezone.utc),
    )
    decision = RiskDecision(allowed=True, reason="risk ok", policy_hits=["chat_confirmation"])
    report = ExecutionReport(
        status=ExecutionStatus.SIMULATED,
        order_id="order-history-1",
        filled_qty=2,
        avg_price_krw=110.0,
        message="paper fill",
        created_at=datetime(2026, 4, 28, 1, 1, tzinfo=timezone.utc),
    )
    repo.write_order_intent(intent, decision)
    repo.write_execution_report(intent, report)

    repo.set_tenant_id("tenant-b")
    other_intent = intent.model_copy(update={"intent_id": "tenant-b-intent", "rationale": "other tenant"})
    other_report = report.model_copy(update={"order_id": "tenant-b-order"})
    repo.write_order_intent(other_intent, decision)
    repo.write_execution_report(other_intent, other_report)

    rows = repo.recent_trade_history(
        tenant_id="tenant-a",
        ticker="aapl",
        agent_id="gpt",
        scope="agent_sleeve",
        days=3650,
        limit=10,
        statuses=["SIMULATED"],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["order_id"] == "order-history-1"
    assert row["ticker"] == "AAPL"
    assert row["rationale"] == "사용자와 투자챗봇이 일부 익절을 판단함"
    assert row["risk_reason"] == "risk ok"
    assert row["policy_hits"] == ["chat_confirmation"]
    assert row["strategy_refs"] == ["scope:agent_sleeve", "judgment:user+investment_chat"]


def test_sleeve_snapshot_replays_simulated_execution(repo):
    repo.ensure_agent_state_checkpoints(
        agent_ids=["gpt"],
        total_cash_krw=1_000.0,
        checkpoint_at=datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc),
    )
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2,
        price_krw=100.0,
        rationale="test",
        created_at=datetime(2026, 4, 28, 1, 0, tzinfo=timezone.utc),
    )
    report = ExecutionReport(
        status=ExecutionStatus.SIMULATED,
        order_id="order-2",
        filled_qty=2,
        avg_price_krw=100.0,
        message="paper fill",
        created_at=datetime(2026, 4, 28, 1, 1, tzinfo=timezone.utc),
    )
    repo.write_execution_report(intent, report)

    snapshot, baseline, meta = repo.build_agent_sleeve_snapshot(agent_id="gpt")

    assert baseline == pytest.approx(1_000.0)
    assert snapshot.cash_krw == pytest.approx(800.0)
    assert snapshot.positions["AAPL"].quantity == pytest.approx(2.0)
    assert snapshot.total_equity_krw == pytest.approx(1_000.0)
    assert meta["valuation_source"] == "local_sleeve_replay"


def test_local_sleeve_snapshot_replays_capital_events(repo):
    repo.ensure_agent_state_checkpoints(
        agent_ids=["gpt"],
        total_cash_krw=1_000.0,
        checkpoint_at=datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc),
    )
    written = repo.append_capital_events(
        [
            {
                "event_id": "cap-local-1",
                "occurred_at": datetime(2026, 4, 29, 0, 0, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 250.0,
                "event_type": "INJECTION",
                "reason": "test",
                "created_by": "tester",
            }
        ],
        tenant_id="tenant-a",
    )

    snapshot, baseline, meta = repo.build_agent_sleeve_snapshot(agent_id="gpt")

    assert written == 1
    assert snapshot.cash_krw == pytest.approx(1_250.0)
    assert snapshot.total_equity_krw == pytest.approx(1_250.0)
    assert baseline == pytest.approx(1_250.0)
    assert meta["capital_flow_krw"] == pytest.approx(250.0)
    assert meta["capital_event_count"] == 1


def test_local_retarget_agent_capitals_preserves_positions_and_updates_snapshot(repo):
    repo.ensure_agent_state_checkpoints(
        agent_ids=["gpt"],
        total_cash_krw=1_000.0,
        checkpoint_at=datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc),
    )
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2,
        price_krw=100.0,
        rationale="test",
        created_at=datetime(2026, 4, 28, 1, 0, tzinfo=timezone.utc),
    )
    report = ExecutionReport(
        status=ExecutionStatus.SIMULATED,
        order_id="order-retarget-1",
        filled_qty=2,
        avg_price_krw=100.0,
        message="paper fill",
        created_at=datetime(2026, 4, 28, 1, 1, tzinfo=timezone.utc),
    )
    repo.write_execution_report(intent, report)

    out = repo.retarget_agent_capitals_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=1_500.0,
        occurred_at=datetime(2026, 4, 29, 0, 0, tzinfo=timezone.utc),
        created_by="tester",
    )
    snapshot, baseline, meta = repo.build_agent_sleeve_snapshot(agent_id="gpt")
    events = repo.capital_events_since(
        agent_id="gpt",
        since=datetime(2026, 1, 1, tzinfo=timezone.utc),
        tenant_id="tenant-a",
    )

    assert out["gpt"]["capital_flow_krw"] == pytest.approx(500.0)
    assert out["gpt"]["event_type"] == "INJECTION"
    assert len(events) == 1
    assert events[0]["event_type"] == "INJECTION"
    assert events[0]["amount_krw"] == pytest.approx(500.0)
    assert snapshot.cash_krw == pytest.approx(1_300.0)
    assert snapshot.positions["AAPL"].quantity == pytest.approx(2.0)
    assert snapshot.total_equity_krw == pytest.approx(1_500.0)
    assert baseline == pytest.approx(1_500.0)
    assert meta["seed_source"] == "capital_events.retarget"
    assert meta["capital_event_count"] == 0
