from __future__ import annotations

from arena.data.bigquery.execution_store import ExecutionStore
from arena.models import OrderIntent, RiskDecision, Side
from tests.helpers.bigquery import FakeBigQuerySession


def test_write_order_intent_persists_cycle_and_llm_call_ids() -> None:
    session = FakeBigQuerySession()
    store = ExecutionStore(session)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1,
        price_krw=100_000,
        rationale="test",
        cycle_id="cycle_1",
        llm_call_id="llm_execution_1",
    )

    store.write_order_intent(intent, RiskDecision(allowed=True, reason="ok"))

    sql, params = session.executed[0]
    assert "intent_id, cycle_id, llm_call_id" in sql
    assert params["cycle_id"] == "cycle_1"
    assert params["llm_call_id"] == "llm_execution_1"


def test_recent_trade_history_joins_execution_reports_to_order_intents() -> None:
    session = FakeBigQuerySession(fetch_result=[{"order_id": "order-1", "rationale": "why"}])
    store = ExecutionStore(session)

    rows = store.recent_trade_history(
        tenant_id="Tenant-A",
        ticker="aapl",
        agent_id="gpt",
        scope="agent_sleeve",
        days=30,
        limit=5,
        statuses=["SIMULATED"],
    )

    sql, params = session.fetched[0]
    assert rows == [{"order_id": "order-1", "rationale": "why"}]
    assert "FROM `proj.ds.execution_reports` er" in sql
    assert "LEFT JOIN `proj.ds.agent_order_intents` oi" in sql
    assert "er.tenant_id = @tenant_id" in sql
    assert "er.ticker = @ticker" in sql
    assert "er.agent_id = @agent_id" in sql
    assert "er.status IN UNNEST(@statuses)" in sql
    assert params["tenant_id"] == "Tenant-A"
    assert params["ticker"] == "AAPL"
    assert params["agent_id"] == "gpt"
    assert params["scope"] == "agent_sleeve"
    assert params["days"] == 30
    assert params["limit"] == 5
    assert params["statuses"] == ["SIMULATED"]
