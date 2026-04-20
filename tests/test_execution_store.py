from __future__ import annotations

from arena.data.bigquery.execution_store import ExecutionStore
from arena.models import OrderIntent, RiskDecision, Side


class _FakeSession:
    dataset_fqn = "proj.ds"

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def resolve_tenant_id(self, tenant_id=None) -> str:
        return str(tenant_id or "tenant-a")

    def execute(self, sql: str, params: dict) -> None:
        self.executed.append((sql, dict(params)))


def test_write_order_intent_persists_cycle_and_llm_call_ids() -> None:
    session = _FakeSession()
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
