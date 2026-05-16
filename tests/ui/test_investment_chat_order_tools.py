from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from arena.config import load_settings
from arena.models import AccountSnapshot, ExecutionStatus, Position
from tests.ui.investment_chat_helpers import (
    _ChatOrderRepo,
    _FakeExecutionMemory,
    _FakeToolContext,
    _build_raw_chat_tools,
)


def test_chat_order_draft_does_not_block_rationale_by_phrase(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)

    result = tools["validate_order_draft"](
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="매수 근거",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
    )

    assert result["status"] == "ok"
    assert result["intent"]["rationale"] == "매수 근거"


def test_chat_order_tools_require_confirmation_and_are_idempotent(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_USER_EMAIL
    from arena.agents.investment_chat.drafts import draft_key

    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo, include_internal_bridge=True)

    user_token = REQUEST_USER_EMAIL.set("trader@example.com")
    try:
        draft = tools["validate_order_draft"](
            ticker="AAPL",
            side="BUY",
            quantity=1,
            price_krw=100_000,
            rationale="test buy",
            exchange_code="NASD",
            instrument_id="NASD:AAPL",
        )

        token = str(draft.get("approval_token") or "")
        assert draft["submission_status"] == "not_submitted"
        assert token
        assert repo.get_config("local", draft_key(token))

        blocked = tools["submit_approved_order"](approval_token=token, confirmation_text="승인")

        assert blocked["status"] == "blocked"
        assert "CONFIRM" in blocked["required_confirmation"]
        assert repo.execution_reports == []

        submitted = tools["submit_approved_order"](
            approval_token=token,
            confirmation_text=f"CONFIRM {token}",
        )
        repeated = tools["submit_approved_order"](
            approval_token=token,
            confirmation_text=f"CONFIRM {token}",
        )
    finally:
        REQUEST_USER_EMAIL.reset(user_token)

    assert submitted["status"] == "submitted"
    assert submitted["execution_report"]["status"] == ExecutionStatus.SIMULATED.value
    assert repeated["status"] == "already_submitted"
    assert len(repo.execution_reports) == 1
    assert any(row.get("action") == "chat_order_submit" for row in repo.audit_logs)
    assert {row.get("user_email") for row in repo.audit_logs} == {"trader@example.com"}

    second_draft = tools["validate_order_draft"](
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test buy",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
    )

    assert second_draft["approval_token"] != token
    assert tools["submit_approved_order"](token, f"CONFIRM {token}")["status"] == "already_submitted"


def test_chat_order_tool_uses_adk_confirmation_before_execution(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)
    submit_with_confirmation = tools["submit_order_with_confirmation"]

    first_context = _FakeToolContext()
    waiting = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test buy through ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=first_context,
    )

    assert waiting["status"] == "waiting_for_confirmation"
    assert waiting["submission_status"] == "not_submitted"
    assert first_context.actions.skip_summarization is False
    assert first_context.confirmation_request is not None
    payload = first_context.confirmation_request["payload"]
    assert isinstance(payload, dict)
    assert payload["ticker"] == "AAPL"
    assert "approval_token" not in payload
    assert "ADK Web 확인창" in str(first_context.confirmation_request["hint"])
    assert "Confirmed 체크박스" in str(first_context.confirmation_request["hint"])
    assert repo.execution_reports == []

    confirmed_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(confirmed=True, payload={}),
    )
    submitted = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test buy through ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=confirmed_context,
    )

    assert submitted["status"] == "submitted"
    assert submitted["execution_report"]["status"] == ExecutionStatus.SIMULATED.value
    assert len(repo.execution_reports) == 1


def test_chat_order_tool_rejects_adk_confirmation_without_execution(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)
    submit_with_confirmation = tools["submit_order_with_confirmation"]

    first_context = _FakeToolContext()
    waiting = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test rejected ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=first_context,
    )
    assert waiting["status"] == "waiting_for_confirmation"

    rejected_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(confirmed=False, payload={}),
    )
    rejected = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test rejected ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=rejected_context,
    )

    assert rejected["status"] == "rejected"
    assert repo.execution_reports == []


def test_chat_order_batch_tool_uses_one_adk_confirmation_before_execution(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)
    submit_batch = tools["submit_order_batch_with_confirmation"]
    orders = [
        {
            "ticker": "AAPL",
            "side": "BUY",
            "quantity": 1,
            "price_krw": 100_000,
            "rationale": "AAPL batch buy thesis with account context",
            "exchange_code": "NASD",
            "instrument_id": "NASD:AAPL",
        },
        {
            "ticker": "MSFT",
            "side": "BUY",
            "quantity": 1,
            "price_krw": 100_000,
            "rationale": "MSFT batch buy thesis with account context",
            "exchange_code": "NASD",
            "instrument_id": "NASD:MSFT",
        },
    ]

    first_context = _FakeToolContext(function_call_id="fc-order-batch")
    waiting = submit_batch(orders=orders, tool_context=first_context)

    assert waiting["status"] == "waiting_for_confirmation"
    assert waiting["order_count"] == 2
    assert waiting["submittable_count"] == 2
    assert waiting["submission_status"] == "not_submitted"
    assert first_context.actions.skip_summarization is False
    assert first_context.confirmation_request is not None
    payload = first_context.confirmation_request["payload"]
    assert isinstance(payload, dict)
    assert payload["action"] == "submit_order_batch"
    assert payload["order_count"] == 2
    assert payload["submittable_count"] == 2
    assert repo.execution_reports == []

    confirmed_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(confirmed=True, payload={}),
    )
    submitted = submit_batch(orders=orders, tool_context=confirmed_context)

    assert submitted["status"] == "submitted"
    assert submitted["order_count"] == 2
    assert submitted["submitted_count"] == 2
    assert len(repo.execution_reports) == 2


def test_chat_order_batch_tool_rejects_one_confirmation_without_execution(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)
    submit_batch = tools["submit_order_batch_with_confirmation"]
    orders = [
        {"ticker": "AAPL", "side": "BUY", "quantity": 1, "price_krw": 100_000, "rationale": "batch buy AAPL"},
        {"ticker": "MSFT", "side": "BUY", "quantity": 1, "price_krw": 100_000, "rationale": "batch buy MSFT"},
    ]

    first_context = _FakeToolContext(function_call_id="fc-order-batch-reject")
    waiting = submit_batch(orders=orders, tool_context=first_context)
    assert waiting["status"] == "waiting_for_confirmation"

    rejected_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(confirmed=False, payload={}),
    )
    rejected = submit_batch(orders=orders, tool_context=rejected_context)

    assert rejected["status"] == "rejected"
    assert rejected["order_count"] == 2
    assert repo.execution_reports == []


def test_chat_order_tool_explains_unchecked_adk_confirmation(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)
    submit_with_confirmation = tools["submit_order_with_confirmation"]

    first_context = _FakeToolContext()
    waiting = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test unchecked ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=first_context,
    )
    assert waiting["status"] == "waiting_for_confirmation"

    unchecked_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(
            confirmed=False,
            payload={
                "ticker": "AAPL",
                "side": "BUY",
                "quantity": 1,
                "price_krw": 100_000,
            },
        ),
    )
    rejected = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test unchecked ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=unchecked_context,
    )

    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "confirmed_checkbox_unchecked"
    assert "Confirmed 체크박스" in rejected["message"]
    assert repo.execution_reports == []


def test_get_trade_history_tool_reads_tenant_scoped_execution_history(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    repo.trade_history_rows = [
        {
            "order_id": "order-1",
            "intent_id": "intent-1",
            "created_at": datetime(2026, 4, 28, 1, 1, tzinfo=timezone.utc),
            "trading_mode": "paper",
            "agent_id": "gpt",
            "ticker": "AAPL",
            "exchange_code": "NASD",
            "instrument_id": "NASD:AAPL",
            "side": "SELL",
            "requested_qty": 2.0,
            "filled_qty": 1.0,
            "avg_price_krw": 150_000.0,
            "avg_price_native": 100.0,
            "quote_currency": "USD",
            "fx_rate": 1500.0,
            "status": "SIMULATED",
            "message": "paper fill",
            "rationale": "사용자와 투자챗봇이 과열 구간 일부 익절을 판단함",
            "risk_reason": "ok",
            "policy_hits": ["chat_confirmation"],
            "strategy_refs": ["scope:agent_sleeve", "judgment:user+investment_chat"],
        }
    ]
    tools = _build_raw_chat_tools(monkeypatch, repo)

    result = tools["get_trade_history"](
        ticker="aapl",
        agent_id="gpt",
        scope="agent_sleeve",
        days=30,
        limit=5,
    )

    assert result["status"] == "ok"
    assert result["tenant_id"] == "local"
    assert result["count"] == 1
    assert repo.trade_history_calls == [
        {
            "tenant_id": "local",
            "ticker": "AAPL",
            "agent_id": "gpt",
            "scope": "agent_sleeve",
            "days": 30,
            "limit": 5,
            "statuses": ["FILLED", "SIMULATED", "SUBMITTED"],
        }
    ]
    trade = result["trades"][0]
    assert trade["ticker"] == "AAPL"
    assert trade["scope"] == "agent_sleeve"
    assert trade["judgment_source"] == "user+investment_chat"
    assert trade["notional_krw"] == 150_000.0
    assert trade["rationale"] == "사용자와 투자챗봇이 과열 구간 일부 익절을 판단함"


def test_chat_sleeve_order_uses_sleeve_snapshot_and_syncs_target_agent_memory(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_USER_EMAIL

    repo = _ChatOrderRepo(
        account_snapshot=AccountSnapshot(cash_krw=9_000_000.0, total_equity_krw=10_000_000.0, positions={}),
        sleeve_snapshot=AccountSnapshot(
            cash_krw=1_000_000.0,
            total_equity_krw=1_260_000.0,
            usd_krw_rate=1400.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL",
                    quantity=2,
                    avg_price_krw=120_000,
                    market_price_krw=130_000,
                )
            },
        ),
    )
    tools = _build_raw_chat_tools(monkeypatch, repo, include_internal_bridge=True)

    user_token = REQUEST_USER_EMAIL.set("trader@example.com")
    try:
        draft = tools["validate_order_draft"](
            ticker="AAPL",
            side="SELL",
            quantity=1,
            price_krw=130_000,
            rationale="사용자가 AAPL 비중을 낮추고 현금 여력을 확보하기로 판단함",
            scope="agent_sleeve",
            agent_id="gpt",
            exchange_code="NASD",
            instrument_id="NASD:AAPL",
        )
        token = str(draft["approval_token"])
        submitted = tools["submit_approved_order"](
            approval_token=token,
            confirmation_text=f"CONFIRM {token}",
        )
    finally:
        REQUEST_USER_EMAIL.reset(user_token)

    assert draft["status"] == "ok"
    assert draft["risk"]["allowed"] is True
    assert draft["scope"] == "agent_sleeve"
    assert draft["intent"]["agent_id"] == "gpt"
    assert submitted["status"] == "submitted"
    assert len(repo.sleeve_calls) >= 2
    assert {str(call["agent_id"]) for call in repo.sleeve_calls} == {"gpt"}
    assert len(repo.order_intents) == 1
    intent = repo.order_intents[0]["intent"]
    assert intent.agent_id == "gpt"
    assert "scope:agent_sleeve" in intent.strategy_refs
    assert "judgment:user+investment_chat" in intent.strategy_refs
    assert "approved_by:trader@example.com" in intent.strategy_refs
    assert len(_FakeExecutionMemory.instances) == 1
    memory = _FakeExecutionMemory.instances[0]
    assert [row["intent"].agent_id for row in memory.executions] == ["gpt"]
    assert [row["intent"].agent_id for row in memory.theses] == ["gpt"]
    assert [row["agent_id"] for row in memory.reflections] == ["gpt"]
    reflection = memory.reflections[0]
    assert "사용자+투자챗봇 판단" in reflection["summary"]
    assert "AAPL 비중을 낮추고" in reflection["summary"]
    assert reflection["payload"]["source"] == "investment_chat_order_decision"
    assert reflection["payload"]["judgment_source"] == "user+investment_chat"
    assert reflection["payload"]["scope"] == "agent_sleeve"
    assert reflection["payload"]["approved_by"] == "trader@example.com"


def test_chat_order_submit_blocks_live_mode_without_explicit_permission(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    settings = load_settings()
    settings.trading_mode = "live"
    settings.allow_live_trading = False
    settings.kis_target_market = "us"

    tools = _build_raw_chat_tools(monkeypatch, repo, settings=settings, include_internal_bridge=True)

    draft = tools["validate_order_draft"](
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test buy",
    )
    token = str(draft["approval_token"])
    submitted = tools["submit_approved_order"](
        approval_token=token,
        confirmation_text=f"CONFIRM {token}",
    )

    assert submitted["status"] == "blocked"
    assert "live trading" in submitted["error"]
    assert repo.execution_reports == []
