from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from arena.config import AgentConfig, load_settings
from arena.models import AccountSnapshot, ExecutionReport, ExecutionStatus, Position
from arena.tools.registry import ToolEntry, ToolRegistry
from arena.ui.layout import tailwind_layout
from arena.ui.server import _build_app
from tests.direct_route_client import DirectRouteClient
from tests.ui.helpers import _DummyRepo

from tests.ui.investment_chat_helpers import (
    _ChatOrderRepo,
    _FakeExecutionMemory,
    _FakeToolContext,
    _build_fake_chat_agent,
    _build_raw_chat_tools,
)

def test_investment_chat_account_tools_expose_available_agent_ids() -> None:
    from arena.agents.investment_chat.account_tools import build_account_tool_entries

    settings = load_settings()
    repo = _ChatOrderRepo()
    tools = {
        entry.name: entry.callable
        for entry in build_account_tool_entries(repo=repo, settings=settings, tenant_id="local")
    }

    payload = tools["get_account_snapshot"]()

    assert payload["available_agent_ids"] == ["gemini", "gpt", "claude"]


def test_investment_chat_sleeve_tool_normalizes_model_aliases_to_agent_ids() -> None:
    from arena.agents.investment_chat.account_tools import build_account_tool_entries

    settings = load_settings()
    repo = _ChatOrderRepo()
    tools = {
        entry.name: entry.callable
        for entry in build_account_tool_entries(repo=repo, settings=settings, tenant_id="local")
    }

    for alias, expected_agent_id in [
        ("gpt4o", "gpt"),
        ("gemini_2_0_flash_exp", "gemini"),
        ("claude_3_7_sonnet", "claude"),
    ]:
        payload = tools["get_agent_sleeve_snapshot"](agent_id=alias)

        assert payload["agent_id"] == expected_agent_id
        assert payload["requested_agent_id"] == alias
        assert payload["available_agent_ids"] == ["gemini", "gpt", "claude"]
        assert repo.sleeve_calls[-1]["agent_id"] == expected_agent_id


def test_investment_chat_sleeve_tool_rejects_unknown_agent_id() -> None:
    from arena.agents.investment_chat.account_tools import build_account_tool_entries

    settings = load_settings()
    repo = _ChatOrderRepo()
    tools = {
        entry.name: entry.callable
        for entry in build_account_tool_entries(repo=repo, settings=settings, tenant_id="local")
    }

    payload = tools["get_agent_sleeve_snapshot"](agent_id="quant_bot")

    assert payload["status"] == "blocked"
    assert payload["requested_agent_id"] == "quant_bot"
    assert payload["available_agent_ids"] == ["gemini", "gpt", "claude"]
    assert repo.sleeve_calls == []


def test_investment_chat_wrapped_adk_confirmation_tool_builds_declaration(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "submit_order_with_confirmation")

    declaration = FunctionTool(tool)._get_declaration()

    assert declaration is not None
    assert declaration.name == "submit_order_with_confirmation"
    assert "tool_context" not in json.dumps(declaration.model_dump(), default=str)


def test_chat_order_tool_schema_describes_ontology_friendly_rationale(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "submit_order_with_confirmation")

    declaration = FunctionTool(tool)._get_declaration()
    dumped = json.dumps(declaration.model_dump(), ensure_ascii=False, default=str)

    assert "ontology-friendly investment memo" in dumped
    assert "explicit ticker names" in dumped


def test_chat_order_tool_schema_preserves_required_fields_and_enums(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "submit_order_with_confirmation")

    declaration = FunctionTool(tool)._get_declaration()
    params = declaration.parameters.model_dump(mode="json", exclude_none=True)
    props = params["properties"]

    assert set(params["required"]) >= {"ticker", "side", "quantity", "price_krw", "rationale"}
    assert props["side"]["enum"] == ["BUY", "SELL"]
    assert props["scope"]["enum"] == ["account", "agent_sleeve"]
    assert props["price_native"]["type"] == "NUMBER"
    assert props["price_native"]["nullable"] is True


def test_chat_config_tools_expose_structured_schema(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {candidate.__name__: candidate for candidate in agent.tools}

    assert "propose_config_change" not in tools
    assert {
        "propose_agent_config_change",
        "propose_chat_agent_config_change",
        "propose_tenant_config_change",
    }.issubset(tools)

    declaration = FunctionTool(tools["propose_agent_config_change"])._get_declaration()
    params = declaration.parameters.model_dump(mode="json", exclude_none=True)
    props = params["properties"]

    assert "change_json" not in props
    assert "agent_id" in params["required"]
    assert props["action"]["enum"] == ["update", "upsert", "add", "remove"]
    assert props["capital_allocation_mode"]["enum"] == ["unchanged", "fixed_krw", "account_percent", "whole_account"]


def test_chat_tool_schemas_do_not_emit_empty_enum_values_for_gemini(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool
    from arena.agents.investment_chat import factory
    from arena.agents.investment_chat import memory as chat_memory

    def walk_schema(schema, path: str = ""):
        if isinstance(schema, dict):
            enum_values = schema.get("enum")
            if enum_values:
                assert "" not in enum_values, path
            for key, value in schema.items():
                walk_schema(value, f"{path}.{key}" if path else key)
        elif isinstance(schema, list):
            for index, value in enumerate(schema):
                walk_schema(value, f"{path}[{index}]")

    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    repo = _ChatOrderRepo()

    _FakeExecutionMemory.instances.clear()
    monkeypatch.setattr(chat_memory, "MemoryStore", _FakeExecutionMemory, raising=False)
    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=None,
    )

    for tool in agent.tools:
        declaration = FunctionTool(tool)._get_declaration()
        payload = declaration.model_dump(mode="json", exclude_none=True)
        walk_schema(payload, getattr(tool, "__name__", "tool"))


def test_chat_analysis_tool_schema_keeps_required_fields_with_optional_params(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    from arena.agents.investment_chat import factory
    from arena.agents.investment_chat import memory as chat_memory

    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    repo = _ChatOrderRepo()

    _FakeExecutionMemory.instances.clear()
    monkeypatch.setattr(chat_memory, "MemoryStore", _FakeExecutionMemory, raising=False)
    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=None,
    )
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "optimize_portfolio")

    declaration = FunctionTool(tool)._get_declaration()
    params = declaration.parameters.model_dump(mode="json", exclude_none=True)

    assert "tickers" in params["required"]
    assert params["properties"]["tickers"]["items"]["type"] == "STRING"
    assert params["properties"]["strategy"]["enum"] == ["sharpe", "risk_parity", "forecast"]
    assert params["properties"]["forecast_mode"]["enum"] == ["default", "all", "stacked", "base", "balanced", "lgbm", "ridge", "avg"]


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


def test_chat_config_change_tool_requires_button_approval(monkeypatch) -> None:
    from arena.agents.investment_chat.drafts import config_draft_key

    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "agents_config",
        json.dumps(
            [
                {
                    "id": "gpt",
                    "provider": "gpt",
                    "model": "gpt-5.2",
                    "capital_krw": 1_000_000,
                    "target_market": "us",
                }
            ]
        ),
        "seed",
    )
    tools = _build_raw_chat_tools(monkeypatch, repo, include_internal_bridge=True)

    proposed = tools["propose_agent_config_change"](
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        capital_allocation_mode="account_percent",
        capital_allocation_percent=50,
        target_market="us",
        disabled_tools=["screen_market"],
        memory_compaction_model="gpt-5.4",
        rationale="gpt sleeve should manage half of the account",
    )

    token = str(proposed.get("approval_token") or "")
    assert proposed["status"] == "ok"
    assert proposed["approval_required"] is True
    assert token
    assert repo.get_config("local", config_draft_key(token))
    saved_before_approval = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert saved_before_approval[0]["model"] == "gpt-5.2"

    status = tools["get_config_change_status"](approval_token=token)

    assert status["status"] == "ok"
    assert status["drafts"][0]["approval_token"] == token
    assert status["drafts"][0]["submittable"] is True
    assert "gpt" in status["drafts"][0]["summary"]

    blocked = tools["apply_approved_config_change"](approval_token=token, confirmation_text="승인")

    assert blocked["status"] == "blocked"
    assert "CONFIRM" in blocked["required_confirmation"]
    assert repo.capital_sync_calls == []

    applied = tools["apply_approved_config_change"](
        approval_token=token,
        confirmation_text=f"CONFIRM {token}",
    )
    repeated = tools["apply_approved_config_change"](
        approval_token=token,
        confirmation_text=f"CONFIRM {token}",
    )

    assert applied["status"] == "applied"
    assert repeated["status"] == "already_applied"
    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert saved[0]["id"] == "gpt"
    assert saved[0]["model"] == "gpt-5.5"
    assert saved[0]["capital_krw"] == 5_000_000
    assert saved[0]["disabled_tools"] == ["screen_market"]
    assert saved[0]["memory_compaction_model"] == "gpt-5.4"
    assert repo.capital_sync_calls
    assert repo.capital_sync_calls[-1]["target_capitals"]["gpt"] == 5_000_000


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
    assert first_context.actions.skip_summarization is True
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


def test_refresh_account_snapshot_tool_calls_sync_service(monkeypatch) -> None:
    from arena.agents.investment_chat import account_tools

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {"kis_secret_name": "local-local-kis"}
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}
    calls: dict[str, object] = {}

    class _FakeAccountSyncService:
        def __init__(self, *, settings, repo):
            calls["settings"] = settings
            calls["repo"] = repo

        def sync_account_snapshot(self):
            calls["synced_at"] = datetime.now(timezone.utc)
            return AccountSnapshot(cash_krw=1.0, total_equity_krw=2.0, positions={})

    monkeypatch.setattr(account_tools, "AccountSyncService", _FakeAccountSyncService)

    result = tools["refresh_account_snapshot"]()

    assert result["status"] == "ok"
    assert result["total_equity_krw"] == 2.0
    assert calls["repo"] is repo


def test_refresh_account_snapshot_logs_unexpected_sync_failure(monkeypatch, caplog) -> None:
    from arena.agents.investment_chat import account_tools

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {"kis_secret_name": "local-local-kis"}
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}

    class _FailingAccountSyncService:
        def __init__(self, *, settings, repo):
            _ = settings, repo

        def sync_account_snapshot(self):
            raise RuntimeError("sync boom")

    monkeypatch.setattr(account_tools, "AccountSyncService", _FailingAccountSyncService)

    with caplog.at_level(logging.WARNING):
        result = tools["refresh_account_snapshot"]()

    assert result["status"] == "error"
    failure_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", "") == "chat_account_refresh_failed"
    )
    assert failure_record.exc_info is not None


def test_refresh_account_snapshot_defaults_to_total_account_markets(monkeypatch) -> None:
    from arena.agents.investment_chat import account_tools

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {"kis_secret_name": "local-local-kis"}
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}
    calls: dict[str, object] = {}

    class _FakeAccountSyncService:
        def __init__(self, *, settings, repo):
            calls["market"] = settings.kis_target_market
            calls["repo"] = repo

        def sync_account_snapshot(self):
            return AccountSnapshot(cash_krw=1.0, total_equity_krw=2.0, positions={})

    monkeypatch.setattr(account_tools, "AccountSyncService", _FakeAccountSyncService)

    result = tools["refresh_account_snapshot"]()

    assert result["status"] == "ok"
    assert calls["market"] == "us,kospi"
    assert repo.audit_logs[-1]["detail"]["target_market"] == "us,kospi"


def test_refresh_account_snapshot_uses_chat_account_market_override(monkeypatch) -> None:
    from arena.agents.investment_chat import account_tools

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {"kis_secret_name": "local-local-kis"}
    repo.set_config("local", "investment_chat_account_markets", "us,kospi", "tester")
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}
    calls: dict[str, object] = {}

    class _FakeAccountSyncService:
        def __init__(self, *, settings, repo):
            calls["market"] = settings.kis_target_market
            calls["repo"] = repo

        def sync_account_snapshot(self):
            return AccountSnapshot(cash_krw=1.0, total_equity_krw=2.0, positions={})

    monkeypatch.setattr(account_tools, "AccountSyncService", _FakeAccountSyncService)

    result = tools["refresh_account_snapshot"]()

    assert result["status"] == "ok"
    assert calls["market"] == "us,kospi"
    assert repo.audit_logs[-1]["detail"]["target_market"] == "us,kospi"


def test_refresh_account_snapshot_blocks_server_fallback_credentials(monkeypatch) -> None:
    from arena.agents.investment_chat import account_tools, factory

    repo = _ChatOrderRepo()
    settings = load_settings()
    settings.kis_secret_name = "KISAPI"
    settings.kis_api_key = "server-default-key"
    settings.kis_api_secret = "server-default-secret"
    settings.kis_account_no = "1234567890"
    settings.kis_target_market = "us"

    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    calls: dict[str, object] = {}

    class _FakeAccountSyncService:
        def __init__(self, *, settings, repo):
            calls["settings"] = settings
            calls["repo"] = repo

        def sync_account_snapshot(self):
            calls["synced"] = True
            return AccountSnapshot(cash_krw=1.0, total_equity_krw=2.0, positions={})

    monkeypatch.setattr(account_tools, "AccountSyncService", _FakeAccountSyncService)

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="czxnms",
        registry=ToolRegistry([]),
    )
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}

    result = tools["refresh_account_snapshot"]()

    assert result["status"] == "blocked"
    assert result["tenant_id"] == "czxnms"
    assert "tenant KIS credentials" in result["error"]
    assert "synced" not in calls
