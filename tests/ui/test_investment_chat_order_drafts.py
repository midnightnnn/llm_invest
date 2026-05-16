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

def test_investment_chat_config_draft_api_lists_and_applies_pending_draft(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "agents_config",
        json.dumps([{"id": "gpt", "provider": "gpt", "model": "gpt-5.2", "capital_krw": 1_000_000}]),
        "seed",
    )
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    settings = load_settings()
    tools = _build_raw_chat_tools(monkeypatch, repo, settings=settings, include_internal_bridge=True)
    draft = tools["propose_agent_config_change"](
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        capital_krw=2_000_000,
        rationale="upgrade model and capital",
    )
    token = str(draft["approval_token"])
    client = DirectRouteClient(_build_app(repo=repo, settings=settings))

    listed = client.get("/investment-chat/config-drafts", params={"tenant_id": "local"})

    assert listed.status_code == 200
    payload = listed.json()
    assert payload["status"] == "ok"
    assert payload["drafts"][0]["approval_token"] == token
    assert payload["drafts"][0]["submittable"] is True

    applied = client.post(f"/investment-chat/config-drafts/{token}/apply", params={"tenant_id": "local"})
    repeated = client.post(f"/investment-chat/config-drafts/{token}/apply", params={"tenant_id": "local"})

    assert applied.status_code == 200
    assert applied.json()["status"] == "applied"
    assert "chat_delivery_text" in applied.json()
    assert repeated.json()["status"] == "already_applied"
    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert saved[0]["model"] == "gpt-5.5"
    assert saved[0]["capital_krw"] == 2_000_000


def test_investment_chat_order_draft_api_lists_and_submits_pending_draft(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    validate = registry.get("validate_order_draft").callable

    draft = validate(ticker="AAPL", side="BUY", quantity=1, price_krw=100_000, rationale="button approval test")
    token = str(draft["approval_token"])
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)

    listed = client.get("/investment-chat/order-drafts", params={"tenant_id": "local"})

    assert listed.status_code == 200
    payload = listed.json()
    assert payload["drafts"][0]["approval_token"] == token
    assert payload["drafts"][0]["submittable"] is True
    assert payload["drafts"][0]["intent"]["ticker"] == "AAPL"
    assert "required_confirmation" not in payload["drafts"][0]

    submitted = client.post(f"/investment-chat/order-drafts/{token}/submit", params={"tenant_id": "local"})

    assert submitted.status_code == 200
    result = submitted.json()
    assert result["status"] == "submitted"
    assert len(repo.execution_reports) == 1


def test_investment_chat_order_draft_api_batch_submits_pending_drafts(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    validate = registry.get("validate_order_draft").callable

    first = validate(ticker="AAPL", side="BUY", quantity=1, price_krw=100_000, rationale="button batch approval AAPL")
    second = validate(ticker="MSFT", side="BUY", quantity=1, price_krw=100_000, rationale="button batch approval MSFT")
    tokens = [str(first["approval_token"]), str(second["approval_token"])]
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)

    listed = client.get("/investment-chat/order-drafts", params={"tenant_id": "local"})
    payload = listed.json()

    assert listed.status_code == 200
    assert [item["approval_token"] for item in payload["drafts"][:2]] == list(reversed(tokens))

    submitted = client.post(
        "/investment-chat/order-drafts/batch-submit",
        params={"tenant_id": "local"},
        json={"approval_tokens": tokens},
    )

    assert submitted.status_code == 200
    result = submitted.json()
    assert result["status"] == "submitted"
    assert result["order_count"] == 2
    assert result["submitted_count"] == 2
    assert len(repo.execution_reports) == 2
    assert result["chat_delivery_text"] == "방금 주문 2건 일괄 승인 결과를 확인해서 알려줘."


def test_investment_chat_order_draft_api_hides_adk_confirmation_drafts(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    submit_with_confirmation = registry.get("submit_order_with_confirmation").callable

    waiting = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="ADK confirmation should not create host approval card",
        tool_context=_FakeToolContext(),
    )
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)

    listed = client.get("/investment-chat/order-drafts", params={"tenant_id": "local"})

    assert waiting["status"] == "waiting_for_confirmation"
    assert listed.status_code == 200
    assert listed.json()["drafts"] == []


def test_investment_chat_config_draft_api_hides_adk_confirmation_drafts(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    settings = load_settings()
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "agents_config",
        json.dumps([{"id": "gpt", "provider": "gpt", "model": "gpt-5.2", "capital_krw": 1_000_000}]),
        "seed",
    )
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    propose = registry.get("propose_agent_config_change").callable

    waiting = propose(
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        rationale="ADK confirmation should not create host config approval card",
        tool_context=_FakeToolContext(function_call_id="fc-config-route"),
    )
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)

    listed = client.get("/investment-chat/config-drafts", params={"tenant_id": "local"})

    assert waiting["status"] == "waiting_for_confirmation"
    assert listed.status_code == 200
    assert listed.json()["drafts"] == []


def test_investment_chat_order_draft_api_surfaces_broker_error_message(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())

    class _RejectingBroker:
        def place_order(self, intent, *, fx_rate=None):
            _ = intent, fx_rate
            return ExecutionReport(
                status=ExecutionStatus.ERROR,
                order_id="err_holiday",
                filled_qty=0,
                avg_price_krw=0,
                message="금일은 해외 휴장일로 주문이 불가합니다.",
            )

    monkeypatch.setattr(order_tools, "PaperBroker", _RejectingBroker)
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    validate = registry.get("validate_order_draft").callable

    draft = validate(ticker="AAPL", side="BUY", quantity=1, price_krw=100_000, rationale="button approval test")
    token = str(draft["approval_token"])
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)

    submitted = client.post(f"/investment-chat/order-drafts/{token}/submit", params={"tenant_id": "local"})

    assert submitted.status_code == 200
    result = submitted.json()
    assert result["status"] == "error"
    assert result["message"] == "금일은 해외 휴장일로 주문이 불가합니다."
    assert result["error"] == "금일은 해외 휴장일로 주문이 불가합니다."
    assert result["execution_report"]["message"] == "금일은 해외 휴장일로 주문이 불가합니다."
    assert result["chat_delivery_text"] == "방금 AAPL BUY 1주 주문 승인 결과를 확인해서 알려줘."
    assert "[주문 승인 패널 결과]" not in result["chat_delivery_text"]
    assert "금일은 해외 휴장일로 주문이 불가합니다." not in result["chat_delivery_text"]
    assert "/uapi/" not in result["chat_delivery_text"]
    assert token not in result["chat_delivery_text"]
    assert "CONFIRM" not in result["chat_delivery_text"]


def test_get_order_approval_status_reads_latest_button_result(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())

    class _RejectingBroker:
        def place_order(self, intent, *, fx_rate=None):
            _ = intent, fx_rate
            return ExecutionReport(
                status=ExecutionStatus.ERROR,
                order_id="err_holiday",
                filled_qty=0,
                avg_price_krw=0,
                message="금일은 해외 휴장일로 주문이 불가합니다.",
            )

    monkeypatch.setattr(order_tools, "PaperBroker", _RejectingBroker)
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    validate = registry.get("validate_order_draft").callable
    status_tool = registry.get("get_order_approval_status").callable

    draft = validate(ticker="AAPL", side="BUY", quantity=1, price_krw=100_000, rationale="button approval test")
    token = str(draft["approval_token"])
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)
    client.post(f"/investment-chat/order-drafts/{token}/submit", params={"tenant_id": "local"})

    status = status_tool(approval_token=token)

    assert status["status"] == "ok"
    assert status["orders"][0]["status"] == "error"
    assert status["orders"][0]["ticker"] == "AAPL"
    assert status["orders"][0]["message"] == "금일은 해외 휴장일로 주문이 불가합니다."


def test_get_order_approval_status_reads_latest_button_result_from_detail_json(monkeypatch) -> None:
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.order_tools import build_order_bridge_tool_entries
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())

    class _RejectingBroker:
        def place_order(self, intent, *, fx_rate=None):
            _ = intent, fx_rate
            return ExecutionReport(
                status=ExecutionStatus.ERROR,
                order_id="err_holiday",
                filled_qty=0,
                avg_price_krw=0,
                message="금일은 해외 휴장일로 주문이 불가합니다.",
            )

    monkeypatch.setattr(order_tools, "PaperBroker", _RejectingBroker)
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    validate = registry.get("validate_order_draft").callable
    assert registry.get("submit_approved_order") is None
    bridge_entries = build_order_bridge_tool_entries(repo=repo, settings=settings, tenant_id="local")
    submit = {entry.name: entry.callable for entry in bridge_entries}["submit_approved_order"]
    status_tool = registry.get("get_order_approval_status").callable

    draft = validate(ticker="AAPL", side="BUY", quantity=1, price_krw=100_000, rationale="button approval test")
    token = str(draft["approval_token"])
    submit(approval_token=token, confirmation_text=f"CONFIRM {token}")
    for row in repo.audit_logs:
        detail = row.pop("detail", None)
        if detail is not None:
            row["detail_json"] = json.dumps(detail)

    status = status_tool()

    assert status["status"] == "ok"
    assert status["count"] == 1
    assert status["orders"][0]["approval_token"] == token
    assert status["orders"][0]["status"] == "error"
    assert status["orders"][0]["message"] == "금일은 해외 휴장일로 주문이 불가합니다."
