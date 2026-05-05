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

def test_investment_chat_loader_request_selection_overrides_stale_adk_app_name(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_MODEL, REQUEST_PROVIDER, REQUEST_TENANT
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}
    settings = load_settings()

    def fake_build_agent(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(name="investment_chat", description="chat")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )

    tenant_token = REQUEST_TENANT.set("czxnms")
    provider_token = REQUEST_PROVIDER.set("gemini")
    model_token = REQUEST_MODEL.set("gemini-3-flash-preview")
    try:
        stale_app_name = "investment_chat__midnightnnn__gpt__m_Z3B0LTUuMg"
        loader.load_agent(stale_app_name)
    finally:
        REQUEST_MODEL.reset(model_token)
        REQUEST_PROVIDER.reset(provider_token)
        REQUEST_TENANT.reset(tenant_token)

    assert calls["tenant_id"] == "czxnms"
    assert calls["provider"] == "gemini"
    assert calls["model_override"] == "gemini-3-flash-preview"


def test_investment_chat_adk_api_requires_ui_auth(monkeypatch) -> None:
    import asyncio

    from arena.ui import investment_chat_adk

    def fake_get_fast_api_app(**kwargs):
        _ = kwargs
        app = FastAPI(title="fake-adk")

        @app.get("/list-apps")
        def list_apps():
            return ["investment_chat"]

        return app

    monkeypatch.setattr(investment_chat_adk, "get_fast_api_app", fake_get_fast_api_app)
    monkeypatch.setattr(investment_chat_adk, "_mount_adk_static", lambda app, url_prefix: None)

    blocked_app = investment_chat_adk.build_investment_chat_adk_app(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: load_settings(),
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
        auth_enabled=True,
        current_user=lambda request: None,
    )
    allowed_app = investment_chat_adk.build_investment_chat_adk_app(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: load_settings(),
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
        auth_enabled=True,
        current_user=lambda request: {"email": "user@example.com"},
    )

    def make_request(headers: dict[str, str] | None = None) -> Request:
        raw_headers = [
            (str(key).lower().encode("latin-1"), str(value).encode("latin-1"))
            for key, value in (headers or {}).items()
        ]
        return Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/list-apps",
                "raw_path": b"/list-apps",
                "query_string": b"",
                "headers": raw_headers,
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
                "session": {},
            },
            receive=lambda: {"type": "http.request", "body": b"", "more_body": False},
        )

    async def passthrough(_request: Request):
        return JSONResponse(["investment_chat"])

    blocked_dispatch = blocked_app.user_middleware[0].kwargs["dispatch"]
    allowed_dispatch = allowed_app.user_middleware[0].kwargs["dispatch"]
    blocked_response = asyncio.run(blocked_dispatch(make_request({"accept": "application/json"}), passthrough))
    allowed_response = asyncio.run(allowed_dispatch(make_request(), passthrough))

    assert blocked_response.status_code == 401
    assert allowed_response.status_code == 200
    assert json.loads(allowed_response.body) == ["investment_chat"]


def test_investment_chat_adk_auth_middleware_uses_provider_model_query(monkeypatch) -> None:
    import asyncio

    from arena.agents.investment_chat.context import REQUEST_MODEL, REQUEST_PROVIDER
    from arena.ui import investment_chat_adk

    def fake_get_fast_api_app(**kwargs):
        _ = kwargs
        return FastAPI(title="fake-adk")

    monkeypatch.setattr(investment_chat_adk, "get_fast_api_app", fake_get_fast_api_app)
    monkeypatch.setattr(investment_chat_adk, "_mount_adk_static", lambda app, url_prefix: None)

    app = investment_chat_adk.build_investment_chat_adk_app(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: load_settings(),
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
        auth_enabled=True,
        current_user=lambda request: {"email": "user@example.com"},
    )
    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/dev-ui/",
            "raw_path": b"/dev-ui/",
            "query_string": b"provider=gpt&model=gpt-5.4",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "session": {},
        },
        receive=lambda: {"type": "http.request", "body": b"", "more_body": False},
    )

    async def passthrough(_request: Request):
        return JSONResponse(
            {
                "provider": REQUEST_PROVIDER.get(),
                "model": REQUEST_MODEL.get(),
                "session_provider": _request.session.get("investment_chat_provider"),
                "session_model": _request.session.get("investment_chat_model"),
            }
        )

    dispatch = app.user_middleware[0].kwargs["dispatch"]
    response = asyncio.run(dispatch(request, passthrough))

    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["provider"] == "gpt"
    assert payload["model"] == "gpt-5.4"
    assert payload["session_provider"] == "gpt"
    assert payload["session_model"] == "gpt-5.4"


def test_investment_chat_adk_auth_middleware_blocks_stale_run_body_app(monkeypatch) -> None:
    import asyncio

    from arena.ui import investment_chat_adk

    def fake_get_fast_api_app(**kwargs):
        _ = kwargs
        return FastAPI(title="fake-adk")

    monkeypatch.setattr(investment_chat_adk, "get_fast_api_app", fake_get_fast_api_app)
    monkeypatch.setattr(investment_chat_adk, "_mount_adk_static", lambda app, url_prefix: None)

    app = investment_chat_adk.build_investment_chat_adk_app(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: load_settings(),
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
        auth_enabled=True,
        current_user=lambda request: {"email": "user@example.com"},
    )
    body = json.dumps(
        {
            "app_name": "investment_chat__local__gemini__m_Z2VtaW5pLTMuMS1mbGFzaC1wcmV2aWV3",
            "user_id": "user",
            "session_id": "s1",
        }
    ).encode("utf-8")

    async def receive_body():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/run_sse",
            "raw_path": b"/run_sse",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "session": {
                "investment_chat_tenant_id": "local",
                "investment_chat_provider": "gpt",
                "investment_chat_model": "gpt-5.4",
            },
        },
        receive=receive_body,
    )

    async def passthrough(_request: Request):
        return JSONResponse({"status": "unexpected"})

    dispatch = app.user_middleware[0].kwargs["dispatch"]
    response = asyncio.run(dispatch(request, passthrough))

    assert response.status_code == 409
    payload = json.loads(response.body)
    assert payload["error"] == "stale adk app_name provider"
    assert payload["provider"] == "gpt"
    assert payload["app_name_provider"] == "gemini"


def test_investment_chat_adk_rejects_stale_path_app_name_tenant() -> None:
    import asyncio

    from arena.ui import investment_chat_adk

    stale_app_name = investment_chat_adk._chat_app_name("midnightnnn", "gpt", "gpt-5.5")
    request = SimpleNamespace(url=SimpleNamespace(path=f"/apps/{stale_app_name}/app-info"))

    response = asyncio.run(
        investment_chat_adk._stale_app_name_response(
            request,
            tenant="czxnms",
            provider="gemini",
            model="gemini-3-flash-preview",
        )
    )

    assert response is not None
    assert response.status_code == 409
    payload = json.loads(response.body)
    assert payload["error"] == "stale adk app_name tenant"
    assert payload["tenant_id"] == "czxnms"
    assert payload["app_name_tenant"] == "midnightnnn"


def test_investment_chat_adk_defaults_to_data_sqlite_sessions(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}

    def fake_get_fast_api_app(**kwargs):
        calls.update(kwargs)
        return FastAPI(title="fake-adk")

    monkeypatch.delenv("ARENA_CHAT_SESSION_SERVICE_URI", raising=False)
    monkeypatch.setattr(investment_chat_adk, "get_fast_api_app", fake_get_fast_api_app)
    monkeypatch.setattr(investment_chat_adk, "_mount_adk_static", lambda app, url_prefix: None)

    investment_chat_adk.build_investment_chat_adk_app(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: load_settings(),
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )

    expected = investment_chat_adk.Path(investment_chat_adk.__file__).resolve().parents[2]
    expected = expected / "data" / "arena-investment-chat-adk-sessions.sqlite"
    assert calls["session_service_uri"] == f"sqlite:///{expected}"
