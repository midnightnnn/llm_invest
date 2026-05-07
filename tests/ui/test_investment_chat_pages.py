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

def test_ui_registers_investment_chat_page_and_adk_mount(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )

    app = _build_app(repo=_DummyRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/investment-chat", params={"tenant_id": "local", "provider": "gpt", "model": "gpt-5.2"})

    assert response.status_code == 200
    assert "투자챗봇" in response.text
    assert ">투자챗봇</h2>" not in response.text
    assert "/investment-chat/adk/dev-ui/?tenant_id=local&amp;provider=gpt&amp;model=gpt-5.2" in response.text
    assert 'id="sidebar-collapse-toggle"' in response.text
    assert response.text.index('id="arena-sidebar"') < response.text.index('id="sidebar-collapse-toggle"') < response.text.index("<nav")
    collapsed_rule = response.text.split("body.sidebar-collapsed .arena-sidebar {", 1)[1].split("}", 1)[0]
    assert "margin-left" not in collapsed_rule
    assert "width: var(--sidebar-collapsed-w)" in collapsed_rule
    assert "flex-basis: var(--sidebar-collapsed-w)" in collapsed_rule
    assert "transform:" not in collapsed_rule
    assert "body.sidebar-collapsed #arena-sidebar nav" in response.text
    assert "body.sidebar-collapsed .sidebar-footer" in response.text
    assert client.session["investment_chat_tenant_id"] == "local"
    assert client.session["investment_chat_provider"] == "gpt"
    assert client.session["investment_chat_model"] == "gpt-5.2"
    # Selector form has been removed (provider/model now selected in /settings).
    assert "data-chat-selector-form" not in response.text
    assert "data-chat-provider" not in response.text
    assert "data-chat-model" not in response.text
    assert 'list="investment-chat-model-presets"' not in response.text
    assert "investment-chat-model-map" not in response.text
    assert "model.addEventListener('change', submitSelection)" not in response.text
    # Iframe + approval toast wiring still present.
    assert "data-adk-chat-frame" in response.text
    assert "submitResultMessage" in response.text
    assert "execution_report.message" in response.text
    assert "deliverOrderResultToChat" in response.text
    assert "Config Approval" in response.text
    assert "data-config-draft-panel" in response.text
    assert "/investment-chat/config-drafts" in response.text
    assert "data-config-draft-apply" in response.text
    assert "isAdkChatBusy" in response.text
    assert "mat-progress-bar" in response.text
    assert "textarea.chat-input-box" in response.text
    assert "button.send-message-btn" in response.text
    assert 'data-active="investment_chat"' in response.text
    assert "investment-chat-shell" in response.text
    assert "calc(100dvh - var(--mobile-topbar-h))" in response.text
    assert "investment-chat-frame" in response.text
    assert 'body[data-active="investment_chat"] .sidebar-backdrop.open' in response.text
    assert any(getattr(route, "path", "") == "/investment-chat/adk" for route in app.routes)


def test_investment_chat_page_normalizes_removed_gemini_flash_preview(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )

    app = _build_app(repo=_DummyRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get(
        "/investment-chat",
        params={
            "tenant_id": "local",
            "provider": "gemini",
            "model": "gemini-3.1-flash-preview",
        },
    )

    assert response.status_code == 200
    assert "/investment-chat/adk/dev-ui/?tenant_id=local&amp;provider=gemini&amp;model=gemini-3-flash-preview" in response.text
    assert "gemini-3.1-flash-preview" not in response.text
    assert client.session["investment_chat_model"] == "gemini-3-flash-preview"


def test_investment_chat_model_select_renders_all_provider_presets(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )

    app = _build_app(repo=_DummyRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get(
        "/investment-chat",
        params={"tenant_id": "local", "provider": "claude", "model": "claude-sonnet-4-6"},
    )

    # Selector form moved to /settings; the chat page just renders the iframe
    # against the resolved provider/model and saves the choice in the session.
    assert response.status_code == 200
    assert '<select\n      name="model"' not in response.text
    assert "data-chat-selector-form" not in response.text
    assert (
        "/investment-chat/adk/dev-ui/?tenant_id=local&amp;provider=claude&amp;model=claude-sonnet-4-6"
        in response.text
    )
    assert client.session["investment_chat_provider"] == "claude"
    assert client.session["investment_chat_model"] == "claude-sonnet-4-6"


def test_investment_chat_provider_options_come_from_adk_provider_registry(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )

    app = _build_app(repo=_DummyRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get(
        "/investment-chat",
        params={"tenant_id": "local", "provider": "deepseek", "model": "deepseek-reasoner"},
    )

    # Selector form is gone; the route still recognizes provider tokens registered
    # in the ADK provider registry and resolves them into the iframe URL + session.
    assert response.status_code == 200
    assert "data-chat-selector-form" not in response.text
    assert (
        "/investment-chat/adk/dev-ui/?tenant_id=local&amp;provider=deepseek&amp;model=deepseek-reasoner"
        in response.text
    )
    assert client.session["investment_chat_provider"] == "deepseek"
    assert client.session["investment_chat_model"] == "deepseek-reasoner"


def test_investment_chat_page_defaults_to_stored_chat_agent_config(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _DummyRepo()
    repo.set_config(
        "local",
        "investment_chat_config",
        json.dumps({"provider": "claude", "model": "claude-opus-4-7"}),
        "seed",
    )
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/investment-chat", params={"tenant_id": "local"})

    assert response.status_code == 200
    assert "data-chat-selector-form" not in response.text
    assert "/investment-chat/adk/dev-ui/?tenant_id=local&amp;provider=claude&amp;model=claude-opus-4-7" in response.text
    assert client.session["investment_chat_provider"] == "claude"
    assert client.session["investment_chat_model"] == "claude-opus-4-7"


def test_investment_chat_page_defaults_to_tenant_agent_model(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _DummyRepo()
    repo.set_config(
        "cxznms",
        "agents_config",
        json.dumps([
            {"id": "claude", "provider": "claude", "model": "claude-sonnet-4-6", "capital_krw": 1_000_000}
        ]),
        "seed",
    )
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/investment-chat", params={"tenant_id": "cxznms"})

    assert response.status_code == 200
    assert "data-chat-selector-form" not in response.text
    assert "/investment-chat/adk/dev-ui/?tenant_id=cxznms&amp;provider=claude&amp;model=claude-sonnet-4-6" in response.text
    assert client.session["investment_chat_provider"] == "claude"
    assert client.session["investment_chat_model"] == "claude-sonnet-4-6"


def test_investment_chat_page_ignores_stale_session_provider_for_new_tenant(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _DummyRepo()
    repo.set_config(
        "cxznms",
        "agents_config",
        json.dumps([
            {"id": "claude", "provider": "claude", "model": "claude-sonnet-4-6", "capital_krw": 1_000_000}
        ]),
        "seed",
    )
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)
    client.session["investment_chat_tenant_id"] = "midnightnnn"
    client.session["investment_chat_provider"] = "gemini"
    client.session["investment_chat_model"] = "gemini-3-flash-preview"

    response = client.get("/investment-chat", params={"tenant_id": "cxznms"})

    assert response.status_code == 200
    assert "data-chat-selector-form" not in response.text
    assert "/investment-chat/adk/dev-ui/?tenant_id=cxznms&amp;provider=claude&amp;model=claude-sonnet-4-6" in response.text
    assert client.session["investment_chat_tenant_id"] == "cxznms"
    assert client.session["investment_chat_provider"] == "claude"
    assert client.session["investment_chat_model"] == "claude-sonnet-4-6"


def test_investment_chat_provider_options_are_limited_to_tenant_model_keys(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _DummyRepo()
    repo.runtime_credentials["czxnms"] = {
        "tenant_id": "czxnms",
        "model_secret_name": "local-czxnms-models",
        "has_openai": False,
        "has_gemini": False,
        "has_anthropic": True,
    }
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get(
        "/investment-chat",
        params={"tenant_id": "czxnms", "provider": "gemini", "model": "gemini-3-flash-preview"},
    )

    # Even without a selector UI, the route still filters provider tokens against
    # tenant credentials and falls back to the only available provider (claude).
    assert response.status_code == 200
    assert "data-chat-selector-form" not in response.text
    assert "/investment-chat/adk/dev-ui/?tenant_id=czxnms&amp;provider=claude" in response.text
    assert "gemini-3-flash-preview" not in response.text
    assert client.session["investment_chat_provider"] == "claude"
    assert client.session["investment_chat_model"].startswith("claude-")


def test_investment_chat_provider_options_show_no_iframe_when_no_tenant_model_keys(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _DummyRepo()
    repo.runtime_credentials["czxnms"] = {
        "tenant_id": "czxnms",
        "model_secret_name": "local-czxnms-models",
        "has_openai": False,
        "has_gemini": False,
        "has_anthropic": False,
    }
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/investment-chat", params={"tenant_id": "czxnms"})

    assert response.status_code == 200
    # Empty-state message now points to /settings (provider/model selector moved there).
    assert "등록된 LLM API key가 없거나 챗봇 모델이 설정되지 않았습니다" in response.text
    assert "환경설정" in response.text
    assert "<iframe" not in response.text
