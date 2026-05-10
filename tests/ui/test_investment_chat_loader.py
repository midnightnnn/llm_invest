from __future__ import annotations

import json
from types import SimpleNamespace

from arena.config import AgentConfig, load_settings
from arena.tools.registry import ToolRegistry
from tests.ui.helpers import _DummyRepo


def test_adk_browser_overrides_do_not_hide_or_overlay_debug_drawer() -> None:
    from arena.ui import investment_chat_adk

    css = investment_chat_adk._MOBILE_OVERRIDE_CSS

    assert "side-panel-container" not in css
    assert "mat-drawer.mat-drawer-side" not in css
    assert "position: absolute !important" not in css
    assert "mat-toolbar > div:first-child > button:first-child" not in css


def test_adk_browser_overrides_blur_chat_input_after_mobile_submit() -> None:
    from arena.ui import investment_chat_adk

    script = investment_chat_adk._MOBILE_KEYBOARD_DISMISSAL_SCRIPT

    assert "installArenaMobileKeyboardDismissal" in script
    assert "textarea.chat-input-box" in script
    assert "button.send-message-btn" in script
    assert "active.blur()" in script
    assert "keydown" in script
    assert "Enter" in script


def test_adk_browser_mobile_chat_input_uses_compact_bottom_spacing() -> None:
    from arena.ui import investment_chat_adk

    css = investment_chat_adk._MOBILE_OVERRIDE_CSS

    assert ".chat-input-container { padding: 0 !important; }" in css
    assert ".chat-input {\n    width: 100% !important;\n    padding: 6px 10px 0 !important;" in css
    assert ".chat-input-actions { margin-top: 4px !important;" in css
    assert "padding-bottom: 0 !important;" in css
    assert "safe area / viewport sizing" in css
    assert "max(16px, env(safe-area-inset-bottom))" not in css


def test_adk_browser_overrides_hide_live_call_controls() -> None:
    from arena.ui import investment_chat_adk

    css = investment_chat_adk._MOBILE_OVERRIDE_CSS

    assert "app-call-controls" in css
    assert ".call-btn-container" in css
    assert "button.audio-rec-btn" in css
    assert "Live calls" in css


def test_investment_chat_loader_binds_default_tenant(monkeypatch) -> None:
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
        default_tenant="MidNightNnN",
    )

    agent = loader.load_agent("investment_chat")

    assert agent.name == "investment_chat"
    assert loader.list_agents() == ["investment_chat__midnightnnn__gemini__m_Z2VtaW5pLTMtZmxhc2gtcHJldmlldw"]
    assert calls["tenant_id"] == "midnightnnn"
    assert calls["settings"] is settings
    assert calls["registry"] is None

    loader.load_agent("investment_chat__research")
    assert calls["tenant_id"] == "research"


def test_investment_chat_loader_separates_model_selection(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_MODEL, REQUEST_PROVIDER
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

    provider_token = REQUEST_PROVIDER.set("gpt")
    model_token = REQUEST_MODEL.set("gpt-5.5")
    try:
        listed = loader.list_agents()
        loader.load_agent(listed[0])
    finally:
        REQUEST_MODEL.reset(model_token)
        REQUEST_PROVIDER.reset(provider_token)

    assert listed == ["investment_chat__local__gpt__m_Z3B0LTUuNQ"]
    assert calls["tenant_id"] == "local"
    assert calls["provider"] == "gpt"
    assert calls["model_override"] == "gpt-5.5"

    calls.clear()
    second_loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )
    second_loader.load_agent(listed[0])

    assert calls["provider"] == "gpt"
    assert calls["model_override"] == "gpt-5.5"

    calls.clear()
    gemini_provider = REQUEST_PROVIDER.set("gemini")
    gemini_model = REQUEST_MODEL.set("gemini-3.1-pro-preview")
    try:
        gemini_listed = second_loader.list_agents()
    finally:
        REQUEST_MODEL.reset(gemini_model)
        REQUEST_PROVIDER.reset(gemini_provider)
    third_loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )
    third_loader.load_agent(gemini_listed[0])

    assert gemini_listed == ["investment_chat__local__gemini__m_Z2VtaW5pLTMuMS1wcm8tcHJldmlldw"]
    assert calls["provider"] == "gemini"
    assert calls["model_override"] == "gemini-3.1-pro-preview"


def test_investment_chat_loader_rebuilds_after_settings_fingerprint_changes(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    builds: list[dict[str, object]] = []
    settings = load_settings()
    settings.openai_api_key = "old-key"

    def fake_build_agent(**kwargs):
        builds.append(dict(kwargs))
        return SimpleNamespace(name=f"agent-{len(builds)}")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )
    app_name = "investment_chat__local__gpt__m_Z3B0LTUuNQ"

    first = loader.load_agent(app_name)
    second = loader.load_agent(app_name)
    settings.openai_api_key = "new-key"
    third = loader.load_agent(app_name)

    assert first is second
    assert third is not first
    assert [item["model_override"] for item in builds] == ["gpt-5.5", "gpt-5.5"]


def test_investment_chat_loader_rebuilds_after_chat_config_fingerprint_changes(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    builds: list[dict[str, object]] = []
    settings = load_settings()
    repo = _DummyRepo()

    def fake_build_agent(**kwargs):
        builds.append(dict(kwargs))
        return SimpleNamespace(name=f"agent-{len(builds)}")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=repo,
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )
    app_name = "investment_chat__local__gpt__m_Z3B0LTUuNQ"

    first = loader.load_agent(app_name)
    second = loader.load_agent(app_name)
    repo.set_config(
        "local",
        "investment_chat_config",
        json.dumps({"model_routing": {"cheap_model_by_provider": {"gpt": "gpt-5.4-mini"}}}),
        "tester",
    )
    third = loader.load_agent(app_name)

    assert first is second
    assert third is not first
    assert [item["model_override"] for item in builds] == ["gpt-5.5", "gpt-5.5"]


def test_investment_chat_loader_uses_encoded_claude_selection_without_request_context(monkeypatch) -> None:
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

    claude_app_name = "investment_chat__local__claude__m_Y2xhdWRlLXNvbm5ldC00LTY"
    loader.load_agent(claude_app_name)

    assert calls["provider"] == "claude"
    assert calls["model_override"] == "claude-sonnet-4-6"


def test_investment_chat_loader_normalizes_removed_gemini_flash_preview(monkeypatch) -> None:
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

    removed_model_app_name = "investment_chat__local__gemini__m_Z2VtaW5pLTMuMS1mbGFzaC1wcmV2aWV3"
    loader.load_agent(removed_model_app_name)

    assert calls["provider"] == "gemini"
    assert calls["model_override"] == "gemini-3-flash-preview"


def test_investment_chat_loader_restricts_selection_to_tenant_model_keys(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_MODEL, REQUEST_PROVIDER
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}
    settings = load_settings()
    repo = _DummyRepo()
    repo.runtime_credentials["czxnms"] = {
        "tenant_id": "czxnms",
        "model_secret_name": "local-czxnms-models",
        "has_openai": False,
        "has_gemini": False,
        "has_anthropic": True,
    }

    def fake_build_agent(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(name="investment_chat", description="chat")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=repo,
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="czxnms",
    )

    provider_token = REQUEST_PROVIDER.set("gemini")
    model_token = REQUEST_MODEL.set("gemini-3-flash-preview")
    try:
        listed = loader.list_agents()
        loader.load_agent(listed[0])
    finally:
        REQUEST_MODEL.reset(model_token)
        REQUEST_PROVIDER.reset(provider_token)

    assert listed[0].startswith("investment_chat__czxnms__claude__m_")
    assert calls["provider"] == "claude"
    assert str(calls["model_override"]).startswith("claude-")


def test_investment_chat_loader_defaults_to_tenant_agent_model(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}
    settings = load_settings()
    settings.agent_ids = ["claude"]
    settings.agent_configs = {
        "claude": AgentConfig(
            agent_id="claude",
            provider="claude",
            model="claude-sonnet-4-6",
            capital_krw=1_000_000,
        )
    }
    repo = _DummyRepo()

    def fake_build_agent(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(name="investment_chat", description="chat")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=repo,
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="cxznms",
    )

    listed = loader.list_agents()
    loader.load_agent(listed[0])

    assert listed[0] == "investment_chat__cxznms__claude__m_Y2xhdWRlLXNvbm5ldC00LTY"
    assert calls["provider"] == "claude"
    assert calls["model_override"] == "claude-sonnet-4-6"


def test_investment_chat_loader_ignores_stale_stored_advisor_model(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}
    settings = load_settings()
    settings.agent_ids = ["gpt", "gemini", "claude"]
    settings.agent_configs = {
        "gpt": AgentConfig(
            agent_id="gpt",
            provider="gpt",
            model="gpt-5.5",
            capital_krw=1_000_000,
        ),
        "gemini": AgentConfig(
            agent_id="gemini",
            provider="gemini",
            model="gemini-3.1-pro-preview",
            capital_krw=1_000_000,
        ),
        "claude": AgentConfig(
            agent_id="claude",
            provider="claude",
            model="claude-opus-4-7",
            capital_krw=1_000_000,
        ),
    }
    settings.openai_model = "gpt-5.5"
    settings.gemini_model = "gemini-3.1-pro-preview"
    settings.anthropic_model = "claude-opus-4-7"
    repo = _DummyRepo()
    repo.set_config(
        "midnightnnn",
        "investment_chat_config",
        json.dumps({"provider": "gpt", "model": "gpt-5.2"}),
        "seed",
    )

    def fake_build_agent(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(name="investment_chat", description="chat")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=repo,
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="midnightnnn",
    )

    listed = loader.list_agents()
    loader.load_agent(listed[0])

    assert listed[0] == "investment_chat__midnightnnn__gpt__m_Z3B0LTUuNQ"
    assert calls["provider"] == "gpt"
    assert calls["model_override"] == "gpt-5.5"
