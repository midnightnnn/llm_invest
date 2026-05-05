from __future__ import annotations

from types import SimpleNamespace

from arena.config import AgentConfig, load_settings
from arena.tools.registry import ToolRegistry
from tests.ui.helpers import _DummyRepo


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
