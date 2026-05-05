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

def test_investment_chat_factory_delegates_tool_implementations() -> None:
    import inspect

    from arena.agents.investment_chat import factory

    source = inspect.getsource(factory)

    assert "def get_account_snapshot(" not in source
    assert "def validate_order_draft(" not in source
    assert "def submit_approved_order(" not in source


def test_default_layout_places_investment_chat_under_memory_nav() -> None:
    html = tailwind_layout("Board", "<div>body</div>", active="investment_chat")

    assert "/investment-chat" in html
    assert "투자챗봇" in html
    assert 'href="/investment-chat" class="sidebar-link active"' in html
    assert html.index("기억관리") < html.index("투자챗봇")
    assert "bottom_nav_links" not in html


def test_layout_preserves_tenant_in_investment_chat_nav() -> None:
    html = tailwind_layout("Board", "<div>body</div>", active="board", tenant="MidNightNnN")

    assert 'href="/investment-chat?tenant_id=midnightnnn"' in html


def test_build_investment_chat_agent_filters_write_tools(monkeypatch) -> None:
    from arena.agents.investment_chat import factory

    settings = load_settings()
    repo = _DummyRepo()
    registry = ToolRegistry(
        [
            ToolEntry(
                tool_id="recommend_opportunities",
                name="recommend_opportunities",
                description="read tool",
                category="quant",
                callable=lambda top_n=8: {"top_n": top_n},
            ),
            ToolEntry(
                tool_id="execute_order",
                name="execute_order",
                description="write tool",
                category="execution",
                callable=lambda: {"submitted": True},
            ),
            ToolEntry(
                tool_id="screen_market",
                name="screen_market",
                description="safe diagnostic read tool",
                category="quant",
                callable=lambda bucket="momentum": {"bucket": bucket},
            ),
            ToolEntry(
                tool_id="scratch_run_python",
                name="scratch_run_python",
                description="scratch tool",
                category="analysis",
                callable=lambda code="": {"code": code},
            ),
        ]
    )

    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=registry,
    )

    tool_names = {getattr(tool, "__name__", "") for tool in agent.tools}
    assert "recommend_opportunities" in tool_names
    assert "screen_market" in tool_names
    assert "get_account_snapshot" in tool_names
    assert "get_agent_sleeve_snapshot" in tool_names
    assert "get_trade_history" in tool_names
    assert "get_order_approval_status" in tool_names
    assert "submit_order_with_confirmation" in tool_names
    assert "validate_order_draft" in tool_names
    assert "refresh_account_snapshot" in tool_names
    assert "submit_approved_order" not in tool_names
    assert "execute_order" not in tool_names
    assert "submit_order" not in tool_names
    assert "scratch_run_python" not in tool_names
    assert "live" not in agent.instruction.lower()


def test_investment_chat_builds_analysis_tools_with_total_account_market_scope(monkeypatch) -> None:
    from arena.agents.investment_chat import factory
    from arena.agents.investment_chat import registry as chat_registry

    settings = load_settings()
    settings.kis_target_market = "us"
    repo = _ChatOrderRepo()
    repo.set_config("local", "investment_chat_account_markets", "us,kospi", "tester")
    captured: dict[str, object] = {}

    def fake_default_registry(repo, settings, *, tenant_id="local"):
        _ = repo, tenant_id
        captured["kis_target_market"] = settings.kis_target_market
        return ToolRegistry([])

    monkeypatch.setattr(chat_registry, "build_default_registry", fake_default_registry)
    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=None,
    )

    assert captured["kis_target_market"] == "us,kospi"


def test_build_investment_chat_agent_injects_tool_memory_for_request_tenant(monkeypatch) -> None:
    from arena.agents.investment_chat import factory
    from arena.memory.query_builders import MemoryQuerySpec

    settings = load_settings()
    repo = _ChatOrderRepo()
    registry = ToolRegistry(
        [
            ToolEntry(
                tool_id="recommend_opportunities",
                name="recommend_opportunities",
                description="read tool",
                category="quant",
                callable=lambda top_n=8: {"top_n": top_n},
            )
        ]
    )

    class _VectorStore:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def search_similar_memories(self, **kwargs):
            self.calls.append(dict(kwargs))
            return [
                {
                    "event_id": "mem-chat-1",
                    "summary": "AAPL 급등 후 추격매수보다 분할 접근이 나았다.",
                    "importance_score": 0.82,
                }
            ]

    class _MemoryStore:
        instances: list["_MemoryStore"] = []

        def __init__(self, *, repo, trading_mode, memory_policy):
            self.repo = repo
            self.trading_mode = trading_mode
            self.memory_policy = memory_policy
            self.vector_store = _VectorStore()
            self.__class__.instances.append(self)

        def _tenant(self) -> str:
            return self.repo.resolve_tenant_id()

    captured: dict[str, object] = {}

    def fake_build_tool_wrapper(
        entry,
        *,
        settings,
        agent_id,
        tool_events,
        update_candidate_ledger,
        search_tool_memories,
        apply_tool_schema_metadata,
    ):
        _ = settings, tool_events, update_candidate_ledger, apply_tool_schema_metadata
        captured["agent_id"] = agent_id

        def wrapped():
            return search_tool_memories(
                MemoryQuerySpec(
                    tool_name="recommend_opportunities",
                    key_type="ticker",
                    keys=("AAPL",),
                    query="AAPL opportunity",
                )
            )

        wrapped.__name__ = str(entry.name)
        return wrapped

    monkeypatch.setattr(factory, "MemoryStore", _MemoryStore, raising=False)
    monkeypatch.setattr(factory, "build_tool_wrapper", fake_build_tool_wrapper)
    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="czxnms",
        registry=registry,
    )

    memories = agent.tools[0]()

    assert captured["agent_id"] == "investment_chat"
    assert memories == [
        {
            "summary": "AAPL 급등 후 추격매수보다 분할 접근이 나았다.",
            "importance_score": 0.82,
        }
    ]
    store = _MemoryStore.instances[0]
    assert store.vector_store.calls[0]["agent_id"] == "investment_chat"
    assert store.vector_store.calls[0]["tenant_id"] == "czxnms"
    assert repo.resolve_tenant_id() == "local"


def test_build_investment_chat_agent_uses_stored_chat_agent_config(monkeypatch) -> None:
    from arena.agents.investment_chat import factory

    settings = load_settings()
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "investment_chat_config",
        json.dumps(
            {
                "provider": "gpt",
                "model": "gpt-5.5",
                "llm_params": {"reasoning_effort": "high", "verbosity": "low"},
            }
        ),
        "seed",
    )
    captured: dict[str, object] = {}

    def fake_resolve_model(provider, settings, *, model_override="", llm_params=None):
        captured["provider"] = provider
        captured["model_override"] = model_override
        captured["llm_params"] = dict(llm_params or {})
        return "fake-model"

    monkeypatch.setattr(factory, "_resolve_model", fake_resolve_model)
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=ToolRegistry([]),
    )

    assert captured["provider"] == "gpt"
    assert captured["model_override"] == "gpt-5.5"
    assert captured["llm_params"] == {"reasoning_effort": "high", "verbosity": "low"}


def test_build_investment_chat_agent_applies_stored_chat_tool_filter(monkeypatch) -> None:
    from arena.agents.investment_chat import factory

    settings = load_settings()
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "investment_chat_config",
        json.dumps({"disabled_tools": ["recommend_opportunities"]}),
        "seed",
    )
    registry = ToolRegistry(
        [
            ToolEntry(
                tool_id="recommend_opportunities",
                name="recommend_opportunities",
                description="read tool",
                category="quant",
                callable=lambda top_n=8: {"top_n": top_n},
            )
        ]
    )

    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=registry,
    )

    tool_names = {getattr(tool, "__name__", "") for tool in agent.tools}
    assert "recommend_opportunities" not in tool_names
    assert "get_account_snapshot" in tool_names


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
