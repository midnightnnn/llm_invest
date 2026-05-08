from __future__ import annotations

import json
from types import SimpleNamespace

from arena.config import load_settings
from arena.tools.registry import ToolEntry, ToolRegistry
from tests.ui.helpers import _DummyRepo
from tests.ui.investment_chat_helpers import (
    _ChatOrderRepo,
    _chat_advisor_agent,
    _chat_tool_names,
    _chat_utility_agent,
)


def test_investment_chat_factory_delegates_tool_implementations() -> None:
    import inspect

    from arena.agents.investment_chat import factory

    source = inspect.getsource(factory)

    assert "def get_account_snapshot(" not in source
    assert "def validate_order_draft(" not in source
    assert "def submit_approved_order(" not in source


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

    advisor = _chat_advisor_agent(agent)
    tool_names = _chat_tool_names(advisor)
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
    assert "live" not in advisor.instruction.lower()


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

    assert captured["kis_target_market"] == "us,kospi,kosdaq"


def test_investment_chat_ignores_legacy_single_market_chat_scope(monkeypatch) -> None:
    from arena.agents.investment_chat import factory
    from arena.agents.investment_chat import registry as chat_registry

    settings = load_settings()
    settings.kis_target_market = "us"
    repo = _ChatOrderRepo()
    repo.set_config("local", "investment_chat_account_markets", "us", "legacy")
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

    assert captured["kis_target_market"] == "us,kospi,kosdaq"


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

    advisor = _chat_advisor_agent(agent)
    memories = advisor.tools[0]()

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
    captured: list[dict[str, object]] = []

    def fake_resolve_model(provider, settings, *, model_override="", llm_params=None):
        captured.append(
            {
                "provider": provider,
                "model_override": model_override,
                "llm_params": dict(llm_params or {}),
            }
        )
        return f"{provider}:{model_override}"

    monkeypatch.setattr(factory, "_resolve_model", fake_resolve_model)
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=ToolRegistry([]),
    )

    assert {
        (item["provider"], item["model_override"], tuple(sorted(item["llm_params"].items())))
        for item in captured
    } >= {
        ("gpt", "gpt-5.4-mini", ()),
        ("gpt", "gpt-5.5", (("reasoning_effort", "high"), ("verbosity", "low"))),
    }


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

    advisor = _chat_advisor_agent(agent)
    tool_names = _chat_tool_names(advisor)
    assert "recommend_opportunities" not in tool_names
    assert "get_account_snapshot" in tool_names


def test_build_investment_chat_agent_builds_cheap_router_tree(monkeypatch) -> None:
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
                "llm_params": {"reasoning_effort": "high"},
                "model_routing": {
                    "cheap_model_by_provider": {"gpt": "gpt-5.4-mini"},
                    "router_llm_params": {"verbosity": "low"},
                },
            }
        ),
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
    resolved: list[dict[str, object]] = []

    def fake_resolve_model(provider, settings, *, model_override="", llm_params=None):
        resolved.append(
            {
                "provider": provider,
                "model_override": model_override,
                "llm_params": dict(llm_params or {}),
            }
        )
        return f"{provider}:{model_override}"

    monkeypatch.setattr(factory, "_resolve_model", fake_resolve_model)
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=registry,
    )

    advisor = _chat_advisor_agent(agent)
    utility = _chat_utility_agent(agent)
    advisor_tools = _chat_tool_names(advisor)
    utility_tools = _chat_tool_names(utility)

    assert agent.name == "investment_chat"
    assert agent.model == "gpt:gpt-5.4-mini"
    assert getattr(agent, "tools", []) == []
    assert {child.name for child in agent.sub_agents} == {"investment_chat_advisor", "investment_chat_utility"}
    assert advisor.model == "gpt:gpt-5.5"
    assert utility.model == "gpt:gpt-5.4-mini"
    assert advisor.disallow_transfer_to_parent is True
    assert utility.disallow_transfer_to_parent is True
    assert advisor.disallow_transfer_to_peers is False
    assert utility.disallow_transfer_to_peers is False
    assert {"recommend_opportunities", "submit_order_with_confirmation", "validate_order_draft"}.issubset(
        advisor_tools
    )
    assert {"get_account_snapshot", "get_trade_history", "get_order_approval_status"}.issubset(utility_tools)
    assert "submit_order_with_confirmation" not in utility_tools
    assert "validate_order_draft" not in utility_tools
    assert "recommend_opportunities" not in utility_tools
    assert {
        (item["provider"], item["model_override"], tuple(sorted(item["llm_params"].items())))
        for item in resolved
    } >= {
        ("gpt", "gpt-5.4-mini", (("verbosity", "low"),)),
        ("gpt", "gpt-5.5", (("reasoning_effort", "high"),)),
    }


def test_build_investment_chat_agent_defaults_claude_cheap_router_to_haiku(monkeypatch) -> None:
    from arena.agents.investment_chat import factory

    settings = load_settings()
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "investment_chat_config",
        json.dumps({"provider": "claude", "model": "claude-sonnet-4-6"}),
        "seed",
    )

    monkeypatch.setattr(
        factory,
        "_resolve_model",
        lambda provider, settings, *, model_override="", llm_params=None: f"{provider}:{model_override}",
    )
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=ToolRegistry([]),
    )

    assert agent.model == "claude:claude-haiku-4-5-20251001"
    assert _chat_advisor_agent(agent).model == "claude:claude-sonnet-4-6"


def test_build_investment_chat_agent_uses_prompt_pack_for_role_prompts(monkeypatch) -> None:
    from arena.agents.investment_chat import factory

    settings = load_settings()
    repo = _ChatOrderRepo()
    captured: dict[str, dict[str, object]] = {}

    def fake_advisor_instruction(**kwargs):
        captured["advisor"] = dict(kwargs)
        return "ADVISOR FILE PROMPT"

    def fake_router_instruction(**kwargs):
        captured["router"] = dict(kwargs)
        return "ROUTER FILE PROMPT"

    def fake_utility_instruction(**kwargs):
        captured["utility"] = dict(kwargs)
        return "UTILITY FILE PROMPT"

    monkeypatch.setattr(
        factory.PromptPack,
        "render_investment_chat_instruction",
        staticmethod(fake_advisor_instruction),
    )
    monkeypatch.setattr(
        factory.PromptPack,
        "render_investment_chat_router_instruction",
        staticmethod(fake_router_instruction),
    )
    monkeypatch.setattr(
        factory.PromptPack,
        "render_investment_chat_utility_instruction",
        staticmethod(fake_utility_instruction),
    )
    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=ToolRegistry([]),
        provider="gpt",
        model_override="gpt-5.5",
    )

    assert agent.instruction == "ROUTER FILE PROMPT"
    assert _chat_advisor_agent(agent).instruction == "ADVISOR FILE PROMPT"
    assert _chat_utility_agent(agent).instruction == "UTILITY FILE PROMPT"
    assert captured["advisor"]["utility_agent_name"] == "investment_chat_utility"
    assert captured["router"]["advisor_agent_name"] == "investment_chat_advisor"
    assert captured["router"]["utility_agent_name"] == "investment_chat_utility"
    assert captured["utility"]["advisor_agent_name"] == "investment_chat_advisor"
