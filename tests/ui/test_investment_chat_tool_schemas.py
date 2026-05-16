from __future__ import annotations

import json
from types import SimpleNamespace

from arena.config import load_settings
from tests.ui.investment_chat_helpers import (
    _ChatOrderRepo,
    _FakeExecutionMemory,
    _build_fake_chat_agent,
    _chat_advisor_agent,
)


def test_investment_chat_wrapped_adk_confirmation_tool_builds_declaration(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "submit_order_with_confirmation")

    declaration = FunctionTool(tool)._get_declaration()

    assert declaration is not None
    assert declaration.name == "submit_order_with_confirmation"
    assert "tool_context" not in json.dumps(declaration.model_dump(), default=str)


def test_investment_chat_batch_order_confirmation_tool_builds_declaration(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "submit_order_batch_with_confirmation")

    declaration = FunctionTool(tool)._get_declaration()
    params = declaration.parameters.model_dump(mode="json", exclude_none=True)

    assert declaration is not None
    assert declaration.name == "submit_order_batch_with_confirmation"
    assert "tool_context" not in json.dumps(declaration.model_dump(), default=str)
    assert params["properties"]["orders"]["type"] == "ARRAY"
    assert params["properties"]["orders"]["items"]["type"] == "OBJECT"


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
        "list_chat_model_options",
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
    assert props["capital_allocation_mode"]["enum"] == [
        "unchanged",
        "fixed_krw",
        "add_krw",
        "account_percent",
        "whole_account",
    ]


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

    for tool in _chat_advisor_agent(agent).tools:
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
    tool = next(candidate for candidate in _chat_advisor_agent(agent).tools if candidate.__name__ == "optimize_portfolio")

    declaration = FunctionTool(tool)._get_declaration()
    params = declaration.parameters.model_dump(mode="json", exclude_none=True)

    assert "tickers" in params["required"]
    assert params["properties"]["tickers"]["items"]["type"] == "STRING"
    assert params["properties"]["strategy"]["enum"] == ["sharpe", "risk_parity", "forecast"]
    assert params["properties"]["forecast_mode"]["enum"] == ["default", "all", "stacked", "base", "balanced", "lgbm", "ridge", "avg"]


def test_chat_research_tool_schema_exposes_on_demand_refresh_and_category_enum(monkeypatch) -> None:
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
    tool = next(candidate for candidate in _chat_advisor_agent(agent).tools if candidate.__name__ == "get_research_briefing")

    declaration = FunctionTool(tool)._get_declaration()
    params = declaration.parameters.model_dump(mode="json", exclude_none=True)
    props = params["properties"]

    assert "refresh_missing" in props
    assert props["refresh_missing"]["type"] == "BOOLEAN"
    assert props["categories"]["items"]["enum"] == [
        "global_market",
        "geopolitical",
        "sector_trends",
        "sector",
    ]
    assert 'categories=["geopolitical"]' not in declaration.description
    assert 'tickers=["AAPL"]' not in declaration.description
