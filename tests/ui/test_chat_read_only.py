from __future__ import annotations

from arena.prompts.prompt_pack import PromptPack


def test_render_investment_chat_instruction_default_no_read_only_notice() -> None:
    text = PromptPack.render_investment_chat_instruction(
        tenant_id="local",
        provider="gpt",
        model_id="gpt-5.5",
    )
    assert "보기 전용" not in text


def test_render_investment_chat_instruction_read_only_appends_notice() -> None:
    text = PromptPack.render_investment_chat_instruction(
        tenant_id="local",
        provider="gpt",
        model_id="gpt-5.5",
        read_only=True,
    )
    assert "보기 전용" in text
    assert "주문" in text and "설정" in text  # mentions both blocked tool families


def test_build_chat_tool_entries_default_includes_order_and_config(monkeypatch) -> None:
    from arena.agents.investment_chat import tools as chat_tools
    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    settings = load_settings()
    repo = _DummyRepo()
    entries = chat_tools.build_chat_tool_entries(
        repo=repo,
        settings=settings,
        tenant_id="local",
    )
    names = {str(e.name or e.tool_id or "") for e in entries}
    assert "submit_approved_order" in names or any("submit" in n for n in names)
    assert any("config" in n.lower() or "approve_config" in n.lower() for n in names)


def test_build_chat_tool_entries_read_only_strips_order_and_config() -> None:
    from arena.agents.investment_chat import tools as chat_tools
    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    settings = load_settings()
    repo = _DummyRepo()
    entries = chat_tools.build_chat_tool_entries(
        repo=repo,
        settings=settings,
        tenant_id="local",
        read_only=True,
    )
    names = {str(e.name or e.tool_id or "") for e in entries}
    # No order submission, no config approval/apply tools.
    assert "submit_approved_order" not in names
    assert not any("approve_config" in n.lower() for n in names)
    assert not any("apply_config" in n.lower() for n in names)


def test_build_chat_registry_read_only_strips_order_and_config() -> None:
    from arena.agents.investment_chat.registry import build_chat_registry
    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    settings = load_settings()
    repo = _DummyRepo()
    registry = build_chat_registry(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=None,
        read_only=True,
    )
    tool_ids = {str(entry.tool_id or "").lower() for entry in registry.list_entries()}
    assert "submit_approved_order" not in tool_ids
    assert not any("approve_config" in tid for tid in tool_ids)
    assert not any("apply_config" in tid for tid in tool_ids)


def test_build_investment_chat_agent_read_only_excludes_write_tools(monkeypatch) -> None:
    from types import SimpleNamespace

    from arena.agents.investment_chat import factory
    from arena.config import load_settings
    from tests.ui.helpers import _DummyRepo

    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    settings = load_settings()
    repo = _DummyRepo()
    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=None,
        read_only=True,
    )

    # Top-level agent is the router; sub-agents do the real work.
    sub_agents = {sub.name: sub for sub in (agent.sub_agents or [])}
    advisor = sub_agents.get(factory.ADVISOR_AGENT_NAME)
    utility = sub_agents.get(factory.UTILITY_AGENT_NAME)
    assert advisor is not None and utility is not None

    advisor_names = {getattr(t, "__name__", "") for t in advisor.tools}
    assert "submit_approved_order" not in advisor_names
    assert not any("approve_config" in n.lower() for n in advisor_names)
    assert not any("apply_config" in n.lower() for n in advisor_names)
    assert "보기 전용" in advisor.instruction

    utility_names = {getattr(t, "__name__", "") for t in utility.tools}
    assert "submit_approved_order" not in utility_names
    assert not any("propose_" in n.lower() and "config_change" in n.lower() for n in utility_names)
    assert not any("apply_config" in n.lower() for n in utility_names)
