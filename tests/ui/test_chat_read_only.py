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
