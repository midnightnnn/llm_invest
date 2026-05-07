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
