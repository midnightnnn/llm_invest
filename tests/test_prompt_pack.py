from __future__ import annotations

from arena.prompts.prompt_pack import PromptPack
from arena.tools.registry import ToolEntry, ToolRegistry


def test_prompt_pack_renders_explore_prompt_from_single_entrypoint() -> None:
    prompt = PromptPack.render_decision_prompt(
        {
            "cycle_phase": "explore",
            "share_explore_summary": False,
            "portfolio": {"cash_krw": 1000},
        },
        [],
        max_tool_calls=7,
    )

    assert "## explore phase 규칙" in prompt
    assert '"explore_status": "complete"' in prompt
    assert "Context payload JSON" in prompt
    assert '"max_tool_calls": 7' in prompt


def test_legacy_agent_prompt_pack_import_stays_compatible() -> None:
    from arena.agents.prompts.prompt_pack import PromptPack as LegacyPromptPack

    assert LegacyPromptPack is PromptPack


def test_prompt_pack_loads_text_templates_from_central_package() -> None:
    from arena.prompts.loader import prompt_path

    assert prompt_path("adk", "core_prompt.txt").exists()
    assert prompt_path("adk", "system_prompt.txt").exists()
    assert prompt_path("adk", "explore_shared_format.txt").exists()
    assert prompt_path("adk", "explore_solo_format.txt").exists()
    assert prompt_path("adk", "execution_format.txt").exists()
    assert prompt_path("adk", "board_format.txt").exists()
    assert prompt_path("investment_chat", "system_prompt.txt").exists()
    assert "{agent_id}" in PromptPack.file_core_prompt()
    assert "적극적인 포트폴리오 관리" in PromptPack.file_user_prompt_default()


def test_prompt_pack_renders_resume_and_board_prompts() -> None:
    resume = PromptPack.render_resume_prompt(
        {
            "board_context": "peer note",
            "order_budget": {"max_buy_notional_krw": 100},
            "risk_policy": {"max_position_ratio": 0.5},
        },
        analysis_funnel={"analyzed_held": 2},
        max_tool_events=5,
    )
    board = PromptPack.render_board_prompt("주문 없음")

    assert resume.startswith("cycle_phase: execution")
    assert "이전 explore 단계의 분석" in resume
    assert "## 주문 규칙" in resume
    assert '"max_tool_calls": 5' in resume
    assert board.startswith("cycle_phase: board")
    assert "주문 없음" in board


def test_execution_prompt_describes_ontology_friendly_order_rationale() -> None:
    prompt = PromptPack.render_decision_prompt(
        {
            "cycle_phase": "execution",
            "portfolio": {"cash_krw": 1000},
            "order_budget": {"max_buy_notional_krw": 1000},
        },
        [],
        max_tool_calls=5,
    )

    assert "ontology-friendly investment memo" in prompt
    assert "explicit ticker names" in prompt
    assert "catalyst/risk/thesis/outcome" in prompt
    assert "memory/thesis summary" not in prompt
    assert "generic placeholders" not in prompt
    assert "2-4" not in prompt


def test_execution_prompt_requires_quantity_based_orders() -> None:
    prompt = PromptPack.render_decision_prompt(
        {
            "cycle_phase": "execution",
            "portfolio": {"cash_krw": 1000},
            "order_budget": {"max_buy_notional_krw": 1000},
        },
        [],
        max_tool_calls=5,
    )

    assert "quantity" in prompt
    assert "target_weight" not in prompt
    assert "sell_ratio" not in prompt


def test_prompt_pack_builds_tool_catalog_payload_from_registry() -> None:
    registry = ToolRegistry(
        [
            ToolEntry(
                tool_id="scratch_run_python",
                name="scratch_run_python",
                description="Temporary Python scratch workspace.",
                category="analysis",
                tier="core",
                callable=lambda code: {"status": "ok"},
            ),
            ToolEntry(
                tool_id="disabled_tool",
                name="disabled_tool",
                description="Hidden tool.",
                category="other",
                callable=lambda: {},
            ),
        ]
    )

    payload = PromptPack.tool_catalog_payload(
        registry,
        disabled_tool_ids={"disabled_tool"},
        mcp_toolset_count=1,
    )

    assert payload[0] == {
        "tool_id": "scratch_run_python",
        "name": "scratch_run_python",
        "category": "analysis",
        "tier": "core",
        "description": "Temporary Python scratch workspace.",
    }
    assert [row["tool_id"] for row in payload] == ["scratch_run_python", "mcp_toolsets"]


def test_prompt_pack_renders_investment_chat_instruction() -> None:
    prompt = PromptPack.render_investment_chat_instruction(
        tenant_id="MidNightNnN",
        provider="gpt",
        model_id="gpt-5.5",
    )

    assert "tenant 'midnightnnn'" in prompt
    assert "submit_order_with_confirmation" in prompt
    assert "validate_order_draft" in prompt
    assert "get_order_approval_status" in prompt
    assert "get_trade_history" in prompt
    assert "scope='agent_sleeve'" in prompt
    assert "user+investment_chat judgment" in prompt
    assert "Default to Korean" in prompt
    assert "approval card" in prompt
    assert "Confirmed 체크박스" in prompt
    assert "Do not ask the user to type CONFIRM" in prompt
    assert "exact confirmation phrase" not in prompt
    assert "memory/thesis summaries" not in prompt
    assert "generic placeholders" not in prompt
    assert "2-4" not in prompt
