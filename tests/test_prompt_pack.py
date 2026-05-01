from __future__ import annotations

from arena.agents.prompts.prompt_pack import PromptPack
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
