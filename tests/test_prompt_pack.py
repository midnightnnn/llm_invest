from __future__ import annotations

import json

from arena.prompts.loader import prompt_path
from arena.prompts.prompt_pack import PromptPack
from arena.tools.registry import ToolEntry, ToolRegistry


def _decision_payload_from_prompt(prompt: str) -> dict:
    marker = "Context payload JSON"
    json_start = prompt.index("{", prompt.index(marker))
    return json.loads(prompt[json_start:])


def _json_suffix_from_text(text: str) -> dict:
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            return json.loads(text[idx:])
        except json.JSONDecodeError:
            continue
    raise AssertionError("JSON suffix not found")


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


def test_execution_prompt_schema_requests_structured_thesis_fields() -> None:
    prompt = PromptPack.render_decision_prompt(
        {
            "cycle_phase": "execution",
            "portfolio": {"cash_krw": 1000},
            "market_context": [],
        },
        [],
        max_tool_calls=7,
    )

    assert '"thesis_core"' in prompt
    assert '"supporting_factors"' in prompt
    assert '"risk_factors"' in prompt
    assert '"invalidation_conditions"' in prompt
    assert "catalyst|event|indicator|metric|macro_factor|regime|strategy_tag|thesis|scenario" in prompt
    assert "rationale은 사람이 읽는 주문 판단 요약문" in prompt


def test_decision_payload_includes_runtime_clock_when_present() -> None:
    context = {
        "cycle_phase": "execution",
        "_runtime_clock": {"now_kst": "2026-05-15T15:25:14+09:00"},
    }

    payload = PromptPack.decision_payload(context, max_tool_calls=10)

    assert payload["_runtime_clock"] == {
        "now_kst": "2026-05-15T15:25:14+09:00",
    }
    assert "supervisor" not in str(payload).lower()
    assert "deadline" not in str(payload).lower()


def test_execution_payload_omits_empty_context_fields() -> None:
    payload = PromptPack.decision_payload(
        {
            "cycle_phase": "execution",
            "performance_context": "",
            "active_thesis_context": "",
            "memory_context": "",
            "board_context": "",
            "market_context": [],
            "research_context": "",
            "portfolio": {},
            "ticker_names": {},
            "risk_policy": {},
            "order_budget": {},
            "analysis_funnel": {},
            "candidate_cases": [],
            "decision_frame": "",
            "investment_style_context": "",
        },
        max_tool_calls=10,
    )

    assert payload == {
        "cycle_phase": "execution",
        "tool_budget": {
            "max_tool_calls": 10,
            "final_json_before_exhaustion": True,
        },
    }


def test_prompt_pack_uses_phase_specific_active_thesis_projection() -> None:
    context = {
        "active_thesis_context": "Active Thesis:\n- full projection",
        "active_thesis_context_explore": "Active Thesis:\n- explore projection",
        "active_thesis_context_execution": "Active Thesis:\n- execution projection",
        "portfolio": {"cash_krw": 1000},
        "market_context": [],
    }

    explore_prompt = PromptPack.render_decision_prompt(
        {**context, "cycle_phase": "explore"},
        [],
        max_tool_calls=7,
    )
    execution_prompt = PromptPack.render_decision_prompt(
        {**context, "cycle_phase": "execution"},
        [],
        max_tool_calls=7,
    )

    explore_payload = _decision_payload_from_prompt(explore_prompt)
    execution_payload = _decision_payload_from_prompt(execution_prompt)
    assert explore_payload["active_thesis_context"] == "Active Thesis:\n- explore projection"
    assert execution_payload["active_thesis_context"] == "Active Thesis:\n- execution projection"


def test_explore_payload_uses_positions_brief_and_omits_raw_position_duplicates() -> None:
    prompt = PromptPack.render_decision_prompt(
        {
            "cycle_phase": "explore",
            "share_explore_summary": True,
            "positions_brief": ["AAPL qty=2 weight=80.0% avg=$70.00 price=$80.00"],
            "portfolio": {"positions": {"AAPL": {"quantity": 2, "market_price_krw": 120_000}}},
            "market_context": [{"ticker": "AAPL", "close_price_native": 80.0}],
            "performance_context": "NAV 300,000 KRW",
            "active_thesis_context": "",
            "memory_context": "",
            "board_context": "",
            "research_context": "- [AAPL] Research stays intact.",
            "ticker_names": {},
            "risk_policy": {"max_position_ratio": 0.35},
            "order_budget": {"max_buy_notional_krw": 30_000},
            "candidate_cases": [],
            "decision_frame": "",
            "investment_style_context": "",
            "relation_context": "",
            "graph_context": "",
        },
        [],
        max_tool_calls=7,
    )

    payload = _decision_payload_from_prompt(prompt)

    assert payload["positions_brief"] == ["AAPL qty=2 weight=80.0% avg=$70.00 price=$80.00"]
    assert "portfolio" not in payload
    assert "market_context" not in payload
    assert "board_context" not in payload
    assert "ticker_names" not in payload
    assert "candidate_cases" not in payload
    assert "decision_frame" not in payload
    assert "memory_context" not in payload
    assert "relation_context" not in payload
    assert "graph_context" not in payload
    assert payload["research_context"] == "- [AAPL] Research stays intact."
    assert payload["risk_policy"] == {"max_position_ratio": 0.35}
    assert payload["order_budget"] == {"max_buy_notional_krw": 30_000}


def test_explore_payload_preserves_nonheld_market_rows_when_positions_brief_exists() -> None:
    prompt = PromptPack.render_decision_prompt(
        {
            "cycle_phase": "explore",
            "positions_brief": ["AAPL qty=2 weight=80.0% avg=$70.00 price=$80.00"],
            "portfolio": {"positions": {"AAPL": {"quantity": 2}}},
            "market_context": [
                {"ticker": "AAPL", "close_price_native": 80.0},
                {"ticker": "MSFT", "close_price_native": 210.0},
            ],
        },
        [],
        max_tool_calls=7,
    )

    payload = _decision_payload_from_prompt(prompt)

    assert payload["positions_brief"] == ["AAPL qty=2 weight=80.0% avg=$70.00 price=$80.00"]
    assert payload["market_context"] == [{"ticker": "MSFT", "close_price_native": 210.0}]
    assert "portfolio" not in payload


def test_explore_payload_compacts_budget_policy_funnel_and_tool_budget() -> None:
    prompt = PromptPack.render_decision_prompt(
        {
            "cycle_phase": "explore",
            "analysis_funnel": {
                "discovered_candidates": 0,
                "screened_only_candidates": 0,
                "fully_analyzed_candidates": 0,
                "analyzed_held_positions": 0,
                "ordered_candidates": 0,
                "intended_candidates": 0,
                "executed_candidates": 0,
                "skipped_candidates": 0,
            },
            "risk_policy": {
                "max_order_krw": 100_000_000.0,
                "max_daily_turnover_ratio": 10.0,
                "max_position_ratio": 1.0,
                "min_cash_buffer_ratio": 0.1,
                "ticker_cooldown_seconds": 120,
                "max_daily_orders": None,
                "max_daily_orders_unlimited": True,
                "single_share_buy_exception_enabled": True,
                "sleeve_capital_krw": 5_000_000.0,
            },
            "order_budget": {
                "display_currency": "KRW",
                "cash": 749_133.5890466672,
                "cash_krw": 749_133.5890466672,
                "min_cash_required": 484_888.5189046667,
                "min_cash_required_krw": 484_888.5189046667,
                "max_buy_notional_by_cash": 264_245.0701420005,
                "max_buy_notional_by_cash_krw": 264_245.0701420005,
                "daily_turnover_limit": 48_488_851.89046667,
                "remaining_turnover": 48_200_986.81046667,
                "remaining_turnover_krw": 48_200_986.81046667,
                "max_order": 100_000_000.0,
                "max_order_krw": 100_000_000.0,
                "max_buy_notional_by_sleeve": 264_245.0701420005,
                "max_buy_notional_by_sleeve_krw": 264_245.0701420005,
                "max_buy_notional": 264_245.0701420005,
                "max_buy_notional_krw": 264_245.0701420005,
                "today_intents": 1,
                "daily_orders_cap": None,
                "remaining_daily_orders": None,
            },
        },
        [],
        max_tool_calls=7,
    )

    payload = _decision_payload_from_prompt(prompt)

    assert payload["analysis_funnel"] == {"status": "none"}
    assert payload["tool_budget"] == {"max_tool_calls": 7, "final_json_before_exhaustion": True}
    assert payload["risk_policy"] == {
        "max_position_ratio": 1.0,
        "min_cash_buffer_ratio": 0.1,
        "ticker_cooldown_seconds": 120,
        "single_share_buy_exception_enabled": True,
    }
    assert payload["order_budget"] == {
        "cash_krw": 749_134,
        "min_cash_required_krw": 484_889,
        "max_buy_notional_krw": 264_245,
        "buy_caps_krw": {
            "cash": 264_245,
            "sleeve": 264_245,
            "turnover": 48_200_987,
            "order": 100_000_000,
        },
        "today_intents": 1,
        "daily_orders": "unlimited",
    }
    assert "cash" not in payload["order_budget"]
    assert "max_buy_notional" not in payload["order_budget"]
    assert "remaining_turnover" not in payload["order_budget"]
    assert "max_daily_orders_unlimited" not in payload["risk_policy"]
    assert "sleeve_capital_krw" not in payload["risk_policy"]


def test_execution_payload_keeps_raw_portfolio_and_market_context() -> None:
    prompt = PromptPack.render_decision_prompt(
        {
            "cycle_phase": "execution",
            "positions_brief": ["AAPL compact line"],
            "portfolio": {"positions": {"AAPL": {"quantity": 2}}},
            "market_context": [{"ticker": "AAPL", "close_price_native": 80.0}],
            "order_budget": {"max_buy_notional_krw": 30_000},
        },
        [],
        max_tool_calls=5,
    )

    payload = _decision_payload_from_prompt(prompt)

    assert payload["portfolio"] == {"positions": {"AAPL": {"quantity": 2}}}
    assert payload["market_context"] == [{"ticker": "AAPL", "close_price_native": 80.0}]
    assert "positions_brief" not in payload


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
    assert prompt_path("investment_chat", "advisor_prompt.txt").exists()
    assert not prompt_path("investment_chat", "system_prompt.txt").exists()
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
    assert '"max_tool_calls": 5' not in resume
    assert '"order_budget"' not in resume
    assert '"risk_policy"' not in resume
    assert board.startswith("cycle_phase: board")
    assert "주문 없음" in board


def test_resume_prompt_omits_phase_duplicate_payload_and_keeps_runtime_clock() -> None:
    resume = PromptPack.render_resume_prompt(
        {
            "board_context": "",
            "_runtime_clock": {"now_kst": "2026-05-15T15:26:18+09:00"},
            "order_budget": {
                "display_currency": "KRW",
                "cash": 749_133.5890466672,
                "cash_krw": 749_133.5890466672,
                "min_cash_required": 484_888.5189046667,
                "min_cash_required_krw": 484_888.5189046667,
                "max_buy_notional_by_cash": 264_245.0701420005,
                "max_buy_notional_by_cash_krw": 264_245.0701420005,
                "remaining_turnover": 48_200_986.81046667,
                "remaining_turnover_krw": 48_200_986.81046667,
                "max_order": 100_000_000.0,
                "max_order_krw": 100_000_000.0,
                "max_buy_notional_by_sleeve": 264_245.0701420005,
                "max_buy_notional_by_sleeve_krw": 264_245.0701420005,
                "max_buy_notional": 264_245.0701420005,
                "max_buy_notional_krw": 264_245.0701420005,
                "today_intents": 1,
                "daily_orders_cap": None,
                "remaining_daily_orders": None,
            },
            "risk_policy": {
                "max_order_krw": 100_000_000.0,
                "max_daily_turnover_ratio": 10.0,
                "max_position_ratio": 1.0,
                "min_cash_buffer_ratio": 0.1,
                "ticker_cooldown_seconds": 120,
                "max_daily_orders": None,
                "max_daily_orders_unlimited": True,
                "single_share_buy_exception_enabled": True,
                "sleeve_capital_krw": 5_000_000.0,
            },
            "candidate_cases": [],
            "decision_frame": "",
        },
        analysis_funnel={"discovered_nonheld": 0, "pending_nonheld": 0},
        max_tool_events=5,
    )

    payload = _json_suffix_from_text(resume)

    assert payload == {"_runtime_clock": {"now_kst": "2026-05-15T15:26:18+09:00"}}
    assert '"analysis_funnel"' not in resume
    assert '"tool_budget"' not in resume
    assert '"risk_policy"' not in resume
    assert '"order_budget"' not in resume
    assert '"candidate_cases"' not in resume
    assert '"decision_frame"' not in resume


def test_board_prompt_includes_runtime_clock_when_present() -> None:
    prompt = PromptPack.render_board_prompt(
        "주문 없음",
        runtime_clock={"now_kst": "2026-05-15T15:26:18+09:00"},
    )

    assert prompt.startswith("cycle_phase: board")
    assert '"_runtime_clock": {"now_kst": "2026-05-15T15:26:18+09:00"}' in prompt
    assert "주문 없음" in prompt


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

    assert "rationale은 사람이 읽는 주문 판단 요약문" in prompt
    assert "thesis_core, supporting_factors, risk_factors" in prompt
    assert "source-grounded relation triple" in prompt
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


def test_execution_prompt_requires_sell_lifecycle_strategy_ref() -> None:
    prompt = PromptPack.render_decision_prompt(
        {
            "cycle_phase": "execution",
            "portfolio": {"cash_krw": 1000},
            "order_budget": {"max_buy_notional_krw": 1000},
        },
        [],
        max_tool_calls=5,
    )

    assert "strategy_refs" in prompt
    assert "thesis_invalidated" in prompt
    assert "thesis_realized" in prompt
    assert "risk_reduction" in prompt
    assert "rebalancing" in prompt
    assert "thesis_invalidation" not in prompt


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
        utility_agent_name="investment_chat_utility",
    )

    assert "tenant 'midnightnnn'" in prompt
    assert "investment advisor sub-agent" in prompt
    assert "investment_chat_utility" in prompt
    assert "mixed or ambiguous" in prompt
    assert "Always consider the current account situation" in prompt
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
    assert not prompt_path("investment_chat", "advisor_routing_note.txt").exists()


def test_prompt_pack_renders_investment_chat_router_instruction() -> None:
    prompt = PromptPack.render_investment_chat_router_instruction(
        tenant_id="MidNightNnN",
        provider="gpt",
        advisor_model_id="gpt-5.5",
        cheap_model_id="gpt-5.4-mini",
        advisor_agent_name="investment_chat_advisor",
        utility_agent_name="investment_chat_utility",
    )

    assert prompt_path("investment_chat", "router_prompt.txt").exists()
    assert "Tenant: midnightnnn" in prompt
    assert "Provider: gpt" in prompt
    assert "Advisor model: gpt-5.5" in prompt
    assert "Utility model: gpt-5.4-mini" in prompt
    assert "investment_chat_advisor" in prompt
    assert "investment_chat_utility" in prompt
    assert "Deterministic model/tool/sleeve-capital settings changes" in prompt
    assert "asks you to choose an allocation" in prompt
    assert "Do not provide substantive investment analysis in the router" in prompt


def test_prompt_pack_renders_investment_chat_utility_instruction() -> None:
    prompt = PromptPack.render_investment_chat_utility_instruction(
        tenant_id="MidNightNnN",
        provider="claude",
        model_id="claude-haiku-4-5-20251001",
        advisor_agent_name="investment_chat_advisor",
    )

    assert prompt_path("investment_chat", "utility_prompt.txt").exists()
    assert "Tenant: midnightnnn" in prompt
    assert "Provider: claude" in prompt
    assert "Model: claude-haiku-4-5-20251001" in prompt
    assert "configuration-change drafts" in prompt
    assert 'capital_allocation_mode="add_krw"' in prompt
    assert 'capital_allocation_mode="fixed_krw"' in prompt
    assert "routed investment chat" in prompt
    assert "appropriate specialist agent" in prompt
    assert "provider-level credentials" in prompt
    assert "portfolio/risk diagnosis" not in prompt
    assert "ticker discussion" not in prompt
    assert "Do not transfer deterministic configuration changes" in prompt
    assert "Do not give investment advice" in prompt
    assert "investment_chat_advisor" in prompt
