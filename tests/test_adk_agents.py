from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from arena.agents.adk_agents import (
    _ADKDecisionRunner,
    _apply_tool_schema_metadata,
    _compact_tool_result_for_prompt,
    _ContextTools,
    _is_retryable_adk_error,
    _load_disabled_tool_ids,
    _system_prompt,
    _user_prompt,
)
from arena.agents.adk_agent_flow import (
    explore_phase_output,
    extract_decision_payload,
    retry_policy_from_env,
)
from arena.agents.adk_decision_flow import (
    build_tool_summary_memory_record,
    parse_board_response,
    prepare_decision_prompt,
)
from arena.agents.adk_runner_bootstrap import build_tool_wrapper, resolve_max_tool_events, runner_identity
from arena.agents.adk_runner_runtime import AdkToolBudgetExceeded
from arena.config import AgentConfig, load_settings
from arena.models import OrderIntent, Side
from arena.tools.default_registry import build_default_registry
from arena.tools.registry import ToolEntry, ToolRegistry
from arena.tools.scratch_workspace import ScratchWorkspace


def test_apply_tool_schema_metadata_prefers_registry_description() -> None:
    def original_tool(ticker: str) -> dict[str, str]:
        """Original docstring that should not leak to the model."""
        return {"ticker": ticker}

    entry = ToolEntry(
        tool_id="screen_market",
        name="screen_market",
        description="Canonical registry description for the model schema.",
        category="quant",
        callable=original_tool,
    )

    wrapped = _apply_tool_schema_metadata(
        original_tool,
        entry=entry,
        sig=inspect.signature(original_tool),
    )

    assert wrapped.__name__ == "screen_market"
    assert wrapped.__doc__ == "Canonical registry description for the model schema."


def test_batch_default_tool_schema_preserves_required_fields_and_enums() -> None:
    from google.adk.tools.function_tool import FunctionTool

    from arena.agents.adk_tool_helpers import noop_search_tool_memories, noop_update_candidate_ledger

    settings = load_settings()
    repo = SimpleNamespace(get_config=lambda *args, **kwargs: "")
    registry = build_default_registry(repo, settings, tenant_id="local")
    scratch = ScratchWorkspace(agent_id="gpt", tenant_id="local", tool_events=[])
    registry.bind("scratch_run_python", scratch.run_python)

    def declaration(tool_id: str):
        entry = registry.get(tool_id)
        assert entry is not None
        assert entry.callable is not None
        wrapped = build_tool_wrapper(
            entry,
            settings=settings,
            agent_id="gpt",
            tool_events=[],
            update_candidate_ledger=noop_update_candidate_ledger,
            search_tool_memories=noop_search_tool_memories,
            apply_tool_schema_metadata=_apply_tool_schema_metadata,
        )
        return FunctionTool(wrapped)._get_declaration()

    screen_params = declaration("screen_market").parameters.model_dump(mode="json", exclude_none=True)
    assert screen_params["properties"]["bucket"]["enum"] == [
        "auto",
        "balanced",
        "momentum",
        "pullback",
        "recovery",
        "defensive",
        "value",
    ]
    assert screen_params["properties"]["sort_by"]["enum"] == [
        "none",
        "as_of_ts",
        "ret_20d",
        "ret_5d",
        "volatility_20d",
        "sentiment_score",
        "close_price_krw",
    ]
    assert screen_params["properties"]["order"]["enum"] == ["asc", "desc"]

    optimize_params = declaration("optimize_portfolio").parameters.model_dump(mode="json", exclude_none=True)
    assert "tickers" in optimize_params["required"]
    assert optimize_params["properties"]["strategy"]["enum"] == ["sharpe", "risk_parity", "forecast"]
    assert optimize_params["properties"]["forecast_mode"]["enum"] == [
        "default",
        "all",
        "stacked",
        "base",
        "balanced",
        "lgbm",
        "ridge",
        "avg",
    ]

    opportunity_params = declaration("recommend_opportunities").parameters.model_dump(mode="json", exclude_none=True)
    assert opportunity_params["properties"]["buckets"]["items"]["enum"] == ["momentum", "pullback", "recovery"]
    assert opportunity_params["properties"]["profiles"]["items"]["enum"] == [
        "aggressive",
        "balanced",
        "defensive",
        "value",
        "tactical",
        "tactical_leverage",
        "tactical_inverse",
        "tactical_hedge",
    ]

    scratch_params = declaration("scratch_run_python").parameters.model_dump(mode="json", exclude_none=True)
    assert "code" in scratch_params["required"]
    assert scratch_params["properties"]["inputs"]["type"] == "OBJECT"
    assert scratch_params["properties"]["inputs"]["nullable"] is True


class _RepoForPrompt:
    def __init__(self, value: str | None):
        self._value = value

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        _ = tenant_id, config_key
        return self._value


def test_system_prompt_uses_db_when_available() -> None:
    repo = _RepoForPrompt("You are a trading agent.")
    out = _system_prompt("test-agent", repo=repo, tenant_id="tenant-a")
    assert "test-agent" in out
    assert "You are a trading agent." in out


def test_system_prompt_uses_agent_config_override() -> None:
    repo = _RepoForPrompt("Global prompt from DB.")
    ac = AgentConfig(
        agent_id="custom",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        system_prompt="Custom per-agent prompt.",
    )
    out = _system_prompt("custom", repo=repo, tenant_id="tenant-a", agent_config=ac)
    assert "Custom per-agent prompt." in out
    # Should NOT include the DB global prompt
    assert "Global prompt from DB." not in out


class _RepoForTools:
    def __init__(self, disabled: str | None):
        self.disabled = disabled

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        _ = tenant_id, config_key
        return self.disabled


def test_load_disabled_tool_ids_uses_tool_id_tokens() -> None:
    repo = _RepoForTools('["fetch_reddit_sentiment","optimize_portfolio"]')
    out = _load_disabled_tool_ids(repo, "tenant-a")
    assert out == {"fetch_reddit_sentiment", "optimize_portfolio"}


def test_runner_identity_uses_agent_scoped_names() -> None:
    identity = runner_identity("gpt")

    assert identity.app_name == "llm_arena_gpt"
    assert identity.user_id == "arena"
    assert identity.session_id == "gpt_react"


def test_resolve_max_tool_events_clamps_invalid_and_high_values() -> None:
    settings = load_settings()
    settings.adk_max_tool_events = "oops"
    assert resolve_max_tool_events(settings) == 120

    settings.adk_max_tool_events = 999
    assert resolve_max_tool_events(settings) == 400


def test_prepare_decision_prompt_resume_reuses_session_and_includes_board_context() -> None:
    session_id, prompt, needs_new_session = prepare_decision_prompt(
        {
            "board_context": "peer conviction is rising",
            "order_budget": {"max_buy_notional_krw": 1_000_000},
            "risk_policy": {"max_position_ratio": 0.2},
            "decision_frame": "Compare opportunities against weakest holding.",
            "candidate_cases": [{"ticker": "MSFT", "case_for": "screened candidate"}],
        },
        default_universe=["AAPL"],
        phase="execution",
        base_session_id="gpt_react",
        max_tool_events=12,
        resume_session_id="resume_1",
        analysis_funnel={"pending_nonheld": 2},
    )

    assert session_id == "resume_1"
    assert needs_new_session is False
    assert "peer conviction is rising" in prompt
    assert "Compare opportunities against weakest holding." in prompt
    assert "screened candidate" in prompt
    assert '"max_tool_calls": 12' in prompt


def test_build_tool_summary_memory_record_keeps_token_usage_even_without_events() -> None:
    record = build_tool_summary_memory_record(
        [],
        registry=ToolRegistry([]),
        phase="explore",
        analysis_funnel={"discovered_nonheld": 0},
        cycle_id="cycle_1",
        token_usage={"llm_calls": 1, "prompt_tokens": 120},
    )

    assert record is not None
    summary, payload = record
    assert "ReAct tools used (explore): 0" in summary
    assert payload["token_usage"]["prompt_tokens"] == 120
    assert payload["analysis_funnel"]["discovered_nonheld"] == 0


def test_parse_board_response_raises_on_plain_text_body() -> None:
    with pytest.raises(Exception):
        parse_board_response("plain board body")


def test_retry_policy_from_env_clamps_extreme_values(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_ADK_RETRY_MAX", "99")
    monkeypatch.setenv("ARENA_ADK_RETRY_BACKOFF_SECONDS", "99")

    retry_limit, retry_delay = retry_policy_from_env()

    assert retry_limit == 4
    assert retry_delay == 10.0


def test_adk_own_timeout_is_not_retryable() -> None:
    assert _is_retryable_adk_error(TimeoutError("ADK coroutine timed out after 1530s")) is False
    assert _is_retryable_adk_error(TimeoutError("ADK tool-budget finalization timed out after 60s")) is False
    assert _is_retryable_adk_error(AdkToolBudgetExceeded("ADK tool budget exceeded after 121 tool calls")) is False
    assert _is_retryable_adk_error(RuntimeError("429 RESOURCE_EXHAUSTED")) is True
    assert _is_retryable_adk_error(RuntimeError("litellm.BadGatewayError: 502 Bad gateway")) is True


def test_user_prompt_omits_sleeve_state_payload() -> None:
    prompt = _user_prompt(
        {
            "cycle_phase": "execution",
            "portfolio": {"cash_krw": 0},
            "risk_policy": {},
            "order_budget": {"max_buy_notional_krw": 0.0},
            "sleeve_state": {"buy_blocked": True, "over_target": True},
            "analysis_funnel": {"discovered_nonheld": 3, "analyzed_nonheld": 1, "pending_nonheld": 2},
            "active_thesis_context": "Active Thesis:\n- [AAPL | open] compact thesis",
            "active_theses": [{"ticker": "AAPL", "payload_json": '{"raw": "large"}'}],
            "opportunity_working_set": [{"ticker": "TSLA", "status": "pending"}],
            "decision_frame": "Compare self-discovered opportunities against cash first.",
            "market_context": [{"ticker": "AAPL", "close": 123.45}],
            "research_context": "- [AAPL] New product cycle - Demand watchlist.",
            "relation_context": "Relation Hints:\n- contains ticker AAPL: prior risk lesson.",
            "graph_context": "Decision Paths:\n- AAPL prior entry connects to a later win.",
            "memory_context": "Portfolio Memory:\n- [AAPL | BUY] Keep this compressed lesson.",
            "memory_events": [{"summary": "Do not duplicate this raw memory summary."}],
        },
        default_universe=[],
        max_tool_calls=5,
    )
    marker = "Context payload JSON"
    json_start = prompt.index("{", prompt.index(marker))
    payload = json.loads(prompt[json_start:])
    assert "sleeve_state" not in payload
    assert payload["active_thesis_context"] == "Active Thesis:\n- [AAPL | open] compact thesis"
    assert "active_theses" not in payload
    assert payload["analysis_funnel"]["screened_only_candidates"] == 2
    assert "pending_nonheld" not in payload["analysis_funnel"]
    assert "opportunity_working_set" not in payload
    assert payload["candidate_cases"] == []
    assert payload["decision_frame"] == "Compare self-discovered opportunities against cash first."
    assert payload["market_context"] == [{"ticker": "AAPL", "close": 123.45}]
    assert payload["research_context"] == "- [AAPL] New product cycle - Demand watchlist."
    assert payload["relation_context"] == "Relation Hints:\n- contains ticker AAPL: prior risk lesson."
    assert payload["graph_context"] == "Decision Paths:\n- AAPL prior entry connects to a later win."
    assert payload["memory_context"] == "Portfolio Memory:\n- [AAPL | BUY] Keep this compressed lesson."
    assert "memory_events" not in payload
    assert "recent_memory_summaries" not in payload
    assert payload["tool_budget"]["max_tool_calls"] == 5


def test_prompt_context_sections_collects_prompt_details() -> None:
    sections = _ADKDecisionRunner._prompt_context_sections(
        {
            "portfolio": {"cash_krw": 1000},
            "market_features": [{"ticker": "AAPL", "close": 123.45}],
            "board_posts": [{"post_id": "board_1", "summary": "Hold watchlist"}],
            "research_context": "- [AAPL] New product cycle - Demand watchlist.",
            "relation_context": "Relation Hints:\n- contains ticker AAPL: prior risk lesson.",
            "graph_context": "Decision Paths:\n- AAPL prior entry connects to a later win.",
            "memory_context": "Memory:\n- Prefer staged entries.",
        }
    )

    assert sections["portfolio_context"] == {"cash_krw": 1000}
    assert sections["market_context"] == [{"ticker": "AAPL", "close": 123.45}]
    assert sections["board_context"] == [{"post_id": "board_1", "summary": "Hold watchlist"}]
    assert sections["research_context"] == "- [AAPL] New product cycle - Demand watchlist."
    assert sections["relation_context"] == "Relation Hints:\n- contains ticker AAPL: prior risk lesson."
    assert sections["graph_context"] == "Decision Paths:\n- AAPL prior entry connects to a later win."
    assert sections["memory_context"] == "Memory:\n- Prefer staged entries."


def test_extract_decision_payload_normalizes_non_list_orders() -> None:
    explore_summary, orders = extract_decision_payload(
        {
            "explore_summary": "  concise explore  ",
            "orders": {"ticker": "AAPL"},
        }
    )

    assert explore_summary == "concise explore"
    assert orders == []


def test_explore_phase_output_uses_distinct_tickers() -> None:
    out = explore_phase_output(
        agent_id="gpt",
        cycle_id="cycle_explore_1",
        explore_summary="summary",
        orders=[
            {"ticker": "AAPL"},
            {"ticker": "MSFT"},
            {"ticker": "AAPL"},
        ],
        share_summary=True,
    )

    assert out.intents == []
    assert out.board_post.title == "탐색 요약"
    assert out.board_post.body == "summary"
    assert out.board_post.explore_summary == "summary"
    assert out.board_post.tickers == ["AAPL", "MSFT"]


def test_compact_validate_order_draft_hides_manual_confirmation_phrase() -> None:
    out = _compact_tool_result_for_prompt(
        "validate_order_draft",
        {
            "status": "ok",
            "tenant_id": "local",
            "scope": "account",
            "target_agent_id": "investment_chat",
            "judgment_source": "user+investment_chat",
            "approval_token": "abc123",
            "required_confirmation": "CONFIRM abc123",
            "submission_status": "not_submitted",
            "approval_required": True,
            "notional_krw": 100000,
            "intent": {"ticker": "AAPL", "side": "BUY", "quantity": 1, "rationale": "test"},
            "risk": {"allowed": True, "reason": "ok", "policy_hits": []},
        },
    )

    assert out["approval_required"] is True
    assert out["approval_ui"] == "approval_card"
    assert out["submission_status"] == "not_submitted"
    assert out["intent"]["ticker"] == "AAPL"
    assert out["risk"] == {"allowed": True, "reason": "ok", "policy_hits": []}
    assert "approval_token" not in out
    assert "required_confirmation" not in out


def test_compact_tool_result_reddit_drops_url_and_trims_text() -> None:
    out = _compact_tool_result_for_prompt(
        "fetch_reddit_sentiment",
        [
            {
                "title": "AAPL sentiment is ripping higher on wallstreetbets and this title is intentionally very long",
                "subreddit": "wallstreetbets",
                "score": 123,
                "num_comments": 45,
                "created": "2026-03-14T00:00:00+00:00",
                "url": "https://reddit.com/r/x",
                "selftext_snippet": "x" * 400,
            }
        ],
        args={"ticker": "AAPL"},
    )

    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["subreddit"] == "wallstreetbets"
    assert "url" not in out[0]
    assert len(out[0]["selftext_snippet"]) <= 140


def test_compact_tool_result_technical_signals_multi_returns_summary_rows() -> None:
    out = _compact_tool_result_for_prompt(
        "technical_signals",
        {
            "tickers": ["AAPL", "MSFT"],
            "count": 2,
            "rows": [
                {
                    "ticker": "AAPL",
                    "price": 100.0,
                    "rsi_14": 61.2,
                    "rsi_state": "neutral",
                    "macd": {"line": 1.0, "signal": 0.8, "hist": 0.2, "state": "bullish"},
                    "moving_averages": {"sma_20": 98.0, "sma_50": 95.0, "price_vs_sma20": 0.0204},
                    "bollinger_20_2": {"upper": 102.0, "mid": 98.0, "lower": 94.0, "state": "inside_bands"},
                    "trend_state": "uptrend",
                }
            ],
        },
    )

    assert out["count"] == 1
    assert out["rows"][0]["ticker"] == "AAPL"
    assert out["rows"][0]["macd_state"] == "bullish"
    assert "macd" not in out["rows"][0]


def test_compact_tool_result_technical_signals_reports_truncation() -> None:
    raw_rows = [
        {
            "ticker": f"T{i:02d}",
            "price": 100.0 + i,
            "rsi_14": 50.0,
            "rsi_state": "neutral",
            "macd": {"state": "neutral"},
            "moving_averages": {"price_vs_sma20": 0.01},
            "bollinger_20_2": {"state": "inside_bands"},
            "trend_state": "flat",
        }
        for i in range(11)
    ]

    out = _compact_tool_result_for_prompt(
        "technical_signals",
        {"tickers": [row["ticker"] for row in raw_rows], "count": 11, "rows": raw_rows},
        args={"tickers": [row["ticker"] for row in raw_rows]},
    )

    assert len(out["rows"]) == 10
    assert out["compaction"] == {
        "requested_count": 11,
        "returned_count": 11,
        "visible_count": 10,
        "visible_limit": 10,
        "truncated": True,
    }


def test_compact_tool_result_earnings_calendar_reports_truncation() -> None:
    rows = [
        {"date": "2026-05-01", "symbol": f"T{i:02d}", "name": "Name", "time": "AMC", "eps_forecast": "1.00"}
        for i in range(11)
    ]

    out = _compact_tool_result_for_prompt(
        "earnings_calendar",
        {"ticker": None, "tickers": [row["symbol"] for row in rows], "count": 11, "rows": rows},
        args={"tickers": [row["symbol"] for row in rows]},
    )

    assert len(out["rows"]) == 10
    assert out["compaction"]["requested_count"] == 11
    assert out["compaction"]["returned_count"] == 11
    assert out["compaction"]["visible_limit"] == 10
    assert out["compaction"]["truncated"] is True


def test_compact_tool_result_screen_market_keeps_bucket_reason_and_value_fields() -> None:
    out = _compact_tool_result_for_prompt(
        "screen_market",
        [
            {
                "ticker": "PBR",
                "bucket": "value",
                "bucket_rank": 1,
                "score": 2.14,
                "reason": "Valuation support: PER 6.2, PBR 1.1",
                "reason_for": "Valuation support: PER 6.2, PBR 1.1",
                "reason_risk": "Screen-only evidence; confirm first.",
                "ret_20d": 0.11,
                "ret_5d": -0.02,
                "volatility_20d": 0.21,
                "sentiment_score": 0.08,
                "per": 6.2,
                "pbr": 1.1,
                "roe": 18.0,
                "debt_ratio": 72.0,
                "close_price_krw": 18340.0,
            }
        ],
    )

    assert out[0]["ticker"] == "PBR"
    assert out[0]["bucket"] == "value"
    assert out[0]["reason"].startswith("Valuation support")
    assert out[0]["reason_for"].startswith("Valuation support")
    assert out[0]["reason_risk"] == "Screen-only evidence; confirm first."
    assert out[0]["per"] == 6.2
    assert out[0]["pbr"] == 1.1


def test_compact_tool_result_recommend_opportunities_keeps_validation_fields() -> None:
    out = _compact_tool_result_for_prompt(
        "recommend_opportunities",
        {
            "status": "ok",
            "recommendations": [
                {
                    "ticker": "PBR",
                    "profile": "value",
                    "bucket": "value",
                    "recommendation_rank": 1,
                    "recommendation_score": 1.7,
                    "score_components": {"forecast": 0.5, "technical": 0.2},
                    "signal_contributions": [{"signal": "ep", "contribution": 0.4}],
                    "confidence": "high",
                    "action": "candidate",
                    "reason_for": "Validated value candidate",
                    "reason_risk": "valuation risk",
                    "optimizer_weight": 0.18,
                    "evidence_level": "validated",
                }
            ],
            "optimizer": {"status": "ok", "strategy": "forecast_max_sharpe", "weights": {"PBR": 0.18}},
            "diagnostics": {
                "score_policy": {
                    "version": "heuristic_ranker_v1",
                    "score_formula": "0.40*screen_rank_score + ...",
                },
                "selection_scope": {
                    "mode": "ranked_union",
                    "global_limit": 8,
                    "per_profile_limit": 8,
                    "loaded_rows": 73,
                    "requested_buckets": ["value"],
                },
            },
        },
    )

    assert out["status"] == "ok"
    assert out["recommendations"][0]["ticker"] == "PBR"
    assert out["recommendations"][0]["profile"] == "value"
    assert out["recommendations"][0]["signal_contributions"] == [{"signal": "ep", "contribution": 0.4}]
    assert out["recommendations"][0]["optimizer_weight"] == 0.18
    assert out["recommendations"][0]["score_components"]["forecast"] == 0.5
    assert out["optimizer"]["weights"] == {"PBR": 0.18}
    assert out["score_policy"]["version"] == "heuristic_ranker_v1"
    assert out["selection_scope"]["global_limit"] == 8
    assert out["selection_scope"]["per_profile_limit"] == 8
    assert out["selection_scope"]["loaded_rows"] == 73


def test_tool_wrapper_injects_memory_for_macro_tools_with_typed_query() -> None:
    captured: dict[str, object] = {}

    def macro_snapshot() -> dict:
        return {
            "indicators": {
                "fed_funds_rate": {"value": 5.25, "unit": "%"},
                "treasury_10y": {"value": 4.8, "unit": "%"},
            },
            "source": "fred",
        }

    def search_tool_memories(query):
        captured["query"] = query
        return [{"event_id": "mem_macro", "summary": "High-rate regimes require smaller gross exposure."}]

    wrapper = build_tool_wrapper(
        ToolEntry(
            tool_id="macro_snapshot",
            name="macro_snapshot",
            description="Fetch macro indicators.",
            category="macro",
            callable=macro_snapshot,
        ),
        settings=SimpleNamespace(memory_policy=None),
        agent_id="gpt",
        tool_events=[],
        update_candidate_ledger=lambda *args: None,
        search_tool_memories=search_tool_memories,
        apply_tool_schema_metadata=lambda fn, **kwargs: fn,
    )

    out = wrapper()

    assert out["_memory_context"][0]["summary"] == "High-rate regimes require smaller gross exposure."
    query = captured["query"]
    assert getattr(query, "key_type") == "regime"
    assert "regime:high_rates" in query.search_text()


def test_compact_tool_result_get_fundamentals_reduces_meta_lists() -> None:
    out = _compact_tool_result_for_prompt(
        "get_fundamentals",
        {
            "requested": ["AAPL", "MSFT", "XYZ"],
            "eligible": ["AAPL", "MSFT"],
            "excluded": ["XYZ"],
            "rows": [
                {"ticker": "AAPL", "market": "us", "per": 31.5, "pbr": 45.2, "eps": 6.38, "currency": "USD", "exchange": "NAS"},
                {"ticker": "MSFT", "market": "us", "per": 34.0, "pbr": 12.1, "eps": 12.5, "currency": "USD", "exchange": "NAS"},
            ],
            "errors": [{"ticker": "XYZ", "error": "ticker not found in upstream fundamentals payload"}],
        },
    )

    assert out["requested_count"] == 3
    assert out["eligible_count"] == 2
    assert out["excluded_count"] == 1
    assert out["excluded"] == ["XYZ"]
    assert out["rows"][0]["ticker"] == "AAPL"
    assert out["errors"][0]["ticker"] == "XYZ"
