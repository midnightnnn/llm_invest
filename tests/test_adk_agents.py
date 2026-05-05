from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from arena.agents.adk_agents import (
    AdkTradingAgent,
    _ADKDecisionRunner,
    _apply_tool_schema_metadata,
    _agent_config_payload,
    _compact_tool_result_for_prompt,
    _ContextTools,
    _has_credentials,
    _is_retryable_adk_error,
    _load_disabled_tool_ids,
    _resolve_disabled_tool_ids,
    _resolve_model,
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
from arena.models import BoardPost, ExecutionReport, ExecutionStatus, OrderIntent, Side, utc_now
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


class _RepoForAdkGenerate:
    def latest_market_features(self, tickers, limit, sources=None):
        _ = (tickers, limit, sources)
        return []


class _FakeRunner:
    def __init__(self) -> None:
        self.board_calls: list[tuple[str, str, str]] = []

    def decide_orders(self, *, context, default_universe, resume_session_id=None):
        _ = (context, default_universe, resume_session_id)
        return (
            {
                "orders": [
                    {
                        "ticker": "AAPL",
                        "side": "BUY",
                        "target_weight": 0.5,
                        "rationale": "fx repricing",
                    }
                ],
            },
            "sid_1",
        )

    def decide_board(self, session_id, orders_summary, *, cycle_id=""):
        self.board_calls.append((session_id, orders_summary, cycle_id))
        return {"board_title": "confirmed", "board_body": orders_summary}


class _FakeKospiRunner(_FakeRunner):
    def decide_orders(self, *, context, default_universe, resume_session_id=None):
        _ = (context, default_universe, resume_session_id)
        return (
            {
                "orders": [
                    {
                        "ticker": "025860",
                        "side": "BUY",
                        "target_weight": 0.2,
                        "rationale": "momentum continuation",
                    }
                ],
            },
            "sid_kospi_1",
        )

    def decide_board(self, session_id, orders_summary, *, cycle_id=""):
        self.board_calls.append((session_id, orders_summary, cycle_id))
        return {
            "board_title": "이녹스첨단소재를 다시 담다",
            "board_body": "**이녹스첨단소재(025860)** BUY 48주 체결\n전날 27주에 이어 오늘 48주.",
        }


class _FailRunner:
    def decide_orders(self, *, context, default_universe, resume_session_id=None):
        _ = (context, default_universe, resume_session_id)
        raise RuntimeError("runner boom")


def test_generate_reprices_us_order_with_live_fx(monkeypatch) -> None:
    runner = _FakeRunner()
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: runner)
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    out = agent.generate(
        {
            "cycle_phase": "execution",
            "cycle_id": "cycle_fx_1",
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "usd_krw_rate": 1450.0,
                "positions": {},
            },
            "market_features": [
                {
                    "ticker": "AAPL",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:AAPL",
                    "close_price_krw": 130000.0,
                    "close_price_native": 100.0,
                    "quote_currency": "USD",
                    "fx_rate_used": 1300.0,
                }
            ],
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        }
    )

    assert len(out.intents) == 1
    intent = out.intents[0]
    assert intent.price_krw == pytest.approx(145000.0)
    assert intent.price_native == pytest.approx(100.0)
    assert intent.quote_currency == "USD"
    assert intent.fx_rate == pytest.approx(1450.0)
    assert runner.board_calls == []


def test_generate_raises_when_market_features_missing(monkeypatch) -> None:
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: _FakeRunner())
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    with pytest.raises(RuntimeError, match="market_features missing"):
        agent.generate({"cycle_phase": "execution", "cycle_id": "cycle_missing_rows", "market_features": []})


def test_generate_raises_when_decision_fails(monkeypatch) -> None:
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: _FailRunner())
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    with pytest.raises(RuntimeError, match="ADK decision failed"):
        agent.generate(
            {
                "cycle_phase": "execution",
                "cycle_id": "cycle_decision_fail",
                "market_features": [
                    {
                        "ticker": "AAPL",
                        "exchange_code": "NASD",
                        "instrument_id": "NASD:AAPL",
                        "close_price_krw": 130000.0,
                        "close_price_native": 100.0,
                        "quote_currency": "USD",
                        "fx_rate_used": 1300.0,
                    }
                ],
            }
        )


def test_agent_config_payload_serializes_dataclass() -> None:
    payload = _agent_config_payload(
        AgentConfig(
            agent_id="claude",
            provider="claude",
            model="claude-sonnet-4-6",
            capital_krw=2_000_000.0,
            target_market="kospi",
            system_prompt="focus on risk",
            risk_overrides={"max_position_ratio": 0.2},
            disabled_tools=["trade_performance"],
        )
    )

    assert payload == {
        "agent_id": "claude",
        "provider": "claude",
        "model": "claude-sonnet-4-6",
        "capital_krw": 2_000_000.0,
        "target_market": "kospi",
        "system_prompt": "focus on risk",
        "risk_overrides": {"max_position_ratio": 0.2},
        "disabled_tools": ["trade_performance"],
        "llm_params": None,
        "memory_compaction_model": "",
    }


def test_finalize_board_post_uses_execution_summary(monkeypatch) -> None:
    runner = _FakeRunner()
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: runner)
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    out = agent.generate(
        {
            "cycle_phase": "execution",
            "cycle_id": "cycle_fx_2",
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "usd_krw_rate": 1450.0,
                "positions": {},
            },
            "market_features": [
                {
                    "ticker": "AAPL",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:AAPL",
                    "close_price_krw": 130000.0,
                    "close_price_native": 100.0,
                    "quote_currency": "USD",
                    "fx_rate_used": 1300.0,
                }
            ],
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        }
    )

    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_1",
        filled_qty=1.0,
        avg_price_krw=145145.0,
        avg_price_native=100.1,
        quote_currency="USD",
        fx_rate=1450.0,
        message="confirmed",
        created_at=utc_now(),
    )

    post = agent.finalize_board_post(
        cycle_id="cycle_fx_2",
        initial_post=BoardPost(
            agent_id="gpt",
            title="placeholder",
            body="pending",
            tickers=["AAPL"],
            cycle_id="cycle_fx_2",
        ),
        intents=out.intents,
        reports=[report],
    )

    assert len(runner.board_calls) == 1
    _, summary, board_cycle_id = runner.board_calls[0]
    assert "실제 실행 결과" in summary
    assert "AAPL BUY 1주 FILLED" in summary
    assert board_cycle_id == "cycle_fx_2"
    assert post.body == summary


def test_finalize_board_post_keeps_freeform_board_text(monkeypatch) -> None:
    runner = _FakeKospiRunner()
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: runner)
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "kospi"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    settings.default_universe = ["025860"]

    agent = AdkTradingAgent(
        agent_id="claude",
        provider="claude",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    out = agent.generate(
        {
            "cycle_phase": "execution",
            "cycle_id": "cycle_kospi_1",
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_500_000.0,
                "positions": {
                    "025860": {
                        "quantity": 27.0,
                        "avg_price_krw": 8290.0,
                        "market_price_krw": 8270.0,
                        "ticker_name": "남해화학",
                    }
                },
            },
            "market_features": [
                {
                    "ticker": "025860",
                    "exchange_code": "KRX",
                    "instrument_id": "KRX:025860",
                    "close_price_krw": 8270.0,
                    "close_price_native": 8270.0,
                    "quote_currency": "KRW",
                    "fx_rate_used": 1.0,
                }
            ],
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        }
    )

    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_kospi_1",
        filled_qty=48.0,
        avg_price_krw=8290.0,
        quote_currency="KRW",
        fx_rate=1.0,
        message="confirmed",
        created_at=utc_now(),
    )

    post = agent.finalize_board_post(
        cycle_id="cycle_kospi_1",
        initial_post=out.board_post,
        intents=out.intents,
        reports=[report],
    )

    assert len(runner.board_calls) == 1
    assert runner.board_calls[0][2] == "cycle_kospi_1"
    assert post.title == "이녹스첨단소재를 다시 담다"
    assert "**이녹스첨단소재(025860)** BUY 48주 체결" in post.body
    assert "전날 27주에 이어 오늘 48주." in post.body


def test_generate_skips_mixed_us_order_when_exchange_is_unresolved(monkeypatch) -> None:
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: _FakeRunner())
    settings = load_settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "us"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    settings.default_universe = ["AAPL"]

    agent = AdkTradingAgent(
        agent_id="gpt",
        provider="gpt",
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )

    out = agent.generate(
        {
            "cycle_phase": "execution",
            "cycle_id": "cycle_fx_2",
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "usd_krw_rate": 1450.0,
                "positions": {},
            },
            "market_features": [
                {
                    "ticker": "AAPL",
                    "exchange_code": "",
                    "instrument_id": "",
                    "close_price_krw": 130000.0,
                    "close_price_native": 100.0,
                    "quote_currency": "USD",
                    "fx_rate_used": 1300.0,
                }
            ],
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        }
    )

    assert out.intents == []


class _MemoryStoreForToolSummary:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_memory(self, **kwargs) -> None:
        self.calls.append(kwargs)


class _MemoryStoreForCandidateMemory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_candidate_memories(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return 1


class _VectorStoreForToolMemory:
    def search_similar_memories(self, **kwargs):
        _ = kwargs
        from datetime import datetime

        return [
            {
                "summary": "Macro-sensitive trim discipline mattered.",
                "importance_score": 0.8,
                "created_at": datetime.fromisoformat("2026-03-05T00:00:00+00:00"),
                "outcome_score": 0.8,
            }
        ]


class _VectorStoreForDedupedToolMemory:
    def search_similar_memories(self, **kwargs):
        _ = kwargs
        from datetime import datetime

        return [
            {
                "event_id": "mem_seen",
                "summary": "Already injected lesson.",
                "importance_score": 0.9,
                "created_at": datetime.fromisoformat("2026-03-05T00:00:00+00:00"),
                "outcome_score": 0.8,
            },
            {
                "event_id": "mem_new",
                "summary": "Fresh trim discipline lesson.",
                "importance_score": 0.7,
                "created_at": datetime.fromisoformat("2026-03-04T00:00:00+00:00"),
                "outcome_score": 0.2,
            },
        ]


class _VectorStoreForContextToolSearch:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_similar_memories(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {"event_id": "mem_seen", "summary": "Already prompt-injected lesson."},
            {"event_id": "mem_new", "summary": "Fresh trim discipline lesson."},
            {"event_id": "mem_extra", "summary": "Second fresh lesson."},
        ]


class _MemoryStoreForToolMemory:
    def __init__(self) -> None:
        self.vector_store = _VectorStoreForToolMemory()

    def _tenant(self) -> str:
        return "local"


class _MemoryStoreForDedupedToolMemory:
    def __init__(self) -> None:
        self.vector_store = _VectorStoreForDedupedToolMemory()

    def _tenant(self) -> str:
        return "local"


class _RepoForToolSummary:
    def __init__(self) -> None:
        self.events = []

    def write_memory_event(self, event) -> None:
        self.events.append(event)


def test_persist_tool_summary_memory_prefers_memory_store() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForToolSummary()
    runner.repo = _RepoForToolSummary()

    runner._persist_tool_summary_memory(
        summary="ReAct tools used (explore): 2",
        payload={
            "tool_events": [{"tool": "technical_signals"}],
            "phase": "explore",
            "token_usage": {"llm_calls": 2, "prompt_tokens": 1200, "completion_tokens": 180, "total_tokens": 1380},
        },
    )

    assert len(runner._memory_store.calls) == 1
    call = runner._memory_store.calls[0]
    assert call["agent_id"] == "gpt"
    assert call["event_type"] == "react_tools_summary"
    assert call["score"] == pytest.approx(0.6)
    assert call["payload"]["token_usage"]["total_tokens"] == 1380
    assert runner.repo.events == []


def test_persist_candidate_memories_uses_candidate_ledger() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner._memory_store = _MemoryStoreForCandidateMemory()
    runner._candidate_ledger = {
        "MSFT": {
            "source_tools": {"screen_market:value"},
            "discovery_count": 1,
            "last_seen_rank": 2,
            "discovery_evidence": {"reason_for": "Valuation support"},
        }
    }
    runner._held_tickers_cache = {"AAPL"}
    runner._current_phase = "execution"

    written = runner._persist_candidate_memories(cycle_id="cycle_candidate")

    assert written == 1
    call = runner._memory_store.calls[0]
    assert call["agent_id"] == "gpt"
    assert call["held_tickers"] == {"AAPL"}
    assert call["cycle_id"] == "cycle_candidate"
    assert call["phase"] == "execution"
    assert "MSFT" in call["candidate_ledger"]


def test_decide_orders_keeps_tool_events_reference_for_wrapped_tools(monkeypatch) -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    shared_tool_events = [{"tool": "stale_event"}]
    runner._tool_events = shared_tool_events
    runner._seen_memory_ids = set()
    runner._candidate_ledger = {}
    runner._current_phase = "unknown"
    runner._current_context = None
    runner._held_tickers_cache = set()
    runner._session_id = "sid_base"
    runner._max_tool_events = 5
    runner._run_config = object()
    runner._runner = object()
    runner._user_id = "arena"
    runner.agent_id = "gpt"
    runner._registry = SimpleNamespace(
        set_context=lambda context: None,
        list_entries=lambda **kwargs: [],
    )
    runner._toolbox = SimpleNamespace(set_context=lambda context: None)
    runner._memory_store = None
    runner._seed_seen_memory_ids = lambda context: None
    runner._extract_held_tickers = lambda context: set()
    runner._sync_pipeline_context = lambda: None
    runner._funnel_metrics = lambda: {}
    runner._persist_tool_summary_memory = lambda *, summary, payload: None
    runner._run_on_loop = lambda value: value
    runner._disabled_tool_ids = set()
    runner._mcp_toolset_count = 0
    runner._system_prompt_snapshot = ""
    runner._agent_config = None
    runner._prompt_snapshots = []
    runner._llm_call_ids_by_phase = {}
    runner._latest_llm_call_id = ""
    runner.provider = "gpt"
    runner.settings = SimpleNamespace(trading_mode="paper", kis_target_market="", memory_policy=None)
    runner.tenant_id = "local"
    runner.repo = SimpleNamespace()

    def _fake_run_async(_runner, session_id, prompt):
        _ = (_runner, session_id, prompt)
        shared_tool_events.append(
            {
                "tool": "technical_signals",
                "args": {"ticker": "AAPL"},
                "result": {"ticker": "AAPL", "trend_state": "uptrend"},
            }
        )
        return '{"orders": []}'

    runner._run_async = _fake_run_async

    monkeypatch.setattr(
        "arena.agents.adk_agents.prepare_decision_prompt",
        lambda *args, **kwargs: ("sid_test", "prompt", False),
    )
    monkeypatch.setattr("arena.agents.adk_agents.parse_decision_response", lambda text: {"orders": []})
    monkeypatch.setattr("arena.agents.adk_agents.tag_phase_tool_events", lambda *args, **kwargs: None)

    captured: dict[str, object] = {}

    def _capture_summary(tool_events, **kwargs):
        _ = kwargs
        captured["tool_names"] = [str(event.get("tool") or "") for event in tool_events]
        captured["tool_events_id"] = id(tool_events)
        return None

    monkeypatch.setattr("arena.agents.adk_agents.build_tool_summary_memory_record", _capture_summary)

    decision, session_id = runner.decide_orders({"cycle_phase": "execution"}, [])

    assert decision == {"orders": []}
    assert session_id == "sid_test"
    assert runner._tool_events is shared_tool_events
    assert captured["tool_events_id"] == id(shared_tool_events)
    assert captured["tool_names"] == ["technical_signals"]


def test_search_past_experiences_skips_cycle_seen_memory_ids() -> None:
    vector_store = _VectorStoreForContextToolSearch()
    tool = _ContextTools.__new__(_ContextTools)
    tool.agent_id = "gpt"
    tool.tenant_id = "local"
    tool.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    tool._vector_store = vector_store
    tool._seen_memory_ids = set()
    tool._seen_memory_ids_shared = False

    tool.set_context({"memory_events": [{"event_id": "mem_seen"}]})
    rows = tool.search_past_experiences("trim discipline", limit=2)

    assert [row["event_id"] for row in rows] == ["mem_new", "mem_extra"]
    assert vector_store.calls[0]["limit"] == 5
    assert tool._seen_memory_ids == {"mem_seen", "mem_new", "mem_extra"}


def test_search_tool_memories_includes_created_date() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForToolMemory()
    runner._seen_memory_ids = set()

    rows = runner._search_tool_memories("macro regime trim discipline")

    assert rows is not None
    assert rows[0]["created_date"] == "2026-03-05"
    assert rows[0]["created_at"].startswith("2026-03-05T00:00:00")
    assert rows[0]["outcome_label"] == "win"


def test_search_tool_memories_skips_initially_injected_event_ids() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner.agent_id = "gpt"
    runner.settings = type("SettingsStub", (), {"trading_mode": "paper", "memory_policy": None})()
    runner._memory_store = _MemoryStoreForDedupedToolMemory()
    runner._seen_memory_ids = {"mem_seen"}

    rows = runner._search_tool_memories("trim discipline")

    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["summary"] == "Fresh trim discipline lesson."
    assert "mem_new" in runner._seen_memory_ids


def test_seed_seen_memory_ids_uses_initial_context_memory_rows() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._seen_memory_ids = set()

    runner._seed_seen_memory_ids(
        {
            "memory_events": [
                {"event_id": "mem_a"},
                {"event_id": "mem_b"},
                {"summary": "no id"},
            ]
        }
    )

    assert runner._seen_memory_ids == {"mem_a", "mem_b"}


# ──────────────────────────────────────────────────
# Per-agent: _resolve_disabled_tool_ids
# ──────────────────────────────────────────────────

def test_resolve_disabled_tool_ids_uses_agent_config_override() -> None:
    repo = _RepoForTools('["tool_a","tool_b"]')
    ac = AgentConfig(
        agent_id="custom",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        disabled_tools=["tool_x"],
    )
    result = _resolve_disabled_tool_ids(repo, "tenant-a", ac)
    assert result == {"tool_x"}


def test_resolve_disabled_tool_ids_falls_back_to_global() -> None:
    repo = _RepoForTools('["tool_a"]')
    ac = AgentConfig(
        agent_id="custom",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        disabled_tools=None,
    )
    result = _resolve_disabled_tool_ids(repo, "tenant-a", ac)
    assert result == {"tool_a"}


def test_resolve_disabled_tool_ids_without_agent_config() -> None:
    repo = _RepoForTools('["tool_a"]')
    result = _resolve_disabled_tool_ids(repo, "tenant-a", None)
    assert result == {"tool_a"}


# ──────────────────────────────────────────────────
# Per-agent: _has_credentials
# ──────────────────────────────────────────────────

def test_has_credentials_gpt() -> None:
    s = load_settings()
    s.openai_api_key = "sk-test"
    assert _has_credentials("gpt", s) is True

    s.openai_api_key = ""
    assert _has_credentials("gpt", s) is False


def test_has_credentials_claude() -> None:
    s = load_settings()
    s.anthropic_api_key = "ak-test"
    s.anthropic_use_vertexai = False
    assert _has_credentials("claude", s) is True

    s.anthropic_api_key = ""
    s.anthropic_use_vertexai = True
    assert _has_credentials("claude", s) is True

    s.anthropic_use_vertexai = False
    assert _has_credentials("claude", s) is False


def test_has_credentials_unknown() -> None:
    s = load_settings()
    assert _has_credentials("unknown", s) is False


def test_resolve_model_openai_uses_instance_scoped_api_key() -> None:
    settings = load_settings()
    settings.openai_api_key = "tenant-openai"
    settings.llm_timeout_seconds = 1500

    model = _resolve_model("gpt", settings, model_override="gpt-5.4")

    assert model.model == "openai/gpt-5.4"
    assert model._additional_args["api_key"] == "tenant-openai"
    assert model._additional_args["timeout"] == 1500


def test_resolve_model_claude_direct_uses_instance_scoped_api_key() -> None:
    settings = load_settings()
    settings.anthropic_api_key = "tenant-anthropic"
    settings.anthropic_use_vertexai = False
    settings.anthropic_model = "claude-sonnet-4-6"
    settings.llm_timeout_seconds = 1500

    model = _resolve_model("claude", settings)

    assert model.model == "anthropic/claude-sonnet-4-6"
    assert model._additional_args["api_key"] == "tenant-anthropic"
    assert model._additional_args["timeout"] == 1500
    assert model._additional_args["cache_control_injection_points"] == [
        {"location": "message", "role": "system"},
    ]


def test_resolve_model_deepseek_uses_provider_payload_api_key_and_base_url() -> None:
    settings = load_settings()
    settings.provider_secrets = {
        "deepseek": {
            "api_key": "tenant-deepseek",
            "model": "deepseek-chat",
            "base_url": "https://custom.deepseek/v1",
        }
    }

    model = _resolve_model("deepseek", settings)

    assert model.model == "deepseek/deepseek-chat"
    assert model._additional_args["api_key"] == "tenant-deepseek"
    assert model._additional_args["base_url"] == "https://custom.deepseek/v1"
