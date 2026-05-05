from __future__ import annotations

from arena.agents.adk_agents import _ADKDecisionRunner
from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, Side


def test_candidate_ledger_tracks_discovery_and_analysis_from_result_tickers() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {}
    runner._held_tickers_cache = {"AAPL"}
    runner._current_phase = "explore"
    runner._current_context = {}
    runner._tool_events = []

    runner._update_candidate_ledger(
        "screen_market",
        {},
        [
            {"ticker": "AAPL", "bucket": "momentum"},
            {
                "ticker": "MSFT",
                "bucket": "value",
                "score": 1.7,
                "reason_for": "Valuation support: PER 14.0",
                "reason_risk": "Screen-only evidence; confirm first.",
                "ret_20d": 0.04,
            },
            {"ticker": "TSLA", "bucket": "pullback", "reason": "Recent pullback"},
        ],
    )

    assert set(runner._candidate_ledger.keys()) == {"MSFT", "TSLA"}
    assert runner._candidate_ledger["MSFT"]["source_tools"] == {"screen_market:value"}
    assert runner._candidate_ledger["TSLA"]["source_tools"] == {"screen_market:pullback"}
    assert runner._current_context["_candidate_tickers"] == ["MSFT", "TSLA"]
    assert runner._current_context["_discovered_candidate_tickers"] == ["MSFT", "TSLA"]
    assert [row["ticker"] for row in runner._current_context["opportunity_working_set"]] == ["MSFT", "TSLA"]
    assert runner._current_context["opportunity_working_set"][0]["status"] == "screened_only"
    assert runner._current_context["opportunity_working_set"][0]["workflow_status"] == "pending"
    assert runner._current_context["opportunity_working_set"][0]["discovery_buckets"] == ["value"]
    assert runner._current_context["analysis_funnel_prompt"]["screened_only_candidates"] == 2
    assert runner._current_context["candidate_cases"][0]["ticker"] == "MSFT"
    assert runner._current_context["candidate_cases"][0]["case_for"].startswith("Valuation support")
    assert runner._current_context["candidate_cases"][0]["case_risk"] == "Screen-only evidence; confirm first."
    assert runner._current_context["candidate_cases"][0]["evidence_level"] == "screened_only"
    assert "thesis_summary" not in runner._current_context["candidate_cases"][0]

    runner._update_candidate_ledger(
        "forecast_returns",
        {},
        [{"ticker": "MSFT", "exp_return_period": 0.02}],
    )

    assert runner._candidate_ledger["MSFT"]["analyzed_by"] == {"forecast_returns"}
    assert runner._current_context["analysis_funnel"]["analyzed_nonheld"] == 1
    assert runner._current_context["analysis_funnel"]["pending_nonheld"] == 1
    assert runner._current_context["analysis_funnel_prompt"]["fully_analyzed_candidates"] == 1
    assert runner._current_context["analysis_funnel_prompt"]["screened_only_candidates"] == 1
    assert runner._current_context["opportunity_working_set"][0]["ticker"] == "TSLA"


def test_candidate_ledger_records_screen_market_momentum_bucket_source() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {}
    runner._held_tickers_cache = set()
    runner._current_phase = "explore"
    runner._current_context = {}
    runner._tool_events = []

    runner._update_candidate_ledger(
        "screen_market",
        {},
        [{"ticker": "PBR", "bucket": "momentum", "score": 1.25}],
    )

    assert runner._candidate_ledger["PBR"]["source_tools"] == {"screen_market:momentum"}
    assert runner._current_context["opportunity_working_set"][0]["discovery_buckets"] == ["momentum"]
    assert runner._current_context["opportunity_working_set"][0]["status"] == "screened_only"
    assert runner._current_context["candidate_cases"][0]["ticker"] == "PBR"


def test_candidate_ledger_records_recommend_opportunities_profile_source() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {}
    runner._held_tickers_cache = set()
    runner._current_phase = "explore"
    runner._current_context = {}
    runner._tool_events = []

    runner._update_candidate_ledger(
        "recommend_opportunities",
        {},
        {
            "rows": [
                {
                    "ticker": "MSFT",
                    "profile": "defensive",
                    "bucket": "defensive",
                    "buckets": ["defensive"],
                    "recommendation_score": 1.4,
                    "confidence": "medium",
                    "reason_for": "Validated defensive candidate",
                    "reason_risk": "No major gaps.",
                    "evidence_level": "validated",
                }
            ]
        },
    )

    assert runner._candidate_ledger["MSFT"]["source_tools"] == {"recommend_opportunities:defensive"}
    assert runner._current_context["opportunity_working_set"][0]["discovery_buckets"] == ["defensive"]
    assert runner._current_context["candidate_cases"][0]["case_for"] == "Validated defensive candidate"


def test_funnel_metrics_counts_held_analysis_from_result_tickers() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {
        "MSFT": {"analyzed_by": {"forecast_returns"}},
        "TSLA": {"analyzed_by": set()},
    }
    runner._held_tickers_cache = {"AAPL"}
    runner._tool_events = [
        {
            "tool": "forecast_returns",
            "args": {},
            "result": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
        },
        {
            "tool": "technical_signals",
            "args": {"ticker": "AAPL"},
            "result": {"ticker": "AAPL", "trend_state": "uptrend"},
        },
    ]

    metrics = runner._funnel_metrics()

    assert metrics == {
        "discovered_nonheld": 2,
        "analyzed_nonheld": 1,
        "pending_nonheld": 1,
        "analyzed_held": 1,
        "ordered_nonheld": 0,
        "intended_nonheld": 0,
        "executed_nonheld": 0,
        "skipped_nonheld": 0,
        "skip_reasons": {},
    }


def test_candidate_ledger_tracks_order_and_execution_funnel() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {
        "MSFT": {"analyzed_by": {"forecast_returns"}},
        "TSLA": {"analyzed_by": set()},
    }
    runner._held_tickers_cache = {"AAPL"}
    runner._current_phase = "execution"
    runner._current_context = {}
    runner._tool_events = []

    runner.record_candidate_orders(
        [
            {"ticker": "MSFT", "side": "BUY", "target_weight": 0.2},
            {"ticker": "AAPL", "side": "BUY", "target_weight": 0.1},
        ]
    )
    runner.record_candidate_order_feedback(
        [
            {"ticker": "MSFT", "side": "BUY", "status": "intent_built"},
            {"ticker": "TSLA", "side": "BUY", "status": "skipped", "reason": "no_price"},
        ]
    )
    runner.record_candidate_executions(
        [
            OrderIntent(
                agent_id="gpt",
                ticker="MSFT",
                side=Side.BUY,
                quantity=1.0,
                price_krw=1000.0,
                rationale="candidate buy",
            )
        ],
        [
            ExecutionReport(
                status=ExecutionStatus.FILLED,
                order_id="ord_1",
                filled_qty=1.0,
                avg_price_krw=1000.0,
                message="filled",
            )
        ],
    )

    metrics = runner._funnel_metrics()

    assert metrics["ordered_nonheld"] == 1
    assert metrics["intended_nonheld"] == 1
    assert metrics["executed_nonheld"] == 1
    assert metrics["skipped_nonheld"] == 1
    assert metrics["skip_reasons"] == {"no_price": 1}


def test_sync_pipeline_context_adds_decision_frame_when_opportunities_have_budget() -> None:
    runner = _ADKDecisionRunner.__new__(_ADKDecisionRunner)
    runner._candidate_ledger = {"MSFT": {"source_tools": {"screen_market"}, "analyzed_by": set()}}
    runner._held_tickers_cache = {"AAPL"}
    runner._current_phase = "execution"
    runner._tool_events = []
    runner._current_context = {"order_budget": {"max_buy_notional_krw": 500000.0}}

    runner._sync_pipeline_context()

    assert "Compare any self-discovered opportunities" in runner._current_context["decision_frame"]
    assert runner._current_context["candidate_cases"][0]["ticker"] == "MSFT"
    assert runner._current_context["candidate_cases"][0]["candidate_status"] == "screened_only"
