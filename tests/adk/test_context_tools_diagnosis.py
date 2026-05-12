from __future__ import annotations

from datetime import datetime, timezone

import pytest

from arena.agents.adk_agents import _ContextTools, _compact_tool_result_for_prompt
from arena.config import load_settings
from tests.adk.context_tools_helpers import (
    _RepoForPortfolioDiagnosis,
    _RepoForPortfolioDiagnosisExact,
    _RepoForPortfolioDiagnosisRaises,
)


def test_context_portfolio_weights_require_market_price_not_avg_cost() -> None:
    tool = _ContextTools.__new__(_ContextTools)
    tool._context = {
        "portfolio": {
            "positions": {
                "AAPL": {"quantity": 10.0, "avg_price_krw": 100.0},
            },
        },
    }

    weights, stock_mv, cash = tool._portfolio_weights()

    assert weights == {}
    assert stock_mv == 0.0
    assert cash == 0.0


def test_portfolio_diagnosis_returns_derived_fields_not_raw_portfolio_echo() -> None:
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForPortfolioDiagnosis()
    tool.settings = load_settings()
    tool.settings.kis_target_market = "nasdaq"
    tool._context = {
        "portfolio": {
            "cash_krw": 1_000.0,
            "positions": {
                "AAPL": {"quantity": 10.0, "avg_price_krw": 100.0},
                "MSFT": {"quantity": 5.0, "avg_price_krw": 200.0},
            },
        },
        "market_features": [
            {"ticker": "AAPL", "close_price_krw": 110.0, "volatility_20d": 0.2, "ret_20d": 0.08, "ret_5d": 0.03},
            {"ticker": "MSFT", "close_price_krw": 208.0, "volatility_20d": 0.1, "ret_20d": 0.04, "ret_5d": 0.01},
        ],
        "performance": {
            "initialized_at": "2026-01-01T00:00:00+00:00",
            "pnl_ratio": 0.12,
        },
        "risk_policy": {
            "min_cash_buffer_ratio": 0.10,
            "max_position_ratio": 0.60,
        },
    }

    out = tool.portfolio_diagnosis(mdd_days=5, top_n=2)

    assert "cash_krw" not in out
    assert "stock_market_value_krw" not in out
    assert "weights" not in out
    assert "performance" not in out
    assert "top_weights" not in out
    assert "cash_weight" not in out
    assert "gross_exposure" not in out
    assert "risk_contribution" in out
    assert "mdd" in out
    assert out["benchmark"]["ticker"] == "QQQ"
    assert "rebalance_plan" not in out
    assert "hrp_allocation" not in out


def test_portfolio_diagnosis_adds_joint_policy_scores_for_current_holdings() -> None:
    class _Repo(_RepoForPortfolioDiagnosis):
        def __init__(self) -> None:
            self.ranker_kwargs: dict[str, object] | None = None

        def latest_opportunity_ranker_scores(self, **kwargs):  # noqa: ANN001
            self.ranker_kwargs = dict(kwargs)
            return [
                {
                    "ticker": "AAPL",
                    "market": "us",
                    "profile": "balanced",
                    "bucket": "momentum",
                    "recommendation_rank": 4,
                    "recommendation_score": 0.12,
                    "score_source": "joint_policy_v1",
                    "ranker_version": "opportunity_ranker_joint_policy_20260512_test",
                    "model_confidence": "medium",
                    "action": "watchlist",
                    "explanation_json": {
                        "top_contributions": [
                            {"signal": "forecast_er", "contribution": 0.07},
                            {"signal": "ret_20d", "contribution": 0.05},
                        ]
                    },
                },
                {
                    "ticker": "MSFT",
                    "market": "us",
                    "profile": "balanced",
                    "bucket": "defensive",
                    "recommendation_rank": 9,
                    "recommendation_score": -0.03,
                    "score_source": "joint_policy_v1",
                    "ranker_version": "opportunity_ranker_joint_policy_20260512_test",
                    "model_confidence": "low",
                    "action": "watchlist",
                    "explanation_json": {"top_contributions": []},
                },
            ]

    tool = _ContextTools.__new__(_ContextTools)
    repo = _Repo()
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.kis_target_market = "nasdaq"
    tool._context = {
        "portfolio": {
            "cash_krw": 1_000.0,
            "positions": {
                "AAPL": {"quantity": 10.0, "avg_price_krw": 100.0},
                "MSFT": {"quantity": 5.0, "avg_price_krw": 200.0},
            },
        },
        "market_features": [
            {"ticker": "AAPL", "close_price_krw": 110.0, "volatility_20d": 0.2, "ret_20d": 0.08, "ret_5d": 0.03},
            {"ticker": "MSFT", "close_price_krw": 208.0, "volatility_20d": 0.1, "ret_20d": 0.04, "ret_5d": 0.01},
        ],
    }

    out = tool.portfolio_diagnosis(mdd_days=5, top_n=2)

    assert repo.ranker_kwargs is not None
    assert repo.ranker_kwargs["tickers"] == ["AAPL", "MSFT"]
    assert repo.ranker_kwargs["score_sources"] == ["joint_policy_v1"]
    assert repo.ranker_kwargs["markets"] == ["us"]
    joint = out["joint_policy"]
    assert joint["status"] == "ok"
    assert joint["coverage"] == {"held": 2, "scored": 2, "missing": 0}
    assert joint["weighted_score"] == pytest.approx(0.032102, abs=1e-6)
    assert joint["holdings"][0]["ticker"] == "AAPL"
    assert joint["holdings"][0]["score"] == 0.12
    assert joint["holdings"][0]["top_contributions"][0] == {"signal": "forecast_er", "contribution": 0.07}


def test_trade_performance_handles_mixed_naive_and_aware_execution_times() -> None:
    class _Repo:
        def filled_execution_reports_since(self, **kwargs):
            _ = kwargs
            return [
                {
                    "agent_id": "gemini",
                    "ticker": "AAPL",
                    "side": "BUY",
                    "filled_qty": 1,
                    "avg_price_krw": 100_000,
                    "created_at": datetime(2026, 5, 1, 9, 0, 0),
                },
                {
                    "agent_id": "gemini",
                    "ticker": "AAPL",
                    "side": "SELL",
                    "filled_qty": 1,
                    "avg_price_krw": 110_000,
                    "created_at": datetime(2026, 5, 3, 10, 0, 0, tzinfo=timezone.utc),
                },
            ]

    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _Repo()
    tool.settings = load_settings()
    tool.agent_id = "gemini"
    tool.tenant_id = "local"
    tool._context = {}

    out = tool.trade_performance(lookback_days=30)

    assert out["round_trips"]["closed"] == 1
    assert out["round_trips"]["avg_return_pct"] == 10.0
    assert out["round_trips"]["avg_holding_days"] == 2
    assert out["recent_streak"]["last_5"][0]["ticker"] == "AAPL"


def test_portfolio_diagnosis_aligns_benchmark_period_with_current_sleeve_return(monkeypatch) -> None:
    tool = _ContextTools.__new__(_ContextTools)
    repo = _RepoForPortfolioDiagnosisExact()
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.kis_target_market = "nasdaq"
    tool._context = {
        "portfolio": {
            "cash_krw": 1_000.0,
            "positions": {
                "AAPL": {"quantity": 10.0, "avg_price_krw": 100.0},
                "MSFT": {"quantity": 5.0, "avg_price_krw": 200.0},
            },
        },
        "market_features": [
            {"ticker": "AAPL", "close_price_krw": 110.0, "volatility_20d": 0.2, "ret_20d": 0.08, "ret_5d": 0.03},
            {"ticker": "MSFT", "close_price_krw": 208.0, "volatility_20d": 0.1, "ret_20d": 0.04, "ret_5d": 0.01},
        ],
        "performance": {
            "initialized_at": "2026-01-01T00:00:00+00:00",
            "cumulative_pnl_ratio": 0.30,
            "current_sleeve_initialized_at": "2026-03-01T00:00:00+00:00",
            "current_sleeve_pnl_ratio": 0.05,
            "pnl_ratio": 0.05,
        },
        "risk_policy": {
            "min_cash_buffer_ratio": 0.10,
            "max_position_ratio": 0.60,
        },
    }
    monkeypatch.setattr(
        "arena.agents.adk_context_tools.utc_now",
        lambda: datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc),
    )

    out = tool.portfolio_diagnosis(mdd_days=5, top_n=2)

    assert repo.frame_calls == [
        {
            "tickers": ["QQQ"],
            "start": datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc).date(),
            "end": datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc).date(),
            "sources": tool._sources(),
        },
        {
            "tickers": ["QQQ"],
            "start": datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).date(),
            "end": datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc).date(),
            "sources": tool._sources(),
        },
    ]
    assert out["benchmark"]["period_alignment"] == "exact"
    assert out["benchmark"]["portfolio_start_date"] == "2026-03-01"
    assert out["benchmark"]["benchmark_start_date"] == "2026-03-03"
    assert out["benchmark"]["benchmark_end_date"] == "2026-03-27"
    assert out["benchmark"]["agent_return_metric"] == "current_sleeve_pnl_ratio"
    assert out["benchmark"]["comparison_scope"] == "current_sleeve"
    assert out["benchmark"]["currency_basis"] == "KRW"
    assert out["benchmark"]["price_basis"] == "close_price_krw"
    assert out["benchmark"]["source_basis"] == "quote_aware"
    assert out["benchmark"]["agent_return"] == pytest.approx(0.05, abs=1e-6)
    assert out["benchmark"]["excess_return_vs_benchmark"] == pytest.approx(0.05, abs=1e-6)
    assert out["benchmark"]["return"] == pytest.approx(0.0, abs=1e-6)
    assert "alpha_vs_benchmark" not in out["benchmark"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["current_sleeve"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["cumulative"]
    assert "not risk-adjusted alpha" in out["benchmark"]["alpha_definition"]
    assert "quote-aware KRW" in out["benchmark"]["note"]
    assert "2026-03-01 -> 2026-03-03" in out["benchmark"]["note"]
    assert out["benchmark"] == out["benchmarks"]["current_sleeve"]
    assert set(out["benchmarks"]) == {"current_sleeve", "cumulative"}
    cumulative = out["benchmarks"]["cumulative"]
    assert cumulative["comparison_scope"] == "cumulative"
    assert cumulative["portfolio_start_date"] == "2026-01-01"
    assert cumulative["benchmark_start_date"] == "2026-01-02"
    assert cumulative["agent_return_metric"] == "cumulative_pnl_ratio"
    assert cumulative["agent_return"] == pytest.approx(0.30, abs=1e-6)
    assert cumulative["return_krw"] == pytest.approx((100.0 / 90.0) - 1.0, abs=1e-6)
    assert cumulative["excess_return_vs_benchmark"] == pytest.approx(0.30 - ((100.0 / 90.0) - 1.0), abs=1e-6)
    assert "cumulative/TWR" in cumulative["note"]


def test_portfolio_diagnosis_logs_warning_when_mdd_calculation_fails(caplog) -> None:
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForPortfolioDiagnosisRaises()
    tool.settings = load_settings()
    tool.settings.kis_target_market = "nasdaq"
    tool.agent_id = "gpt"
    tool._context = {
        "portfolio": {
            "cash_krw": 1_000.0,
            "positions": {
                "AAPL": {"quantity": 10.0, "avg_price_krw": 100.0},
                "MSFT": {"quantity": 5.0, "avg_price_krw": 200.0},
            },
        },
        "market_features": [
            {"ticker": "AAPL", "close_price_krw": 110.0, "volatility_20d": 0.2, "ret_20d": 0.08, "ret_5d": 0.03},
            {"ticker": "MSFT", "close_price_krw": 208.0, "volatility_20d": 0.1, "ret_20d": 0.04, "ret_5d": 0.01},
        ],
        "risk_policy": {
            "min_cash_buffer_ratio": 0.10,
            "max_position_ratio": 0.60,
        },
    }

    with caplog.at_level("WARNING"):
        out = tool.portfolio_diagnosis(mdd_days=5, top_n=2)

    assert "mdd" not in out
    assert "portfolio diagnosis MDD calculation failed" in caplog.text


def test_compact_portfolio_diagnosis_no_hrp_allocation() -> None:
    out = _compact_tool_result_for_prompt(
        "portfolio_diagnosis",
        {
            "risk_contribution": [
                {"ticker": "AAPL", "rc": 0.6},
                {"ticker": "MSFT", "rc": 0.4},
            ],
            "concentration_top3": 0.82,
            "hhi": 0.34,
            "momentum_20d_weighted": 0.07,
            "momentum_5d_weighted": 0.02,
            "volatility_20d_weighted": 0.18,
            "joint_policy": {
                "status": "ok",
                "score_source": "joint_policy_v1",
                "coverage": {"held": 2, "scored": 2, "missing": 0},
                "weighted_score": 0.032102,
                "holdings": [
                    {"ticker": "AAPL", "weight": 0.35, "score": 0.12, "rank": 4},
                    {"ticker": "MSFT", "weight": 0.33, "score": -0.03, "rank": 9},
                ],
            },
            "mdd": {"days": 60, "value": -0.12},
            "benchmark": {
                "ticker": "SPY",
                "return_krw": 0.05,
                "agent_return": -0.01,
                "alpha_vs_benchmark": -0.06,
                "alpha_definition": "simple excess return: agent_return - benchmark return_krw; not risk-adjusted alpha",
            },
            "benchmarks": {
                "current_sleeve": {
                    "ticker": "SPY",
                    "return_krw": 0.05,
                    "agent_return": -0.01,
                    "alpha_vs_benchmark": -0.06,
                    "alpha_definition": "simple excess return: agent_return - benchmark return_krw; not risk-adjusted alpha",
                },
                "cumulative": {
                    "ticker": "SPY",
                    "return_krw": 0.08,
                    "agent_return": 0.02,
                    "alpha_vs_benchmark": -0.06,
                    "alpha_definition": "simple excess return: agent_return - benchmark return_krw; not risk-adjusted alpha",
                },
            },
        },
    )

    assert "hrp_allocation" not in out
    assert "alpha_vs_benchmark" not in out["benchmark"]
    assert out["benchmark"]["excess_return_vs_benchmark"] == -0.06
    assert out["joint_policy"]["weighted_score"] == 0.032102
    assert out["joint_policy"]["holdings"][0] == {"ticker": "AAPL", "weight": 0.35, "score": 0.12, "rank": 4}
    assert "not risk-adjusted alpha" in out["benchmark"]["alpha_definition"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["current_sleeve"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["cumulative"]
    assert out["benchmarks"]["current_sleeve"]["excess_return_vs_benchmark"] == -0.06
