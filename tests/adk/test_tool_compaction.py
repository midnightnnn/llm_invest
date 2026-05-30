from __future__ import annotations

import asyncio
from types import SimpleNamespace

from arena.agents.adk_agents import _compact_tool_result_for_prompt
from arena.agents.adk_runner_bootstrap import build_tool_wrapper
from arena.tools.registry import ToolEntry


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
                },
                {
                    "ticker": "MSFT",
                    "price": 250.0,
                    "rsi_14": 48.0,
                    "rsi_state": "neutral",
                    "macd": {"line": 0.1, "signal": 0.1, "hist": 0.0, "state": "neutral"},
                    "moving_averages": {"sma_20": 252.0, "sma_50": 248.0, "price_vs_sma20": -0.0079},
                    "bollinger_20_2": {"upper": 260.0, "mid": 252.0, "lower": 244.0, "state": "inside_bands"},
                    "trend_state": "sideways",
                },
            ],
        },
    )

    assert "count" not in out
    assert "tickers" not in out
    assert "compaction" not in out
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


def test_compact_tool_result_earnings_calendar_omits_untruncated_derived_meta() -> None:
    out = _compact_tool_result_for_prompt(
        "earnings_calendar",
        {
            "ticker": None,
            "tickers": ["AAPL", "MSFT"],
            "start_date": "2026-05-19",
            "days_ahead": 14,
            "count": 2,
            "rows": [
                {"date": "2026-05-20", "symbol": "AAPL", "name": "Apple", "time": "AMC", "eps_forecast": "1.60"},
                {"date": "2026-05-21", "symbol": "MSFT", "name": "Microsoft", "time": "BMO", "eps_forecast": "3.10"},
            ],
        },
        args={"tickers": ["AAPL", "MSFT"]},
    )

    assert "compaction" not in out
    assert "count" not in out
    assert "tickers" not in out
    assert [row["symbol"] for row in out["rows"]] == ["AAPL", "MSFT"]


def test_compact_tool_result_reddit_keeps_requested_tickers_when_no_rows() -> None:
    out = _compact_tool_result_for_prompt(
        "fetch_reddit_sentiment",
        {"tickers": ["AAPL", "MSFT"], "count": 0, "rows": []},
        args={"tickers": ["AAPL", "MSFT"]},
    )

    assert out["tickers"] == ["AAPL", "MSFT"]
    assert out["rows"] == []
    assert "count" not in out
    assert "compaction" not in out


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


def test_compact_tool_result_forecast_returns_uses_model_direction() -> None:
    out = _compact_tool_result_for_prompt(
        "forecast_returns",
        [
            {
                "run_date": "2026-05-12",
                "ticker": "AAPL",
                "exp_return_period": 0.042,
                "forecast_horizon": 20,
                "forecast_model": "ensemble_wmae",
                "is_stacked": True,
                "forecast_score": 0.12,
                "prob_up": 0.82,
                "model_votes_up": 6,
                "model_votes_total": 7,
                "consensus": "STRONG_BUY",
            }
        ],
    )

    assert out[0]["ticker"] == "AAPL"
    assert out[0]["model_direction"] == "MODEL_UP_STRONG"
    assert "consensus" not in out[0]


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


def test_compact_recommend_opportunities_drops_empty_optimizer_and_duplicate_confidence_alias() -> None:
    out = _compact_tool_result_for_prompt(
        "recommend_opportunities",
        {
            "status": "ok",
            "recommendations": [
                {
                    "ticker": "AAPL",
                    "profile": "balanced",
                    "bucket": "momentum",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.81,
                    "score_source": "learned_ic",
                    "ranker_version": "v1",
                    "confidence": "medium",
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "reason_for": "strong trend",
                    "reason_risk": "valuation risk",
                }
            ],
            "optimizer": {},
        },
    )

    row = out["recommendations"][0]
    assert "optimizer" not in out
    assert "confidence" not in row
    assert row["model_confidence"] == "medium"
    assert row["recommendation_score"] == 0.81
    assert row["reason_for"] == "strong trend"
    assert row["reason_risk"] == "valuation risk"


def test_compact_memory_context_candidate_uses_lean_payload_without_summary_truncation() -> None:
    long_reason = (
        "Learned IC ranker score=+0.8661; contribs: momentum_20d(+0.2992) "
        "meanrev_5d(+0.2325) pullback(+0.1566) lowvol(+0.1165); prob_up=50.0%"
    )
    out = _compact_tool_result_for_prompt(
        "macro_snapshot",
        {
            "status": "ok",
            "_memory_context": [
                {
                    "event_id": "mem_007610",
                    "created_date": "2026-05-07",
                    "event_type": "candidate_watchlist",
                    "summary": "007610 candidate_watchlist: " + ("x" * 400),
                    "importance_score": 0.38,
                    "payload": {
                        "source": "candidate_discovery",
                        "ticker": "007610",
                        "candidate_status": "watchlist",
                        "source_tools": ["recommend_opportunities:aggressive"],
                        "analyzed_by": ["forecast_returns", "get_fundamentals", "technical_signals"],
                        "last_seen_rank": 1,
                        "discovery_evidence": {
                            "score": 0.86605,
                            "reason_for": long_reason,
                            "reason_risk": "blended_oos_ic=-0.024; signals_scored=17; model_confidence=low",
                        },
                    },
                }
            ],
        },
    )

    memory = out["_memory_context"][0]
    assert memory["event_id"] == "mem_007610"
    assert memory["d"] == "2026-05-07"
    assert memory["type"] == "candidate_watchlist"
    assert memory["t"] == "007610"
    assert memory["src"] == "recommend_opportunities:aggressive"
    assert memory["checked"] == ["forecast_returns", "get_fundamentals", "technical_signals"]
    assert memory["rank"] == 1
    assert memory["score"] == 0.86605
    assert memory["why"] == long_reason
    assert memory["risk"] == "blended_oos_ic=-0.024; signals_scored=17; model_confidence=low"
    assert "xxx" not in str(memory)
    assert "..." not in memory["why"]


def test_compact_memory_context_generic_keeps_full_summary() -> None:
    summary = "Risk lesson: " + ("position sizing discipline matters. " * 20)
    out = _compact_tool_result_for_prompt(
        "macro_snapshot",
        {
            "status": "ok",
            "_memory_context": [
                {
                    "event_id": "mem_lesson",
                    "created_date": "2026-05-01",
                    "event_type": "strategy_reflection",
                    "summary": summary,
                    "importance_score": 0.7,
                    "outcome_label": "win",
                }
            ],
        },
    )

    memory = out["_memory_context"][0]
    assert memory["summary"] == summary
    assert not memory["summary"].endswith("...")
    assert memory["outcome"] == "win"


def _macro_indicator(
    value: float,
    *,
    date: str = "2026-05-30",
    unit: str = "%",
    source: str = "fred",
    series_id: str | None = None,
    class_name: str | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "value": value,
        "date": date,
        "unit": unit,
        "source": source,
    }
    if series_id:
        item["series_id"] = series_id
    if class_name:
        item["class_name"] = class_name
    return item


def test_compact_macro_snapshot_balances_fred_and_ecos_groups() -> None:
    indicators = {
        "fed_funds_rate": _macro_indicator(3.62, series_id="DFF"),
        "sofr": _macro_indicator(3.62, series_id="SOFR"),
        "treasury_10y": _macro_indicator(4.45, series_id="DGS10"),
        "yield_spread_10y_3m": _macro_indicator(0.76, unit="pp", series_id="DGS10-DGS3MO"),
        "core_cpi_yoy": _macro_indicator(2.74, series_id="CPILFESL"),
        "real_gdp": _macro_indicator(24152.656, unit="billions chained 2017 dollars", series_id="GDPC1"),
        "high_yield_oas": _macro_indicator(2.72, series_id="BAMLH0A0HYM2"),
        "vix": _macro_indicator(15.74, unit="index", series_id="VIXCLS"),
        "case_shiller_home_price_yoy": _macro_indicator(3.12, series_id="CSUSHPINSA"),
        "bok_base_rate": _macro_indicator(2.5, source="ecos", class_name="시장금리"),
        "kr_treasury_5y": _macro_indicator(3.924, source="ecos", class_name="시장금리"),
        "kr_m2_money_supply": _macro_indicator(4143515.8, unit="십억원", source="ecos", class_name="통화량"),
        "kr_household_credit": _macro_indicator(1993110.8, unit="십억원", source="ecos", class_name="예금/대출금"),
        "jpy_krw": _macro_indicator(945.56, unit="원", source="ecos", class_name="환율"),
        "kr_current_account": _macro_indicator(37327.1, unit="백만달러", source="ecos", class_name="국제수지"),
        "kr_all_industry_production": _macro_indicator(117.8, unit="2020=100", source="ecos", class_name="생산"),
        "kr_cpi": _macro_indicator(119.37, unit="2020=100", source="ecos", class_name="소비자/생산자 물가"),
        "kr_consumer_sentiment_index": _macro_indicator(106.1, unit="", source="ecos", class_name="심리지표"),
        "kr_house_price_index": _macro_indicator(101.4, unit="2025.03=100", source="ecos", class_name="부동산 가격"),
        "dubai_oil": _macro_indicator(105.3, unit="달러/배럴", source="ecos", class_name="국제원자재가격"),
    }
    for idx in range(60):
        indicators[f"unused_{idx}"] = _macro_indicator(float(idx), series_id=f"UNUSED{idx}")

    out = _compact_tool_result_for_prompt(
        "macro_snapshot",
        {
            "as_of": "2026-05-30",
            "source": "fred+ecos",
            "coverage": {
                "fred": {"requested": 40, "returned": 40},
                "ecos": {"requested": 101, "returned": 100},
            },
            "indicators": indicators,
            "groups": {"large_raw_group": indicators},
        },
    )

    assert out["coverage"] == {"fred": "40/40", "ecos": "100/101"}
    assert out["key_indicators"]["us_policy"]["sofr"]["id"] == "SOFR"
    assert out["key_indicators"]["us_inflation"]["core_cpi_yoy"]["src"] == "fred"
    assert out["key_indicators"]["kr_money_credit"]["kr_m2_money_supply"]["src"] == "ecos"
    assert out["key_indicators"]["kr_fx_external"]["jpy_krw"]["id"] == "환율"
    assert out["key_indicators"]["kr_housing_commodities"]["dubai_oil"]["u"] == "달러/배럴"
    assert "indicators" not in out
    assert "groups" not in out
    assert "unused_0" not in str(out)
    assert out["compaction"]["raw_indicator_count"] == 80
    assert out["compaction"]["omitted_indicator_count"] > 50


def test_compact_macro_snapshot_preserves_regime_card_and_requested_drilldown() -> None:
    out = _compact_tool_result_for_prompt(
        "macro_snapshot",
        {
            "as_of": "2026-05-30",
            "source": "ecos",
            "depth": "full",
            "data_mode": "historical",
            "coverage": {"ecos": {"requested": 68, "returned": 68}},
            "regime_card": {"fx_external": "pressure_high", "rates_curve": "easing"},
            "market_implications": {"usd_exposure": "positive_but_expensive"},
            "groups": {
                "external": {
                    "state": "pressure_high",
                    "evidence": [
                        {
                            "k": "usd_krw",
                            "v": 1410.0,
                            "d": "2026-05-30",
                            "u": "KRW per USD",
                            "chg_3m": 50.0,
                            "z": 1.22,
                            "series": [
                                {"d": "2026-04-01", "v": 1380.0},
                                {"d": "2026-05-30", "v": 1410.0},
                            ],
                        }
                    ],
                }
            },
            "notable_movers": [{"k": "usd_krw", "why": "high percentile and rising"}],
            "omitted": {"indicator_count": 80, "reason": "available via focus/depth"},
            "indicators": {
                "usd_krw": {"value": 1410.0, "date": "2026-05-30", "unit": "KRW per USD"},
                "unused": {"value": 1.0},
            },
        },
    )

    assert out["depth"] == "full"
    assert out["regime_card"]["fx_external"] == "pressure_high"
    assert out["market_implications"]["usd_exposure"] == "positive_but_expensive"
    assert out["groups"]["external"]["evidence"][0]["series"][-1]["v"] == 1410.0
    assert "unused" not in str(out)


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
        return [
            {
                "event_id": "mem_macro",
                "created_date": "2026-05-07",
                "event_type": "candidate_watchlist",
                "payload": {
                    "source": "candidate_discovery",
                    "ticker": "007610",
                    "candidate_status": "watchlist",
                    "source_tools": ["recommend_opportunities:aggressive"],
                    "analyzed_by": ["forecast_returns"],
                    "last_seen_rank": 1,
                    "discovery_evidence": {
                        "score": 0.86605,
                        "reason_for": "Full reason survives in macro tool memory context.",
                        "reason_risk": "Full risk survives too.",
                    },
                },
            }
        ]

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

    out = asyncio.run(wrapper())

    memory = out["_memory_context"][0]
    assert memory["t"] == "007610"
    assert memory["src"] == "recommend_opportunities:aggressive"
    assert memory["why"] == "Full reason survives in macro tool memory context."
    assert memory["risk"] == "Full risk survives too."
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
    assert "eligible_count" not in out
    assert "excluded_count" not in out
    assert out["excluded"] == ["XYZ"]
    assert out["rows"][0]["ticker"] == "AAPL"
    assert out["errors"][0]["ticker"] == "XYZ"


def test_compact_portfolio_diagnosis_drops_duplicate_benchmark_alias() -> None:
    benchmark = {
        "ticker": "SPY",
        "return_krw": 0.05,
        "agent_return": -0.01,
        "excess_return_vs_benchmark": -0.06,
    }
    out = _compact_tool_result_for_prompt(
        "portfolio_diagnosis",
        {
            "risk_contribution": [{"ticker": "AAPL", "rc": 0.6}],
            "concentration_top3": 0.82,
            "hhi": 0.34,
            "momentum_20d_weighted": 0.07,
            "momentum_5d_weighted": 0.02,
            "volatility_20d_weighted": 0.18,
            "benchmark": benchmark,
            "benchmarks": {"current_sleeve": benchmark},
        },
    )

    assert "benchmark" not in out
    assert out["primary_benchmark_scope"] == "current_sleeve"
    assert out["benchmarks"]["current_sleeve"]["ticker"] == "SPY"
