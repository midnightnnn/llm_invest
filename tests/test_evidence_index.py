from __future__ import annotations

from arena.agents.evidence_index import build_evidence_index, record_tool_evidence
from arena.agents.tool_evidence import extract_evidence


def _screen_market_result() -> list[dict]:
    return [
        {"ticker": "cvx", "bucket": "defensive", "bucket_rank": 4, "score": 0.83, "ret_20d": 0.13},
        {"ticker": "MUSA", "bucket": "defensive", "bucket_rank": 7, "score": 0.71},
    ]


def test_extract_evidence_screen_market_normalizes_tickers():
    events = extract_evidence("screen_market", _screen_market_result())
    assert {e["ticker"] for e in events} == {"CVX", "MUSA"}
    cvx = next(e for e in events if e["ticker"] == "CVX")
    assert cvx["scope"] == "ticker"
    assert cvx["summary"]["bucket"] == "defensive"
    assert cvx["summary"]["bucket_rank"] == 4
    assert "ticker" not in cvx["summary"]


def test_extract_evidence_unknown_tool_returns_empty():
    assert extract_evidence("fetch_ohlcv", [{"ticker": "AAPL"}]) == []
    assert extract_evidence("", {"anything": 1}) == []


def test_extract_evidence_technical_signals_rows_shape():
    result = {
        "tickers": ["CVX"],
        "rows": [
            {
                "ticker": "CVX",
                "rsi_14": 58.3,
                "rsi_state": "neutral",
                "trend_state": "uptrend",
                "macd": {"state": "bullish"},
                "moving_averages": {"price_vs_sma20": 0.02},
            }
        ],
    }
    events = extract_evidence("technical_signals", result)
    assert len(events) == 1
    assert events[0]["ticker"] == "CVX"
    assert events[0]["summary"]["macd_state"] == "bullish"
    assert events[0]["summary"]["price_vs_sma20"] == 0.02


def test_extract_evidence_portfolio_diagnosis_portfolio_scope():
    result = {"concentration_top3": 1.0, "hhi": 0.51, "risk_contribution": {"CCEP": 0.42}}
    events = extract_evidence("portfolio_diagnosis", result)
    assert len(events) == 1
    assert events[0]["scope"] == "portfolio"
    assert "ticker" not in events[0]
    assert events[0]["summary"]["hhi"] == 0.51


def test_extract_evidence_portfolio_diagnosis_error_is_empty():
    assert extract_evidence("portfolio_diagnosis", {"error": "no active positions"}) == []


def test_record_tool_evidence_appends_phase_and_tool():
    log: list[dict] = []
    record_tool_evidence(log, tool_name="screen_market", result=_screen_market_result(), phase="draft")
    assert len(log) == 2
    assert all(entry["tool"] == "screen_market" for entry in log)
    assert all(entry["phase"] == "draft" for entry in log)
    assert all(entry["scope"] == "ticker" for entry in log)


def test_build_evidence_index_role_assignment_and_sort():
    log: list[dict] = []
    record_tool_evidence(log, tool_name="screen_market", result=_screen_market_result(), phase="draft")
    record_tool_evidence(
        log,
        tool_name="forecast_returns",
        result=[{"ticker": "CCEP", "consensus": "STRONG_BUY", "prob_up": 1.0}],
        phase="draft",
    )

    index = build_evidence_index(log, held_tickers={"CCEP"})
    cases = index["security_cases"]
    assert [c["ticker"] for c in cases] == ["CCEP", "CVX", "MUSA"]
    assert cases[0]["role"] == "held"
    assert cases[1]["role"] == "candidate"


def test_build_evidence_index_merges_multiple_tools_per_ticker():
    log: list[dict] = []
    record_tool_evidence(log, tool_name="screen_market", result=_screen_market_result(), phase="draft")
    record_tool_evidence(
        log,
        tool_name="technical_signals",
        result={"rows": [{"ticker": "CVX", "rsi_state": "neutral", "trend_state": "uptrend"}]},
        phase="draft",
    )

    index = build_evidence_index(log, held_tickers=set())
    cvx = next(c for c in index["security_cases"] if c["ticker"] == "CVX")
    assert cvx["sources"] == ["screen_market", "technical_signals"]
    assert len(cvx["latest_evidence"]) == 2
    tools = {ev["tool"] for ev in cvx["latest_evidence"]}
    assert tools == {"screen_market", "technical_signals"}


def test_build_evidence_index_duplicate_tool_replaces_previous():
    log: list[dict] = []
    record_tool_evidence(
        log,
        tool_name="technical_signals",
        result={"rows": [{"ticker": "CVX", "rsi_state": "oversold", "trend_state": "downtrend"}]},
        phase="draft",
    )
    record_tool_evidence(
        log,
        tool_name="technical_signals",
        result={"rows": [{"ticker": "CVX", "rsi_state": "neutral", "trend_state": "uptrend"}]},
        phase="execution",
    )

    index = build_evidence_index(log, held_tickers=set())
    cvx = next(c for c in index["security_cases"] if c["ticker"] == "CVX")
    assert cvx["sources"] == ["technical_signals"]
    assert len(cvx["latest_evidence"]) == 1
    latest = cvx["latest_evidence"][0]
    assert latest["summary"]["rsi_state"] == "neutral"
    assert latest["phase"] == "execution"


def test_build_evidence_index_non_ticker_events_go_to_cycle_evidence():
    log: list[dict] = []
    record_tool_evidence(
        log,
        tool_name="portfolio_diagnosis",
        result={"hhi": 0.51, "concentration_top3": 1.0},
        phase="draft",
    )

    index = build_evidence_index(log, held_tickers={"CCEP"})
    assert index["security_cases"] == []
    assert len(index["cycle_evidence"]) == 1
    ev = index["cycle_evidence"][0]
    assert ev["tool"] == "portfolio_diagnosis"
    assert ev["scope"] == "portfolio"
    assert ev["summary"]["hhi"] == 0.51
    assert ev["phase"] == "draft"


def test_build_evidence_index_ignores_unknown_tools():
    log: list[dict] = []
    record_tool_evidence(log, tool_name="fetch_ohlcv", result={"rows": []}, phase="draft")
    index = build_evidence_index(log, held_tickers=set())
    assert index == {"cycle_evidence": [], "security_cases": []}


def test_build_evidence_index_empty_log():
    index = build_evidence_index([], held_tickers={"CCEP"})
    assert index == {"cycle_evidence": [], "security_cases": []}


def test_extract_evidence_get_research_briefing_splits_by_ticker():
    result = [
        {"ticker": "CVX", "headline": "Oil rally", "category": "single-name"},
        {"headline": "Macro update", "category": "global"},
    ]
    events = extract_evidence("get_research_briefing", result)
    scopes = [e["scope"] for e in events]
    assert scopes == ["ticker", "market"]
    assert events[0]["ticker"] == "CVX"
    assert "ticker" not in events[1]
