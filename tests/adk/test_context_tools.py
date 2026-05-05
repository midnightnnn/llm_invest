from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from arena.agents.adk_agents import _ContextTools, _compact_tool_result_for_prompt
from arena.config import load_settings


class _RepoForPortfolioDiagnosis:
    def get_daily_closes(self, tickers, lookback_days, sources=None):
        _ = lookback_days, sources
        base = {
            "AAPL": [100.0, 101.0, 103.0, 104.0, 106.0, 108.0, 109.0, 111.0, 112.0, 114.0, 116.0, 118.0],
            "MSFT": [200.0, 199.0, 198.0, 201.0, 202.0, 204.0, 205.0, 207.0, 209.0, 210.0, 212.0, 214.0],
            "QQQ": [300.0, 301.0, 302.0, 304.0, 306.0, 307.0, 309.0, 311.0, 312.0, 314.0, 316.0, 318.0],
        }
        return {ticker: base.get(ticker, []) for ticker in tickers}

    def get_daily_close_frame(self, *, tickers, start, end, sources=None, price_field="close_price_krw"):
        _ = (sources, price_field)
        series = {
            "QQQ": [
                ("2026-01-02", 300.0),
                ("2026-01-03", 310.0),
                ("2026-01-04", 315.0),
            ],
        }
        frame = pd.DataFrame(
            {
                token: [px for _, px in series.get(token, [])]
                for token in tickers
                if token in series
            },
            index=pd.to_datetime([ts for ts, _ in series.get(next(iter(tickers), ""), [])]),
        )
        if frame.empty:
            return frame
        mask = (frame.index.date >= start) & (frame.index.date <= end)
        return frame.loc[mask]


class _RepoForPortfolioDiagnosisExact(_RepoForPortfolioDiagnosis):
    def __init__(self) -> None:
        self.frame_calls: list[dict[str, object]] = []

    def get_daily_close_frame(self, *, tickers, start, end, sources=None):  # noqa: ANN001
        self.frame_calls.append(
            {
                "tickers": list(tickers),
                "start": start,
                "end": end,
                "sources": list(sources) if isinstance(sources, list) else sources,
            }
        )
        frame = pd.DataFrame(
            {"QQQ": [90.0, 100.0, 100.0]},
            index=pd.to_datetime(["2026-01-02", "2026-03-03", "2026-03-27"]),
        )
        mask = (frame.index.date >= start) & (frame.index.date <= end)
        return frame.loc[mask]


class _RepoForPortfolioDiagnosisRaises(_RepoForPortfolioDiagnosis):
    def get_daily_closes(self, tickers, lookback_days, sources=None):
        if int(lookback_days) <= 10:
            raise RuntimeError("no closes")
        return super().get_daily_closes(tickers, lookback_days, sources=sources)


class _RepoForPeerLessons:
    def memory_events_by_ids_any_agent(self, *, event_ids, trading_mode="paper", tenant_id=None):
        _ = (trading_mode, tenant_id)
        rows = {
            "mem_peer": {
                "event_id": "mem_peer",
                "agent_id": "gemini",
                "payload_json": json.dumps({"source": "thesis_chain_compaction"}),
            },
            "mem_manual": {
                "event_id": "mem_manual",
                "agent_id": "claude",
                "payload_json": json.dumps({"source": "manual_note"}),
            },
        }
        return [rows[eid] for eid in event_ids if eid in rows]


class _VectorStoreForPeerLessons:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search_peer_lessons(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "event_id": "mem_peer",
                "agent_id": "gemini",
                "summary": "Trim single-name exposure after fast gains.",
                "created_date": "2026-03-07",
            },
            {
                "event_id": "mem_manual",
                "agent_id": "claude",
                "summary": "Manual reflection that should be filtered out.",
                "created_date": "2026-03-06",
            },
        ]


class _RepoForResearchBriefingFallback:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_research_briefings(self, *, tickers=None, categories=None, limit=10, trading_mode="paper", tenant_id=None):
        self.calls.append(
            {
                "tickers": list(tickers) if tickers else None,
                "categories": list(categories) if categories else None,
                "limit": limit,
                "trading_mode": trading_mode,
                "tenant_id": tenant_id,
            }
        )
        tenant = str(tenant_id or "").strip().lower()
        if tenant == "tenant-a":
            return []
        if tenant == "midnightnnn":
            rows = [
                {
                    "briefing_id": "pub_global",
                    "category": "global_market",
                    "ticker": "GLOBAL",
                    "headline": "Global",
                    "summary": "global summary",
                    "sources": "[]",
                },
                {
                    "briefing_id": "pub_geo",
                    "category": "geopolitical",
                    "ticker": "GEOPOLITICAL",
                    "headline": "Geo",
                    "summary": "geo summary",
                    "sources": "[]",
                },
                {
                    "briefing_id": "pub_sector",
                    "category": "sector_trends",
                    "ticker": "SECTOR",
                    "headline": "Sector",
                    "summary": "sector summary",
                    "sources": "[]",
                },
            ]
            if categories:
                allowed = {str(token).strip().lower() for token in categories if str(token).strip()}
                rows = [row for row in rows if str(row.get("category") or "").strip().lower() in allowed]
            return rows[:limit]
        return []


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
    assert "not risk-adjusted alpha" in out["benchmark"]["alpha_definition"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["current_sleeve"]
    assert "alpha_vs_benchmark" not in out["benchmarks"]["cumulative"]
    assert out["benchmarks"]["current_sleeve"]["excess_return_vs_benchmark"] == -0.06


def test_search_peer_lessons_returns_only_compactor_reflections() -> None:
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForPeerLessons()
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.agent_id = "gpt"
    tool.tenant_id = "local"
    tool._vector_store = _VectorStoreForPeerLessons()

    out = tool.search_peer_lessons("concentration risk", limit=5)

    assert len(out) == 1
    assert out[0]["event_id"] == "mem_peer"
    assert out[0]["agent_id"] == "gemini"
    assert out[0]["author_id"] == "gemini"
    assert out[0]["memory_source"] == "thesis_chain_compaction"
    assert tool._vector_store.calls[0]["agent_id"] == "gpt"


def test_get_research_briefing_falls_back_to_public_demo_for_no_key_tenant(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_PUBLIC_DEMO_TENANT", "midnightnnn")
    monkeypatch.delenv("ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)

    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForResearchBriefingFallback()
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.settings.research_enabled = True
    tool.settings.gemini_api_key = ""
    tool.settings.research_gemini_api_key = ""
    tool.settings.research_gemini_source = ""
    tool.settings.research_gemini_source_tenant = ""
    tool.tenant_id = "tenant-a"

    out = tool.get_research_briefing(limit=2)

    assert [row["briefing_id"] for row in out] == ["pub_global", "pub_geo"]
    assert all(row["public_fallback"] is True for row in out)
    assert all(row["source_tenant_id"] == "midnightnnn" for row in out)
    assert tool.repo.calls == [
        {
            "tickers": None,
            "categories": None,
            "limit": 2,
            "trading_mode": "paper",
            "tenant_id": "tenant-a",
        },
        {
            "tickers": None,
            "categories": ["global_market", "geopolitical", "sector_trends"],
            "limit": 2,
            "trading_mode": "paper",
            "tenant_id": "midnightnnn",
        },
    ]


def test_get_research_briefing_does_not_fallback_for_ticker_queries(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_PUBLIC_DEMO_TENANT", "midnightnnn")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)

    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForResearchBriefingFallback()
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.settings.research_enabled = True
    tool.settings.gemini_api_key = ""
    tool.settings.research_gemini_api_key = ""
    tool.settings.research_gemini_source = ""
    tool.settings.research_gemini_source_tenant = ""
    tool.tenant_id = "tenant-a"

    out = tool.get_research_briefing(tickers=["AAPL"], limit=2)

    assert out == []
    assert tool.repo.calls == [
        {
            "tickers": ["AAPL"],
            "categories": None,
            "limit": 2,
            "trading_mode": "paper",
            "tenant_id": "tenant-a",
        }
    ]
