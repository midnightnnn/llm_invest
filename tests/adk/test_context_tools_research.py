from __future__ import annotations

from arena.agents.adk_agents import _ContextTools
from arena.config import load_settings
from tests.adk.context_tools_helpers import (
    _RepoForPeerLessons,
    _RepoForResearchBriefingFallback,
    _VectorStoreForPeerLessons,
)


class _RepoForStructuredResearch:
    def get_research_briefings(self, *, tickers=None, categories=None, limit=10, trading_mode="paper", tenant_id=None):
        _ = (tickers, categories, trading_mode, tenant_id)
        return [
            {
                "briefing_id": "brf_vz",
                "ticker": "VZ",
                "category": "held",
                "headline": "VZ research brief",
                "summary": "EPS beat; FY26 guide raised; postpaid adds positive.",
                "detail_json": {
                    "summary": "EPS beat; FY26 guide raised; postpaid adds positive.",
                    "key_points": ["Q1 EPS beat", "FY26 guide raised"],
                    "risks": ["valuation after rally"],
                    "sentiment": "positive",
                    "confidence": 0.74,
                },
            }
        ][:limit]


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


def test_get_research_briefing_returns_full_structured_schema() -> None:
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForStructuredResearch()
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.tenant_id = "tenant-a"

    out = tool.get_research_briefing(tickers=["VZ"], limit=1)

    assert out[0]["detail_json"]["key_points"] == ["Q1 EPS beat", "FY26 guide raised"]
    assert out[0]["detail_json"]["risks"] == ["valuation after rally"]
    assert out[0]["detail_json"]["confidence"] == 0.74
