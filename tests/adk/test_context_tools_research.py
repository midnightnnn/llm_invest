from __future__ import annotations

import asyncio

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


class _RepoForResearchDocuments:
    def __init__(self) -> None:
        self.briefing_calls: list[dict] = []
        self.document_calls: list[dict] = []
        self.updates: list[dict] = []

    def get_research_briefings(self, *, tickers=None, categories=None, limit=10, trading_mode="paper", tenant_id=None):
        self.briefing_calls.append(
            {
                "tickers": tickers,
                "categories": categories,
                "limit": limit,
                "trading_mode": trading_mode,
                "tenant_id": tenant_id,
            }
        )
        return [
            {
                "briefing_id": "brf_aapl",
                "ticker": "AAPL",
                "category": "held",
                "headline": "AAPL Gemini-grounded research",
                "summary": "Gemini + Google Search research context.",
            }
        ][:limit]

    def get_research_documents(
        self,
        *,
        source_doc_ids=None,
        tickers=None,
        categories=None,
        limit=10,
        trading_mode="paper",
        tenant_id=None,
    ):
        self.document_calls.append(
            {
                "source_doc_ids": source_doc_ids,
                "tickers": tickers,
                "categories": categories,
                "limit": limit,
                "trading_mode": trading_mode,
                "tenant_id": tenant_id,
            }
        )
        return [
            {
                "source_doc_id": "research:google_news:aapl:abc",
                "published_at": "2026-06-03T12:00:00+00:00",
                "source": "google_news",
                "feed_id": "google_news_aapl",
                "category": "held",
                "market": "us",
                "ticker": "AAPL",
                "publisher": "CNBC",
                "title": "Apple shares rise after earnings",
                "source_url": "https://example.test/aapl",
                "snippet": "Apple reported stronger iPhone sales.",
                "text_char_count": 36,
                "status": "listed",
            }
        ][:limit]

    def update_research_document_snapshot(self, source_doc_id, **kwargs):
        self.updates.append({"source_doc_id": source_doc_id, **kwargs})


class _RepoForOnDemandResearch:
    def __init__(self):
        self.rows: list[dict] = []
        self.calls: list[dict] = []
        self.inserts: list[list[dict]] = []

    def get_research_briefings(self, *, tickers=None, categories=None, limit=10, trading_mode="paper", tenant_id=None):
        self.calls.append(
            {
                "tickers": tickers,
                "categories": categories,
                "limit": limit,
                "trading_mode": trading_mode,
                "tenant_id": tenant_id,
            }
        )
        rows = list(self.rows)
        filters = []
        if tickers:
            allowed_tickers = {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
            filters.append(lambda row: str(row.get("ticker") or "").strip().upper() in allowed_tickers)
        if categories:
            allowed_categories = {str(category).strip().lower() for category in categories if str(category).strip()}
            filters.append(lambda row: str(row.get("category") or "").strip().lower() in allowed_categories)
        if filters:
            rows = [row for row in rows if any(check(row) for check in filters)]
        return rows[:limit]

    def insert_research_briefings(self, rows):
        inserted = [dict(row) for row in rows]
        self.inserts.append(inserted)
        self.rows.extend(inserted)


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


def test_get_research_briefing_prefers_gemini_briefings_over_document_store() -> None:
    repo = _RepoForResearchDocuments()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.tenant_id = "tenant-a"
    tool._research_refresher = lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not call refresher"))

    out = asyncio.run(tool.get_research_briefing(tickers=["aapl"], limit=1))

    assert repo.briefing_calls[0]["tickers"] == ["AAPL"]
    assert repo.document_calls == []
    assert out == [
        {
            "briefing_id": "brf_aapl",
            "ticker": "AAPL",
            "category": "held",
            "headline": "AAPL Gemini-grounded research",
            "summary": "Gemini + Google Search research context.",
        }
    ]


def test_get_research_briefing_refreshes_missing_tickers_and_sector_alias(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    repo = _RepoForOnDemandResearch()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.settings.research_enabled = True
    tool.settings.gemini_api_key = "tenant-gemini"
    tool.settings.research_gemini_api_key = ""
    tool.settings.research_gemini_source = ""
    tool.settings.research_gemini_source_tenant = ""
    tool.tenant_id = "tenant-a"
    refresh_calls: list[dict] = []

    async def _refresh(*, tickers=None, categories=None):
        refresh_calls.append({"tickers": tickers, "categories": categories})
        rows = [
            {
                "briefing_id": "brf_aapl",
                "ticker": "AAPL",
                "category": "held",
                "headline": "AAPL fresh research",
                "summary": "Fresh Apple context.",
            },
            {
                "briefing_id": "brf_sector",
                "ticker": "SECTOR",
                "category": "sector_trends",
                "headline": "Sector fresh research",
                "summary": "Fresh sector context.",
            },
        ]
        repo.insert_research_briefings(rows)
        return rows

    tool._research_refresher = _refresh

    out = asyncio.run(
        tool.get_research_briefing(
            tickers=["aapl"],
            categories=["sector"],
            refresh_missing=True,
            limit=5,
        )
    )

    assert refresh_calls == [{"tickers": ["AAPL"], "categories": ["sector_trends"]}]
    assert {row["briefing_id"] for row in out} == {"brf_aapl", "brf_sector"}
    assert repo.calls[0]["categories"] == ["sector_trends"]
    assert repo.calls[-1]["categories"] == ["sector_trends"]
    assert not hasattr(tool, "get_research_briefing_adk")


def test_get_research_briefing_refreshes_public_categories_when_unfiltered(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    repo = _RepoForOnDemandResearch()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.settings.research_enabled = True
    tool.settings.gemini_api_key = "tenant-gemini"
    tool.settings.research_gemini_api_key = ""
    tool.settings.research_gemini_source = ""
    tool.settings.research_gemini_source_tenant = ""
    tool.tenant_id = "tenant-a"
    refresh_calls: list[dict] = []

    async def _refresh(*, tickers=None, categories=None):
        refresh_calls.append({"tickers": tickers, "categories": categories})
        rows = [
            {
                "briefing_id": f"brf_{category}",
                "ticker": category.upper(),
                "category": category,
                "headline": f"{category} fresh research",
                "summary": f"Fresh {category} context.",
            }
            for category in categories
        ]
        repo.insert_research_briefings(rows)
        return rows

    tool._research_refresher = _refresh

    out = asyncio.run(tool.get_research_briefing(refresh_missing=True, limit=5))

    assert refresh_calls == [
        {
            "tickers": [],
            "categories": ["global_market", "geopolitical", "sector_trends"],
        }
    ]
    assert [row["category"] for row in out] == ["global_market", "geopolitical", "sector_trends"]


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

    out = asyncio.run(tool.get_research_briefing(limit=2))

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

    out = asyncio.run(tool.get_research_briefing(tickers=["AAPL"], limit=2))

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

    out = asyncio.run(tool.get_research_briefing(tickers=["VZ"], limit=1))

    assert out[0]["detail_json"]["key_points"] == ["Q1 EPS beat", "FY26 guide raised"]
    assert out[0]["detail_json"]["risks"] == ["valuation after rally"]
    assert out[0]["detail_json"]["confidence"] == 0.74
