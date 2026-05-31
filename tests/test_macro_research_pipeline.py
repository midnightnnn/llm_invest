from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from arena.config import load_settings


RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>BOK Research</title>
    <item>
      <title>Monetary Policy and Household Credit</title>
      <link>https://www.bok.or.kr/portal/bbs/P0000559/view.do?nttId=1001&amp;menuNo=201156</link>
      <description><![CDATA[
        <p>Household credit remains sensitive to policy rates.</p>
        <a href="https://file-cdn.bok.or.kr/research-note.pdf?token=abc">PDF</a>
      ]]></description>
      <pubDate>Fri, 29 May 2026 09:00:00 +0900</pubDate>
      <guid>https://www.bok.or.kr/portal/bbs/P0000559/view.do?nttId=1001&amp;menuNo=201156</guid>
    </item>
  </channel>
</rss>
"""


class _Response:
    def __init__(self, *, text: str = "", content: bytes | None = None) -> None:
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")

    def raise_for_status(self) -> None:
        return None


class _Http:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(self, url: str, timeout: int = 20) -> _Response:
        self.calls.append(url)
        if url.endswith("/rss"):
            return _Response(text=RSS_SAMPLE)
        if "research-note.pdf" in url:
            return _Response(content=b"%PDF-1.4 macro research")
        raise AssertionError(f"unexpected url {url}")


class _ObjectStore:
    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes]] = []

    def put_bytes(self, path: str, data: bytes, *, content_type: str) -> str:
        self.writes.append((path, data))
        return f"gs://arena-macro-research/{path}"

    def put_text(self, path: str, text: str, *, content_type: str = "text/plain; charset=utf-8") -> str:
        self.writes.append((path, text.encode("utf-8")))
        return f"gs://arena-macro-research/{path}"


class _Repo:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.briefings: dict[str, dict[str, Any]] = {}

    def get_macro_research_document(self, source_doc_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        _ = tenant_id
        return self.docs.get(source_doc_id)

    def upsert_macro_research_document(self, row: dict[str, Any], *, tenant_id: str | None = None) -> None:
        payload = dict(row)
        payload["tenant_id"] = tenant_id or payload.get("tenant_id")
        self.docs[str(payload["source_doc_id"])] = payload

    def upsert_macro_research_briefing(self, row: dict[str, Any], *, tenant_id: str | None = None) -> None:
        payload = dict(row)
        payload["tenant_id"] = tenant_id or payload.get("tenant_id")
        self.briefings[str(payload["source_doc_id"])] = payload


class _Summarizer:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def summarize(self, document: Any) -> Any:
        from arena.macro_research import MacroResearchSummary

        self.calls.append(document)
        return MacroResearchSummary(
            headline="BOK flags credit sensitivity",
            executive_summary=(
                "The note links household credit sensitivity to policy rates and preserves the mechanism: "
                "borrowers with floating-rate exposure respond quickly to funding-cost changes, while fixed-rate "
                "migration depends on rate expectations and lending spreads."
            ),
            key_findings=["Credit is rate-sensitive", "Policy transmission remains active"],
            methodology="Uses official household credit and mortgage-rate choice evidence to identify rate exposure.",
            macro_channels=["policy rates", "household credit", "bank lending spreads"],
            asset_implications=["KR duration risk should be monitored", "Bank credit beta may rise in hiking cycles"],
            watch_indicators=["BOK base rate", "household credit growth", "mortgage fixed-floating spread"],
            caveats=["Summary depends on a central-bank research note"],
            market_implication="KR duration and bank credit risk should be monitored.",
            themes=["monetary_policy", "credit"],
            confidence=0.82,
            detail_json={"source_language": "en"},
        )


def test_research_model_default_uses_gemini_3_flash_preview(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_MODEL", raising=False)

    settings = load_settings()

    assert settings.research_gemini_model == "gemini-3-flash-preview"


def test_default_macro_research_feeds_exclude_low_signal_feeds() -> None:
    from arena.macro_research import DEFAULT_MACRO_RESEARCH_FEEDS

    feed_ids = {feed.feed_id for feed in DEFAULT_MACRO_RESEARCH_FEEDS}
    doc_types = {feed.doc_type for feed in DEFAULT_MACRO_RESEARCH_FEEDS}

    assert "bok_mpc_minutes" not in feed_ids
    assert "fred_blog" not in feed_ids
    assert "minutes" not in doc_types
    assert "data_research_blog" not in doc_types


def test_macro_research_service_ingests_bok_rss_to_gcs_and_summary() -> None:
    from arena.macro_research import MacroResearchFeed, MacroResearchService

    settings = load_settings()
    settings.research_gemini_model = "gemini-3-flash-preview"
    settings.macro_research_gcs_bucket = "arena-macro-research"
    repo = _Repo()
    object_store = _ObjectStore()
    summarizer = _Summarizer()
    service = MacroResearchService(
        settings=settings,
        repo=repo,
        feeds=[
            MacroResearchFeed(
                source="bok",
                feed_id="bok_issue_notes",
                title="BOK Issue Notes",
                url="https://bok.local/rss",
                doc_type="issue_note",
                region="kr",
                market="kr",
                themes=("monetary_policy", "credit"),
            )
        ],
        http=_Http(),
        object_store=object_store,
        summarizer=summarizer,
        tenant_id="tenant-a",
    )

    result = service.refresh(max_items_per_feed=5)

    assert result.discovered == 1
    assert result.inserted == 1
    assert result.summarized == 1
    assert result.skipped == 0
    source_doc_id = "bok:bok_issue_notes:1001:201156"
    doc = repo.docs[source_doc_id]
    assert doc["source"] == "bok"
    assert doc["doc_type"] == "issue_note"
    assert doc["content_hash"]
    assert doc["source_url"].endswith("nttId=1001&menuNo=201156")
    assert doc["raw_gcs_uri"].startswith("gs://arena-macro-research/macro_research/bok/bok_issue_notes/")
    assert doc["content_gcs_uri"].startswith("gs://arena-macro-research/macro_research/bok/bok_issue_notes/")
    assert doc["pdf_gcs_uri"].startswith("gs://arena-macro-research/macro_research/bok/bok_issue_notes/")
    assert len(object_store.writes) == 3
    briefing = repo.briefings[source_doc_id]
    assert briefing["model"] == "gemini-3-flash-preview"
    assert briefing["headline"] == "BOK flags credit sensitivity"
    assert briefing["summary"].startswith("The note links household credit sensitivity")
    assert briefing["key_points"] == ["Credit is rate-sensitive", "Policy transmission remains active"]
    assert briefing["risk_flags"] == ["Summary depends on a central-bank research note"]
    assert briefing["themes"] == ["monetary_policy", "credit"]
    assert briefing["detail_json"]["schema_version"] == "macro_research_summary.v2"
    assert briefing["detail_json"]["methodology"].startswith("Uses official household credit")
    assert briefing["detail_json"]["macro_channels"] == ["policy rates", "household credit", "bank lending spreads"]
    assert briefing["detail_json"]["asset_implications"][0] == "KR duration risk should be monitored"
    assert briefing["detail_json"]["watch_indicators"][0] == "BOK base rate"
    assert briefing["detail_json"]["caveats"] == ["Summary depends on a central-bank research note"]
    assert summarizer.calls[0].source_doc_id == source_doc_id


def test_macro_research_service_skips_unchanged_documents() -> None:
    from arena.macro_research import MacroResearchFeed, MacroResearchService

    settings = load_settings()
    settings.research_gemini_model = "gemini-3-flash-preview"
    settings.macro_research_gcs_bucket = "arena-macro-research"
    repo = _Repo()
    object_store = _ObjectStore()
    summarizer = _Summarizer()
    service = MacroResearchService(
        settings=settings,
        repo=repo,
        feeds=[
            MacroResearchFeed(
                source="bok",
                feed_id="bok_issue_notes",
                title="BOK Issue Notes",
                url="https://bok.local/rss",
                doc_type="issue_note",
                region="kr",
                market="kr",
            )
        ],
        http=_Http(),
        object_store=object_store,
        summarizer=summarizer,
        tenant_id="tenant-a",
    )

    first = service.refresh(max_items_per_feed=5)
    second = service.refresh(max_items_per_feed=5)

    assert first.summarized == 1
    assert second.discovered == 1
    assert second.inserted == 0
    assert second.summarized == 0
    assert second.skipped == 1
    assert len(summarizer.calls) == 1
    assert len(repo.docs) == 1
    assert len(repo.briefings) == 1


class _RepoForMacroTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_macro_research_briefings(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return [
            {
                "source_doc_id": "bok:bok_issue_notes:1001:201156",
                "published_at": datetime(2026, 5, 29, 0, 0, tzinfo=timezone.utc),
                "source": "bok",
                "feed_id": "bok_issue_notes",
                "doc_type": "issue_note",
                "market": "kr",
                "title": "Monetary Policy and Household Credit",
                "headline": "BOK flags credit sensitivity",
                "summary": "The note links household credit sensitivity to policy rates.",
                "market_implication": "KR duration and bank credit risk should be monitored.",
                "themes": ["monetary_policy", "credit"],
                "source_url": "https://www.bok.or.kr/portal/bbs/P0000559/view.do?nttId=1001&menuNo=201156",
                "key_points": ["Credit is rate-sensitive"],
                "risk_flags": ["Central-bank research note"],
                "confidence": 0.82,
                "model": "gemini-3-flash-preview",
                "detail_json": {
                    "schema_version": "macro_research_summary.v2",
                    "key_findings": ["Credit is rate-sensitive"],
                    "methodology": "Uses official household credit and mortgage-rate choice evidence.",
                    "macro_channels": ["policy rates", "household credit"],
                    "asset_implications": ["KR duration risk should be monitored"],
                    "watch_indicators": ["BOK base rate"],
                    "caveats": ["Central-bank research note"],
                },
            }
        ]


def test_get_macro_research_briefing_returns_compact_rows() -> None:
    from arena.agents.adk_agents import _ContextTools

    repo = _RepoForMacroTool()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.tenant_id = "tenant-a"

    out = asyncio.run(
        tool.get_macro_research_briefing(
            scope="week",
            market="kr",
            sources=["bok"],
            doc_types=["issue_note"],
            themes=["credit"],
            detail_level="compact",
            limit=2,
        )
    )

    assert repo.calls[0]["tenant_id"] == "tenant-a"
    assert repo.calls[0]["market"] == "kr"
    assert repo.calls[0]["sources"] == ["bok"]
    assert repo.calls[0]["doc_types"] == ["issue_note"]
    assert repo.calls[0]["themes"] == ["credit"]
    assert repo.calls[0]["since"] is not None
    assert out == [
        {
            "published_at": "2026-05-29T00:00:00+00:00",
            "source": "bok",
            "doc_type": "issue_note",
            "market": "kr",
            "title": "Monetary Policy and Household Credit",
            "headline": "BOK flags credit sensitivity",
            "summary": "The note links household credit sensitivity to policy rates.",
            "market_implication": "KR duration and bank credit risk should be monitored.",
            "themes": ["monetary_policy", "credit"],
            "source_doc_id": "bok:bok_issue_notes:1001:201156",
        }
    ]


def test_get_macro_research_briefing_facts_returns_research_detail() -> None:
    from arena.agents.adk_agents import _ContextTools

    repo = _RepoForMacroTool()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.tenant_id = "tenant-a"

    out = asyncio.run(
        tool.get_macro_research_briefing(
            scope="all",
            market="kr",
            sources=["bok"],
            detail_level="facts",
            limit=1,
        )
    )

    assert out[0]["key_findings"] == ["Credit is rate-sensitive"]
    assert out[0]["methodology"] == "Uses official household credit and mortgage-rate choice evidence."
    assert out[0]["macro_channels"] == ["policy rates", "household credit"]
    assert out[0]["asset_implications"] == ["KR duration risk should be monitored"]
    assert out[0]["watch_indicators"] == ["BOK base rate"]
    assert out[0]["caveats"] == ["Central-bank research note"]
    assert out[0]["confidence"] == 0.82


def test_get_macro_research_briefing_accepts_source_doc_ids_for_drilldown() -> None:
    from arena.agents.adk_agents import _ContextTools

    repo = _RepoForMacroTool()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.tenant_id = "tenant-a"

    out = asyncio.run(
        tool.get_macro_research_briefing(
            source_doc_ids=["bok:bok_issue_notes:1001:201156"],
            detail_level="facts",
            limit=1,
        )
    )

    assert repo.calls[0]["source_doc_ids"] == ["bok:bok_issue_notes:1001:201156"]
    assert out[0]["source_doc_id"] == "bok:bok_issue_notes:1001:201156"
    assert out[0]["key_findings"] == ["Credit is rate-sensitive"]


def test_macro_research_schema_and_registry_are_exposed() -> None:
    from arena.data.local.schema import table_specs
    from arena.tools.default_registry import build_default_registry
    from tests.test_new_tools import _FakeRepo, _settings

    specs = {spec.name: {col.name for col in spec.columns} for spec in table_specs()}
    assert {"source_doc_id", "content_hash", "raw_gcs_uri", "content_gcs_uri"} <= specs["macro_research_documents"]
    assert {"source_doc_id", "headline", "summary", "market_implication", "model"} <= specs[
        "macro_research_briefings"
    ]

    reg = build_default_registry(repo=_FakeRepo(), settings=_settings())
    entry = reg.get("get_macro_research_briefing")
    assert entry.category == "macro"
    assert entry.tier == "optional"


def test_local_macro_research_store_upserts_and_filters(tmp_path) -> None:
    from arena.data.local.repository import LocalRepository

    repo = LocalRepository(tenant_id="tenant-a", settings=load_settings(), db_path=str(tmp_path / "arena.duckdb"))
    repo.ensure_tables()
    published_at = datetime(2026, 5, 29, 0, 0, tzinfo=timezone.utc)

    repo.upsert_macro_research_document(
        {
            "source_doc_id": "bok:bok_issue_notes:1001:201156",
            "source": "bok",
            "feed_id": "bok_issue_notes",
            "doc_type": "issue_note",
            "region": "kr",
            "market": "kr",
            "title": "Monetary Policy and Household Credit",
            "source_url": "https://www.bok.or.kr/portal/bbs/P0000559/view.do?nttId=1001&menuNo=201156",
            "published_at": published_at,
            "fetched_at": published_at,
            "content_hash": "abc123",
            "raw_gcs_uri": "gs://bucket/raw.xml",
            "content_gcs_uri": "gs://bucket/content.txt",
            "pdf_gcs_uri": None,
            "text_char_count": 42,
            "status": "summarized",
            "summary_status": "summarized",
            "error_message": None,
            "themes": ["monetary_policy", "credit"],
            "detail_json": {"pdf_url": ""},
        },
        tenant_id="tenant-a",
    )
    repo.upsert_macro_research_briefing(
        {
            "source_doc_id": "bok:bok_issue_notes:1001:201156",
            "created_at": published_at,
            "published_at": published_at,
            "source": "bok",
            "feed_id": "bok_issue_notes",
            "doc_type": "issue_note",
            "region": "kr",
            "market": "kr",
            "title": "Monetary Policy and Household Credit",
            "source_url": "https://www.bok.or.kr/portal/bbs/P0000559/view.do?nttId=1001&menuNo=201156",
            "headline": "BOK flags credit sensitivity",
            "summary": "The note links household credit sensitivity to policy rates.",
            "key_points": ["Credit is rate-sensitive"],
            "market_implication": "KR duration and bank credit risk should be monitored.",
            "risk_flags": ["Central-bank research note"],
            "themes": ["monetary_policy", "credit"],
            "confidence": 0.82,
            "model": "gemini-3-flash-preview",
            "detail_json": {"schema_version": "macro_research_summary.v1"},
        },
        tenant_id="tenant-a",
    )

    doc = repo.get_macro_research_document("bok:bok_issue_notes:1001:201156", tenant_id="tenant-a")
    rows = repo.get_macro_research_briefings(
        source_doc_ids=["bok:bok_issue_notes:1001:201156"],
        sources=["bok"],
        doc_types=["issue_note"],
        themes=["credit"],
        market="kr",
        limit=5,
        tenant_id="tenant-a",
    )

    assert doc is not None
    assert doc["content_hash"] == "abc123"
    assert rows[0]["source_doc_id"] == "bok:bok_issue_notes:1001:201156"
    assert rows[0]["themes"] == ["monetary_policy", "credit"]
    assert rows[0]["detail_json"]["schema_version"] == "macro_research_summary.v1"
