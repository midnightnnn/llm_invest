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
        self.theses: dict[str, list[dict[str, Any]]] = {}

    def get_macro_research_document(self, source_doc_id: str, *, tenant_id: str | None = None) -> dict[str, Any] | None:
        _ = tenant_id
        return self.docs.get(source_doc_id)

    def upsert_macro_research_document(self, row: dict[str, Any], *, tenant_id: str | None = None) -> None:
        _ = tenant_id
        payload = dict(row)
        self.docs[str(payload["source_doc_id"])] = payload

    def upsert_macro_research_briefing(self, row: dict[str, Any], *, tenant_id: str | None = None) -> None:
        _ = tenant_id
        payload = dict(row)
        self.briefings[str(payload["source_doc_id"])] = payload

    def replace_macro_research_theses(
        self,
        source_doc_id: str,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        _ = tenant_id
        self.theses[str(source_doc_id)] = [dict(row) for row in rows]


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
            investment_theses=[
                {
                    "theme_key": "credit_transmission",
                    "horizon": "quarters",
                    "thesis": "Rate-sensitive household credit can pressure KR lenders and duration-sensitive assets.",
                    "transmission_channels": ["policy rates", "household credit", "bank lending spreads"],
                    "affected_sectors": ["banks", "duration_sensitive_assets"],
                    "candidate_queries": ["KR banks", "duration-sensitive assets"],
                    "watch_indicators": ["BOK base rate", "household credit growth"],
                    "invalidation_conditions": ["Household credit growth stabilizes despite higher rates"],
                    "confidence_label": "medium",
                }
            ],
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


def test_macro_research_service_ingests_bok_rss_to_gcs_and_summary(monkeypatch) -> None:
    import arena.macro_research as macro_research
    from arena.macro_research import MacroResearchFeed, MacroResearchService

    monkeypatch.setattr(macro_research, "_extract_pdf_text", lambda value: "PDF body text about credit pass-through.")

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
    content_write = next(data for path, data in object_store.writes if path.endswith("/content.txt"))
    assert b"PDF body text about credit pass-through." in content_write
    briefing = repo.briefings[source_doc_id]
    assert briefing["model"] == "gemini-3-flash-preview"
    assert briefing["headline"] == "BOK flags credit sensitivity"
    assert briefing["summary"].startswith("The note links household credit sensitivity")
    assert briefing["key_points"] == ["Credit is rate-sensitive", "Policy transmission remains active"]
    assert briefing["risk_flags"] == ["Summary depends on a central-bank research note"]
    assert briefing["themes"] == ["monetary_policy", "credit"]
    assert briefing["detail_json"]["schema_version"] == "macro_research_summary.v3"
    assert briefing["detail_json"]["methodology"].startswith("Uses official household credit")
    assert briefing["detail_json"]["macro_channels"] == ["policy rates", "household credit", "bank lending spreads"]
    assert briefing["detail_json"]["asset_implications"][0] == "KR duration risk should be monitored"
    assert briefing["detail_json"]["watch_indicators"][0] == "BOK base rate"
    assert briefing["detail_json"]["caveats"] == ["Summary depends on a central-bank research note"]
    assert briefing["detail_json"]["investment_theses"][0]["theme_key"] == "credit_transmission"
    thesis_rows = repo.theses[source_doc_id]
    assert len(thesis_rows) == 1
    assert thesis_rows[0]["source_doc_id"] == source_doc_id
    assert thesis_rows[0]["market"] == "kr"
    assert thesis_rows[0]["theme_key"] == "credit_transmission"
    assert thesis_rows[0]["thesis"].startswith("Rate-sensitive household credit")
    assert thesis_rows[0]["candidate_queries"] == ["KR banks", "duration-sensitive assets"]
    assert thesis_rows[0]["status"] == "active"
    assert summarizer.calls[0].source_doc_id == source_doc_id
    assert "PDF body text about credit pass-through." in summarizer.calls[0].content_text


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
    assert len(repo.theses) == 1


def test_macro_research_service_can_store_metadata_without_gcs_or_summary() -> None:
    from arena.macro_research import MacroResearchFeed, MacroResearchService

    settings = load_settings()
    settings.macro_research_gcs_bucket = ""
    repo = _Repo()
    http = _Http()
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
        http=http,
        object_store=None,
        summarizer=None,
        tenant_id="tenant-a",
    )

    result = service.refresh(max_items_per_feed=5)

    assert result.discovered == 1
    assert result.inserted == 1
    assert result.summarized == 0
    assert http.calls == ["https://bok.local/rss"]
    doc = repo.docs["bok:bok_issue_notes:1001:201156"]
    assert doc["status"] == "listed"
    assert doc["summary_status"] == "pending"
    assert doc["content_gcs_uri"] is None
    assert doc["pdf_gcs_uri"] is None
    assert doc["detail_json"]["pdf_url"] == "https://file-cdn.bok.or.kr/research-note.pdf?token=abc"
    assert repo.briefings == {}
    assert repo.theses == {}


def test_macro_research_theme_phrases_are_canonicalized() -> None:
    from arena.macro_research import _clean_theme_codes

    assert _clean_theme_codes(["Inflation Dynamics", "Monetary Policy Transmission", "Credit Risk"]) == [
        "inflation",
        "monetary_policy",
        "credit",
    ]


class _RepoForMacroBriefingOnlyTool:
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
                    "investment_theses": [
                        {
                            "theme_key": "credit_transmission",
                            "thesis": "Household credit sensitivity may pressure KR lenders.",
                        }
                    ],
                },
            }
        ]


class _RepoForMacroDocumentTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    def get_macro_research_briefings(self, **kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("get_macro_research_briefings should not be used by the macro document tool")

    def get_macro_research_documents(self, **kwargs: Any) -> list[dict[str, Any]]:
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
                "source_url": "https://www.bok.or.kr/portal/bbs/P0000559/view.do?nttId=1001&menuNo=201156",
                "themes": ["monetary_policy", "credit"],
                "text_char_count": 128,
                "status": "listed",
                "detail_json": {"pdf_url": ""},
            }
        ]

    def update_macro_research_document_snapshot(self, source_doc_id: str, **kwargs: Any) -> None:
        self.updates.append({"source_doc_id": source_doc_id, **kwargs})


def test_read_official_macro_research_lists_document_metadata_when_available() -> None:
    from arena.agents.adk_agents import _ContextTools

    repo = _RepoForMacroDocumentTool()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.tenant_id = "tenant-a"

    out = asyncio.run(
        tool.read_official_macro_research(
            market="kr",
            sources=["bok"],
            limit=1,
        )
    )

    assert repo.calls[0] == {
        "source_doc_ids": None,
        "sources": ["bok"],
        "market": "kr",
        "since": None,
        "limit": 1,
    }
    assert out == [
        {
            "published_at": "2026-05-29T00:00:00+00:00",
            "source": "bok",
            "feed_id": "bok_issue_notes",
            "doc_type": "issue_note",
            "market": "kr",
            "title": "Monetary Policy and Household Credit",
            "source_url": "https://www.bok.or.kr/portal/bbs/P0000559/view.do?nttId=1001&menuNo=201156",
            "source_doc_id": "bok:bok_issue_notes:1001:201156",
            "text_char_count": 128,
            "status": "listed",
        }
    ]


def test_read_official_macro_research_reads_live_source_by_doc_id(monkeypatch) -> None:
    from arena.agents import adk_context_tools
    from arena.agents.adk_agents import _ContextTools
    from arena.research_documents import LiveDocumentRead

    repo = _RepoForMacroDocumentTool()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.settings.macro_research_gcs_bucket = ""
    tool.tenant_id = "tenant-a"
    body = "Official document body " * 20

    monkeypatch.setattr(
        adk_context_tools,
        "fetch_live_document",
        lambda url: LiveDocumentRead(
            source_url=url,
            final_url=url,
            content_type="text/html",
            content_text=body,
            content_hash="hash-read",
            retrieved_at=datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc),
        ),
    )

    out = asyncio.run(
        tool.read_official_macro_research(
            source_doc_ids=["bok:bok_issue_notes:1001:201156"],
            offset=0,
            limit=1,
        )
    )

    assert repo.calls[0]["source_doc_ids"] == ["bok:bok_issue_notes:1001:201156"]
    assert repo.calls[0]["since"] is None
    assert out[0]["content_text"] == body
    assert "next_offset" not in out[0]
    assert out[0]["content_hash"] == "hash-read"
    assert repo.updates[0]["status"] == "read"
    assert repo.updates[0]["content_hash"] == "hash-read"


def test_read_official_macro_research_returns_read_response_when_snapshot_update_fails(monkeypatch) -> None:
    from arena.agents import adk_context_tools
    from arena.agents.adk_agents import _ContextTools
    from arena.research_documents import LiveDocumentRead

    repo = _RepoForMacroDocumentTool()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.settings.macro_research_gcs_bucket = ""
    tool.tenant_id = "tenant-a"

    def _raise_snapshot_update(*args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)
        raise RuntimeError("streaming buffer update blocked")

    repo.update_macro_research_document_snapshot = _raise_snapshot_update  # type: ignore[method-assign]
    monkeypatch.setattr(
        adk_context_tools,
        "fetch_live_document",
        lambda url: LiveDocumentRead(
            source_url=url,
            final_url=url,
            content_type="text/html",
            content_text="Fresh official document body " * 20,
            content_hash="hash-fresh",
            retrieved_at=datetime(2026, 5, 30, 0, 0, tzinfo=timezone.utc),
        ),
    )

    out = asyncio.run(
        tool.read_official_macro_research(
            source_doc_ids=["bok:bok_issue_notes:1001:201156"],
            offset=0,
            limit=1,
        )
    )

    assert out[0]["content_text"].startswith("Fresh official document body")
    assert out[0]["content_hash"] == "hash-fresh"


def test_read_official_macro_research_does_not_fallback_to_summarized_briefings() -> None:
    from arena.agents.adk_agents import _ContextTools

    repo = _RepoForMacroBriefingOnlyTool()
    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = repo
    tool.settings = load_settings()
    tool.settings.trading_mode = "paper"
    tool.tenant_id = "tenant-a"

    out = asyncio.run(
        tool.read_official_macro_research(
            market="kr",
            sources=["bok"],
            limit=2,
        )
    )

    assert out == []
    assert repo.calls == []


def test_read_official_macro_research_schema_exposes_minimal_document_browser_params() -> None:
    from google.adk.tools.function_tool import FunctionTool

    from arena.agents.adk_agents import _ContextTools
    from arena.macro_research_taxonomy import MACRO_RESEARCH_MARKETS, MACRO_RESEARCH_SOURCES

    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForMacroBriefingOnlyTool()
    tool.settings = load_settings()
    tool.tenant_id = "tenant-a"

    params = FunctionTool(tool.read_official_macro_research)._get_declaration().parameters.model_dump(
        mode="json",
        exclude_none=True,
    )
    props = params["properties"]

    assert set(props) == {"market", "sources", "source_doc_ids", "offset", "limit"}
    assert props["market"]["enum"] == list(MACRO_RESEARCH_MARKETS)
    assert props["sources"]["items"]["enum"] == list(MACRO_RESEARCH_SOURCES)


def test_read_official_macro_research_description_frames_forward_looking_reading_value() -> None:
    from google.adk.tools.function_tool import FunctionTool

    from arena.agents.adk_agents import _ContextTools

    tool = _ContextTools.__new__(_ContextTools)
    tool.repo = _RepoForMacroBriefingOnlyTool()
    tool.settings = load_settings()
    tool.tenant_id = "tenant-a"

    declaration = FunctionTool(tool.read_official_macro_research)._get_declaration()
    description = declaration.description

    assert declaration.name == "read_official_macro_research"
    assert "forward-looking" in description
    assert "before they are obvious in prices or indicators" in description
    assert "Choose documents freely" in description


def test_macro_research_schema_and_registry_are_exposed() -> None:
    from arena.data.local.schema import table_specs
    from arena.tools.default_registry import build_default_registry
    from tests.test_new_tools import _FakeRepo, _settings

    specs = {spec.name: {col.name for col in spec.columns} for spec in table_specs()}
    assert {"source_doc_id", "content_hash", "raw_gcs_uri", "content_gcs_uri"} <= specs["macro_research_documents"]
    assert {"source_doc_id", "headline", "summary", "market_implication", "model"} <= specs[
        "macro_research_briefings"
    ]
    assert {"thesis_id", "source_doc_id", "theme_key", "candidate_queries", "status"} <= specs[
        "macro_research_theses"
    ]
    assert "tenant_id" not in specs["macro_research_documents"]
    assert "tenant_id" not in specs["macro_research_briefings"]
    assert "tenant_id" not in specs["macro_research_theses"]

    reg = build_default_registry(repo=_FakeRepo(), settings=_settings())
    entry = reg.get("read_official_macro_research")
    assert entry is not None
    assert entry.category == "macro"
    assert entry.tier == "optional"
    assert "FRED" not in entry.description.splitlines()[0]
    assert '["fred"]' not in entry.description
    assert "detail_level" not in entry.description
    assert "doc_types" not in entry.description
    assert "themes" not in entry.description
    assert "FRED" not in entry.description_ko
    assert "source_doc_id" in entry.description
    assert "드릴다운" in entry.description_ko
    assert reg.get("get_macro_research_briefing") is None


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
    repo.upsert_macro_research_briefing(
        {
            "source_doc_id": "stlouisfed:review:inflation-dynamics",
            "created_at": published_at,
            "published_at": published_at,
            "source": "stlouisfed",
            "feed_id": "stlouisfed_review",
            "doc_type": "journal_article",
            "region": "us",
            "market": "us",
            "title": "Labor Costs and Inflation Dynamics",
            "source_url": "https://www.stlouisfed.org/review/example",
            "headline": "St. Louis Fed links labor costs and inflation dynamics",
            "summary": "The paper studies inflation dynamics and monetary policy transmission.",
            "key_points": ["Inflation dynamics are persistent"],
            "market_implication": "US duration risk should be monitored.",
            "risk_flags": ["Research article"],
            "themes": ["Inflation Dynamics", "Monetary Policy Transmission"],
            "confidence": 0.77,
            "model": "gemini-3-flash-preview",
            "detail_json": {"schema_version": "macro_research_summary.v1"},
        },
        tenant_id="tenant-a",
    )
    repo.replace_macro_research_theses(
        "bok:bok_issue_notes:1001:201156",
        [
            {
                "thesis_id": "mrt_credit",
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
                "theme_key": "credit_transmission",
                "horizon": "quarters",
                "thesis": "Rate-sensitive household credit can pressure KR lenders and duration-sensitive assets.",
                "transmission_channels": ["policy rates", "household credit"],
                "affected_sectors": ["banks"],
                "candidate_queries": ["KR banks"],
                "watch_indicators": ["BOK base rate"],
                "invalidation_conditions": ["Household credit stabilizes"],
                "confidence_label": "medium",
                "status": "active",
                "evidence_json": {"source_doc_id": "bok:bok_issue_notes:1001:201156"},
                "detail_json": {"schema_version": "macro_research_thesis.v1"},
            }
        ],
        tenant_id="tenant-a",
    )

    doc = repo.get_macro_research_document("bok:bok_issue_notes:1001:201156", tenant_id="tenant-b")
    rows = repo.get_macro_research_briefings(
        source_doc_ids=["bok:bok_issue_notes:1001:201156"],
        sources=["bok"],
        doc_types=["issue_note"],
        themes=["credit"],
        market="kr",
        limit=5,
        tenant_id="tenant-b",
    )

    assert doc is not None
    assert doc["content_hash"] == "abc123"
    assert rows[0]["source_doc_id"] == "bok:bok_issue_notes:1001:201156"
    assert rows[0]["themes"] == ["monetary_policy", "credit"]
    assert rows[0]["detail_json"]["schema_version"] == "macro_research_summary.v1"
    us_rows = repo.get_macro_research_briefings(
        sources=["stlouisfed"],
        themes=["inflation", "monetary_policy"],
        market="us",
        limit=5,
        tenant_id="tenant-b",
    )
    assert us_rows[0]["source_doc_id"] == "stlouisfed:review:inflation-dynamics"
    assert us_rows[0]["themes"] == ["inflation dynamics", "monetary policy transmission"]
    theses = repo.get_macro_research_theses(
        market="kr",
        themes=["credit_transmission"],
        limit=5,
        tenant_id="tenant-b",
    )
    assert theses[0]["theme_key"] == "credit_transmission"
    assert theses[0]["candidate_queries"] == ["KR banks"]
    assert theses[0]["detail_json"]["schema_version"] == "macro_research_thesis.v1"
