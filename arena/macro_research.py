from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import parse_qs, unquote, urlparse
from xml.etree import ElementTree as ET

import requests

from arena.config import Settings, effective_research_gemini_api_key, research_generation_status
from arena.models import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MacroResearchFeed:
    source: str
    feed_id: str
    title: str
    url: str
    doc_type: str
    region: str = ""
    market: str = "all"
    themes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class MacroResearchDocument:
    source_doc_id: str
    source: str
    feed_id: str
    doc_type: str
    region: str
    market: str
    title: str
    source_url: str
    published_at: datetime | None
    fetched_at: datetime
    content_hash: str
    content_text: str
    raw_text: str
    themes: tuple[str, ...] = field(default_factory=tuple)
    pdf_url: str = ""
    pdf_bytes: bytes = b""
    raw_gcs_uri: str = ""
    content_gcs_uri: str = ""
    pdf_gcs_uri: str = ""


@dataclass(frozen=True, slots=True)
class MacroResearchSummary:
    headline: str
    executive_summary: str
    key_findings: list[str] = field(default_factory=list)
    methodology: str = ""
    macro_channels: list[str] = field(default_factory=list)
    asset_implications: list[str] = field(default_factory=list)
    watch_indicators: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    market_implication: str = ""
    themes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    detail_json: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        return self.executive_summary


@dataclass(frozen=True, slots=True)
class MacroResearchRefreshResult:
    discovered: int = 0
    inserted: int = 0
    summarized: int = 0
    skipped: int = 0
    failed: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)


class MacroResearchObjectStore(Protocol):
    def put_bytes(self, path: str, data: bytes, *, content_type: str) -> str:
        ...

    def put_text(self, path: str, text: str, *, content_type: str = "text/plain; charset=utf-8") -> str:
        ...


class MacroResearchSummarizer(Protocol):
    def summarize(self, document: MacroResearchDocument) -> MacroResearchSummary | None:
        ...


DEFAULT_MACRO_RESEARCH_FEEDS: tuple[MacroResearchFeed, ...] = (
    MacroResearchFeed(
        source="bok",
        feed_id="bok_issue_notes",
        title="BOK Issue Notes",
        url="https://www.bok.or.kr/portal/bbs/P0002353/news.rss?menuNo=200433",
        doc_type="issue_note",
        region="kr",
        market="kr",
        themes=("monetary_policy", "credit", "growth"),
    ),
    MacroResearchFeed(
        source="bok",
        feed_id="bok_economic_research_ko",
        title="BOK Economic Research Korean",
        url="https://www.bok.or.kr/imer/bbs/P0002455/news.rss?menuNo=500788",
        doc_type="working_paper",
        region="kr",
        market="kr",
        themes=("growth", "inflation", "financial_stability"),
    ),
    MacroResearchFeed(
        source="bok",
        feed_id="bok_economic_outlook_deep_dive",
        title="BOK Economic Outlook Core Issues",
        url="https://www.bok.or.kr/imer/bbs/B0000368/news.rss?menuNo=201140",
        doc_type="research_report",
        region="kr",
        market="kr",
        themes=("growth", "inflation", "external"),
    ),
    MacroResearchFeed(
        source="stlouisfed",
        feed_id="stlouisfed_review",
        title="Federal Reserve Bank of St. Louis Review",
        url="https://www.stlouisfed.org/rss/page-resources/publications/review",
        doc_type="journal_article",
        region="us",
        market="us",
        themes=("monetary_policy", "growth", "inflation"),
    ),
    MacroResearchFeed(
        source="stlouisfed",
        feed_id="stlouisfed_on_the_economy",
        title="St. Louis Fed On the Economy",
        url="https://www.stlouisfed.org/on-the-economy/rss",
        doc_type="research_blog",
        region="us",
        market="us",
        themes=("growth", "labor", "inflation"),
    ),
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _clean_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", _text(value).lower()).strip("_")


def _safe_path_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", _text(value)).strip("_")
    return token or hashlib.sha256(_text(value).encode("utf-8")).hexdigest()[:16]


def _parse_datetime(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = parsedate_to_datetime(text)
    except Exception:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clean_html(value: Any) -> str:
    text = html.unescape(_text(value))
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()


def _extract_pdf_urls(value: Any) -> list[str]:
    text = html.unescape(_text(value))
    urls: list[str] = []
    for match in re.finditer(r"""href=["']([^"']+\.pdf(?:\?[^"']*)?)["']""", text, flags=re.IGNORECASE):
        urls.append(html.unescape(match.group(1)))
    for match in re.finditer(r"https?://[^\s\"'<>]+?\.pdf(?:\?[^\s\"'<>]+)?", text, flags=re.IGNORECASE):
        urls.append(html.unescape(match.group(0)))
    clean: list[str] = []
    for url in urls:
        token = url.strip()
        if token and token not in clean:
            clean.append(token)
    return clean


def _child_text(item: ET.Element, names: tuple[str, ...]) -> str:
    wanted = {name.lower() for name in names}
    for child in list(item):
        name = child.tag.rsplit("}", 1)[-1].lower()
        if name in wanted:
            return _text(child.text)
    return ""


def _item_xml(item: ET.Element) -> str:
    try:
        return ET.tostring(item, encoding="unicode")
    except Exception:
        return ""


def _source_doc_id(feed: MacroResearchFeed, *, guid: str, link: str, title: str, published_at: datetime | None) -> str:
    candidate = html.unescape(guid or link or "")
    parsed = urlparse(candidate)
    query = parse_qs(parsed.query)
    ntt_id = _text((query.get("nttId") or query.get("nttid") or [""])[0])
    menu_no = _text((query.get("menuNo") or query.get("menuno") or [""])[0])
    if feed.source == "bok" and ntt_id and menu_no:
        return f"{feed.source}:{feed.feed_id}:{ntt_id}:{menu_no}"
    stable = "|".join(
        [
            feed.source,
            feed.feed_id,
            guid or "",
            link or "",
            title or "",
            published_at.isoformat() if published_at else "",
        ]
    )
    digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]
    return f"{feed.source}:{feed.feed_id}:{digest}"


def _parse_summary_json(text: str) -> dict[str, Any]:
    raw = _text(text)
    if not raw:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    first = raw.find("{")
    last = raw.rfind("}")
    if 0 <= first < last:
        candidates.append(raw[first : last + 1])
    candidates.append(raw)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _clean_str_list(value: Any, *, max_items: int = 5, max_len: int = 180) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = re.sub(r"\s+", " ", _text(item))
        if not text:
            continue
        if len(text) > max_len:
            text = text[: max_len - 3].rstrip() + "..."
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _bounded_text(value: Any, *, max_len: int) -> str:
    text = re.sub(r"\s+", " ", _text(value))
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


class GcsMacroResearchObjectStore:
    def __init__(self, *, bucket_name: str, project: str | None = None) -> None:
        if not _text(bucket_name):
            raise ValueError("bucket_name is required")
        from google.cloud import storage

        self.bucket_name = bucket_name
        self.client = storage.Client(project=project or None)
        self.bucket = self.client.bucket(bucket_name)

    def put_bytes(self, path: str, data: bytes, *, content_type: str) -> str:
        blob = self.bucket.blob(path)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{self.bucket_name}/{path}"

    def put_text(self, path: str, text: str, *, content_type: str = "text/plain; charset=utf-8") -> str:
        return self.put_bytes(path, _text(text).encode("utf-8"), content_type=content_type)


class GeminiMacroResearchSummarizer:
    def __init__(self, *, settings: Settings, model: str | None = None) -> None:
        self.settings = settings
        self.model = _text(model or settings.research_gemini_model) or "gemini-3-flash-preview"
        self.status = research_generation_status(settings)
        if not self.status.get("can_generate"):
            raise RuntimeError("Gemini macro research summarizer is not available")
        from google import genai
        from google.genai import types

        self._types = types
        api_key = effective_research_gemini_api_key(settings)
        if api_key and not self.status.get("uses_vertex"):
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(
                vertexai=bool(self.status.get("uses_vertex")),
                project=settings.google_cloud_project or None,
                location=os.getenv("GOOGLE_CLOUD_LOCATION") or settings.bq_location or None,
            )

    def summarize(self, document: MacroResearchDocument) -> MacroResearchSummary | None:
        source_hint = f"{document.source}/{document.feed_id}/{document.doc_type}"
        prompt = (
            "You are summarizing official central-bank and macroeconomic research for an investment agent. "
            "Return only valid JSON in English. Preserve decision-useful research content, not generic prose. "
            "Do not force a fixed length. Be concise for simple sources, and include more detail when the source "
            "contains methodology, assumptions, transmission channels, empirical findings, market implications, "
            "or limitations that would matter for investment judgment. "
            "Use this schema exactly: "
            '{"headline":"short title","executive_summary":"decision-useful summary",'
            '"key_findings":["research findings"],"methodology":"data, method, model, or analytical setup",'
            '"macro_channels":["economic transmission channels"],"asset_implications":["market or asset-class implications"],'
            '"watch_indicators":["indicators to monitor"],"caveats":["limitations or risks"],'
            '"market_implication":"portfolio implication","themes":["macro tags"],"confidence":0.0}. '
            f"Source: {source_hint}. Title: {document.title}. URL: {document.source_url}. "
            f"Document text:\n{document.content_text[:18000]}"
        )
        try:
            config = self._types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            raw = _text(getattr(response, "text", ""))
            parsed = _parse_summary_json(raw)
        except Exception as exc:
            logger.warning(
                "[yellow]Macro research Gemini summary failed[/yellow] source_doc_id=%s err=%s",
                document.source_doc_id,
                str(exc),
                exc_info=True,
            )
            return None
        if not parsed:
            return None
        try:
            confidence = float(parsed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        themes = _clean_str_list(parsed.get("themes"), max_items=8, max_len=48)
        if not themes:
            themes = list(document.themes)
        key_findings = _clean_str_list(
            parsed.get("key_findings") or parsed.get("key_points"),
            max_items=20,
            max_len=800,
        )
        macro_channels = _clean_str_list(parsed.get("macro_channels"), max_items=20, max_len=500)
        asset_implications = _clean_str_list(parsed.get("asset_implications"), max_items=20, max_len=800)
        watch_indicators = _clean_str_list(parsed.get("watch_indicators"), max_items=20, max_len=240)
        caveats = _clean_str_list(parsed.get("caveats") or parsed.get("risk_flags"), max_items=20, max_len=500)
        methodology = _bounded_text(parsed.get("methodology"), max_len=2400)
        executive_summary = _bounded_text(
            _first_text(parsed.get("executive_summary"), parsed.get("summary"), document.content_text),
            max_len=5000,
        )
        market_implication = _bounded_text(
            _first_text(parsed.get("market_implication"), "; ".join(asset_implications[:3])),
            max_len=1600,
        )
        return MacroResearchSummary(
            headline=_bounded_text(parsed.get("headline") or document.title, max_len=140),
            executive_summary=executive_summary,
            key_findings=key_findings,
            methodology=methodology,
            macro_channels=macro_channels,
            asset_implications=asset_implications,
            watch_indicators=watch_indicators,
            caveats=caveats,
            market_implication=market_implication,
            themes=themes,
            confidence=confidence,
            detail_json={
                "schema_version": "macro_research_summary.v2",
                "source_doc_id": document.source_doc_id,
                "source_url": document.source_url,
                "executive_summary": executive_summary,
                "key_findings": key_findings,
                "methodology": methodology,
                "macro_channels": macro_channels,
                "asset_implications": asset_implications,
                "watch_indicators": watch_indicators,
                "caveats": caveats,
                "market_implication": market_implication,
                "themes": themes,
                "raw_model_response": parsed,
            },
        )


class MacroResearchService:
    def __init__(
        self,
        *,
        settings: Settings,
        repo: Any,
        feeds: tuple[MacroResearchFeed, ...] | list[MacroResearchFeed] = DEFAULT_MACRO_RESEARCH_FEEDS,
        http: Any | None = None,
        object_store: MacroResearchObjectStore | None = None,
        summarizer: MacroResearchSummarizer | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.feeds = tuple(feeds)
        self.http = http or requests.Session()
        self.object_store = object_store
        self.summarizer = summarizer
        self.tenant_id = _text(tenant_id or getattr(repo, "tenant_id", "") or os.getenv("ARENA_TENANT_ID")) or "local"

    @classmethod
    def from_settings(cls, *, settings: Settings, repo: Any, tenant_id: str | None = None) -> "MacroResearchService":
        bucket = _text(getattr(settings, "macro_research_gcs_bucket", ""))
        object_store = GcsMacroResearchObjectStore(
            bucket_name=bucket,
            project=settings.google_cloud_project or None,
        )
        summarizer: MacroResearchSummarizer | None = None
        try:
            summarizer = GeminiMacroResearchSummarizer(settings=settings)
        except Exception as exc:
            logger.warning(
                "[yellow]Macro research summarizer disabled[/yellow] err=%s",
                str(exc),
            )
        return cls(settings=settings, repo=repo, object_store=object_store, summarizer=summarizer, tenant_id=tenant_id)

    def refresh(self, *, max_items_per_feed: int | None = None) -> MacroResearchRefreshResult:
        run_cap = max(1, int(getattr(self.settings, "macro_research_max_docs_per_run", 12) or 12))
        default_per_feed_cap = max(1, (run_cap + max(len(self.feeds), 1) - 1) // max(len(self.feeds), 1))
        per_feed_cap = max(1, int(max_items_per_feed)) if max_items_per_feed is not None else default_per_feed_cap
        remaining = run_cap
        discovered = inserted = summarized = skipped = failed = 0
        source_counts: dict[str, int] = {}
        for feed in self.feeds:
            if max_items_per_feed is None and remaining <= 0:
                break
            item_cap = min(per_feed_cap, remaining) if max_items_per_feed is None else per_feed_cap
            try:
                items = self._fetch_feed_items(feed)[:item_cap]
            except Exception as exc:
                failed += 1
                logger.warning(
                    "[yellow]Macro research feed fetch failed[/yellow] feed=%s err=%s",
                    feed.feed_id,
                    str(exc),
                    exc_info=True,
                )
                continue
            if max_items_per_feed is None:
                remaining -= len(items)
            source_counts[feed.source] = source_counts.get(feed.source, 0) + len(items)
            for item in items:
                discovered += 1
                try:
                    document = self._document_from_item(feed, item)
                    existing = self._get_existing_document(document.source_doc_id)
                    if (
                        existing
                        and _text(existing.get("content_hash")) == document.content_hash
                        and _text(existing.get("status")).lower() == "summarized"
                    ):
                        skipped += 1
                        continue
                    document = self._write_objects(document)
                    summary = self.summarizer.summarize(document) if self.summarizer is not None else None
                    status = "summarized" if summary is not None else "stored"
                    self._upsert_document(document, status=status)
                    inserted += 1
                    if summary is not None:
                        self._upsert_briefing(document, summary)
                        summarized += 1
                except Exception as exc:
                    failed += 1
                    logger.warning(
                        "[yellow]Macro research item failed[/yellow] feed=%s err=%s",
                        feed.feed_id,
                        str(exc),
                        exc_info=True,
                    )
        return MacroResearchRefreshResult(
            discovered=discovered,
            inserted=inserted,
            summarized=summarized,
            skipped=skipped,
            failed=failed,
            source_counts=source_counts,
        )

    def _fetch_feed_items(self, feed: MacroResearchFeed) -> list[dict[str, Any]]:
        response = self.http.get(feed.url, timeout=30)
        response.raise_for_status()
        raw = response.text
        root = ET.fromstring(raw.encode("utf-8"))
        items = root.findall(".//item")
        if not items and root.tag.rsplit("}", 1)[-1].lower() == "feed":
            items = root.findall(".//{*}entry")
        out: list[dict[str, Any]] = []
        for item in items:
            title = _child_text(item, ("title",))
            link = _child_text(item, ("link",))
            if not link:
                for child in list(item):
                    if child.tag.rsplit("}", 1)[-1].lower() == "link" and child.attrib.get("href"):
                        link = child.attrib["href"]
                        break
            guid = _child_text(item, ("guid", "id"))
            desc = _child_text(item, ("description", "summary", "content", "encoded"))
            pub = _child_text(item, ("pubDate", "published", "updated", "dc:date"))
            out.append(
                {
                    "title": html.unescape(title),
                    "link": html.unescape(link),
                    "guid": html.unescape(guid),
                    "description": desc,
                    "published_at": _parse_datetime(pub),
                    "raw_text": _item_xml(item),
                }
            )
        return out

    def _document_from_item(self, feed: MacroResearchFeed, item: dict[str, Any]) -> MacroResearchDocument:
        title = _bounded_text(item.get("title"), max_len=300)
        link = unquote(html.unescape(_text(item.get("link") or item.get("guid"))))
        guid = unquote(html.unescape(_text(item.get("guid") or link)))
        published_at = item.get("published_at") if isinstance(item.get("published_at"), datetime) else None
        description = _text(item.get("description"))
        cleaned = _clean_html(description)
        pdf_url = (_extract_pdf_urls(description) or [""])[0]
        pdf_bytes = b""
        if pdf_url:
            try:
                response = self.http.get(pdf_url, timeout=45)
                response.raise_for_status()
                pdf_bytes = bytes(getattr(response, "content", b"") or b"")
            except Exception as exc:
                logger.warning(
                    "[yellow]Macro research PDF fetch skipped[/yellow] url=%s err=%s",
                    pdf_url,
                    str(exc),
                )
        source_doc_id = _source_doc_id(feed, guid=guid, link=link, title=title, published_at=published_at)
        content_text = "\n\n".join(part for part in (title, cleaned, f"PDF: {pdf_url}" if pdf_url else "") if part)
        hash_input = content_text.encode("utf-8") + b"\n" + hashlib.sha256(pdf_bytes).hexdigest().encode("ascii")
        return MacroResearchDocument(
            source_doc_id=source_doc_id,
            source=_clean_token(feed.source),
            feed_id=_clean_token(feed.feed_id),
            doc_type=_clean_token(feed.doc_type),
            region=_clean_token(feed.region),
            market=_clean_token(feed.market or "all"),
            title=title,
            source_url=link,
            published_at=published_at,
            fetched_at=utc_now(),
            content_hash=hashlib.sha256(hash_input).hexdigest(),
            content_text=content_text,
            raw_text=_text(item.get("raw_text")),
            themes=tuple(_clean_token(theme) for theme in feed.themes if _clean_token(theme)),
            pdf_url=pdf_url,
            pdf_bytes=pdf_bytes,
        )

    def _base_object_path(self, document: MacroResearchDocument) -> str:
        dt = document.published_at or document.fetched_at
        day = dt.date().isoformat()
        return "/".join(
            [
                "macro_research",
                _safe_path_token(document.source),
                _safe_path_token(document.feed_id),
                day,
                _safe_path_token(document.source_doc_id),
            ]
        )

    def _write_objects(self, document: MacroResearchDocument) -> MacroResearchDocument:
        if self.object_store is None:
            return document
        base = self._base_object_path(document)
        raw_uri = self.object_store.put_text(
            f"{base}/raw.xml",
            document.raw_text,
            content_type="application/rss+xml; charset=utf-8",
        )
        content_uri = self.object_store.put_text(
            f"{base}/content.txt",
            document.content_text,
            content_type="text/plain; charset=utf-8",
        )
        pdf_uri = ""
        if document.pdf_bytes:
            pdf_uri = self.object_store.put_bytes(
                f"{base}/source.pdf",
                document.pdf_bytes,
                content_type="application/pdf",
            )
        return replace(
            document,
            raw_gcs_uri=raw_uri,
            content_gcs_uri=content_uri,
            pdf_gcs_uri=pdf_uri,
        )

    def _get_existing_document(self, source_doc_id: str) -> dict[str, Any] | None:
        getter = getattr(self.repo, "get_macro_research_document", None)
        if not callable(getter):
            return None
        return getter(source_doc_id, tenant_id=self.tenant_id)

    def _upsert_document(self, document: MacroResearchDocument, *, status: str) -> None:
        writer = getattr(self.repo, "upsert_macro_research_document", None)
        if not callable(writer):
            return
        writer(
            {
                "source_doc_id": document.source_doc_id,
                "source": document.source,
                "feed_id": document.feed_id,
                "doc_type": document.doc_type,
                "region": document.region or None,
                "market": document.market or "all",
                "title": document.title,
                "source_url": document.source_url,
                "published_at": document.published_at,
                "fetched_at": document.fetched_at,
                "content_hash": document.content_hash,
                "raw_gcs_uri": document.raw_gcs_uri or None,
                "content_gcs_uri": document.content_gcs_uri or None,
                "pdf_gcs_uri": document.pdf_gcs_uri or None,
                "text_char_count": len(document.content_text),
                "status": status,
                "summary_status": "summarized" if status == "summarized" else "pending",
                "error_message": None,
                "themes": list(document.themes),
                "detail_json": {
                    "pdf_url": document.pdf_url,
                    "content_preview": document.content_text[:1000],
                },
            },
            tenant_id=self.tenant_id,
        )

    def _upsert_briefing(self, document: MacroResearchDocument, summary: MacroResearchSummary) -> None:
        writer = getattr(self.repo, "upsert_macro_research_briefing", None)
        if not callable(writer):
            return
        detail_json = {
            **summary.detail_json,
            "schema_version": "macro_research_summary.v2",
            "source_doc_id": document.source_doc_id,
            "source_url": document.source_url,
            "executive_summary": summary.executive_summary,
            "key_findings": summary.key_findings,
            "methodology": summary.methodology,
            "macro_channels": summary.macro_channels,
            "asset_implications": summary.asset_implications,
            "watch_indicators": summary.watch_indicators,
            "caveats": summary.caveats,
            "market_implication": summary.market_implication,
            "themes": summary.themes or list(document.themes),
            "document": {
                "raw_gcs_uri": document.raw_gcs_uri,
                "content_gcs_uri": document.content_gcs_uri,
                "pdf_gcs_uri": document.pdf_gcs_uri,
            },
        }
        writer(
            {
                "source_doc_id": document.source_doc_id,
                "created_at": utc_now(),
                "published_at": document.published_at,
                "source": document.source,
                "feed_id": document.feed_id,
                "doc_type": document.doc_type,
                "region": document.region or None,
                "market": document.market or "all",
                "title": document.title,
                "source_url": document.source_url,
                "headline": summary.headline,
                "summary": summary.summary,
                "key_points": summary.key_findings,
                "market_implication": summary.market_implication,
                "risk_flags": summary.caveats,
                "themes": summary.themes or list(document.themes),
                "confidence": summary.confidence,
                "model": _text(getattr(self.settings, "research_gemini_model", "")) or "gemini-3-flash-preview",
                "detail_json": detail_json,
            },
            tenant_id=self.tenant_id,
        )


def macro_research_bucket_default(project: str) -> str:
    project_token = _text(project)
    return f"{project_token}-macro-research" if project_token else ""
