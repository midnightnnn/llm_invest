from __future__ import annotations

import hashlib
import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse
from xml.etree import ElementTree as ET

import requests

from arena.config import Settings
from arena.models import utc_now

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ResearchDocumentFeed:
    source: str
    feed_id: str
    category: str
    url: str
    market: str = "us"
    ticker: str = ""


@dataclass(frozen=True, slots=True)
class LiveDocumentRead:
    source_url: str
    final_url: str
    content_type: str
    content_text: str
    content_hash: str
    retrieved_at: datetime
    error: str = ""

    @property
    def text_char_count(self) -> int:
        return len(self.content_text)


DEFAULT_RESEARCH_FEEDS: tuple[ResearchDocumentFeed, ...] = (
    ResearchDocumentFeed(
        source="marketwatch",
        feed_id="marketwatch_topstories",
        category="global_market",
        url="https://www.marketwatch.com/rss/topstories",
        market="us",
    ),
    ResearchDocumentFeed(
        source="marketwatch",
        feed_id="marketwatch_bulletins",
        category="global_market",
        url="https://www.marketwatch.com/rss/bulletins",
        market="us",
    ),
    ResearchDocumentFeed(
        source="sec",
        feed_id="sec_press_releases",
        category="geopolitical",
        url="https://www.sec.gov/news/pressreleases.rss",
        market="us",
    ),
)

_CATEGORY_QUERIES: dict[str, str] = {
    "global_market": "global stock market today S&P 500 Nasdaq Treasury yields oil dollar Fed",
    "geopolitical": "geopolitical risk markets sanctions tariffs war oil supply chain",
    "sector_trends": "US stock market sector rotation technology semiconductors energy financials healthcare",
}

_HTTP_HEADERS = {
    "User-Agent": "llm-arena-research/1.0 (+https://github.com/) Python requests",
    "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf,text/plain,*/*",
}


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
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr)\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s+", "\n", text)
    return text.strip()


def _child_text(item: ET.Element, names: tuple[str, ...]) -> str:
    wanted = {name.lower() for name in names}
    for child in list(item):
        name = child.tag.rsplit("}", 1)[-1].lower()
        if name in wanted:
            return _text(child.text)
    return ""


def _child_attr(item: ET.Element, name: str, attr: str) -> str:
    wanted = name.lower()
    for child in list(item):
        child_name = child.tag.rsplit("}", 1)[-1].lower()
        if child_name == wanted:
            return _text(child.attrib.get(attr))
    return ""


def _google_news_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"


def _source_doc_id(*, source: str, feed_id: str, guid: str, link: str, title: str, published_at: datetime | None) -> str:
    stable = "|".join([source, feed_id, guid or "", link or "", title or "", published_at.isoformat() if published_at else ""])
    digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:18]
    return f"research:{_clean_token(source)}:{_clean_token(feed_id)}:{digest}"


def _parse_feed_items(raw: str) -> list[dict[str, Any]]:
    root = ET.fromstring(raw.encode("utf-8"))
    items = root.findall(".//item")
    if not items and root.tag.rsplit("}", 1)[-1].lower() == "feed":
        items = root.findall(".//{*}entry")
    out: list[dict[str, Any]] = []
    for item in items:
        link = _child_text(item, ("link",))
        if not link:
            link = _child_attr(item, "link", "href")
        title = html.unescape(_child_text(item, ("title",)))
        guid = html.unescape(_child_text(item, ("guid", "id")))
        desc = _child_text(item, ("description", "summary", "content", "encoded"))
        published_at = _parse_datetime(_child_text(item, ("pubDate", "published", "updated", "dc:date")))
        publisher = html.unescape(_child_text(item, ("source",)))
        publisher_url = _child_attr(item, "source", "url")
        out.append(
            {
                "title": title,
                "link": html.unescape(link),
                "guid": guid,
                "description": desc,
                "published_at": published_at,
                "publisher": publisher,
                "publisher_url": publisher_url,
            }
        )
    return out


def _document_row_from_item(
    feed: ResearchDocumentFeed,
    item: dict[str, Any],
    *,
    trading_mode: str,
) -> dict[str, Any] | None:
    title = re.sub(r"\s+", " ", _text(item.get("title")))[:300].strip()
    link = html.unescape(_text(item.get("link") or item.get("guid")))
    if not title or not link:
        return None
    source_url = urljoin(feed.url, link)
    published_at = item.get("published_at") if isinstance(item.get("published_at"), datetime) else None
    snippet = _clean_html(item.get("description"))[:1200]
    source_doc_id = _source_doc_id(
        source=feed.source,
        feed_id=feed.feed_id,
        guid=_text(item.get("guid")),
        link=source_url,
        title=title,
        published_at=published_at,
    )
    listing_hash = hashlib.sha256(
        "\n".join([source_doc_id, title, source_url, snippet, published_at.isoformat() if published_at else ""]).encode(
            "utf-8"
        )
    ).hexdigest()
    publisher = _text(item.get("publisher"))
    publisher_url = _text(item.get("publisher_url"))
    if not publisher:
        host = urlparse(publisher_url or source_url).netloc
        publisher = host.replace("www.", "")
    return {
        "source_doc_id": source_doc_id,
        "source": _clean_token(feed.source),
        "feed_id": _clean_token(feed.feed_id),
        "category": _clean_token(feed.category) or "global_market",
        "market": _clean_token(feed.market) or "us",
        "ticker": _text(feed.ticker).upper() or None,
        "publisher": publisher or None,
        "publisher_url": publisher_url or None,
        "title": title,
        "source_url": source_url,
        "published_at": published_at,
        "fetched_at": utc_now(),
        "snippet": snippet or None,
        "content_hash": listing_hash,
        "content_gcs_uri": None,
        "text_char_count": len(snippet) if snippet else None,
        "status": "listed",
        "error_message": None,
        "detail_json": {"feed_url": feed.url},
        "trading_mode": str(trading_mode or "paper").strip().lower() or "paper",
    }


class ResearchDocumentService:
    def __init__(self, *, settings: Settings, repo: Any, http: Any | None = None, tenant_id: str | None = None) -> None:
        self.settings = settings
        self.repo = repo
        self.http = http or requests.Session()
        self.tenant_id = tenant_id

    def _feeds(self, *, tickers: list[str] | None, categories: list[str] | None) -> list[ResearchDocumentFeed]:
        clean_categories = [_clean_token(category) for category in (categories or []) if _clean_token(category)]
        if not clean_categories and not tickers:
            clean_categories = list(_CATEGORY_QUERIES)
        feeds: list[ResearchDocumentFeed] = []
        category_set = set(clean_categories)
        for feed in DEFAULT_RESEARCH_FEEDS:
            if not category_set or feed.category in category_set:
                feeds.append(feed)
        for category in clean_categories:
            query = _CATEGORY_QUERIES.get(category)
            if query:
                feeds.append(
                    ResearchDocumentFeed(
                        source="google_news",
                        feed_id=f"google_news_{category}",
                        category=category,
                        url=_google_news_url(query),
                        market="us",
                    )
                )
        for ticker in [str(item or "").strip().upper() for item in (tickers or []) if str(item or "").strip()]:
            feeds.append(
                ResearchDocumentFeed(
                    source="google_news",
                    feed_id=f"google_news_{ticker.lower()}",
                    category="held",
                    url=_google_news_url(f"{ticker} stock earnings guidance shares"),
                    market="us",
                    ticker=ticker,
                )
            )
        return feeds

    def refresh(
        self,
        *,
        tickers: list[str] | None = None,
        categories: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        writer = getattr(self.repo, "upsert_research_document", None)
        if not callable(writer):
            return []
        max_items = max(1, min(int(limit), 20))
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for feed in self._feeds(tickers=tickers, categories=categories):
            try:
                response = self.http.get(feed.url, headers=_HTTP_HEADERS, timeout=30)
                response.raise_for_status()
                items = _parse_feed_items(response.text)
            except Exception as exc:
                logger.warning("[yellow]Research document feed fetch failed[/yellow] feed=%s err=%s", feed.feed_id, str(exc))
                continue
            for item in items[:max_items]:
                row = _document_row_from_item(feed, item, trading_mode=getattr(self.settings, "trading_mode", "paper"))
                if not row:
                    continue
                source_doc_id = str(row.get("source_doc_id") or "")
                if source_doc_id in seen:
                    continue
                seen.add(source_doc_id)
                try:
                    writer(row, tenant_id=self.tenant_id)
                except Exception as exc:
                    logger.warning(
                        "[yellow]Research document upsert failed[/yellow] source_doc_id=%s err=%s",
                        source_doc_id,
                        str(exc),
                    )
                    continue
                rows.append(row)
        return rows


def fetch_live_document(source_url: str, *, http: Any | None = None) -> LiveDocumentRead:
    url = _text(source_url)
    retrieved_at = utc_now()
    if not url:
        return LiveDocumentRead("", "", "", "", "", retrieved_at, error="missing source_url")
    session = http or requests.Session()
    try:
        response = session.get(url, headers=_HTTP_HEADERS, timeout=45)
        response.raise_for_status()
        content_type = str(response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        final_url = str(getattr(response, "url", "") or url)
        raw_bytes = bytes(getattr(response, "content", b"") or b"")
        if content_type == "application/pdf" or final_url.lower().split("?", 1)[0].endswith(".pdf"):
            from arena.macro_research import _extract_pdf_text

            text = _extract_pdf_text(raw_bytes)
        else:
            text = _clean_html(getattr(response, "text", "") or raw_bytes.decode("utf-8", errors="replace"))
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""
        return LiveDocumentRead(url, final_url, content_type, text, digest, retrieved_at)
    except Exception as exc:
        return LiveDocumentRead(url, "", "", "", "", retrieved_at, error=str(exc)[:500])


def write_research_snapshot(
    settings: Settings,
    *,
    namespace: str,
    row: dict[str, Any],
    content_text: str,
) -> str:
    bucket = str(getattr(settings, "macro_research_gcs_bucket", "") or "").strip()
    if not bucket or not content_text:
        return ""
    try:
        from arena.macro_research import GcsMacroResearchObjectStore

        store = GcsMacroResearchObjectStore(bucket_name=bucket, project=getattr(settings, "google_cloud_project", "") or None)
        dt = utc_now().date().isoformat()
        path = "/".join(
            [
                "research_evidence",
                _safe_path_token(namespace),
                _safe_path_token(row.get("source") or "source"),
                _safe_path_token(row.get("feed_id") or "feed"),
                dt,
                _safe_path_token(row.get("source_doc_id") or "document"),
                "content.txt",
            ]
        )
        return store.put_text(path, content_text, content_type="text/plain; charset=utf-8")
    except Exception as exc:
        logger.warning("[yellow]Research evidence snapshot skipped[/yellow] err=%s", str(exc))
        return ""
