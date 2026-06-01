from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from arena.models import utc_now

if TYPE_CHECKING:
    from arena.data.bigquery.session import BigQuerySession

logger = logging.getLogger(__name__)


_DOCUMENT_COLUMNS = (
    "source_doc_id",
    "source",
    "feed_id",
    "doc_type",
    "region",
    "market",
    "title",
    "source_url",
    "published_at",
    "fetched_at",
    "content_hash",
    "raw_gcs_uri",
    "content_gcs_uri",
    "pdf_gcs_uri",
    "text_char_count",
    "status",
    "summary_status",
    "error_message",
    "themes",
    "detail_json",
)

_BRIEFING_COLUMNS = (
    "source_doc_id",
    "created_at",
    "published_at",
    "source",
    "feed_id",
    "doc_type",
    "region",
    "market",
    "title",
    "source_url",
    "headline",
    "summary",
    "key_points",
    "market_implication",
    "risk_flags",
    "themes",
    "confidence",
    "model",
    "detail_json",
)

_THESIS_COLUMNS = (
    "thesis_id",
    "source_doc_id",
    "created_at",
    "published_at",
    "source",
    "feed_id",
    "doc_type",
    "region",
    "market",
    "title",
    "source_url",
    "theme_key",
    "horizon",
    "thesis",
    "transmission_channels",
    "affected_sectors",
    "candidate_queries",
    "watch_indicators",
    "invalidation_conditions",
    "confidence_label",
    "status",
    "evidence_json",
    "detail_json",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if str(item).strip()]


def _json_or_none(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    try:
        return json.loads(str(value))
    except Exception:
        return None


class MacroResearchStore:
    def __init__(self, session: BigQuerySession) -> None:
        self.session = session

    def get_macro_research_document(
        self,
        source_doc_id: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        _ = tenant_id
        rows = self.session.fetch_rows(
            f"""
            SELECT *
            FROM `{self.session.dataset_fqn}.macro_research_documents`
            WHERE source_doc_id = @source_doc_id
            LIMIT 1
            """,
            {
                "source_doc_id": str(source_doc_id or "").strip(),
            },
        )
        if not rows:
            return None
        rows[0]["detail_json"] = _json_or_none(rows[0].get("detail_json"))
        return rows[0]

    def upsert_macro_research_document(self, row: dict[str, Any], *, tenant_id: str | None = None) -> None:
        _ = tenant_id
        source_doc_id = str(row.get("source_doc_id") or "").strip()
        if not source_doc_id:
            raise ValueError("source_doc_id is required")
        self.session.execute(
            f"""
            DELETE FROM `{self.session.dataset_fqn}.macro_research_documents`
            WHERE source_doc_id = @source_doc_id
            """,
            {"source_doc_id": source_doc_id},
        )
        payload = dict(row)
        payload["source_doc_id"] = source_doc_id
        payload["fetched_at"] = payload.get("fetched_at") or utc_now()
        payload["themes"] = _str_list(payload.get("themes"))
        payload["detail_json"] = _json_value(payload.get("detail_json") or {})
        self._insert_json("macro_research_documents", [{col: _json_safe(payload.get(col)) for col in _DOCUMENT_COLUMNS}])

    def upsert_macro_research_briefing(self, row: dict[str, Any], *, tenant_id: str | None = None) -> None:
        _ = tenant_id
        source_doc_id = str(row.get("source_doc_id") or "").strip()
        if not source_doc_id:
            raise ValueError("source_doc_id is required")
        self.session.execute(
            f"""
            DELETE FROM `{self.session.dataset_fqn}.macro_research_briefings`
            WHERE source_doc_id = @source_doc_id
            """,
            {"source_doc_id": source_doc_id},
        )
        payload = dict(row)
        payload["source_doc_id"] = source_doc_id
        payload["created_at"] = payload.get("created_at") or utc_now()
        payload["key_points"] = [str(item).strip() for item in (payload.get("key_points") or []) if str(item).strip()]
        payload["risk_flags"] = [str(item).strip() for item in (payload.get("risk_flags") or []) if str(item).strip()]
        payload["themes"] = _str_list(payload.get("themes"))
        payload["detail_json"] = _json_value(payload.get("detail_json") or {})
        self._insert_json("macro_research_briefings", [{col: _json_safe(payload.get(col)) for col in _BRIEFING_COLUMNS}])

    def get_macro_research_briefings(
        self,
        *,
        source_doc_ids: list[str] | None = None,
        sources: list[str] | None = None,
        doc_types: list[str] | None = None,
        themes: list[str] | None = None,
        market: str | None = None,
        since: datetime | None = None,
        limit: int = 10,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        _ = tenant_id
        conditions = ["TRUE"]
        params: dict[str, Any] = {
            "limit": max(1, min(int(limit), 50)),
        }
        clean_source_doc_ids = [str(item or "").strip() for item in (source_doc_ids or []) if str(item or "").strip()]
        clean_sources = _str_list(sources)
        clean_doc_types = _str_list(doc_types)
        clean_themes = _str_list(themes)
        clean_market = str(market or "").strip().lower()
        if clean_source_doc_ids:
            conditions.append("source_doc_id IN UNNEST(@source_doc_ids)")
            params["source_doc_ids"] = clean_source_doc_ids
        if clean_sources:
            conditions.append("source IN UNNEST(@sources)")
            params["sources"] = clean_sources
        if clean_doc_types:
            conditions.append("doc_type IN UNNEST(@doc_types)")
            params["doc_types"] = clean_doc_types
        if clean_themes:
            conditions.append(
                """
                EXISTS (
                  SELECT 1
                  FROM UNNEST(themes) AS theme
                  CROSS JOIN UNNEST(@themes) AS requested
                  WHERE LOWER(theme) = requested
                     OR REPLACE(LOWER(theme), ' ', '_') = requested
                     OR LOWER(theme) LIKE CONCAT('%', REPLACE(requested, '_', ' '), '%')
                     OR REPLACE(LOWER(theme), ' ', '_') LIKE CONCAT('%', requested, '%')
                )
                """
            )
            params["themes"] = clean_themes
        if clean_market and clean_market != "all":
            conditions.append("(market = @market OR market = 'all')")
            params["market"] = clean_market
        if since is not None:
            conditions.append("COALESCE(published_at, created_at) >= @since")
            params["since"] = since
        rows = self.session.fetch_rows(
            f"""
            SELECT *
            FROM `{self.session.dataset_fqn}.macro_research_briefings`
            WHERE {' AND '.join(conditions)}
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT @limit
            """,
            params,
        )
        for row in rows:
            row["detail_json"] = _json_or_none(row.get("detail_json"))
        return rows

    def replace_macro_research_theses(
        self,
        source_doc_id: str,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        _ = tenant_id
        clean_source_doc_id = str(source_doc_id or "").strip()
        if not clean_source_doc_id:
            raise ValueError("source_doc_id is required")
        self.session.execute(
            f"""
            DELETE FROM `{self.session.dataset_fqn}.macro_research_theses`
            WHERE source_doc_id = @source_doc_id
            """,
            {"source_doc_id": clean_source_doc_id},
        )
        payload_rows: list[dict[str, Any]] = []
        for row in rows or []:
            payload = dict(row)
            payload["source_doc_id"] = clean_source_doc_id
            payload["thesis_id"] = str(payload.get("thesis_id") or "").strip()
            payload["created_at"] = payload.get("created_at") or utc_now()
            payload["status"] = str(payload.get("status") or "active").strip().lower() or "active"
            payload["theme_key"] = str(payload.get("theme_key") or "").strip().lower() or None
            for key in (
                "transmission_channels",
                "affected_sectors",
                "candidate_queries",
                "watch_indicators",
                "invalidation_conditions",
            ):
                payload[key] = [str(item).strip() for item in (payload.get(key) or []) if str(item).strip()]
            payload["evidence_json"] = _json_value(payload.get("evidence_json") or {})
            payload["detail_json"] = _json_value(payload.get("detail_json") or {})
            if not payload["thesis_id"] or not str(payload.get("thesis") or "").strip():
                continue
            payload_rows.append({col: _json_safe(payload.get(col)) for col in _THESIS_COLUMNS})
        self._insert_json("macro_research_theses", payload_rows)

    def get_macro_research_theses(
        self,
        *,
        source_doc_ids: list[str] | None = None,
        themes: list[str] | None = None,
        market: str | None = None,
        status: str | None = "active",
        since: datetime | None = None,
        limit: int = 10,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        _ = tenant_id
        conditions = ["TRUE"]
        params: dict[str, Any] = {"limit": max(1, min(int(limit), 50))}
        clean_source_doc_ids = [str(item or "").strip() for item in (source_doc_ids or []) if str(item or "").strip()]
        clean_themes = _str_list(themes)
        clean_market = str(market or "").strip().lower()
        clean_status = str(status or "").strip().lower()
        if clean_source_doc_ids:
            conditions.append("source_doc_id IN UNNEST(@source_doc_ids)")
            params["source_doc_ids"] = clean_source_doc_ids
        if clean_themes:
            conditions.append("theme_key IN UNNEST(@themes)")
            params["themes"] = clean_themes
        if clean_market and clean_market != "all":
            conditions.append("(market = @market OR market = 'all')")
            params["market"] = clean_market
        if clean_status:
            conditions.append("status = @status")
            params["status"] = clean_status
        if since is not None:
            conditions.append("COALESCE(published_at, created_at) >= @since")
            params["since"] = since
        rows = self.session.fetch_rows(
            f"""
            SELECT *
            FROM `{self.session.dataset_fqn}.macro_research_theses`
            WHERE {' AND '.join(conditions)}
            ORDER BY COALESCE(published_at, created_at) DESC, created_at DESC
            LIMIT @limit
            """,
            params,
        )
        for row in rows:
            row["evidence_json"] = _json_or_none(row.get("evidence_json"))
            row["detail_json"] = _json_or_none(row.get("detail_json"))
        return rows

    def _insert_json(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table_id = f"{self.session.dataset_fqn}.{table}"
        errors = self.session.client.insert_rows_json(table_id, rows)
        if errors:
            logger.error("[red]BigQuery macro research insert failed[/red] table=%s errors=%s", table_id, errors)
            raise RuntimeError(f"BigQuery insert failed for {table}: {errors}")
