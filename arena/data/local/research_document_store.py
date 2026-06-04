from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from arena.data.local.session import DuckDBSession
from arena.models import utc_now


def _json_cell(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _json_or_none(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    try:
        return json.loads(str(value))
    except Exception:
        return None


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if str(item).strip()]


class LocalResearchDocumentStore:
    def __init__(self, session: DuckDBSession) -> None:
        self.session = session

    def upsert_research_document(self, row: dict[str, Any], *, tenant_id: str | None = None) -> None:
        tenant = self.session.resolve_tenant_id(tenant_id)
        source_doc_id = str(row.get("source_doc_id") or "").strip()
        if not source_doc_id:
            raise ValueError("source_doc_id is required")
        trading_mode = str(row.get("trading_mode") or "paper").strip().lower() or "paper"
        self.session.execute(
            """
            DELETE FROM research_documents
            WHERE tenant_id = $tenant_id
              AND source_doc_id = $source_doc_id
              AND trading_mode = $trading_mode
            """,
            {"tenant_id": tenant, "source_doc_id": source_doc_id, "trading_mode": trading_mode},
        )
        payload = dict(row)
        payload["tenant_id"] = tenant
        payload["source_doc_id"] = source_doc_id
        payload["fetched_at"] = payload.get("fetched_at") or utc_now()
        payload["trading_mode"] = trading_mode
        payload["status"] = str(payload.get("status") or "listed").strip().lower() or "listed"
        payload["detail_json"] = _json_cell(payload.get("detail_json") or {})
        self.session.insert_dicts("research_documents", [payload])

    def get_research_document(
        self,
        source_doc_id: str,
        *,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        tenant = self.session.resolve_tenant_id(tenant_id)
        rows = self.session.fetch_rows(
            """
            SELECT *
            FROM research_documents
            WHERE tenant_id = $tenant_id
              AND source_doc_id = $source_doc_id
              AND trading_mode = $trading_mode
            LIMIT 1
            """,
            {
                "tenant_id": tenant,
                "source_doc_id": str(source_doc_id or "").strip(),
                "trading_mode": str(trading_mode or "paper").strip().lower() or "paper",
            },
        )
        if not rows:
            return None
        row = rows[0]
        row["detail_json"] = _json_or_none(row.get("detail_json"))
        return row

    def get_research_documents(
        self,
        *,
        source_doc_ids: list[str] | None = None,
        tickers: list[str] | None = None,
        categories: list[str] | None = None,
        sources: list[str] | None = None,
        market: str | None = None,
        since: datetime | None = None,
        limit: int = 10,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self.session.resolve_tenant_id(tenant_id)
        conditions = ["tenant_id = $tenant_id", "trading_mode = $trading_mode"]
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "trading_mode": str(trading_mode or "paper").strip().lower() or "paper",
            "limit": max(1, min(int(limit), 50)),
        }
        clean_source_doc_ids = [str(item or "").strip() for item in (source_doc_ids or []) if str(item or "").strip()]
        clean_tickers = [str(item or "").strip().upper() for item in (tickers or []) if str(item or "").strip()]
        clean_categories = _str_list(categories)
        clean_sources = _str_list(sources)
        clean_market = str(market or "").strip().lower()
        if clean_source_doc_ids:
            conditions.append("source_doc_id IN (SELECT unnest($source_doc_ids))")
            params["source_doc_ids"] = clean_source_doc_ids
        filters: list[str] = []
        if clean_tickers:
            filters.append("ticker IN (SELECT unnest($tickers))")
            params["tickers"] = clean_tickers
        if clean_categories:
            filters.append("category IN (SELECT unnest($categories))")
            params["categories"] = clean_categories
        if filters:
            conditions.append(f"({' OR '.join(filters)})")
        if clean_sources:
            conditions.append("source IN (SELECT unnest($sources))")
            params["sources"] = clean_sources
        if clean_market and clean_market != "all":
            conditions.append("(market = $market OR market = 'all')")
            params["market"] = clean_market
        if since is not None:
            conditions.append("COALESCE(published_at, fetched_at) >= $since")
            params["since"] = since
        rows = self.session.fetch_rows(
            f"""
            SELECT *
            FROM research_documents
            WHERE {' AND '.join(conditions)}
            ORDER BY COALESCE(published_at, fetched_at) DESC, fetched_at DESC
            LIMIT $limit
            """,
            params,
        )
        for row in rows:
            row["detail_json"] = _json_or_none(row.get("detail_json"))
        return rows

    def update_research_document_snapshot(
        self,
        source_doc_id: str,
        *,
        content_hash: str | None = None,
        content_gcs_uri: str | None = None,
        text_char_count: int | None = None,
        status: str = "read",
        error_message: str | None = None,
        trading_mode: str = "paper",
        tenant_id: str | None = None,
    ) -> None:
        tenant = self.session.resolve_tenant_id(tenant_id)
        self.session.execute(
            """
            UPDATE research_documents
            SET content_hash = COALESCE($content_hash, content_hash),
                content_gcs_uri = COALESCE($content_gcs_uri, content_gcs_uri),
                text_char_count = COALESCE($text_char_count, text_char_count),
                status = $status,
                error_message = $error_message
            WHERE tenant_id = $tenant_id
              AND source_doc_id = $source_doc_id
              AND trading_mode = $trading_mode
            """,
            {
                "tenant_id": tenant,
                "source_doc_id": str(source_doc_id or "").strip(),
                "trading_mode": str(trading_mode or "paper").strip().lower() or "paper",
                "content_hash": content_hash,
                "content_gcs_uri": content_gcs_uri,
                "text_char_count": text_char_count,
                "status": str(status or "read").strip().lower() or "read",
                "error_message": error_message,
            },
        )
