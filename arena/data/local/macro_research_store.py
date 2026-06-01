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


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if str(item).strip()]


def _theme_matches(row_themes: Any, requested_themes: set[str]) -> bool:
    if not requested_themes:
        return True
    stored = _str_list(row_themes)
    for theme in stored:
        compact = theme.replace(" ", "_")
        for requested in requested_themes:
            phrase = requested.replace("_", " ")
            if theme == requested or compact == requested:
                return True
            if phrase and phrase in theme:
                return True
            if requested and requested in compact:
                return True
    return False


def _json_or_none(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    try:
        return json.loads(str(value))
    except Exception:
        return None


class LocalMacroResearchStore:
    def __init__(self, session: DuckDBSession) -> None:
        self.session = session

    def get_macro_research_document(
        self,
        source_doc_id: str,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        _ = tenant_id
        rows = self.session.fetch_rows(
            """
            SELECT *
            FROM macro_research_documents
            WHERE source_doc_id = $source_doc_id
            LIMIT 1
            """,
            {
                "source_doc_id": str(source_doc_id or "").strip(),
            },
        )
        if not rows:
            return None
        row = rows[0]
        row["detail_json"] = _json_or_none(row.get("detail_json"))
        return row

    def upsert_macro_research_document(self, row: dict[str, Any], *, tenant_id: str | None = None) -> None:
        _ = tenant_id
        source_doc_id = str(row.get("source_doc_id") or "").strip()
        if not source_doc_id:
            raise ValueError("source_doc_id is required")
        self.session.execute(
            """
            DELETE FROM macro_research_documents
            WHERE source_doc_id = $source_doc_id
            """,
            {"source_doc_id": source_doc_id},
        )
        payload = dict(row)
        payload["source_doc_id"] = source_doc_id
        payload["fetched_at"] = payload.get("fetched_at") or utc_now()
        payload["status"] = str(payload.get("status") or "stored").strip().lower()
        payload["themes"] = _str_list(payload.get("themes"))
        payload["detail_json"] = _json_cell(payload.get("detail_json") or {})
        self.session.insert_dicts("macro_research_documents", [payload])

    def upsert_macro_research_briefing(self, row: dict[str, Any], *, tenant_id: str | None = None) -> None:
        _ = tenant_id
        source_doc_id = str(row.get("source_doc_id") or "").strip()
        if not source_doc_id:
            raise ValueError("source_doc_id is required")
        self.session.execute(
            """
            DELETE FROM macro_research_briefings
            WHERE source_doc_id = $source_doc_id
            """,
            {"source_doc_id": source_doc_id},
        )
        payload = dict(row)
        payload["source_doc_id"] = source_doc_id
        payload["created_at"] = payload.get("created_at") or utc_now()
        payload["key_points"] = [str(item).strip() for item in (payload.get("key_points") or []) if str(item).strip()]
        payload["risk_flags"] = [str(item).strip() for item in (payload.get("risk_flags") or []) if str(item).strip()]
        payload["themes"] = _str_list(payload.get("themes"))
        payload["detail_json"] = _json_cell(payload.get("detail_json") or {})
        self.session.insert_dicts("macro_research_briefings", [payload])

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
        params: dict[str, Any] = {"limit": max(1, min(int(limit), 50))}
        clean_source_doc_ids = [str(item or "").strip() for item in (source_doc_ids or []) if str(item or "").strip()]
        clean_sources = _str_list(sources)
        clean_doc_types = _str_list(doc_types)
        clean_market = str(market or "").strip().lower()
        if clean_source_doc_ids:
            conditions.append("source_doc_id IN (SELECT unnest($source_doc_ids))")
            params["source_doc_ids"] = clean_source_doc_ids
        if clean_sources:
            conditions.append("source IN (SELECT unnest($sources))")
            params["sources"] = clean_sources
        if clean_doc_types:
            conditions.append("doc_type IN (SELECT unnest($doc_types))")
            params["doc_types"] = clean_doc_types
        if clean_market and clean_market != "all":
            conditions.append("(market = $market OR market = 'all')")
            params["market"] = clean_market
        if since is not None:
            conditions.append("COALESCE(published_at, created_at) >= $since")
            params["since"] = since
        rows = self.session.fetch_rows(
            f"""
            SELECT *
            FROM macro_research_briefings
            WHERE {' AND '.join(conditions)}
            ORDER BY COALESCE(published_at, created_at) DESC
            LIMIT $limit
            """,
            params,
        )
        clean_themes = set(_str_list(themes))
        out: list[dict[str, Any]] = []
        for row in rows:
            row["detail_json"] = _json_or_none(row.get("detail_json"))
            if not _theme_matches(row.get("themes"), clean_themes):
                continue
            out.append(row)
            if len(out) >= params["limit"]:
                break
        return out

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
            """
            DELETE FROM macro_research_theses
            WHERE source_doc_id = $source_doc_id
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
            payload["evidence_json"] = _json_cell(payload.get("evidence_json") or {})
            payload["detail_json"] = _json_cell(payload.get("detail_json") or {})
            if not payload["thesis_id"] or not str(payload.get("thesis") or "").strip():
                continue
            payload_rows.append(payload)
        if payload_rows:
            self.session.insert_dicts("macro_research_theses", payload_rows)

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
        clean_themes = set(_str_list(themes))
        clean_market = str(market or "").strip().lower()
        clean_status = str(status or "").strip().lower()
        if clean_source_doc_ids:
            conditions.append("source_doc_id IN (SELECT unnest($source_doc_ids))")
            params["source_doc_ids"] = clean_source_doc_ids
        if clean_market and clean_market != "all":
            conditions.append("(market = $market OR market = 'all')")
            params["market"] = clean_market
        if clean_status:
            conditions.append("status = $status")
            params["status"] = clean_status
        if since is not None:
            conditions.append("COALESCE(published_at, created_at) >= $since")
            params["since"] = since
        rows = self.session.fetch_rows(
            f"""
            SELECT *
            FROM macro_research_theses
            WHERE {' AND '.join(conditions)}
            ORDER BY COALESCE(published_at, created_at) DESC, created_at DESC
            LIMIT $limit
            """,
            params,
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            row["evidence_json"] = _json_or_none(row.get("evidence_json"))
            row["detail_json"] = _json_or_none(row.get("detail_json"))
            if clean_themes and str(row.get("theme_key") or "").strip().lower() not in clean_themes:
                continue
            out.append(row)
            if len(out) >= params["limit"]:
                break
        return out
