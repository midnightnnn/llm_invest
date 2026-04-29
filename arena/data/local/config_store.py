"""Local DuckDB-backed implementation of the ``ConfigStore`` protocol.

Mirrors the behaviour of ``arena/data/bigquery/runtime_store.py``'s config
methods: append-only writes to ``arena_config``, latest row per
``(tenant_id, config_key)`` wins.

DuckDB SQL is written from scratch — we never reuse BigQuery dialect text.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from arena.data.local.session import DuckDBSession


_VALID_CONFIG_KEY_RE = re.compile(r"^[a-zA-Z0-9_.\-]+$")


def _norm_config_key(value: str | None) -> str:
    token = str(value or "").strip()
    if not token or not _VALID_CONFIG_KEY_RE.match(token):
        return ""
    return token


class LocalConfigStore:
    """Append-only ``arena_config`` table backed by DuckDB."""

    def __init__(self, session: DuckDBSession) -> None:
        self.session = session

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        tenant = str(tenant_id or "").strip().lower()
        key = _norm_config_key(config_key)
        if not tenant or not key:
            return None
        rows = self.session.fetch_rows(
            """
            SELECT config_value
            FROM arena_config
            WHERE tenant_id = $tenant_id
              AND config_key = $config_key
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            {"tenant_id": tenant, "config_key": key},
        )
        if not rows:
            return None
        value = rows[0].get("config_value")
        return None if value is None else str(value)

    def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]:
        tenant = str(tenant_id or "").strip().lower()
        keys = [_norm_config_key(k) for k in (config_keys or [])]
        keys = [k for k in keys if k]
        if not tenant or not keys:
            return {}
        rows = self.session.fetch_rows(
            """
            WITH ranked AS (
              SELECT
                config_key,
                config_value,
                ROW_NUMBER() OVER (
                  PARTITION BY config_key
                  ORDER BY updated_at DESC
                ) AS rn
              FROM arena_config
              WHERE tenant_id = $tenant_id
                AND config_key IN (SELECT unnest($config_keys))
            )
            SELECT config_key, config_value
            FROM ranked
            WHERE rn = 1
            """,
            {"tenant_id": tenant, "config_keys": keys},
        )
        out: dict[str, str] = {}
        for row in rows:
            key = _norm_config_key(str(row.get("config_key") or ""))
            if key:
                out[key] = str(row.get("config_value") or "")
        return out

    def set_config(
        self,
        tenant_id: str,
        config_key: str,
        value: str,
        updated_by: str | None = None,
        *,
        updated_at: datetime | None = None,
    ) -> None:
        tenant = str(tenant_id or "").strip().lower()
        key = _norm_config_key(config_key)
        if not tenant:
            raise ValueError("tenant_id is required")
        if not key:
            raise ValueError("config_key is required")

        ts = updated_at or datetime.now(timezone.utc)
        self.session.execute(
            """
            INSERT INTO arena_config (tenant_id, config_key, config_value, updated_at, updated_by)
            VALUES ($tenant_id, $config_key, $config_value, $updated_at, $updated_by)
            """,
            {
                "tenant_id": tenant,
                "config_key": key,
                "config_value": str(value or ""),
                "updated_at": ts,
                "updated_by": str(updated_by or "").strip() or None,
            },
        )

    def list_runtime_tenants(self, *, limit: int = 200) -> list[str]:
        """Lists tenants seen in arena_config."""
        lim = max(1, min(int(limit), 2000))
        rows = self.session.fetch_rows(
            """
            SELECT DISTINCT tenant_id
            FROM arena_config
            WHERE tenant_id IS NOT NULL AND TRIM(tenant_id) != ''
            ORDER BY tenant_id ASC
            LIMIT $limit
            """,
            {"limit": lim},
        )
        return [str(r.get("tenant_id") or "").strip().lower() for r in rows if r.get("tenant_id")]

    def upsert_runtime_credentials(
        self,
        *,
        tenant_id: str,
        updated_at: datetime | None = None,
        updated_by: str | None = None,
        kis_secret_name: str | None = None,
        model_secret_name: str | None = None,
        kis_account_no_masked: str | None = None,
        kis_env: str | None = None,
        has_openai: bool = False,
        has_gemini: bool = False,
        has_anthropic: bool = False,
        notes: str | None = None,
    ) -> None:
        tenant = str(tenant_id or "").strip().lower()
        if not tenant:
            raise ValueError("tenant_id is required")
        self.session.insert_dict(
            "runtime_credentials",
            {
                "tenant_id": tenant,
                "updated_at": updated_at or datetime.now(timezone.utc),
                "updated_by": str(updated_by or "").strip() or None,
                "kis_secret_name": str(kis_secret_name or "").strip() or None,
                "model_secret_name": str(model_secret_name or "").strip() or None,
                "kis_account_no_masked": str(kis_account_no_masked or "").strip() or None,
                "kis_env": str(kis_env or "").strip().lower() or None,
                "has_openai": bool(has_openai),
                "has_gemini": bool(has_gemini),
                "has_anthropic": bool(has_anthropic),
                "notes": str(notes or "").strip() or None,
            },
        )

    def latest_runtime_credentials(self, *, tenant_id: str) -> dict[str, Any] | None:
        tenant = str(tenant_id or "").strip().lower()
        if not tenant:
            return None
        rows = self.session.fetch_rows(
            """
            SELECT tenant_id, updated_at, updated_by, kis_secret_name, model_secret_name,
                   kis_account_no_masked, kis_env, has_openai, has_gemini, has_anthropic, notes
            FROM runtime_credentials
            WHERE tenant_id = $tenant_id
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            {"tenant_id": tenant},
        )
        return rows[0] if rows else None

    def recent_runtime_credentials(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self.session.fetch_rows(
            """
            SELECT tenant_id, updated_at, updated_by, kis_secret_name, model_secret_name,
                   kis_account_no_masked, kis_env, has_openai, has_gemini, has_anthropic, notes
            FROM runtime_credentials
            ORDER BY updated_at DESC
            LIMIT $limit
            """,
            {"limit": max(1, min(int(limit), 200))},
        )
