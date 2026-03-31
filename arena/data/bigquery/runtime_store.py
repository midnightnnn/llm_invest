"""Runtime credential and admin store — config, credentials, audit, access management."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

from arena.models import utc_now

if TYPE_CHECKING:
    from arena.data.bigquery.session import BigQuerySession


class RuntimeStore:
    """Reads and writes runtime credential/admin data via a shared session."""

    def __init__(self, session: BigQuerySession) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): RuntimeStore._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [RuntimeStore._json_safe(item) for item in value]
        return value

    @staticmethod
    def _norm_config_key(config_key: str) -> str:
        """Normalizes config key token for stable DB lookups."""
        return str(config_key or "").strip().lower()

    # ------------------------------------------------------------------
    # Runtime credentials
    # ------------------------------------------------------------------

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
        """Appends one runtime credential metadata row."""
        tenant = str(tenant_id or "").strip()
        if not tenant:
            raise ValueError("tenant_id is required")

        ts = updated_at or utc_now()
        table_id = f"{self.session.dataset_fqn}.runtime_credentials"
        row = {
            "tenant_id": tenant,
            "updated_at": ts.isoformat(),
            "updated_by": str(updated_by or "").strip() or None,
            "kis_secret_name": str(kis_secret_name or "").strip() or None,
            "model_secret_name": str(model_secret_name or "").strip() or None,
            "kis_account_no_masked": str(kis_account_no_masked or "").strip() or None,
            "kis_env": str(kis_env or "").strip().lower() or None,
            "has_openai": bool(has_openai),
            "has_gemini": bool(has_gemini),
            "has_anthropic": bool(has_anthropic),
            "notes": str(notes or "").strip() or None,
        }
        errors = self.session.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"runtime_credentials insert failed: {errors}")

    def latest_runtime_credentials(self, *, tenant_id: str) -> dict[str, Any] | None:
        """Returns latest runtime credential metadata for one tenant."""
        tenant = str(tenant_id or "").strip()
        if not tenant:
            return None
        sql = f"""
        SELECT tenant_id, updated_at, updated_by, kis_secret_name, model_secret_name,
               kis_account_no_masked, kis_env, has_openai, has_gemini, has_anthropic, notes
        FROM `{self.session.dataset_fqn}.runtime_credentials`
        WHERE tenant_id = @tenant_id
        ORDER BY updated_at DESC
        LIMIT 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant})
        return rows[0] if rows else None

    def recent_runtime_credentials(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Returns latest runtime credential metadata rows across tenants."""
        lim = max(1, min(int(limit), 200))
        sql = f"""
        SELECT tenant_id, updated_at, updated_by, kis_secret_name, model_secret_name,
               kis_account_no_masked, kis_env, has_openai, has_gemini, has_anthropic, notes
        FROM `{self.session.dataset_fqn}.runtime_credentials`
        ORDER BY updated_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, {"limit": lim})

    # ------------------------------------------------------------------
    # Arena config
    # ------------------------------------------------------------------

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        """Returns latest config value for one tenant/config key."""
        tenant = str(tenant_id or "").strip().lower()
        key = self._norm_config_key(config_key)
        if not tenant or not key:
            return None

        sql = f"""
        SELECT config_value
        FROM `{self.session.dataset_fqn}.arena_config`
        WHERE tenant_id = @tenant_id
          AND config_key = @config_key
        ORDER BY updated_at DESC
        LIMIT 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant, "config_key": key})
        if not rows:
            return None
        value = rows[0].get("config_value")
        return None if value is None else str(value)

    def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]:
        """Returns latest config values for one tenant across many config keys."""
        tenant = str(tenant_id or "").strip().lower()
        keys = [self._norm_config_key(k) for k in (config_keys or []) if self._norm_config_key(k)]
        if not tenant or not keys:
            return {}

        sql = f"""
        SELECT config_key, config_value
        FROM (
          SELECT
            config_key,
            config_value,
            ROW_NUMBER() OVER (
              PARTITION BY config_key
              ORDER BY updated_at DESC
            ) AS rn
          FROM `{self.session.dataset_fqn}.arena_config`
          WHERE tenant_id = @tenant_id
            AND config_key IN UNNEST(@config_keys)
        )
        WHERE rn = 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant, "config_keys": keys})
        out: dict[str, str] = {}
        for row in rows:
            key = self._norm_config_key(str(row.get("config_key") or ""))
            if not key:
                continue
            out[key] = str(row.get("config_value") or "")
        return out

    def latest_config_values(self, *, config_key: str, tenant_ids: list[str] | None = None) -> dict[str, str]:
        """Returns latest config value per tenant for one config key."""
        key = self._norm_config_key(config_key)
        if not key:
            return {}

        clean_tenants = [
            str(token or "").strip().lower()
            for token in (tenant_ids or [])
            if str(token or "").strip()
        ]
        clean_tenants = list(dict.fromkeys(clean_tenants))

        conditions = ["config_key = @config_key"]
        params: dict[str, Any] = {"config_key": key}
        if clean_tenants:
            conditions.append("tenant_id IN UNNEST(@tenant_ids)")
            params["tenant_ids"] = clean_tenants

        sql = f"""
        SELECT tenant_id, config_value
        FROM (
          SELECT
            tenant_id,
            config_value,
            ROW_NUMBER() OVER (
              PARTITION BY tenant_id
              ORDER BY updated_at DESC
            ) AS rn
          FROM `{self.session.dataset_fqn}.arena_config`
          WHERE {' AND '.join(conditions)}
        )
        WHERE rn = 1
        """
        rows = self.session.fetch_rows(sql, params)
        out: dict[str, str] = {}
        for row in rows:
            tenant = str(row.get("tenant_id") or "").strip().lower()
            if not tenant:
                continue
            out[tenant] = str(row.get("config_value") or "")
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
        """Writes one tenant config row (append-only, latest row wins)."""
        tenant = str(tenant_id or "").strip().lower()
        key = self._norm_config_key(config_key)
        if not tenant:
            raise ValueError("tenant_id is required")
        if not key:
            raise ValueError("config_key is required")

        ts = updated_at or utc_now()
        row = {
            "tenant_id": tenant,
            "config_key": key,
            "config_value": str(value or ""),
            "updated_at": ts.isoformat(),
            "updated_by": str(updated_by or "").strip() or None,
        }
        table_id = f"{self.session.dataset_fqn}.arena_config"
        errors = self.session.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"arena_config insert failed: {errors}")

    # ------------------------------------------------------------------
    # User-tenant mappings
    # ------------------------------------------------------------------

    def ensure_runtime_user_tenant(
        self,
        *,
        user_email: str,
        tenant_id: str,
        role: str = "owner",
        created_by: str | None = None,
    ) -> None:
        """Creates user-tenant mapping if missing."""
        user = str(user_email or "").strip().lower()
        tenant = str(tenant_id or "").strip().lower()
        if not user or not tenant:
            return

        sql = f"""
        SELECT COUNT(1) AS cnt
        FROM `{self.session.dataset_fqn}.runtime_user_tenants`
        WHERE user_email = @user_email AND tenant_id = @tenant_id
        """
        rows = self.session.fetch_rows(sql, {"user_email": user, "tenant_id": tenant})
        if int(rows[0].get("cnt") or 0) > 0:
            return

        table_id = f"{self.session.dataset_fqn}.runtime_user_tenants"
        row = {
            "user_email": user,
            "tenant_id": tenant,
            "role": str(role or "owner").strip().lower() or "owner",
            "created_at": utc_now().isoformat(),
            "created_by": str(created_by or "").strip() or user,
        }
        errors = self.session.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"runtime_user_tenants insert failed: {errors}")

    def list_runtime_user_tenants(self, *, user_email: str) -> list[dict[str, Any]]:
        """Lists tenants accessible by user."""
        user = str(user_email or "").strip().lower()
        if not user:
            return []
        sql = f"""
        SELECT user_email, tenant_id, role, created_at, created_by
        FROM `{self.session.dataset_fqn}.runtime_user_tenants`
        WHERE user_email = @user_email
        ORDER BY created_at ASC
        """
        return self.session.fetch_rows(sql, {"user_email": user})

    def list_runtime_tenants(self, *, limit: int = 200) -> list[str]:
        """Lists tenant ids that have runtime credentials registered (batch-ready only)."""
        lim = max(1, min(int(limit), 2000))
        sql = f"""
        SELECT DISTINCT tenant_id
        FROM `{self.session.dataset_fqn}.runtime_credentials`
        WHERE tenant_id IS NOT NULL
          AND TRIM(tenant_id) != ''
        ORDER BY tenant_id ASC
        LIMIT @limit
        """
        rows = self.session.fetch_rows(sql, {"limit": lim})
        out: list[str] = []
        for row in rows:
            token = str(row.get("tenant_id") or "").strip().lower()
            if token and token not in out:
                out.append(token)
        return out

    def has_runtime_user_tenant(self, *, user_email: str, tenant_id: str) -> bool:
        """Returns True if user has access to tenant."""
        user = str(user_email or "").strip().lower()
        tenant = str(tenant_id or "").strip().lower()
        if not user or not tenant:
            return False
        sql = f"""
        SELECT COUNT(1) AS cnt
        FROM `{self.session.dataset_fqn}.runtime_user_tenants`
        WHERE user_email = @user_email AND tenant_id = @tenant_id
        """
        rows = self.session.fetch_rows(sql, {"user_email": user, "tenant_id": tenant})
        return int(rows[0].get("cnt") or 0) > 0

    # ------------------------------------------------------------------
    # Access requests
    # ------------------------------------------------------------------

    def append_runtime_access_request(
        self,
        *,
        user_email: str,
        user_name: str | None = None,
        google_sub: str | None = None,
        status: str = "pending",
        note: str | None = None,
        decided_by: str | None = None,
        requested_at: datetime | None = None,
        decided_at: datetime | None = None,
    ) -> None:
        """Appends one UI access-request row."""
        user = str(user_email or "").strip().lower()
        if not user:
            raise ValueError("user_email is required")

        ts = requested_at or utc_now()
        row = {
            "user_email": user,
            "user_name": str(user_name or "").strip() or None,
            "google_sub": str(google_sub or "").strip() or None,
            "requested_at": ts.isoformat(),
            "status": str(status or "pending").strip().lower() or "pending",
            "decided_at": decided_at.isoformat() if isinstance(decided_at, datetime) else None,
            "decided_by": str(decided_by or "").strip().lower() or None,
            "note": str(note or "").strip() or None,
        }
        table_id = f"{self.session.dataset_fqn}.runtime_access_requests"
        errors = self.session.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"runtime_access_requests insert failed: {errors}")

    def latest_runtime_access_request(self, *, user_email: str) -> dict[str, Any] | None:
        """Returns the latest UI access request row for one email."""
        user = str(user_email or "").strip().lower()
        if not user:
            return None
        sql = f"""
        SELECT user_email, user_name, google_sub, requested_at, status, decided_at, decided_by, note
        FROM `{self.session.dataset_fqn}.runtime_access_requests`
        WHERE user_email = @user_email
        ORDER BY requested_at DESC
        LIMIT 1
        """
        rows = self.session.fetch_rows(sql, {"user_email": user})
        return rows[0] if rows else None

    def ensure_runtime_access_request_pending(
        self,
        *,
        user_email: str,
        user_name: str | None = None,
        google_sub: str | None = None,
    ) -> dict[str, Any] | None:
        """Creates a pending access request if the latest row is not already pending."""
        latest = self.latest_runtime_access_request(user_email=user_email)
        latest_status = str((latest or {}).get("status") or "").strip().lower()
        if latest and latest_status == "pending":
            return latest
        self.append_runtime_access_request(
            user_email=user_email,
            user_name=user_name,
            google_sub=google_sub,
            status="pending",
        )
        return self.latest_runtime_access_request(user_email=user_email)

    # ------------------------------------------------------------------
    # Audit logs
    # ------------------------------------------------------------------

    def append_runtime_audit_log(
        self,
        *,
        action: str,
        status: str,
        user_email: str | None = None,
        tenant_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Appends one audit log row for runtime admin actions."""
        table_id = f"{self.session.dataset_fqn}.runtime_audit_logs"
        row = {
            "created_at": utc_now().isoformat(),
            "user_email": str(user_email or "").strip().lower() or None,
            "tenant_id": str(tenant_id or "").strip().lower() or None,
            "action": str(action or "").strip() or "unknown",
            "status": str(status or "").strip().lower() or "unknown",
            "detail_json": json.dumps(detail or {}, ensure_ascii=False),
        }
        errors = self.session.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"runtime_audit_logs insert failed: {errors}")

    def recent_runtime_audit_logs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Returns recent audit log rows across all tenants."""
        lim = max(1, min(int(limit), 500))
        sql = f"""
        SELECT created_at, user_email, tenant_id, action, status, detail_json
        FROM `{self.session.dataset_fqn}.runtime_audit_logs`
        ORDER BY created_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, {"limit": lim})

    # ------------------------------------------------------------------
    # Migration states
    # ------------------------------------------------------------------

    def append_runtime_migration_state(
        self,
        *,
        tenant_id: str,
        migration_key: str,
        run_id: str,
        stage: str,
        status: str,
        trading_mode: str | None = None,
        updated_by: str | None = None,
        recorded_at: datetime | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Appends one migration state row for preview/apply/verify operations."""
        tenant = str(tenant_id or "").strip().lower()
        key = str(migration_key or "").strip().lower()
        token = str(run_id or "").strip()
        if not tenant:
            raise ValueError("tenant_id is required")
        if not key:
            raise ValueError("migration_key is required")
        if not token:
            raise ValueError("run_id is required")
        row = {
            "tenant_id": tenant,
            "migration_key": key,
            "run_id": token,
            "recorded_at": (recorded_at or utc_now()).isoformat(),
            "trading_mode": str(trading_mode or "").strip().lower() or None,
            "stage": str(stage or "").strip().lower() or "unknown",
            "status": str(status or "").strip().lower() or "unknown",
            "updated_by": str(updated_by or "").strip() or None,
            "detail_json": json.dumps(self._json_safe(detail or {}), ensure_ascii=False, separators=(",", ":")),
        }
        table_id = f"{self.session.dataset_fqn}.runtime_migration_states"
        errors = self.session.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"runtime_migration_states insert failed: {errors}")

    def latest_runtime_migration_states(
        self,
        *,
        tenant_id: str,
        limit: int = 20,
        migration_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns recent migration state rows for a tenant."""
        tenant = str(tenant_id or "").strip().lower()
        if not tenant:
            return []
        lim = max(1, min(int(limit), 200))
        filters = ["tenant_id = @tenant_id"]
        params: dict[str, Any] = {"tenant_id": tenant, "limit": lim}
        key = str(migration_key or "").strip().lower()
        if key:
            filters.append("migration_key = @migration_key")
            params["migration_key"] = key
        where = " AND ".join(filters)
        sql = f"""
        SELECT tenant_id, migration_key, run_id, recorded_at, trading_mode, stage, status, updated_by, detail_json
        FROM `{self.session.dataset_fqn}.runtime_migration_states`
        WHERE {where}
        ORDER BY recorded_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    # ------------------------------------------------------------------
    # Tenant run statuses
    # ------------------------------------------------------------------

    def append_tenant_run_status(
        self,
        *,
        tenant_id: str,
        run_id: str,
        run_type: str,
        status: str,
        reason_code: str | None = None,
        stage: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        recorded_at: datetime | None = None,
        message: str | None = None,
        job_name: str | None = None,
        execution_name: str | None = None,
        log_uri: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Appends one tenant-scoped run status row."""
        tenant = str(tenant_id or "").strip().lower()
        if not tenant:
            raise ValueError("tenant_id is required")
        token = str(run_id or "").strip()
        if not token:
            raise ValueError("run_id is required")
        row = {
            "tenant_id": tenant,
            "run_id": token,
            "recorded_at": (recorded_at or utc_now()).isoformat(),
            "run_type": str(run_type or "").strip().lower() or "unknown",
            "status": str(status or "").strip().lower() or "unknown",
            "reason_code": str(reason_code or "").strip().lower() or None,
            "stage": str(stage or "").strip().lower() or None,
            "started_at": started_at.isoformat() if isinstance(started_at, datetime) else None,
            "finished_at": finished_at.isoformat() if isinstance(finished_at, datetime) else None,
            "message": str(message or "").strip() or None,
            "job_name": str(job_name or "").strip() or None,
            "execution_name": str(execution_name or "").strip() or None,
            "log_uri": str(log_uri or "").strip() or None,
            "detail_json": json.dumps(detail or {}, ensure_ascii=False, separators=(",", ":")),
        }
        table_id = f"{self.session.dataset_fqn}.tenant_run_statuses"
        errors = self.session.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"tenant_run_statuses insert failed: {errors}")

    def all_tenant_run_statuses(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Returns the latest run status per tenant across ALL tenants."""
        lim = max(1, min(int(limit), 500))
        sql = f"""
        SELECT tenant_id, run_id, recorded_at, run_type, status, reason_code, stage,
               started_at, finished_at, message, job_name, execution_name, log_uri, detail_json
        FROM (
          SELECT *, ROW_NUMBER() OVER (PARTITION BY tenant_id ORDER BY recorded_at DESC) AS rn
          FROM `{self.session.dataset_fqn}.tenant_run_statuses`
          WHERE tenant_id IS NOT NULL AND TRIM(tenant_id) != ''
        )
        WHERE rn = 1
        ORDER BY recorded_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, {"limit": lim})

    def latest_tenant_run_status(
        self,
        *,
        tenant_id: str,
        run_type: str | None = None,
        exclude_statuses: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Returns the latest run status row for a tenant."""
        tenant = str(tenant_id or "").strip().lower()
        if not tenant:
            return None
        filter_sql = ""
        params: dict[str, Any] = {"tenant_id": tenant}
        token = str(run_type or "").strip().lower()
        if token:
            filter_sql = "AND run_type = @run_type"
            params["run_type"] = token
        if exclude_statuses:
            filter_sql += " AND status NOT IN UNNEST(@exclude_statuses)"
            params["exclude_statuses"] = [str(s).strip().lower() for s in exclude_statuses]
        sql = f"""
        SELECT tenant_id, run_id, recorded_at, run_type, status, reason_code, stage,
               started_at, finished_at, message, job_name, execution_name, log_uri, detail_json
        FROM `{self.session.dataset_fqn}.tenant_run_statuses`
        WHERE tenant_id = @tenant_id
          {filter_sql}
        ORDER BY recorded_at DESC
        LIMIT 1
        """
        rows = self.session.fetch_rows(sql, params)
        return rows[0] if rows else None
