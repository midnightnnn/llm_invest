from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any

from arena.ui.access import access_rows_for_user, default_tenant_for_email, safe_tenant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProvisionedUserAccess:
    access_rows: list[dict[str, str]]
    tenant_id: str
    created_tenant: bool = False
    blocked_status: str = ""
    blocked_note: str = ""


class UIProvisioner:
    """Provision a per-user tenant on first successful UI login."""

    def __init__(self, *, repo: Any) -> None:
        self.repo = repo

    def _latest_access_request(self, user_email: str) -> dict[str, Any] | None:
        loader = getattr(self.repo, "latest_runtime_access_request", None)
        if not callable(loader):
            return None
        try:
            return loader(user_email=str(user_email or "").strip().lower())
        except Exception:
            return None

    def _tenant_exists(self, tenant_id: str) -> bool:
        tenant = str(tenant_id or "").strip().lower()
        if not tenant:
            return False
        fetch_rows = getattr(self.repo, "fetch_rows", None)
        dataset_fqn = str(getattr(self.repo, "dataset_fqn", "") or "").strip()
        if not callable(fetch_rows) or not dataset_fqn:
            return False
        sql = f"""
        SELECT tenant_id
        FROM (
          SELECT tenant_id
          FROM `{dataset_fqn}.runtime_user_tenants`
          WHERE tenant_id = @tenant_id
          UNION ALL
          SELECT tenant_id
          FROM `{dataset_fqn}.runtime_credentials`
          WHERE tenant_id = @tenant_id
          UNION ALL
          SELECT tenant_id
          FROM `{dataset_fqn}.arena_config`
          WHERE tenant_id = @tenant_id
        )
        LIMIT 1
        """
        try:
            return bool(fetch_rows(sql, {"tenant_id": tenant}))
        except Exception:
            return False

    def _choose_tenant_id(self, user_email: str) -> str:
        email = str(user_email or "").strip().lower()
        base = default_tenant_for_email(email)
        candidates: list[str] = [base]
        if "@" in email:
            domain = safe_tenant(email.split("@", 1)[1])
            if domain and domain != base:
                candidates.append(f"{base}-{domain}")
        suffix = hashlib.sha1(email.encode("utf-8")).hexdigest()[:6]
        candidates.append(f"{base}-{suffix}")
        for candidate in candidates:
            if not self._tenant_exists(candidate):
                return candidate
        for idx in range(2, 1000):
            candidate = f"{base}-{idx}"
            if not self._tenant_exists(candidate):
                return candidate
        return f"{base}-{suffix}"

    def _public_demo_tenant_id(self) -> str:
        raw = str(os.getenv("ARENA_PUBLIC_DEMO_TENANT", "") or "").strip().lower()
        return safe_tenant(raw) if raw else ""

    def _ensure_public_demo_view_access(self, *, user_email: str, created_by: str) -> None:
        demo_tenant = self._public_demo_tenant_id()
        if not demo_tenant or not self._tenant_exists(demo_tenant):
            return
        rows = access_rows_for_user(self.repo, user_email)
        if any(str(row.get("tenant_id") or "").strip().lower() == demo_tenant for row in rows):
            return
        ensure_mapping = getattr(self.repo, "ensure_runtime_user_tenant", None)
        if not callable(ensure_mapping):
            return
        ensure_mapping(
            user_email=user_email,
            tenant_id=demo_tenant,
            role="viewer",
            created_by=created_by,
        )

    def _set_config_if_missing(self, tenant_id: str, config_key: str, value: str, updated_by: str) -> bool:
        getter = getattr(self.repo, "get_config", None)
        setter = getattr(self.repo, "set_config", None)
        if not callable(setter):
            return False
        existing = None
        if callable(getter):
            try:
                existing = getter(tenant_id, config_key)
            except Exception:
                existing = None
        if str(existing or "").strip():
            return False
        setter(tenant_id, config_key, value, updated_by)
        return True

    def _seed_default_tenant_config(self, tenant_id: str, updated_by: str) -> None:
        self._set_config_if_missing(tenant_id, "distribution_mode", "simulated_only", updated_by)
        self._set_config_if_missing(tenant_id, "real_trading_approved", "false", updated_by)

    def ensure_user_access(self, user: dict[str, Any] | None) -> ProvisionedUserAccess:
        if not isinstance(user, dict):
            return ProvisionedUserAccess(access_rows=[], tenant_id="")
        user_email = str(user.get("email") or "").strip().lower()
        if not user_email:
            return ProvisionedUserAccess(access_rows=[], tenant_id="")

        rows = access_rows_for_user(self.repo, user_email)
        if rows:
            self._ensure_public_demo_view_access(user_email=user_email, created_by=user_email)
            rows = access_rows_for_user(self.repo, user_email)
            return ProvisionedUserAccess(
                access_rows=rows,
                tenant_id=str(rows[0].get("tenant_id") or "").strip().lower(),
            )

        latest_request = self._latest_access_request(user_email)
        latest_status = str((latest_request or {}).get("status") or "").strip().lower()
        if latest_status == "rejected":
            return ProvisionedUserAccess(
                access_rows=[],
                tenant_id="",
                blocked_status="rejected",
                blocked_note=str((latest_request or {}).get("note") or "").strip(),
            )

        tenant_id = self._choose_tenant_id(user_email)
        creator = user_email
        ensure_mapping = getattr(self.repo, "ensure_runtime_user_tenant", None)
        if not callable(ensure_mapping):
            return ProvisionedUserAccess(access_rows=[], tenant_id="")

        ensure_mapping(
            user_email=user_email,
            tenant_id=tenant_id,
            role="owner",
            created_by=creator,
        )
        self._seed_default_tenant_config(tenant_id, creator)
        self._ensure_public_demo_view_access(user_email=user_email, created_by=creator)

        rows = access_rows_for_user(self.repo, user_email)
        if not rows:
            rows = [{"tenant_id": tenant_id, "role": "owner"}]

        append_audit = getattr(self.repo, "append_runtime_audit_log", None)
        if callable(append_audit):
            try:
                append_audit(
                    action="auth_auto_provision_tenant",
                    status="ok",
                    user_email=user_email,
                    tenant_id=tenant_id,
                    detail={
                        "distribution_mode": "simulated_only",
                        "owner_email": user_email,
                    },
                )
            except Exception:
                logger.warning("auth_auto_provision_tenant audit skipped user=%s tenant=%s", user_email, tenant_id)

        return ProvisionedUserAccess(
            access_rows=rows,
            tenant_id=tenant_id,
            created_tenant=True,
        )
