from __future__ import annotations

import re
from typing import Any


def build_operator_emails(raw_ops: str, *, auth_enabled: bool) -> set[str]:
    operators = {email.strip().lower() for email in str(raw_ops or "").split(",") if email.strip()}
    if not auth_enabled:
        operators.add("local@localhost")
    return operators


def is_operator(user: dict[str, Any] | None, operator_emails: set[str]) -> bool:
    if not user:
        return False
    return str(user.get("email") or "").strip().lower() in operator_emails


def safe_tenant(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9-]", "-", str(value or "").strip().lower())
    token = re.sub(r"-+", "-", token).strip("-")
    return token or "local"


def default_tenant_for_email(email: str) -> str:
    local = str(email or "").split("@", 1)[0]
    return safe_tenant(local or "local")


def access_rows_for_user(repo: Any, user_email: str) -> list[dict[str, str]]:
    user = str(user_email or "").strip().lower()
    if not user:
        return []
    loader = getattr(repo, "list_runtime_user_tenants", None)
    if not callable(loader):
        return []
    try:
        raw_rows = loader(user_email=user) or []
    except Exception:
        raw_rows = []
    rows: list[dict[str, str]] = []
    for raw in raw_rows:
        tenant = str((raw or {}).get("tenant_id") or "").strip().lower()
        if not tenant:
            continue
        role = str((raw or {}).get("role") or "viewer").strip().lower() or "viewer"
        rows.append({"tenant_id": tenant, "role": role})
    return rows


def tenant_list_for_roles(access_rows: list[dict[str, str]], allowed_roles: set[str]) -> list[str]:
    tenants: list[str] = []
    for row in access_rows:
        tenant = str((row or {}).get("tenant_id") or "").strip().lower()
        role = str((row or {}).get("role") or "viewer").strip().lower() or "viewer"
        if not tenant or role not in allowed_roles or tenant in tenants:
            continue
        tenants.append(tenant)
    return tenants
