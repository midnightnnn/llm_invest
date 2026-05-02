from __future__ import annotations

import logging
from typing import Any

from arena.agents.investment_chat.context import REQUEST_USER_EMAIL

logger = logging.getLogger(__name__)


def config_get(repo: Any, tenant_id: str, key: str) -> str | None:
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return None
    try:
        return getter(tenant_id, key)
    except TypeError:
        return getter(tenant_id=tenant_id, config_key=key)


def config_set(
    repo: Any,
    tenant_id: str,
    key: str,
    value: str,
    *,
    updated_by: str = "investment_chat",
) -> None:
    setter = getattr(repo, "set_config", None)
    if not callable(setter):
        return
    try:
        setter(tenant_id, key, value, updated_by)
    except TypeError:
        setter(tenant_id=tenant_id, config_key=key, value=value, updated_by=updated_by)


def append_chat_audit(
    repo: Any,
    *,
    tenant_id: str,
    action: str,
    status: str,
    detail: dict[str, Any],
    user_email: str = "",
) -> None:
    append = getattr(repo, "append_runtime_audit_log", None)
    if not callable(append):
        return
    actor_email = str(user_email or REQUEST_USER_EMAIL.get() or "").strip().lower() or None
    try:
        append(
            action=action,
            status=status,
            tenant_id=tenant_id,
            user_email=actor_email,
            detail=detail,
        )
    except Exception:
        logger.warning("[yellow]chat audit append failed[/yellow] action=%s tenant=%s", action, tenant_id)
