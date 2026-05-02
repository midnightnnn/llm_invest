from __future__ import annotations

import hashlib
import json
from typing import Any

from arena.agents.investment_chat.audit import config_get, config_set


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def approval_token(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json_dumps(payload).encode("utf-8")).hexdigest()[:24]


def draft_key(token: str) -> str:
    return f"chat_order_draft.{str(token or '').strip()}"


def load_draft(repo: Any, *, tenant_id: str, token: str) -> dict[str, Any] | None:
    raw = config_get(repo, tenant_id, draft_key(token))
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def save_draft(
    repo: Any,
    *,
    tenant_id: str,
    token: str,
    draft: dict[str, Any],
    updated_by: str = "investment_chat",
) -> None:
    config_set(repo, tenant_id, draft_key(token), json_dumps(draft), updated_by=updated_by)
