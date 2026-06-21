from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable

from arena.agents.investment_chat.market_scope import account_market_override, account_snapshot_market_scope
from arena.agents.investment_chat.utils import repo_tenant_scope
from arena.config import Settings
from arena.logging_utils import failure_extra
from arena.open_trading import sync as open_trading_sync

logger = logging.getLogger(__name__)


def sync_account_snapshot_after_kis_save(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    settings_for_tenant: Callable[[str], Settings] | None = None,
    updated_by: str = "",
) -> dict[str, Any]:
    """Best-effort first account snapshot after KIS credentials are saved."""
    tenant = str(tenant_id or "").strip().lower()
    market_scope = account_market_override(repo, tenant_id=tenant)
    try:
        base_settings = settings_for_tenant(tenant) if callable(settings_for_tenant) else settings
        account_settings = deepcopy(base_settings)
        account_settings.kis_target_market = market_scope
        with repo_tenant_scope(repo, tenant):
            snapshot = open_trading_sync.AccountSyncService(settings=account_settings, repo=repo).sync_account_snapshot(
                market_scope=account_snapshot_market_scope()
            )
        detail = {
            "target_market": market_scope,
            "snapshot_market_scope": account_snapshot_market_scope(),
            "cash_krw": float(getattr(snapshot, "cash_krw", 0.0) or 0.0),
            "total_equity_krw": float(getattr(snapshot, "total_equity_krw", 0.0) or 0.0),
            "position_count": len(getattr(snapshot, "positions", {}) or {}),
        }
        audit = getattr(repo, "append_runtime_audit_log", None)
        if callable(audit):
            audit(
                action="kis_credentials_account_snapshot_sync",
                status="ok",
                user_email=updated_by or "system",
                tenant_id=tenant,
                detail=detail,
            )
        return {"status": "ok", **detail}
    except Exception as exc:
        logger.warning(
            "[yellow]Initial account snapshot sync after KIS save failed[/yellow] tenant=%s err=%s",
            tenant,
            str(exc),
            extra=failure_extra(
                "kis_credentials_account_snapshot_sync_failed",
                exc,
                tenant_id=tenant,
                target_market=market_scope,
            ),
            exc_info=True,
        )
        detail = {"target_market": market_scope, "error": str(exc)[:500]}
        audit = getattr(repo, "append_runtime_audit_log", None)
        if callable(audit):
            audit(
                action="kis_credentials_account_snapshot_sync",
                status="error",
                user_email=updated_by or "system",
                tenant_id=tenant,
                detail=detail,
            )
        return {"status": "error", **detail}
