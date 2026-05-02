from __future__ import annotations

from typing import Any

from arena.agents.investment_chat.audit import append_chat_audit
from arena.agents.investment_chat.context import normalize_tenant
from arena.agents.investment_chat.market_scope import account_market_override, account_scope_settings
from arena.agents.investment_chat.utils import (
    latest_account_snapshot,
    repo_tenant_scope,
    safe_float,
    snapshot_payload,
    sources_for_settings,
)
from arena.config import Settings
from arena.open_trading.sync import AccountSyncService
from arena.tools.registry import ToolEntry


def _tenant_has_kis_credentials(repo: Any, *, tenant_id: str) -> bool:
    loader = getattr(repo, "latest_runtime_credentials", None)
    if not callable(loader):
        return True
    try:
        row = loader(tenant_id=tenant_id) or {}
    except Exception:
        return False
    return bool(str((row or {}).get("kis_secret_name") or "").strip())


def _account_market_override(repo: Any, *, tenant_id: str) -> str:
    return account_market_override(repo, tenant_id=tenant_id)


def _account_sync_settings(repo: Any, *, tenant_id: str, settings: Settings) -> Settings:
    return account_scope_settings(repo, tenant_id=tenant_id, settings=settings)


def build_account_tool_entries(*, repo: Any, settings: Settings, tenant_id: str) -> list[ToolEntry]:
    tenant = normalize_tenant(tenant_id)

    def get_account_snapshot(source: str = "latest", max_positions: int = 50) -> dict[str, Any]:
        """Reads the latest persisted total account snapshot for the current Arena tenant."""
        source_token = str(source or "latest").strip().lower()
        if source_token not in {"latest", "db", "stored"}:
            return {
                "status": "blocked",
                "error": "Only persisted account snapshots are enabled in chat. Run account sync outside chat, then retry with source='latest'.",
                "requested_source": source_token,
            }
        snapshot = latest_account_snapshot(repo, tenant_id=tenant)
        if snapshot is None:
            return {
                "status": "missing",
                "tenant_id": tenant,
                "error": "No account snapshot is stored for this tenant.",
            }
        return snapshot_payload(snapshot, tenant_id=tenant, max_positions=max_positions)

    def refresh_account_snapshot(max_positions: int = 50) -> dict[str, Any]:
        """Reads the broker account through the configured KIS account sync path and stores the latest snapshot."""
        if not _tenant_has_kis_credentials(repo, tenant_id=tenant):
            append_chat_audit(
                repo,
                tenant_id=tenant,
                action="chat_account_refresh",
                status="blocked",
                detail={"reason": "tenant_kis_credentials_missing"},
            )
            return {
                "status": "blocked",
                "tenant_id": tenant,
                "error": "This tenant KIS credentials are not configured. Refusing to use server fallback credentials for account refresh.",
            }
        try:
            account_settings = _account_sync_settings(repo, tenant_id=tenant, settings=settings)
            with repo_tenant_scope(repo, tenant):
                snapshot = AccountSyncService(settings=account_settings, repo=repo).sync_account_snapshot()
        except Exception as exc:
            append_chat_audit(
                repo,
                tenant_id=tenant,
                action="chat_account_refresh",
                status="error",
                detail={"error": str(exc)[:500]},
            )
            return {"status": "error", "tenant_id": tenant, "error": str(exc)}
        append_chat_audit(
            repo,
            tenant_id=tenant,
            action="chat_account_refresh",
            status="ok",
            detail={
                "position_count": len(getattr(snapshot, "positions", {}) or {}),
                "target_market": str(getattr(account_settings, "kis_target_market", "") or ""),
            },
        )
        return snapshot_payload(snapshot, tenant_id=tenant, max_positions=max_positions)

    def get_agent_sleeve_snapshot(agent_id: str, max_positions: int = 50) -> dict[str, Any]:
        """Reads one batch agent sleeve snapshot so chat advice can distinguish total account vs sleeve scope."""
        agent = str(agent_id or "").strip().lower()
        if not agent:
            return {"status": "error", "error": "agent_id is required"}
        builder = getattr(repo, "build_agent_sleeve_snapshot", None)
        if not callable(builder):
            return {"status": "unavailable", "error": "sleeve snapshot reader is unavailable"}
        try:
            snapshot, baseline_equity_krw, meta = builder(
                agent_id=agent,
                sources=sources_for_settings(settings),
                include_simulated=True,
                tenant_id=tenant,
            )
        except TypeError:
            snapshot, baseline_equity_krw, meta = builder(
                agent_id=agent,
                sources=sources_for_settings(settings),
                include_simulated=True,
            )
        payload = snapshot_payload(snapshot, tenant_id=tenant, max_positions=max_positions)
        payload.update(
            {
                "agent_id": agent,
                "scope": "agent_sleeve",
                "baseline_equity_krw": safe_float(baseline_equity_krw),
                "metadata": meta if isinstance(meta, dict) else {},
            }
        )
        return payload

    return [
        ToolEntry(
            tool_id="get_account_snapshot",
            name="get_account_snapshot",
            description="Reads the latest stored total account snapshot for this Arena tenant. No broker call and no execution side effect.",
            category="account",
            callable=get_account_snapshot,
            tier="core",
            label_ko="계좌 스냅샷",
            sort_order=1,
        ),
        ToolEntry(
            tool_id="refresh_account_snapshot",
            name="refresh_account_snapshot",
            description="Refreshes the account snapshot from the configured broker account and persists it. Read-only broker operation; no order execution.",
            category="account",
            callable=refresh_account_snapshot,
            tier="core",
            label_ko="계좌 즉시 조회",
            sort_order=2,
        ),
        ToolEntry(
            tool_id="get_agent_sleeve_snapshot",
            name="get_agent_sleeve_snapshot",
            description="Reads one batch agent sleeve snapshot. Use this to compare sleeve-level state against the total account.",
            category="account",
            callable=get_agent_sleeve_snapshot,
            tier="core",
            label_ko="에이전트 슬리브",
            sort_order=3,
        ),
    ]
