from __future__ import annotations

import logging
from typing import Any, Literal

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
from arena.logging_utils import failure_extra
from arena.open_trading.sync import AccountSyncService
from arena.providers.registry import canonical_provider, provider_alias_map
from arena.tools.registry import ToolEntry

logger = logging.getLogger(__name__)

AccountSnapshotSource = Literal["latest", "db", "stored"]

_PROVIDER_MODEL_ALIASES = {
    "gpt": {"chatgpt", "gpt4", "gpt4o", "gpt-4", "gpt-4o", "openai"},
    "gemini": {"gemini", "google", "gemini_2_0_flash_exp", "gemini-2.0-flash-exp"},
    "claude": {"anthropic", "claude", "claude_3_7_sonnet", "claude-3-7-sonnet", "sonnet", "opus"},
}


def _compact_agent_alias(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "").replace("_", "").replace(".", "")


def _available_agent_ids(settings: Settings) -> list[str]:
    tokens = [
        str(agent_id or "").strip().lower()
        for agent_id in (getattr(settings, "agent_ids", []) or [])
        if str(agent_id or "").strip()
    ]
    if not tokens:
        tokens = [
            str(agent_id or "").strip().lower()
            for agent_id in (getattr(settings, "agent_configs", {}) or {}).keys()
            if str(agent_id or "").strip()
        ]
    return list(dict.fromkeys(tokens))


def _agent_provider(settings: Settings, agent_id: str) -> str:
    agent = str(agent_id or "").strip().lower()
    config = (getattr(settings, "agent_configs", {}) or {}).get(agent)
    configured = canonical_provider(getattr(config, "provider", "") if config is not None else "")
    return configured or canonical_provider(agent)


def _agent_model(settings: Settings, agent_id: str) -> str:
    agent = str(agent_id or "").strip().lower()
    config = (getattr(settings, "agent_configs", {}) or {}).get(agent)
    return str(getattr(config, "model", "") if config is not None else "").strip().lower()


def _single_agent_for_provider(settings: Settings, provider: str, available: list[str]) -> str:
    provider_token = canonical_provider(provider)
    if not provider_token:
        return ""
    matches = [agent_id for agent_id in available if _agent_provider(settings, agent_id) == provider_token]
    return matches[0] if len(matches) == 1 else ""


def _resolve_available_agent_id(settings: Settings, requested_agent_id: str) -> tuple[str, list[str]]:
    available = _available_agent_ids(settings)
    requested = str(requested_agent_id or "").strip().lower()
    if not requested:
        return "", available
    if requested in available:
        return requested, available

    provider_token = canonical_provider(requested)
    if provider_token:
        provider_agent = _single_agent_for_provider(settings, provider_token, available)
        if provider_agent:
            return provider_agent, available

    alias_provider = provider_alias_map().get(requested) or ""
    if alias_provider:
        provider_agent = _single_agent_for_provider(settings, alias_provider, available)
        if provider_agent:
            return provider_agent, available

    compact_requested = _compact_agent_alias(requested)
    for provider, aliases in _PROVIDER_MODEL_ALIASES.items():
        compact_aliases = {_compact_agent_alias(alias) for alias in aliases}
        provider_prefixes = {
            "gpt": ("gpt", "openai"),
            "gemini": ("gemini", "google"),
            "claude": ("claude", "anthropic", "sonnet", "opus"),
        }.get(provider, ())
        if (
            requested in aliases
            or compact_requested in compact_aliases
            or any(compact_requested.startswith(prefix) for prefix in provider_prefixes)
            or any(prefix in compact_requested for prefix in ("sonnet", "opus") if provider == "claude")
        ):
            provider_agent = _single_agent_for_provider(settings, provider, available)
            if provider_agent:
                return provider_agent, available

    for agent_id in available:
        model = _agent_model(settings, agent_id)
        if requested == model or compact_requested == _compact_agent_alias(model):
            return agent_id, available
        short_model = model.split("/", 1)[-1] if "/" in model else model
        if requested == short_model or compact_requested == _compact_agent_alias(short_model):
            return agent_id, available
    return "", available


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


def _account_scope_payload(repo: Any, *, tenant_id: str) -> dict[str, str]:
    return {
        "scope": "account",
        "market_scope": _account_market_override(repo, tenant_id=tenant_id),
    }


def build_account_tool_entries(*, repo: Any, settings: Settings, tenant_id: str) -> list[ToolEntry]:
    tenant = normalize_tenant(tenant_id)
    available_agent_ids = _available_agent_ids(settings)

    def get_account_snapshot(source: AccountSnapshotSource = "latest", max_positions: int = 50) -> dict[str, Any]:
        """Reads the latest persisted total account snapshot for the current Arena tenant."""
        source_token = str(source or "latest").strip().lower()
        if source_token not in {"latest", "db", "stored"}:
            return {
                "status": "blocked",
                "error": "Only persisted account snapshots are enabled in chat. Run account sync outside chat, then retry with source='latest'.",
                "requested_source": source_token,
                "available_agent_ids": available_agent_ids,
            }
        market_scope = _account_market_override(repo, tenant_id=tenant)
        snapshot = latest_account_snapshot(repo, tenant_id=tenant, market_scope=market_scope)
        if snapshot is None:
            return {
                "status": "missing",
                "tenant_id": tenant,
                **_account_scope_payload(repo, tenant_id=tenant),
                "error": "No account snapshot is stored for this tenant.",
                "available_agent_ids": available_agent_ids,
            }
        payload = snapshot_payload(snapshot, tenant_id=tenant, max_positions=max_positions)
        payload.update(_account_scope_payload(repo, tenant_id=tenant))
        payload["available_agent_ids"] = available_agent_ids
        return payload

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
                **_account_scope_payload(repo, tenant_id=tenant),
                "error": "This tenant KIS credentials are not configured. Refusing to use server fallback credentials for account refresh.",
            }
        try:
            account_settings = _account_sync_settings(repo, tenant_id=tenant, settings=settings)
            with repo_tenant_scope(repo, tenant):
                snapshot = AccountSyncService(settings=account_settings, repo=repo).sync_account_snapshot()
        except Exception as exc:
            logger.warning(
                "[yellow]Investment chat account refresh failed[/yellow] tenant=%s err=%s",
                tenant,
                str(exc),
                extra=failure_extra(
                    "chat_account_refresh_failed",
                    exc,
                    tenant_id=tenant,
                ),
                exc_info=True,
            )
            append_chat_audit(
                repo,
                tenant_id=tenant,
                action="chat_account_refresh",
                status="error",
                detail={"error": str(exc)[:500]},
            )
            return {
                "status": "error",
                "tenant_id": tenant,
                **_account_scope_payload(repo, tenant_id=tenant),
                "error": str(exc),
            }
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
        payload = snapshot_payload(snapshot, tenant_id=tenant, max_positions=max_positions)
        payload.update(_account_scope_payload(repo, tenant_id=tenant))
        return payload

    def get_agent_sleeve_snapshot(agent_id: str, max_positions: int = 50) -> dict[str, Any]:
        """Reads one batch agent sleeve snapshot so chat advice can distinguish total account vs sleeve scope."""
        requested_agent = str(agent_id or "").strip().lower()
        agent, current_available_agent_ids = _resolve_available_agent_id(settings, requested_agent)
        if not requested_agent:
            return {
                "status": "error",
                "error": "agent_id is required",
                "available_agent_ids": current_available_agent_ids,
            }
        if not agent:
            return {
                "status": "blocked",
                "requested_agent_id": requested_agent,
                "available_agent_ids": current_available_agent_ids,
                "error": "agent_id must name one of the configured batch agents.",
            }
        builder = getattr(repo, "build_agent_sleeve_snapshot", None)
        if not callable(builder):
            return {
                "status": "unavailable",
                "requested_agent_id": requested_agent,
                "available_agent_ids": current_available_agent_ids,
                "error": "sleeve snapshot reader is unavailable",
            }
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
                "requested_agent_id": requested_agent,
                "available_agent_ids": current_available_agent_ids,
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
            description=(
                "Reads one configured batch agent sleeve snapshot. agent_id should be one of "
                "available_agent_ids from get_account_snapshot; provider/model aliases like "
                "gpt4o, gemini_2_0_flash_exp, or claude_3_7_sonnet are normalized only when "
                "they match one configured agent unambiguously."
            ),
            category="account",
            callable=get_agent_sleeve_snapshot,
            tier="core",
            label_ko="에이전트 슬리브",
            sort_order=3,
        ),
    ]
