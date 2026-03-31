from __future__ import annotations

import json
import logging

from arena.memory.policy import (
    MEMORY_POLICY_CONFIG_KEY,
    load_memory_policy,
    normalize_memory_policy,
    serialize_memory_policy,
)
from arena.memory.tuning import run_memory_forgetting_tuner

logger = logging.getLogger(__name__)


def _cli():
    import arena.cli as cli

    return cli


def cmd_approve_live_tenant(
    *,
    tenant_id: str,
    approved: bool,
    updated_by: str = "cli-admin",
    note: str = "",
) -> None:
    """Approves or revokes real KIS trading for one tenant."""
    cli = _cli()
    tenant = str(tenant_id or "").strip().lower()
    if not tenant:
        raise SystemExit("tenant_id is required")

    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    repo = cli._repo_or_exit(settings, tenant_id=tenant)
    repo.ensure_dataset()
    repo.ensure_tables()

    repo.set_config(
        tenant,
        "real_trading_approved",
        "true" if approved else "false",
        updated_by,
    )
    if str(note or "").strip():
        repo.set_config(
            tenant,
            "real_trading_approval_note",
            str(note).strip(),
            updated_by,
        )
    repo.append_runtime_audit_log(
        action="approve_live_tenant",
        status="ok",
        user_email=updated_by,
        tenant_id=tenant,
        detail={
            "approved": bool(approved),
            "note": str(note or "").strip() or None,
        },
    )
    logger.info(
        "[green]Live trading approval updated[/green] tenant=%s approved=%s note=%s",
        tenant,
        "true" if approved else "false",
        str(note or "").strip() or "-",
    )


def _admin_repo_for_tenant(*, tenant_id: str):
    cli = _cli()
    tenant = str(tenant_id or "").strip().lower()
    if not tenant:
        raise SystemExit("tenant_id is required")

    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    repo = cli._repo_or_exit(settings, tenant_id=tenant)
    repo.ensure_dataset()
    repo.ensure_tables()
    return tenant, repo


def _admin_runtime():
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    repo = cli._repo_or_exit(settings, tenant_id=cli._tenant_id() or "local")
    repo.ensure_dataset()
    repo.ensure_tables()
    return cli, settings, repo


def _resolve_runtime_tenants(repo, *, tenant_ids: list[str] | None = None) -> list[str]:
    explicit = [str(token or "").strip().lower() for token in (tenant_ids or []) if str(token or "").strip()]
    if explicit:
        return list(dict.fromkeys(explicit))
    return list(repo.list_runtime_tenants(limit=2000))


def _with_forgetting_shadow(policy: dict[str, object] | None) -> dict[str, object]:
    normalized = normalize_memory_policy(policy)
    forgetting = normalized.setdefault("forgetting", {})
    if not isinstance(forgetting, dict):
        forgetting = {}
        normalized["forgetting"] = forgetting
    forgetting["enabled"] = True
    forgetting["access_log_enabled"] = True
    tuning = forgetting.setdefault("tuning", {})
    if not isinstance(tuning, dict):
        tuning = {}
        forgetting["tuning"] = tuning
    tuning["enabled"] = True
    tuning["mode"] = "shadow"
    tuning["auto_promote_enabled"] = False
    tuning["auto_demote_enabled"] = False
    return normalize_memory_policy(normalized)


def _normalize_market_tokens(raw_market: object) -> list[str]:
    alias = {"kr": "kospi", "korea": "kospi"}
    allowed = {"us", "nasdaq", "nyse", "amex", "kospi", "kosdaq"}
    tokens: list[str] = []
    for token in str(raw_market or "").split(","):
        market = alias.get(str(token).strip().lower(), str(token).strip().lower())
        if not market or market not in allowed or market in tokens:
            continue
        tokens.append(market)
    if "us" in tokens:
        tokens = [token for token in tokens if token == "us" or token not in {"nasdaq", "nyse", "amex"}]
    return tokens


def _derive_market_from_agents_config(agents_config_raw: str) -> str:
    text = str(agents_config_raw or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except Exception:
        return ""
    if not isinstance(parsed, list):
        return ""
    tokens: list[str] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        for market in _normalize_market_tokens(entry.get("target_market")):
            if market not in tokens:
                tokens.append(market)
    return ",".join(tokens)


def cmd_promote_tenant_live(
    *,
    tenant_id: str,
    updated_by: str = "cli-admin",
    note: str = "",
) -> None:
    """Promote one tenant to live-capable private mode."""
    tenant, repo = _admin_repo_for_tenant(tenant_id=tenant_id)
    repo.set_config(tenant, "distribution_mode", "private", updated_by)
    repo.set_config(tenant, "real_trading_approved", "true", updated_by)
    if str(note or "").strip():
        repo.set_config(tenant, "real_trading_approval_note", str(note).strip(), updated_by)
    repo.append_runtime_audit_log(
        action="promote_tenant_live",
        status="ok",
        user_email=updated_by,
        tenant_id=tenant,
        detail={"distribution_mode": "private", "approved": True, "note": str(note or "").strip() or None},
    )
    logger.info(
        "[green]Tenant promoted to live[/green] tenant=%s distribution_mode=private approved=true note=%s",
        tenant,
        str(note or "").strip() or "-",
    )


def cmd_set_tenant_simulated(
    *,
    tenant_id: str,
    updated_by: str = "cli-admin",
    note: str = "",
) -> None:
    """Demote one tenant to simulated-only onboarding mode."""
    tenant, repo = _admin_repo_for_tenant(tenant_id=tenant_id)
    repo.set_config(tenant, "distribution_mode", "simulated_only", updated_by)
    repo.set_config(tenant, "real_trading_approved", "false", updated_by)
    repo.append_runtime_audit_log(
        action="set_tenant_simulated",
        status="ok",
        user_email=updated_by,
        tenant_id=tenant,
        detail={"distribution_mode": "simulated_only", "approved": False, "note": str(note or "").strip() or None},
    )
    logger.info(
        "[green]Tenant set to simulated-only[/green] tenant=%s distribution_mode=simulated_only note=%s",
        tenant,
        str(note or "").strip() or "-",
    )


def cmd_backfill_tenant_markets(
    *,
    tenant_ids: list[str] | None = None,
    updated_by: str = "cli-admin",
) -> None:
    """Backfills tenant-level kis_target_market from agents_config target_market entries."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    repo = cli._repo_or_exit(settings, tenant_id="local")
    repo.ensure_dataset()
    repo.ensure_tables()

    explicit = [str(token or "").strip().lower() for token in (tenant_ids or []) if str(token or "").strip()]
    if explicit:
        tenants = list(dict.fromkeys(explicit))
    else:
        tenants = list(repo.list_runtime_tenants(limit=2000))

    updated = 0
    skipped = 0
    for tenant in tenants:
        values = repo.get_configs(tenant, ["agents_config", "kis_target_market"])
        next_market = _derive_market_from_agents_config(str(values.get("agents_config") or ""))
        current_market = str(values.get("kis_target_market") or "").strip().lower()
        if not next_market or next_market == current_market:
            skipped += 1
            continue
        repo.set_config(tenant, "kis_target_market", next_market, updated_by)
        repo.append_runtime_audit_log(
            action="backfill_tenant_market",
            status="ok",
            user_email=updated_by,
            tenant_id=tenant,
            detail={"previous_market": current_market or None, "kis_target_market": next_market},
        )
        updated += 1
        logger.info(
            "[green]Tenant market backfilled[/green] tenant=%s kis_target_market=%s previous=%s",
            tenant,
            next_market,
            current_market or "-",
        )

    logger.info(
        "[bold green]Tenant market backfill done[/bold green] tenants=%d updated=%d skipped=%d",
        len(tenants),
        updated,
        skipped,
    )


def cmd_enable_memory_forgetting(
    *,
    tenant_ids: list[str] | None = None,
    updated_by: str = "cli-admin",
) -> None:
    """Enables forgetting + access logs + shadow tuning for runtime tenants."""
    _cli_handle, settings, repo = _admin_runtime()
    tenants = _resolve_runtime_tenants(repo, tenant_ids=tenant_ids)

    updated = 0
    skipped = 0
    for tenant in tenants:
        current = load_memory_policy(repo, tenant, defaults=settings.memory_policy)
        next_policy = _with_forgetting_shadow(current)
        if next_policy == normalize_memory_policy(current, defaults=settings.memory_policy):
            skipped += 1
            continue
        repo.set_config(tenant, MEMORY_POLICY_CONFIG_KEY, serialize_memory_policy(next_policy), updated_by)
        repo.append_runtime_audit_log(
            action="enable_memory_forgetting",
            status="ok",
            user_email=updated_by,
            tenant_id=tenant,
            detail={
                "forgetting_enabled": True,
                "access_log_enabled": True,
                "tuning_enabled": True,
                "tuning_mode": "shadow",
                "auto_promote_enabled": False,
                "auto_demote_enabled": False,
            },
        )
        updated += 1
        logger.info(
            "[green]Memory forgetting enabled[/green] tenant=%s forgetting=true access_log=true tuning=shadow",
            tenant,
        )

    logger.info(
        "[bold green]Memory forgetting enable complete[/bold green] tenants=%d updated=%d skipped=%d",
        len(tenants),
        updated,
        skipped,
    )


def cmd_run_memory_forgetting_tuner(
    *,
    tenant_ids: list[str] | None = None,
    updated_by: str = "cli-memory-tuner",
) -> None:
    """Runs forgetting tuner for runtime tenants so the command can be scheduled externally."""
    _cli_handle, settings, repo = _admin_runtime()
    tenants = _resolve_runtime_tenants(repo, tenant_ids=tenant_ids)

    for tenant in tenants:
        state = run_memory_forgetting_tuner(
            repo,
            settings,
            tenant_id=tenant,
            updated_by=updated_by,
            persist_state=True,
        )
        sample = state.get("sample") if isinstance(state.get("sample"), dict) else {}
        gates = state.get("gates") if isinstance(state.get("gates"), dict) else {}
        transition = state.get("transition") if isinstance(state.get("transition"), dict) else {}
        logger.info(
            "Memory forgetting tuner tenant=%s reason=%s mode=%s effective=%s access=%s prompt_uses=%s unique=%s apply_allowed=%s transition=%s",
            tenant,
            str(state.get("reason") or "-"),
            str(state.get("configured_mode") or state.get("mode") or "-"),
            str(state.get("effective_mode") or "-"),
            int(sample.get("access_events") or 0),
            int(sample.get("prompt_uses") or 0),
            int(sample.get("unique_memories") or 0),
            "true" if bool(gates.get("apply_allowed")) else "false",
            str(transition.get("action") or "-"),
        )
