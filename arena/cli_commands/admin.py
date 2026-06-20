from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from arena.memory.policy import (
    MEMORY_POLICY_CONFIG_KEY,
    load_memory_policy,
    normalize_memory_policy,
    serialize_memory_policy,
)
from arena.memory.candidate_structured import backfill_candidate_memory_structures
from arena.memory.watch_items import build_watch_record
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


_WATCH_TICKER_RE = re.compile(r"\b([A-Z]{1,5}|\d{6})\b")


def _json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        return list(parsed) if isinstance(parsed, list) else []
    return []


def _parse_dt(value: object) -> datetime | None:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _sql_text(value: object) -> str:
    return "'" + str(value or "").replace("'", "''") + "'"


def _extract_ticker(payload: dict[str, object], summary: str) -> str:
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    for candidate in (
        payload.get("ticker"),
        intent.get("ticker") if isinstance(intent, dict) else "",
    ):
        token = str(candidate or "").strip().upper()
        if token:
            return token
    match = _WATCH_TICKER_RE.search(summary or "")
    return str(match.group(1) or "").strip().upper() if match else ""


def _candidate_watch_record(row: dict[str, object], *, agent_id: str, source_phase: str) -> dict[str, object] | None:
    summary = str(row.get("summary") or "").strip()
    payload = _json_object(row.get("payload_json"))
    event_type = str(row.get("event_type") or "").strip().lower()
    if not summary and not payload:
        return None
    ticker = _extract_ticker(payload, summary)
    source_doc_ids = _json_list(payload.get("source_doc_ids"))
    if not source_doc_ids:
        source_doc_id = str(payload.get("source_doc_id") or "").strip()
        if source_doc_id:
            source_doc_ids = [source_doc_id]
    status_map = {
        "candidate_screen_hit": "active",
        "candidate_watchlist": "active",
        "candidate_thesis": "active",
        "candidate_rejected": "archived",
    }
    horizon_map = {
        "candidate_screen_hit": 14,
        "candidate_watchlist": 30,
        "candidate_thesis": 60,
        "candidate_rejected": 45,
    }
    priority_map = {
        "candidate_screen_hit": 0.25,
        "candidate_watchlist": 0.38,
        "candidate_thesis": 0.55,
        "candidate_rejected": 0.32,
    }
    watch_payload = {
        "source": "candidate_memory_backfill",
        "memory_event_id": str(row.get("event_id") or "").strip(),
        "memory_event_type": event_type,
        "payload": payload,
    }
    watch_indicators = _json_list(payload.get("suggested_next_checks"))
    if not watch_indicators:
        watch_indicators = _json_list(payload.get("source_tools"))
    if not watch_indicators:
        watch_indicators = _json_list(payload.get("analyzed_by"))
    context_tags = {"watch_indicators": [str(item).strip() for item in watch_indicators if str(item).strip()]}
    record = build_watch_record(
        agent_id=agent_id,
        watch_kind="candidate",
        summary=summary or f"{ticker or 'candidate'} watch",
        title=str(payload.get("title") or f"{ticker or 'candidate'} candidate watch").strip(),
        status=status_map.get(event_type, "active"),
        ticker=ticker or None,
        source_doc_id=str(payload.get("source_doc_id") or "").strip() or None,
        source_doc_ids=[str(item or "").strip() for item in source_doc_ids if str(item or "").strip()],
        payload=watch_payload,
        cycle_id=str(row.get("cycle_id") or payload.get("cycle_id") or "").strip(),
        llm_call_id=str(row.get("llm_call_id") or payload.get("llm_call_id") or "").strip(),
        source_phase=source_phase,
        source_event=event_type or "candidate_backfill",
        priority_score=priority_map.get(event_type, 0.35),
        time_horizon_days=horizon_map.get(event_type, 30),
        created_at=_parse_dt(row.get("created_at")),
        updated_at=_parse_dt(row.get("updated_at")),
        context_tags=context_tags,
    )
    return record


def _post_exit_watch_record(row: dict[str, object], *, agent_id: str, source_phase: str) -> dict[str, object] | None:
    summary = str(row.get("summary") or "").strip()
    payload = _json_object(row.get("payload_json"))
    event_type = str(row.get("event_type") or "").strip().lower()
    if event_type not in {"thesis_invalidated", "thesis_realized"}:
        return None
    intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
    ticker = _extract_ticker(payload, summary)
    record = build_watch_record(
        agent_id=agent_id,
        watch_kind="post_exit",
        summary=summary or f"{ticker or 'post exit'} watch",
        title=str(payload.get("title") or f"{ticker or 'post exit'} {event_type}").strip(),
        status="resolved",
        ticker=ticker or None,
        payload={
            "source": "thesis_lifecycle_backfill",
            "memory_event_id": str(row.get("event_id") or "").strip(),
            "memory_event_type": event_type,
            "payload": payload,
            "decision": decision,
            "report": report,
        },
        cycle_id=str(row.get("cycle_id") or payload.get("cycle_id") or "").strip(),
        llm_call_id=str(row.get("llm_call_id") or payload.get("llm_call_id") or "").strip(),
        source_phase=source_phase,
        source_event=event_type,
        priority_score=0.78 if event_type == "thesis_invalidated" else 0.74,
        time_horizon_days=_parse_int(payload.get("time_horizon_days")),
        resolved_at=_parse_dt(row.get("updated_at") or row.get("created_at")),
        resolution=event_type,
        created_at=_parse_dt(row.get("created_at")),
        updated_at=_parse_dt(row.get("updated_at")),
        context_tags={
            "thesis_id": str(payload.get("thesis_id") or "").strip(),
            "position_action": str(payload.get("position_action") or "").strip(),
        },
    )
    return record


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


def cmd_backfill_candidate_memory_structures(
    *,
    tenant_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    live: bool = False,
    dry_run: bool = False,
    include_existing: bool = False,
    limit_per_agent: int = 1000,
) -> None:
    """Backfills candidate memory payload_json.structured_memory for prompt-safe recall."""
    cli = _cli()
    settings = cli.load_settings()
    if live:
        settings.trading_mode = "live"
    cli.configure_logging(settings.log_level, settings.log_format)
    repo = cli._repo_or_exit(settings, tenant_id="local")
    repo.ensure_dataset()
    repo.ensure_tables()

    explicit_tenants = [str(token or "").strip().lower() for token in (tenant_ids or []) if str(token or "").strip()]
    if explicit_tenants:
        tenants = list(dict.fromkeys(explicit_tenants))
    else:
        tenants = list(repo.list_runtime_tenants(limit=2000))
    if not tenants:
        tenants = ["local"]

    explicit_agents = [str(token or "").strip() for token in (agent_ids or []) if str(token or "").strip()]
    agents = list(dict.fromkeys(explicit_agents or [str(agent or "").strip() for agent in settings.agent_ids if str(agent or "").strip()]))
    if not agents:
        raise SystemExit("No agent ids configured; pass --agent")

    scanned = updated = would_update = skipped = 0
    quality_counts: dict[str, int] = {}
    for tenant in tenants:
        result = backfill_candidate_memory_structures(
            repo,
            agent_ids=agents,
            trading_mode=settings.trading_mode,
            tenant_id=tenant,
            limit_per_agent=max(1, int(limit_per_agent or 1000)),
            dry_run=dry_run,
            include_existing=include_existing,
        )
        scanned += result.scanned
        updated += result.updated
        would_update += result.would_update
        skipped += result.skipped
        for key, value in result.quality_counts.items():
            quality_counts[key] = quality_counts.get(key, 0) + int(value)
        logger.info(
            "[green]Candidate memory structure backfill[/green] tenant=%s scanned=%d updated=%d would_update=%d dry_run=%s qualities=%s",
            tenant,
            result.scanned,
            result.updated,
            result.would_update,
            dry_run,
            result.quality_counts,
        )

    logger.info(
        "[bold green]Candidate memory structure backfill done[/bold green] tenants=%d agents=%d scanned=%d updated=%d would_update=%d skipped=%d dry_run=%s qualities=%s",
        len(tenants),
        len(agents),
        scanned,
        updated,
        would_update,
        skipped,
        dry_run,
        quality_counts,
    )


def cmd_backfill_watch_items(
    *,
    tenant_ids: list[str] | None = None,
    agent_ids: list[str] | None = None,
    live: bool = False,
    dry_run: bool = False,
    include_existing: bool = False,
    limit_per_agent: int = 1000,
) -> None:
    """Backfills durable watch items from candidate and thesis memory history."""
    cli = _cli()
    settings = cli.load_settings()
    if live:
        settings.trading_mode = "live"
    cli.configure_logging(settings.log_level, settings.log_format)
    repo = cli._repo_or_exit(settings, tenant_id="local")
    repo.ensure_dataset()
    repo.ensure_tables()

    explicit_tenants = [str(token or "").strip().lower() for token in (tenant_ids or []) if str(token or "").strip()]
    if explicit_tenants:
        tenants = list(dict.fromkeys(explicit_tenants))
    else:
        tenants = list(repo.list_runtime_tenants(limit=2000))
    if not tenants:
        tenants = ["local"]

    explicit_agents = [str(token or "").strip() for token in (agent_ids or []) if str(token or "").strip()]
    agents = list(dict.fromkeys(explicit_agents or [str(agent or "").strip() for agent in settings.agent_ids if str(agent or "").strip()]))
    if not agents:
        raise SystemExit("No agent ids configured; pass --agent")

    limit = max(1, int(limit_per_agent or 1000))
    total_scanned = 0
    total_written = 0
    total_skipped = 0

    for tenant in tenants:
        tenant_scanned = 0
        tenant_written = 0
        tenant_skipped = 0
        for agent in agents:
            candidate_rows = []
            candidate_loader = getattr(repo, "candidate_memory_events_for_structured_backfill", None)
            if callable(candidate_loader):
                try:
                    candidate_rows = candidate_loader(
                        agent_id=agent,
                        trading_mode=settings.trading_mode,
                        tenant_id=tenant,
                        limit=limit,
                        include_existing=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "[yellow]watch candidate backfill failed[/yellow] tenant=%s agent=%s err=%s",
                        tenant,
                        agent,
                        str(exc),
                    )
                    candidate_rows = []
            thesis_rows = []
            try:
                thesis_rows = repo.fetch_rows(
                    f"""
                    SELECT event_id, created_at, updated_at, agent_id, event_type, summary, cycle_id, llm_call_id, payload_json
                    FROM `{repo.dataset_fqn}.agent_memory_events`
                    WHERE tenant_id = {_sql_text(tenant)}
                      AND agent_id = {_sql_text(agent)}
                      AND trading_mode = {_sql_text(settings.trading_mode)}
                      AND event_type IN ('thesis_invalidated', 'thesis_realized')
                    ORDER BY created_at DESC
                    LIMIT {limit}
                    """,
                    None,
                )
            except Exception as exc:
                logger.warning(
                    "[yellow]watch thesis backfill failed[/yellow] tenant=%s agent=%s err=%s",
                    tenant,
                    agent,
                    str(exc),
                )
                thesis_rows = []

            for row in list(candidate_rows or []) + list(thesis_rows or []):
                if not isinstance(row, dict):
                    continue
                event_type = str(row.get("event_type") or "").strip().lower()
                if event_type.startswith("candidate_"):
                    record = _candidate_watch_record(row, agent_id=agent, source_phase="backfill")
                else:
                    record = _post_exit_watch_record(row, agent_id=agent, source_phase="backfill")
                if record is None:
                    continue
                tenant_scanned += 1
                watch_key = str(record.get("watch_key") or "").strip()
                if watch_key and not include_existing:
                    try:
                        existing = repo.watch_item_by_key(watch_key=watch_key, tenant_id=tenant)
                    except Exception:
                        existing = None
                    if existing:
                        tenant_skipped += 1
                        continue
                if not dry_run:
                    repo.upsert_watch_item(record, tenant_id=tenant)
                tenant_written += 1

        total_scanned += tenant_scanned
        total_written += tenant_written
        total_skipped += tenant_skipped
        logger.info(
            "[green]Watch backfill[/green] tenant=%s scanned=%d written=%d skipped=%d dry_run=%s",
            tenant,
            tenant_scanned,
            tenant_written,
            tenant_skipped,
            dry_run,
        )

    logger.info(
        "[bold green]Watch backfill done[/bold green] tenants=%d agents=%d scanned=%d written=%d skipped=%d dry_run=%s",
        len(tenants),
        len(agents),
        total_scanned,
        total_written,
        total_skipped,
        dry_run,
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
