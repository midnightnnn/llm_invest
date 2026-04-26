from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.memory.policy import memory_event_enabled, memory_graph_semantic_triples_enabled
from arena.memory.tuning import run_memory_forgetting_tuner
from arena.orchestrator import ArenaOrchestrator

logger = logging.getLogger(__name__)


_COMPACTION_SOURCE_EVENT_TYPES = (
    "trade_execution",
    "react_tools_summary",
    "thesis_open",
    "thesis_update",
    "thesis_invalidated",
    "thesis_realized",
)


def _cli():
    import arena.cli as cli

    return cli


def _sync_broker_trade_ledger(*, live: bool, settings: Settings, repo: BigQueryRepository, tenant: str) -> None:
    """Refreshes broker trade ledger before reconciliation/cycle in live mode."""
    cli = _cli()
    if not live:
        return
    if not cli._truthy_env("ARENA_BROKER_TRADE_SYNC_ENABLED", True):
        return
    days = max(cli._int_env("ARENA_BROKER_TRADE_SYNC_DAYS", 14), 1)
    result = cli.BrokerTradeSyncService(settings=settings, repo=repo).sync_broker_trade_events(days=days)
    logger.info(
        "[cyan]Broker trade ledger sync[/cyan] tenant=%s inserted=%d scanned=%d skipped=%d failed_scopes=%d",
        tenant,
        result.inserted_events,
        result.scanned_rows,
        result.skipped_existing,
        len(result.failed_scopes),
    )


def _sync_broker_cash_ledger(*, live: bool, settings: Settings, repo: BigQueryRepository, tenant: str) -> None:
    """Refreshes broker cash ledger before reconciliation/cycle in live mode."""
    cli = _cli()
    if not live:
        return
    if not cli._truthy_env("ARENA_BROKER_CASH_SYNC_ENABLED", True):
        return
    days = max(cli._int_env("ARENA_BROKER_CASH_SYNC_DAYS", 14), 1)
    result = cli.BrokerCashSyncService(settings=settings, repo=repo).sync_broker_cash_events(days=days)
    logger.info(
        "[cyan]Broker cash ledger sync[/cyan] tenant=%s inserted=%d scanned=%d skipped=%d failed_scopes=%d",
        tenant,
        result.inserted_events,
        result.scanned_rows,
        result.skipped_existing,
        len(result.failed_scopes),
    )


def _run_reconciliation_guard(
    *,
    live: bool,
    settings: Settings,
    repo: BigQueryRepository,
    orchestrator: ArenaOrchestrator,
    tenant: str,
    snapshot=None,
    auto_recover: bool = True,
):
    """Runs pre-cycle reconciliation and fails closed on unresolved mismatches."""
    cli = _cli()
    if not live:
        return snapshot
    if not cli._truthy_env("ARENA_RECONCILIATION_ENABLED", True):
        return snapshot

    agent_ids = [
        str(getattr(agent, "agent_id", "") or "").strip()
        for agent in getattr(orchestrator, "agents", [])
        if str(getattr(agent, "agent_id", "") or "").strip()
    ] or list(settings.agent_ids)
    qty_tolerance = max(cli._float_env("ARENA_RECONCILE_QTY_TOLERANCE", 1e-9), 0.0)
    cash_tolerance_krw = max(cli._float_env("ARENA_RECONCILE_CASH_TOLERANCE_KRW", 50_000.0), 0.0)
    cash_reconciliation_enabled = cli._truthy_env("ARENA_RECONCILE_CASH_ENABLED", True)

    result = cli.StateReconciliationService(
        settings=settings,
        repo=repo,
        excluded_tickers=cli._reconcile_excluded_tickers(settings),
        qty_tolerance=qty_tolerance,
        cash_tolerance_krw=cash_tolerance_krw,
        cash_reconciliation_enabled=cash_reconciliation_enabled,
    ).reconcile_positions(
        agent_ids=agent_ids,
        tenant_id=tenant,
        include_simulated=False,
        auto_recover=bool(auto_recover),
        account_snapshot=snapshot,
        sync_account_snapshot=cli.AccountSyncService(settings=settings, repo=repo).sync_account_snapshot if auto_recover else None,
    )

    snapshot = result.account_snapshot or snapshot
    if not result.ok and auto_recover and cli._truthy_env("ARENA_RECONCILIATION_RECOVER_ENABLED", True):
        recovery = cli.StateRecoveryService(
            settings=settings,
            repo=repo,
            excluded_tickers=cli._reconcile_excluded_tickers(settings),
            qty_tolerance=qty_tolerance,
            cash_tolerance_krw=cash_tolerance_krw,
            cash_reconciliation_enabled=cash_reconciliation_enabled,
        ).recover_and_reconcile(
            agent_ids=agent_ids,
            tenant_id=tenant,
            include_simulated=False,
            auto_recover=False,
            account_snapshot=snapshot,
            created_by="cli_reconciliation_guard",
            allow_checkpoint_rebuild=cli._allow_checkpoint_rebuild_recovery(),
        )
        result = recovery.after
        snapshot = result.account_snapshot or snapshot
        logger.warning(
            "[yellow]Reconciliation recovery attempted[/yellow] tenant=%s status=%s checkpoints=%d",
            tenant,
            recovery.status,
            recovery.applied_checkpoints,
        )
    if result.ok:
        return snapshot

    issue_keys = ", ".join(f"{issue.issue_type}:{issue.entity_key}" for issue in result.issues[:5])
    message = (
        f"tenant={tenant} reconciliation_status={result.status} "
        f"issues={len(result.issues)} recoveries={','.join(result.recoveries) or '-'} "
        f"detail={issue_keys or '-'}"
    )
    if cli._truthy_env("ARENA_RECONCILE_FAIL_CLOSED", True):
        logger.error("[red]Cycle blocked by reconciliation[/red] %s", message)
        raise SystemExit(3)

    logger.warning("[yellow]Reconciliation mismatch ignored[/yellow] %s", message)
    return snapshot


def _run_memory_compaction(
    *,
    settings: Settings,
    repo: BigQueryRepository,
    orchestrator: ArenaOrchestrator,
    tenant: str,
) -> None:
    """Runs post-cycle memory compaction as a separate ADK phase."""
    if not settings.memory_compaction_enabled:
        return

    cycle_id = str(getattr(orchestrator, "last_cycle_id", "") or "").strip()
    if not cycle_id:
        return

    memory_store = getattr(getattr(orchestrator, "gateway", None), "memory_store", None)
    if memory_store is None:
        return

    agent_ids = [
        str(getattr(agent, "agent_id", "") or "").strip()
        for agent in getattr(orchestrator, "agents", [])
        if str(getattr(agent, "agent_id", "") or "").strip()
    ]
    if not agent_ids:
        return

    from arena.agents.memory_compaction_agent import MemoryCompactionAgent

    try:
        compactor = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)
        saved = asyncio.run(compactor.run(cycle_id=cycle_id, agent_ids=agent_ids))
    except Exception as exc:
        logger.warning(
            "[yellow]Memory compaction failed; continuing[/yellow] tenant=%s cycle_id=%s err=%s",
            tenant,
            cycle_id,
            str(exc),
        )
        return

    logger.info("[cyan]Memory compaction[/cyan] tenant=%s cycle_id=%s reflections=%d", tenant, cycle_id, len(saved))


def _memory_compaction_source_event_types(settings: Settings) -> list[str]:
    return [
        event_type
        for event_type in _COMPACTION_SOURCE_EVENT_TYPES
        if memory_event_enabled(settings.memory_policy, event_type, True)
    ]


def _resolve_memory_compaction_cycle_id(
    *,
    repo: BigQueryRepository,
    settings: Settings,
    agent_ids: list[str],
    cycle_id: str,
) -> str:
    clean_cycle = str(cycle_id or "").strip()
    if clean_cycle and clean_cycle.lower() not in {"latest", "last"}:
        return clean_cycle

    loader = getattr(repo, "latest_memory_compaction_cycle_id", None)
    if not callable(loader):
        raise RuntimeError("repo.latest_memory_compaction_cycle_id is required when --cycle-id is omitted")

    latest = str(
        loader(
            agent_ids=agent_ids,
            event_types=_memory_compaction_source_event_types(settings),
            trading_mode=settings.trading_mode,
        )
        or ""
    ).strip()
    return latest


def _clear_model_credentials(settings: Settings) -> None:
    settings.openai_api_key = ""
    settings.gemini_api_key = ""
    settings.anthropic_api_key = ""
    settings.research_gemini_api_key = ""
    settings.research_gemini_source = ""
    settings.research_gemini_source_tenant = ""


def _build_memory_compaction_runtime(
    *,
    live: bool,
    tenant: str,
    market_override: str,
    require_tenant_runtime_credentials: bool,
) -> tuple[Settings, BigQueryRepository]:
    """Builds just enough tenant runtime for memory compaction."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)

    clean_tenant = str(tenant or cli._tenant_id() or "local").strip().lower() or "local"
    repo = cli._repo_or_exit(settings, tenant_id=clean_tenant)
    repo.ensure_dataset()
    repo.ensure_tables()

    if require_tenant_runtime_credentials:
        _clear_model_credentials(settings)

    runtime_row = cli._apply_tenant_runtime_credentials(settings, repo, tenant_id=clean_tenant)
    if require_tenant_runtime_credentials:
        if not runtime_row:
            raise RuntimeError(f"tenant runtime credentials missing: tenant={clean_tenant}")
        if not str(runtime_row.get("model_secret_name") or "").strip():
            raise RuntimeError(f"tenant model_secret_name missing: tenant={clean_tenant}")

    cli.apply_runtime_overrides(settings, repo, tenant_id=clean_tenant)
    cli._apply_market_override(settings, market_override)
    settings.trading_mode = "live" if live else "paper"
    cli._validate_or_exit(settings, require_kis=False, require_llm=True, live=live)
    return settings, repo


def _memory_compaction_agent_ids(settings: Settings, requested_agents: list[str] | None = None) -> list[str]:
    out: list[str] = []
    for token in list(requested_agents or []) or list(getattr(settings, "agent_ids", [])):
        agent_id = str(token or "").strip()
        if agent_id and agent_id not in out:
            out.append(agent_id)
    return out


def _run_memory_compaction_for_tenant(
    *,
    live: bool,
    tenant: str,
    cycle_id: str,
    market_override: str,
    agent_ids: list[str] | None,
    timeout_seconds: int,
    dry_run: bool,
    force: bool,
    require_tenant_runtime_credentials: bool,
) -> dict[str, Any]:
    cli = _cli()
    tenant_id = str(tenant or cli._tenant_id() or "local").strip().lower() or "local"
    run_id = cli._new_run_id("memory_compaction")
    started_at = cli.utc_now()
    settings, repo = _build_memory_compaction_runtime(
        live=live,
        tenant=tenant_id,
        market_override=market_override,
        require_tenant_runtime_credentials=require_tenant_runtime_credentials,
    )
    if timeout_seconds > 0:
        settings.llm_timeout_runtime_override_seconds = int(timeout_seconds)

    selected_agents = _memory_compaction_agent_ids(settings, agent_ids)
    if not selected_agents:
        raise RuntimeError(f"no memory compaction agents selected: tenant={tenant_id}")

    resolved_cycle_id = _resolve_memory_compaction_cycle_id(
        repo=repo,
        settings=settings,
        agent_ids=selected_agents,
        cycle_id=cycle_id,
    )
    if not resolved_cycle_id:
        now = cli.utc_now()
        cli._append_tenant_run_status(
            repo,
            settings,
            tenant=tenant_id,
            run_id=run_id,
            run_type="memory_compaction",
            status="skipped",
            reason_code="cycle_not_found",
            stage="cycle_resolve",
            started_at=started_at,
            finished_at=now,
            message="컴팩션할 최신 사이클을 찾지 못했습니다.",
            detail={"live": bool(live), "market": market_override or settings.kis_target_market, "agents": selected_agents},
        )
        return {
            "tenant_id": tenant_id,
            "status": "skipped",
            "reason": "cycle_not_found",
            "cycle_id": "",
            "agent_ids": selected_agents,
            "saved_count": 0,
            "dry_run": bool(dry_run),
        }

    cli._append_tenant_run_status(
        repo,
        settings,
        tenant=tenant_id,
        run_id=run_id,
        run_type="memory_compaction",
        status="running",
        stage="compaction",
        started_at=started_at,
        message="메모리 컴팩션 실행 중입니다.",
        detail={
            "live": bool(live),
            "market": market_override or settings.kis_target_market,
            "cycle_id": resolved_cycle_id,
            "agents": selected_agents,
            "dry_run": bool(dry_run),
        },
    )

    from arena.agents.memory_compaction_agent import MemoryCompactionAgent
    from arena.memory.store import MemoryStore

    memory_store = MemoryStore(repo, trading_mode=settings.trading_mode, memory_policy=settings.memory_policy)
    compactor = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)
    try:
        if dry_run:
            preview = asyncio.run(compactor.preview(cycle_id=resolved_cycle_id, agent_ids=selected_agents))
            saved: list[dict[str, Any]] = []
            error_count = sum(1 for row in preview if str(row.get("error") or "").strip())
            result_detail: dict[str, Any] = {
                "preview_count": sum(int(row.get("reflection_count") or 0) for row in preview),
                "error_count": error_count,
                "previews": preview,
            }
        else:
            saved = asyncio.run(compactor.run(cycle_id=resolved_cycle_id, agent_ids=selected_agents, force=force))
            result_detail = {"saved": saved, "error_count": 0}
    except Exception as exc:
        cli._append_tenant_run_status(
            repo,
            settings,
            tenant=tenant_id,
            run_id=run_id,
            run_type="memory_compaction",
            status="failed",
            reason_code="compaction_failed",
            stage="compaction",
            started_at=started_at,
            finished_at=cli.utc_now(),
            message=str(exc),
            detail={"error": str(exc), "cycle_id": resolved_cycle_id, "agents": selected_agents},
        )
        raise

    saved_count = len(saved)
    error_count = int(result_detail.get("error_count") or 0)
    status = "warning" if error_count else "success"
    cli._append_tenant_run_status(
        repo,
        settings,
        tenant=tenant_id,
        run_id=run_id,
        run_type="memory_compaction",
        status=status,
        reason_code="compaction_preview_errors" if error_count else None,
        stage="complete",
        started_at=started_at,
        finished_at=cli.utc_now(),
        message=(
            f"메모리 컴팩션 dry-run이 경고와 함께 완료되었습니다. errors={error_count}"
            if dry_run and error_count
            else ("메모리 컴팩션 dry-run이 완료되었습니다." if dry_run else "메모리 컴팩션이 완료되었습니다.")
        ),
        detail={
            "live": bool(live),
            "market": market_override or settings.kis_target_market,
            "cycle_id": resolved_cycle_id,
            "agents": selected_agents,
            "dry_run": bool(dry_run),
            "saved_count": saved_count,
            **result_detail,
        },
    )
    return {
        "tenant_id": tenant_id,
        "status": status,
        "error_count": error_count,
        "cycle_id": resolved_cycle_id,
        "agent_ids": selected_agents,
        "dry_run": bool(dry_run),
        "saved_count": saved_count,
        **result_detail,
    }


def _resolve_memory_compaction_tenants(
    *,
    repo: BigQueryRepository,
    fallback: str,
    tenant_ids: list[str],
    all_tenants: bool,
    market_override: str,
) -> list[str]:
    cli = _cli()
    explicit: list[str] = []
    for raw in tenant_ids:
        for token in cli._parse_tenant_tokens(raw):
            if token not in explicit:
                explicit.append(token)
    if explicit:
        tenants = explicit
    elif all_tenants:
        tenants = cli._resolve_batch_tenants(repo, fallback=fallback)
    else:
        tenants = [fallback]
    tenants = cli._filter_tenants_by_market(repo, tenants, market_override)
    if all_tenants:
        tenants = cli._partition_tenants_for_task(tenants)
    return tenants


def cmd_run_memory_compaction(
    *,
    live: bool = False,
    all_tenants: bool = False,
    tenant_ids: list[str] | None = None,
    cycle_id: str = "",
    market_override: str = "",
    agent_ids: list[str] | None = None,
    timeout_seconds: int = 0,
    dry_run: bool = False,
    force: bool = False,
) -> None:
    """Runs the memory compactor as a first-class CLI command."""
    cli = _cli()
    bootstrap_settings = cli.load_settings()
    cli.configure_logging(bootstrap_settings.log_level, bootstrap_settings.log_format)
    cli._validate_or_exit(bootstrap_settings)
    fallback = cli._tenant_id() or "local"
    bootstrap_repo = cli._repo_or_exit(bootstrap_settings, tenant_id=fallback)
    bootstrap_repo.ensure_dataset()
    bootstrap_repo.ensure_tables()

    tenants = _resolve_memory_compaction_tenants(
        repo=bootstrap_repo,
        fallback=fallback,
        tenant_ids=list(tenant_ids or []),
        all_tenants=all_tenants,
        market_override=market_override,
    )
    if not tenants:
        logger.info("[yellow]No tenants selected for memory compaction; skipping[/yellow]")
        print(json.dumps({"status": "skipped", "reason": "no_tenants_selected"}, ensure_ascii=False))
        return

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for tenant in tenants:
        require_runtime = all_tenants or bool(tenant_ids) or str(tenant or "").strip().lower() != "local"
        try:
            results.append(
                _run_memory_compaction_for_tenant(
                    live=live,
                    tenant=tenant,
                    cycle_id=cycle_id,
                    market_override=market_override,
                    agent_ids=agent_ids,
                    timeout_seconds=timeout_seconds,
                    dry_run=dry_run,
                    force=force,
                    require_tenant_runtime_credentials=require_runtime,
                )
            )
        except Exception as exc:
            failures.append({"tenant_id": tenant, "error": str(exc)})
            logger.exception("[red]Memory compaction tenant failed[/red] tenant=%s err=%s", tenant, str(exc))

    output = {
        "status": "failed" if failures else ("warning" if any(row.get("status") == "warning" for row in results) else "success"),
        "tenant_count": len(tenants),
        "result_count": len(results),
        "failure_count": len(failures),
        "results": results,
        "failures": failures,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    if failures:
        raise SystemExit(1)


def _run_memory_relation_extraction_post_cycle(
    *,
    settings: Settings,
    repo: BigQueryRepository,
    tenant: str,
) -> None:
    """Runs semantic relation extraction after cycle writes without blocking trading."""
    cli = _cli()
    if not cli._truthy_env("ARENA_MEMORY_RELATION_EXTRACTION_POST_CYCLE_ENABLED", True):
        return
    if not memory_graph_semantic_triples_enabled(getattr(settings, "memory_policy", None)):
        return

    limit = max(1, cli._int_env("ARENA_MEMORY_RELATION_EXTRACTION_POST_CYCLE_LIMIT", 12))
    source_table = str(os.getenv("ARENA_MEMORY_RELATION_EXTRACTION_SOURCE_TABLE", "") or "").strip()
    event_types = cli._csv_env("ARENA_MEMORY_RELATION_EXTRACTION_EVENT_TYPES")
    min_confidence = max(0.0, min(cli._float_env("ARENA_MEMORY_RELATION_EXTRACTION_MIN_CONFIDENCE", 0.65), 1.0))
    max_triples = max(1, min(cli._int_env("ARENA_MEMORY_RELATION_EXTRACTION_MAX_TRIPLES", 6), 12))

    from arena.memory.semantic_extractor import SemanticRelationExtractor

    try:
        extractor = SemanticRelationExtractor(
            settings=settings,
            repo=repo,
            min_confidence=min_confidence,
            max_triples_per_source=max_triples,
        )
        rows = asyncio.run(
            extractor.run_pending(
                tenant_id=tenant,
                limit=limit,
                source_table=source_table or None,
                event_types=event_types or None,
                dry_run=False,
            )
        )
    except Exception as exc:
        logger.warning(
            "[yellow]Memory relation extraction failed; continuing[/yellow] tenant=%s err=%s",
            tenant,
            str(exc),
        )
        return

    accepted = sum(int(row.get("accepted_count") or 0) for row in rows)
    rejected = sum(int(row.get("rejected_count") or 0) for row in rows)
    logger.info(
        "[cyan]Memory relation extraction[/cyan] tenant=%s sources=%d accepted=%d rejected=%d",
        tenant,
        len(rows),
        accepted,
        rejected,
    )


def _run_memory_relation_tuner_post_cycle(
    *,
    settings: Settings,
    repo: BigQueryRepository,
    tenant: str,
) -> None:
    """Evaluates semantic relation gates and auto-switches shadow/inject when justified."""
    cli = _cli()
    if not cli._truthy_env("ARENA_MEMORY_RELATION_TUNER_POST_CYCLE_ENABLED", True):
        return

    from arena.memory.semantic_tuning import run_memory_relation_tuner

    try:
        state = run_memory_relation_tuner(
            repo,
            settings,
            tenant_id=tenant,
            updated_by="post-cycle-relation-tuner",
        )
    except Exception as exc:
        logger.warning(
            "[yellow]Memory relation tuner failed; continuing[/yellow] tenant=%s err=%s",
            tenant,
            str(exc),
        )
        return

    metrics = state.get("metrics") if isinstance(state.get("metrics"), dict) else {}
    transition = state.get("transition") if isinstance(state.get("transition"), dict) else {}
    logger.info(
        "[cyan]Memory relation tuner[/cyan] tenant=%s mode=%s effective=%s recommended=%s sources=%s accepted=%s unsafe=%s health=%s stability=%s transition=%s",
        tenant,
        str(state.get("configured_mode") or "-"),
        str(state.get("effective_mode") or "-"),
        str(state.get("recommended_mode") or "-"),
        int(metrics.get("source_count") or 0),
        int(metrics.get("accepted_count") or 0),
        int(metrics.get("unsafe_reject_count") or 0),
        "true" if bool((state.get("gates") or {}).get("health_ok")) else "false",
        "true" if bool((state.get("gates") or {}).get("stability_ok")) else "false",
        str(transition.get("action") or "-"),
    )


def _run_memory_forgetting_tuner_post_cycle(
    *,
    settings: Settings,
    repo: BigQueryRepository,
    tenant: str,
) -> None:
    """Runs forgetting tuning as a non-blocking post-cycle maintenance step."""
    cli = _cli()
    if not cli._truthy_env("ARENA_MEMORY_FORGETTING_TUNER_POST_CYCLE_ENABLED", True):
        return

    try:
        state = run_memory_forgetting_tuner(
            repo,
            settings,
            tenant_id=tenant,
            updated_by="post-cycle-forgetting-tuner",
        )
    except Exception as exc:
        logger.warning(
            "[yellow]Memory forgetting tuner failed; continuing[/yellow] tenant=%s err=%s",
            tenant,
            str(exc),
        )
        return

    sample = state.get("sample") if isinstance(state.get("sample"), dict) else {}
    transition = state.get("transition") if isinstance(state.get("transition"), dict) else {}
    gates = state.get("gates") if isinstance(state.get("gates"), dict) else {}
    logger.info(
        "[cyan]Memory forgetting tuner[/cyan] tenant=%s reason=%s mode=%s effective=%s access=%s prompt_uses=%s unique=%s apply_allowed=%s transition=%s",
        tenant,
        str(state.get("reason") or "").strip() or "-",
        str(state.get("mode") or "").strip() or "-",
        str(state.get("effective_mode") or "").strip() or "-",
        int(sample.get("access_events") or 0),
        int(sample.get("prompt_uses") or 0),
        int(sample.get("unique_memories") or 0),
        "true" if bool(gates.get("apply_allowed")) else "false",
        str(transition.get("action") or "").strip() or "-",
    )
