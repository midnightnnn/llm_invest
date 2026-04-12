from __future__ import annotations

import asyncio
import logging
import os

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.memory.policy import memory_graph_semantic_triples_enabled
from arena.memory.tuning import run_memory_forgetting_tuner
from arena.orchestrator import ArenaOrchestrator

logger = logging.getLogger(__name__)


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
