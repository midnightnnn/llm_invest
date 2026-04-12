from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.orchestrator import ArenaOrchestrator
from arena.cli_commands.run_shared import _MARKET_ALIAS

logger = logging.getLogger(__name__)


def _cli():
    import arena.cli as cli

    return cli


def _run_post_cycle_maintenance(
    cli_module,
    *,
    settings: Settings,
    repo: BigQueryRepository,
    orchestrator: ArenaOrchestrator,
    tenant: str,
) -> None:
    """Runs post-cycle maintenance without failing the trading cycle."""
    try:
        run_compaction = getattr(cli_module, "_run_memory_compaction")
        run_compaction(settings=settings, repo=repo, orchestrator=orchestrator, tenant=tenant)
    except Exception as exc:
        logger.warning(
            "[yellow]Post-cycle memory compaction failed; continuing[/yellow] tenant=%s err=%s",
            tenant,
            str(exc),
        )

    try:
        run_relation_extraction = getattr(cli_module, "_run_memory_relation_extraction_post_cycle")
        run_relation_extraction(settings=settings, repo=repo, tenant=tenant)
    except Exception as exc:
        logger.warning(
            "[yellow]Post-cycle memory relation extraction failed; continuing[/yellow] tenant=%s err=%s",
            tenant,
            str(exc),
        )

    try:
        run_relation_tuner = getattr(cli_module, "_run_memory_relation_tuner_post_cycle")
        run_relation_tuner(settings=settings, repo=repo, tenant=tenant)
    except Exception as exc:
        logger.warning(
            "[yellow]Post-cycle memory relation tuner failed; continuing[/yellow] tenant=%s err=%s",
            tenant,
            str(exc),
        )

    try:
        run_forgetting = getattr(cli_module, "_run_memory_forgetting_tuner_post_cycle")
        run_forgetting(settings=settings, repo=repo, tenant=tenant)
    except Exception as exc:
        logger.warning(
            "[yellow]Post-cycle memory forgetting tuner failed; continuing[/yellow] tenant=%s err=%s",
            tenant,
            str(exc),
        )


def cmd_run_cycle(live: bool) -> None:
    """Runs one multi-agent trading cycle."""
    cli = _cli()
    logger.warning("[yellow]run-cycle is deprecated[/yellow] use `run-agent-cycle` or `run-pipeline` instead")
    settings, repo, orchestrator = cli._build_runtime(live=live, require_kis=live)
    mode = "LIVE" if live else "PAPER"
    logger.info("[bold]Arena cycle start[/bold] mode=%s dataset=%s.%s", mode, settings.google_cloud_project, settings.bq_dataset)

    snapshot = None
    if live:
        reconciled = orchestrator.gateway.reconcile_submitted_orders()
        if reconciled:
            logger.info("[cyan]Pre-cycle reconcile[/cyan] updated=%d", reconciled)
        cli._sync_broker_trade_ledger(live=live, settings=settings, repo=repo, tenant=cli._tenant_id() or "local")
        cli._sync_broker_cash_ledger(live=live, settings=settings, repo=repo, tenant=cli._tenant_id() or "local")
        snapshot = cli.AccountSyncService(settings=settings, repo=repo).sync_account_snapshot()
        snapshot = cli._run_reconciliation_guard(
            live=live,
            settings=settings,
            repo=repo,
            orchestrator=orchestrator,
            tenant=cli._tenant_id() or "local",
            snapshot=snapshot,
        )

    if repo.recent_market_count() == 0:
        logger.warning("[yellow]No recent market_features rows[/yellow] run 'sync-market' first")

    reports = orchestrator.run_cycle(snapshot=snapshot)
    executed = sum(1 for report in reports if report.status.value in {"SIMULATED", "FILLED"})
    submitted = sum(1 for report in reports if report.status.value == "SUBMITTED")
    logger.info("[bold green]Arena cycle done[/bold green] executed=%d submitted=%d total=%d", executed, submitted, len(reports))


def _run_agent_cycle_once_guarded(
    live: bool,
    *,
    settings: Settings,
    repo: BigQueryRepository,
    orchestrator: ArenaOrchestrator,
    tenant: str,
    run_id: str | None = None,
    market_override: str = "",
) -> None:
    """Runs one tenant cycle with an optional per-market execution lease."""
    cli = _cli()
    execution_market = cli._execution_market_key(market_override, settings=settings)
    execution_source = cli._execution_source()
    run_token = str(run_id or "").strip() or cli._new_run_id("agent_cycle")
    owner_execution = (
        str(os.getenv("CLOUD_RUN_EXECUTION") or "").strip()
        or str(os.getenv("CLOUD_RUN_JOB") or "").strip()
        or "local"
    )
    shard = cli._task_shard_spec()
    if shard is not None:
        owner_execution = f"{owner_execution}:task{shard[0]}"

    lease_store = None
    lease_id = ""
    if live and execution_market and cli._tenant_lease_enabled():
        lease_store = cli.FirestoreTenantLeaseStore(
            project=settings.google_cloud_project,
            collection=str(os.getenv("ARENA_TENANT_LEASE_COLLECTION") or "tenant_cycle_leases").strip(),
        )
        acquired = lease_store.acquire(
            tenant_id=tenant,
            market=execution_market,
            trading_date=cli._execution_trading_date(execution_market),
            run_type="agent_cycle",
            execution_source=execution_source,
            owner_execution=owner_execution,
            run_id=run_token,
            lease_ttl_minutes=max(5, cli._int_env("ARENA_TENANT_LEASE_TTL_MINUTES", 120)),
            detail={
                "job_name": str(os.getenv("CLOUD_RUN_JOB") or "").strip() or None,
                "execution_name": str(os.getenv("CLOUD_RUN_EXECUTION") or "").strip() or None,
                "execution_source": execution_source or None,
            },
        )
        if not acquired.acquired:
            now = cli.utc_now()
            logger.info(
                "[yellow]Tenant cycle skipped by lease[/yellow] tenant=%s market=%s reason=%s",
                tenant,
                execution_market,
                acquired.reason,
            )
            cli._append_tenant_run_status(
                repo,
                settings,
                tenant=tenant,
                run_id=run_token,
                run_type="agent_cycle",
                status="skipped",
                reason_code=acquired.reason,
                stage="lease",
                started_at=now,
                finished_at=now,
                message="동일 tenant 실행 lease가 이미 존재해 건너뛰었습니다.",
                detail={
                    "market": execution_market,
                    "lease_id": acquired.lease_id,
                    "live": bool(live),
                    "execution_source": execution_source or None,
                },
            )
            return
        lease_id = acquired.lease_id

    try:
        cli._run_agent_cycle_once(
            live,
            settings=settings,
            repo=repo,
            orchestrator=orchestrator,
            tenant=tenant,
            run_id=run_token,
        )
    except BaseException as exc:
        if lease_store is not None and lease_id:
            try:
                lease_store.complete(
                    lease_id=lease_id,
                    status="failed",
                    owner_execution=owner_execution,
                    message=str(exc),
                    detail={"market": execution_market, "live": bool(live), "execution_source": execution_source or None},
                )
            except Exception as lease_exc:
                logger.warning(
                    "[yellow]Tenant lease completion failed[/yellow] tenant=%s status=failed err=%s",
                    tenant,
                    str(lease_exc),
                )
        raise
    else:
        if lease_store is not None and lease_id:
            try:
                lease_store.complete(
                    lease_id=lease_id,
                    status="success",
                    owner_execution=owner_execution,
                    message="agent cycle completed",
                    detail={"market": execution_market, "live": bool(live), "execution_source": execution_source or None},
                )
            except Exception as lease_exc:
                logger.warning(
                    "[yellow]Tenant lease completion failed[/yellow] tenant=%s status=success err=%s",
                    tenant,
                    str(lease_exc),
                )


def _run_agent_cycle_once(
    live: bool,
    *,
    settings: Settings,
    repo: BigQueryRepository,
    orchestrator: ArenaOrchestrator,
    tenant: str,
    run_id: str | None = None,
) -> None:
    """Runs one tenant-scoped agent cycle (no sync, no forecast)."""
    cli = _cli()
    logger.info("[bold]Agent cycle start[/bold] tenant=%s live=%s", tenant, live)
    run_token = str(run_id or "").strip() or cli._new_run_id("agent_cycle")
    started_at = cli.utc_now()
    current_stage = "start"
    cli._append_tenant_run_status(
        repo,
        settings,
        tenant=tenant,
        run_id=run_token,
        run_type="agent_cycle",
        status="running",
        stage=current_stage,
        started_at=started_at,
        message="에이전트 사이클 실행 중입니다.",
        detail={"live": bool(live)},
    )

    snapshot = None
    try:
        if live:
            current_stage = "sync"
            account_service = cli.AccountSyncService(settings=settings, repo=repo)
            reconciled = orchestrator.gateway.reconcile_submitted_orders()
            if reconciled:
                logger.info("[cyan]Pre-cycle reconcile[/cyan] updated=%d", reconciled)
            cli._sync_broker_trade_ledger(live=live, settings=settings, repo=repo, tenant=tenant)
            cli._sync_broker_cash_ledger(live=live, settings=settings, repo=repo, tenant=tenant)
            snapshot = account_service.sync_account_snapshot()
            current_stage = "reconcile"
            snapshot = cli._run_reconciliation_guard(
                live=live,
                settings=settings,
                repo=repo,
                orchestrator=orchestrator,
                tenant=tenant,
                snapshot=snapshot,
            )

        current_stage = "research"
        held_tickers = repo.get_all_held_tickers(market=settings.kis_target_market) if repo else []
        from arena.agents.research_agent import ResearchAgent

        research_agent = ResearchAgent(settings=settings, repo=repo)
        briefings = asyncio.run(research_agent.run(held_tickers))
        logger.info("[cyan]Research phase[/cyan] briefings=%d held=%s", len(briefings), held_tickers)

        current_stage = "trade"
        reports = orchestrator.run_cycle(snapshot=snapshot)
        current_stage = "post_cycle_maintenance"
        _run_post_cycle_maintenance(
            cli,
            settings=settings,
            repo=repo,
            orchestrator=orchestrator,
            tenant=tenant,
        )
        executed = sum(1 for report in reports if report.status.value in {"SIMULATED", "FILLED"})
        submitted = sum(1 for report in reports if report.status.value == "SUBMITTED")
        rejected = sum(1 for report in reports if report.status.value == "REJECTED")
        errored = sum(1 for report in reports if report.status.value == "ERROR")
        status = "warning" if (rejected or errored) else "success"
        reason_code = "broker_order_rejected" if rejected else ("cycle_report_error" if errored else None)
        logger.info(
            "[bold green]Agent cycle done[/bold green] tenant=%s executed=%d submitted=%d total=%d",
            tenant,
            executed,
            submitted,
            len(reports),
        )
        cli._append_tenant_run_status(
            repo,
            settings,
            tenant=tenant,
            run_id=run_token,
            run_type="agent_cycle",
            status=status,
            reason_code=reason_code,
            stage="complete",
            started_at=started_at,
            finished_at=cli.utc_now(),
            message="주문 반려가 있어 경고로 종료했습니다." if status == "warning" else "에이전트 사이클이 정상 완료되었습니다.",
            detail={
                "live": bool(live),
                "report_count": len(reports),
                "executed_count": executed,
                "submitted_count": submitted,
                "rejected_count": rejected,
                "error_count": errored,
            },
        )
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else None
        reason_code = "reconciliation_failed" if exit_code == 3 else "system_exit"
        status = "blocked" if exit_code == 3 else "failed"
        cli._append_tenant_run_status(
            repo,
            settings,
            tenant=tenant,
            run_id=run_token,
            run_type="agent_cycle",
            status=status,
            reason_code=reason_code,
            stage=current_stage,
            started_at=started_at,
            finished_at=cli.utc_now(),
            message="실계좌와 AI 장부가 맞지 않아 거래를 중단했습니다." if exit_code == 3 else f"에이전트 사이클이 중단되었습니다 (exit={exc.code}).",
            detail={"exit_code": exc.code, "live": bool(live)},
        )
        raise
    except Exception as exc:
        cli._append_tenant_run_status(
            repo,
            settings,
            tenant=tenant,
            run_id=run_token,
            run_type="agent_cycle",
            status="failed",
            reason_code="unexpected_exception",
            stage=current_stage,
            started_at=started_at,
            finished_at=cli.utc_now(),
            message=str(exc),
            detail={"error": str(exc), "live": bool(live)},
        )
        raise


def cmd_run_agent_cycle(live: bool, *, all_tenants: bool = False, market_override: str = "") -> None:
    """Runs the agent trading cycle only (no sync, no forecast)."""
    cli = _cli()
    if market_override.strip() and not all_tenants:
        os.environ["KIS_TARGET_MARKET"] = _MARKET_ALIAS.get(market_override.strip().lower(), market_override.strip().lower())
        logger.info("[cyan]Market override[/cyan] --market=%s", market_override)

    if not all_tenants:
        tenant = cli._tenant_id() or "local"
        run_id = cli._new_run_id("agent_cycle")
        try:
            settings, repo, orchestrator = cli._build_runtime(
                live=live,
                require_kis=live,
                tenant_id=tenant,
                execution_market=market_override,
            )
        except BaseException as exc:
            try:
                bootstrap_settings = cli.load_settings()
                cli.configure_logging(bootstrap_settings.log_level, bootstrap_settings.log_format)
                bootstrap_repo = cli._repo_or_exit(bootstrap_settings, tenant_id=tenant)
                bootstrap_repo.ensure_dataset()
                bootstrap_repo.ensure_tables()
                cli._append_tenant_run_status(
                    bootstrap_repo,
                    bootstrap_settings,
                    tenant=tenant,
                    run_id=run_id,
                    run_type="agent_cycle",
                    status="failed",
                    reason_code="runtime_build_failed",
                    stage="runtime",
                    finished_at=cli.utc_now(),
                    message=str(exc),
                    detail={"error": str(exc)},
                )
            except Exception:
                pass
            raise
        if live and cli._live_agent_cycle_market_closed(settings):
            logger.info("[yellow]All configured markets closed (holiday/weekend)[/yellow] — skipping agent cycle tenant=%s", tenant)
            now = cli.utc_now()
            cli._append_tenant_run_status(
                repo,
                settings,
                tenant=tenant,
                run_id=run_id,
                run_type="agent_cycle",
                status="skipped",
                reason_code="market_closed",
                stage="market_guard",
                started_at=now,
                finished_at=now,
                message="휴장일 또는 주말이라 에이전트 사이클을 건너뛰었습니다.",
                detail={"live": bool(live)},
            )
            return
        cli._run_agent_cycle_once(live, settings=settings, repo=repo, orchestrator=orchestrator, tenant=tenant, run_id=run_id)
        return

    bootstrap_settings = cli.load_settings()
    cli.configure_logging(bootstrap_settings.log_level, bootstrap_settings.log_format)
    cli._validate_or_exit(bootstrap_settings)
    bootstrap_repo = cli._repo_or_exit(bootstrap_settings, tenant_id="local")
    bootstrap_repo.ensure_dataset()
    bootstrap_repo.ensure_tables()

    fallback = cli._tenant_id() or "local"
    tenants = cli._resolve_batch_tenants(bootstrap_repo, fallback=fallback)
    tenants = cli._filter_tenants_by_market(bootstrap_repo, tenants, market_override)
    if not tenants:
        logger.info("[yellow]No tenants match --market=%s; skipping[/yellow]", market_override or "all")
        return
    tenants = cli._partition_tenants_for_task(tenants)
    if not tenants:
        logger.info("[yellow]No tenants assigned to this task shard; skipping[/yellow]")
        return
    run_ids = {tenant: cli._new_run_id("agent_cycle") for tenant in tenants}
    logger.info("[bold]Agent cycle multi-tenant start[/bold] tenants=%s live=%s", ",".join(tenants), live)

    max_workers = int(os.getenv("ARENA_BATCH_PARALLEL", "3") or "3")
    build_failures: list[tuple[str, str]] = []
    runtimes: list[tuple[str, Settings, BigQueryRepository, ArenaOrchestrator]] = []

    def _build_agent_tenant(tenant_id: str):
        return tenant_id, *cli._build_runtime(
            live=live,
            require_kis=live,
            tenant_id=tenant_id,
            require_tenant_runtime_credentials=True,
            execution_market=market_override,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_build_agent_tenant, tenant_id): tenant_id for tenant_id in tenants}
        for future in as_completed(futures):
            tenant = futures[future]
            try:
                runtimes.append(future.result())
            except SystemExit as exc:
                build_failures.append((tenant, f"SystemExit({exc.code})"))
                cli._append_tenant_run_status(
                    bootstrap_repo,
                    bootstrap_settings,
                    tenant=tenant,
                    run_id=run_ids.get(tenant) or cli._new_run_id("agent_cycle"),
                    run_type="agent_cycle",
                    status="failed",
                    reason_code="runtime_build_failed",
                    stage="runtime",
                    finished_at=cli.utc_now(),
                    message=f"runtime build blocked: SystemExit({exc.code})",
                    detail={"exit_code": int(exc.code) if isinstance(exc.code, int) else str(exc.code)},
                )
                logger.exception("[red]Agent cycle tenant build blocked[/red] tenant=%s code=%s", tenant, exc.code)
            except Exception as exc:
                build_failures.append((tenant, str(exc)))
                cli._append_tenant_run_status(
                    bootstrap_repo,
                    bootstrap_settings,
                    tenant=tenant,
                    run_id=run_ids.get(tenant) or cli._new_run_id("agent_cycle"),
                    run_type="agent_cycle",
                    status="failed",
                    reason_code="runtime_build_failed",
                    stage="runtime",
                    finished_at=cli.utc_now(),
                    message=str(exc),
                    detail={"error": str(exc)},
                )
                logger.exception("[red]Agent cycle tenant build failed[/red] tenant=%s err=%s", tenant, str(exc))

    if not runtimes:
        logger.error("[red]Agent cycle multi-tenant failed[/red] no tenants initialized")
        raise SystemExit(1)

    if live:
        filtered: list[tuple[str, Settings, BigQueryRepository, ArenaOrchestrator]] = []
        skipped_closed = 0
        for runtime in runtimes:
            tenant, settings, repo, orchestrator = runtime
            if cli._live_agent_cycle_market_closed(settings):
                skipped_closed += 1
                logger.info("[yellow]All configured markets closed (holiday/weekend)[/yellow] — skipping agent cycle tenant=%s", tenant)
                now = cli.utc_now()
                cli._append_tenant_run_status(
                    repo,
                    settings,
                    tenant=tenant,
                    run_id=run_ids.get(tenant) or cli._new_run_id("agent_cycle"),
                    run_type="agent_cycle",
                    status="skipped",
                    reason_code="market_closed",
                    stage="market_guard",
                    started_at=now,
                    finished_at=now,
                    message="휴장일 또는 주말이라 에이전트 사이클을 건너뛰었습니다.",
                    detail={"live": bool(live)},
                )
                continue
            filtered.append((tenant, settings, repo, orchestrator))
        runtimes = filtered
        if skipped_closed and not runtimes:
            logger.info("[yellow]All selected tenants skipped due to market closure[/yellow]")
            return

    exec_failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                cli._run_agent_cycle_once_guarded,
                live,
                settings=settings,
                repo=repo,
                orchestrator=orchestrator,
                tenant=tenant,
                run_id=run_ids.get(tenant) or cli._new_run_id("agent_cycle"),
                market_override=market_override,
            ): tenant
            for tenant, settings, repo, orchestrator in runtimes
        }
        for future in as_completed(futures):
            tenant = futures[future]
            try:
                future.result()
            except SystemExit as exc:
                exec_failures.append((tenant, f"SystemExit({exc.code})"))
                logger.exception("[red]Agent cycle tenant blocked[/red] tenant=%s code=%s", tenant, exc.code)
            except Exception as exc:
                exec_failures.append((tenant, str(exc)))
                logger.exception("[red]Agent cycle tenant failed[/red] tenant=%s err=%s", tenant, str(exc))

    failures = build_failures + exec_failures
    if failures:
        logger.error(
            "[red]Agent cycle multi-tenant completed with failures[/red] failed=%s",
            ", ".join([f"{tenant}:{error}" for tenant, error in failures]),
        )
        raise SystemExit(1)
    logger.info("[bold green]Agent cycle multi-tenant done[/bold green] tenants=%d", len(runtimes))
