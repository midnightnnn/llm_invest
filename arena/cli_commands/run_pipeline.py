from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.logging_utils import event_extra, failure_extra
from arena.market_hours import MarketWindow
from arena.orchestrator import ArenaOrchestrator
from arena.cli_commands.run_shared import _MARKET_ALIAS

logger = logging.getLogger(__name__)


def _cli():
    import arena.cli as cli

    return cli


def _batch_tenant_work(
    phase: str | None,
    live: bool,
    *,
    settings: Settings,
    repo: BigQueryRepository,
    orchestrator: ArenaOrchestrator,
    tenant: str,
    window: MarketWindow | None,
) -> None:
    """Runs tenant-scoped work based on batch phase. Thread-safe."""
    cli = _cli()
    if phase in ("open_cycle", "general"):
        snapshot = None
        if live:
            reconciled = orchestrator.gateway.reconcile_submitted_orders()
            if reconciled:
                logger.info("[cyan]Pre-cycle reconcile[/cyan] tenant=%s updated=%d", tenant, reconciled)
            cli._sync_broker_trade_ledger(live=live, settings=settings, repo=repo, tenant=tenant)
            cli._sync_broker_cash_ledger(live=live, settings=settings, repo=repo, tenant=tenant)
            snapshot = cli.AccountSyncService(settings=settings, repo=repo).sync_account_snapshot()

            if settings.dividend_sync_enabled:
                try:
                    usd_krw = snapshot.usd_krw_rate if snapshot and snapshot.usd_krw_rate > 0 else None
                    div_result = cli.DividendSyncService(settings=settings, repo=repo).sync_dividends(usd_krw_override=usd_krw)
                    logger.info(
                        "[cyan]Dividend sync[/cyan] tenant=%s inserted=%d skipped=%d cash=%d fx=%.2f",
                        tenant,
                        div_result.events_inserted,
                        div_result.skipped_duplicate,
                        div_result.broker_cash_events_inserted,
                        usd_krw or settings.usd_krw_rate,
                    )
                except Exception as exc:
                    logger.warning("[yellow]Dividend sync failed; continuing[/yellow] tenant=%s err=%s", tenant, str(exc))

            snapshot = cli._run_reconciliation_guard(
                live=live,
                settings=settings,
                repo=repo,
                orchestrator=orchestrator,
                tenant=tenant,
                snapshot=snapshot,
            )

        held_tickers = repo.get_all_held_tickers(market=settings.kis_target_market) if repo else []
        from arena.agents.research_agent import ResearchAgent

        research_agent = ResearchAgent(settings=settings, repo=repo)
        briefings = asyncio.run(research_agent.run(held_tickers))
        logger.info("[cyan]Research phase[/cyan] tenant=%s briefings=%d held=%s", tenant, len(briefings), held_tickers)

        reports = orchestrator.run_cycle(snapshot=snapshot)
        cli._run_post_cycle_maintenance(
            cli,
            settings=settings,
            repo=repo,
            orchestrator=orchestrator,
            tenant=tenant,
        )
        executed = sum(1 for report in reports if report.status.value in {"SIMULATED", "FILLED"})
        submitted = sum(1 for report in reports if report.status.value == "SUBMITTED")
        logger.info(
            "[bold green]Batch done[/bold green] tenant=%s executed=%d submitted=%d total_reports=%d",
            tenant,
            executed,
            submitted,
            len(reports),
        )

    elif phase == "report":
        cli.AccountSyncService(settings=settings, repo=repo).sync_account_snapshot()
        logger.info("[bold green]Batch done[/bold green] tenant=%s mode=report", tenant)

    elif phase == "closed":
        reconciled = orchestrator.gateway.reconcile_submitted_orders()
        if reconciled:
            logger.info("[cyan]Off-hours reconcile[/cyan] tenant=%s updated=%d", tenant, reconciled)
        logger.info("[cyan]Market closed; skipping cycle[/cyan] tenant=%s phase=%s", tenant, window.phase if window else "unknown")

    elif phase == "seed":
        logger.info("[bold green]Batch done[/bold green] tenant=%s mode=seed", tenant)


def _run_batch_once(
    live: bool,
    *,
    settings: Settings,
    repo: BigQueryRepository,
    orchestrator: ArenaOrchestrator,
    tenant: str,
) -> None:
    """Runs one tenant-scoped batch cycle (single-tenant path)."""
    cli = _cli()
    logger.info("[bold]Batch start[/bold] tenant=%s live=%s market=%s", tenant, live, settings.kis_target_market)

    phase, window = cli._batch_phase(live, settings, repo)
    cli._batch_market_sync(phase, settings, repo, window)
    if phase == "seed" or phase is None:
        if phase == "seed":
            logger.info("[bold green]Batch done[/bold green] tenant=%s mode=seed", tenant)
        return
    cli._batch_tenant_work(phase, live, settings=settings, repo=repo, orchestrator=orchestrator, tenant=tenant, window=window)


def cmd_run_shared_prep(
    live: bool,
    *,
    market_override: str = "",
    dispatch_job: str = "",
) -> None:
    """Runs shared market sync + forecast steps once, then optionally dispatches the agent job."""
    cli = _cli()
    bootstrap_settings = cli.load_settings()
    cli.configure_logging(bootstrap_settings.log_level, bootstrap_settings.log_format)
    cli._validate_or_exit(bootstrap_settings)
    logger.info(
        "[bold]Shared prep start[/bold] live=%s market=%s dispatch_job=%s",
        live,
        market_override or "all",
        dispatch_job or "-",
    )

    cli._apply_market_override(bootstrap_settings, market_override)
    orig_market_env = os.environ.get("KIS_TARGET_MARKET")
    if market_override.strip():
        os.environ["KIS_TARGET_MARKET"] = bootstrap_settings.kis_target_market

    try:
        markets = cli._parse_cli_markets(bootstrap_settings)
        us_open = False
        kr_open = False

        if cli._has_us(markets):
            us_win = cli.nasdaq_window()
            us_open = not (us_win.now_local.weekday() >= 5 or cli.is_nasdaq_holiday(us_win.trading_date))

        if cli._has_kr(markets):
            kr_win = cli.kospi_window()
            kr_open = not cli.is_kospi_holiday(kr_win.trading_date)

        if not us_open and not kr_open:
            logger.info("[yellow]All markets closed (holiday/weekend)[/yellow] — skipping shared prep")
            return

        repo = cli._repo_or_exit(bootstrap_settings, tenant_id=cli._tenant_id() or "local")
        repo.ensure_dataset()
        repo.ensure_tables()
        cli._apply_tenant_runtime_credentials(bootstrap_settings, repo)

        phase, window = cli._batch_phase(live, bootstrap_settings, repo)
        if phase is None:
            logger.info("[yellow]Shared prep: not a scheduled cycle time — skipping[/yellow]")
            return

        logger.info("[bold cyan]Shared prep step 1/4: sync-market[/bold cyan]")
        cli._batch_market_sync(phase, bootstrap_settings, repo, window)
        if phase == "seed":
            logger.info("[bold green]Shared prep done[/bold green] mode=seed")
            return

        logger.info("[bold cyan]Shared prep step 2/4: build-forecasts[/bold cyan]")
        fc_args = type(
            "Args",
            (),
            {
                "top_n": 50,
                "lookback_days": 360,
                "horizon": 20,
                "min_series_length": 160,
                "max_steps": 200,
            },
        )()
        cli.cmd_build_forecasts(fc_args)

        logger.info("[bold cyan]Shared prep step 3/4: refresh-fundamentals-derived[/bold cyan]")
        fund_args = type(
            "Args",
            (),
            {"lookback_days": 600},
        )()
        try:
            cli.cmd_refresh_fundamentals_derived(fund_args)
        except Exception as exc:
            logger.warning("[yellow]fundamentals derived refresh skipped (non-fatal)[/yellow] err=%s", exc)

        logger.info("[bold cyan]Shared prep step 4/4: build-opportunity-ranker[/bold cyan]")
        ranker_args = type(
            "Args",
            (),
            {
                "lookback_days": 540,
                "horizon": 20,
                "max_scoring_rows": 500,
                "min_ic_dates": 60,
                "min_valid_signals": 3,
            },
        )()
        cli.cmd_build_opportunity_ranker(ranker_args)

        if dispatch_job.strip():
            cli._dispatch_agent_job(bootstrap_settings, job_name=dispatch_job.strip())

        logger.info("[bold green]Shared prep done[/bold green]")
    finally:
        if market_override.strip():
            if orig_market_env is not None:
                os.environ["KIS_TARGET_MARKET"] = orig_market_env
            else:
                os.environ.pop("KIS_TARGET_MARKET", None)


def cmd_run_pipeline(live: bool, *, all_tenants: bool = False, market_override: str = "") -> None:
    """Runs sync → forecast → opportunity-ranker → agent cycle sequentially."""
    cli = _cli()
    bootstrap_settings = cli.load_settings()
    cli.configure_logging(bootstrap_settings.log_level, bootstrap_settings.log_format)
    cli._validate_or_exit(bootstrap_settings)
    logger.info("[bold]Pipeline start[/bold] live=%s all_tenants=%s market=%s", live, all_tenants, market_override or "all")
    fallback = cli._tenant_id() or "local"

    settings_peek = bootstrap_settings
    cli._apply_market_override(settings_peek, market_override)

    original_market_env = os.environ.get("KIS_TARGET_MARKET")
    if market_override.strip():
        os.environ["KIS_TARGET_MARKET"] = settings_peek.kis_target_market

    markets = cli._parse_cli_markets(settings_peek)
    us_open = False
    kr_open = False

    if cli._has_us(markets):
        us_win = cli.nasdaq_window()
        us_open = not (us_win.now_local.weekday() >= 5 or cli.is_nasdaq_holiday(us_win.trading_date))

    if cli._has_kr(markets):
        kr_win = cli.kospi_window()
        kr_open = not cli.is_kospi_holiday(kr_win.trading_date)

    if not us_open and not kr_open:
        logger.info("[yellow]All markets closed (holiday/weekend)[/yellow] — skipping pipeline")
        if market_override.strip():
            if original_market_env is not None:
                os.environ["KIS_TARGET_MARKET"] = original_market_env
            else:
                os.environ.pop("KIS_TARGET_MARKET", None)
        return

    bootstrap_repo = cli._repo_or_exit(bootstrap_settings, tenant_id=fallback)
    bootstrap_repo.ensure_dataset()
    bootstrap_repo.ensure_tables()
    pipeline_targets = cli._resolve_batch_tenants(bootstrap_repo, fallback=fallback) if all_tenants else [fallback]
    pipeline_run_ids = {tenant: cli._new_run_id("pipeline") for tenant in pipeline_targets}

    current_stage = "sync"
    try:
        logger.info("[bold cyan]Pipeline step 1/7: sync-market[/bold cyan]")
        pipeline_repo = bootstrap_repo
        cli._apply_tenant_runtime_credentials(bootstrap_settings, pipeline_repo)
        pipeline_phase, pipeline_window = cli._batch_phase(live, settings_peek, pipeline_repo)
        if pipeline_phase is None:
            logger.info("[yellow]Pipeline: not a scheduled cycle time — skipping[/yellow]")
            now = cli.utc_now()
            cli._append_tenant_run_status_many(
                bootstrap_repo,
                bootstrap_settings,
                tenants=pipeline_targets,
                run_ids=pipeline_run_ids,
                run_type="pipeline",
                status="skipped",
                reason_code="schedule_closed",
                stage="schedule_guard",
                started_at=now,
                finished_at=now,
                message="예약된 실행 시간이 아니라 배치를 건너뛰었습니다.",
                detail={"live": bool(live), "market_override": market_override or None},
            )
            return
        cli._batch_market_sync(pipeline_phase, settings_peek, pipeline_repo, pipeline_window)

        if pipeline_phase == "seed":
            logger.info("[bold green]Pipeline done[/bold green] mode=seed (daily history seeding)")
            return

        current_stage = "forecast"
        logger.info("[bold cyan]Pipeline step 2/7: build-forecasts[/bold cyan]")
        fc_args = type(
            "Args",
            (),
            {
                "top_n": 50,
                "lookback_days": 360,
                "horizon": 20,
                "min_series_length": 160,
                "max_steps": 200,
            },
        )()
        cli.cmd_build_forecasts(fc_args)

        current_stage = "fundamentals_derived"
        logger.info("[bold cyan]Pipeline step 3/7: refresh-fundamentals-derived[/bold cyan]")
        fund_args = type("Args", (), {"lookback_days": 600})()
        try:
            cli.cmd_refresh_fundamentals_derived(fund_args)
        except Exception as exc:
            logger.warning("[yellow]fundamentals derived refresh skipped[/yellow] err=%s", exc)

        current_stage = "opportunity_ranker"
        logger.info("[bold cyan]Pipeline step 4/7: build-opportunity-ranker[/bold cyan]")
        ranker_args = type(
            "Args",
            (),
            {
                "lookback_days": 540,
                "horizon": 20,
                "max_scoring_rows": 500,
                "min_ic_dates": 60,
                "min_valid_signals": 3,
            },
        )()
        cli.cmd_build_opportunity_ranker(ranker_args)

        if market_override.strip():
            if original_market_env is not None:
                os.environ["KIS_TARGET_MARKET"] = original_market_env
            else:
                os.environ.pop("KIS_TARGET_MARKET", None)

        current_stage = "agent_cycle"
        logger.info("[bold cyan]Pipeline step 5/7: run-agent-cycle[/bold cyan]")
        cli.cmd_run_agent_cycle(live=live, all_tenants=all_tenants, market_override=market_override)

        logger.info("[bold cyan]Pipeline step 6/7: mtm-score-update[/bold cyan]")
        try:
            cli._run_mtm_score_update(bootstrap_settings)
        except Exception as exc:
            logger.warning("[yellow]MTM score update failed (non-fatal): %s[/yellow]", exc)

        now_utc = cli.utc_now()
        if now_utc.weekday() == 0:
            logger.info("[bold cyan]Pipeline step 7/7: memory-cleanup[/bold cyan]")
            try:
                cli._run_memory_cleanup(bootstrap_settings)
            except Exception as exc:
                logger.warning("[yellow]Memory cleanup failed (non-fatal): %s[/yellow]", exc)
        else:
            logger.info("[dim]Pipeline step 7/7: memory-cleanup — skipped (not Monday)[/dim]")

        logger.info("[bold green]Pipeline done[/bold green]")
    except Exception as exc:
        if current_stage != "agent_cycle":
            cli._append_tenant_run_status_many(
                bootstrap_repo,
                bootstrap_settings,
                tenants=pipeline_targets,
                run_ids=pipeline_run_ids,
                run_type="pipeline",
                status="failed",
                reason_code=f"{current_stage}_failed",
                stage=current_stage,
                finished_at=cli.utc_now(),
                message=str(exc),
                detail={"error": str(exc), "live": bool(live), "market_override": market_override or None},
            )
        raise
    finally:
        if market_override.strip():
            if original_market_env is not None:
                os.environ["KIS_TARGET_MARKET"] = original_market_env
            else:
                os.environ.pop("KIS_TARGET_MARKET", None)


def cmd_run_batch(live: bool, *, all_tenants: bool = False, market_override: str = "") -> None:
    """Runs data sync + trading cycle; optionally across all runtime tenants."""
    cli = _cli()
    if market_override.strip() and not all_tenants:
        os.environ["KIS_TARGET_MARKET"] = _MARKET_ALIAS.get(market_override.strip().lower(), market_override.strip().lower())
        logger.info("[cyan]Market override[/cyan] --market=%s", market_override)

    if not all_tenants:
        tenant = cli._tenant_id() or "local"
        settings, repo, orchestrator = cli._build_runtime(
            live=live,
            require_kis=True,
            tenant_id=tenant,
            execution_market=market_override,
        )
        cli._run_batch_once(live, settings=settings, repo=repo, orchestrator=orchestrator, tenant=tenant)
        return

    bootstrap_settings = cli.load_settings()
    cli.configure_logging(bootstrap_settings.log_level, bootstrap_settings.log_format)
    cli._validate_or_exit(bootstrap_settings)
    bootstrap_repo = cli._repo_or_exit(bootstrap_settings, tenant_id="local")
    bootstrap_repo.ensure_dataset()
    bootstrap_repo.ensure_tables()

    fallback = cli._tenant_id() or "local"
    tenants = cli._resolve_batch_tenants(bootstrap_repo, fallback=fallback)
    tenants = cli._partition_tenants_for_task(tenants)
    if not tenants:
        logger.info(
            "[yellow]No tenants assigned to this task shard; skipping[/yellow]",
            extra=event_extra(
                "batch_task_shard_empty",
                market=market_override or "all",
            ),
        )
        return
    logger.info(
        "[bold]Batch multi-tenant start[/bold] tenants=%s live=%s",
        ",".join(tenants),
        live,
        extra=event_extra(
            "batch_multi_tenant_start",
            live=live,
            market=market_override or "all",
            tenant_count=len(tenants),
            tenants=tenants,
        ),
    )

    max_workers = int(os.getenv("ARENA_BATCH_PARALLEL", "3") or "3")
    build_failures: list[tuple[str, str]] = []
    runtimes: list[tuple[str, Settings, BigQueryRepository, ArenaOrchestrator]] = []

    def _build_batch_tenant(tenant_id: str):
        return tenant_id, *cli._build_runtime(
            live=live,
            require_kis=True,
            tenant_id=tenant_id,
            require_tenant_runtime_credentials=True,
            execution_market=market_override,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_build_batch_tenant, tenant_id): tenant_id for tenant_id in tenants}
        for future in as_completed(futures):
            tenant = futures[future]
            try:
                runtimes.append(future.result())
            except SystemExit as exc:
                build_failures.append((tenant, f"SystemExit({exc.code})"))
                logger.exception(
                    "[red]Batch tenant build blocked[/red] tenant=%s code=%s",
                    tenant,
                    exc.code,
                    extra=failure_extra(
                        "batch_tenant_build_blocked",
                        exc,
                        tenant_id=tenant,
                        market=market_override or "all",
                        stage="runtime",
                    ),
                )
            except Exception as exc:
                build_failures.append((tenant, str(exc)))
                logger.exception(
                    "[red]Batch tenant build failed[/red] tenant=%s err=%s",
                    tenant,
                    str(exc),
                    extra=failure_extra(
                        "batch_tenant_build_failed",
                        exc,
                        tenant_id=tenant,
                        market=market_override or "all",
                        stage="runtime",
                    ),
                )

    if not runtimes:
        logger.error(
            "[red]Batch multi-tenant failed[/red] no tenants initialized",
            extra=event_extra(
                "batch_multi_tenant_failed",
                reason="no_tenants_initialized",
                live=live,
                market=market_override or "all",
                tenant_count=len(tenants),
                runtime_count=0,
                build_failed_count=len(build_failures),
                execution_failed_count=0,
            ),
        )
        raise SystemExit(1)

    if market_override.strip():
        pre_count = len(runtimes)
        runtimes = [
            (tenant, settings, repo, orchestrator)
            for tenant, settings, repo, orchestrator in runtimes
            if cli._market_filter_matches(settings, market_override)
        ]
        skipped = pre_count - len(runtimes)
        if skipped:
            logger.info(
                "[cyan]Market filter[/cyan] --market=%s matched=%d skipped=%d",
                market_override,
                len(runtimes),
                skipped,
                extra=event_extra(
                    "batch_market_filter",
                    market=market_override,
                    matched=len(runtimes),
                    skipped=skipped,
                ),
            )
        if not runtimes:
            logger.info(
                "[yellow]No tenants match --market=%s; skipping[/yellow]",
                market_override,
                extra=event_extra(
                    "batch_market_filter_empty",
                    market=market_override,
                ),
            )
            return

    _, first_settings, first_repo, _ = runtimes[0]
    phase, window = cli._batch_phase(live, first_settings, first_repo)
    cli._batch_market_sync(phase, first_settings, first_repo, window)

    if phase == "seed" or phase is None:
        if phase == "seed":
            logger.info(
                "[bold green]Batch multi-tenant done[/bold green] mode=seed tenants=%d",
                len(runtimes),
                extra=event_extra(
                    "batch_multi_tenant_done",
                    mode="seed",
                    live=live,
                    market=market_override or "all",
                    tenant_count=len(tenants),
                    runtime_count=len(runtimes),
                ),
            )
        return

    exec_failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                cli._batch_tenant_work,
                phase,
                live,
                settings=settings,
                repo=repo,
                orchestrator=orchestrator,
                tenant=tenant,
                window=window,
            ): tenant
            for tenant, settings, repo, orchestrator in runtimes
        }
        for future in as_completed(futures):
            tenant = futures[future]
            try:
                future.result()
            except SystemExit as exc:
                exec_failures.append((tenant, f"SystemExit({exc.code})"))
                logger.exception(
                    "[red]Batch tenant blocked[/red] tenant=%s code=%s",
                    tenant,
                    exc.code,
                    extra=failure_extra(
                        "batch_tenant_blocked",
                        exc,
                        tenant_id=tenant,
                        market=market_override or "all",
                        stage="execution",
                        phase=phase,
                    ),
                )
            except Exception as exc:
                exec_failures.append((tenant, str(exc)))
                logger.exception(
                    "[red]Batch tenant failed[/red] tenant=%s err=%s",
                    tenant,
                    str(exc),
                    extra=failure_extra(
                        "batch_tenant_failed",
                        exc,
                        tenant_id=tenant,
                        market=market_override or "all",
                        stage="execution",
                        phase=phase,
                    ),
                )

    failures = build_failures + exec_failures
    if failures:
        logger.error(
            "[red]Batch multi-tenant completed with failures[/red] failed=%s",
            ", ".join([f"{tenant}:{error}" for tenant, error in failures]),
            extra=event_extra(
                "batch_multi_tenant_completed_with_failures",
                live=live,
                market=market_override or "all",
                phase=phase,
                tenant_count=len(tenants),
                runtime_count=len(runtimes),
                build_failed_count=len(build_failures),
                execution_failed_count=len(exec_failures),
                failed_count=len(failures),
                failed_tenants=[tenant for tenant, _ in failures],
            ),
        )
        raise SystemExit(1)
    logger.info(
        "[bold green]Batch multi-tenant done[/bold green] tenants=%d",
        len(runtimes),
        extra=event_extra(
            "batch_multi_tenant_done",
            live=live,
            market=market_override or "all",
            phase=phase,
            tenant_count=len(tenants),
            runtime_count=len(runtimes),
        ),
    )
