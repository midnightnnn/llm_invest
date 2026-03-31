from __future__ import annotations

import logging
from math import ceil

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.market_sources import live_market_sources_for_markets, parse_markets
from arena.runtime_universe import resolve_runtime_universe
from arena.tools.screening import DISCOVERY_BUCKETS, build_discovery_rows, momentum_scores

logger = logging.getLogger(__name__)


def _cli():
    import arena.cli as cli

    return cli


def cmd_init_bq() -> None:
    """Initializes the arena dataset and all runtime tables."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()
    logger.info("[green]BigQuery bootstrap completed[/green] dataset=%s.%s", settings.google_cloud_project, settings.bq_dataset)


def cmd_seed_demo_market() -> None:
    """Seeds market_features with deterministic demo rows for immediate testing."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()
    rows = cli._seed_rows(settings)
    repo.insert_market_features(rows)
    logger.info("[green]Seeded demo market rows[/green] count=%d", len(rows))


def _prepare_kis_command_repo(settings: Settings) -> BigQueryRepository:
    """Builds a tenant-scoped repo and applies runtime KIS overrides before validation."""
    cli = _cli()
    tenant = cli._tenant_id() or "local"
    repo = cli._repo_or_exit(settings, tenant_id=tenant)
    repo.ensure_dataset()
    repo.ensure_tables()
    cli._apply_tenant_runtime_credentials(settings, repo, tenant_id=tenant)
    cli.apply_runtime_overrides(settings, repo, tenant_id=tenant)
    cli._validate_or_exit(settings, require_kis=True)
    return repo


def cmd_sync_market() -> None:
    """Pulls market data from open-trading API and inserts feature rows."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = _prepare_kis_command_repo(settings)
    result = cli.MarketDataSyncService(settings=settings, repo=repo).sync_market_features()
    logger.info(
        "[bold green]Market sync finished[/bold green] inserted=%d attempted=%d failed=%d",
        result.inserted_rows,
        result.attempted_tickers,
        len(result.failed_tickers),
    )


def cmd_sync_market_quotes() -> None:
    """Pulls intraday quotes from open-trading API and inserts hourly feature rows."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = _prepare_kis_command_repo(settings)
    result = cli.MarketDataSyncService(settings=settings, repo=repo).sync_market_quotes()
    logger.info(
        "[bold green]Quote sync finished[/bold green] inserted=%d attempted=%d failed=%d",
        result.inserted_rows,
        result.attempted_tickers,
        len(result.failed_tickers),
    )


def cmd_sync_account() -> None:
    """Pulls live account snapshot from open-trading API into BigQuery."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = _prepare_kis_command_repo(settings)
    snapshot = cli.AccountSyncService(settings=settings, repo=repo).sync_account_snapshot()
    logger.info(
        "[bold green]Account sync finished[/bold green] cash=%.0f equity=%.0f positions=%d",
        snapshot.cash_krw,
        snapshot.total_equity_krw,
        len(snapshot.positions),
    )


def cmd_sync_broker_trades(*, days: int = 7) -> None:
    """Pulls raw broker trade events from KIS inquiry APIs into BigQuery."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = _prepare_kis_command_repo(settings)
    result = cli.BrokerTradeSyncService(settings=settings, repo=repo).sync_broker_trade_events(days=days)
    logger.info(
        "[bold green]Broker trade sync finished[/bold green] inserted=%d scanned=%d skipped=%d failed_scopes=%d",
        result.inserted_events,
        result.scanned_rows,
        result.skipped_existing,
        len(result.failed_scopes),
    )


def cmd_sync_broker_cash(*, days: int = 7) -> None:
    """Pulls signed broker cash settlement events from KIS inquiry APIs into BigQuery."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = _prepare_kis_command_repo(settings)
    result = cli.BrokerCashSyncService(settings=settings, repo=repo).sync_broker_cash_events(days=days)
    logger.info(
        "[bold green]Broker cash sync finished[/bold green] inserted=%d scanned=%d skipped=%d failed_scopes=%d",
        result.inserted_events,
        result.scanned_rows,
        result.skipped_existing,
        len(result.failed_scopes),
    )


def cmd_sync_dividends() -> None:
    """Discovers overseas dividends and attributes them to agent sleeves."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    if settings.trading_mode != "live":
        logger.info("[cyan]Dividend sync skipped[/cyan] mode=%s (live only)", settings.trading_mode)
        return
    if not settings.dividend_sync_enabled:
        logger.info("[cyan]Dividend sync disabled by config[/cyan]")
        return
    repo = _prepare_kis_command_repo(settings)
    result = cli.DividendSyncService(settings=settings, repo=repo).sync_dividends()
    logger.info(
        "[bold green]Dividend sync finished[/bold green] tickers=%d found=%d inserted=%d skipped=%d cash=%d",
        result.tickers_checked,
        result.dividends_found,
        result.events_inserted,
        result.skipped_duplicate,
        result.broker_cash_events_inserted,
    )


def _build_forecast_tickers(repo, settings: Settings, top_n: int) -> list[str]:
    """Selects discovery-bucket candidates + current holdings for forecast computation."""
    sources = live_market_sources_for_markets(parse_markets(settings.kis_target_market)) or None
    universe = resolve_runtime_universe(settings, repo=repo)

    latest_rows: list[dict] = []
    latest_loader = getattr(repo, "latest_market_features", None)
    if callable(latest_loader) and universe:
        latest_rows = latest_loader(
            tickers=universe,
            limit=max(50, len(universe)),
            sources=sources,
        ) or []

    latest_tickers = [
        str(row.get("ticker") or "").strip().upper()
        for row in latest_rows
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    ]

    momentum_rows: list[dict] = []
    closes_loader = getattr(repo, "get_daily_closes", None)
    if callable(closes_loader) and latest_tickers:
        closes = closes_loader(
            tickers=latest_tickers,
            lookback_days=128,
            sources=sources,
        ) or {}
        momentum_rows = momentum_scores(closes, windows=[20, 60, 126], vol_adjust=True) if closes else []

    fundamentals_rows: list[dict] = []
    fundamentals_loader = getattr(repo, "latest_fundamentals_snapshot", None)
    if callable(fundamentals_loader) and latest_tickers:
        fundamentals_rows = fundamentals_loader(
            tickers=latest_tickers,
            limit=max(50, len(latest_tickers)),
        ) or []

    bucket_cap = max(1, int(ceil(max(1, int(top_n)) / max(1, len(DISCOVERY_BUCKETS)))))
    discovery_tickers: list[str] = []
    bucket_counts: dict[str, int] = {}
    for bucket_name in DISCOVERY_BUCKETS:
        bucket_rows = build_discovery_rows(
            latest_rows,
            momentum_rows=momentum_rows,
            fundamentals_rows=fundamentals_rows,
            bucket=bucket_name,
            top_n=bucket_cap,
        )
        bucket_tokens = [
            str(row.get("ticker") or "").strip().upper()
            for row in bucket_rows
            if isinstance(row, dict) and str(row.get("ticker") or "").strip()
        ]
        bucket_counts[bucket_name] = len(bucket_tokens)
        for ticker in bucket_tokens:
            if ticker not in discovery_tickers:
                discovery_tickers.append(ticker)

    held_tickers: list[str] = []
    try:
        held_tickers = repo.get_latest_position_tickers(
            market=settings.kis_target_market,
            all_tenants=True,
        )
    except Exception as exc:
        logger.warning("[yellow]Failed to load held tickers for forecast[/yellow] err=%s", str(exc))

    combined = list(dict.fromkeys(discovery_tickers + held_tickers))
    logger.info(
        "[cyan]Forecast ticker selection[/cyan] bucket_cap=%d discovery=%d held=%d combined=%d buckets=%s",
        bucket_cap,
        len(discovery_tickers),
        len(held_tickers),
        len(combined),
        ",".join(f"{name}:{bucket_counts.get(name, 0)}" for name in DISCOVERY_BUCKETS),
    )
    return combined


def cmd_build_forecasts(args: object) -> None:
    """Builds stacked forecast rows and writes predicted_expected_returns."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)

    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()

    top_n = max(10, int(getattr(args, "top_n", 50)))
    forecast_tickers = _build_forecast_tickers(repo, settings, top_n=top_n)

    from arena.forecasting import build_and_store_stacked_forecasts

    result = build_and_store_stacked_forecasts(
        repo,
        settings,
        lookback_days=max(180, int(getattr(args, "lookback_days", 360))),
        horizon=max(5, int(getattr(args, "horizon", 20))),
        min_series_length=max(80, int(getattr(args, "min_series_length", 160))),
        max_steps=max(50, int(getattr(args, "max_steps", 200))),
        tickers=forecast_tickers or None,
    )
    logger.info(
        "[bold green]Forecast build finished[/bold green] run_date=%s rows=%d tickers=%d used_neuralforecast=%s models=%s note=%s",
        result.run_date,
        int(result.rows_written),
        int(result.tickers_used),
        str(bool(result.used_neuralforecast)).lower(),
        ",".join(result.model_names),
        result.note,
    )


def cmd_list_strategies() -> None:
    """Prints strategy reference cards for quick inspection."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    cards = cli.list_cards()
    for card in cards:
        logger.info(
            "[cyan]%s[/cyan] | %s | %s",
            card["strategy_id"],
            card["category"],
            card["name"],
        )


def cmd_recover_sleeves(*, live: bool = False) -> None:
    """Runs deterministic checkpoint recovery, then re-runs reconciliation."""
    cli = _cli()
    tenant = cli._tenant_id() or "local"
    settings, repo, orchestrator = cli._build_runtime(live=live, require_kis=live, tenant_id=tenant)
    agent_ids = [
        str(getattr(agent, "agent_id", "") or "").strip()
        for agent in getattr(orchestrator, "agents", [])
        if str(getattr(agent, "agent_id", "") or "").strip()
    ] or list(settings.agent_ids)

    snapshot = None
    if live:
        cli._sync_broker_trade_ledger(live=True, settings=settings, repo=repo, tenant=tenant)
        cli._sync_broker_cash_ledger(live=True, settings=settings, repo=repo, tenant=tenant)
        snapshot = cli.AccountSyncService(settings=settings, repo=repo).sync_account_snapshot()

    recovery = cli.StateRecoveryService(
        settings=settings,
        repo=repo,
        excluded_tickers=cli._reconcile_excluded_tickers(settings),
        qty_tolerance=max(cli._float_env("ARENA_RECONCILE_QTY_TOLERANCE", 1e-9), 0.0),
        cash_tolerance_krw=max(cli._float_env("ARENA_RECONCILE_CASH_TOLERANCE_KRW", 50_000.0), 0.0),
        cash_reconciliation_enabled=cli._truthy_env("ARENA_RECONCILE_CASH_ENABLED", True),
    ).recover_and_reconcile(
        agent_ids=agent_ids,
        tenant_id=tenant,
        include_simulated=not live and settings.trading_mode != "live",
        auto_recover=True,
        account_snapshot=snapshot,
        sync_account_snapshot=cli.AccountSyncService(settings=settings, repo=repo).sync_account_snapshot if live else None,
        created_by="cli_recover_sleeves",
        allow_checkpoint_rebuild=cli._allow_checkpoint_rebuild_recovery(explicit=True),
    )

    logger.info(
        "[cyan]Recover sleeves[/cyan] tenant=%s status=%s checkpoints=%d before=%s after=%s",
        tenant,
        recovery.status,
        recovery.applied_checkpoints,
        recovery.before.status,
        recovery.after.status,
    )
    if recovery.ok:
        return
    raise SystemExit(1)
