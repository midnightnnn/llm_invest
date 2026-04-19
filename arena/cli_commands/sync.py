from __future__ import annotations

import logging
from math import ceil

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.market_feature_normalization import (
    daily_history_sources,
    normalize_market_feature_rows_from_closes,
)
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
            sources=daily_history_sources(sources),
        ) or {}
        latest_rows = normalize_market_feature_rows_from_closes(latest_rows, closes)
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


def cmd_build_opportunity_ranker(args: object) -> None:
    """Builds the learned opportunity ranker and writes precomputed scores."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)

    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()

    from arena.recommendation import build_and_store_opportunity_ranker

    result = build_and_store_opportunity_ranker(
        repo,
        settings,
        lookback_days=max(120, int(getattr(args, "lookback_days", 540))),
        horizon_days=max(5, int(getattr(args, "horizon", 20))),
        max_scoring_rows=max(50, int(getattr(args, "max_scoring_rows", 500))),
        min_ic_dates=max(20, int(getattr(args, "min_ic_dates", 60))),
        min_valid_signals=max(1, int(getattr(args, "min_valid_signals", 3))),
    )
    log_fn = logger.info if result.status == "ok" else logger.warning
    log_fn(
        "[bold %s]Opportunity ranker build %s[/bold %s] version=%s status=%s training=%d validation=%d scoring=%d written=%d ic=%s hit=%s note=%s",
        "green" if result.status == "ok" else "yellow",
        "finished" if result.status == "ok" else "degraded",
        "green" if result.status == "ok" else "yellow",
        result.ranker_version or "-",
        result.status,
        int(result.training_rows),
        int(result.validation_rows),
        int(result.scoring_rows),
        int(result.scores_written),
        "-" if result.oos_ic_20d is None else f"{result.oos_ic_20d:.4f}",
        "-" if result.oos_hit_rate_20d is None else f"{result.oos_hit_rate_20d:.4f}",
        result.note or "-",
    )


def _signal_refresh_bootstrap() -> tuple[object, object, object]:
    """Loads settings/repo/market for signal-refresh debug CLIs."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()
    return cli, settings, repo


def cmd_refresh_signals(args: object) -> None:
    """Recomputes signal_daily_values (debug)."""
    from arena.market_sources import live_market_sources_for_markets, parse_markets

    _, settings, repo = _signal_refresh_bootstrap()
    market = str(settings.kis_target_market or "").strip().lower()
    sources = live_market_sources_for_markets(parse_markets(settings.kis_target_market)) or None
    repo.refresh_signal_daily_values(
        lookback_days=max(40, int(getattr(args, "lookback_days", 540))),
        horizon_days=max(5, int(getattr(args, "horizon", 20))),
        sources=sources,
        market=market,
    )
    logger.info("[bold green]refresh-signals done[/bold green] market=%s", market)


def cmd_refresh_signal_ic(args: object) -> None:
    """Recomputes signal_daily_ic (debug)."""
    _, settings, repo = _signal_refresh_bootstrap()
    market = str(settings.kis_target_market or "").strip().lower()
    repo.refresh_signal_daily_ic(
        lookback_days=max(40, int(getattr(args, "lookback_days", 540))),
        horizon_days=max(5, int(getattr(args, "horizon", 20))),
        market=market,
    )
    logger.info("[bold green]refresh-signal-ic done[/bold green] market=%s", market)


def cmd_refresh_regime_features(args: object) -> None:
    """Recomputes regime_daily_features (debug)."""
    _, settings, repo = _signal_refresh_bootstrap()
    market = str(settings.kis_target_market or "").strip().lower()
    repo.refresh_regime_daily_features(
        lookback_days=max(40, int(getattr(args, "lookback_days", 540))),
        market=market,
    )
    logger.info("[bold green]refresh-regime-features done[/bold green] market=%s", market)


def _load_backfill_tickers(args: object) -> list[str]:
    raw = getattr(args, "tickers", None)
    if raw:
        text = str(raw).strip()
        if text:
            return [t.strip().upper() for t in text.split(",") if t.strip()]
    path_val = getattr(args, "tickers_file", None)
    if path_val:
        try:
            with open(str(path_val), encoding="utf-8") as fh:
                return [line.strip().upper() for line in fh if line.strip()]
        except OSError as exc:
            logger.warning("ticker file read failed path=%s err=%s", path_val, exc)
    return []


def cmd_fundamentals_backfill_kr(args: object) -> None:
    """Runs the KIS-based fundamentals backfill for Korean tickers."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()
    from arena.open_trading.client import OpenTradingClient
    from arena.open_trading.kis_fundamentals_ingestor import KISFundamentalsIngestor

    tickers = _load_backfill_tickers(args)
    if not tickers:
        session = getattr(repo, "session", None)
        if session is not None and hasattr(session, "fetch_rows"):
            try:
                rows = session.fetch_rows(
                    f"""
                    SELECT DISTINCT ticker
                    FROM `{session.dataset_fqn}.universe_candidates`
                    WHERE SAFE_CAST(ticker AS INT64) IS NOT NULL
                      AND LENGTH(ticker) = 6
                      AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
                    """,
                    {"days": 14},
                ) or []
                tickers = sorted({str(r.get("ticker") or "").strip() for r in rows if r.get("ticker")})
            except Exception as exc:
                logger.warning("KR universe query failed err=%s", exc)
    if not tickers:
        logger.warning("[yellow]no tickers supplied for KR backfill[/yellow]")
        return

    client = OpenTradingClient(settings=settings)
    ingestor = KISFundamentalsIngestor(
        client=client,
        repo=repo,
        div_cls_code=str(getattr(args, "period", "quarter") or "quarter").lower() == "annual" and "0" or "1",
    )
    logger.info("[cyan]KR fundamentals backfill start[/cyan] tickers=%d period=%s", len(tickers), getattr(args, "period", "quarter"))
    result = ingestor.run(tickers=tickers, market="kospi")
    logger.info(
        "[bold green]KR fundamentals backfill %s[/bold green] run=%s attempted=%d succeeded=%d quarters=%d",
        result.status,
        result.run_id,
        result.tickers_attempted,
        result.tickers_succeeded,
        result.quarters_inserted,
    )


def cmd_fundamentals_backfill_us(args: object) -> None:
    """Runs the SEC/FMP-based fundamentals backfill for US tickers."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    import os
    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()

    tickers = _load_backfill_tickers(args)
    if not tickers:
        session = getattr(repo, "session", None)
        if session is not None and hasattr(session, "fetch_rows"):
            try:
                rows = session.fetch_rows(
                    f"""
                    SELECT DISTINCT ticker
                    FROM `{session.dataset_fqn}.universe_candidates`
                    WHERE SAFE_CAST(ticker AS INT64) IS NULL
                      AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
                    """,
                    {"days": 14},
                ) or []
                tickers = sorted({str(r.get("ticker") or "").strip().upper() for r in rows if r.get("ticker")})
            except Exception as exc:
                logger.warning("US universe query failed err=%s", exc)
    if not tickers:
        logger.warning("[yellow]no tickers supplied for US backfill[/yellow]")
        return

    source = str(getattr(args, "source", "sec") or "sec").strip().lower()
    logger.info("[cyan]US fundamentals backfill start[/cyan] source=%s tickers=%d period=%s", source, len(tickers), getattr(args, "period", "quarter"))
    if source == "fmp":
        api_key = str(getattr(args, "api_key", "") or os.environ.get("FMP_API_KEY", "") or "").strip()
        if not api_key:
            logger.error("[red]FMP_API_KEY env var or --api-key required for --source fmp[/red]")
            raise SystemExit(2)
        from arena.open_trading.fmp_fundamentals_ingestor import FMPFundamentalsIngestor

        ingestor = FMPFundamentalsIngestor(
            api_key=api_key,
            repo=repo,
            period=str(getattr(args, "period", "quarter") or "quarter"),
            limit=int(getattr(args, "limit", 40) or 40),
        )
    else:
        from arena.open_trading.sec_fundamentals_ingestor import SECFundamentalsIngestor

        sec_user_agent = str(getattr(args, "sec_user_agent", "") or os.environ.get("SEC_USER_AGENT", "") or "").strip()
        if not sec_user_agent:
            contact = str(os.environ.get("ARENA_OPERATOR_EMAILS", "") or "").split(",")[0].strip()
            if contact:
                sec_user_agent = f"LLM Arena {contact}"
        if not sec_user_agent:
            logger.error("[red]SEC_USER_AGENT or ARENA_OPERATOR_EMAILS required for --source sec[/red]")
            raise SystemExit(2)
        ingestor = SECFundamentalsIngestor(
            repo=repo,
            user_agent=sec_user_agent,
            sleep_seconds=float(getattr(args, "sleep_seconds", 0.15) or 0.15),
        )
    result = ingestor.run(tickers=tickers, market="us")
    logger.info(
        "[bold green]US fundamentals backfill %s[/bold green] run=%s attempted=%d succeeded=%d quarters=%d",
        result.status,
        result.run_id,
        result.tickers_attempted,
        result.tickers_succeeded,
        result.quarters_inserted,
    )


def cmd_fundamentals_coverage(args: object) -> None:
    """Prints a fundamentals coverage report by market/year."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = cli._repo_or_exit(settings)
    session = getattr(repo, "session", None)
    if session is None or not hasattr(session, "fetch_rows"):
        logger.error("repo does not expose BigQuery session for coverage report")
        return
    sql = f"""
    SELECT
      market,
      MIN(fiscal_year) AS min_year,
      MAX(fiscal_year) AS max_year,
      COUNT(DISTINCT ticker) AS tickers,
      COUNT(*) AS rows,
      COUNT(DISTINCT CONCAT(CAST(fiscal_year AS STRING), '-Q', CAST(fiscal_quarter AS STRING))) AS distinct_periods
    FROM `{session.dataset_fqn}.fundamentals_history_raw`
    GROUP BY market
    ORDER BY market
    """
    rows = session.fetch_rows(sql, {}) or []
    logger.info("[bold]Fundamentals coverage[/bold]")
    for row in rows:
        logger.info(
            "  market=%s years=%s~%s tickers=%s rows=%s periods=%s",
            row.get("market"),
            row.get("min_year"),
            row.get("max_year"),
            row.get("tickers"),
            row.get("rows"),
            row.get("distinct_periods"),
        )


def cmd_refresh_fundamentals_derived(args: object) -> None:
    """Recomputes fundamentals_derived_daily for a lookback window."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)
    repo = cli._repo_or_exit(settings)
    repo.ensure_dataset()
    repo.ensure_tables()
    market = str(settings.kis_target_market or "").strip().lower()
    repo.refresh_fundamentals_derived_daily(
        lookback_days=max(40, int(getattr(args, "lookback_days", 600))),
        market=market,
    )
    logger.info("[bold green]refresh-fundamentals-derived done[/bold green] market=%s", market)


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
