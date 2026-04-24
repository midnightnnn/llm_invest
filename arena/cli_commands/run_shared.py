from __future__ import annotations
import json
import logging
import os
from datetime import date, timedelta

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.market_sources import live_market_sources_for_markets
from arena.market_hours import MarketWindow

logger = logging.getLogger(__name__)

_US_MARKET_KEYS = {"nasdaq", "nyse", "amex", "us"}
_KR_MARKET_KEYS = {"kospi", "kosdaq"}
_MARKET_ALIAS: dict[str, str] = {
    "us": "us",
    "nasdaq": "nasdaq",
    "nyse": "nyse",
    "amex": "amex",
    "kospi": "kospi",
    "kosdaq": "kosdaq",
    "kr": "kospi",
    "korea": "kospi",
}


def _cli():
    import arena.cli as cli

    return cli


def _parse_cli_markets(settings: Settings) -> set[str]:
    raw = (settings.kis_target_market or "").lower().strip()
    return {token.strip() for token in raw.split(",") if token.strip()}


def _has_us(markets: set[str]) -> bool:
    return bool(markets & _US_MARKET_KEYS)


def _has_kr(markets: set[str]) -> bool:
    return bool(markets & _KR_MARKET_KEYS)


def _apply_market_override(settings: Settings, market_override: str) -> None:
    """Overrides ``kis_target_market`` when ``--market`` flag is provided."""
    raw = market_override.strip().lower()
    if not raw:
        return
    resolved = _MARKET_ALIAS.get(raw, raw)
    logger.info(
        "[cyan]Market override[/cyan] --market=%s → kis_target_market=%s (was %s)",
        raw,
        resolved,
        settings.kis_target_market,
    )
    settings.kis_target_market = resolved


def _execution_market_key(market_override: str = "", settings: Settings | None = None) -> str:
    """Returns the canonical market key for one scheduled execution."""
    raw = str(market_override or "").strip().lower()
    if not raw and settings is not None:
        raw = str(settings.kis_target_market or "").strip().lower()
        if "," in raw:
            raw = raw.split(",", 1)[0].strip()
    if not raw:
        return ""
    resolved = _MARKET_ALIAS.get(raw, raw)
    if resolved in _US_MARKET_KEYS:
        return "us"
    if resolved in _KR_MARKET_KEYS:
        return "kospi"
    return resolved


def _execution_trading_date(market_key: str) -> date:
    """Returns trading date anchor for one scheduled market execution."""
    cli = _cli()
    clean = _execution_market_key(market_key)
    if clean == "us":
        return cli.nasdaq_window().trading_date
    if clean == "kospi":
        return cli.kospi_window().trading_date
    return cli.utc_now().date()


def _execution_source(default: str = "") -> str:
    """Returns the execution source token used to scope one-off Cloud Run executions."""
    raw = str(os.getenv("ARENA_EXECUTION_SOURCE") or "").strip().lower()
    if raw:
        return raw
    if os.getenv("CLOUD_RUN_JOB"):
        return "manual"
    return str(default or "").strip().lower()


def _cloud_run_execution_body(*, source: str = "") -> dict[str, object]:
    """Builds a Cloud Run RunJob request body with execution-scoped env overrides."""
    clean_source = str(source or "").strip().lower()
    if not clean_source:
        return {}
    return {
        "overrides": {
            "containerOverrides": [
                {
                    "env": [
                        {
                            "name": "ARENA_EXECUTION_SOURCE",
                            "value": clean_source,
                        }
                    ]
                }
            ]
        }
    }


def _daily_history_sources_for_markets(markets: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    """Returns live market sources that carry daily history, excluding quote snapshots."""
    return [
        source
        for source in live_market_sources_for_markets(markets)
        if not str(source or "").endswith("_quote")
    ]


def _probe_daily_coverage(
    repo: BigQueryRepository,
    markets_to_probe: list[str],
    *,
    trading_day: date,
) -> tuple[str | None, int]:
    """Returns the best same-day daily-history coverage across candidate sources."""
    sources = _daily_history_sources_for_markets(markets_to_probe)
    if not sources:
        return None, 0

    coverage_fn = getattr(repo, "market_daily_ticker_coverage", None)
    if not callable(coverage_fn):
        distinct_fn = getattr(repo, "market_source_distinct_tickers", None)
        if not callable(distinct_fn):
            return None, 0
        best_source = None
        best_coverage = 0
        for source in sources:
            coverage = int(distinct_fn(source=source) or 0)
            if coverage > best_coverage:
                best_source = source
                best_coverage = coverage
        return best_source, best_coverage

    best_source = None
    best_coverage = 0
    for source in sources:
        coverage = int(coverage_fn(source=source, day=trading_day) or 0)
        if coverage > best_coverage:
            best_source = source
            best_coverage = coverage
    return best_source, best_coverage


def _probe_recent_daily_coverage(
    repo: BigQueryRepository,
    markets_to_probe: list[str],
    *,
    anchor_day: date,
    lookback_days: int = 7,
) -> tuple[str | None, date | None, int]:
    """Returns the most recent non-empty daily-history coverage within a short lookback window."""
    clean_lookback = max(int(lookback_days or 0), 0)
    for offset in range(clean_lookback + 1):
        probe_day = anchor_day - timedelta(days=offset)
        source, coverage = _probe_daily_coverage(repo, markets_to_probe, trading_day=probe_day)
        if coverage > 0:
            return source, probe_day if source else None, coverage
    return None, None, 0


def _tenant_lease_enabled() -> bool:
    """Enables tenant execution leases by default in Cloud Run jobs only."""
    cli = _cli()
    return cli._truthy_env("ARENA_TENANT_LEASE_ENABLED", default=bool(os.getenv("CLOUD_RUN_JOB")))


def _dispatch_agent_job(settings: Settings, *, job_name: str) -> None:
    """Triggers the downstream Cloud Run agent job after shared prep succeeds."""
    cli = _cli()
    region = str(
        os.getenv("ARENA_CLOUD_RUN_REGION")
        or os.getenv("GOOGLE_CLOUD_REGION")
        or settings.bq_location
        or ""
    ).strip()
    execution_source = cli._execution_source()
    response = cli.run_cloud_run_job(
        project=settings.google_cloud_project,
        region=region,
        job_name=job_name,
        body=cli._cloud_run_execution_body(source=execution_source),
        timeout_seconds=max(5, cli._int_env("ARENA_CLOUD_RUN_DISPATCH_TIMEOUT_SECONDS", 30)),
    )
    logger.info(
        "[green]Agent job dispatched[/green] job=%s region=%s source=%s response=%s",
        job_name,
        region,
        execution_source or "-",
        json.dumps(response or {}, ensure_ascii=False)[:500],
    )


def _market_filter_matches(tenant_settings: Settings, market_filter: str) -> bool:
    """True when tenant's configured market overlaps with the --market schedule filter."""
    raw = market_filter.strip().lower()
    if not raw:
        return True
    resolved = _MARKET_ALIAS.get(raw, raw)
    filter_set = {resolved}
    tenant_set = _parse_cli_markets(tenant_settings)
    if not tenant_set:
        return False
    if _has_us(filter_set) and _has_us(tenant_set):
        return True
    if _has_kr(filter_set) and _has_kr(tenant_set):
        return True
    return False


def _market_value_matches(market_value: str, market_filter: str) -> bool:
    """True when a raw market config value overlaps with a schedule filter."""
    raw_filter = str(market_filter or "").strip().lower()
    if not raw_filter:
        return True
    filter_token = _MARKET_ALIAS.get(raw_filter, raw_filter)
    tenant_tokens = {
        _MARKET_ALIAS.get(token.strip().lower(), token.strip().lower())
        for token in str(market_value or "").split(",")
        if token.strip()
    }
    if not tenant_tokens:
        return False
    if filter_token in _US_MARKET_KEYS:
        return bool(tenant_tokens & _US_MARKET_KEYS)
    if filter_token in _KR_MARKET_KEYS:
        return bool(tenant_tokens & _KR_MARKET_KEYS)
    return filter_token in tenant_tokens


def _filter_tenants_by_market(
    repo: BigQueryRepository,
    tenants: list[str],
    market_filter: str,
) -> list[str]:
    """Filters tenant ids by latest kis_target_market config when available."""
    if not market_filter.strip() or not tenants:
        return list(tenants)
    fetch = getattr(repo, "latest_config_values", None)
    if not callable(fetch):
        raise RuntimeError("latest_config_values is required for market-scoped multi-tenant runs")

    market_map = fetch(config_key="kis_target_market", tenant_ids=tenants) or {}
    matched = [tenant for tenant in tenants if _market_value_matches(str(market_map.get(tenant) or ""), market_filter)]
    logger.info(
        "[cyan]Tenant market prefilter[/cyan] --market=%s matched=%d skipped=%d",
        market_filter,
        len(matched),
        len(tenants) - len(matched),
    )
    return matched


def _batch_phase(
    live: bool,
    settings: Settings,
    repo: BigQueryRepository,
) -> tuple[str | None, MarketWindow | None]:
    """Determines batch phase from market window. Shared across tenants."""
    cli = _cli()
    if not live:
        return "general", None

    markets = _parse_cli_markets(settings)
    has_us = _has_us(markets)
    has_kr = _has_kr(markets)

    if not has_us and not has_kr:
        return "general", None

    now = cli.utc_now()
    try:
        recent_lookback_days = int(os.getenv("ARENA_BATCH_DAILY_FRESHNESS_LOOKBACK_DAYS", "7") or "7")
    except ValueError:
        recent_lookback_days = 7

    if has_kr and not has_us:
        window = cli.kospi_window(now)
        logger.info(
            "[cyan]KOSPI window[/cyan] phase=%s now_kst=%s",
            window.phase,
            window.now_local.strftime("%Y-%m-%d %H:%M"),
            extra={"event": "market_window", "phase": window.phase, "trading_date_kst": window.trading_date.isoformat()},
        )
        daily_source, daily_day, daily_tickers = _probe_recent_daily_coverage(
            repo,
            ["kospi"],
            anchor_day=window.trading_date,
            lookback_days=recent_lookback_days,
        )
        if daily_tickers < 30:
            logger.info(
                "[cyan]Daily history missing or stale; seeding[/cyan] phase=%s source=%s freshest_day=%s coverage=%d lookback_days=%d",
                window.phase,
                daily_source or "-",
                daily_day.isoformat() if daily_day else "-",
                daily_tickers,
                recent_lookback_days,
            )
            return "seed", window

        disable_guard = str(os.getenv("ARENA_KOSPI_DISABLE_SCHEDULE_GUARD", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
        if window.phase == "OPEN" or disable_guard:
            from arena.market_hours import format_local_times, parse_local_times, should_run_scheduled_cycle

            times = parse_local_times(os.getenv("ARENA_KOSPI_CYCLE_TIMES_KST"), default=["14:30"])
            try:
                tolerance = int(os.getenv("ARENA_KOSPI_CYCLE_TOLERANCE_MINUTES", "20") or "20")
            except ValueError:
                tolerance = 20
            if disable_guard:
                logger.warning("[yellow]KOSPI schedule guard disabled[/yellow] phase=%s", window.phase)
            elif not should_run_scheduled_cycle(window, times_local=times, tolerance_minutes=tolerance):
                logger.info(
                    "[cyan]KOSPI not scheduled; skipping[/cyan] now_kst=%s schedule=%s",
                    window.now_local.strftime("%H:%M"),
                    format_local_times(times),
                )
                return None, window
            return "open_cycle", window

        if cli.is_report_window(window):
            return "report", window
        return "closed", window

    window = cli.nasdaq_window(now)
    logger.info(
        "[cyan]Market window[/cyan] phase=%s now_et=%s open_utc=%s close_utc=%s",
        window.phase,
        window.now_local.strftime("%Y-%m-%d %H:%M"),
        window.open_utc.isoformat(),
        window.close_utc.isoformat(),
        extra={"event": "market_window", "phase": window.phase, "trading_date_et": window.trading_date.isoformat()},
    )

    daily_source, daily_day, daily_tickers = _probe_recent_daily_coverage(
        repo,
        sorted(markets & _US_MARKET_KEYS),
        anchor_day=window.trading_date,
        lookback_days=recent_lookback_days,
    )
    if daily_tickers < 80:
        logger.info(
            "[cyan]Daily history missing or stale; seeding[/cyan] phase=%s source=%s freshest_day=%s coverage=%d lookback_days=%d",
            window.phase,
            daily_source or "-",
            daily_day.isoformat() if daily_day else "-",
            daily_tickers,
            recent_lookback_days,
        )
        return "seed", window

    disable_schedule_guard = str(os.getenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
    if window.phase == "OPEN" or disable_schedule_guard:
        from arena.market_hours import format_local_times, parse_local_times, should_run_scheduled_cycle

        times = parse_local_times(os.getenv("ARENA_NASDAQ_CYCLE_TIMES_ET"), default=["15:30"])
        try:
            tolerance = int(os.getenv("ARENA_NASDAQ_CYCLE_TOLERANCE_MINUTES", "20") or "20")
        except ValueError:
            tolerance = 20

        if disable_schedule_guard:
            logger.warning(
                "[yellow]Schedule guard disabled; forcing cycle[/yellow] phase=%s now_et=%s",
                window.phase,
                window.now_local.strftime("%H:%M"),
            )
        elif not should_run_scheduled_cycle(window, times_local=times, tolerance_minutes=tolerance):
            logger.info(
                "[cyan]Not a scheduled cycle; skipping[/cyan] now_et=%s schedule=%s tol_min=%d",
                window.now_local.strftime("%H:%M"),
                format_local_times(times),
                tolerance,
            )
            return None, window

        return "open_cycle", window

    if cli.is_report_window(window):
        return "report", window
    return "closed", window


def _live_agent_cycle_market_closed(settings: Settings) -> bool:
    """True when all configured markets are closed for live agent execution."""
    cli = _cli()
    markets = _parse_cli_markets(settings)
    has_us = _has_us(markets)
    has_kr = _has_kr(markets)
    if not has_us and not has_kr:
        return False

    us_closed = False
    kr_closed = False

    if has_us:
        us_win = cli.nasdaq_window()
        disable_us_guard = str(os.getenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
        us_closed = not disable_us_guard and (us_win.now_local.weekday() >= 5 or cli.is_nasdaq_holiday(us_win.trading_date))

    if has_kr:
        kr_win = cli.kospi_window()
        disable_kr_guard = str(os.getenv("ARENA_KOSPI_DISABLE_SCHEDULE_GUARD", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
        kr_closed = not disable_kr_guard and cli.is_kospi_holiday(kr_win.trading_date)

    return (not has_us or us_closed) and (not has_kr or kr_closed)


def _batch_market_sync(
    phase: str | None,
    settings: Settings,
    repo: BigQueryRepository,
    window: MarketWindow | None,
) -> object | None:
    """Runs shared market data sync once based on batch phase."""
    cli = _cli()
    market_service = cli.MarketDataSyncService(settings=settings, repo=repo)

    if phase in ("seed", "general"):
        return market_service.sync_market_features()
    if phase == "open_cycle":
        return market_service.sync_market_quotes()
    if phase == "report" and window is not None:
        markets = _parse_cli_markets(settings)
        if _has_kr(markets) and not _has_us(markets):
            probe_markets = ["kospi"]
            threshold = 30
        else:
            probe_markets = sorted(markets & _US_MARKET_KEYS)
            threshold = 80
        source, coverage = _probe_daily_coverage(repo, probe_markets, trading_day=window.trading_date)
        if coverage < threshold:
            logger.info(
                "[cyan]Daily sync required[/cyan] date=%s source=%s coverage=%d",
                window.trading_date.isoformat(),
                source or "-",
                coverage,
            )
            return market_service.sync_market_features()
    return None


def _run_mtm_score_update(settings: Settings) -> int:
    """BUY 기억의 score를 미실현 손익 기준으로 업데이트."""
    import math
    from google.cloud import firestore as firestore

    cli = _cli()
    repo = cli.BigQueryRepository(project=settings.google_cloud_project, dataset=settings.bq_dataset, location=settings.bq_location)
    db = firestore.Client(project=settings.google_cloud_project)
    updated = 0

    for agent_id in settings.agent_ids:
        snapshot, _, _ = repo.build_agent_sleeve_snapshot(agent_id=agent_id, include_simulated=(settings.trading_mode != "live"))
        if not snapshot.positions:
            continue
        for ticker, position in snapshot.positions.items():
            market_price = position.market_price_krw
            if market_price <= 0:
                continue
            buy_memories = repo.find_buy_memories_for_ticker(agent_id=agent_id, ticker=ticker, limit=10, trading_mode=settings.trading_mode)
            for memory in buy_memories:
                buy_price = cli.MemoryStore._extract_buy_price(memory)
                if buy_price <= 0:
                    continue
                pnl_ratio = (market_price - buy_price) / buy_price
                new_score = max(0.1, min(0.5 + 0.5 * math.tanh(pnl_ratio * 3), 1.0))
                old_score = float(memory.get("score") or 1.0)
                if abs(new_score - old_score) < 0.02:
                    continue
                event_id = str(memory.get("event_id", "")).strip()
                repo.update_memory_score(event_id, new_score)
                try:
                    db.collection("agent_memories").document(event_id).update({"score": float(new_score)})
                except Exception as fs_exc:
                    logger.warning("Firestore sync failed for %s: %s", event_id[:8], fs_exc)
                logger.info("[MTM] %s (%s) score: %.2f → %.2f (PnL: %+.1f%%)", event_id[:8], ticker, old_score, new_score, pnl_ratio * 100)
                updated += 1

    logger.info("MTM score update complete: %d memories updated", updated)
    return updated


def _run_memory_cleanup(settings: Settings) -> int:
    """Runs policy-driven cleanup across discovered runtime tenants."""
    cli = _cli()
    repo = cli._repo_or_exit(settings, tenant_id=cli._tenant_id() or "local")
    summary = cli.run_memory_cleanup_for_all_tenants(repo, settings, require_enabled=True)
    for tenant_result in summary.get("tenants", []):
        tenant = str(tenant_result.get("tenant_id") or "local")
        if not bool(tenant_result.get("enabled")):
            logger.info("[dim]Memory cleanup skipped[/dim] tenant=%s reason=%s", tenant, str(tenant_result.get("reason") or "disabled"))
            continue
        logger.info(
            "Memory cleanup tenant=%s candidates=%d deleted_bq=%d deleted_fs=%d",
            tenant,
            int(tenant_result.get("candidate_count") or 0),
            int(tenant_result.get("deleted_bigquery") or 0),
            int(tenant_result.get("deleted_firestore") or 0),
        )
        if tenant_result.get("firestore_error"):
            logger.warning(
                "[yellow]Memory cleanup Firestore warning[/yellow] tenant=%s err=%s",
                tenant,
                str(tenant_result.get("firestore_error") or ""),
            )
    logger.info(
        "Memory cleanup complete: candidates=%d deleted_bq=%d deleted_fs=%d",
        int(summary.get("total_candidates") or 0),
        int(summary.get("total_deleted_bigquery") or 0),
        int(summary.get("total_deleted_firestore") or 0),
    )
    return int(summary.get("total_deleted_bigquery") or 0)
