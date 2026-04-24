from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any

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
                logger.info(
                    "[cyan]Pre-cycle reconcile[/cyan] tenant=%s updated=%d",
                    tenant,
                    reconciled,
                    extra=event_extra(
                        "batch_pre_cycle_reconcile",
                        tenant_id=tenant,
                        phase=phase,
                        updated=reconciled,
                        live=live,
                    ),
                )
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
                        extra=event_extra(
                            "batch_dividend_sync",
                            tenant_id=tenant,
                            phase=phase,
                            events_inserted=div_result.events_inserted,
                            skipped_duplicate=div_result.skipped_duplicate,
                            broker_cash_events_inserted=div_result.broker_cash_events_inserted,
                            usd_krw=usd_krw or settings.usd_krw_rate,
                        ),
                    )
                except Exception as exc:
                    logger.warning(
                        "[yellow]Dividend sync failed; continuing[/yellow] tenant=%s err=%s",
                        tenant,
                        str(exc),
                        extra=failure_extra(
                            "batch_dividend_sync_failed",
                            exc,
                            tenant_id=tenant,
                            phase=phase,
                        ),
                    )

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
        logger.info(
            "[cyan]Research phase[/cyan] tenant=%s briefings=%d held=%s",
            tenant,
            len(briefings),
            held_tickers,
            extra=event_extra(
                "batch_research_phase",
                tenant_id=tenant,
                phase=phase,
                briefing_count=len(briefings),
                held_tickers=held_tickers,
            ),
        )

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
            extra=event_extra(
                "batch_tenant_done",
                tenant_id=tenant,
                phase=phase,
                executed=executed,
                submitted=submitted,
                total_reports=len(reports),
            ),
        )

    elif phase == "report":
        cli.AccountSyncService(settings=settings, repo=repo).sync_account_snapshot()
        logger.info(
            "[bold green]Batch done[/bold green] tenant=%s mode=report",
            tenant,
            extra=event_extra(
                "batch_tenant_done",
                tenant_id=tenant,
                phase=phase,
                mode="report",
            ),
        )

    elif phase == "closed":
        reconciled = orchestrator.gateway.reconcile_submitted_orders()
        if reconciled:
            logger.info(
                "[cyan]Off-hours reconcile[/cyan] tenant=%s updated=%d",
                tenant,
                reconciled,
                extra=event_extra(
                    "batch_off_hours_reconcile",
                    tenant_id=tenant,
                    phase=phase,
                    updated=reconciled,
                ),
            )
        logger.info(
            "[cyan]Market closed; skipping cycle[/cyan] tenant=%s phase=%s",
            tenant,
            window.phase if window else "unknown",
            extra=event_extra(
                "batch_market_closed",
                tenant_id=tenant,
                phase=window.phase if window else "unknown",
            ),
        )

    elif phase == "seed":
        logger.info(
            "[bold green]Batch done[/bold green] tenant=%s mode=seed",
            tenant,
            extra=event_extra(
                "batch_tenant_done",
                tenant_id=tenant,
                phase=phase,
                mode="seed",
            ),
        )


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
    logger.info(
        "[bold]Batch start[/bold] tenant=%s live=%s market=%s",
        tenant,
        live,
        settings.kis_target_market,
        extra=event_extra(
            "batch_tenant_start",
            tenant_id=tenant,
            live=live,
            market=settings.kis_target_market,
        ),
    )

    phase, window = cli._batch_phase(live, settings, repo)
    cli._batch_market_sync(phase, settings, repo, window)
    if phase == "seed" or phase is None:
        if phase == "seed":
            logger.info(
                "[bold green]Batch done[/bold green] tenant=%s mode=seed",
                tenant,
                extra=event_extra(
                    "batch_tenant_done",
                    tenant_id=tenant,
                    mode="seed",
                ),
            )
        return
    cli._batch_tenant_work(phase, live, settings=settings, repo=repo, orchestrator=orchestrator, tenant=tenant, window=window)


_MARKET_TRADING_TZ: dict[str, str] = {
    "kospi": "Asia/Seoul",
    "us": "America/New_York",
}

_MARKET_ALIASES_US: frozenset[str] = frozenset({"us", "nasdaq", "nyse", "amex"})
_MARKET_ALIASES_KR: frozenset[str] = frozenset({"kospi", "kosdaq", "kr", "kr_stock"})

_QUOTE_SOURCE_BY_MARKET: dict[str, tuple[str, ...]] = {
    "us": ("open_trading_us_quote",),
    "kospi": ("open_trading_kospi_quote",),
}

_DAILY_SOURCE_BY_MARKET: dict[str, str] = {
    "us": "open_trading_us",
    "kospi": "open_trading_kospi",
}

# Daily EOD rows must be at most this many days behind the trading_date.
# 5 days covers Friday->Monday (3-day gap) + one missed business day of slack.
_UPSTREAM_MAX_DAILY_AGE_DAYS = int(os.getenv("ARENA_UPSTREAM_MAX_DAILY_AGE_DAYS", "5") or 5)


def _canonical_market_key(raw: str) -> str:
    """Canonicalizes a KIS_TARGET_MARKET-style token into the row-level
    vocabulary used by BQ data (``us`` / ``kospi``). Accepts aliases such as
    ``nasdaq``/``nyse``/``amex`` (US) and ``kosdaq``/``kr`` (Korea). For
    comma-separated configs, the first token wins — the shared-prep flow
    always runs one market at a time after _apply_market_override.
    """
    token = str(raw or "").strip().lower()
    if "," in token:
        token = token.split(",", 1)[0].strip()
    if token in _MARKET_ALIASES_US:
        return "us"
    if token in _MARKET_ALIASES_KR:
        return "kospi"
    return token


def _trading_date_for_market(market: str) -> Any:
    """Returns today's trading date for the market's local timezone.

    Assumes scheduler only fires on weekdays (cron '1-5'); no holiday-aware
    rollback is applied. Same-session means 'same civil date in market TZ'.
    Canonicalizes the incoming market so ``nasdaq``/``nyse``/``amex`` land on
    America/New_York instead of silently falling back to UTC.
    """
    from zoneinfo import ZoneInfo

    canonical = _canonical_market_key(market)
    tz_name = _MARKET_TRADING_TZ.get(canonical, "UTC")
    try:
        return datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        # Fallback to UTC date on any tz lookup failure.
        return datetime.now(timezone.utc).date()


def _upstream_market_freshness(
    repo: BigQueryRepository,
    *,
    market: str,
    trading_date: Any,
    max_age_days: int = _UPSTREAM_MAX_DAILY_AGE_DAYS,
) -> tuple[bool, dict[str, Any]]:
    """Verifies that daily EOD rows for this market are not absurdly stale.

    Fast quote sync requires a recent daily base (see open_trading/sync.py
    _has_fresh_daily_feature_metrics), and the ranker trains on
    signal_daily_values derived from market_features daily rows. A feed
    that is weeks behind would silently train on outdated prices, so refuse.

    Fails closed on query errors.
    """
    canonical = _canonical_market_key(market)
    daily_source = _DAILY_SOURCE_BY_MARKET.get(canonical)
    if not daily_source:
        return True, {"reason": "no_daily_source_mapping", "market": canonical}

    try:
        sql = (
            f"SELECT MAX(DATE(as_of_ts)) AS max_date "
            f"FROM `{repo.session.dataset_fqn}.market_features` "
            f"WHERE source = @source"
        )
        rows = list(repo.session.fetch_rows(sql, {"source": daily_source}))
    except Exception as exc:
        return False, {
            "reason": "query_failed",
            "error": str(exc),
            "market": canonical,
            "source": daily_source,
        }

    if not rows or rows[0].get("max_date") is None:
        return False, {
            "reason": "no_daily_rows",
            "market": canonical,
            "source": daily_source,
        }
    max_date = rows[0]["max_date"]
    try:
        td = trading_date if hasattr(trading_date, "toordinal") else datetime.strptime(
            str(trading_date), "%Y-%m-%d"
        ).date()
    except Exception:
        td = datetime.now(timezone.utc).date()
    try:
        age_days = int((td - max_date).days)
    except Exception:
        age_days = 10**6
    if age_days > int(max_age_days):
        return False, {
            "reason": "stale_daily",
            "market": canonical,
            "source": daily_source,
            "max_daily_date": str(max_date),
            "trading_date": str(td),
            "age_days": age_days,
            "threshold_days": int(max_age_days),
        }
    return True, {
        "market": canonical,
        "source": daily_source,
        "max_daily_date": str(max_date),
        "age_days": age_days,
    }


def _same_day_quote_rows_present(
    repo: BigQueryRepository,
    *,
    market: str,
    trading_date: Any,
) -> tuple[bool, dict[str, Any]]:
    """Checks whether same-day intraday quote rows already exist in
    market_features for this market. Fails closed: any query error is
    reported as 'may be tainted' so slow prep refuses rather than quietly
    training on partial intraday data.

    Returns (tainted, info_dict). ``tainted=True`` means a quote row for
    today's trading_date already landed in market_features, which would be
    picked up by refresh_signal_daily_values (latest as_of_ts wins per date)
    and drift the ranker off prior-EOD.
    """
    canonical = _canonical_market_key(market)
    sources = _QUOTE_SOURCE_BY_MARKET.get(canonical, ())
    try:
        dataset_fqn = repo.session.dataset_fqn
        if sources:
            sql = (
                f"SELECT COUNT(*) AS cnt "
                f"FROM `{dataset_fqn}.market_features` "
                f"WHERE DATE(as_of_ts) = @trading_date "
                f"  AND source IN UNNEST(@sources)"
            )
            params = {"trading_date": trading_date, "sources": list(sources)}
        else:
            # Unknown market: fall back to wildcard quote suffix, still scoped
            # to today, so we do not starve legitimate runs on seed/bootstrap.
            sql = (
                f"SELECT COUNT(*) AS cnt "
                f"FROM `{dataset_fqn}.market_features` "
                f"WHERE DATE(as_of_ts) = @trading_date "
                f"  AND ENDS_WITH(source, '_quote')"
            )
            params = {"trading_date": trading_date}
        rows = list(repo.session.fetch_rows(sql, params))
    except Exception as exc:
        return True, {
            "reason": "query_failed",
            "error": str(exc),
            "market": canonical,
            "trading_date": str(trading_date),
        }
    if not rows:
        return False, {"market": canonical, "trading_date": str(trading_date), "count": 0}
    cnt = int(rows[0].get("cnt") or 0)
    return (cnt > 0), {
        "market": canonical,
        "trading_date": str(trading_date),
        "count": cnt,
    }


def _shared_prep_session_ready(
    repo: BigQueryRepository,
    *,
    market: str,
    trading_date: Any,
) -> tuple[bool, dict[str, Any]]:
    """Checks that a prior prep produced a same-session success marker for
    this market/trading_date. Accepts either a 'slow' marker (normal split
    workflow) or an 'all' marker (legacy single-shot run), whichever is more
    recent. Returns (is_ready, info_dict). Fails closed on any error.

    Scope is the exact market token (e.g., ``nasdaq``/``nyse``/``amex``), not
    the canonicalized ``us``, because forecast/ranker prep itself is scoped
    by the raw ``settings.kis_target_market`` string. Collapsing all US
    aliases here would let a ``nasdaq`` prep satisfy a ``nyse`` fast dispatch
    (and vice versa) even though they validate different artifact sets.
    """
    market_key = str(market or "").strip().lower()
    candidates: list[dict[str, Any]] = []
    try:
        for stage_token in ("slow", "all"):
            session = repo.get_latest_shared_prep_session(
                market=market_key,
                trading_date=trading_date,
                stage=stage_token,
            )
            if session:
                candidates.append(session)
    except Exception as exc:
        return False, {
            "reason": "query_failed",
            "error": str(exc),
            "market": market_key,
            "trading_date": str(trading_date),
        }

    if not candidates:
        return False, {
            "reason": "no_session",
            "market": market_key,
            "trading_date": str(trading_date),
        }

    def _ts_key(row: dict[str, Any]) -> Any:
        ts = row.get("created_at")
        # datetime.min without tz makes comparisons blow up when other rows
        # are tz-aware; fall back to epoch-0 UTC instead.
        return ts if ts is not None else datetime.fromtimestamp(0, tz=timezone.utc)

    session = max(candidates, key=_ts_key)
    status = str(session.get("status") or "").strip().lower()
    forecast_rows = int(session.get("forecast_rows_written") or 0)
    ranker_scores = int(session.get("ranker_scores_written") or 0)
    if status != "ok" or forecast_rows <= 0 or ranker_scores <= 0:
        return False, {
            "reason": "incomplete_session",
            "market": market_key,
            "trading_date": str(trading_date),
            "status": status,
            "forecast_rows_written": forecast_rows,
            "ranker_scores_written": ranker_scores,
        }

    return True, {
        "market": market_key,
        "trading_date": str(trading_date),
        "session_id": session.get("session_id"),
        "forecast_run_id": session.get("forecast_run_id"),
        "ranker_run_id": session.get("ranker_run_id"),
        "matched_stage": str(session.get("stage") or "").strip().lower(),
    }


def _record_shared_prep_session(
    repo: BigQueryRepository,
    *,
    market: str,
    trading_date: Any,
    stage: str,
    status: str,
    forecast_run_id: str | None = None,
    forecast_rows_written: int = 0,
    ranker_run_id: str | None = None,
    ranker_scores_written: int = 0,
    detail: dict[str, Any] | None = None,
) -> None:
    """Writes a shared_prep_sessions marker row. Logs at warning on failure
    because the fast-stage gate depends on these rows; a silent insert failure
    would turn into a silent dispatch block on the next fast run.
    """
    from uuid import uuid4

    inserter = getattr(repo, "insert_shared_prep_session", None)
    if not callable(inserter):
        logger.warning("[yellow]shared_prep_sessions insert skipped: accessor missing[/yellow]")
        return
    row = {
        "session_id": f"sp_{uuid4().hex[:24]}",
        # Store the raw prep scope token (e.g., 'nasdaq') rather than the
        # canonical bucket ('us') so marker scope matches the scope actually
        # used by forecast/ranker prep (settings.kis_target_market).
        "market": str(market or "").strip().lower(),
        "trading_date": str(trading_date),
        "stage": str(stage or "").strip().lower(),
        "status": str(status or "").strip().lower(),
        "forecast_run_id": forecast_run_id or None,
        "forecast_rows_written": int(forecast_rows_written or 0),
        "ranker_run_id": ranker_run_id or None,
        "ranker_scores_written": int(ranker_scores_written or 0),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "detail_json": dict(detail or {}),
    }
    try:
        inserter(row)
    except Exception as exc:
        logger.warning(
            "[yellow]shared_prep_sessions insert failed[/yellow] err=%s row=%s",
            str(exc),
            {k: v for k, v in row.items() if k != "detail_json"},
        )


def cmd_run_shared_prep(
    live: bool,
    *,
    market_override: str = "",
    dispatch_job: str = "",
    stage: str = "all",
) -> None:
    """Runs shared prep steps. stage='slow' skips sync-market (run early, ML-heavy);
    stage='fast' runs only sync-market + dispatch (run just before agent);
    stage='all' runs everything (legacy single-shot behavior).

    Invariant the slow/fast split relies on
    ----------------------------------------
    Market quotes reach market_features ONLY via _batch_market_sync (inside the
    'all' and 'fast' paths of this command). The ranker consumes
    signal_daily_values, which derives from market_features by picking the
    latest as_of_ts per (ticker, date). Because slow runs without
    _batch_market_sync, no intraday row lands in market_features during slow,
    so signal_daily_values has no partition for today and the ranker stays
    anchored on the prior EOD. The fast stage then adds fresh intraday quotes
    but does NOT re-run the ranker, so the dispatched agent sees:
      - ranker/forecast outputs anchored on prior-day dailies (stable),
      - plus fresh intraday prices in context.market_rows (volatile).
    If a future change lets another path write to market_features (e.g. an
    always-on quote streamer), or moves sync into slow, this invariant breaks
    and fast will need to re-run the ranker after its sync.

    Fast readiness gating
    ---------------------
    stage='fast' calls _shared_prep_session_ready BEFORE _batch_market_sync.
    A failed gate therefore exits without writing intraday rows, which keeps
    the invariant above intact even under partial failures. The gate matches
    on same-session (today's trading_date in the market's TZ) and requires
    forecast_rows_written > 0 AND ranker_scores_written > 0 AND status='ok'
    recorded by the preceding slow/all run. Fast also fails closed when the
    open-cycle quote sync writes zero rows, because dispatching on stale or
    missing intraday snapshots is worse than skipping the cycle.
    """
    stage_norm = str(stage or "all").strip().lower()
    if stage_norm not in {"all", "slow", "fast"}:
        raise ValueError(f"invalid stage={stage!r}; expected one of: all, slow, fast")
    run_sync = stage_norm in {"all", "fast"}
    run_ml_prep = stage_norm in {"all", "slow"}
    allow_dispatch = stage_norm in {"all", "fast"}

    cli = _cli()
    bootstrap_settings = cli.load_settings()
    cli.configure_logging(bootstrap_settings.log_level, bootstrap_settings.log_format)
    cli._validate_or_exit(bootstrap_settings)
    logger.info(
        "[bold]Shared prep start[/bold] stage=%s live=%s market=%s dispatch_job=%s",
        stage_norm,
        live,
        market_override or "all",
        dispatch_job or "-",
        extra=event_extra(
            "shared_prep_start",
            stage=stage_norm,
            live=live,
            market=market_override or "all",
            dispatch_job=dispatch_job or None,
        ),
    )

    cli._apply_market_override(bootstrap_settings, market_override)
    orig_market_env = os.environ.get("KIS_TARGET_MARKET")
    if market_override.strip():
        os.environ["KIS_TARGET_MARKET"] = bootstrap_settings.kis_target_market

    try:
        markets = cli._parse_cli_markets(bootstrap_settings)
        # Shared-prep is single-market by design: readiness markers, taint
        # guards, and the trading_date helper all key on exactly one market.
        # Reject mixed configs here so callers cannot silently get per-market
        # blind spots. Scheduler fan-out (--market us / --market kospi) is
        # the supported pattern.
        if cli._has_us(markets) and cli._has_kr(markets):
            logger.error(
                "[red]Shared prep refused: multi-market config not supported; pin with --market[/red] markets=%s",
                sorted(markets),
                extra=event_extra(
                    "shared_prep_multi_market_refused",
                    stage=stage_norm,
                    market=market_override or "all",
                    configured_markets=sorted(markets),
                ),
            )
            raise SystemExit(5)

        us_open = False
        kr_open = False

        if cli._has_us(markets):
            us_win = cli.nasdaq_window()
            us_open = not (us_win.now_local.weekday() >= 5 or cli.is_nasdaq_holiday(us_win.trading_date))

        if cli._has_kr(markets):
            kr_win = cli.kospi_window()
            kr_open = not cli.is_kospi_holiday(kr_win.trading_date)

        if not us_open and not kr_open:
            logger.info(
                "[yellow]All markets closed (holiday/weekend)[/yellow] — skipping shared prep",
                extra=event_extra(
                    "shared_prep_market_closed",
                    live=live,
                    market=market_override or "all",
                ),
            )
            return

        repo = cli._repo_or_exit(bootstrap_settings, tenant_id=cli._tenant_id() or "local")
        repo.ensure_dataset()
        repo.ensure_tables()
        cli._apply_tenant_runtime_credentials(bootstrap_settings, repo)

        phase, window = cli._batch_phase(live, bootstrap_settings, repo)
        if phase is None:
            logger.info(
                "[yellow]Shared prep: not a scheduled cycle time — skipping[/yellow]",
                extra=event_extra(
                    "shared_prep_schedule_closed",
                    live=live,
                    market=market_override or "all",
                ),
            )
            return

        # Prep run scope key = raw kis_target_market token (e.g., 'nasdaq').
        # The canonical bucket ('us') is still used for tz/taint helpers, but
        # readiness markers must isolate distinct exchanges.
        market_for_session = str(bootstrap_settings.kis_target_market or "").strip().lower()
        trading_date = _trading_date_for_market(market_for_session)

        if phase == "seed":
            # Seed/bootstrap: ML artifacts from sparse history are worse than
            # nothing, so ML is always refused. But daily EOD sync is safe —
            # in fact it is the only way to leave seed state, so slow MUST
            # run it here even though run_sync is False for the split flow.
            if run_sync:
                # stage=all / fast: sync_market(phase=seed) == sync_market_features
                logger.info("[bold cyan]Shared prep: sync-market[/bold cyan]")
                cli._batch_market_sync(phase, bootstrap_settings, repo, window)
            elif stage_norm == "slow":
                logger.info("[bold cyan]Shared prep: seed+slow -> daily EOD sync[/bold cyan]")
                try:
                    cli.MarketDataSyncService(
                        settings=bootstrap_settings, repo=repo
                    ).sync_market_features()
                except Exception as exc:
                    logger.error(
                        "[red]Slow seed: daily sync failed[/red] err=%s",
                        str(exc),
                        extra=failure_extra(
                            "shared_prep_seed_daily_sync_failed",
                            exc,
                            stage=stage_norm,
                            market=market_override or "all",
                        ),
                    )
                    raise SystemExit(8)
            logger.info(
                "[bold green]Shared prep done[/bold green] mode=seed stage=%s",
                stage_norm,
                extra=event_extra(
                    "shared_prep_seed_skip_ml",
                    stage=stage_norm,
                    market=market_override or "all",
                ),
            )
            return

        # Fast stage: verify readiness BEFORE syncing, REGARDLESS of whether
        # a downstream dispatch is requested. Intraday writes without a
        # matching slow/all session taint later ranker rebuilds, so the gate
        # fires on every fast invocation (manual or scheduled).
        if stage_norm == "fast":
            ready, info = _shared_prep_session_ready(
                repo,
                market=market_for_session,
                trading_date=trading_date,
            )
            if not ready:
                logger.error(
                    "[red]Fast prep: prior prep session not ready; aborting BEFORE sync[/red] info=%s",
                    info,
                    extra=event_extra(
                        "shared_prep_fast_abort_not_ready",
                        stage=stage_norm,
                        market=market_override or "all",
                        dispatch_job=dispatch_job or None,
                        gate_info=info,
                    ),
                )
                raise SystemExit(3)
            logger.info(
                "[cyan]Fast prep: prior session verified[/cyan] info=%s",
                info,
            )

        sync_result = None
        if run_sync:
            logger.info("[bold cyan]Shared prep: sync-market[/bold cyan]")
            sync_result = cli._batch_market_sync(phase, bootstrap_settings, repo, window)
            if stage_norm == "fast" and phase == "open_cycle":
                sync_inserted_rows = int(getattr(sync_result, "inserted_rows", 0) or 0)
                sync_attempted_tickers = int(getattr(sync_result, "attempted_tickers", 0) or 0)
                sync_failed_tickers = list(getattr(sync_result, "failed_tickers", []) or [])
                if sync_inserted_rows <= 0:
                    logger.error(
                        "[red]Fast prep: quote sync wrote zero rows; refusing dispatch[/red] attempted=%d failed=%d",
                        sync_attempted_tickers,
                        len(sync_failed_tickers),
                        extra=event_extra(
                            "shared_prep_fast_abort_zero_quote_rows",
                            stage=stage_norm,
                            market=market_override or "all",
                            dispatch_job=dispatch_job or None,
                            attempted_tickers=sync_attempted_tickers,
                            failed_tickers=len(sync_failed_tickers),
                        ),
                    )
                    raise SystemExit(7)

        forecast_rows_written = 0
        ranker_scores_written = 0
        ranker_status = ""
        ranker_version = ""
        # Default for non-ML stages (fast / seed early return). Fast has
        # already passed the readiness gate, so treat as 'ok' for the
        # dispatch guard below.
        session_status = "ok"
        if run_ml_prep:
            # Taint guard applies ONLY to stage='slow'. Rationale:
            #   - stage='slow' runs without _batch_market_sync, so any same-day
            #     intraday quote row present in market_features must have been
            #     written by another path (manual sync-market-quotes, sidecar,
            #     a prior fast run). That drifts ranker off prior-EOD, so
            #     refuse.
            #   - stage='all' runs sync BEFORE ML by design; the same-day
            #     quote rows visible here are the ones this invocation just
            #     wrote, which is the legacy single-shot semantics. Running
            #     the taint guard there would abort the default deploy path.
            if stage_norm == "slow":
                tainted, taint_info = _same_day_quote_rows_present(
                    repo,
                    market=market_for_session,
                    trading_date=trading_date,
                )
                if tainted:
                    logger.error(
                        "[red]Slow prep refused: same-day intraday quote rows present[/red] info=%s",
                        taint_info,
                        extra=event_extra(
                            "shared_prep_ml_abort_tainted",
                            stage=stage_norm,
                            market=market_override or "all",
                            taint_info=taint_info,
                        ),
                    )
                    raise SystemExit(4)

            # Daily EOD sync MUST run before ML (slow only). Rationale:
            #   - Live scheduler phases never hit 'general', so there is no
            #     other automated path that populates open_trading_{market}
            #     daily rows. Without this, latest_market_features returns a
            #     stale base, quote sync later fails the 'fresh daily
            #     features' guard, and ranker/forecast train on prior-month
            #     prices for weeks without anyone noticing.
            #   - stage='all' already calls _batch_market_sync above, which
            #     covers daily on seed/general phases.
            if stage_norm == "slow":
                logger.info("[bold cyan]Shared prep: sync-market-features (daily EOD)[/bold cyan]")
                try:
                    daily_result = cli.MarketDataSyncService(
                        settings=bootstrap_settings, repo=repo
                    ).sync_market_features()
                    logger.info(
                        "[cyan]Daily sync[/cyan] inserted=%d attempted=%d failed=%d",
                        int(getattr(daily_result, "inserted_rows", 0) or 0),
                        int(getattr(daily_result, "attempted_tickers", 0) or 0),
                        len(getattr(daily_result, "failed_tickers", []) or []),
                    )
                except Exception as exc:
                    logger.error(
                        "[red]Slow prep: daily sync failed[/red] err=%s",
                        str(exc),
                        extra=failure_extra(
                            "shared_prep_daily_sync_failed",
                            exc,
                            stage=stage_norm,
                            market=market_override or "all",
                        ),
                    )
                    raise SystemExit(8)

            # Upstream freshness guard: refuse if daily EOD data is so stale
            # that the ML step would train on month-old prices. 'stale' here
            # is not a data-pipeline subtlety — it is 'the feed is broken'.
            fresh, fresh_info = _upstream_market_freshness(
                repo,
                market=market_for_session,
                trading_date=trading_date,
            )
            if not fresh:
                logger.error(
                    "[red]Shared prep ML refused: upstream daily feed is stale[/red] info=%s",
                    fresh_info,
                    extra=event_extra(
                        "shared_prep_ml_abort_stale_feed",
                        stage=stage_norm,
                        market=market_override or "all",
                        freshness_info=fresh_info,
                    ),
                )
                raise SystemExit(7)

            logger.info("[bold cyan]Shared prep: build-forecasts[/bold cyan]")
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
            forecast_result = cli.cmd_build_forecasts(fc_args)
            forecast_rows_written = int(getattr(forecast_result, "rows_written", 0) or 0)

            logger.info("[bold cyan]Shared prep: refresh-fundamentals-derived[/bold cyan]")
            fund_args = type(
                "Args",
                (),
                {"lookback_days": 600},
            )()
            try:
                cli.cmd_refresh_fundamentals_derived(fund_args)
            except Exception as exc:
                logger.warning(
                    "[yellow]fundamentals derived refresh skipped (non-fatal)[/yellow] err=%s",
                    exc,
                    extra=failure_extra(
                        "shared_prep_fundamentals_derived_skipped",
                        exc,
                        stage="fundamentals_derived",
                        live=live,
                        market=market_override or "all",
                    ),
                )

            logger.info("[bold cyan]Shared prep: build-opportunity-ranker[/bold cyan]")
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
            ranker_result = cli.cmd_build_opportunity_ranker(ranker_args)
            ranker_scores_written = int(getattr(ranker_result, "scores_written", 0) or 0)
            ranker_status = str(getattr(ranker_result, "status", "") or "").strip().lower()
            ranker_version = str(getattr(ranker_result, "ranker_version", "") or "")

            # Record a same-session readiness marker. Status='ok' is required
            # for the fast gate to pass; anything else (no forecast rows, no
            # ranker scores, ranker 'unusable'/'failed') is explicitly recorded
            # so operators see WHY fast aborted.
            if (
                ranker_status == "ok"
                and forecast_rows_written > 0
                and ranker_scores_written > 0
            ):
                session_status = "ok"
            elif ranker_status == "ok":
                # Ranker claimed ok but one of the required artifact counts
                # was zero; downgrade so fast gate treats this as incomplete.
                session_status = "partial"
            else:
                session_status = ranker_status or "failed"
            session_stage = "slow" if stage_norm == "slow" else "all"
            _record_shared_prep_session(
                repo,
                market=market_for_session,
                trading_date=trading_date,
                stage=session_stage,
                status=session_status,
                forecast_rows_written=forecast_rows_written,
                ranker_scores_written=ranker_scores_written,
                detail={
                    "ranker_version": ranker_version,
                    "ranker_status": ranker_status,
                    "live": live,
                },
            )
            logger.info(
                "[cyan]Shared prep session marker[/cyan] stage=%s status=%s forecast_rows=%d ranker_scores=%d",
                session_stage,
                session_status,
                forecast_rows_written,
                ranker_scores_written,
            )

        if allow_dispatch and dispatch_job.strip():
            # Legacy single-shot (stage='all') must refuse to dispatch when
            # its own prep marker is not 'ok'. Without this the code would
            # detect partial/failed prep, record it, and then launch trading
            # on known-bad artifacts anyway. Fast already passed the gate
            # upstream so it skips this check.
            if stage_norm == "all" and session_status != "ok":
                logger.error(
                    "[red]stage=all: prep marker status=%s; refusing dispatch[/red] forecast_rows=%d ranker_scores=%d",
                    session_status,
                    forecast_rows_written,
                    ranker_scores_written,
                    extra=event_extra(
                        "shared_prep_all_abort_bad_status",
                        stage=stage_norm,
                        market=market_override or "all",
                        dispatch_job=dispatch_job,
                        session_status=session_status,
                        forecast_rows_written=forecast_rows_written,
                        ranker_scores_written=ranker_scores_written,
                    ),
                )
                raise SystemExit(6)
            cli._dispatch_agent_job(bootstrap_settings, job_name=dispatch_job.strip())
            if stage_norm == "fast":
                # Record a fast-session marker after successful dispatch for
                # auditability. Not consumed by the gate; purely observability.
                _record_shared_prep_session(
                    repo,
                    market=market_for_session,
                    trading_date=trading_date,
                    stage="fast",
                    status="ok",
                    detail={"dispatch_job": dispatch_job, "live": live},
                )

        logger.info(
            "[bold green]Shared prep done[/bold green] stage=%s",
            stage_norm,
            extra=event_extra(
                "shared_prep_done",
                stage=stage_norm,
                live=live,
                market=market_override or "all",
                dispatch_job=dispatch_job or None,
            ),
        )
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
    logger.info(
        "[bold]Pipeline start[/bold] live=%s all_tenants=%s market=%s",
        live,
        all_tenants,
        market_override or "all",
        extra=event_extra(
            "pipeline_start",
            live=live,
            all_tenants=all_tenants,
            market=market_override or "all",
        ),
    )
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
        logger.info(
            "[yellow]All markets closed (holiday/weekend)[/yellow] — skipping pipeline",
            extra=event_extra(
                "pipeline_market_closed",
                live=live,
                all_tenants=all_tenants,
                market=market_override or "all",
            ),
        )
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
            logger.info(
                "[yellow]Pipeline: not a scheduled cycle time — skipping[/yellow]",
                extra=event_extra(
                    "pipeline_schedule_closed",
                    live=live,
                    all_tenants=all_tenants,
                    market=market_override or "all",
                ),
            )
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
            logger.warning(
                "[yellow]fundamentals derived refresh skipped[/yellow] err=%s",
                exc,
                extra=failure_extra(
                    "pipeline_fundamentals_derived_skipped",
                    exc,
                    stage=current_stage,
                    live=live,
                    market=market_override or "all",
                ),
            )

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
            logger.warning(
                "[yellow]MTM score update failed (non-fatal): %s[/yellow]",
                exc,
                extra=failure_extra(
                    "pipeline_mtm_score_update_failed",
                    exc,
                    stage="mtm_score_update",
                    live=live,
                    market=market_override or "all",
                ),
            )

        now_utc = cli.utc_now()
        if now_utc.weekday() == 0:
            logger.info("[bold cyan]Pipeline step 7/7: memory-cleanup[/bold cyan]")
            try:
                cli._run_memory_cleanup(bootstrap_settings)
            except Exception as exc:
                logger.warning(
                    "[yellow]Memory cleanup failed (non-fatal): %s[/yellow]",
                    exc,
                    extra=failure_extra(
                        "pipeline_memory_cleanup_failed",
                        exc,
                        stage="memory_cleanup",
                        live=live,
                        market=market_override or "all",
                    ),
                )
        else:
            logger.info("[dim]Pipeline step 7/7: memory-cleanup — skipped (not Monday)[/dim]")

        logger.info(
            "[bold green]Pipeline done[/bold green]",
            extra=event_extra(
                "pipeline_done",
                live=live,
                all_tenants=all_tenants,
                market=market_override or "all",
            ),
        )
    except Exception as exc:
        logger.exception(
            "[red]Pipeline failed[/red] stage=%s err=%s",
            current_stage,
            str(exc),
            extra=failure_extra(
                "pipeline_failed",
                exc,
                stage=current_stage,
                live=live,
                all_tenants=all_tenants,
                market=market_override or "all",
            ),
        )
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
