from __future__ import annotations

import argparse
import logging

from arena.board.store import BoardStore
from arena.broker.open_trading import KISOpenTradingBroker
from arena.broker.paper import KISHttpBroker, PaperBroker
from arena.cloud_run_jobs import run_cloud_run_job
from arena.config import (
    Settings,
    SettingsError,
    apply_distribution_mode,
    apply_runtime_overrides,
    load_settings,
    validate_settings,
)
from arena.context import ContextBuilder
from arena.data.bq import BigQueryRepository
from arena.execution.gateway import ExecutionGateway
from arena.logging_utils import configure_logging
from arena.memory.cleanup import run_memory_cleanup_for_all_tenants
from arena.memory.store import MemoryStore
from arena.models import utc_now
from arena.market_hours import MarketWindow, is_kospi_holiday, is_nasdaq_holiday, is_report_window, kospi_window, nasdaq_window
from arena.open_trading.sync import AccountSyncService, BrokerCashSyncService, BrokerTradeSyncService, DividendSyncService, MarketDataSyncService
from arena.orchestrator import ArenaOrchestrator
from arena.providers.credentials import apply_model_secret_payload
from arena.reconciliation import StateReconciliationService, StateRecoveryService
from arena.risk import RiskEngine
from arena.strategy.catalog import list_cards
from arena.strategy.mcp_server import serve_mcp
from arena.tenant_leases import FirestoreTenantLeaseStore

from arena.cli_commands.admin import (
    cmd_approve_live_tenant,
    cmd_backfill_tenant_markets,
    cmd_enable_memory_forgetting,
    cmd_promote_tenant_live,
    cmd_run_memory_forgetting_tuner,
    cmd_set_tenant_simulated,
)
from arena.cli_commands.memory_relations import cmd_extract_memory_relations
from arena.cli_commands.run import (
    _apply_market_override,
    _batch_market_sync,
    _batch_phase,
    _batch_tenant_work,
    _cloud_run_execution_body,
    _dispatch_agent_job,
    _execution_market_key,
    _execution_source,
    _execution_trading_date,
    _filter_tenants_by_market,
    _has_kr,
    _has_us,
    _live_agent_cycle_market_closed,
    _market_filter_matches,
    _market_value_matches,
    _parse_cli_markets,
    _run_agent_cycle_once,
    _run_agent_cycle_once_guarded,
    _run_batch_once,
    _run_memory_cleanup,
    _run_memory_compaction,
    _run_memory_relation_extraction_post_cycle,
    _run_memory_relation_tuner_post_cycle,
    _run_memory_forgetting_tuner_post_cycle,
    _run_mtm_score_update,
    _run_post_cycle_maintenance,
    _run_reconciliation_guard,
    _sync_broker_cash_ledger,
    _sync_broker_trade_ledger,
    _tenant_lease_enabled,
    cmd_run_agent_cycle,
    cmd_run_batch,
    cmd_run_cycle,
    cmd_run_memory_compaction,
    cmd_run_pipeline,
    cmd_run_shared_prep,
)
from arena.cli_commands.serve import (
    _provider_credentials_ready,
    _run_research_smoke,
    _run_thesis_compaction_smoke,
    cmd_serve_strategy_mcp,
    cmd_serve_ui,
    cmd_smoke_research,
    cmd_smoke_thesis_compaction,
)
from arena.cli_commands.sync import (
    _build_forecast_tickers,
    _prepare_kis_command_repo,
    cmd_build_forecasts,
    cmd_build_opportunity_ranker,
    cmd_fundamentals_backfill_kr,
    cmd_fundamentals_backfill_us,
    cmd_fundamentals_coverage,
    cmd_init_bq,
    cmd_refresh_fundamentals_derived,
    cmd_refresh_regime_features,
    cmd_refresh_signal_ic,
    cmd_refresh_signals,
    cmd_list_strategies,
    cmd_recover_sleeves,
    cmd_seed_demo_market,
    cmd_sync_account,
    cmd_sync_broker_cash,
    cmd_sync_broker_trades,
    cmd_sync_dividends,
    cmd_sync_market,
    cmd_sync_market_quotes,
)
from arena.cli_runtime import (
    _allow_checkpoint_rebuild_recovery,
    _append_tenant_run_status,
    _append_tenant_run_status_many,
    _apply_shared_research_gemini,
    _apply_tenant_runtime_credentials,
    _build_agents,
    _build_runtime,
    _cloud_run_log_uri,
    _csv_env,
    _float_env,
    _int_env,
    _load_secret_json,
    _new_run_id,
    _parse_tenant_tokens,
    _partition_tenants_for_task,
    _reconcile_excluded_tickers,
    _repo_or_exit,
    _resolve_batch_tenants,
    _seed_rows,
    _task_shard_spec,
    _tenant_id,
    _truthy_env,
    _validate_or_exit,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Creates the command-line parser for arena operations."""
    parser = argparse.ArgumentParser(prog="llm-arena", description="LLM investment arena runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-bq", help="Create dataset/tables")
    sub.add_parser("seed-demo-market", help="Insert demo market_features rows")
    sub.add_parser("sync-market", help="Sync market_features from open-trading API")
    sub.add_parser("sync-market-quotes", help="Sync intraday quotes into market_features")
    sub.add_parser("sync-account", help="Sync account snapshot from open-trading API")
    sync_broker_trades = sub.add_parser("sync-broker-trades", help="Sync raw broker trade events into ledger tables")
    sync_broker_trades.add_argument("--days", type=int, default=7, help="Recent day window to scan")
    sync_broker_cash = sub.add_parser("sync-broker-cash", help="Sync signed broker cash settlement events into ledger tables")
    sync_broker_cash.add_argument("--days", type=int, default=7, help="Recent day window to scan")
    recover_sleeves = sub.add_parser("recover-sleeves", help="Rebuild agent checkpoints from current state and re-run reconciliation")
    recover_sleeves.add_argument("--live", action="store_true", help="Refresh live broker snapshot/ledger before recovery")
    sub.add_parser("sync-dividends", help="Discover and attribute overseas dividends to agent sleeves")
    build_fc = sub.add_parser("build-forecasts", help="Build stacked forecasts into predicted_expected_returns")
    build_fc.add_argument("--lookback-days", type=int, default=360, help="History window (calendar days)")
    build_fc.add_argument("--horizon", type=int, default=20, help="Forecast horizon (business days)")
    build_fc.add_argument("--min-series-length", type=int, default=160, help="Minimum observations per ticker")
    build_fc.add_argument("--max-steps", type=int, default=200, help="Max training steps for neuralforecast models")
    build_fc.add_argument("--top-n", type=int, default=50, help="Approximate discovery-candidate budget to forecast before adding holdings")
    build_ranker = sub.add_parser("build-opportunity-ranker", help="Build learned opportunity ranker scores (signal-IC meta-learner)")
    build_ranker.add_argument("--lookback-days", type=int, default=540, help="Historical window for IC training")
    build_ranker.add_argument("--horizon", type=int, default=20, help="IC horizon in trading rows")
    build_ranker.add_argument("--max-scoring-rows", type=int, default=500, help="Maximum latest candidates to score")
    build_ranker.add_argument("--min-ic-dates", type=int, default=60, help="Minimum distinct IC dates required")
    build_ranker.add_argument("--min-valid-signals", type=int, default=3, help="Minimum signals with trained IC models")
    refresh_signals = sub.add_parser("refresh-signals", help="Recompute signal_daily_values only (debugging)")
    refresh_signals.add_argument("--lookback-days", type=int, default=540)
    refresh_signals.add_argument("--horizon", type=int, default=20)
    refresh_ic = sub.add_parser("refresh-signal-ic", help="Recompute signal_daily_ic only (debugging)")
    refresh_ic.add_argument("--lookback-days", type=int, default=540)
    refresh_ic.add_argument("--horizon", type=int, default=20)
    refresh_regime = sub.add_parser("refresh-regime-features", help="Recompute regime_daily_features only (debugging)")
    refresh_regime.add_argument("--lookback-days", type=int, default=540)
    refresh_derived = sub.add_parser("refresh-fundamentals-derived", help="Recompute fundamentals_derived_daily")
    refresh_derived.add_argument("--lookback-days", type=int, default=600)
    fund_kr = sub.add_parser("fundamentals-backfill-kr", help="KIS-based KR fundamentals backfill into fundamentals_history_raw")
    fund_kr.add_argument("--tickers", type=str, default="", help="Comma-separated tickers; empty → use universe")
    fund_kr.add_argument("--tickers-file", type=str, default="", help="Optional path to a newline-delimited ticker file")
    fund_kr.add_argument("--period", type=str, default="quarter", choices=["quarter", "annual"])
    fund_us = sub.add_parser("fundamentals-backfill-us", help="SEC/FMP-based US fundamentals backfill into fundamentals_history_raw")
    fund_us.add_argument("--tickers", type=str, default="")
    fund_us.add_argument("--tickers-file", type=str, default="")
    fund_us.add_argument("--source", type=str, default="sec", choices=["sec", "fmp"], help="US fundamentals source; SEC is keyless and default")
    fund_us.add_argument("--api-key", type=str, default="", help="FMP API key; only used with --source fmp")
    fund_us.add_argument("--sec-user-agent", type=str, default="", help="SEC declared User-Agent; if omitted uses SEC_USER_AGENT or ARENA_OPERATOR_EMAILS")
    fund_us.add_argument("--period", type=str, default="quarter", choices=["quarter", "annual"])
    fund_us.add_argument("--limit", type=int, default=40, help="Periods per ticker")
    fund_us.add_argument("--sleep-seconds", type=float, default=0.15, help="SEC sleep between companyfacts requests")
    sub.add_parser("fundamentals-coverage", help="Print fundamentals_history_raw coverage report")
    sub.add_parser("list-strategies", help="Print strategy reference cards")

    run_cycle = sub.add_parser("run-cycle", help="Run one arena cycle (deprecated; use run-agent-cycle or run-pipeline)")
    run_cycle.add_argument("--live", action="store_true", help="Use live broker")

    run_batch = sub.add_parser("run-batch", help="Sync data + run one cycle (manual shortcut)")
    run_batch.add_argument("--live", action="store_true", help="Use live broker")
    run_batch.add_argument("--all-tenants", action="store_true", help="Run batch for all runtime tenants")
    run_batch.add_argument("--market", type=str, default="", help="Override target market for this run (us, kospi)")

    run_pipeline = sub.add_parser("run-pipeline", help="sync → forecast → agent cycle (single Cloud Run Job)")
    run_pipeline.add_argument("--live", action="store_true", help="Use live broker")
    run_pipeline.add_argument("--all-tenants", action="store_true", help="Run for all runtime tenants")
    run_pipeline.add_argument("--market", type=str, default="", help="Override target market for this run (us, kospi)")

    run_shared_prep = sub.add_parser("run-shared-prep", help="Run shared sync + forecast once, then optionally dispatch agent job")
    run_shared_prep.add_argument("--live", action="store_true", help="Use live broker")
    run_shared_prep.add_argument("--market", type=str, default="", help="Override target market for this run (us, kospi)")
    run_shared_prep.add_argument("--dispatch-job", type=str, default="", help="Optional downstream Cloud Run agent job name")
    run_shared_prep.add_argument(
        "--stage",
        choices=["all", "slow", "fast"],
        default="all",
        help="'slow'=forecasts+fundamentals+ranker (ML, run early), 'fast'=sync-market only (run before agent), 'all'=both (legacy).",
    )

    run_agent = sub.add_parser("run-agent-cycle", help="Run agent trading cycle only")
    run_agent.add_argument("--live", action="store_true", help="Use live broker")
    run_agent.add_argument("--all-tenants", action="store_true", help="Run for all runtime tenants")
    run_agent.add_argument("--market", type=str, default="", help="Override target market for this run (us, kospi)")

    run_memory_compaction = sub.add_parser("run-memory-compaction", help="Run memory compaction only")
    run_memory_compaction.add_argument("--live", action="store_true", help="Use live trading-mode memories")
    run_memory_compaction.add_argument("--all-tenants", action="store_true", help="Run for all runtime tenants")
    run_memory_compaction.add_argument("--tenant", action="append", default=[], help="Optional tenant id (repeatable or comma-separated)")
    run_memory_compaction.add_argument("--market", type=str, default="", help="Override target market for this run (us, kospi)")
    run_memory_compaction.add_argument("--cycle-id", type=str, default="", help="Cycle id to compact; empty/latest uses the newest eligible cycle")
    run_memory_compaction.add_argument("--agent", action="append", default=[], help="Optional agent id (repeatable). Defaults to configured agents")
    run_memory_compaction.add_argument("--timeout", type=int, default=0, help="Optional LLM timeout override in seconds")
    run_memory_compaction.add_argument("--dry-run", action="store_true", help="Run LLM compaction without saving reflections")
    run_memory_compaction.add_argument("--force", action="store_true", help="Save even if this cycle already has compaction reflections")

    smoke_research = sub.add_parser("smoke-research", help="Smoke-test one provider with google_search research flow")
    smoke_research.add_argument("--provider", choices=["gpt", "gemini", "claude"], required=True, help="Provider to test")
    smoke_research.add_argument("--prompt", type=str, default="", help="Optional research prompt override")
    smoke_research.add_argument("--timeout", type=int, default=0, help="Optional timeout override in seconds")
    smoke_thesis = sub.add_parser("smoke-thesis-compaction", help="Smoke-test thesis-chain compaction for one cycle")
    smoke_thesis.add_argument("--cycle-id", required=True, help="Cycle id whose closed thesis chains should be compacted")
    smoke_thesis.add_argument("--agent", action="append", default=[], help="Optional agent id (repeatable). Defaults to configured agents")
    smoke_thesis.add_argument("--timeout", type=int, default=0, help="Optional timeout override in seconds")
    smoke_thesis.add_argument("--save", action="store_true", help="Persist generated reflections instead of previewing them")
    approve_live = sub.add_parser("approve-live-tenant", help="Approve or revoke real KIS trading for one tenant")
    approve_live.add_argument("--tenant", required=True, help="Tenant id to update")
    approve_live.add_argument("--approved", choices=["true", "false"], default="true", help="Whether real trading is approved")
    approve_live.add_argument("--updated-by", default="cli-admin", help="Actor id/email recorded in audit log")
    approve_live.add_argument("--note", default="", help="Optional audit note")
    promote_live = sub.add_parser("promote-tenant-live", help="Promote one tenant to private live-trading mode")
    promote_live.add_argument("--tenant", required=True, help="Tenant id to update")
    promote_live.add_argument("--updated-by", default="cli-admin", help="Actor id/email recorded in audit log")
    promote_live.add_argument("--note", default="", help="Optional audit note")
    set_simulated = sub.add_parser("set-tenant-simulated", help="Demote one tenant back to simulated-only mode")
    set_simulated.add_argument("--tenant", required=True, help="Tenant id to update")
    set_simulated.add_argument("--updated-by", default="cli-admin", help="Actor id/email recorded in audit log")
    set_simulated.add_argument("--note", default="", help="Optional audit note")
    backfill_market = sub.add_parser("backfill-tenant-markets", help="Backfill tenant kis_target_market from agents_config")
    backfill_market.add_argument("--tenant", action="append", default=[], help="Optional tenant id (repeatable). Defaults to all runtime tenants")
    backfill_market.add_argument("--updated-by", default="cli-admin", help="Actor id/email recorded in audit log")
    enable_memory_forgetting = sub.add_parser(
        "enable-memory-forgetting",
        help="Enable forgetting, access logs, and shadow tuning for runtime tenants",
    )
    enable_memory_forgetting.add_argument("--tenant", action="append", default=[], help="Optional tenant id (repeatable). Defaults to all runtime tenants")
    enable_memory_forgetting.add_argument("--updated-by", default="cli-admin", help="Actor id/email recorded in audit log")
    run_memory_tuner = sub.add_parser(
        "run-memory-forgetting-tuner",
        help="Run forgetting tuner for runtime tenants so this command can be scheduled",
    )
    run_memory_tuner.add_argument("--tenant", action="append", default=[], help="Optional tenant id (repeatable). Defaults to all runtime tenants")
    run_memory_tuner.add_argument("--updated-by", default="cli-memory-tuner", help="Actor id/email recorded in audit log")

    extract_rel = sub.add_parser(
        "extract-memory-relations",
        help="Extract semantic relation triples from stored memory sources",
    )
    extract_rel.add_argument("--tenant", action="append", default=[], help="Optional tenant id (repeatable or comma-separated)")
    extract_rel.add_argument("--limit", type=int, default=25, help="Maximum pending sources per tenant")
    extract_rel.add_argument(
        "--source-table",
        default="",
        help="Optional source table: agent_memory_events, board_posts, research_briefings",
    )
    extract_rel.add_argument("--event-type", action="append", default=[], help="Memory event type filter (repeatable)")
    extract_rel.add_argument("--dry-run", action="store_true", help="Preview extraction without writing triples or run rows")
    extract_rel.add_argument("--timeout", type=int, default=0, help="Optional LLM timeout override in seconds")
    extract_rel.add_argument("--provider", default="", help="Optional helper provider override")
    extract_rel.add_argument("--model", default="", help="Optional helper model override")
    extract_rel.add_argument("--min-confidence", type=float, default=0.65, help="Minimum accepted relation confidence")
    extract_rel.add_argument("--max-triples", type=int, default=6, help="Maximum triples to request per source")

    sub.add_parser("serve-strategy-mcp", help="Run MCP strategy reference server")
    sub.add_parser("serve-ui", help="Serve read-only UI for board/memory/NAV")
    return parser


def _dispatch_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    handlers = {
        "init-bq": lambda ns: cmd_init_bq(),
        "seed-demo-market": lambda ns: cmd_seed_demo_market(),
        "sync-market": lambda ns: cmd_sync_market(),
        "sync-market-quotes": lambda ns: cmd_sync_market_quotes(),
        "sync-account": lambda ns: cmd_sync_account(),
        "sync-broker-trades": lambda ns: cmd_sync_broker_trades(days=int(getattr(ns, "days", 7) or 7)),
        "sync-broker-cash": lambda ns: cmd_sync_broker_cash(days=int(getattr(ns, "days", 7) or 7)),
        "recover-sleeves": lambda ns: cmd_recover_sleeves(live=bool(getattr(ns, "live", False))),
        "sync-dividends": lambda ns: cmd_sync_dividends(),
        "build-forecasts": lambda ns: cmd_build_forecasts(ns),
        "build-opportunity-ranker": lambda ns: cmd_build_opportunity_ranker(ns),
        "refresh-signals": lambda ns: cmd_refresh_signals(ns),
        "refresh-signal-ic": lambda ns: cmd_refresh_signal_ic(ns),
        "refresh-regime-features": lambda ns: cmd_refresh_regime_features(ns),
        "refresh-fundamentals-derived": lambda ns: cmd_refresh_fundamentals_derived(ns),
        "fundamentals-backfill-kr": lambda ns: cmd_fundamentals_backfill_kr(ns),
        "fundamentals-backfill-us": lambda ns: cmd_fundamentals_backfill_us(ns),
        "fundamentals-coverage": lambda ns: cmd_fundamentals_coverage(ns),
        "list-strategies": lambda ns: cmd_list_strategies(),
        "run-cycle": lambda ns: cmd_run_cycle(live=bool(ns.live)),
        "run-batch": lambda ns: cmd_run_batch(
            live=bool(ns.live),
            all_tenants=bool(getattr(ns, "all_tenants", False)),
            market_override=str(getattr(ns, "market", "") or ""),
        ),
        "run-pipeline": lambda ns: cmd_run_pipeline(
            live=bool(ns.live),
            all_tenants=bool(getattr(ns, "all_tenants", False)),
            market_override=str(getattr(ns, "market", "") or ""),
        ),
        "run-shared-prep": lambda ns: cmd_run_shared_prep(
            live=bool(ns.live),
            market_override=str(getattr(ns, "market", "") or ""),
            dispatch_job=str(getattr(ns, "dispatch_job", "") or ""),
            stage=str(getattr(ns, "stage", "all") or "all"),
        ),
        "run-agent-cycle": lambda ns: cmd_run_agent_cycle(
            live=bool(ns.live),
            all_tenants=bool(getattr(ns, "all_tenants", False)),
            market_override=str(getattr(ns, "market", "") or ""),
        ),
        "run-memory-compaction": lambda ns: cmd_run_memory_compaction(
            live=bool(getattr(ns, "live", False)),
            all_tenants=bool(getattr(ns, "all_tenants", False)),
            tenant_ids=list(getattr(ns, "tenant", [])),
            cycle_id=str(getattr(ns, "cycle_id", "") or ""),
            market_override=str(getattr(ns, "market", "") or ""),
            agent_ids=list(getattr(ns, "agent", [])),
            timeout_seconds=int(getattr(ns, "timeout", 0) or 0),
            dry_run=bool(getattr(ns, "dry_run", False)),
            force=bool(getattr(ns, "force", False)),
        ),
        "smoke-research": lambda ns: cmd_smoke_research(
            str(getattr(ns, "provider", "") or ""),
            prompt=str(getattr(ns, "prompt", "") or ""),
            timeout_seconds=int(getattr(ns, "timeout", 0) or 0),
        ),
        "smoke-thesis-compaction": lambda ns: cmd_smoke_thesis_compaction(
            str(getattr(ns, "cycle_id", "") or ""),
            agent_ids=list(getattr(ns, "agent", [])),
            timeout_seconds=int(getattr(ns, "timeout", 0) or 0),
            save=bool(getattr(ns, "save", False)),
        ),
        "approve-live-tenant": lambda ns: cmd_approve_live_tenant(
            tenant_id=str(getattr(ns, "tenant", "") or ""),
            approved=str(getattr(ns, "approved", "true") or "true").strip().lower() == "true",
            updated_by=str(getattr(ns, "updated_by", "") or "cli-admin"),
            note=str(getattr(ns, "note", "") or ""),
        ),
        "promote-tenant-live": lambda ns: cmd_promote_tenant_live(
            tenant_id=str(getattr(ns, "tenant", "") or ""),
            updated_by=str(getattr(ns, "updated_by", "") or "cli-admin"),
            note=str(getattr(ns, "note", "") or ""),
        ),
        "set-tenant-simulated": lambda ns: cmd_set_tenant_simulated(
            tenant_id=str(getattr(ns, "tenant", "") or ""),
            updated_by=str(getattr(ns, "updated_by", "") or "cli-admin"),
            note=str(getattr(ns, "note", "") or ""),
        ),
        "backfill-tenant-markets": lambda ns: cmd_backfill_tenant_markets(
            tenant_ids=list(getattr(ns, "tenant", [])),
            updated_by=str(getattr(ns, "updated_by", "") or "cli-admin"),
        ),
        "enable-memory-forgetting": lambda ns: cmd_enable_memory_forgetting(
            tenant_ids=list(getattr(ns, "tenant", [])),
            updated_by=str(getattr(ns, "updated_by", "") or "cli-admin"),
        ),
        "run-memory-forgetting-tuner": lambda ns: cmd_run_memory_forgetting_tuner(
            tenant_ids=list(getattr(ns, "tenant", [])),
            updated_by=str(getattr(ns, "updated_by", "") or "cli-memory-tuner"),
        ),
        "extract-memory-relations": lambda ns: cmd_extract_memory_relations(
            tenant_ids=list(getattr(ns, "tenant", [])),
            limit=int(getattr(ns, "limit", 25) or 25),
            source_table=str(getattr(ns, "source_table", "") or ""),
            event_types=list(getattr(ns, "event_type", [])),
            dry_run=bool(getattr(ns, "dry_run", False)),
            timeout_seconds=int(getattr(ns, "timeout", 0) or 0),
            provider=str(getattr(ns, "provider", "") or ""),
            model=str(getattr(ns, "model", "") or ""),
            min_confidence=float(getattr(ns, "min_confidence", 0.65) or 0.65),
            max_triples_per_source=int(getattr(ns, "max_triples", 6) or 6),
        ),
        "serve-strategy-mcp": lambda ns: cmd_serve_strategy_mcp(),
        "serve-ui": lambda ns: cmd_serve_ui(),
    }
    handler = handlers.get(args.command)
    if handler is None:
        parser.error("Unknown command")
    handler(args)


def main() -> None:
    """Dispatches CLI commands to the runtime handlers."""
    parser = build_parser()
    args = parser.parse_args()
    _dispatch_command(args, parser)


if __name__ == "__main__":
    main()
