from __future__ import annotations

import json
import logging
import os
from datetime import timezone
from uuid import uuid4

from google.auth.exceptions import DefaultCredentialsError

from arena.config import (
    Settings,
    SettingsError,
    effective_research_gemini_api_key,
    normalize_distribution_mode,
    validate_settings,
)
from arena.data.bq import BigQueryRepository
from arena.providers.credentials import parse_model_secret_providers

logger = logging.getLogger(__name__)


def _cli():
    import arena.cli as cli

    return cli


def _validate_or_exit(settings: Settings, **kwargs) -> None:
    """Validates settings and exits with an actionable message on failure."""
    try:
        validate_settings(settings, **kwargs)
    except SettingsError as exc:
        logger.error("[red]Settings invalid[/red] %s", str(exc))
        raise SystemExit(2)


def _repo_or_exit(settings: Settings, *, tenant_id: str | None = None) -> BigQueryRepository:
    """Creates a BigQuery repository and exits with actionable guidance on auth failure."""
    try:
        tenant = str(tenant_id or _tenant_id() or "local").strip().lower() or "local"
        return BigQueryRepository(
            project=settings.google_cloud_project,
            dataset=settings.bq_dataset,
            location=settings.bq_location,
            tenant_id=tenant,
        )
    except DefaultCredentialsError:
        logger.error("[red]BigQuery auth failed[/red] Application Default Credentials are missing")
        logger.error("Run these commands first:")
        logger.error("  gcloud auth login")
        logger.error("  gcloud auth application-default login")
        logger.error("  gcloud config set project %s", settings.google_cloud_project)
        raise SystemExit(2)


def _tenant_id() -> str:
    """Returns active tenant id for runtime credential resolution."""
    return (os.getenv("ARENA_TENANT_ID", "") or "").strip().lower()


def _parse_tenant_tokens(raw: str | None) -> list[str]:
    """Parses comma/pipe/semicolon/space-separated tenant ids."""
    text = str(raw or "")
    if not text.strip():
        return []
    import re

    parts = re.split(r"[,|;\s]+", text)
    out: list[str] = []
    for token in parts:
        tenant = str(token or "").strip().lower()
        if tenant and tenant not in out:
            out.append(tenant)
    return out


def _truthy_env(name: str, default: bool = False) -> bool:
    """Parses boolean env flags with a caller-provided default."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _csv_env(name: str) -> list[str]:
    """Returns normalized comma/pipe/semicolon-separated env tokens."""
    raw = os.getenv(name)
    if raw is None:
        return []
    return _parse_tenant_tokens(raw)


def _shared_research_gemini_source_tenant() -> str:
    """Returns the tenant whose Gemini key may be shared for approved live research."""
    return str(os.getenv("ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT", "") or "").strip().lower()


def _apply_shared_research_gemini(
    settings: Settings,
    repo: BigQueryRepository,
    *,
    tenant_id: str | None = None,
) -> str:
    """Applies operator-managed Gemini for research to approved live/private tenants only."""
    tenant = str(tenant_id or _tenant_id() or "").strip().lower()
    source_tenant = _shared_research_gemini_source_tenant()
    if not tenant or not source_tenant or tenant == source_tenant:
        return ""
    if not bool(getattr(settings, "research_enabled", False)):
        return ""
    if _truthy_env("GOOGLE_GENAI_USE_VERTEXAI", False):
        return ""
    if effective_research_gemini_api_key(settings):
        return ""
    if normalize_distribution_mode(getattr(settings, "distribution_mode", "private")) != "private":
        return ""
    if not bool(getattr(settings, "real_trading_approved", False)):
        return ""

    row = repo.latest_runtime_credentials(tenant_id=source_tenant) or {}
    model_secret_name = str(row.get("model_secret_name") or "").strip()
    if not model_secret_name:
        logger.warning(
            "[yellow]Shared research Gemini skipped[/yellow] tenant=%s source_tenant=%s reason=missing_model_secret",
            tenant,
            source_tenant,
        )
        return ""

    cli = _cli()
    try:
        payload = cli._load_secret_json(
            project=settings.google_cloud_project,
            secret_name=model_secret_name,
            version="latest",
        )
    except Exception as exc:
        logger.warning(
            "[yellow]Shared research Gemini secret load failed[/yellow] tenant=%s source_tenant=%s secret=%s err=%s",
            tenant,
            source_tenant,
            model_secret_name,
            str(exc),
        )
        return ""

    providers = parse_model_secret_providers(payload if isinstance(payload, dict) else {})
    gemini_key = str((providers.get("gemini") or {}).get("api_key") or "").strip()
    if not gemini_key:
        logger.warning(
            "[yellow]Shared research Gemini skipped[/yellow] tenant=%s source_tenant=%s reason=missing_gemini_key",
            tenant,
            source_tenant,
        )
        return ""

    settings.research_gemini_api_key = gemini_key
    settings.research_gemini_source = "shared_live_tenant"
    settings.research_gemini_source_tenant = source_tenant
    logger.info(
        "[cyan]Shared research Gemini applied[/cyan] tenant=%s source_tenant=%s",
        tenant,
        source_tenant,
    )
    return source_tenant


def _reconcile_excluded_tickers(settings: Settings) -> list[str]:
    """Returns normalized reconciliation exclusions from runtime settings/env."""
    configured = list(getattr(settings, "reconcile_excluded_tickers", []) or [])
    if not configured:
        configured = _csv_env("ARENA_RECONCILE_EXCLUDED_TICKERS")
    out: list[str] = []
    for token in configured:
        normalized = str(token or "").strip().upper()
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _allow_checkpoint_rebuild_recovery(*, explicit: bool = False) -> bool:
    """Returns whether reconciliation recovery may append fresh checkpoints."""
    if explicit:
        return True
    return _truthy_env("ARENA_RECONCILIATION_CHECKPOINT_REBUILD_ENABLED", False)


def _int_env(name: str, default: int) -> int:
    """Parses integer env values with fallback."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError:
        return int(default)


def _float_env(name: str, default: float) -> float:
    """Parses float env values with fallback."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        return float(default)


def _new_run_id(prefix: str) -> str:
    """Builds a compact append-only run token."""
    token = str(prefix or "").strip().lower() or "run"
    return f"{token}_{uuid4().hex[:12]}"


def _cloud_run_log_uri(settings: Settings, *, job_name: str | None = None, execution_name: str | None = None) -> str | None:
    """Builds a Cloud Logging console URL when execution metadata is available."""
    job = str(job_name or os.getenv("CLOUD_RUN_JOB") or "").strip()
    execution = str(execution_name or os.getenv("CLOUD_RUN_EXECUTION") or "").strip()
    project = str(settings.google_cloud_project or "").strip()
    location = str(os.getenv("ARENA_CLOUD_RUN_REGION") or os.getenv("GOOGLE_CLOUD_REGION") or settings.bq_location or "").strip()
    if not project or not job or not execution or not location:
        return None
    filt = (
        'resource.type="cloud_run_job"\n'
        f'resource.labels.job_name="{job}"\n'
        f'resource.labels.location="{location}"\n'
        f'labels."run.googleapis.com/execution_name"="{execution}"'
    )
    from urllib.parse import quote

    return f"https://console.cloud.google.com/logs/viewer?project={project}&advancedFilter={quote(filt, safe='')}"


def _append_tenant_run_status(
    repo: object,
    settings: Settings,
    *,
    tenant: str,
    run_id: str,
    run_type: str,
    status: str,
    reason_code: str | None = None,
    stage: str | None = None,
    started_at=None,
    finished_at=None,
    message: str | None = None,
    detail: dict | None = None,
) -> None:
    """Appends one tenant run status row when the repository supports it."""
    append = getattr(repo, "append_tenant_run_status", None)
    if not callable(append):
        return
    try:
        append(
            tenant_id=str(tenant or "").strip().lower() or "local",
            run_id=str(run_id or "").strip() or _new_run_id(run_type),
            run_type=str(run_type or "").strip().lower() or "unknown",
            status=str(status or "").strip().lower() or "unknown",
            reason_code=str(reason_code or "").strip().lower() or None,
            stage=str(stage or "").strip().lower() or None,
            started_at=started_at,
            finished_at=finished_at,
            message=str(message or "").strip() or None,
            job_name=str(os.getenv("CLOUD_RUN_JOB") or "").strip() or None,
            execution_name=str(os.getenv("CLOUD_RUN_EXECUTION") or "").strip() or None,
            log_uri=_cloud_run_log_uri(settings),
            detail=detail or {},
        )
    except Exception as exc:
        logger.warning("[yellow]Tenant run status append failed[/yellow] tenant=%s err=%s", tenant, str(exc))


def _append_tenant_run_status_many(
    repo: object,
    settings: Settings,
    *,
    tenants: list[str],
    run_ids: dict[str, str] | None = None,
    run_type: str,
    status: str,
    reason_code: str | None = None,
    stage: str | None = None,
    started_at=None,
    finished_at=None,
    message: str | None = None,
    detail: dict | None = None,
) -> None:
    """Appends the same run status across many tenants."""
    ids = dict(run_ids or {})
    for tenant in tenants:
        token = str(tenant or "").strip().lower()
        if not token:
            continue
        _append_tenant_run_status(
            repo,
            settings,
            tenant=token,
            run_id=ids.get(token) or _new_run_id(run_type),
            run_type=run_type,
            status=status,
            reason_code=reason_code,
            stage=stage,
            started_at=started_at,
            finished_at=finished_at,
            message=message,
            detail=detail,
        )


def _resolve_batch_tenants(repo: BigQueryRepository, *, fallback: str = "local") -> list[str]:
    """Resolves tenant list for multi-tenant batch execution."""
    demo_tokens = _parse_tenant_tokens(os.getenv("ARENA_PUBLIC_DEMO_TENANT"))
    demo_tenant = demo_tokens[0] if demo_tokens else ""
    explicit = _parse_tenant_tokens(os.getenv("ARENA_BATCH_TENANTS"))
    if explicit:
        if demo_tenant and demo_tenant not in explicit:
            explicit.append(demo_tenant)
        return explicit
    try:
        tenants = list(repo.list_runtime_tenants(limit=500))
    except Exception as exc:
        logger.warning("[yellow]Failed to list runtime tenants[/yellow] err=%s", str(exc))
        tenants = []
    if tenants:
        if demo_tenant and demo_tenant not in tenants:
            tenants.append(demo_tenant)
        return tenants
    if demo_tenant:
        return [demo_tenant]
    raise RuntimeError(
        f"no runtime tenants resolved and no demo tenant configured (fallback={str(fallback or '').strip().lower() or 'local'})"
    )


def _task_shard_spec() -> tuple[int, int] | None:
    """Returns task shard (index, count) from env when configured."""
    raw_index = os.getenv("ARENA_TASK_SHARD_INDEX", os.getenv("CLOUD_RUN_TASK_INDEX"))
    raw_count = os.getenv("ARENA_TASK_SHARD_COUNT", os.getenv("CLOUD_RUN_TASK_COUNT"))
    if raw_index is None or raw_count is None:
        return None
    try:
        index = int(str(raw_index).strip())
        count = int(str(raw_count).strip())
    except ValueError:
        return None
    if count <= 1 or index < 0 or index >= count:
        return None
    return index, count


def _partition_tenants_for_task(tenants: list[str]) -> list[str]:
    """Applies deterministic round-robin tenant sharding per Cloud Run task."""
    clean = [str(t or "").strip().lower() for t in tenants if str(t or "").strip()]
    if not clean:
        return []
    shard = _task_shard_spec()
    if shard is None:
        return clean
    index, count = shard
    assigned = [tenant for pos, tenant in enumerate(sorted(clean)) if pos % count == index]
    logger.info(
        "[cyan]Task shard[/cyan] index=%d count=%d assigned=%d total=%d",
        index,
        count,
        len(assigned),
        len(clean),
    )
    return assigned


def _load_secret_json(*, project: str, secret_name: str, version: str = "latest") -> dict:
    """Loads one JSON secret payload from Secret Manager."""
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_name}/versions/{version}"
    response = client.access_secret_version(request={"name": name})
    payload = response.payload.data.decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, dict):
        return data
    return {}


def _apply_tenant_runtime_credentials(
    settings: Settings,
    repo: BigQueryRepository,
    *,
    tenant_id: str | None = None,
) -> dict | None:
    """Hydrates runtime keys/secrets from tenant metadata + Secret Manager."""
    cli = _cli()
    tenant = str(tenant_id or _tenant_id() or "").strip().lower()
    if not tenant:
        return None

    row = repo.latest_runtime_credentials(tenant_id=tenant) or {}
    if not row:
        logger.warning("[yellow]Tenant runtime credentials not found[/yellow] tenant=%s", tenant)
        return None

    kis_secret_name = str(row.get("kis_secret_name") or "").strip()
    model_secret_name = str(row.get("model_secret_name") or "").strip()
    kis_env = str(row.get("kis_env") or "").strip().lower()

    if kis_secret_name:
        settings.kis_secret_name = kis_secret_name
    if kis_env in {"real", "demo"}:
        settings.kis_env = kis_env

    if model_secret_name:
        try:
            payload = cli._load_secret_json(
                project=settings.google_cloud_project,
                secret_name=model_secret_name,
                version="latest",
            )
        except Exception as exc:
            logger.warning(
                "[yellow]Model secret load failed[/yellow] tenant=%s secret=%s err=%s",
                tenant,
                model_secret_name,
                str(exc),
            )
            payload = {}

        if isinstance(payload, dict):
            cli.apply_model_secret_payload(settings, payload)

    logger.info(
        "[cyan]Tenant runtime credentials applied[/cyan] tenant=%s kis_secret=%s model_secret=%s openai=%s gemini=%s anthropic=%s",
        tenant,
        (settings.kis_secret_name or "-"),
        (model_secret_name or "-"),
        "yes" if bool(settings.openai_api_key) else "no",
        "yes" if bool(settings.gemini_api_key) else "no",
        "yes" if bool(settings.anthropic_api_key) else "no",
    )
    cli.apply_distribution_mode(settings)
    return row


def _seed_rows(settings: Settings) -> list[dict]:
    """Builds deterministic demo market rows for quick cycle testing."""
    cli = _cli()
    now = cli.utc_now().astimezone(timezone.utc)
    base_rows = [
        ("005930", 74500.0, 0.021, 0.047, 0.018, 0.11),
        ("000660", 183000.0, -0.013, 0.028, 0.026, 0.07),
        ("035420", 214000.0, 0.018, 0.052, 0.022, 0.18),
        ("AAPL", 247000.0, 0.009, 0.031, 0.017, 0.05),
        ("MSFT", 601000.0, 0.006, 0.026, 0.014, 0.06),
        ("TSLA", 308000.0, -0.024, -0.011, 0.039, -0.08),
    ]
    ticker_filter = set(settings.default_universe)
    rows = []
    for ticker, px, ret5, ret20, vol20, sent in base_rows:
        if ticker in ticker_filter:
            rows.append(
                {
                    "as_of_ts": now,
                    "ticker": ticker,
                    "exchange_code": ("KRX" if ticker[:1].isdigit() else "NASD"),
                    "instrument_id": (f"KRX:{ticker}" if ticker[:1].isdigit() else f"NASD:{ticker}"),
                    "close_price_krw": px,
                    "ret_5d": ret5,
                    "ret_20d": ret20,
                    "volatility_20d": vol20,
                    "sentiment_score": sent,
                    "source": "seed_demo",
                }
            )
    return rows


def _build_agents(settings: Settings, repo: BigQueryRepository, *, tenant_id: str):
    """Builds agents according to selected mode and configured providers."""
    mode = settings.agent_mode.strip().lower()

    if mode != "adk":
        raise SystemExit("ARENA_AGENT_MODE=adk only (llm/heuristic modes removed)")

    from arena.agents.adk_agents import build_adk_agents

    agents = build_adk_agents(settings, repo, tenant_id=tenant_id)
    if not agents:
        raise SystemExit("ARENA_AGENT_MODE=adk requires OPENAI_API_KEY and/or GEMINI_API_KEY and/or ANTHROPIC_API_KEY with matching ARENA_AGENT_IDS")
    logger.info("[green]ADK agent mode[/green] active_agents=%s", ",".join([a.agent_id for a in agents]))
    return agents


def _build_runtime(
    live: bool,
    *,
    require_kis: bool,
    tenant_id: str | None = None,
    require_tenant_runtime_credentials: bool = False,
    execution_market: str = "",
):
    """Builds the runtime graph for one arena command execution."""
    cli = _cli()
    settings = cli.load_settings()
    cli.configure_logging(settings.log_level, settings.log_format)
    cli._validate_or_exit(settings)

    tenant = str(tenant_id or _tenant_id() or "local").strip().lower() or "local"
    repo = cli._repo_or_exit(settings, tenant_id=tenant)
    repo.ensure_dataset()
    repo.ensure_tables()
    if require_tenant_runtime_credentials:
        settings.kis_secret_name = ""
        settings.kis_api_key = ""
        settings.kis_api_secret = ""
        settings.kis_paper_api_key = ""
        settings.kis_paper_api_secret = ""
        settings.kis_account_no = ""
        settings.openai_api_key = ""
        settings.gemini_api_key = ""
        settings.anthropic_api_key = ""
        settings.research_gemini_api_key = ""
        settings.research_gemini_source = ""
        settings.research_gemini_source_tenant = ""

    runtime_row = cli._apply_tenant_runtime_credentials(settings, repo, tenant_id=tenant)
    if require_tenant_runtime_credentials:
        if not runtime_row:
            raise RuntimeError(f"tenant runtime credentials missing: tenant={tenant}")
        kis_secret = str(runtime_row.get("kis_secret_name") or "").strip()
        model_secret = str(runtime_row.get("model_secret_name") or "").strip()
        if require_kis and not kis_secret:
            raise RuntimeError(f"tenant kis_secret_name missing: tenant={tenant}")
        if not model_secret:
            raise RuntimeError(f"tenant model_secret_name missing: tenant={tenant}")

    cli.apply_runtime_overrides(settings, repo, tenant_id=tenant)
    cli._apply_shared_research_gemini(settings, repo, tenant_id=tenant)
    cli._apply_market_override(settings, execution_market)
    settings.trading_mode = "live" if live else "paper"
    cli._validate_or_exit(settings, require_kis=require_kis, require_llm=True, live=live)

    memory = cli.MemoryStore(repo, trading_mode=settings.trading_mode, memory_policy=settings.memory_policy)
    board = cli.BoardStore(repo, trading_mode=settings.trading_mode)
    context_builder = cli.ContextBuilder(repo=repo, memory=memory, board=board, settings=settings)
    risk_engine = cli.RiskEngine(settings=settings)

    if live:
        if settings.kis_order_endpoint:
            broker = cli.KISHttpBroker(settings=settings)
            logger.info("[cyan]Live broker[/cyan] mode=http-endpoint")
        else:
            broker = cli.KISOpenTradingBroker(settings=settings)
            logger.info("[cyan]Live broker[/cyan] mode=open-trading-api env=%s", settings.kis_env)
    else:
        broker = cli.PaperBroker()

    gateway = cli.ExecutionGateway(
        repo=repo,
        risk_engine=risk_engine,
        broker=broker,
        memory_store=memory,
        agent_configs=settings.agent_configs,
    )

    agents = cli._build_agents(settings, repo, tenant_id=tenant)
    orchestrator = cli.ArenaOrchestrator(
        settings=settings,
        context_builder=context_builder,
        board_store=board,
        gateway=gateway,
        agents=agents,
    )
    return settings, repo, orchestrator
