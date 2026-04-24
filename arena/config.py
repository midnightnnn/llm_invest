from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any

from dotenv import load_dotenv
from arena.memory.policy import apply_memory_policy_to_settings, default_memory_policy, normalize_memory_policy
from arena.providers.registry import (
    canonical_provider,
    default_model_for_provider,
    list_adk_provider_specs,
    provider_alias_map,
    provider_has_credentials,
)

load_dotenv()
logger = logging.getLogger(__name__)


def _to_bool(value: str | None, default: bool = False) -> bool:
    """Parses common truthy strings into a boolean."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float(value: str | None, default: float) -> float:
    """Parses float with a safe default fallback."""
    if value is None or not value.strip():
        return default
    return float(value)


def _to_int(value: str | None, default: int) -> int:
    """Parses integer with a safe default fallback."""
    if value is None or not value.strip():
        return default
    return int(value)


def _to_optional_int(value: str | None) -> int | None:
    """Parses integer returning None when unset or blank."""
    if value is None or not value.strip():
        return None
    return int(value)


def _csv(value: str | None, default: list[str]) -> list[str]:
    """Parses delimited values into a clean list (comma/pipe/semicolon)."""
    if value is None or not value.strip():
        return default
    tokens = re.split(r"[,|;]", value)
    return [token.strip() for token in tokens if token.strip()]


def _market_tokens(value: str | None) -> list[str]:
    """Parses comma-delimited market config into normalized tokens."""
    return [token.strip().lower() for token in str(value or "").split(",") if token.strip()]


def _to_bool_token(value: Any, default: bool) -> bool:
    """Parses truthy runtime-config values into a boolean."""
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.lower() in {"1", "true", "yes", "y", "on"}



class SettingsError(RuntimeError):
    """Raised when required settings are missing or inconsistent."""


@dataclass(slots=True)
class Settings:
    """Holds runtime configuration for arena services."""

    google_cloud_project: str
    bq_dataset: str
    bq_location: str

    agent_ids: list[str]
    agent_mode: str
    base_currency: str
    sleeve_capital_krw: float
    log_level: str
    log_format: str

    trading_mode: str
    kis_order_endpoint: str
    kis_api_key: str
    kis_api_secret: str
    kis_paper_api_key: str
    kis_paper_api_secret: str
    kis_account_no: str
    kis_account_product_code: str
    kis_account_key_suffix: str
    kis_env: str
    kis_target_market: str
    kis_overseas_quote_excd: str
    kis_overseas_order_excd: str
    kis_us_natn_cd: str
    kis_us_tr_mket_cd: str
    kis_secret_name: str
    kis_secret_version: str
    kis_http_timeout_seconds: int
    kis_http_max_retries: int
    kis_http_backoff_base_seconds: float
    kis_http_backoff_max_seconds: float
    kis_confirm_fills: bool
    kis_confirm_timeout_seconds: int
    kis_confirm_poll_seconds: float

    usd_krw_rate: float
    market_sync_history_days: int

    max_order_krw: float
    max_daily_turnover_ratio: float
    max_position_ratio: float
    min_cash_buffer_ratio: float
    ticker_cooldown_seconds: int
    max_daily_orders: int
    estimated_fee_bps: float

    context_max_board_posts: int
    context_max_memory_events: int
    context_max_market_rows: int

    openai_api_key: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str
    research_gemini_model: str
    llm_timeout_seconds: int

    default_universe: list[str]
    allow_live_trading: bool
    live_slippage_bps_base: float = 8.0
    live_slippage_bps_impact: float = 12.0
    live_slippage_bps_max: float = 80.0
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_use_vertexai: bool = False
    research_gemini_api_key: str = ""
    research_gemini_source: str = ""
    research_gemini_source_tenant: str = ""

    # Role-specific LLM timeout overrides. None falls back to llm_timeout_seconds.
    llm_timeout_trading_seconds: int | None = None
    llm_timeout_research_seconds: int | None = None
    llm_timeout_compaction_seconds: int | None = None
    # Runtime override (e.g., CLI --timeout flag). Highest precedence when set.
    llm_timeout_runtime_override_seconds: int | None = None

    research_enabled: bool = True
    research_max_tickers: int = 5
    research_mover_top_n: int = 3
    research_earnings_lookahead_days: int = 7
    adk_max_tool_events: int = 120
    universe_run_top_n: int = 400
    universe_per_exchange_cap: int = 200
    us_quote_exchanges: list[str] = field(default_factory=lambda: ["NAS", "NYS"])
    reddit_sentiment_enabled: bool = False
    forecast_mode: str = "all"
    forecast_table: str = ""
    fred_api_key: str = ""
    ecos_api_key: str = ""
    usd_krw_fx_symbol: str = ""
    usd_krw_fx_market_div_code: str = "X"
    reconcile_excluded_tickers: list[str] = field(default_factory=list)
    agent_capitals: dict[str, float] = field(default_factory=dict)

    dividend_sync_enabled: bool = True
    dividend_lookback_days: int = 90
    dividend_withholding_rate_us: float = 0.15

    agent_configs: dict[str, "AgentConfig"] = field(default_factory=dict)
    memory_compaction_enabled: bool = True
    memory_compaction_cycle_event_limit: int = 12
    memory_compaction_recent_lessons_limit: int = 4
    memory_compaction_max_reflections: int = 3
    memory_policy: dict[str, Any] = field(default_factory=dict)
    distribution_mode: str = "private"
    real_trading_approved: bool = False
    provider_secrets: dict[str, dict[str, str]] = field(default_factory=dict)
    autonomy_working_set_enabled: bool = True
    autonomy_tool_default_candidates_enabled: bool = True
    autonomy_opportunity_context_enabled: bool = True

    def timeout_for(self, role: str) -> int:
        """Resolves LLM timeout (seconds) for a given agent role.

        Precedence: runtime override (CLI --timeout) > role-specific env var >
        llm_timeout_seconds fallback. Unknown roles use the same chain.
        """
        runtime = self.llm_timeout_runtime_override_seconds
        if runtime is not None and runtime > 0:
            return int(runtime)
        override = {
            "trading": self.llm_timeout_trading_seconds,
            "research": self.llm_timeout_research_seconds,
            "compaction": self.llm_timeout_compaction_seconds,
        }.get(role)
        if override is not None and override > 0:
            return int(override)
        return int(self.llm_timeout_seconds)


# Provider alias mapping: agent_id keyword -> canonical provider name
PROVIDER_ALIASES: dict[str, str] = provider_alias_map()

_DISTRIBUTION_MODES = {"private", "paper_only", "simulated_only"}


def normalize_distribution_mode(value: str | None) -> str:
    """Normalizes release distribution mode tokens."""
    token = str(value or "").strip().lower()
    if token in _DISTRIBUTION_MODES:
        return token
    return "private"


def distribution_allows_real_kis_credentials(settings: Settings) -> bool:
    """Returns True when real KIS credentials should be accepted/exposed for this tenant."""
    return (
        normalize_distribution_mode(getattr(settings, "distribution_mode", "private")) == "private"
        and bool(getattr(settings, "real_trading_approved", False))
    )


def distribution_allows_paper_kis_credentials(settings: Settings) -> bool:
    """Returns True when paper/demo KIS credentials should be accepted/exposed."""
    return normalize_distribution_mode(getattr(settings, "distribution_mode", "private")) in {"private", "paper_only"}


def distribution_uses_broker_credentials(settings: Settings) -> bool:
    """Returns True when this deployment mode expects any broker credentials."""
    return normalize_distribution_mode(getattr(settings, "distribution_mode", "private")) != "simulated_only"


def apply_distribution_mode(settings: Settings) -> Settings:
    """Applies deployment-mode guardrails while keeping private mode unchanged."""
    mode = normalize_distribution_mode(getattr(settings, "distribution_mode", "private"))
    settings.distribution_mode = mode
    if mode == "paper_only":
        settings.kis_env = "demo"
        settings.allow_live_trading = True
    elif mode == "simulated_only":
        settings.kis_env = "demo"
        settings.allow_live_trading = False
    return settings


def effective_research_gemini_api_key(settings: Settings) -> str:
    """Returns the Gemini API key used for research generation."""
    research_key = getattr(settings, "research_gemini_api_key", "").strip()
    if research_key:
        return research_key
    return getattr(settings, "gemini_api_key", "").strip()


def research_generation_status(settings: Settings) -> dict[str, Any]:
    """Returns tenant-scoped Gemini research generation capability state."""
    enabled_by_config = bool(getattr(settings, "research_enabled", False))
    has_gemini_key = bool(getattr(settings, "gemini_api_key", "").strip())
    has_research_gemini_key = bool(effective_research_gemini_api_key(settings))
    research_source = getattr(settings, "research_gemini_source", "").strip().lower()
    research_source_tenant = getattr(settings, "research_gemini_source_tenant", "").strip().lower()
    use_vertex = _to_bool(os.getenv("GOOGLE_GENAI_USE_VERTEXAI"), False)
    shared_source_tenant = str(os.getenv("ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT", "") or "").strip().lower()
    public_fallback_tenant = str(os.getenv("ARENA_PUBLIC_DEMO_TENANT", "") or shared_source_tenant or "").strip().lower()
    vertex_limited_to_live_tenants = bool(use_vertex and shared_source_tenant)
    vertex_allowed = True
    if vertex_limited_to_live_tenants:
        vertex_allowed = (
            normalize_distribution_mode(getattr(settings, "distribution_mode", "private")) == "private"
            and bool(getattr(settings, "real_trading_approved", False))
        )

    code = "enabled"
    message = "Gemini native grounding is enabled for research briefing generation."
    if not enabled_by_config:
        code = "disabled_by_config"
        message = "Research briefing generation is disabled by config."
    elif research_source == "shared_live_tenant" and has_research_gemini_key:
        code = "shared_live_tenant"
        source_label = research_source_tenant or "configured operator tenant"
        message = (
            "Research briefing generation uses operator-managed Gemini shared from "
            f"{source_label} for approved live tenants."
        )
    elif has_research_gemini_key:
        code = "enabled"
        message = "Gemini native grounding is enabled for research briefing generation."
    elif use_vertex and vertex_allowed:
        code = "vertex_enabled"
        if vertex_limited_to_live_tenants:
            source_label = shared_source_tenant or "operator project"
            message = (
                "Research briefing generation uses operator-managed Gemini via Vertex AI "
                f"for approved live tenants. Source tenant: {source_label}."
            )
        else:
            message = "Research briefing generation uses Gemini via Vertex AI."
    else:
        code = "missing_gemini_key"
        message = (
            "Gemini native grounding is unavailable for new research briefings. "
            "Add GEMINI_API_KEY or enable GOOGLE_GENAI_USE_VERTEXAI. Cached briefings remain readable."
        )

    return {
        "code": code,
        "enabled_by_config": enabled_by_config,
        "has_gemini_key": has_gemini_key,
        "has_research_gemini_key": has_research_gemini_key,
        "uses_vertex": use_vertex,
        "vertex_limited_to_live_tenants": vertex_limited_to_live_tenants,
        "shared_source_tenant": shared_source_tenant,
        "public_fallback_tenant": public_fallback_tenant,
        "research_source": research_source,
        "research_source_tenant": research_source_tenant,
        "can_generate": code in {"enabled", "vertex_enabled", "shared_live_tenant"},
        "model": getattr(settings, "research_gemini_model", "").strip(),
        "message": message,
    }


@dataclass(slots=True)
class AgentConfig:
    """Per-agent configuration — overrides global Settings when present."""

    agent_id: str                                        # free-form name (e.g. "aggressive-gpt")
    provider: str                                        # canonical: "gpt" | "gemini" | "claude"
    model: str                                           # e.g. "gpt-5.2"
    capital_krw: float
    target_market: str = ""                              # "" → use global kis_target_market
    system_prompt: str | None = None                     # None → use global prompt
    risk_overrides: dict[str, float | int] | None = None # None → use global risk
    disabled_tools: list[str] | None = None              # None → use global tool config
    llm_params: dict[str, Any] | None = None             # provider-native SDK params (effort/thinking_level/temperature/…)


def _default_model_for_provider(settings: Settings, provider: str) -> str:
    """Returns the default model token for a canonical provider."""
    return default_model_for_provider(settings, provider)


def _provider_has_usable_credentials(settings: Settings, provider_id: str) -> bool:
    """Checks whether a provider has credentials OR an equivalent transport fallback."""
    token = str(provider_id or "").strip().lower()
    if not token:
        return False
    if token == "claude" and settings.anthropic_use_vertexai:
        return bool(settings.google_cloud_project.strip())
    return provider_has_credentials(settings, token)


def _drop_agents_missing_credentials(settings: Settings) -> list[str]:
    """Removes agents whose providers have no usable credentials.

    Behavior: silently skipping a missing key is graceful when a tenant simply
    doesn't want that provider (e.g., gpt-only tenants inherit a global
    ARENA_AGENT_IDS=gemini,gpt,claude). A warning is emitted per drop so
    operator typos aren't hidden. Returns the dropped agent_ids.
    """
    dropped: list[str] = []
    kept_ids: list[str] = []
    for agent_id in list(settings.agent_ids):
        config = settings.agent_configs.get(agent_id)
        provider = str(config.provider).strip().lower() if config else ""
        if not provider:
            kept_ids.append(agent_id)
            continue
        if _provider_has_usable_credentials(settings, provider):
            kept_ids.append(agent_id)
            continue
        dropped.append(agent_id)
        logger.warning(
            "[yellow]Agent skipped: no credentials[/yellow] agent_id=%s provider=%s "
            "(drop this agent from ARENA_AGENT_IDS or set the provider's API key)",
            agent_id,
            provider,
        )
    if dropped:
        settings.agent_ids = kept_ids
        for aid in dropped:
            settings.agent_configs.pop(aid, None)
            settings.agent_capitals.pop(aid, None)
    return dropped


def normalize_agent_settings(settings: Settings) -> Settings:
    """Normalizes agent ids/configs/capitals into a single canonical shape."""
    normalized_ids: list[str] = []
    normalized_capitals: dict[str, float] = {}
    normalized_configs: dict[str, AgentConfig] = {}

    for raw_agent_id in settings.agent_ids:
        agent_id = str(raw_agent_id or "").strip().lower()
        if not agent_id or agent_id in normalized_ids:
            continue
        normalized_ids.append(agent_id)

        existing = settings.agent_configs.get(agent_id)
        provider = ""
        model = ""
        capital = settings.agent_capitals.get(agent_id, settings.sleeve_capital_krw)
        target_market = ""
        system_prompt: str | None = None
        risk_overrides: dict[str, float | int] | None = None
        disabled_tools: list[str] | None = None
        llm_params: dict[str, Any] | None = None

        if existing is not None:
            provider = str(existing.provider or "").strip().lower()
            model = str(existing.model or "").strip()
            target_market = str(existing.target_market or "").strip().lower()
            if isinstance(existing.system_prompt, str) and existing.system_prompt.strip():
                system_prompt = existing.system_prompt.strip()
            if isinstance(existing.risk_overrides, dict) and existing.risk_overrides:
                risk_overrides = dict(existing.risk_overrides)
            if isinstance(existing.disabled_tools, list):
                cleaned_tools = [str(x).strip() for x in existing.disabled_tools if str(x).strip()]
                disabled_tools = cleaned_tools or None
            if isinstance(existing.llm_params, dict) and existing.llm_params:
                llm_params = dict(existing.llm_params)
            try:
                existing_capital = float(existing.capital_krw)
                if existing_capital > 0:
                    capital = existing_capital
            except (TypeError, ValueError):
                pass

        if not provider:
            provider = canonical_provider(agent_id)

        try:
            capital = float(capital)
        except (TypeError, ValueError):
            capital = float(settings.sleeve_capital_krw)
        if capital <= 0:
            capital = float(settings.sleeve_capital_krw)
        normalized_capitals[agent_id] = capital

        if not provider:
            continue

        normalized_configs[agent_id] = AgentConfig(
            agent_id=agent_id,
            provider=provider,
            model=model or _default_model_for_provider(settings, provider),
            capital_krw=capital,
            target_market=target_market,
            system_prompt=system_prompt,
            risk_overrides=risk_overrides,
            disabled_tools=disabled_tools,
            llm_params=llm_params,
        )

    settings.agent_ids = normalized_ids
    settings.agent_capitals = normalized_capitals
    settings.agent_configs = normalized_configs
    return settings


def merge_agent_risk_settings(settings: Settings, agent_config: "AgentConfig | None") -> Settings:
    """Returns Settings with per-agent risk overrides merged. Returns original if no overrides."""
    if agent_config is None or agent_config.risk_overrides is None:
        return settings
    overrides = agent_config.risk_overrides
    if not overrides:
        return settings
    kwargs: dict[str, Any] = {}
    float_fields = (
        "max_order_krw", "max_daily_turnover_ratio", "max_position_ratio",
        "min_cash_buffer_ratio", "estimated_fee_bps",
    )
    int_fields = ("ticker_cooldown_seconds", "max_daily_orders")
    for key in float_fields:
        if key in overrides:
            try:
                kwargs[key] = float(overrides[key])
            except (TypeError, ValueError):
                pass
    for key in int_fields:
        if key in overrides:
            try:
                kwargs[key] = int(overrides[key])
            except (TypeError, ValueError):
                pass
    if not kwargs:
        return settings
    return replace(settings, **kwargs)


def load_settings() -> Settings:
    """Loads application settings from environment variables."""
    mode = os.getenv("ARENA_TRADING_MODE", "paper").strip().lower()
    agent_mode = os.getenv("ARENA_AGENT_MODE", "adk").strip().lower()
    context_max_memory_events = _to_int(os.getenv("ARENA_CONTEXT_MAX_MEMORY_EVENTS"), 32)
    memory_compaction_enabled = _to_bool(os.getenv("ARENA_MEMORY_COMPACTION_ENABLED"), True)
    memory_compaction_cycle_event_limit = _to_int(os.getenv("ARENA_MEMORY_COMPACTION_CYCLE_EVENT_LIMIT"), 12)
    memory_compaction_recent_lessons_limit = _to_int(os.getenv("ARENA_MEMORY_COMPACTION_RECENT_LESSONS_LIMIT"), 4)
    memory_compaction_max_reflections = _to_int(os.getenv("ARENA_MEMORY_COMPACTION_MAX_REFLECTIONS"), 3)
    memory_embed_cache_max = _to_int(os.getenv("ARENA_MEMORY_EMBED_CACHE_MAX"), 128)
    settings = Settings(
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        bq_dataset=os.getenv("BQ_DATASET", "llm_arena"),
        bq_location=os.getenv("BQ_LOCATION", "asia-northeast3"),
        agent_ids=_csv(
            os.getenv("ARENA_AGENT_IDS"),
            ["gemini", "gpt", "claude"],
        ),
        agent_mode=agent_mode,
        base_currency=os.getenv("ARENA_BASE_CURRENCY", "KRW"),
        sleeve_capital_krw=_to_float(os.getenv("ARENA_SLEEVE_CAPITAL_KRW"), 1_000_000),
        log_level=os.getenv("ARENA_LOG_LEVEL", "INFO").upper(),
        log_format=os.getenv("ARENA_LOG_FORMAT", "").strip().lower(),
        trading_mode=mode,
        kis_order_endpoint=os.getenv("KIS_ORDER_ENDPOINT", "").strip(),
        kis_api_key=(os.getenv("KIS_API_KEY") or os.getenv("KIS_APP_KEY") or "").strip(),
        kis_api_secret=(os.getenv("KIS_API_SECRET") or os.getenv("KIS_APP_SECRET") or "").strip(),
        kis_paper_api_key=(os.getenv("KIS_PAPER_API_KEY") or os.getenv("KIS_PAPER_APP_KEY") or "").strip(),
        kis_paper_api_secret=(os.getenv("KIS_PAPER_API_SECRET") or os.getenv("KIS_PAPER_APP_SECRET") or "").strip(),
        kis_account_no=os.getenv("KIS_ACCOUNT_NO", "").strip(),
        kis_account_product_code=(os.getenv("KIS_ACCOUNT_PRODUCT_CODE") or "01").strip(),
        kis_account_key_suffix=(os.getenv("KIS_ACCOUNT_KEY_SUFFIX") or "").strip().upper(),
        kis_env=os.getenv("KIS_ENV", "demo").strip().lower(),
        kis_target_market=os.getenv("KIS_TARGET_MARKET", "nasdaq").strip().lower(),
        kis_overseas_quote_excd=os.getenv("KIS_OVERSEAS_QUOTE_EXCD", "NAS").strip().upper(),
        kis_overseas_order_excd=os.getenv("KIS_OVERSEAS_ORDER_EXCD", "NASD").strip().upper(),
        kis_us_natn_cd=os.getenv("KIS_US_NATN_CD", "840").strip(),
        kis_us_tr_mket_cd=os.getenv("KIS_US_TR_MKET_CD", "01").strip(),
        kis_secret_name=os.getenv("KIS_SECRET_NAME", "KISAPI").strip(),
        kis_secret_version=os.getenv("KIS_SECRET_VERSION", "latest").strip(),
        kis_http_timeout_seconds=_to_int(os.getenv("KIS_HTTP_TIMEOUT_SECONDS"), 20),
        kis_http_max_retries=_to_int(os.getenv("KIS_HTTP_MAX_RETRIES"), 3),
        kis_http_backoff_base_seconds=_to_float(os.getenv("KIS_HTTP_BACKOFF_BASE_SECONDS"), 0.8),
        kis_http_backoff_max_seconds=_to_float(os.getenv("KIS_HTTP_BACKOFF_MAX_SECONDS"), 8.0),
        kis_confirm_fills=_to_bool(os.getenv("KIS_CONFIRM_FILLS"), False),
        kis_confirm_timeout_seconds=_to_int(os.getenv("KIS_CONFIRM_TIMEOUT_SECONDS"), 25),
        kis_confirm_poll_seconds=_to_float(os.getenv("KIS_CONFIRM_POLL_SECONDS"), 2.0),
        usd_krw_rate=_to_float(os.getenv("ARENA_USD_KRW_RATE"), 0.0),  # no default; must come from live FX or env
        market_sync_history_days=_to_int(os.getenv("ARENA_MARKET_SYNC_HISTORY_DAYS"), 60),
        reconcile_excluded_tickers=[
            str(token).strip().upper()
            for token in _csv(os.getenv("ARENA_RECONCILE_EXCLUDED_TICKERS"), [])
            if str(token).strip()
        ],
        max_order_krw=_to_float(os.getenv("ARENA_MAX_ORDER_KRW"), 100_000_000.0),
        max_daily_turnover_ratio=_to_float(
            os.getenv("ARENA_MAX_DAILY_TURNOVER_RATIO"),
            0.65,
        ),
        max_position_ratio=_to_float(os.getenv("ARENA_MAX_POSITION_RATIO"), 1.0),
        min_cash_buffer_ratio=_to_float(os.getenv("ARENA_MIN_CASH_BUFFER_RATIO"), 0.10),
        ticker_cooldown_seconds=_to_int(
            os.getenv("ARENA_TICKER_COOLDOWN_SECONDS"),
            120,
        ),
        max_daily_orders=_to_int(os.getenv("ARENA_MAX_DAILY_ORDERS"), 0),
        estimated_fee_bps=_to_float(os.getenv("ARENA_ESTIMATED_FEE_BPS"), 10.0),
        context_max_board_posts=_to_int(os.getenv("ARENA_CONTEXT_MAX_BOARD_POSTS"), 24),
        context_max_memory_events=context_max_memory_events,
        context_max_market_rows=_to_int(os.getenv("ARENA_CONTEXT_MAX_MARKET_ROWS"), 64),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.2").strip(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview").strip(),
        research_gemini_model=os.getenv("ARENA_RESEARCH_GEMINI_MODEL", "gemini-2.5-flash").strip(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip(),
        anthropic_use_vertexai=_to_bool(os.getenv("ANTHROPIC_USE_VERTEXAI"), False),
        research_gemini_api_key=os.getenv("ARENA_RESEARCH_GEMINI_API_KEY", "").strip(),
        research_gemini_source=("env" if os.getenv("ARENA_RESEARCH_GEMINI_API_KEY", "").strip() else ""),
        llm_timeout_seconds=_to_int(os.getenv("ARENA_LLM_TIMEOUT_SECONDS"), 90),
        llm_timeout_trading_seconds=_to_optional_int(os.getenv("ARENA_LLM_TIMEOUT_TRADING_SECONDS")),
        llm_timeout_research_seconds=_to_optional_int(os.getenv("ARENA_LLM_TIMEOUT_RESEARCH_SECONDS")),
        llm_timeout_compaction_seconds=_to_optional_int(os.getenv("ARENA_LLM_TIMEOUT_COMPACTION_SECONDS")),
        default_universe=[],
        allow_live_trading=_to_bool(os.getenv("ARENA_ALLOW_LIVE_TRADING"), False),
        live_slippage_bps_base=_to_float(os.getenv("ARENA_LIVE_SLIPPAGE_BPS_BASE"), 8.0),
        live_slippage_bps_impact=_to_float(os.getenv("ARENA_LIVE_SLIPPAGE_BPS_IMPACT"), 12.0),
        live_slippage_bps_max=_to_float(os.getenv("ARENA_LIVE_SLIPPAGE_BPS_MAX"), 80.0),
        research_enabled=_to_bool(os.getenv("ARENA_RESEARCH_ENABLED"), True),
        research_max_tickers=_to_int(os.getenv("ARENA_RESEARCH_MAX_TICKERS"), 5),
        research_mover_top_n=_to_int(os.getenv("ARENA_RESEARCH_MOVER_TOP_N"), 3),
        research_earnings_lookahead_days=_to_int(os.getenv("ARENA_RESEARCH_EARNINGS_LOOKAHEAD_DAYS"), 7),
        adk_max_tool_events=_to_int(os.getenv("ARENA_ADK_MAX_TOOL_EVENTS"), 120),
        universe_run_top_n=_to_int(os.getenv("ARENA_UNIVERSE_RUN_TOP_N"), 400),
        universe_per_exchange_cap=_to_int(os.getenv("ARENA_UNIVERSE_PER_EXCHANGE_CAP"), 200),
        us_quote_exchanges=[
            str(x).strip().upper()
            for x in _csv(os.getenv("ARENA_US_QUOTE_EXCHANGES"), ["NAS", "NYS"])
            if str(x).strip()
        ],
        reddit_sentiment_enabled=_to_bool(os.getenv("ARENA_REDDIT_SENTIMENT_ENABLED"), True),
        forecast_mode=(os.getenv("ARENA_FORECAST_MODE", "all").strip().lower() or "all"),
        forecast_table=os.getenv("ARENA_FORECAST_TABLE", "").strip(),
        fred_api_key=os.getenv("FRED_API_KEY", "").strip(),
        ecos_api_key=os.getenv("ECOS_API_KEY", "").strip(),
        usd_krw_fx_symbol=os.getenv("ARENA_USD_KRW_FX_SYMBOL", "").strip().upper(),
        usd_krw_fx_market_div_code=(os.getenv("ARENA_USD_KRW_FX_MARKET_DIV_CODE", "X").strip().upper() or "X"),
        dividend_sync_enabled=_to_bool(os.getenv("ARENA_DIVIDEND_SYNC_ENABLED"), True),
        dividend_lookback_days=_to_int(os.getenv("ARENA_DIVIDEND_LOOKBACK_DAYS"), 90),
        dividend_withholding_rate_us=_to_float(os.getenv("ARENA_DIVIDEND_WITHHOLDING_RATE_US"), 0.15),
        memory_compaction_enabled=memory_compaction_enabled,
        memory_compaction_cycle_event_limit=memory_compaction_cycle_event_limit,
        memory_compaction_recent_lessons_limit=memory_compaction_recent_lessons_limit,
        memory_compaction_max_reflections=memory_compaction_max_reflections,
        memory_policy=default_memory_policy(
            context_limit=context_max_memory_events,
            embed_cache_max=memory_embed_cache_max,
            compaction_enabled=memory_compaction_enabled,
            cycle_event_limit=memory_compaction_cycle_event_limit,
            recent_lessons_limit=memory_compaction_recent_lessons_limit,
            max_reflections=memory_compaction_max_reflections,
        ),
        distribution_mode=normalize_distribution_mode(os.getenv("ARENA_DISTRIBUTION_MODE", "private")),
        autonomy_working_set_enabled=_to_bool(os.getenv("ARENA_AUTONOMY_WORKING_SET_ENABLED"), True),
        autonomy_tool_default_candidates_enabled=_to_bool(os.getenv("ARENA_AUTONOMY_TOOL_DEFAULT_CANDIDATES_ENABLED"), True),
        autonomy_opportunity_context_enabled=_to_bool(os.getenv("ARENA_AUTONOMY_OPPORTUNITY_CONTEXT_ENABLED"), True),
    )
    return normalize_agent_settings(apply_distribution_mode(settings))


def _json_list(value: str | None) -> list[Any]:
    """Parses JSON array safely; returns [] on parse/type failure."""
    raw = str(value or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        return parsed
    return []


def _json_list_or_none(value: str | None) -> list[Any] | None:
    """Parses JSON array safely; returns None when blank or invalid."""
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def _json_object(value: str | None) -> dict[str, Any]:
    """Parses JSON object safely; returns {} on parse/type failure."""
    raw = str(value or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def apply_runtime_overrides(settings: Settings, repo: Any, tenant_id: str) -> Settings:
    """Applies tenant-scoped runtime overrides from arena_config into Settings."""
    tenant = str(tenant_id or "").strip().lower() or "local"
    get_configs = getattr(repo, "get_configs", None)
    get_config = getattr(repo, "get_config", None)
    if not callable(get_configs) and not callable(get_config):
        return settings

    keys = [
        "risk_policy",
        "sleeve_capital_krw",
        "agents_config",
        "memory_policy",
        "kis_account_no",
        "kis_account_product_code",
        "kis_account_key_suffix",
        "kis_target_market",
        "reconcile_excluded_tickers",
        "universe_run_top_n",
        "universe_per_exchange_cap",
        "forecast_mode",
        "reddit_sentiment_enabled",
        "research_max_tickers",
        "research_mover_top_n",
        "research_earnings_lookahead_days",
        "distribution_mode",
        "real_trading_approved",
        "agent_autonomy_config",
    ]
    values: dict[str, str] = {}
    if callable(get_configs):
        try:
            values = dict(get_configs(tenant, keys) or {})
        except Exception as exc:
            logger.warning(
                "[yellow]Runtime config bulk load failed[/yellow] tenant=%s err=%s",
                tenant,
                str(exc),
            )
            values = {}
    if callable(get_config):
        for key in keys:
            if key in values:
                continue
            try:
                value = get_config(tenant, key)
            except Exception:
                value = None
            if value is not None:
                values[key] = str(value)

    # risk_policy
    risk = _json_object(values.get("risk_policy"))
    float_fields = (
        "max_order_krw",
        "max_daily_turnover_ratio",
        "max_position_ratio",
        "min_cash_buffer_ratio",
        "estimated_fee_bps",
    )
    int_fields = (
        "ticker_cooldown_seconds",
        "max_daily_orders",
    )
    for key in float_fields:
        if key not in risk:
            continue
        try:
            setattr(settings, key, float(risk[key]))
        except (TypeError, ValueError):
            logger.warning("[yellow]Runtime risk override skipped[/yellow] key=%s value=%s", key, risk[key])
    for key in int_fields:
        if key not in risk:
            continue
        try:
            setattr(settings, key, int(risk[key]))
        except (TypeError, ValueError):
            logger.warning("[yellow]Runtime risk override skipped[/yellow] key=%s value=%s", key, risk[key])

    # sleeve_capital_krw
    sleeve_raw = str(values.get("sleeve_capital_krw") or "").strip()
    if sleeve_raw:
        try:
            settings.sleeve_capital_krw = float(sleeve_raw)
        except ValueError:
            logger.warning(
                "[yellow]Runtime sleeve override skipped[/yellow] tenant=%s value=%s",
                tenant,
                sleeve_raw,
            )

    kis_account_no_raw = str(values.get("kis_account_no") or "").strip()
    if kis_account_no_raw:
        settings.kis_account_no = kis_account_no_raw

    kis_account_product_code_raw = str(values.get("kis_account_product_code") or "").strip()
    if kis_account_product_code_raw:
        settings.kis_account_product_code = kis_account_product_code_raw

    kis_account_key_suffix_raw = str(values.get("kis_account_key_suffix") or "").strip().upper()
    if kis_account_key_suffix_raw:
        settings.kis_account_key_suffix = kis_account_key_suffix_raw

    # kis_target_market — tenant-scoped market override
    market_raw = str(values.get("kis_target_market") or "").strip().lower()
    if market_raw:
        settings.kis_target_market = market_raw
        logger.info(
            "[cyan]Runtime market override[/cyan] tenant=%s kis_target_market=%s",
            tenant,
            market_raw,
        )

    autonomy = _json_object(values.get("agent_autonomy_config"))
    if autonomy:
        settings.autonomy_working_set_enabled = _to_bool_token(
            autonomy.get("working_set_enabled"),
            settings.autonomy_working_set_enabled,
        )
        settings.autonomy_tool_default_candidates_enabled = _to_bool_token(
            autonomy.get("tool_default_candidates_enabled"),
            settings.autonomy_tool_default_candidates_enabled,
        )
        settings.autonomy_opportunity_context_enabled = _to_bool_token(
            autonomy.get("opportunity_context_enabled"),
            settings.autonomy_opportunity_context_enabled,
        )

    runtime_excluded_tickers = [
        str(token).strip().upper()
        for token in _csv(values.get("reconcile_excluded_tickers"), [])
        if str(token).strip()
    ]
    if runtime_excluded_tickers:
        merged: list[str] = []
        for token in [*list(getattr(settings, "reconcile_excluded_tickers", [])), *runtime_excluded_tickers]:
            normalized = str(token).strip().upper()
            if normalized and normalized not in merged:
                merged.append(normalized)
        settings.reconcile_excluded_tickers = merged

    universe_run_top_n_raw = str(values.get("universe_run_top_n") or "").strip()
    if universe_run_top_n_raw:
        try:
            parsed = int(float(universe_run_top_n_raw))
            if parsed > 0:
                settings.universe_run_top_n = parsed
        except ValueError:
            logger.warning(
                "[yellow]Runtime universe override skipped[/yellow] tenant=%s key=universe_run_top_n value=%s",
                tenant,
                universe_run_top_n_raw,
            )

    universe_per_exchange_cap_raw = str(values.get("universe_per_exchange_cap") or "").strip()
    if universe_per_exchange_cap_raw:
        try:
            parsed = int(float(universe_per_exchange_cap_raw))
            if parsed > 0:
                settings.universe_per_exchange_cap = parsed
        except ValueError:
            logger.warning(
                "[yellow]Runtime universe override skipped[/yellow] tenant=%s key=universe_per_exchange_cap value=%s",
                tenant,
                universe_per_exchange_cap_raw,
            )

    forecast_mode_raw = str(values.get("forecast_mode") or "").strip().lower()
    if forecast_mode_raw:
        settings.forecast_mode = forecast_mode_raw

    if values.get("reddit_sentiment_enabled") is not None:
        settings.reddit_sentiment_enabled = _to_bool_token(
            values.get("reddit_sentiment_enabled"),
            settings.reddit_sentiment_enabled,
        )

    if values.get("real_trading_approved") is not None:
        settings.real_trading_approved = _to_bool_token(
            values.get("real_trading_approved"),
            settings.real_trading_approved,
        )

    if values.get("distribution_mode") is not None:
        settings.distribution_mode = normalize_distribution_mode(values.get("distribution_mode"))

    research_max_tickers_raw = str(values.get("research_max_tickers") or "").strip()
    if research_max_tickers_raw:
        try:
            parsed = int(float(research_max_tickers_raw))
            if parsed > 0:
                settings.research_max_tickers = parsed
        except ValueError:
            logger.warning(
                "[yellow]Runtime research override skipped[/yellow] tenant=%s key=research_max_tickers value=%s",
                tenant,
                research_max_tickers_raw,
            )

    research_mover_top_n_raw = str(values.get("research_mover_top_n") or "").strip()
    if research_mover_top_n_raw:
        try:
            parsed = int(float(research_mover_top_n_raw))
            if parsed > 0:
                settings.research_mover_top_n = parsed
        except ValueError:
            logger.warning(
                "[yellow]Runtime research override skipped[/yellow] tenant=%s key=research_mover_top_n value=%s",
                tenant,
                research_mover_top_n_raw,
            )

    research_earnings_lookahead_days_raw = str(values.get("research_earnings_lookahead_days") or "").strip()
    if research_earnings_lookahead_days_raw:
        try:
            parsed = int(float(research_earnings_lookahead_days_raw))
            if parsed > 0:
                settings.research_earnings_lookahead_days = parsed
        except ValueError:
            logger.warning(
                "[yellow]Runtime research override skipped[/yellow] tenant=%s key=research_earnings_lookahead_days value=%s",
                tenant,
                research_earnings_lookahead_days_raw,
            )

    # agents_config — unified per-agent settings (overrides agent_ids, models, capitals)
    agents_config_raw = _json_list_or_none(values.get("agents_config"))
    if agents_config_raw is not None:
        parsed_ids: list[str] = []
        parsed_capitals: dict[str, float] = {}
        parsed_models: dict[str, str] = {}
        parsed_agent_configs: dict[str, AgentConfig] = {}
        for entry in agents_config_raw:
            if not isinstance(entry, dict):
                continue
            aid = str(entry.get("id") or "").strip().lower()
            if not aid:
                continue
            parsed_ids.append(aid)
            try:
                cap = float(entry.get("capital_krw") or 0)
                if cap > 0:
                    parsed_capitals[aid] = cap
            except (TypeError, ValueError):
                pass
            model = str(entry.get("model") or "").strip()
            if model:
                parsed_models[aid] = model

            # Per-agent provider (new field; fallback: infer from id)
            provider = str(entry.get("provider") or "").strip().lower()
            if not provider:
                provider = canonical_provider(aid)

            # Per-agent system_prompt
            sys_prompt = entry.get("system_prompt")
            if isinstance(sys_prompt, str) and sys_prompt.strip():
                sys_prompt = sys_prompt.strip()
            else:
                sys_prompt = None

            # Per-agent risk_policy overrides
            risk_raw = entry.get("risk_policy")
            risk_overrides: dict[str, float | int] | None = None
            if isinstance(risk_raw, dict) and risk_raw:
                risk_overrides = {}
                for rk, rv in risk_raw.items():
                    try:
                        if "." in str(rv):
                            risk_overrides[str(rk)] = float(rv)
                        else:
                            risk_overrides[str(rk)] = int(rv)
                    except (TypeError, ValueError):
                        risk_overrides[str(rk)] = float(rv)

            # Per-agent disabled_tools
            dt_raw = entry.get("disabled_tools")
            disabled_tools: list[str] | None = None
            if isinstance(dt_raw, list):
                disabled_tools = [str(x).strip() for x in dt_raw if str(x).strip()]

            # Per-agent target_market
            agent_market = str(entry.get("target_market") or "").strip().lower()

            # Per-agent llm_params (provider-native SDK keys; validation happens in llm_params module)
            llm_raw = entry.get("llm_params")
            llm_params: dict[str, Any] | None = None
            if isinstance(llm_raw, dict) and llm_raw:
                try:
                    from arena.agents.llm_params import sanitize_llm_params
                    llm_params = sanitize_llm_params(provider, llm_raw) or None
                except Exception:
                    llm_params = None

            if provider:
                parsed_agent_configs[aid] = AgentConfig(
                    agent_id=aid,
                    provider=provider,
                    model=model or "",
                    capital_krw=parsed_capitals.get(aid, settings.sleeve_capital_krw),
                    target_market=agent_market,
                    system_prompt=sys_prompt,
                    risk_overrides=risk_overrides if risk_overrides else None,
                    disabled_tools=disabled_tools,
                    llm_params=llm_params,
                )

        settings.agent_ids = parsed_ids
        settings.agent_capitals = parsed_capitals
        settings.agent_configs = parsed_agent_configs
        # Apply per-provider models from agents_config (backward compat)
        provider_model_map = {"gpt": "openai_model", "openai": "openai_model",
                              "gemini": "gemini_model", "google": "gemini_model",
                              "claude": "anthropic_model", "anthropic": "anthropic_model"}
        for aid, model in parsed_models.items():
            attr = provider_model_map.get(aid)
            if attr:
                setattr(settings, attr, model)
    else:
        settings.agent_configs = {}
        if sleeve_raw:
            settings.agent_capitals = {}

    normalize_agent_settings(settings)

    memory_policy = normalize_memory_policy(
        values.get("memory_policy"),
        defaults=getattr(settings, "memory_policy", None) or default_memory_policy(
            context_limit=settings.context_max_memory_events,
            compaction_enabled=settings.memory_compaction_enabled,
            cycle_event_limit=settings.memory_compaction_cycle_event_limit,
            recent_lessons_limit=settings.memory_compaction_recent_lessons_limit,
            max_reflections=settings.memory_compaction_max_reflections,
        ),
    )
    apply_memory_policy_to_settings(settings, memory_policy)

    return apply_distribution_mode(settings)


def validate_settings(
    settings: Settings,
    *,
    require_kis: bool = False,
    require_llm: bool = False,
    live: bool = False,
) -> None:
    """Validates settings and raises SettingsError with actionable messages."""
    errors: list[str] = []

    if not settings.google_cloud_project.strip():
        errors.append("GOOGLE_CLOUD_PROJECT is required")
    if not settings.bq_dataset.strip():
        errors.append("BQ_DATASET is required")
    if not settings.bq_location.strip():
        errors.append("BQ_LOCATION is required")

    if settings.trading_mode not in {"paper", "live"}:
        errors.append("ARENA_TRADING_MODE must be 'paper' or 'live'")

    if settings.agent_mode != "adk":
        errors.append("ARENA_AGENT_MODE must be 'adk'")

    if live and settings.distribution_mode == "simulated_only":
        errors.append("ARENA_DISTRIBUTION_MODE=simulated_only does not support --live")
    elif live and not settings.allow_live_trading:
        errors.append("Set ARENA_ALLOW_LIVE_TRADING=true before using --live")

    if require_llm and settings.agent_mode == "adk":
        normalize_agent_settings(settings)
        _drop_agents_missing_credentials(settings)

        agent_providers = {
            str(ac.provider).strip().lower()
            for aid in settings.agent_ids
            for ac in [settings.agent_configs.get(aid)]
            if ac and str(ac.provider).strip()
        }
        use_vertex = _to_bool(os.getenv("GOOGLE_GENAI_USE_VERTEXAI"), False)

        configured_adk_specs = [
            spec for spec in list_adk_provider_specs()
            if spec.provider_id in agent_providers
        ]

        if not configured_adk_specs:
            supported_tokens = ", ".join(spec.provider_id for spec in list_adk_provider_specs())
            errors.append(
                "No agents have usable credentials. ARENA_AGENT_IDS must include at least one of: "
                f"{supported_tokens} whose provider has API credentials (or vertex fallback configured)."
            )

    if require_kis:
        market_tokens = _market_tokens(settings.kis_target_market)
        allowed_markets = {"nasdaq", "nyse", "amex", "us", "kospi"}
        if settings.distribution_mode == "simulated_only":
            errors.append("KIS-backed commands are disabled when ARENA_DISTRIBUTION_MODE=simulated_only")
        if settings.kis_env not in {"real", "demo"}:
            errors.append("KIS_ENV must be 'real' or 'demo'")
        if not market_tokens or any(token not in allowed_markets for token in market_tokens):
            errors.append("KIS_TARGET_MARKET must be one of: nasdaq, nyse, amex, us, kospi")

        has_secret = bool(settings.kis_secret_name.strip())
        has_real_keys = bool(settings.kis_api_key and settings.kis_api_secret)
        has_paper_keys = bool(settings.kis_paper_api_key and settings.kis_paper_api_secret)

        if settings.distribution_mode != "private" and has_real_keys:
            errors.append("Real KIS credentials are disabled by ARENA_DISTRIBUTION_MODE")
        if settings.distribution_mode == "paper_only" and settings.kis_env != "demo":
            errors.append("ARENA_DISTRIBUTION_MODE=paper_only requires KIS_ENV=demo")
        if settings.kis_env == "real" and not settings.real_trading_approved:
            errors.append("Tenant is not approved for real KIS trading")

        if settings.kis_env == "real" and not (has_secret or has_real_keys):
            errors.append("KIS credentials missing: set KIS_SECRET_NAME or KIS_API_KEY/KIS_API_SECRET")
        if settings.kis_env == "demo" and not (has_secret or has_paper_keys or has_real_keys):
            errors.append("KIS credentials missing: set KIS_SECRET_NAME or KIS_PAPER_API_KEY/KIS_PAPER_API_SECRET")

        if not settings.kis_account_no and not has_secret:
            errors.append("KIS account missing: set KIS_ACCOUNT_NO or KIS_SECRET_NAME")

        if settings.kis_http_timeout_seconds <= 0:
            errors.append("KIS_HTTP_TIMEOUT_SECONDS must be positive")
        if settings.kis_http_max_retries < 0:
            errors.append("KIS_HTTP_MAX_RETRIES must be >= 0")
        if settings.kis_http_backoff_base_seconds <= 0:
            errors.append("KIS_HTTP_BACKOFF_BASE_SECONDS must be positive")
        if settings.kis_http_backoff_max_seconds < settings.kis_http_backoff_base_seconds:
            errors.append("KIS_HTTP_BACKOFF_MAX_SECONDS must be >= KIS_HTTP_BACKOFF_BASE_SECONDS")

        for role_var, value in (
            ("ARENA_LLM_TIMEOUT_TRADING_SECONDS", settings.llm_timeout_trading_seconds),
            ("ARENA_LLM_TIMEOUT_RESEARCH_SECONDS", settings.llm_timeout_research_seconds),
            ("ARENA_LLM_TIMEOUT_COMPACTION_SECONDS", settings.llm_timeout_compaction_seconds),
        ):
            if value is not None and int(value) <= 0:
                errors.append(f"{role_var} must be positive when set")

        if settings.max_daily_orders < 0:
            errors.append("ARENA_MAX_DAILY_ORDERS must be >= 0")
        if settings.estimated_fee_bps < 0:
            errors.append("ARENA_ESTIMATED_FEE_BPS must be >= 0")
        if settings.live_slippage_bps_base < 0:
            errors.append("ARENA_LIVE_SLIPPAGE_BPS_BASE must be >= 0")
        if settings.live_slippage_bps_impact < 0:
            errors.append("ARENA_LIVE_SLIPPAGE_BPS_IMPACT must be >= 0")
        if settings.live_slippage_bps_max < settings.live_slippage_bps_base:
            errors.append("ARENA_LIVE_SLIPPAGE_BPS_MAX must be >= ARENA_LIVE_SLIPPAGE_BPS_BASE")
        if int(settings.adk_max_tool_events) < 10:
            errors.append("ARENA_ADK_MAX_TOOL_EVENTS must be >= 10")
        if int(settings.universe_run_top_n) <= 0:
            errors.append("ARENA_UNIVERSE_RUN_TOP_N must be >= 1")
        if int(settings.universe_per_exchange_cap) <= 0:
            errors.append("ARENA_UNIVERSE_PER_EXCHANGE_CAP must be >= 1")
        if any(token in {"nasdaq", "nyse", "amex", "us"} for token in market_tokens) and not settings.us_quote_exchanges:
            errors.append("ARENA_US_QUOTE_EXCHANGES must not be empty for US markets")

    if not settings.default_universe and int(settings.universe_run_top_n) <= 0:
        errors.append("default_universe override is required when ARENA_UNIVERSE_RUN_TOP_N is 0")

    if errors:
        raise SettingsError("; ".join(errors))
