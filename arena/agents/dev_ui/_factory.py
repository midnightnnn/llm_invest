from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from arena.agents.adk_runner_bootstrap import (
    build_agent,
    resolve_adk_tools,
    resolve_max_tool_events,
)
from arena.agents.adk_tool_helpers import (
    apply_tool_schema_metadata,
    noop_search_tool_memories,
    noop_update_candidate_ledger,
)
from arena.config import AgentConfig, Settings, apply_runtime_overrides, load_settings
from arena.data.factory import get_repository
from arena.providers.registry import default_model_for_provider
from arena.tools.default_registry import build_default_registry

logger = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = frozenset({"gemini", "gpt", "claude"})


def _env_bool(name: str, default: bool) -> bool:
    token = str(os.getenv(name, "") or "").strip().lower()
    if not token:
        return default
    return token in {"1", "true", "yes", "y", "on"}


def _dev_tenant_id() -> str:
    return (
        str(os.getenv("ARENA_DEV_UI_TENANT_ID") or os.getenv("ARENA_TENANT_ID") or "local")
        .strip()
        .lower()
        or "local"
    )


def _repo_default_local_db_path() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "arena.duckdb"


def _ensure_default_local_db_path() -> None:
    if not str(os.getenv("ARENA_LOCAL_DB_PATH") or "").strip():
        os.environ["ARENA_LOCAL_DB_PATH"] = str(_repo_default_local_db_path())


def _force_local_dev_settings(settings: Settings) -> Settings:
    settings.arena_mode = "local"
    settings.trading_mode = "paper"
    settings.kis_env = "demo"
    settings.allow_live_trading = False
    settings.real_trading_approved = False
    return settings


def _agent_config_for_provider(settings: Settings, provider: str) -> AgentConfig:
    config = settings.agent_configs.get(provider)
    if config is not None:
        return config
    return AgentConfig(
        agent_id=provider,
        provider=provider,
        model=default_model_for_provider(settings, provider),
        capital_krw=float(settings.agent_capitals.get(provider, settings.sleeve_capital_krw)),
    )


def build_dev_agent(provider: str) -> Any:
    """Builds one module-level ADK ``root_agent`` for ``adk web`` discovery."""
    provider_key = str(provider or "").strip().lower()
    if provider_key not in _SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported dev UI provider: {provider!r}. Expected one of: {supported}")

    tenant_id = _dev_tenant_id()
    _ensure_default_local_db_path()
    settings = _force_local_dev_settings(load_settings())
    repo = get_repository(settings, tenant_id=tenant_id)
    if _env_bool("ARENA_DEV_UI_ENSURE_TABLES", True):
        repo.ensure_tables()

    settings = _force_local_dev_settings(apply_runtime_overrides(settings, repo, tenant_id))
    agent_config = _agent_config_for_provider(settings, provider_key)
    registry = build_default_registry(repo, settings, tenant_id=tenant_id)
    max_tool_events = resolve_max_tool_events(settings)
    tool_events: list[dict[str, Any]] = []
    tool_resolution = resolve_adk_tools(
        repo=repo,
        tenant_id=tenant_id,
        agent_config=agent_config,
        registry=registry,
        settings=settings,
        agent_id=provider_key,
        tool_events=tool_events,
        update_candidate_ledger=noop_update_candidate_ledger,
        search_tool_memories=noop_search_tool_memories,
        apply_tool_schema_metadata=apply_tool_schema_metadata,
    )
    logger.info(
        "Built ADK dev UI agent provider=%s tenant=%s tools=%d",
        provider_key,
        tenant_id,
        len(tool_resolution.adk_tools),
    )
    return build_agent(
        agent_id=provider_key,
        provider=provider_key,
        settings=settings,
        repo=repo,
        tenant_id=tenant_id,
        agent_config=agent_config,
        max_tool_events=max_tool_events,
        adk_tools=tool_resolution.adk_tools,
    )
