from __future__ import annotations

import functools
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from google.adk import Agent, Runner
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.run_config import RunConfig
from google.adk.apps.app import App
from google.adk.sessions import InMemorySessionService
from google.genai import types

from arena.agents.adk_models import _resolve_model
from arena.agents.adk_prompting import _safe_json, _system_prompt
from arena.agents.adk_runner_runtime import (
    append_builtin_tool_event,
    replace_last_tool_event_result,
)
from arena.agents.adk_tool_compaction import _compact_tool_result_for_prompt
from arena.agents.adk_tool_config import _load_mcp_toolsets, _resolve_disabled_tool_ids
from arena.config import AgentConfig, Settings
from arena.data.bq import BigQueryRepository
from arena.memory.policy import memory_react_injection_enabled
from arena.memory.query_builders import build_memory_query
from arena.tools.registry import ToolEntry, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerIdentity:
    app_name: str
    user_id: str
    session_id: str


@dataclass(frozen=True)
class ToolResolution:
    adk_tools: list[Any]
    wrapped_tool_names: set[str]
    disabled_tool_ids: set[str]
    builtin_count: int
    mcp_toolset_count: int


@dataclass(frozen=True)
class RunnerBootstrap:
    agent: Agent
    runner: Runner
    session_service: InMemorySessionService


def runner_identity(agent_id: str) -> RunnerIdentity:
    return RunnerIdentity(
        app_name=f"llm_arena_{agent_id}",
        user_id="arena",
        session_id=f"{agent_id}_react",
    )


def resolve_max_tool_events(settings: Settings) -> int:
    try:
        max_tools = int(getattr(settings, "adk_max_tool_events", 120) or 120)
    except ValueError:
        max_tools = 120
    return max(10, min(max_tools, 400))


def build_run_config(max_tool_events: int) -> RunConfig:
    return RunConfig(max_llm_calls=max_tool_events)


def build_tool_wrapper(
    entry: ToolEntry,
    *,
    settings: Settings,
    agent_id: str,
    tool_events: list[dict[str, Any]],
    update_candidate_ledger: Callable[[str, dict[str, Any], Any], None],
    search_tool_memories: Callable[[str], list[dict[str, Any]] | None],
    apply_tool_schema_metadata: Callable[..., Any],
) -> Any:
    fn = entry.callable
    if fn is None:
        raise ValueError(f"tool entry missing callable: {entry.tool_id}")
    sig = inspect.signature(fn)
    name = str(entry.name or entry.tool_id or getattr(fn, "__name__", "tool")).strip() or "tool"

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        t0_epoch = time.time()
        err = None
        try:
            res = fn(*args, **kwargs)
        except Exception as exc:
            err = str(exc)
            res = {"error": err}
        dt_ms = int((time.perf_counter() - t0) * 1000)

        try:
            args_preview = _safe_json(kwargs)
        except Exception:
            args_preview = {}

        append_builtin_tool_event(
            tool_events,
            tool_name=name,
            args_preview=args_preview,
            started_at_ms=int(t0_epoch * 1000),
            elapsed_ms=dt_ms,
            result=res,
            error=err,
        )
        if err is None:
            try:
                update_candidate_ledger(name, args_preview, res)
            except Exception:
                pass
        if err:
            logger.warning(
                "[yellow]TOOL_ERR[/yellow] agent=%s tool=%s err=%s",
                agent_id,
                name,
                err[:200],
            )

        if err is None and memory_react_injection_enabled(settings.memory_policy, name):
            try:
                mem_query = build_memory_query(name, kwargs, res)
                if mem_query:
                    memories = search_tool_memories(mem_query)
                    if memories:
                        if isinstance(res, dict):
                            res["_memory_context"] = memories
                        elif isinstance(res, list):
                            res = {"data": res, "_memory_context": memories}
                        logger.info(
                            "[cyan]REACT_MEM[/cyan] agent=%s tool=%s query=%s hits=%d",
                            agent_id,
                            name,
                            mem_query[:80],
                            len(memories),
                        )
            except Exception:
                pass

        compact_res = _compact_tool_result_for_prompt(name, res, args=args_preview)

        if not err:
            logger.info(
                "[cyan]TOOL[/cyan] agent=%s tool=%s args=%s elapsed=%dms",
                agent_id,
                name,
                args_preview,
                dt_ms,
            )

        replace_last_tool_event_result(tool_events, compact_res)
        return compact_res

    return apply_tool_schema_metadata(wrapper, entry=entry, sig=sig)


def resolve_adk_tools(
    *,
    repo: BigQueryRepository,
    tenant_id: str,
    agent_config: AgentConfig | None,
    registry: ToolRegistry,
    settings: Settings,
    agent_id: str,
    tool_events: list[dict[str, Any]],
    update_candidate_ledger: Callable[[str, dict[str, Any], Any], None],
    search_tool_memories: Callable[[str], list[dict[str, Any]] | None],
    apply_tool_schema_metadata: Callable[..., Any],
) -> ToolResolution:
    disabled_tool_ids = _resolve_disabled_tool_ids(repo, tenant_id, agent_config)
    active_entries = [
        entry
        for entry in registry.list_entries(require_callable=True)
        if entry.tool_id not in disabled_tool_ids
    ]
    all_tools = [
        build_tool_wrapper(
            entry,
            settings=settings,
            agent_id=agent_id,
            tool_events=tool_events,
            update_candidate_ledger=update_candidate_ledger,
            search_tool_memories=search_tool_memories,
            apply_tool_schema_metadata=apply_tool_schema_metadata,
        )
        for entry in active_entries
    ]
    wrapped_tool_names = {
        str(getattr(tool, "__name__", "")).strip()
        for tool in all_tools
        if str(getattr(tool, "__name__", "")).strip()
    }
    mcp_toolsets = _load_mcp_toolsets(repo, tenant_id)
    logger.info(
        "[cyan]ADK tools resolved[/cyan] agent=%s tenant=%s builtin=%d disabled_total=%d mcp_toolsets=%d",
        agent_id,
        tenant_id,
        len(all_tools),
        len(disabled_tool_ids),
        len(mcp_toolsets),
    )
    return ToolResolution(
        adk_tools=[*all_tools, *mcp_toolsets],
        wrapped_tool_names=wrapped_tool_names,
        disabled_tool_ids=disabled_tool_ids,
        builtin_count=len(all_tools),
        mcp_toolset_count=len(mcp_toolsets),
    )


def build_agent(
    *,
    agent_id: str,
    provider: str,
    settings: Settings,
    repo: BigQueryRepository,
    tenant_id: str,
    agent_config: AgentConfig | None,
    max_tool_events: int,
    adk_tools: list[Any],
) -> Agent:
    model_override = (agent_config.model if agent_config else "") or ""
    model = _resolve_model(provider, settings, model_override=model_override)
    generate_cfg = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=False,
            maximum_remote_calls=max_tool_events,
        )
    )
    return Agent(
        name=f"{agent_id}_react",
        model=model,
        instruction=_system_prompt(
            agent_id,
            repo=repo,
            tenant_id=tenant_id,
            agent_config=agent_config,
            target_market=settings.kis_target_market,
        ),
        tools=adk_tools,
        generate_content_config=generate_cfg,
    )


def build_runner_runtime(
    *,
    provider: str,
    agent: Agent,
    identity: RunnerIdentity,
) -> RunnerBootstrap:
    session_service = InMemorySessionService()
    provider_key = provider.strip().lower()
    if provider_key in {"gemini", "google"}:
        app = App(
            name=identity.app_name,
            root_agent=agent,
            context_cache_config=ContextCacheConfig(
                cache_intervals=10,
                ttl_seconds=1800,
                min_tokens=2048,
            ),
        )
        runner = Runner(
            app=app,
            session_service=session_service,
        )
    else:
        runner = Runner(
            app_name=identity.app_name,
            agent=agent,
            session_service=session_service,
        )
    return RunnerBootstrap(
        agent=agent,
        runner=runner,
        session_service=session_service,
    )


def initialize_base_session(
    *,
    session_service: InMemorySessionService,
    identity: RunnerIdentity,
    run_on_loop: Callable[[Any], Any],
) -> None:
    run_on_loop(
        session_service.create_session(
            app_name=identity.app_name,
            user_id=identity.user_id,
            session_id=identity.session_id,
        )
    )
