from __future__ import annotations

import logging
import os
from typing import Any

from google.adk import Agent

from arena.agents.adk_models import _resolve_model
from arena.agents.adk_runner_bootstrap import (
    _build_generate_content_config,
    build_tool_wrapper,
    resolve_max_tool_events,
)
from arena.agents.adk_tool_helpers import (
    apply_tool_schema_metadata,
    noop_update_candidate_ledger,
)
from arena.agents.adk_runner_runtime import search_tool_memories
from arena.agents.investment_chat.config_tools import load_chat_agent_config
from arena.agents.investment_chat.constants import AGENT_ID, APP_NAME
from arena.agents.investment_chat.context import normalize_tenant
from arena.agents.investment_chat.registry import build_chat_registry
from arena.agents.investment_chat.utils import repo_tenant_scope
from arena.config import Settings
from arena.memory.store import MemoryStore
from arena.prompts.prompt_pack import PromptPack
from arena.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _tool_memory_searcher(*, repo: Any, settings: Settings, tenant_id: str):
    seen_memory_ids: set[str] = set()
    try:
        memory_store = MemoryStore(
            repo=repo,
            trading_mode=str(getattr(settings, "trading_mode", "") or "paper").strip().lower() or "paper",
            memory_policy=getattr(settings, "memory_policy", {}) or {},
        )
    except Exception as exc:
        logger.warning(
            "[yellow]investment chat memory search disabled[/yellow] tenant=%s err=%s",
            tenant_id,
            str(exc),
        )
        return lambda query: None

    def _search(query):
        try:
            with repo_tenant_scope(repo, tenant_id):
                return search_tool_memories(
                    memory_store=memory_store,
                    settings=settings,
                    agent_id=AGENT_ID,
                    seen_memory_ids=seen_memory_ids,
                    query=query,
                )
        except Exception as exc:
            logger.warning(
                "[yellow]investment chat tool memory search failed[/yellow] tenant=%s err=%s",
                tenant_id,
                str(exc),
            )
            return None

    return _search


def _wrapped_tools(registry: ToolRegistry, *, repo: Any, settings: Settings, tenant_id: str) -> list[Any]:
    tool_events: list[dict[str, Any]] = []
    search_memories = _tool_memory_searcher(repo=repo, settings=settings, tenant_id=tenant_id)
    return [
        build_tool_wrapper(
            entry,
            settings=settings,
            agent_id=AGENT_ID,
            tool_events=tool_events,
            update_candidate_ledger=noop_update_candidate_ledger,
            search_tool_memories=search_memories,
            apply_tool_schema_metadata=apply_tool_schema_metadata,
        )
        for entry in registry.list_entries(require_callable=True)
    ]


def build_investment_chat_agent(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    registry: ToolRegistry | None = None,
    provider: str | None = None,
    model_override: str | None = None,
) -> Agent:
    tenant = normalize_tenant(tenant_id)
    chat_config = load_chat_agent_config(repo, tenant_id=tenant)
    provider_token = str(
        provider
        or chat_config.get("provider")
        or os.getenv("ARENA_CHAT_PROVIDER")
        or "gemini"
    ).strip().lower() or "gemini"
    model_id = str(model_override or chat_config.get("model") or os.getenv("ARENA_CHAT_MODEL") or "").strip()
    llm_params = chat_config.get("llm_params") if isinstance(chat_config.get("llm_params"), dict) else {}
    max_tool_events = resolve_max_tool_events(settings)
    chat_registry = build_chat_registry(repo=repo, settings=settings, tenant_id=tenant, registry=registry)
    tools = _wrapped_tools(chat_registry, repo=repo, settings=settings, tenant_id=tenant)
    model = _resolve_model(provider_token, settings, model_override=model_id, llm_params=llm_params)
    return Agent(
        name=APP_NAME,
        description="Arena 투자챗봇",
        model=model,
        instruction=PromptPack.render_investment_chat_instruction(
            tenant_id=tenant,
            provider=provider_token,
            model_id=model_id,
        ),
        tools=tools,
        generate_content_config=_build_generate_content_config(
            provider=provider_token,
            llm_params=llm_params,
            max_tool_events=max_tool_events,
        ),
    )
