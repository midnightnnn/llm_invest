from __future__ import annotations

import logging
import os
from typing import Any, Callable

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
from arena.agents.investment_chat.selection import (
    chat_model_routing_config,
    cheap_chat_model_for_provider,
    normalize_chat_model_selection,
    tenant_default_chat_selection,
)
from arena.agents.investment_chat.utils import repo_tenant_scope
from arena.config import Settings
from arena.memory.store import MemoryStore
from arena.prompts.prompt_pack import PromptPack
from arena.tools.registry import ToolEntry, ToolRegistry

logger = logging.getLogger(__name__)

ADVISOR_AGENT_NAME = "investment_chat_advisor"
UTILITY_AGENT_NAME = "investment_chat_utility"

UTILITY_TOOL_IDS = frozenset(
    {
        "get_account_snapshot",
        "refresh_account_snapshot",
        "get_agent_sleeve_snapshot",
        "get_trade_history",
        "get_order_approval_status",
        "propose_agent_config_change",
        "propose_chat_agent_config_change",
        "propose_tenant_config_change",
        "get_config_change_status",
    }
)


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


def _registry_for_tool_ids(registry: ToolRegistry, tool_ids: frozenset[str]) -> ToolRegistry:
    allowed = {str(tool_id or "").strip().lower() for tool_id in tool_ids if str(tool_id or "").strip()}
    entries: list[ToolEntry] = []
    for entry in registry.list_entries(require_callable=True):
        token = str(entry.tool_id or entry.name or "").strip().lower()
        if token in allowed:
            entries.append(entry)
    return ToolRegistry(entries)


def _mapping_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _advisor_instruction(*, tenant: str, provider: str, model_id: str) -> str:
    return PromptPack.render_investment_chat_instruction(
        tenant_id=tenant,
        provider=provider,
        model_id=model_id,
        utility_agent_name=UTILITY_AGENT_NAME,
    )


def _utility_instruction(*, tenant: str, provider: str, model_id: str) -> str:
    return PromptPack.render_investment_chat_utility_instruction(
        tenant_id=tenant,
        provider=provider,
        model_id=model_id,
        advisor_agent_name=ADVISOR_AGENT_NAME,
    )


def build_investment_chat_agent(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    registry: ToolRegistry | None = None,
    provider: str | None = None,
    model_override: str | None = None,
    invalidate_tenant_cache: Callable[..., Any] | None = None,
) -> Agent:
    tenant = normalize_tenant(tenant_id)
    chat_config = load_chat_agent_config(repo, tenant_id=tenant)
    tenant_provider, tenant_model = tenant_default_chat_selection(settings)
    provider_token = str(
        provider
        or chat_config.get("provider")
        or os.getenv("ARENA_CHAT_PROVIDER")
        or tenant_provider
        or "gemini"
    ).strip().lower() or "gemini"
    model_id = str(model_override or chat_config.get("model") or os.getenv("ARENA_CHAT_MODEL") or "").strip()
    if not model_id and provider_token == tenant_provider:
        model_id = tenant_model
    model_id = normalize_chat_model_selection(provider_token, model_id)
    llm_params = chat_config.get("llm_params") if isinstance(chat_config.get("llm_params"), dict) else {}
    routing_config = chat_model_routing_config(chat_config)
    cheap_model_id = cheap_chat_model_for_provider(provider_token, chat_config=chat_config)
    if not cheap_model_id:
        cheap_model_id = model_id
    router_llm_params = _mapping_value(routing_config.get("router_llm_params"))
    utility_llm_params = _mapping_value(routing_config.get("utility_llm_params")) or router_llm_params
    max_tool_events = resolve_max_tool_events(settings)
    chat_registry = build_chat_registry(
        repo=repo,
        settings=settings,
        tenant_id=tenant,
        registry=registry,
        invalidate_tenant_cache=invalidate_tenant_cache,
    )
    advisor_tools = _wrapped_tools(chat_registry, repo=repo, settings=settings, tenant_id=tenant)
    utility_tools = _wrapped_tools(
        _registry_for_tool_ids(chat_registry, UTILITY_TOOL_IDS),
        repo=repo,
        settings=settings,
        tenant_id=tenant,
    )
    router_model = _resolve_model(
        provider_token,
        settings,
        model_override=cheap_model_id,
        llm_params=router_llm_params,
    )
    advisor_model = _resolve_model(provider_token, settings, model_override=model_id, llm_params=llm_params)
    utility_model = _resolve_model(
        provider_token,
        settings,
        model_override=cheap_model_id,
        llm_params=utility_llm_params,
    )
    advisor_agent = Agent(
        name=ADVISOR_AGENT_NAME,
        description="Arena 투자챗봇 투자상담 에이전트",
        model=advisor_model,
        instruction=_advisor_instruction(tenant=tenant, provider=provider_token, model_id=model_id),
        tools=advisor_tools,
        generate_content_config=_build_generate_content_config(
            provider=provider_token,
            llm_params=llm_params,
            max_tool_events=max_tool_events,
        ),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=False,
    )
    utility_agent = Agent(
        name=UTILITY_AGENT_NAME,
        description="Arena 투자챗봇 조회/설정 유틸리티 에이전트",
        model=utility_model,
        instruction=_utility_instruction(tenant=tenant, provider=provider_token, model_id=cheap_model_id),
        tools=utility_tools,
        generate_content_config=_build_generate_content_config(
            provider=provider_token,
            llm_params=utility_llm_params,
            max_tool_events=max_tool_events,
        ),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=False,
    )
    return Agent(
        name=APP_NAME,
        description="Arena 투자챗봇 라우터",
        model=router_model,
        instruction=PromptPack.render_investment_chat_router_instruction(
            tenant_id=tenant,
            provider=provider_token,
            advisor_model_id=model_id,
            cheap_model_id=cheap_model_id,
            advisor_agent_name=ADVISOR_AGENT_NAME,
            utility_agent_name=UTILITY_AGENT_NAME,
        ),
        tools=[],
        sub_agents=[advisor_agent, utility_agent],
        generate_content_config=_build_generate_content_config(
            provider=provider_token,
            llm_params=router_llm_params,
            max_tool_events=max_tool_events,
        ),
    )
