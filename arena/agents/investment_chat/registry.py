from __future__ import annotations

from typing import Any

from arena.agents.adk_context_tools import _ContextTools
from arena.agents.adk_tool_config import _load_disabled_tool_ids
from arena.agents.investment_chat.config_tools import load_chat_agent_config
from arena.agents.investment_chat.constants import AGENT_ID, CHAT_ANALYSIS_TOOL_IDS, WRITE_TOOL_MARKERS
from arena.agents.investment_chat.context import normalize_tenant
from arena.agents.investment_chat.market_scope import account_scope_settings
from arena.agents.investment_chat.tools import build_chat_tool_entries
from arena.config import Settings
from arena.tools.default_registry import build_default_registry
from arena.tools.registry import ToolEntry, ToolRegistry


def build_chat_registry(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    registry: ToolRegistry | None,
) -> ToolRegistry:
    tenant = normalize_tenant(tenant_id)
    tool_settings = account_scope_settings(repo, tenant_id=tenant, settings=settings)
    source = registry.clone() if registry is not None else build_default_registry(repo, tool_settings, tenant_id=tenant)
    disabled = _load_disabled_tool_ids(repo, tenant)
    chat_config = load_chat_agent_config(repo, tenant_id=tenant)
    if isinstance(chat_config.get("disabled_tools"), list):
        disabled.update(str(tool_id).strip() for tool_id in chat_config["disabled_tools"] if str(tool_id).strip())

    context_tools = _ContextTools(repo=repo, settings=tool_settings, agent_id=AGENT_ID, tenant_id=tenant)
    for tool_id, fn in {
        "search_past_experiences": context_tools.search_past_experiences,
        "search_peer_lessons": context_tools.search_peer_lessons,
        "get_research_briefing": context_tools.get_research_briefing,
        "portfolio_diagnosis": context_tools.portfolio_diagnosis,
        "trade_performance": context_tools.trade_performance,
    }.items():
        if source.get(tool_id) is not None:
            source.bind(tool_id, fn)

    entries: list[ToolEntry] = []
    for entry in source.list_entries(require_callable=True):
        token = str(entry.tool_id or entry.name or "").strip().lower()
        if token not in CHAT_ANALYSIS_TOOL_IDS:
            continue
        if any(marker in token for marker in WRITE_TOOL_MARKERS):
            continue
        entries.append(entry)
    entries.extend(build_chat_tool_entries(repo=repo, settings=tool_settings, tenant_id=tenant))
    return ToolRegistry(
        [entry for entry in entries if str(entry.tool_id or "").strip().lower() not in disabled]
    )
