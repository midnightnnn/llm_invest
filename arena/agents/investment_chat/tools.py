from __future__ import annotations

from typing import Any

from arena.agents.investment_chat.account_tools import build_account_tool_entries
from arena.agents.investment_chat.history_tools import build_history_tool_entries
from arena.agents.investment_chat.order_tools import build_order_tool_entries
from arena.config import Settings
from arena.tools.registry import ToolEntry


def build_chat_tool_entries(*, repo: Any, settings: Settings, tenant_id: str) -> list[ToolEntry]:
    return [
        *build_account_tool_entries(repo=repo, settings=settings, tenant_id=tenant_id),
        *build_history_tool_entries(repo=repo, tenant_id=tenant_id),
        *build_order_tool_entries(repo=repo, settings=settings, tenant_id=tenant_id),
    ]
