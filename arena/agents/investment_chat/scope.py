from __future__ import annotations

from typing import Any

from arena.agents.investment_chat.constants import AGENT_ID
from arena.agents.investment_chat.context import REQUEST_USER_EMAIL
from arena.agents.investment_chat.market_scope import account_snapshot_market_scope
from arena.agents.investment_chat.utils import latest_account_snapshot, safe_float, sources_for_settings
from arena.config import Settings
from arena.models import AccountSnapshot


def chat_actor_email() -> str:
    return str(REQUEST_USER_EMAIL.get() or "").strip().lower()


def normalize_order_scope(scope: str | None) -> str:
    token = str(scope or "account").strip().lower()
    if token in {"agent", "sleeve", "agent_sleeve", "agent-sleeve"}:
        return "agent_sleeve"
    if token in {"account", "total_account", "total-account", "portfolio"}:
        return "account"
    return ""


def chat_strategy_refs(*, scope: str, agent_id: str, user_email: str) -> list[str]:
    refs = [
        "source:investment_chat",
        "judgment:user+investment_chat",
        f"scope:{scope}",
        f"target_agent:{str(agent_id or '').strip().lower() or AGENT_ID}",
    ]
    if user_email:
        refs.append(f"approved_by:{user_email}")
    return refs


def snapshot_for_order_scope(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    scope: str,
    agent_id: str,
) -> tuple[AccountSnapshot | None, dict[str, Any]]:
    if scope == "account":
        market_scope = account_snapshot_market_scope()
        return latest_account_snapshot(repo, tenant_id=tenant_id, market_scope=market_scope), {
            "scope": "account",
            "target_agent_id": AGENT_ID,
            "market_scope": market_scope,
        }

    builder = getattr(repo, "build_agent_sleeve_snapshot", None)
    if not callable(builder):
        return None, {"scope": "agent_sleeve", "target_agent_id": agent_id, "error": "sleeve snapshot reader is unavailable"}
    try:
        snapshot, baseline_equity_krw, meta = builder(
            agent_id=agent_id,
            sources=sources_for_settings(settings),
            include_simulated=True,
            tenant_id=tenant_id,
        )
    except TypeError:
        snapshot, baseline_equity_krw, meta = builder(
            agent_id=agent_id,
            sources=sources_for_settings(settings),
            include_simulated=True,
        )
    return snapshot, {
        "scope": "agent_sleeve",
        "target_agent_id": agent_id,
        "baseline_equity_krw": safe_float(baseline_equity_krw),
        "metadata": meta if isinstance(meta, dict) else {},
    }
