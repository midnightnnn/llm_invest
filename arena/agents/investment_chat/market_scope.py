from __future__ import annotations

from copy import deepcopy
from typing import Any

from arena.config import Settings


DEFAULT_ACCOUNT_MARKETS = "us,kospi,kosdaq"


def account_market_override(repo: Any, *, tenant_id: str) -> str:
    """Returns the whole-account market scope for investment chat.

    ``investment_chat_account_markets`` is a legacy key. Chat account context is
    intentionally independent from batch agent target markets, so stale rows or
    env overrides must not narrow the advisor to a single market.
    """
    _ = repo, tenant_id
    return DEFAULT_ACCOUNT_MARKETS


def account_scope_settings(repo: Any, *, tenant_id: str, settings: Settings) -> Settings:
    markets = account_market_override(repo, tenant_id=tenant_id)
    if not markets:
        return settings
    scoped = deepcopy(settings)
    scoped.kis_target_market = markets
    return scoped
