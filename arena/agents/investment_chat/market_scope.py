from __future__ import annotations

from copy import deepcopy
import os
from typing import Any

from arena.config import Settings


ACCOUNT_MARKETS_CONFIG_KEY = "investment_chat_account_markets"
DEFAULT_ACCOUNT_MARKETS = "us,kospi"
_ACCOUNT_MARKET_ALIASES = {
    "kr": "kospi",
    "korea": "kospi",
    "domestic": "kospi",
    "usa": "us",
}
_ALLOWED_ACCOUNT_MARKETS = {"us", "nasdaq", "nyse", "amex", "kospi"}


def account_market_override(repo: Any, *, tenant_id: str) -> str:
    raw = ""
    getter = getattr(repo, "get_config", None)
    if callable(getter):
        try:
            raw = str(getter(tenant_id, ACCOUNT_MARKETS_CONFIG_KEY) or "").strip()
        except Exception:
            raw = ""
    if not raw:
        raw = str(os.getenv("ARENA_INVESTMENT_CHAT_ACCOUNT_MARKETS") or "").strip()
    if not raw:
        raw = DEFAULT_ACCOUNT_MARKETS

    markets: list[str] = []
    for token in raw.replace("|", ",").replace(";", ",").split(","):
        clean = token.strip().lower()
        if clean in {"all", "total", "total_account", "account", "전체"}:
            clean = DEFAULT_ACCOUNT_MARKETS
        for part in clean.split(","):
            market = _ACCOUNT_MARKET_ALIASES.get(part.strip().lower(), part.strip().lower())
            if market in _ALLOWED_ACCOUNT_MARKETS and market not in markets:
                markets.append(market)
    return ",".join(markets)


def account_scope_settings(repo: Any, *, tenant_id: str, settings: Settings) -> Settings:
    markets = account_market_override(repo, tenant_id=tenant_id)
    if not markets:
        return settings
    scoped = deepcopy(settings)
    scoped.kis_target_market = markets
    return scoped
