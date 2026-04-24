from __future__ import annotations

import inspect
import logging
from typing import Any

from arena.config import Settings
from arena.market_sources import KOSPI_MARKETS, US_MARKETS, parse_markets

logger = logging.getLogger(__name__)


def _normalize_tickers(tickers: list[str], *, markets: list[str]) -> list[str]:
    raw = [str(t).strip().upper() for t in tickers if str(t).strip()]
    has_us = bool(set(markets) & US_MARKETS)
    has_kospi = bool(set(markets) & KOSPI_MARKETS)
    if has_us and has_kospi:
        pass
    elif has_us:
        raw = [t for t in raw if not t[:1].isdigit()]
    elif has_kospi:
        raw = [t for t in raw if t.isdigit() and len(t) == 6]
    return list(dict.fromkeys(raw))


def resolve_runtime_universe(
    settings: Settings,
    repo: Any | None = None,
    *,
    markets: str | list[str] | set[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Returns the effective runtime universe for the requested markets.

    Explicit `settings.default_universe` is still honored as an override for tests
    and manual runs. Otherwise, the latest `universe_candidates` rows are loaded
    directly from the repository at read time.
    """
    market_tokens = parse_markets(markets if markets is not None else settings.kis_target_market)

    explicit = _normalize_tickers(list(settings.default_universe), markets=market_tokens)
    if explicit:
        return explicit

    limit = max(0, int(getattr(settings, "universe_run_top_n", 0) or 0))
    if limit <= 0 or repo is None:
        return []

    loader = getattr(repo, "latest_universe_candidate_tickers", None)
    if not callable(loader):
        return []

    supports_markets = False
    try:
        signature = inspect.signature(loader)
        supports_markets = "markets" in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )
    except (TypeError, ValueError):
        supports_markets = False

    try:
        if supports_markets:
            discovered = loader(limit=limit, markets=market_tokens)
        else:
            discovered = loader(limit=limit)
    except Exception as exc:
        logger.warning(
            "[yellow]runtime_universe load failed[/yellow] markets=%s err=%s",
            ",".join(market_tokens) or "-",
            str(exc),
        )
        return []

    return _normalize_tickers(list(discovered or []), markets=market_tokens)
