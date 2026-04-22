from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from arena.market_sources import KOSPI_MARKETS, US_MARKETS, parse_markets


_KR_TICKER_RE = re.compile(r"^\d{6}$")
_KR_MARKET_ALIASES: frozenset[str] = frozenset(KOSPI_MARKETS | {"kosdaq"})


class MarketScopeError(ValueError):
    """Raised when a tool is asked to operate outside its configured market scope."""


@dataclass(frozen=True, slots=True)
class MarketScope:
    """Authoritative per-cycle view of the agent's target markets.

    Built once at tool ``set_context`` time from the agent context (with a
    settings-level fallback), so every tool in the cycle shares an identical,
    normalized view of which markets are in scope.

    The ``markets`` set holds the raw tokens the agent was configured with
    (``us``, ``nasdaq``, ``kospi``, ...). Methods translate between this
    configured view and the row-market tokens stored in BigQuery tables
    (``us``, ``kospi``).
    """

    markets: frozenset[str]

    @classmethod
    def from_context(
        cls,
        context: dict[str, Any] | None,
        *,
        fallback: str | None = None,
    ) -> "MarketScope":
        raw = ""
        if context:
            raw = str(context.get("target_market") or "").strip().lower()
        if not raw:
            raw = str(fallback or "").strip().lower()
        tokens = parse_markets(raw)
        if not tokens:
            raise MarketScopeError(
                "target_market is not configured for this agent. "
                "Set target_market in agent config."
            )
        return cls(markets=frozenset(tokens))

    @property
    def has_us(self) -> bool:
        return bool(self.markets & US_MARKETS)

    @property
    def has_kospi(self) -> bool:
        return bool(self.markets & _KR_MARKET_ALIASES)

    def primary(self) -> str:
        """Returns a single representative market token (first in sorted order)."""
        if not self.markets:
            raise MarketScopeError("MarketScope is empty")
        return sorted(self.markets)[0]

    def as_set(self) -> set[str]:
        """Returns a mutable copy of the raw configured tokens."""
        return set(self.markets)

    def row_market_filter(self) -> list[str]:
        """Row-level ``market`` column values this scope accepts.

        BQ ``opportunity_ranker_scores_latest.market`` holds ``us`` / ``kospi``
        only. An agent configured with ``nasdaq`` still wants ``us`` rows.
        """
        tokens: list[str] = []
        if self.has_us:
            tokens.append("us")
        if self.has_kospi:
            tokens.append("kospi")
        for token in sorted(self.markets):
            if token in US_MARKETS or token in KOSPI_MARKETS:
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens

    def ticker_market(self, ticker: str) -> str | None:
        """Classifies a ticker by shape: ``kospi`` / ``us`` / None."""
        token = str(ticker or "").strip().upper()
        if not token:
            return None
        if _KR_TICKER_RE.match(token):
            return "kospi"
        alpha_core = re.sub(r"[.\-/]", "", token)
        if alpha_core.isalpha():
            return "us"
        return None

    def ticker_in_scope(self, ticker: str) -> bool:
        market = self.ticker_market(ticker)
        if market is None:
            return False
        if market == "us":
            return self.has_us
        if market == "kospi":
            return self.has_kospi
        return market in self.markets

    def row_in_scope(self, row_market: str | None) -> bool:
        token = str(row_market or "").strip().lower()
        if not token:
            return False
        if token in self.markets:
            return True
        if token in US_MARKETS and self.has_us:
            return True
        if token in _KR_MARKET_ALIASES and self.has_kospi:
            return True
        return False

    def filter_tickers(
        self,
        tickers: Iterable[str],
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Splits tickers into (in_scope, excluded_with_reasons).

        Order is preserved for the in-scope list; duplicates are dropped.
        """
        kept: list[str] = []
        excluded: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw in tickers or []:
            token = str(raw or "").strip().upper()
            if not token or token in seen:
                continue
            seen.add(token)
            if self.ticker_in_scope(token):
                kept.append(token)
                continue
            excluded.append(
                {
                    "ticker": token,
                    "reason": "out_of_market_scope",
                    "ticker_market": str(self.ticker_market(token) or "unknown"),
                    "agent_markets": ",".join(sorted(self.markets)),
                }
            )
        return kept, excluded
