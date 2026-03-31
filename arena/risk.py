from __future__ import annotations

import re
from datetime import datetime

from arena.config import Settings
from arena.models import AccountSnapshot, OrderIntent, RiskDecision, Side

_US_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def _ticker_matches_market(ticker: str, market: str) -> bool:
    """Checks whether a ticker format matches the configured target market."""
    token = ticker.strip().upper()
    if not token:
        return False

    if market in {"nasdaq", "nyse", "amex", "us"}:
        return bool(_US_TICKER_RE.fullmatch(token))
    if market == "kospi":
        return token.isdigit() and len(token) == 6

    return True


class RiskEngine:
    """Runs lightweight but critical checks before an order can be executed."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def evaluate(
        self,
        intent: OrderIntent,
        snapshot: AccountSnapshot,
        daily_turnover_krw: float,
        daily_order_count: int,
        last_trade_at: datetime | None,
        now: datetime,
    ) -> RiskDecision:
        """Evaluates one order intent against position, turnover, and cash rules."""
        hits: list[str] = []

        if snapshot.total_equity_krw <= 0:
            hits.append("equity_non_positive")
            return RiskDecision(allowed=False, reason="total equity is not positive", policy_hits=hits)

        if not _ticker_matches_market(intent.ticker, self.settings.kis_target_market):
            hits.append("ticker_market_mismatch")
            return RiskDecision(allowed=False, reason="ticker does not match target market", policy_hits=hits)

        notional = intent.notional_krw
        # Allow one-share buys above max_order_krw when still inside cash/position constraints.
        single_share_buy_exception = (
            intent.side == Side.BUY
            and 0.999 <= float(intent.quantity) <= 1.001
            and snapshot.cash_krw >= notional
            and (notional / snapshot.total_equity_krw) <= self.settings.max_position_ratio
        )
        if notional > self.settings.max_order_krw and not single_share_buy_exception:
            hits.append("max_order_krw")

        turnover_limit = snapshot.total_equity_krw * self.settings.max_daily_turnover_ratio
        if daily_turnover_krw + notional > turnover_limit:
            hits.append("max_daily_turnover_ratio")

        if self.settings.max_daily_orders > 0 and int(daily_order_count) >= int(self.settings.max_daily_orders):
            hits.append("max_daily_orders")

        if last_trade_at is not None:
            delta_sec = (now - last_trade_at).total_seconds()
            if delta_sec < self.settings.ticker_cooldown_seconds:
                hits.append("ticker_cooldown_seconds")

        current = snapshot.positions.get(intent.ticker)
        if intent.side == Side.SELL:
            if current is None or current.quantity <= 0:
                hits.append("no_position")
            elif float(intent.quantity) > float(current.quantity) + 1e-9:
                hits.append("insufficient_position_qty")
        current_value = current.market_value_krw() if current else 0.0
        next_value = current_value + notional if intent.side == Side.BUY else max(current_value - notional, 0.0)
        if next_value / snapshot.total_equity_krw > self.settings.max_position_ratio:
            hits.append("max_position_ratio")

        if intent.side == Side.BUY:
            min_cash = snapshot.total_equity_krw * self.settings.min_cash_buffer_ratio
            if snapshot.cash_krw - notional < min_cash:
                hits.append("min_cash_buffer_ratio")

        if hits:
            return RiskDecision(allowed=False, reason="policy check failed", policy_hits=hits)

        return RiskDecision(allowed=True, reason="approved", policy_hits=[])
