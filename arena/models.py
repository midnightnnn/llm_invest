from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Returns a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Side(str, Enum):
    """Enumerates supported trade sides."""

    BUY = "BUY"
    SELL = "SELL"


class ExecutionStatus(str, Enum):
    """Enumerates gateway execution outcomes."""

    REJECTED = "REJECTED"
    SIMULATED = "SIMULATED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    ERROR = "ERROR"


class OrderIntent(BaseModel):
    """Represents a single order proposal created by an agent."""

    agent_id: str
    ticker: str
    trading_mode: str = "paper"
    exchange_code: str = ""
    instrument_id: str = ""
    side: Side
    quantity: float = Field(gt=0)
    price_krw: float = Field(gt=0)
    price_native: float | None = None
    quote_currency: str = ""
    fx_rate: float = 0.0
    rationale: str
    strategy_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    intent_id: str = Field(default_factory=lambda: f"intent_{uuid4().hex[:12]}")
    cycle_id: str = ""

    @property
    def notional_krw(self) -> float:
        """Returns the order notional in KRW."""
        return self.quantity * self.price_krw


class RiskDecision(BaseModel):
    """Captures pass/fail output from the risk engine."""

    allowed: bool
    reason: str
    policy_hits: list[str] = Field(default_factory=list)


class ExecutionReport(BaseModel):
    """Represents the final status of an order after gateway processing."""

    status: ExecutionStatus
    order_id: str
    filled_qty: float
    avg_price_krw: float
    avg_price_native: float | None = None
    quote_currency: str = ""
    fx_rate: float = 0.0
    message: str
    created_at: datetime = Field(default_factory=utc_now)


class Position(BaseModel):
    """Represents a current position in the account."""

    ticker: str
    exchange_code: str = ""
    instrument_id: str = ""
    quantity: float
    avg_price_krw: float
    market_price_krw: float
    avg_price_native: float | None = None
    market_price_native: float | None = None
    quote_currency: str = ""
    fx_rate: float = 0.0

    def market_value_krw(self) -> float:
        """Returns mark-to-market value in KRW."""
        return self.quantity * self.market_price_krw


class AccountSnapshot(BaseModel):
    """Represents account equity, cash, and live positions."""

    cash_krw: float
    total_equity_krw: float
    positions: dict[str, Position] = Field(default_factory=dict)
    usd_krw_rate: float = 0.0
    cash_foreign: float = 0.0
    cash_foreign_currency: str = ""


class BoardPost(BaseModel):
    """Represents a board message for inter-agent information sharing."""

    agent_id: str
    title: str
    body: str
    draft_summary: str = ""
    trading_mode: str = "paper"
    tickers: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    post_id: str = Field(default_factory=lambda: f"post_{uuid4().hex[:12]}")
    cycle_id: str = ""


class MemoryEvent(BaseModel):
    """Represents a long-memory event for later retrieval."""

    agent_id: str
    event_type: str
    summary: str
    trading_mode: str = "paper"
    payload: dict[str, Any] = Field(default_factory=dict)
    importance_score: float | None = None
    outcome_score: float | None = None
    # Legacy compatibility field. New retrieval should prefer importance/outcome separation.
    score: float = 0.0
    memory_tier: str | None = None
    expires_at: datetime | None = None
    promoted_at: datetime | None = None
    semantic_key: str | None = None
    context_tags: dict[str, Any] = Field(default_factory=dict)
    primary_regime: str | None = None
    primary_strategy_tag: str | None = None
    primary_sector: str | None = None
    access_count: int | None = None
    last_accessed_at: datetime | None = None
    decay_score: float | None = None
    effective_score: float | None = None
    graph_node_id: str | None = None
    causal_chain_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    event_id: str = Field(default_factory=lambda: f"mem_{uuid4().hex[:12]}")
