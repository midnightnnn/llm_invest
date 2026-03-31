"""Narrow Protocol interfaces for the BigQuery data layer.

These protocols allow call-sites to declare the *minimum* set of repository
methods they actually need, rather than depending on the full
:class:`BigQueryRepository` facade.  This makes testing easier (smaller
stubs), improves discoverability for contributors, and opens the door to
alternative storage back-ends in the future.

Usage example::

    from arena.data.protocols import MarketReader

    class MyService:
        def __init__(self, market: MarketReader) -> None: ...
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------


@runtime_checkable
class MarketReader(Protocol):
    """Read-only access to market features and instrument data."""

    def latest_market_features(
        self, tickers: list[str], limit: int, sources: list[str] | None = None
    ) -> list[dict[str, Any]]: ...

    def latest_close_prices(self, *, tickers: list[str], sources: list[str] | None = None) -> dict[str, float]: ...

    def latest_close_prices_with_currency(
        self, *, tickers: list[str], sources: list[str] | None = None, as_of_date: date | None = None
    ) -> dict[str, dict[str, Any]]: ...

    def latest_instrument_map(self, tickers: list[str]) -> dict[str, dict[str, Any]]: ...

    def ticker_name_map(self, *, tickers: list[str] | None = None, limit: int = 500) -> dict[str, str]: ...


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryReader(Protocol):
    """Read-only access to agent memory events."""

    def recent_memory_events(
        self, agent_id: str, limit: int, trading_mode: str = "paper", *, tenant_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    def memory_events_by_ids(
        self, *, agent_id: str, event_ids: list[str], trading_mode: str = "paper", tenant_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    def memory_event_by_id(self, *, event_id: str, tenant_id: str | None = None) -> dict[str, Any] | None: ...

    def find_buy_memories_for_ticker(
        self, agent_id: str, ticker: str, limit: int = 5, trading_mode: str = "paper", *, tenant_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    def memory_graph_neighbors(
        self,
        *,
        seed_node_ids: list[str],
        trading_mode: str = "paper",
        min_confidence: float = 0.0,
        limit: int = 24,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class MemoryWriter(Protocol):
    """Write access to agent memory events and graph."""

    def write_memory_event(self, event: Any, *, tenant_id: str | None = None) -> None: ...

    def update_memory_event(self, *, event_id: str, summary: str, payload: dict[str, Any], score: float, **kwargs: Any) -> None: ...

    def update_memory_score(self, event_id: str, new_score: float, *, tenant_id: str | None = None) -> None: ...

    def append_memory_access_events(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> None: ...


# ---------------------------------------------------------------------------
# Config / Runtime
# ---------------------------------------------------------------------------


@runtime_checkable
class ConfigStore(Protocol):
    """Configuration key-value access."""

    def get_config(self, tenant_id: str, config_key: str) -> str | None: ...

    def set_config(self, tenant_id: str, config_key: str, value: str, updated_by: str | None = None, *, updated_at: datetime | None = None) -> None: ...

    def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]: ...


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@runtime_checkable
class ExecutionReader(Protocol):
    """Read-only access to order/execution data."""

    def recent_intent_count(
        self, day: date, *, agent_id: str | None = None, include_simulated: bool = True,
        include_submitted: bool = True, trading_mode: str | None = None, tenant_id: str | None = None
    ) -> int: ...

    def recent_turnover_krw(
        self, day: date, *, agent_id: str | None = None, include_simulated: bool = True,
        include_submitted: bool = True, trading_mode: str | None = None, tenant_id: str | None = None
    ) -> float: ...

    def last_trade_time(
        self, ticker: str, *, agent_id: str | None = None, exchange_code: str | None = None,
        instrument_id: str | None = None, include_simulated: bool = True, include_submitted: bool = True,
        trading_mode: str | None = None, tenant_id: str | None = None
    ) -> datetime | None: ...


# ---------------------------------------------------------------------------
# Tenant resolution (shared across all stores)
# ---------------------------------------------------------------------------


@runtime_checkable
class TenantResolver(Protocol):
    """Minimal tenant-id resolution."""

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str: ...
