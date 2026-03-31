from __future__ import annotations

from typing import Protocol

from arena.models import ExecutionReport, OrderIntent


class BrokerClient(Protocol):
    """Defines the minimal broker interface required by the gateway."""

    def place_order(self, intent: OrderIntent, *, fx_rate: float | None = None) -> ExecutionReport:
        """Submits one order intent and returns an execution report."""
        ...
