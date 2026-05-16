from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any

from arena.logging_utils import event_extra


logger = logging.getLogger(__name__)


@dataclass
class OperationRecord:
    operation_id: str
    kind: str
    phase: str
    agent_id: str
    started_monotonic: float
    metadata: dict[str, Any] = field(default_factory=dict)
    finished_monotonic: float | None = None


@dataclass
class AgentCycleSupervisor:
    cycle_id: str
    default_model_call_timeout_seconds: int = 300
    _operations: dict[str, OperationRecord] = field(default_factory=dict)

    def start_operation(
        self,
        *,
        kind: str,
        phase: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        operation_id = f"{kind}_{phase}_{agent_id}_{len(self._operations) + 1}"
        operation_metadata = dict(metadata or {})
        self._operations[operation_id] = OperationRecord(
            operation_id=operation_id,
            kind=kind,
            phase=phase,
            agent_id=agent_id,
            started_monotonic=time.monotonic(),
            metadata=operation_metadata,
        )
        fields = {
            **operation_metadata,
            "cycle_id": self.cycle_id,
            "operation_id": operation_id,
            "operation_kind": kind,
            "phase": phase,
            "agent_id": agent_id,
        }
        logger.info(
            "[blue]Agent cycle supervisor operation start[/blue] cycle_id=%s op=%s kind=%s phase=%s agent=%s",
            self.cycle_id or "-",
            operation_id,
            kind,
            phase,
            agent_id,
            extra=event_extra("agent_cycle_supervisor_operation_start", **fields),
        )
        return operation_id

    def finish_operation(
        self,
        operation_id: str,
        *,
        status: str = "finished",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = self._operations[operation_id]
        record.finished_monotonic = time.monotonic()
        elapsed_ms = int((record.finished_monotonic - record.started_monotonic) * 1000)
        fields = {
            **record.metadata,
            **dict(metadata or {}),
            "cycle_id": self.cycle_id,
            "operation_id": operation_id,
            "operation_kind": record.kind,
            "phase": record.phase,
            "agent_id": record.agent_id,
            "status": status,
            "elapsed_ms": elapsed_ms,
        }
        logger.info(
            "[blue]Agent cycle supervisor operation finish[/blue] cycle_id=%s op=%s kind=%s phase=%s elapsed=%dms",
            self.cycle_id or "-",
            operation_id,
            record.kind,
            record.phase,
            elapsed_ms,
            extra=event_extra("agent_cycle_supervisor_operation_finish", **fields),
        )

    def model_call_timeout_seconds(self, *, provider: str, model: str) -> int:
        _ = provider, model
        return self.default_model_call_timeout_seconds

    def summary(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "operation_count": len(self._operations),
        }
