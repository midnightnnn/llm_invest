from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OperationRecord:
    operation_id: str
    kind: str
    phase: str
    agent_id: str
    started_monotonic: float
    finished_monotonic: float | None = None


@dataclass
class AgentCycleSupervisor:
    cycle_id: str
    default_model_call_timeout_seconds: int = 300
    _operations: dict[str, OperationRecord] = field(default_factory=dict)

    def start_operation(self, *, kind: str, phase: str, agent_id: str) -> str:
        operation_id = f"{kind}_{phase}_{agent_id}_{len(self._operations) + 1}"
        self._operations[operation_id] = OperationRecord(
            operation_id=operation_id,
            kind=kind,
            phase=phase,
            agent_id=agent_id,
            started_monotonic=time.monotonic(),
        )
        return operation_id

    def finish_operation(self, operation_id: str) -> None:
        self._operations[operation_id].finished_monotonic = time.monotonic()

    def model_call_timeout_seconds(self, *, provider: str, model: str) -> int:
        _ = provider, model
        return self.default_model_call_timeout_seconds

    def summary(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "operation_count": len(self._operations),
        }
