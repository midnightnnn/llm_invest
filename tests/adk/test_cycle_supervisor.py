from __future__ import annotations

import logging

import pytest

from arena.agents.cycle_supervisor import AgentCycleSupervisor


def test_supervisor_records_operations_without_prompt_text(caplog: pytest.LogCaptureFixture) -> None:
    supervisor = AgentCycleSupervisor(cycle_id="cycle_1")

    with caplog.at_level(logging.INFO, logger="arena.agents.cycle_supervisor"):
        op_id = supervisor.start_operation(
            kind="model_call",
            phase="explore",
            agent_id="claude",
            metadata={"provider": "claude", "llm_call_id": "call_1"},
        )
        supervisor.finish_operation(op_id, status="success")
    summary = supervisor.summary()

    assert summary["cycle_id"] == "cycle_1"
    assert summary["operation_count"] == 1
    assert "prompt" not in str(summary).lower()
    assert "supervisor" not in summary.get("llm_visible_text", "")
    assert [record.event for record in caplog.records] == [
        "agent_cycle_supervisor_operation_start",
        "agent_cycle_supervisor_operation_finish",
    ]
    assert caplog.records[0].cycle_id == "cycle_1"
    assert caplog.records[0].operation_kind == "model_call"
    assert caplog.records[0].phase == "explore"
    assert caplog.records[0].provider == "claude"
    assert caplog.records[0].llm_call_id == "call_1"
    assert caplog.records[1].status == "success"
    assert caplog.records[1].elapsed_ms >= 0


def test_supervisor_model_call_timeout_policy_defaults_to_300_seconds() -> None:
    supervisor = AgentCycleSupervisor(cycle_id="cycle_1")

    assert supervisor.model_call_timeout_seconds(provider="claude", model="claude-opus-4-7") == 300
