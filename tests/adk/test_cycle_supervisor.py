from __future__ import annotations

from arena.agents.cycle_supervisor import AgentCycleSupervisor


def test_supervisor_records_operations_without_prompt_text() -> None:
    supervisor = AgentCycleSupervisor(cycle_id="cycle_1")

    op_id = supervisor.start_operation(kind="model_call", phase="explore", agent_id="claude")
    supervisor.finish_operation(op_id)
    summary = supervisor.summary()

    assert summary["cycle_id"] == "cycle_1"
    assert summary["operation_count"] == 1
    assert "prompt" not in str(summary).lower()
    assert "supervisor" not in summary.get("llm_visible_text", "")


def test_supervisor_model_call_timeout_policy_defaults_to_300_seconds() -> None:
    supervisor = AgentCycleSupervisor(cycle_id="cycle_1")

    assert supervisor.model_call_timeout_seconds(provider="claude", model="claude-opus-4-7") == 300
