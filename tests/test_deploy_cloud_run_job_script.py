from __future__ import annotations

from pathlib import Path


def test_kospi_scheduler_defaults_align_with_runtime_schedule_guard() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "deploy_cloud_run_job.sh"
    ).read_text(encoding="utf-8")

    assert 'SCHEDULER_KR_CRON="${SCHEDULER_KR_CRON:-30 14 * * 1-5}"' in script
    assert "ARENA_KOSPI_CYCLE_TIMES_KST=${ARENA_KOSPI_CYCLE_TIMES_KST:-14:30}" in script
    assert "ARENA_KOSPI_CYCLE_TOLERANCE_MINUTES=${ARENA_KOSPI_CYCLE_TOLERANCE_MINUTES:-20}" in script
    assert "ARENA_KOSPI_DISABLE_SCHEDULE_GUARD=false" in script
    assert 'SCHEDULER_RUN_BODY=' in script
    assert '--message-body "${body}"' in script
    assert '--role "roles/run.jobsExecutorWithOverrides"' in script
    assert "ARENA_LLM_TIMEOUT_SECONDS=1500" in script
    assert "ARENA_LLM_TIMEOUT_TRADING_SECONDS=3000" in script
    assert 'AGENT_TASK_TIMEOUT="${AGENT_TASK_TIMEOUT:-7200s}"' in script
