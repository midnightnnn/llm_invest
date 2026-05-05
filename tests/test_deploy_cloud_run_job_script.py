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


def test_split_deploy_manages_slow_and_fast_prep_jobs() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "deploy_cloud_run_job.sh"
    ).read_text(encoding="utf-8")

    assert 'SCHEDULER_US_SLOW_CRON="${SCHEDULER_US_SLOW_CRON:-0 14 * * 1-5}"' in script
    assert 'SCHEDULER_KR_SLOW_CRON="${SCHEDULER_KR_SLOW_CRON:-30 13 * * 1-5}"' in script
    assert 'SLOW_PREP_US_JOB="${JOB_NAME}-prep-slow-us"' in script
    assert 'SLOW_PREP_KR_JOB="${JOB_NAME}-prep-slow-kospi"' in script
    assert '"${PREP_RUN_ARGS},--market,us,--stage,slow"' in script
    assert '"${PREP_RUN_ARGS},--market,kospi,--stage,slow"' in script
    assert '"${PREP_RUN_ARGS},--market,us,--stage,fast,--dispatch-job,${AGENT_US_JOB}"' in script
    assert '"${PREP_RUN_ARGS},--market,kospi,--stage,fast,--dispatch-job,${AGENT_KR_JOB}"' in script
    assert '"${SCHEDULER_JOB_NAME}-slow-us"' in script
    assert '"${SCHEDULER_JOB_NAME}-slow-kospi"' in script
