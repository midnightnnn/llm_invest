from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace

import arena.cli as cli
import pytest
from arena.config import load_settings

from tests.cli.helpers import _FakeRepo, _stub_shared_prep_environment

def test_dispatch_agent_job_logs_response_without_error(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_location = "asia-northeast3"

    calls: list[dict[str, object]] = []

    def _fake_run_cloud_run_job(**kwargs):
        calls.append(dict(kwargs))
        return {"metadata": {"name": "exec_1"}}

    monkeypatch.setenv("ARENA_CLOUD_RUN_REGION", "asia-northeast3")
    monkeypatch.setattr(cli, "run_cloud_run_job", _fake_run_cloud_run_job)

    cli._dispatch_agent_job(settings, job_name="agent-us")

    assert calls == [
        {
            "project": "proj-x",
            "region": "asia-northeast3",
            "job_name": "agent-us",
            "body": {},
            "timeout_seconds": 30,
        }
    ]


def test_dispatch_agent_job_propagates_execution_source(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_location = "asia-northeast3"

    calls: list[dict[str, object]] = []

    def _fake_run_cloud_run_job(**kwargs):
        calls.append(dict(kwargs))
        return {"metadata": {"name": "exec_1"}}

    monkeypatch.setenv("ARENA_CLOUD_RUN_REGION", "asia-northeast3")
    monkeypatch.setenv("CLOUD_RUN_JOB", "llm-arena-batch-prep-us")
    monkeypatch.setenv("ARENA_EXECUTION_SOURCE", "scheduler")
    monkeypatch.setattr(cli, "run_cloud_run_job", _fake_run_cloud_run_job)

    cli._dispatch_agent_job(settings, job_name="agent-us")

    assert calls == [
        {
            "project": "proj-x",
            "region": "asia-northeast3",
            "job_name": "agent-us",
            "body": {
                "overrides": {
                    "containerOverrides": [
                        {
                            "env": [
                                {
                                    "name": "ARENA_EXECUTION_SOURCE",
                                    "value": "scheduler",
                                }
                            ]
                        }
                    ]
                }
            },
            "timeout_seconds": 30,
        }
    ]


def test_run_agent_cycle_guarded_skips_when_lease_exists(monkeypatch) -> None:
    settings = load_settings()
    settings.arena_mode = "gcp"
    settings.google_cloud_project = "proj-x"
    repo = _FakeRepo()

    class _LeaseStore:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def acquire(self, **kwargs):
            _ = kwargs
            return SimpleNamespace(acquired=False, reason="lease_held", lease_id="lease_1")

    monkeypatch.setattr(cli, "_tenant_lease_enabled", lambda: True)
    monkeypatch.setattr(cli, "FirestoreTenantLeaseStore", _LeaseStore)
    monkeypatch.setattr(cli, "_run_agent_cycle_once", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")))

    cli._run_agent_cycle_once_guarded(
        True,
        settings=settings,
        repo=repo,
        orchestrator=object(),
        tenant="tenant-a",
        run_id="run-1",
        market_override="us",
    )

    assert repo.run_status_rows
    assert repo.run_status_rows[-1]["status"] == "skipped"
    assert repo.run_status_rows[-1]["reason_code"] == "lease_held"


def test_run_agent_cycle_guarded_marks_lease_success(monkeypatch) -> None:
    settings = load_settings()
    settings.arena_mode = "gcp"
    settings.google_cloud_project = "proj-x"
    repo = _FakeRepo()
    completed: list[tuple[str, str]] = []

    class _LeaseStore:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def acquire(self, **kwargs):
            _ = kwargs
            return SimpleNamespace(acquired=True, reason="acquired", lease_id="lease_1")

        def complete(self, **kwargs):
            completed.append((kwargs["lease_id"], kwargs["status"]))

    calls: list[str] = []
    monkeypatch.setattr(cli, "_tenant_lease_enabled", lambda: True)
    monkeypatch.setattr(cli, "FirestoreTenantLeaseStore", _LeaseStore)
    monkeypatch.setattr(cli, "_run_agent_cycle_once", lambda *args, **kwargs: calls.append("run"))

    cli._run_agent_cycle_once_guarded(
        True,
        settings=settings,
        repo=repo,
        orchestrator=object(),
        tenant="tenant-a",
        run_id="run-1",
        market_override="us",
    )

    assert calls == ["run"]
    assert completed == [("lease_1", "success")]


def test_run_agent_cycle_guarded_passes_execution_source_to_lease(monkeypatch) -> None:
    settings = load_settings()
    settings.arena_mode = "gcp"
    settings.google_cloud_project = "proj-x"
    repo = _FakeRepo()
    acquire_calls: list[dict[str, object]] = []

    class _LeaseStore:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def acquire(self, **kwargs):
            acquire_calls.append(dict(kwargs))
            return SimpleNamespace(acquired=False, reason="lease_held", lease_id="lease_1")

    monkeypatch.setenv("CLOUD_RUN_JOB", "llm-arena-batch-agent-us")
    monkeypatch.delenv("ARENA_EXECUTION_SOURCE", raising=False)
    monkeypatch.setattr(cli, "_tenant_lease_enabled", lambda: True)
    monkeypatch.setattr(cli, "FirestoreTenantLeaseStore", _LeaseStore)
    monkeypatch.setattr(cli, "_run_agent_cycle_once", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not run")))

    cli._run_agent_cycle_once_guarded(
        True,
        settings=settings,
        repo=repo,
        orchestrator=object(),
        tenant="tenant-a",
        run_id="run-1",
        market_override="us",
    )

    assert acquire_calls
    assert acquire_calls[-1]["execution_source"] == "manual"
