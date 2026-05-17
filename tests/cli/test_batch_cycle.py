from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace

import arena.cli as cli
import pytest
from arena.config import load_settings

from tests.cli.helpers import _FakeRepo, _stub_shared_prep_environment

def test_run_pipeline_configures_logging_before_weekend_skip(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    calls: list[str] = []

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: calls.append("configure"))
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: calls.append("validate"))
    monkeypatch.setattr(
        cli,
        "nasdaq_window",
        lambda now=None: SimpleNamespace(
            now_local=SimpleNamespace(weekday=lambda: 6),
            trading_date=date(2026, 3, 8),
        ),
    )
    monkeypatch.setattr(cli, "is_nasdaq_holiday", lambda d: False)
    monkeypatch.setattr(cli, "cmd_sync_market", lambda: calls.append("sync"))
    monkeypatch.setattr(cli, "cmd_build_forecasts", lambda args: calls.append("forecast"))
    monkeypatch.setattr(cli, "cmd_run_agent_cycle", lambda **kwargs: calls.append("cycle"))

    cli.cmd_run_pipeline(live=True, all_tenants=False, market_override="us")

    assert calls == ["configure", "validate"]


def test_cmd_run_agent_cycle_skips_single_tenant_when_market_closed(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    tenant = "midnightnnn"
    calls: list[str] = []
    repo = _FakeRepo()

    monkeypatch.setenv("ARENA_TENANT_ID", tenant)
    monkeypatch.setattr(
        cli,
        "_build_runtime",
        lambda **kwargs: (
            settings,
            repo,
            object(),
        ),
    )
    monkeypatch.setattr(cli, "_run_agent_cycle_once", lambda *args, **kwargs: calls.append("run"))
    monkeypatch.setattr(
        cli,
        "nasdaq_window",
        lambda now=None: SimpleNamespace(
            now_local=SimpleNamespace(weekday=lambda: 6),
            trading_date=date(2026, 3, 14),
        ),
    )
    monkeypatch.setattr(cli, "is_nasdaq_holiday", lambda d: False)
    monkeypatch.delenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", raising=False)

    cli.cmd_run_agent_cycle(live=True, all_tenants=False, market_override="us")

    assert calls == []
    assert repo.run_status_rows
    assert repo.run_status_rows[-1]["status"] == "skipped"
    assert repo.run_status_rows[-1]["reason_code"] == "market_closed"


def test_cmd_run_agent_cycle_skips_closed_tenant_in_multi_tenant_mode(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    calls: list[str] = []

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    repo = _Repo(tenants=["midnightnnn"], latest_config_map={"midnightnnn": "us"})
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(
        cli,
        "_build_runtime",
        lambda **kwargs: (
            settings,
            repo,
            object(),
        ),
    )
    monkeypatch.setattr(cli, "_run_agent_cycle_once", lambda *args, **kwargs: calls.append("run"))
    monkeypatch.setattr(
        cli,
        "nasdaq_window",
        lambda now=None: SimpleNamespace(
            now_local=SimpleNamespace(weekday=lambda: 6),
            trading_date=date(2026, 3, 14),
        ),
    )
    monkeypatch.setattr(cli, "is_nasdaq_holiday", lambda d: False)
    monkeypatch.delenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", raising=False)

    cli.cmd_run_agent_cycle(live=True, all_tenants=True, market_override="us")

    assert calls == []
    assert repo.run_status_rows
    assert repo.run_status_rows[-1]["tenant_id"] == "midnightnnn"
    assert repo.run_status_rows[-1]["status"] == "skipped"


def test_run_agent_cycle_once_ignores_post_cycle_maintenance_failures(monkeypatch, caplog) -> None:
    settings = load_settings()

    class _Repo(_FakeRepo):
        def get_all_held_tickers(self, market=None):
            _ = market
            return []

    class _Orchestrator:
        def run_cycle(self, snapshot=None):
            _ = snapshot
            return [SimpleNamespace(status=SimpleNamespace(value="SIMULATED"))]

    class _FakeResearchAgent:
        def __init__(self, settings, repo):
            self.settings = settings
            self.repo = repo

        async def run(self, held_tickers):
            _ = held_tickers
            return []

    repo = _Repo()
    monkeypatch.setattr("arena.agents.research_agent.ResearchAgent", _FakeResearchAgent)
    monkeypatch.setattr(
        cli,
        "_run_memory_compaction",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("compaction boom")),
    )
    monkeypatch.delattr(cli, "_run_memory_forgetting_tuner_post_cycle", raising=False)

    with caplog.at_level(logging.WARNING):
        cli._run_agent_cycle_once(
            False,
            settings=settings,
            repo=repo,
            orchestrator=_Orchestrator(),
            tenant="tenant-a",
            run_id="run-1",
        )

    assert repo.run_status_rows[-1]["status"] == "success"
    assert repo.run_status_rows[-1]["stage"] == "complete"
    failure_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", "") == "post_cycle_memory_compaction_failed"
    )
    assert failure_record.exc_info is not None


def test_run_agent_cycle_once_skips_precycle_research_by_default(monkeypatch) -> None:
    settings = load_settings()

    class _Repo(_FakeRepo):
        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            raise AssertionError("pre-cycle research should not query holdings by default")

    class _Orchestrator:
        def run_cycle(self, snapshot=None):
            _ = snapshot
            return []

    class _FakeResearchAgent:
        def __init__(self, settings, repo):
            raise AssertionError("pre-cycle research should be disabled by default")

    monkeypatch.setattr("arena.agents.research_agent.ResearchAgent", _FakeResearchAgent)
    monkeypatch.setattr(cli, "_run_post_cycle_maintenance", lambda *args, **kwargs: None)

    cli._run_agent_cycle_once(
        False,
        settings=settings,
        repo=_Repo(),
        orchestrator=_Orchestrator(),
        tenant="tenant-a",
        run_id="run-1",
    )


def test_run_agent_cycle_once_research_uses_account_wide_holdings(monkeypatch) -> None:
    settings = load_settings()
    settings.research_precycle_enabled = True
    settings.kis_target_market = "us"
    seen: dict[str, list[str]] = {}

    class _Repo(_FakeRepo):
        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            assert market == "us,kospi,kosdaq"
            assert all_tenants is False
            return ["AAPL", "053580"]

        def get_all_held_tickers(self, market=None):
            _ = market
            return ["AAPL"]

    class _Orchestrator:
        def run_cycle(self, snapshot=None):
            _ = snapshot
            return []

    class _FakeResearchAgent:
        def __init__(self, settings, repo):
            self.settings = settings
            self.repo = repo

        async def run(self, held_tickers):
            seen["held"] = list(held_tickers)
            return []

    repo = _Repo()
    monkeypatch.setattr("arena.agents.research_agent.ResearchAgent", _FakeResearchAgent)
    monkeypatch.setattr(cli, "_run_post_cycle_maintenance", lambda *args, **kwargs: None)

    cli._run_agent_cycle_once(
        False,
        settings=settings,
        repo=repo,
        orchestrator=_Orchestrator(),
        tenant="tenant-a",
        run_id="run-1",
    )

    assert seen["held"] == ["AAPL", "053580"]


def test_post_cycle_maintenance_runs_relation_extraction_after_compaction(monkeypatch) -> None:
    settings = load_settings()
    calls: list[str] = []

    monkeypatch.setattr(
        cli,
        "_run_memory_compaction",
        lambda **kwargs: calls.append("compaction"),
    )
    monkeypatch.setattr(
        cli,
        "_run_memory_relation_extraction_post_cycle",
        lambda **kwargs: calls.append("relations"),
    )
    monkeypatch.setattr(
        cli,
        "_run_memory_relation_tuner_post_cycle",
        lambda **kwargs: calls.append("relation_tuner"),
    )
    monkeypatch.setattr(
        cli,
        "_run_memory_forgetting_tuner_post_cycle",
        lambda **kwargs: calls.append("forgetting"),
    )

    cli._run_post_cycle_maintenance(
        cli,
        settings=settings,
        repo=_FakeRepo(),
        orchestrator=SimpleNamespace(),
        tenant="tenant-a",
    )

    assert calls == ["compaction", "relations", "relation_tuner", "forgetting"]


def test_post_cycle_maintenance_logs_each_stage_progress(monkeypatch, caplog) -> None:
    settings = load_settings()
    calls: list[str] = []

    monkeypatch.setattr(
        cli,
        "_run_memory_compaction",
        lambda **kwargs: calls.append("memory_compaction"),
    )
    monkeypatch.setattr(
        cli,
        "_run_memory_relation_extraction_post_cycle",
        lambda **kwargs: calls.append("memory_relation_extraction"),
    )
    monkeypatch.setattr(
        cli,
        "_run_memory_relation_tuner_post_cycle",
        lambda **kwargs: calls.append("memory_relation_tuner"),
    )
    monkeypatch.setattr(
        cli,
        "_run_memory_forgetting_tuner_post_cycle",
        lambda **kwargs: calls.append("memory_forgetting_tuner"),
    )

    with caplog.at_level(logging.INFO, logger="arena.cli_commands.run_agent"):
        cli._run_post_cycle_maintenance(
            cli,
            settings=settings,
            repo=_FakeRepo(),
            orchestrator=SimpleNamespace(),
            tenant="tenant-a",
        )

    expected_stages = [
        "memory_compaction",
        "memory_relation_extraction",
        "memory_relation_tuner",
        "memory_forgetting_tuner",
    ]
    assert calls == expected_stages

    assert any(getattr(record, "event", "") == "post_cycle_maintenance_start" for record in caplog.records)
    assert any(getattr(record, "event", "") == "post_cycle_maintenance_finish" for record in caplog.records)

    starts = [
        getattr(record, "stage", "")
        for record in caplog.records
        if getattr(record, "event", "") == "post_cycle_maintenance_stage_start"
    ]
    finishes = [
        record
        for record in caplog.records
        if getattr(record, "event", "") == "post_cycle_maintenance_stage_finish"
    ]

    assert starts == expected_stages
    assert [getattr(record, "stage", "") for record in finishes] == expected_stages
    assert all(getattr(record, "status", "") == "ok" for record in finishes)
    assert all(isinstance(getattr(record, "elapsed_ms", None), int) for record in finishes)


def test_cmd_run_agent_cycle_multi_tenant_applies_task_shard_before_building(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    built: list[str] = []
    executed: list[str] = []

    class _BootstrapRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    bootstrap_repo = _BootstrapRepo(
        tenants=["tenant-d", "tenant-c", "tenant-b", "tenant-a"],
        latest_config_map={
            "tenant-a": "us",
            "tenant-b": "us",
            "tenant-c": "us",
            "tenant-d": "us",
        },
    )

    monkeypatch.setenv("CLOUD_RUN_TASK_INDEX", "1")
    monkeypatch.setenv("CLOUD_RUN_TASK_COUNT", "2")
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: bootstrap_repo)
    monkeypatch.setattr(cli, "_resolve_batch_tenants", lambda repo, fallback="local": ["tenant-d", "tenant-c", "tenant-b", "tenant-a"])
    monkeypatch.setattr(
        cli,
        "_build_runtime",
        lambda **kwargs: (
            built.append(kwargs["tenant_id"]) or settings,
            bootstrap_repo,
            object(),
        ),
    )
    monkeypatch.setattr(cli, "_run_agent_cycle_once_guarded", lambda *args, **kwargs: executed.append(kwargs["tenant"]))

    cli.cmd_run_agent_cycle(live=False, all_tenants=True, market_override="us")

    assert built == ["tenant-b", "tenant-d"]
    assert executed == ["tenant-b", "tenant-d"]


def test_cmd_run_agent_cycle_multi_tenant_prefilters_market_before_build(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    built: list[str] = []
    executed: list[str] = []

    class _BootstrapRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    bootstrap_repo = _BootstrapRepo(
        tenants=["tenant-a", "tenant-b", "tenant-c"],
        latest_config_map={
            "tenant-a": "us",
            "tenant-b": "kospi",
            "tenant-c": "nasdaq",
        },
    )

    monkeypatch.delenv("CLOUD_RUN_TASK_INDEX", raising=False)
    monkeypatch.delenv("CLOUD_RUN_TASK_COUNT", raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: bootstrap_repo)
    monkeypatch.setattr(
        cli,
        "_build_runtime",
        lambda **kwargs: (
            built.append(kwargs["tenant_id"]) or settings,
            bootstrap_repo,
            object(),
        ),
    )
    monkeypatch.setattr(cli, "_run_agent_cycle_once_guarded", lambda *args, **kwargs: executed.append(kwargs["tenant"]))

    cli.cmd_run_agent_cycle(live=False, all_tenants=True, market_override="us")

    assert built == ["tenant-a", "tenant-c"]
    assert executed == ["tenant-a", "tenant-c"]


def test_cmd_run_agent_cycle_multi_tenant_failure_summary_tracks_selected_and_runtime_counts(monkeypatch, caplog) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    class _BootstrapRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    bootstrap_repo = _BootstrapRepo(
        tenants=["tenant-a", "tenant-b"],
        latest_config_map={
            "tenant-a": "us",
            "tenant-b": "us",
        },
    )

    monkeypatch.delenv("CLOUD_RUN_TASK_INDEX", raising=False)
    monkeypatch.delenv("CLOUD_RUN_TASK_COUNT", raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: bootstrap_repo)
    monkeypatch.setattr(cli, "_resolve_batch_tenants", lambda repo, fallback="local": ["tenant-a", "tenant-b"])

    def _build_runtime(**kwargs):
        if kwargs["tenant_id"] == "tenant-a":
            raise RuntimeError("build boom")
        return settings, bootstrap_repo, object()

    def _run_agent_cycle_once_guarded(*args, **kwargs):
        raise RuntimeError("exec boom")

    monkeypatch.setattr(cli, "_build_runtime", _build_runtime)
    monkeypatch.setattr(cli, "_run_agent_cycle_once_guarded", _run_agent_cycle_once_guarded)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
        cli.cmd_run_agent_cycle(live=False, all_tenants=True, market_override="us")

    record = next(
        item
        for item in caplog.records
        if getattr(item, "event", "") == "agent_cycle_multi_tenant_completed_with_failures"
    )
    assert record.tenant_count == 2
    assert record.runtime_count == 1
    assert record.build_failed_count == 1
    assert record.execution_failed_count == 1
    assert record.failed_count == 2


def test_cmd_run_batch_multi_tenant_failure_summary_tracks_selected_and_runtime_counts(monkeypatch, caplog) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    class _BootstrapRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    bootstrap_repo = _BootstrapRepo(tenants=["tenant-a", "tenant-b"])

    monkeypatch.delenv("CLOUD_RUN_TASK_INDEX", raising=False)
    monkeypatch.delenv("CLOUD_RUN_TASK_COUNT", raising=False)
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: bootstrap_repo)
    monkeypatch.setattr(cli, "_resolve_batch_tenants", lambda repo, fallback="local": ["tenant-a", "tenant-b"])
    monkeypatch.setattr(cli, "_partition_tenants_for_task", lambda tenants: tenants)

    def _build_runtime(**kwargs):
        if kwargs["tenant_id"] == "tenant-a":
            raise RuntimeError("build boom")
        return settings, bootstrap_repo, object()

    def _batch_tenant_work(*args, **kwargs):
        raise RuntimeError("exec boom")

    monkeypatch.setattr(cli, "_build_runtime", _build_runtime)
    monkeypatch.setattr(cli, "_batch_phase", lambda *args, **kwargs: ("open_cycle", None))
    monkeypatch.setattr(cli, "_batch_market_sync", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_batch_tenant_work", _batch_tenant_work)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
        cli.cmd_run_batch(live=False, all_tenants=True, market_override="")

    record = next(
        item
        for item in caplog.records
        if getattr(item, "event", "") == "batch_multi_tenant_completed_with_failures"
    )
    assert record.tenant_count == 2
    assert record.runtime_count == 1
    assert record.build_failed_count == 1
    assert record.execution_failed_count == 1
    assert record.failed_count == 2
