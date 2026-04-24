from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace

import arena.cli as cli
import pytest
from arena.config import load_settings


class _FakeRepo:
    def __init__(
        self,
        *,
        row: dict | None = None,
        rows_by_tenant: dict[str, dict] | None = None,
        tenants: list[str] | None = None,
        latest_config_map: dict[str, str] | None = None,
        latest_agents_config_map: dict[str, str] | None = None,
    ) -> None:
        self._row = row
        self._rows_by_tenant = {
            str(key).strip().lower(): dict(value)
            for key, value in dict(rows_by_tenant or {}).items()
            if str(key).strip()
        }
        self._tenants = list(tenants or [])
        self._latest_config_map = {
            str(key).strip().lower(): str(value)
            for key, value in dict(latest_config_map or {}).items()
            if str(key).strip()
        }
        self._latest_agents_config_map = {
            str(key).strip().lower(): str(value)
            for key, value in dict(latest_agents_config_map or {}).items()
            if str(key).strip()
        }
        self.latest_tenant: str | None = None
        self.run_status_rows: list[dict[str, object]] = []

    def latest_runtime_credentials(self, *, tenant_id: str) -> dict | None:
        self.latest_tenant = tenant_id
        tenant = str(tenant_id or "").strip().lower()
        if tenant in self._rows_by_tenant:
            return dict(self._rows_by_tenant[tenant])
        return self._row

    def list_runtime_tenants(self, *, limit: int = 200) -> list[str]:
        _ = limit
        return list(self._tenants)

    def latest_config_values(self, *, config_key: str, tenant_ids: list[str] | None = None) -> dict[str, str]:
        ids = [str(token).strip().lower() for token in (tenant_ids or []) if str(token).strip()]
        if config_key == "kis_target_market":
            return {tenant: self._latest_config_map.get(tenant, "") for tenant in ids}
        if config_key == "agents_config":
            return {tenant: self._latest_agents_config_map.get(tenant, "") for tenant in ids}
        return {tenant: "" for tenant in ids}

    def append_tenant_run_status(self, **kwargs) -> None:
        self.run_status_rows.append(dict(kwargs))


def test_parse_tenant_tokens_normalizes_and_dedupes() -> None:
    assert cli._parse_tenant_tokens(" Tenant-A, local|Tenant-A ; ALPHA  ") == ["tenant-a", "local", "alpha"]


def test_resolve_batch_tenants_prefers_env_list(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_PUBLIC_DEMO_TENANT", raising=False)
    monkeypatch.setenv("ARENA_BATCH_TENANTS", "a,b,a")
    repo = _FakeRepo(tenants=["tenant-x"])
    assert cli._resolve_batch_tenants(repo, fallback="local") == ["a", "b"]


def test_resolve_batch_tenants_uses_repo_when_env_missing(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_PUBLIC_DEMO_TENANT", raising=False)
    monkeypatch.delenv("ARENA_BATCH_TENANTS", raising=False)
    repo = _FakeRepo(tenants=["tenant-a", "tenant-b"])
    assert cli._resolve_batch_tenants(repo, fallback="local") == ["tenant-a", "tenant-b"]


def test_resolve_batch_tenants_raises_when_none_found(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_PUBLIC_DEMO_TENANT", raising=False)
    monkeypatch.delenv("ARENA_BATCH_TENANTS", raising=False)
    repo = _FakeRepo(tenants=[])
    with pytest.raises(RuntimeError, match="no runtime tenants resolved"):
        cli._resolve_batch_tenants(repo, fallback="Tenant-Z")


def test_resolve_batch_tenants_appends_public_demo_tenant(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_BATCH_TENANTS", raising=False)
    monkeypatch.setenv("ARENA_PUBLIC_DEMO_TENANT", "midnightnnn")
    repo = _FakeRepo(tenants=["tenant-a"])

    assert cli._resolve_batch_tenants(repo, fallback="local") == ["tenant-a", "midnightnnn"]


def test_resolve_batch_tenants_uses_public_demo_tenant_when_none_registered(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_BATCH_TENANTS", raising=False)
    monkeypatch.setenv("ARENA_PUBLIC_DEMO_TENANT", "midnightnnn")
    repo = _FakeRepo(tenants=[])

    assert cli._resolve_batch_tenants(repo, fallback="local") == ["midnightnnn"]


def test_apply_tenant_runtime_credentials_returns_none_when_missing(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TENANT_ID", "tenant-a")
    settings = load_settings()
    repo = _FakeRepo(row=None)

    out = cli._apply_tenant_runtime_credentials(settings, repo)

    assert out is None
    assert repo.latest_tenant == "tenant-a"


def test_apply_tenant_runtime_credentials_applies_secret_payload(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TENANT_ID", "tenant-a")
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.openai_api_key = "env-openai"
    settings.gemini_api_key = "env-gemini"
    settings.anthropic_api_key = "env-anthropic"

    row = {
        "tenant_id": "tenant-a",
        "kis_secret_name": "kis-tenant-a",
        "model_secret_name": "models-tenant-a",
        "kis_env": "demo",
    }
    repo = _FakeRepo(row=row)

    calls: list[tuple[str, str, str]] = []

    def _fake_load_secret_json(*, project: str, secret_name: str, version: str = "latest") -> dict:
        calls.append((project, secret_name, version))
        return {
            "openai_api_key": "tenant-openai",
            "gemini_api_key": "",
            "anthropic_api_key": "tenant-anthropic",
        }

    monkeypatch.setattr(cli, "_load_secret_json", _fake_load_secret_json)

    out = cli._apply_tenant_runtime_credentials(settings, repo)

    assert out == row
    assert repo.latest_tenant == "tenant-a"
    assert calls == [("proj-x", "models-tenant-a", "latest")]
    assert settings.kis_secret_name == "kis-tenant-a"
    assert settings.kis_env == "demo"
    assert settings.openai_api_key == "tenant-openai"
    assert settings.gemini_api_key == ""
    assert settings.anthropic_api_key == "tenant-anthropic"


def test_apply_tenant_runtime_credentials_applies_provider_secret_payload(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TENANT_ID", "tenant-a")
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.openai_api_key = "env-openai"
    settings.gemini_api_key = "env-gemini"
    settings.anthropic_api_key = "env-anthropic"

    row = {
        "tenant_id": "tenant-a",
        "kis_secret_name": "kis-tenant-a",
        "model_secret_name": "models-tenant-a",
        "kis_env": "demo",
    }
    repo = _FakeRepo(row=row)

    monkeypatch.setattr(
        cli,
        "_load_secret_json",
        lambda **kwargs: {
            "providers": {
                "openai": {"api_key": "tenant-openai"},
                "anthropic": {"api_key": "tenant-anthropic"},
            }
        },
    )

    out = cli._apply_tenant_runtime_credentials(settings, repo)

    assert out == row
    assert settings.openai_api_key == "tenant-openai"
    assert settings.gemini_api_key == ""
    assert settings.anthropic_api_key == "tenant-anthropic"
    assert settings.provider_secrets == {
        "gpt": {"api_key": "tenant-openai"},
        "claude": {"api_key": "tenant-anthropic"},
    }


def test_build_runtime_does_not_restore_shared_gemini_for_research(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT", "midnightnnn")
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.research_enabled = True
    settings.openai_api_key = "shared-openai"
    settings.gemini_api_key = "shared-gemini"
    settings.anthropic_api_key = ""
    settings.kis_secret_name = "shared-kis"
    settings.kis_account_no = "12345678"

    runtime_row = {
        "tenant_id": "tenant-a",
        "kis_secret_name": "kis-tenant-a",
        "model_secret_name": "models-tenant-a",
        "kis_env": "demo",
    }

    class _RuntimeRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    repo = _RuntimeRepo(row=runtime_row)

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(
        cli,
        "_load_secret_json",
        lambda **kwargs: {
            "openai_api_key": "tenant-openai",
            "gemini_api_key": "",
            "anthropic_api_key": "",
        },
    )
    monkeypatch.setattr(cli, "apply_runtime_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "MemoryStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "BoardStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ContextBuilder", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "RiskEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "PaperBroker", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ExecutionGateway", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_build_agents", lambda *args, **kwargs: ["gpt-agent"])

    class _FakeOrchestrator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(cli, "ArenaOrchestrator", _FakeOrchestrator)

    out_settings, _, _ = cli._build_runtime(
        live=False,
        require_kis=True,
        tenant_id="tenant-a",
        require_tenant_runtime_credentials=True,
    )

    assert out_settings.openai_api_key == "tenant-openai"
    assert out_settings.gemini_api_key == ""
    assert out_settings.research_gemini_api_key == ""


def test_build_runtime_restores_shared_research_gemini_for_approved_live_tenant(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT", "midnightnnn")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.research_enabled = True
    settings.openai_api_key = "shared-openai"
    settings.gemini_api_key = "shared-gemini"
    settings.research_gemini_api_key = ""
    settings.anthropic_api_key = ""
    settings.kis_secret_name = "shared-kis"
    settings.kis_account_no = "12345678"

    runtime_rows = {
        "tenant-a": {
            "tenant_id": "tenant-a",
            "kis_secret_name": "kis-tenant-a",
            "model_secret_name": "models-tenant-a",
            "kis_env": "demo",
        },
        "midnightnnn": {
            "tenant_id": "midnightnnn",
            "kis_secret_name": "kis-midnightnnn",
            "model_secret_name": "models-midnightnnn",
            "kis_env": "real",
        },
    }

    class _RuntimeRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    repo = _RuntimeRepo(rows_by_tenant=runtime_rows)

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)

    def _fake_load_secret_json(*, secret_name: str, **kwargs) -> dict:
        if secret_name == "models-tenant-a":
            return {
                "openai_api_key": "tenant-openai",
                "gemini_api_key": "",
                "anthropic_api_key": "",
            }
        if secret_name == "models-midnightnnn":
            return {
                "providers": {
                    "gemini": {"api_key": "shared-research-gemini"},
                }
            }
        raise AssertionError(secret_name)

    monkeypatch.setattr(cli, "_load_secret_json", _fake_load_secret_json)

    def _fake_apply_runtime_overrides(settings, repo, tenant_id):
        _ = repo, tenant_id
        settings.distribution_mode = "private"
        settings.real_trading_approved = True
        return settings

    monkeypatch.setattr(cli, "apply_runtime_overrides", _fake_apply_runtime_overrides)
    monkeypatch.setattr(cli, "MemoryStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "BoardStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ContextBuilder", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "RiskEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "PaperBroker", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ExecutionGateway", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_build_agents", lambda *args, **kwargs: ["gpt-agent"])

    class _FakeOrchestrator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(cli, "ArenaOrchestrator", _FakeOrchestrator)

    out_settings, _, _ = cli._build_runtime(
        live=False,
        require_kis=True,
        tenant_id="tenant-a",
        require_tenant_runtime_credentials=True,
    )

    assert out_settings.openai_api_key == "tenant-openai"
    assert out_settings.gemini_api_key == ""
    assert out_settings.research_gemini_api_key == "shared-research-gemini"
    assert out_settings.research_gemini_source == "shared_live_tenant"
    assert out_settings.research_gemini_source_tenant == "midnightnnn"


def test_prepare_kis_command_repo_applies_runtime_overrides_before_validation(monkeypatch) -> None:
    settings = load_settings()
    settings.real_trading_approved = False
    calls: list[object] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append("dataset")

        def ensure_tables(self):
            calls.append("tables")

    repo = _Repo(row={"tenant_id": "midnightnnn", "kis_secret_name": "kis-midnightnnn", "kis_env": "real"})

    monkeypatch.setenv("ARENA_TENANT_ID", "midnightnnn")
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)

    def _fake_apply_credentials(settings, repo, *, tenant_id=None):
        calls.append(("credentials", tenant_id))
        return {"tenant_id": tenant_id}

    def _fake_apply_runtime_overrides(settings, repo, tenant_id):
        calls.append(("overrides", tenant_id))
        settings.real_trading_approved = True
        return settings

    validations: list[tuple[bool, dict[str, object]]] = []

    def _fake_validate(settings, **kwargs):
        validations.append((settings.real_trading_approved, dict(kwargs)))

    monkeypatch.setattr(cli, "_apply_tenant_runtime_credentials", _fake_apply_credentials)
    monkeypatch.setattr(cli, "apply_runtime_overrides", _fake_apply_runtime_overrides)
    monkeypatch.setattr(cli, "_validate_or_exit", _fake_validate)

    out_repo = cli._prepare_kis_command_repo(settings)

    assert out_repo is repo
    assert calls == [
        "dataset",
        "tables",
        ("credentials", "midnightnnn"),
        ("overrides", "midnightnnn"),
    ]
    assert validations == [(True, {"require_kis": True})]


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


def test_cmd_approve_live_tenant_sets_config_and_audit(monkeypatch) -> None:
    settings = load_settings()
    config_writes: list[tuple[str, str, str, str]] = []
    audit_rows: list[dict[str, object]] = []

    class _ApprovalRepo:
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

        def set_config(self, tenant_id: str, config_key: str, value: str, updated_by: str | None = None, **kwargs):
            _ = kwargs
            config_writes.append((tenant_id, config_key, value, str(updated_by or "")))

        def append_runtime_audit_log(self, **kwargs):
            audit_rows.append(dict(kwargs))

    repo = _ApprovalRepo()
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)

    cli.cmd_approve_live_tenant(
        tenant_id="midnightnnn",
        approved=True,
        updated_by="tester@example.com",
        note="internal allowlist",
    )

    assert ("midnightnnn", "real_trading_approved", "true", "tester@example.com") in config_writes
    assert ("midnightnnn", "real_trading_approval_note", "internal allowlist", "tester@example.com") in config_writes
    assert audit_rows
    assert audit_rows[0]["tenant_id"] == "midnightnnn"
    assert audit_rows[0]["detail"]["approved"] is True


def test_cmd_promote_tenant_live_sets_private_mode_and_approval(monkeypatch) -> None:
    settings = load_settings()
    config_writes: list[tuple[str, str, str, str]] = []
    audit_rows: list[dict[str, object]] = []

    class _ApprovalRepo:
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

        def set_config(self, tenant_id: str, config_key: str, value: str, updated_by: str | None = None, **kwargs):
            _ = kwargs
            config_writes.append((tenant_id, config_key, value, str(updated_by or "")))

        def append_runtime_audit_log(self, **kwargs):
            audit_rows.append(dict(kwargs))

    repo = _ApprovalRepo()
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)

    cli.cmd_promote_tenant_live(
        tenant_id="midnightnnn",
        updated_by="tester@example.com",
        note="graduated from demo",
    )

    assert ("midnightnnn", "distribution_mode", "private", "tester@example.com") in config_writes
    assert ("midnightnnn", "real_trading_approved", "true", "tester@example.com") in config_writes
    assert audit_rows
    assert audit_rows[0]["tenant_id"] == "midnightnnn"
    assert audit_rows[0]["detail"]["distribution_mode"] == "private"


def test_cmd_set_tenant_simulated_resets_mode_and_approval(monkeypatch) -> None:
    settings = load_settings()
    config_writes: list[tuple[str, str, str, str]] = []
    audit_rows: list[dict[str, object]] = []

    class _ApprovalRepo:
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

        def set_config(self, tenant_id: str, config_key: str, value: str, updated_by: str | None = None, **kwargs):
            _ = kwargs
            config_writes.append((tenant_id, config_key, value, str(updated_by or "")))

        def append_runtime_audit_log(self, **kwargs):
            audit_rows.append(dict(kwargs))

    repo = _ApprovalRepo()
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)

    cli.cmd_set_tenant_simulated(
        tenant_id="midnightnnn",
        updated_by="tester@example.com",
        note="reset onboarding",
    )

    assert ("midnightnnn", "distribution_mode", "simulated_only", "tester@example.com") in config_writes
    assert ("midnightnnn", "real_trading_approved", "false", "tester@example.com") in config_writes
    assert audit_rows
    assert audit_rows[0]["tenant_id"] == "midnightnnn"
    assert audit_rows[0]["detail"]["distribution_mode"] == "simulated_only"


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


def test_run_agent_cycle_once_ignores_post_cycle_maintenance_failures(monkeypatch) -> None:
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


def test_partition_tenants_for_task_uses_round_robin(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TASK_SHARD_INDEX", "1")
    monkeypatch.setenv("ARENA_TASK_SHARD_COUNT", "3")

    out = cli._partition_tenants_for_task(["tenant-c", "tenant-a", "tenant-e", "tenant-b", "tenant-d"])

    assert out == ["tenant-b", "tenant-e"]


def test_filter_tenants_by_market_uses_latest_config_values(monkeypatch) -> None:
    settings = load_settings()

    class _Repo:
        def latest_config_values(self, *, config_key: str, tenant_ids: list[str] | None = None) -> dict[str, str]:
            assert config_key == "kis_target_market"
            assert tenant_ids == ["tenant-a", "tenant-b", "tenant-c"]
            return {
                "tenant-a": "us",
                "tenant-b": "kospi",
                "tenant-c": "",
            }

    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    out = cli._filter_tenants_by_market(_Repo(), ["tenant-a", "tenant-b", "tenant-c"], "us")

    assert out == ["tenant-a"]


def test_filter_tenants_by_market_skips_tenant_without_tenant_market(monkeypatch) -> None:
    settings = load_settings()

    class _Repo:
        def latest_config_values(self, *, config_key: str, tenant_ids: list[str] | None = None) -> dict[str, str]:
            assert config_key == "kis_target_market"
            assert tenant_ids == ["tenant-a", "tenant-b"]
            return {"tenant-a": "", "tenant-b": "us"}

    monkeypatch.setattr(cli, "load_settings", lambda: settings)

    out = cli._filter_tenants_by_market(_Repo(), ["tenant-a", "tenant-b"], "us")

    assert out == ["tenant-b"]


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


def test_cmd_backfill_tenant_markets_derives_from_agents_config(monkeypatch) -> None:
    settings = load_settings()
    config_writes: list[tuple[str, str, str, str]] = []
    audit_rows: list[dict[str, object]] = []

    class _Repo:
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

        def list_runtime_tenants(self, *, limit: int = 2000) -> list[str]:
            _ = limit
            return ["tenant-a", "tenant-b", "tenant-c"]

        def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]:
            assert config_keys == ["agents_config", "kis_target_market"]
            payload = {
                "tenant-a": {
                    "agents_config": '[{"id":"gpt","provider":"gpt","target_market":"us"},{"id":"claude","provider":"claude","target_market":"kospi"}]',
                    "kis_target_market": "",
                },
                "tenant-b": {
                    "agents_config": '[{"id":"gpt","provider":"gpt","target_market":"us"}]',
                    "kis_target_market": "us",
                },
                "tenant-c": {
                    "agents_config": "[]",
                    "kis_target_market": "",
                },
            }
            return payload[tenant_id]

        def set_config(self, tenant_id: str, config_key: str, value: str, updated_by: str | None = None, **kwargs):
            _ = kwargs
            config_writes.append((tenant_id, config_key, value, str(updated_by or "")))

        def append_runtime_audit_log(self, **kwargs):
            audit_rows.append(dict(kwargs))

    repo = _Repo()
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)

    cli.cmd_backfill_tenant_markets(updated_by="tester@example.com")

    assert config_writes == [("tenant-a", "kis_target_market", "us,kospi", "tester@example.com")]
    assert audit_rows
    assert audit_rows[0]["tenant_id"] == "tenant-a"
    assert audit_rows[0]["detail"]["kis_target_market"] == "us,kospi"


def test_build_runtime_execution_market_overrides_tenant_market(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.kis_target_market = "kospi"
    settings.openai_api_key = "shared-openai"
    settings.gemini_api_key = ""
    settings.anthropic_api_key = ""
    settings.kis_secret_name = "shared-kis"
    settings.kis_account_no = "12345678"

    class _RuntimeRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    repo = _RuntimeRepo(row={"tenant_id": "tenant-a"})

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "apply_runtime_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "MemoryStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "BoardStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ContextBuilder", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "RiskEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "PaperBroker", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ExecutionGateway", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_build_agents", lambda *args, **kwargs: ["gpt-agent"])
    monkeypatch.setattr(cli, "ArenaOrchestrator", lambda **kwargs: object())

    out_settings, _, _ = cli._build_runtime(
        live=False,
        require_kis=True,
        tenant_id="tenant-a",
        execution_market="us",
    )

    assert out_settings.kis_target_market == "us"


def test_build_runtime_syncs_trading_mode_to_live_flag(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.openai_api_key = "shared-openai"
    settings.gemini_api_key = ""
    settings.anthropic_api_key = ""

    class _RuntimeRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    repo = _RuntimeRepo(row={"tenant_id": "tenant-a"})

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "apply_runtime_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "MemoryStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "BoardStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ContextBuilder", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "RiskEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "KISOpenTradingBroker", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ExecutionGateway", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_build_agents", lambda *args, **kwargs: ["gpt-agent"])
    monkeypatch.setattr(cli, "ArenaOrchestrator", lambda **kwargs: object())

    out_settings, _, _ = cli._build_runtime(
        live=True,
        require_kis=False,
        tenant_id="tenant-a",
    )

    assert out_settings.trading_mode == "live"


def test_build_forecast_tickers_uses_quote_aware_sources() -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    settings.default_universe = []
    settings.universe_run_top_n = 3

    class _Repo:
        def __init__(self) -> None:
            self.last_sources = None

        def latest_universe_candidate_tickers(self, *, limit=200):
            _ = limit
            return ["AAPL", "MSFT", "TSLA"]

        def latest_market_features(self, *, tickers, limit, sources=None):
            self.last_sources = list(sources or [])
            _ = (tickers, limit)
            return [
                {"ticker": "AAPL", "ret_20d": 0.12, "ret_5d": 0.03, "volatility_20d": 0.18, "sentiment_score": 0.2},
                {"ticker": "MSFT", "ret_20d": 0.05, "ret_5d": -0.01, "volatility_20d": 0.12, "sentiment_score": 0.1},
                {"ticker": "TSLA", "ret_20d": -0.08, "ret_5d": 0.04, "volatility_20d": 0.35, "sentiment_score": 0.0},
            ]

        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            _ = (lookback_days, sources)
            base = {"AAPL": 100.0, "MSFT": 120.0, "TSLA": 80.0}
            slopes = {"AAPL": 0.6, "MSFT": 0.2, "TSLA": -0.15}
            out = {}
            for ticker in tickers:
                start = base.get(ticker, 100.0)
                slope = slopes.get(ticker, 0.1)
                out[ticker] = [start + slope * idx for idx in range(140)]
            return out

        def latest_fundamentals_snapshot(self, *, tickers=None, limit=500):
            _ = limit
            allow = set(tickers or [])
            rows = [
                {"ticker": "AAPL", "per": 25.0, "pbr": 8.0, "eps": 5.0, "bps": 18.0, "roe": 16.0, "debt_ratio": 110.0},
                {"ticker": "MSFT", "per": 12.0, "pbr": 2.0, "eps": 9.0, "bps": 35.0, "roe": 19.0, "debt_ratio": 70.0},
            ]
            return [row for row in rows if not allow or row["ticker"] in allow]

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            assert market == "us"
            assert all_tenants is True
            return ["GILD", "AAPL"]

    repo = _Repo()

    out = cli._build_forecast_tickers(repo, settings, top_n=5)

    assert repo.last_sources == [
        "open_trading_us_quote",
        "open_trading_us",
        "open_trading_nasdaq_quote",
        "open_trading_nasdaq",
        "open_trading_nyse_quote",
        "open_trading_nyse",
        "open_trading_amex_quote",
        "open_trading_amex",
    ]
    assert "GILD" in out
    assert {"AAPL", "MSFT"}.issubset(set(out))


def test_build_forecast_tickers_passes_market_scope_to_runtime_universe() -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    settings.default_universe = []
    settings.universe_run_top_n = 3

    class _Repo:
        def __init__(self) -> None:
            self.last_markets = None

        def latest_universe_candidate_tickers(self, *, limit=200, markets=None):
            _ = limit
            self.last_markets = list(markets or [])
            return ["AAPL", "MSFT", "TSLA"]

        def latest_market_features(self, *, tickers, limit, sources=None):
            _ = (tickers, limit, sources)
            return [
                {"ticker": "AAPL", "ret_20d": 0.12, "ret_5d": 0.03, "volatility_20d": 0.18, "sentiment_score": 0.2},
                {"ticker": "MSFT", "ret_20d": 0.05, "ret_5d": -0.01, "volatility_20d": 0.12, "sentiment_score": 0.1},
                {"ticker": "TSLA", "ret_20d": -0.08, "ret_5d": 0.04, "volatility_20d": 0.35, "sentiment_score": 0.0},
            ]

        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            _ = (lookback_days, sources)
            return {str(ticker): [100.0 + idx for idx in range(140)] for ticker in tickers}

        def latest_fundamentals_snapshot(self, *, tickers=None, limit=500):
            _ = (tickers, limit)
            return []

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            _ = (market, all_tenants)
            return []

    repo = _Repo()

    cli._build_forecast_tickers(repo, settings, top_n=5)

    assert repo.last_markets == ["us"]


def test_batch_phase_uses_daily_sources_for_live_us_seed_probe(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    class _Repo:
        def __init__(self) -> None:
            self.coverage_calls: list[tuple[str, date]] = []
            self.distinct_calls: list[str] = []

        def market_daily_ticker_coverage(self, *, source: str, day: date) -> int:
            self.coverage_calls.append((source, day))
            return 0

        def market_source_distinct_tickers(self, *, source: str) -> int:
            self.distinct_calls.append(source)
            return 120

    repo = _Repo()
    now = datetime(2026, 3, 24, 19, 0, tzinfo=timezone.utc)
    window = SimpleNamespace(
        phase="OPEN",
        now_local=now,
        open_utc=now,
        close_utc=now,
        trading_date=date(2026, 3, 24),
    )

    monkeypatch.setenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", "true")
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(cli, "nasdaq_window", lambda _: window)

    phase, out_window = cli._batch_phase(True, settings, repo)

    assert phase == "seed"
    assert out_window is window
    assert repo.coverage_calls[0][0] == "open_trading_us"
    assert all(not source.endswith("_quote") for source, _ in repo.coverage_calls)
    assert repo.distinct_calls == []


def test_batch_phase_uses_recent_daily_coverage_for_live_us_open_cycle(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    class _Repo:
        def __init__(self) -> None:
            self.coverage_calls: list[tuple[str, date]] = []

        def market_daily_ticker_coverage(self, *, source: str, day: date) -> int:
            self.coverage_calls.append((source, day))
            if source == "open_trading_us" and day == date(2026, 3, 23):
                return 120
            return 0

    repo = _Repo()
    now = datetime(2026, 3, 24, 19, 0, tzinfo=timezone.utc)
    window = SimpleNamespace(
        phase="OPEN",
        now_local=now,
        open_utc=now,
        close_utc=now,
        trading_date=date(2026, 3, 24),
    )

    monkeypatch.setenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", "true")
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(cli, "nasdaq_window", lambda _: window)

    phase, out_window = cli._batch_phase(True, settings, repo)

    assert phase == "open_cycle"
    assert out_window is window
    assert ("open_trading_us", date(2026, 3, 24)) in repo.coverage_calls
    assert ("open_trading_us", date(2026, 3, 23)) in repo.coverage_calls


def test_batch_phase_uses_freshest_recent_daily_coverage_for_seed_decision(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    class _Repo:
        def market_daily_ticker_coverage(self, *, source: str, day: date) -> int:
            if source != "open_trading_us":
                return 0
            if day == date(2026, 3, 24):
                return 10
            if day == date(2026, 3, 23):
                return 120
            return 0

    repo = _Repo()
    now = datetime(2026, 3, 24, 19, 0, tzinfo=timezone.utc)
    window = SimpleNamespace(
        phase="OPEN",
        now_local=now,
        open_utc=now,
        close_utc=now,
        trading_date=date(2026, 3, 24),
    )

    monkeypatch.setenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", "true")
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(cli, "nasdaq_window", lambda _: window)

    phase, out_window = cli._batch_phase(True, settings, repo)

    assert phase == "seed"
    assert out_window is window


def test_cmd_run_shared_prep_dispatches_agent_job(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    repo = _Repo()

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "_apply_tenant_runtime_credentials", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_batch_phase", lambda *args, **kwargs: ("open_cycle", None))
    monkeypatch.setattr(cli, "_batch_market_sync", lambda *args, **kwargs: calls.append(("sync", None)))

    def _forecast_fn(args):
        calls.append(("forecast", args.horizon))
        return SimpleNamespace(
            rows_written=42, run_date="2026-03-13", tickers_used=10,
            used_neuralforecast=True, model_names=["nbeatsx"], note="",
        )

    def _ranker_fn(args):
        calls.append(("ranker", args.horizon))
        return SimpleNamespace(
            status="ok", ranker_version="v", training_rows=100,
            validation_rows=10, scoring_rows=50, scores_written=7,
            oos_ic_20d=0.1, oos_hit_rate_20d=0.55, note="",
        )

    monkeypatch.setattr(cli, "cmd_build_forecasts", _forecast_fn)
    monkeypatch.setattr(cli, "cmd_build_opportunity_ranker", _ranker_fn)
    monkeypatch.setattr(cli, "_dispatch_agent_job", lambda settings, job_name: calls.append(("dispatch", job_name)))
    monkeypatch.setattr(
        run_pipeline_mod,
        "_same_day_quote_rows_present",
        lambda *args, **kwargs: (False, {"count": 0}),
    )
    monkeypatch.setattr(
        run_pipeline_mod,
        "_upstream_market_freshness",
        lambda *args, **kwargs: (True, {"age_days": 0}),
    )
    monkeypatch.setattr(
        cli,
        "nasdaq_window",
        lambda now=None: SimpleNamespace(
            now_local=SimpleNamespace(weekday=lambda: 4),
            trading_date=date(2026, 3, 13),
        ),
    )
    monkeypatch.setattr(cli, "is_nasdaq_holiday", lambda d: False)

    cli.cmd_run_shared_prep(live=True, market_override="us", dispatch_job="agent-us")

    assert ("sync", None) in calls
    assert ("forecast", 20) in calls
    assert ("ranker", 20) in calls
    assert ("dispatch", "agent-us") in calls


def _stub_shared_prep_environment(
    monkeypatch,
    settings,
    repo,
    calls: list,
    *,
    phase: str = "open_cycle",
    session_ready: tuple[bool, dict] = (True, {"session_id": "sp_test"}),
    forecast_rows: int = 42,
    ranker_scores: int = 7,
    ranker_status: str = "ok",
    sync_result: object | None = None,
) -> None:
    from arena.cli_commands import run_pipeline as run_pipeline_mod

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "_apply_tenant_runtime_credentials", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_batch_phase", lambda *args, **kwargs: (phase, None))
    effective_sync_result = sync_result
    if effective_sync_result is None:
        effective_sync_result = SimpleNamespace(inserted_rows=11, attempted_tickers=11, failed_tickers=[])

    def _fake_batch_market_sync(*args, **kwargs):
        _ = (args, kwargs)
        calls.append(("sync", None))
        return effective_sync_result

    monkeypatch.setattr(cli, "_batch_market_sync", _fake_batch_market_sync)

    def _fake_forecast(args):
        calls.append(("forecast", args.horizon))
        return SimpleNamespace(rows_written=forecast_rows, run_date="2026-03-13", tickers_used=10, used_neuralforecast=True, model_names=["nbeatsx"], note="")

    def _fake_ranker(args):
        calls.append(("ranker", args.horizon))
        return SimpleNamespace(
            status=ranker_status,
            ranker_version="v-test",
            training_rows=100,
            validation_rows=10,
            scoring_rows=50,
            scores_written=ranker_scores,
            oos_ic_20d=0.1,
            oos_hit_rate_20d=0.55,
            note="",
        )

    monkeypatch.setattr(cli, "cmd_build_forecasts", _fake_forecast)
    monkeypatch.setattr(cli, "cmd_refresh_fundamentals_derived", lambda args: calls.append(("fundamentals", args.lookback_days)))
    monkeypatch.setattr(cli, "cmd_build_opportunity_ranker", _fake_ranker)
    monkeypatch.setattr(cli, "_dispatch_agent_job", lambda settings, job_name: calls.append(("dispatch", job_name)))
    monkeypatch.setattr(
        run_pipeline_mod,
        "_shared_prep_session_ready",
        lambda *args, **kwargs: session_ready,
    )
    monkeypatch.setattr(
        run_pipeline_mod,
        "_record_shared_prep_session",
        lambda *args, **kwargs: calls.append(("marker", kwargs.get("stage"), kwargs.get("status"))),
    )
    # Default: no same-day intraday taint so stage=slow/all tests proceed.
    # Individual tests can override this to exercise the abort path.
    monkeypatch.setattr(
        run_pipeline_mod,
        "_same_day_quote_rows_present",
        lambda *args, **kwargs: (False, {"count": 0}),
    )
    # Default: upstream daily feed fresh. Tests can override for stale abort.
    monkeypatch.setattr(
        run_pipeline_mod,
        "_upstream_market_freshness",
        lambda *args, **kwargs: (True, {"age_days": 0}),
    )

    class _FakeMarketSyncResult:
        inserted_rows = 10
        attempted_tickers = 10
        failed_tickers: list = []

    def _fake_market_service_factory(**kwargs):
        class _S:
            def sync_market_features(self_inner):
                calls.append(("daily_sync", None))
                return _FakeMarketSyncResult()

        return _S()

    monkeypatch.setattr(cli, "MarketDataSyncService", _fake_market_service_factory)
    monkeypatch.setattr(
        cli,
        "nasdaq_window",
        lambda now=None: SimpleNamespace(
            now_local=SimpleNamespace(weekday=lambda: 4),
            trading_date=date(2026, 3, 13),
        ),
    )
    monkeypatch.setattr(cli, "is_nasdaq_holiday", lambda d: False)


def test_cmd_run_shared_prep_slow_runs_only_ml_and_skips_dispatch(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="slow"
    )

    stages = [c[0] for c in calls]
    assert "sync" not in stages, "slow stage must skip sync-market"
    assert "forecast" in stages
    assert "fundamentals" in stages
    assert "ranker" in stages
    assert "dispatch" not in stages, "slow stage must not dispatch downstream agent"
    assert any(
        c[0] == "marker" and c[1] == "slow" and c[2] == "ok" for c in calls
    ), "slow stage must record an ok session marker"


def test_cmd_run_shared_prep_fast_runs_only_sync_and_dispatches(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="fast"
    )

    stages = [c[0] for c in calls]
    assert "sync" in stages
    assert "forecast" not in stages, "fast stage must skip ML forecast"
    assert "fundamentals" not in stages
    assert "ranker" not in stages
    assert ("dispatch", "agent-us") in calls
    assert any(
        c[0] == "marker" and c[1] == "fast" and c[2] == "ok" for c in calls
    ), "fast stage must record an ok session marker after dispatch"


def test_cmd_run_shared_prep_fast_aborts_when_artifacts_stale(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(
        monkeypatch,
        settings,
        _Repo(),
        calls,
        session_ready=(False, {"reason": "no_session", "market": "us"}),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="agent-us", stage="fast"
        )

    assert exc_info.value.code == 3
    stages = [c[0] for c in calls]
    assert "sync" not in stages, "fast gate must abort BEFORE sync when session is not ready"
    assert "dispatch" not in stages, "fast stage must not dispatch when slow session is not ready"
    assert not any(c[0] == "marker" for c in calls), "no marker on aborted fast run"


def test_cmd_run_shared_prep_fast_aborts_when_quote_sync_zero_rows(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(
        monkeypatch,
        settings,
        _Repo(),
        calls,
        sync_result=SimpleNamespace(inserted_rows=0, attempted_tickers=405, failed_tickers=["AAPL"]),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="agent-us", stage="fast"
        )

    assert exc_info.value.code == 7
    stages = [c[0] for c in calls]
    assert "sync" in stages, "fast stage must attempt quote sync before zero-row abort"
    assert "dispatch" not in stages, "fast stage must not dispatch when quote sync wrote zero rows"
    assert not any(c[0] == "marker" for c in calls), "no fast marker on zero-row abort"


def test_cmd_run_shared_prep_slow_records_non_ok_when_forecast_empty(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    # Simulate build_and_store_stacked_forecasts writing zero rows (e.g., all
    # upstream data missing). Ranker still returns 'ok' with scores, but the
    # combined readiness must be not-ok because forecast was empty.
    _stub_shared_prep_environment(
        monkeypatch, settings, _Repo(), calls,
        forecast_rows=0, ranker_scores=5, ranker_status="ok",
    )

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="slow"
    )

    marker_entries = [c for c in calls if c[0] == "marker"]
    assert marker_entries, "slow stage must always record a marker"
    # status must NOT be 'ok' because forecast_rows_written == 0
    assert all(c[2] != "ok" for c in marker_entries), (
        f"forecast=0 must downgrade slow marker status; got {marker_entries}"
    )


def test_cmd_run_shared_prep_fast_without_dispatch_still_runs_gate(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    # Same-session marker missing; the fast gate must fire even though
    # dispatch_job is empty (manual/operator invocation path).
    _stub_shared_prep_environment(
        monkeypatch, settings, _Repo(), calls,
        session_ready=(False, {"reason": "no_session", "market": "us"}),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="", stage="fast"
        )

    assert exc_info.value.code == 3
    stages = [c[0] for c in calls]
    assert "sync" not in stages, "fast gate must abort BEFORE sync even without dispatch_job"
    assert "dispatch" not in stages
    assert not any(c[0] == "marker" for c in calls)


def test_shared_prep_session_ready_accepts_all_marker(monkeypatch) -> None:
    """_shared_prep_session_ready must treat a 'stage=all' marker as valid so
    legacy single-shot runs can hand off to a subsequent fast invocation.
    """
    from datetime import datetime as _dt, timezone as _tz

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    class _Repo:
        def __init__(self) -> None:
            self.queried_stages: list[str] = []

        def get_latest_shared_prep_session(self, *, market, trading_date, stage):
            self.queried_stages.append(stage)
            if stage == "all":
                return {
                    "session_id": "sp_all_ok",
                    "market": market,
                    "trading_date": trading_date,
                    "stage": "all",
                    "status": "ok",
                    "forecast_rows_written": 100,
                    "ranker_scores_written": 25,
                    "created_at": _dt(2026, 3, 13, 5, 0, tzinfo=_tz.utc),
                }
            return None  # no slow marker

    repo = _Repo()
    ready, info = run_pipeline_mod._shared_prep_session_ready(
        repo, market="us", trading_date="2026-03-13"
    )

    assert ready is True, info
    assert info["session_id"] == "sp_all_ok"
    assert info["matched_stage"] == "all"
    assert set(repo.queried_stages) == {"slow", "all"}


def test_shared_prep_session_ready_prefers_newer_of_slow_and_all(monkeypatch) -> None:
    """When both markers exist, the most recent created_at wins."""
    from datetime import datetime as _dt, timezone as _tz

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    slow_row = {
        "session_id": "sp_slow_old",
        "stage": "slow",
        "status": "ok",
        "forecast_rows_written": 50,
        "ranker_scores_written": 10,
        "created_at": _dt(2026, 3, 13, 1, 0, tzinfo=_tz.utc),
    }
    all_row = {
        "session_id": "sp_all_new",
        "stage": "all",
        "status": "ok",
        "forecast_rows_written": 60,
        "ranker_scores_written": 20,
        "created_at": _dt(2026, 3, 13, 4, 30, tzinfo=_tz.utc),
    }

    class _Repo:
        def get_latest_shared_prep_session(self, *, market, trading_date, stage):
            return slow_row if stage == "slow" else all_row

    ready, info = run_pipeline_mod._shared_prep_session_ready(
        _Repo(), market="us", trading_date="2026-03-13"
    )

    assert ready is True
    assert info["session_id"] == "sp_all_new"
    assert info["matched_stage"] == "all"


def test_cmd_run_shared_prep_slow_aborts_on_same_day_intraday_quote(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)
    # Override taint check to report an intraday quote row already present.
    monkeypatch.setattr(
        run_pipeline_mod,
        "_same_day_quote_rows_present",
        lambda *args, **kwargs: (True, {"count": 1, "market": "us"}),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="", stage="slow"
        )

    assert exc_info.value.code == 4
    stages = [c[0] for c in calls]
    # ML must not run, no marker must be recorded.
    assert "forecast" not in stages, "slow must not run forecast when intraday quote is present"
    assert "ranker" not in stages
    assert not any(c[0] == "marker" for c in calls)


def test_canonical_market_key_normalizes_us_aliases() -> None:
    from arena.cli_commands.run_pipeline import _canonical_market_key

    assert _canonical_market_key("us") == "us"
    assert _canonical_market_key("NASDAQ") == "us"
    assert _canonical_market_key("nyse") == "us"
    assert _canonical_market_key("AMEX") == "us"
    assert _canonical_market_key("kospi") == "kospi"
    assert _canonical_market_key("kosdaq") == "kospi"
    assert _canonical_market_key("KR") == "kospi"
    assert _canonical_market_key("us,kospi") == "us", "comma-separated picks first"
    assert _canonical_market_key("") == ""
    assert _canonical_market_key("unknown_venue") == "unknown_venue"


def test_trading_date_handles_us_aliases_without_utc_fallback(monkeypatch) -> None:
    """_trading_date_for_market must route nasdaq/nyse/amex to America/New_York.

    Previously the raw key path would silently fall back to UTC, so a late-evening
    US session could compute a different civil date and store/look up markers
    under the wrong day.
    """
    from datetime import date as _date
    from arena.cli_commands.run_pipeline import _trading_date_for_market

    results = {
        "us": _trading_date_for_market("us"),
        "nasdaq": _trading_date_for_market("nasdaq"),
        "nyse": _trading_date_for_market("nyse"),
        "amex": _trading_date_for_market("amex"),
        "kospi": _trading_date_for_market("kospi"),
        "kosdaq": _trading_date_for_market("kosdaq"),
    }

    for key, value in results.items():
        assert isinstance(value, _date), f"{key} must return a date, got {value!r}"
    # All US aliases must agree on the same civil date (same TZ).
    us_dates = {results[k] for k in ("us", "nasdaq", "nyse", "amex")}
    assert len(us_dates) == 1, f"US aliases diverged: {results}"
    kr_dates = {results[k] for k in ("kospi", "kosdaq")}
    assert len(kr_dates) == 1, f"KR aliases diverged: {results}"


def test_cmd_run_shared_prep_all_live_path_not_blocked_by_self_sync_quotes(monkeypatch) -> None:
    """stage='all' runs sync BEFORE ML on purpose. The taint guard must not
    misinterpret quote rows that this very invocation just wrote as external
    contamination; otherwise the legacy single-shot/live deploy path breaks.
    """
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)
    # Simulate the realistic condition: _batch_market_sync wrote today's
    # quote rows. A naive guard would treat these as taint and SystemExit(4).
    monkeypatch.setattr(
        run_pipeline_mod,
        "_same_day_quote_rows_present",
        lambda *args, **kwargs: (True, {"count": 500, "market": "us"}),
    )

    # No SystemExit expected — stage='all' legacy flow must complete.
    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="all"
    )

    stages = [c[0] for c in calls]
    assert "sync" in stages
    assert "forecast" in stages, "stage='all' must run ML even when same-day quotes exist"
    assert "ranker" in stages
    assert ("dispatch", "agent-us") in calls


def test_cmd_run_shared_prep_rejects_multi_market_config(monkeypatch) -> None:
    """Shared-prep is single-market: mixed KIS_TARGET_MARKET (e.g., 'us,kospi')
    must be rejected so readiness markers and taint checks cannot silently
    ignore one of the markets.
    """
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us,kospi"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="", dispatch_job="agent-x", stage="all"
        )

    assert exc_info.value.code == 5
    stages = [c[0] for c in calls]
    assert "sync" not in stages, "multi-market reject must happen before any sync"
    assert "forecast" not in stages
    assert "ranker" not in stages
    assert "dispatch" not in stages


def test_shared_prep_session_ready_isolates_us_subexchanges(monkeypatch) -> None:
    """Exchange-level isolation: a 'nasdaq' prep marker must NOT satisfy a
    'nyse' readiness check (and vice versa), because forecast/ranker prep
    scopes itself by the raw KIS_TARGET_MARKET token.
    """
    from datetime import datetime as _dt, timezone as _tz
    from arena.cli_commands import run_pipeline as run_pipeline_mod

    nasdaq_row = {
        "session_id": "sp_nasdaq",
        "market": "nasdaq",
        "stage": "slow",
        "status": "ok",
        "forecast_rows_written": 40,
        "ranker_scores_written": 10,
        "created_at": _dt(2026, 3, 13, 5, 0, tzinfo=_tz.utc),
    }

    class _Repo:
        def __init__(self) -> None:
            self.queries: list[dict[str, object]] = []

        def get_latest_shared_prep_session(self, *, market, trading_date, stage):
            self.queries.append({"market": market, "stage": stage})
            # Only return the nasdaq marker when queried for nasdaq.
            if market == "nasdaq":
                return nasdaq_row
            return None

    repo = _Repo()

    ready_nasdaq, info_nasdaq = run_pipeline_mod._shared_prep_session_ready(
        repo, market="nasdaq", trading_date="2026-03-13"
    )
    assert ready_nasdaq is True
    assert info_nasdaq["session_id"] == "sp_nasdaq"

    ready_nyse, info_nyse = run_pipeline_mod._shared_prep_session_ready(
        repo, market="nyse", trading_date="2026-03-13"
    )
    assert ready_nyse is False
    assert info_nyse["reason"] == "no_session"

    # Queries must have been market-scoped, not canonicalized to 'us'.
    queried_markets = {q["market"] for q in repo.queries}
    assert queried_markets == {"nasdaq", "nyse"}


def test_cmd_run_shared_prep_all_refuses_dispatch_on_partial_status(monkeypatch) -> None:
    """stage='all' must fail-closed when its own prep status is not 'ok'.

    The previous flow recorded a 'partial' marker and then dispatched the
    agent anyway — a known-bad prep would still launch live trading.
    """
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    # forecast_rows=0 downgrades status to 'partial' even though ranker
    # claims ok. stage='all' must refuse to dispatch in that case.
    _stub_shared_prep_environment(
        monkeypatch, settings, _Repo(), calls,
        forecast_rows=0, ranker_scores=5, ranker_status="ok",
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="agent-us", stage="all"
        )

    assert exc_info.value.code == 6
    # Marker should have been recorded (so operators see the failure reason),
    # but dispatch must NOT have happened.
    marker_entries = [c for c in calls if c[0] == "marker"]
    assert any(c[2] != "ok" for c in marker_entries), (
        f"expected a non-ok marker; got {marker_entries}"
    )
    assert "dispatch" not in [c[0] for c in calls]


def test_cmd_run_shared_prep_slow_runs_daily_sync_then_ml(monkeypatch) -> None:
    """Slow stage must trigger MarketDataSyncService.sync_market_features()
    BEFORE the ML steps so the daily EOD feed is refreshed — the live
    scheduler phases otherwise never populate daily rows.
    """
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="", stage="slow"
    )

    stages = [c[0] for c in calls]
    # Daily sync must run before forecast/ranker.
    assert "daily_sync" in stages
    daily_idx = stages.index("daily_sync")
    forecast_idx = stages.index("forecast")
    assert daily_idx < forecast_idx, (
        f"daily_sync must precede forecast: daily@{daily_idx} forecast@{forecast_idx}"
    )
    # Marker is ok because freshness default is fresh.
    assert any(c[0] == "marker" and c[2] == "ok" for c in calls)


def test_cmd_run_shared_prep_slow_aborts_when_upstream_stale(monkeypatch) -> None:
    """When daily EOD data is far behind (e.g., feed broken for weeks),
    refuse before training the ranker — otherwise it silently learns on
    stale prices and the fast gate cannot detect that.
    """
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    from arena.cli_commands import run_pipeline as run_pipeline_mod

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)
    monkeypatch.setattr(
        run_pipeline_mod,
        "_upstream_market_freshness",
        lambda *args, **kwargs: (False, {
            "reason": "stale_daily",
            "market": "us",
            "age_days": 27,
            "threshold_days": 5,
        }),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="", stage="slow"
        )

    assert exc_info.value.code == 7
    stages = [c[0] for c in calls]
    assert "daily_sync" in stages, "daily sync must have been attempted"
    assert "forecast" not in stages, "forecast must not run when upstream is stale"
    assert "ranker" not in stages
    assert not any(c[0] == "marker" for c in calls)


def test_cmd_run_shared_prep_slow_aborts_when_daily_sync_fails(monkeypatch) -> None:
    """If MarketDataSyncService raises, slow must abort before ML."""
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls)

    def _boom_factory(**kwargs):
        class _S:
            def sync_market_features(self_inner):
                calls.append(("daily_sync_attempted", None))
                raise RuntimeError("KIS api down")

        return _S()

    monkeypatch.setattr(cli, "MarketDataSyncService", _boom_factory)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_shared_prep(
            live=True, market_override="us", dispatch_job="", stage="slow"
        )

    assert exc_info.value.code == 8
    stages = [c[0] for c in calls]
    assert "daily_sync_attempted" in stages
    assert "forecast" not in stages
    assert "ranker" not in stages


def test_cmd_run_shared_prep_slow_refuses_ml_on_seed_phase(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    calls: list[tuple[str, object]] = []

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            calls.append(("dataset", None))

        def ensure_tables(self):
            calls.append(("tables", None))

    _stub_shared_prep_environment(monkeypatch, settings, _Repo(), calls, phase="seed")

    cli.cmd_run_shared_prep(
        live=True, market_override="us", dispatch_job="agent-us", stage="slow"
    )

    stages = [c[0] for c in calls]
    # ML must still be refused on seed, but the slow path MUST run a daily
    # EOD sync so the feed can bootstrap out of the seed state. Without this,
    # an empty/sparse deployment would deadlock — slow can never populate
    # daily rows and phase would stay 'seed' forever.
    assert "sync" not in stages, "no quote sync on seed"
    assert "daily_sync" in stages, "seed+slow must run daily EOD sync to bootstrap"
    assert "forecast" not in stages, "seed must still refuse ML"
    assert "fundamentals" not in stages
    assert "ranker" not in stages
    assert "dispatch" not in stages


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
