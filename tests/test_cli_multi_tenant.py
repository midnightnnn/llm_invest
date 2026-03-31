from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import arena.cli as cli
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


def test_resolve_batch_tenants_falls_back_when_none_found(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_PUBLIC_DEMO_TENANT", raising=False)
    monkeypatch.delenv("ARENA_BATCH_TENANTS", raising=False)
    repo = _FakeRepo(tenants=[])
    assert cli._resolve_batch_tenants(repo, fallback="Tenant-Z") == ["tenant-z"]


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


def test_batch_phase_uses_quote_sources_for_live_us_probe(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    class _Repo:
        def __init__(self) -> None:
            self.sources: list[str] = []

        def market_source_distinct_tickers(self, *, source: str) -> int:
            self.sources.append(source)
            return 120 if source == "open_trading_us_quote" else 0

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
    assert repo.sources[0] == "open_trading_us_quote"


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

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "_apply_tenant_runtime_credentials", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_batch_phase", lambda *args, **kwargs: ("open_cycle", None))
    monkeypatch.setattr(cli, "_batch_market_sync", lambda *args, **kwargs: calls.append(("sync", None)))
    monkeypatch.setattr(cli, "cmd_build_forecasts", lambda args: calls.append(("forecast", args.horizon)))
    monkeypatch.setattr(cli, "_dispatch_agent_job", lambda settings, job_name: calls.append(("dispatch", job_name)))
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
    assert ("dispatch", "agent-us") in calls


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
