from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace

import arena.cli as cli
import pytest
from arena.config import load_settings

from tests.cli.helpers import _FakeRepo, _stub_shared_prep_environment

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


def test_apply_tenant_runtime_credentials_clears_base_kis_values(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TENANT_ID", "tenant-a")
    settings = load_settings()
    settings.kis_api_key = "server-kis-key"
    settings.kis_api_secret = "server-kis-secret"
    settings.kis_paper_api_key = "server-paper-key"
    settings.kis_paper_api_secret = "server-paper-secret"
    settings.kis_account_no = "1234567801"
    settings.kis_secret_name = "KISAPI"

    repo = _FakeRepo(
        row={
            "tenant_id": "tenant-a",
            "kis_secret_name": "local-tenant-a-kis",
            "model_secret_name": "",
            "kis_env": "real",
        }
    )

    out = cli._apply_tenant_runtime_credentials(settings, repo)

    assert out is not None
    assert settings.kis_secret_name == "local-tenant-a-kis"
    assert settings.kis_api_key == ""
    assert settings.kis_api_secret == ""
    assert settings.kis_paper_api_key == ""
    assert settings.kis_paper_api_secret == ""
    assert settings.kis_account_no == ""


def test_load_secret_json_reads_local_credential_file(monkeypatch, tmp_path) -> None:
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(
        json.dumps(
            {
                "local-tenant-a-models": {
                    "openai_api_key": "tenant-openai",
                    "gemini_api_key": "tenant-gemini",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARENA_LOCAL_CREDENTIALS_FILE", str(credentials_path))

    out = cli._load_secret_json(project="", secret_name="local-tenant-a-models")

    assert out == {
        "openai_api_key": "tenant-openai",
        "gemini_api_key": "tenant-gemini",
    }


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
        return {"tenant_id": tenant_id, "kis_secret_name": "kis-midnightnnn"}

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


def test_prepare_kis_command_repo_rejects_missing_runtime_credentials(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_secret_name = "KISAPI"
    settings.kis_api_key = "server-key"
    settings.kis_api_secret = "server-secret"

    class _Repo(_FakeRepo):
        def ensure_dataset(self):
            pass

        def ensure_tables(self):
            pass

    repo = _Repo(row=None)

    monkeypatch.setenv("ARENA_TENANT_ID", "tenant-a")
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)

    with pytest.raises(RuntimeError, match="tenant runtime credentials missing: tenant=tenant-a"):
        cli._prepare_kis_command_repo(settings)

    assert settings.kis_secret_name == ""
    assert settings.kis_api_key == ""
    assert settings.kis_api_secret == ""


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
