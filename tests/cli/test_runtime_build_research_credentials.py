from __future__ import annotations

from types import SimpleNamespace

import arena.cli as cli
from arena.config import load_settings
from tests.cli.helpers import _FakeRepo


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
