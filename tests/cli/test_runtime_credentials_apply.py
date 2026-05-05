from __future__ import annotations

import json

import arena.cli as cli
from arena.config import load_settings
from tests.cli.helpers import _FakeRepo


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
