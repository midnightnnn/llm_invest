from __future__ import annotations

import pytest

import arena.cli as cli
from arena.config import load_settings
from tests.cli.helpers import _FakeRepo


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
