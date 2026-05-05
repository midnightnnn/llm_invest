from __future__ import annotations

import arena.cli as cli
from arena.config import load_settings


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
