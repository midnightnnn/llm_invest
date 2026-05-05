from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace

import arena.cli as cli
import pytest
from arena.config import load_settings

from tests.cli.helpers import _FakeRepo, _stub_shared_prep_environment

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
