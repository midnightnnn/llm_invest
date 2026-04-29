"""Tests for the ARENA_MODE selector and the storage backend factory.

These guard the contract that:
  * ``ARENA_MODE`` unset behaves identically to the historical GCP path.
  * Settings prefer their own ``arena_mode`` over the env var.
  * ``ARENA_MODE=local`` dispatches to the opt-in local backend, never
    silently falling back to BigQuery.
"""

from __future__ import annotations

import pytest

from arena.config import load_settings
from arena.data import bq as bq_module
from arena.data.factory import (
    get_repository,
    normalize_arena_mode,
    resolve_arena_mode,
)


class _FakeBQRepo:
    """Stand-in for BigQueryRepository so tests run without GCP auth."""

    def __init__(self, *, project: str, dataset: str, location: str, tenant_id: str | None = None):
        self.project = project
        self.dataset = dataset
        self.location = location
        self.tenant_id = tenant_id


# --- normalize_arena_mode -----------------------------------------------------


def test_normalize_returns_gcp_when_value_unknown_or_blank():
    assert normalize_arena_mode(None) == "gcp"
    assert normalize_arena_mode("") == "gcp"
    assert normalize_arena_mode("   ") == "gcp"
    assert normalize_arena_mode("garbage") == "gcp"
    assert normalize_arena_mode("LOCAL_PRO") == "gcp"


def test_normalize_accepts_known_modes_case_insensitively():
    assert normalize_arena_mode("local") == "local"
    assert normalize_arena_mode(" Local ") == "local"
    assert normalize_arena_mode("LOCAL") == "local"
    assert normalize_arena_mode("gcp") == "gcp"
    assert normalize_arena_mode("GCP") == "gcp"


# --- resolve_arena_mode -------------------------------------------------------


def test_resolve_prefers_settings_over_env(monkeypatch):
    monkeypatch.setenv("ARENA_MODE", "local")

    class _Stub:
        arena_mode = "gcp"

    assert resolve_arena_mode(_Stub()) == "gcp"


def test_resolve_falls_back_to_env_when_settings_unset(monkeypatch):
    monkeypatch.setenv("ARENA_MODE", "local")

    class _Stub:
        arena_mode = ""

    assert resolve_arena_mode(_Stub()) == "local"


def test_resolve_default_when_nothing_set(monkeypatch):
    monkeypatch.delenv("ARENA_MODE", raising=False)
    assert resolve_arena_mode(None) == "gcp"


# --- load_settings respects ARENA_MODE ---------------------------------------


def test_load_settings_default_mode_is_gcp(monkeypatch):
    monkeypatch.delenv("ARENA_MODE", raising=False)
    settings = load_settings()
    assert settings.arena_mode == "gcp"


def test_load_settings_explicit_local(monkeypatch):
    monkeypatch.setenv("ARENA_MODE", "local")
    settings = load_settings()
    assert settings.arena_mode == "local"


def test_load_settings_invalid_mode_falls_back_to_gcp(monkeypatch):
    monkeypatch.setenv("ARENA_MODE", "totallybroken")
    settings = load_settings()
    assert settings.arena_mode == "gcp"


# --- factory dispatch --------------------------------------------------------


def test_factory_dispatches_to_bigquery_in_gcp_mode(monkeypatch):
    monkeypatch.delenv("ARENA_MODE", raising=False)
    settings = load_settings()
    settings.google_cloud_project = "test-proj"
    settings.bq_dataset = "test_dataset"
    settings.bq_location = "asia-northeast3"
    monkeypatch.setattr(bq_module, "BigQueryRepository", _FakeBQRepo)

    repo = get_repository(settings, tenant_id="tenant-a")

    assert isinstance(repo, _FakeBQRepo)
    assert repo.project == "test-proj"
    assert repo.dataset == "test_dataset"
    assert repo.location == "asia-northeast3"
    assert repo.tenant_id == "tenant-a"


def test_factory_normalises_blank_tenant_to_local(monkeypatch):
    monkeypatch.delenv("ARENA_MODE", raising=False)
    monkeypatch.delenv("ARENA_TENANT_ID", raising=False)
    settings = load_settings()
    settings.google_cloud_project = "p"
    settings.bq_dataset = "d"
    settings.bq_location = "asia-northeast3"
    monkeypatch.setattr(bq_module, "BigQueryRepository", _FakeBQRepo)

    repo = get_repository(settings, tenant_id="")
    assert repo.tenant_id == "local"


def test_factory_local_mode_returns_local_repository(monkeypatch, tmp_path):
    """ARENA_MODE=local must construct LocalRepository (PR 3+)."""
    pytest.importorskip("duckdb")
    monkeypatch.setenv("ARENA_MODE", "local")
    monkeypatch.setenv("ARENA_LOCAL_DB_PATH", str(tmp_path / "arena.duckdb"))

    settings = load_settings()
    settings.google_cloud_project = "test-proj"

    from arena.data.local.repository import LocalRepository

    repo = get_repository(settings, tenant_id="tenant-a")
    assert isinstance(repo, LocalRepository)
    assert repo.tenant_id == "tenant-a"
