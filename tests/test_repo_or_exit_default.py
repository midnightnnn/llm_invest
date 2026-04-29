"""Regression-prevention gate for ``cli_runtime._repo_or_exit``.

When ``ARENA_MODE`` is unset (or 'gcp'), this single entry point must keep
producing a ``BigQueryRepository`` constructed from the same settings inputs
the legacy direct-construction path used.  Any future PR that breaks one of
these tests is breaking the production GCP execution path.
"""

from __future__ import annotations

import pytest

from arena.cli_runtime import _repo_or_exit
from arena.config import load_settings
from arena.data import bq as bq_module


class _FakeBQRepo:
    def __init__(self, *, project, dataset, location, tenant_id=None):
        self.project = project
        self.dataset = dataset
        self.location = location
        self.tenant_id = tenant_id


def test_repo_or_exit_default_returns_bigquery_repo(monkeypatch):
    monkeypatch.delenv("ARENA_MODE", raising=False)
    monkeypatch.setattr(bq_module, "BigQueryRepository", _FakeBQRepo)
    settings = load_settings()
    settings.google_cloud_project = "p"
    settings.bq_dataset = "d"
    settings.bq_location = "asia-northeast3"

    repo = _repo_or_exit(settings, tenant_id="t1")

    assert isinstance(repo, _FakeBQRepo)
    assert repo.project == "p"
    assert repo.dataset == "d"
    assert repo.location == "asia-northeast3"
    assert repo.tenant_id == "t1"


def test_repo_or_exit_explicit_gcp_mode_returns_bigquery_repo(monkeypatch):
    monkeypatch.setenv("ARENA_MODE", "gcp")
    monkeypatch.setattr(bq_module, "BigQueryRepository", _FakeBQRepo)
    settings = load_settings()
    settings.google_cloud_project = "p"
    settings.bq_dataset = "d"
    settings.bq_location = "asia-northeast3"

    repo = _repo_or_exit(settings, tenant_id="t1")
    assert isinstance(repo, _FakeBQRepo)


def test_repo_or_exit_normalises_tenant_when_omitted(monkeypatch):
    monkeypatch.delenv("ARENA_MODE", raising=False)
    monkeypatch.delenv("ARENA_TENANT_ID", raising=False)
    monkeypatch.setattr(bq_module, "BigQueryRepository", _FakeBQRepo)
    settings = load_settings()
    settings.google_cloud_project = "p"
    settings.bq_dataset = "d"
    settings.bq_location = "asia-northeast3"

    repo = _repo_or_exit(settings)
    assert repo.tenant_id == "local"


def test_repo_or_exit_local_mode_returns_local_repository(monkeypatch, tmp_path):
    """ARENA_MODE=local must dispatch through factory to LocalRepository."""
    pytest.importorskip("duckdb")
    monkeypatch.setenv("ARENA_MODE", "local")
    monkeypatch.setenv("ARENA_LOCAL_DB_PATH", str(tmp_path / "arena.duckdb"))
    settings = load_settings()
    settings.google_cloud_project = "p"
    settings.bq_dataset = "d"
    settings.bq_location = "asia-northeast3"

    from arena.data.local.repository import LocalRepository

    repo = _repo_or_exit(settings, tenant_id="t1")
    assert isinstance(repo, LocalRepository)
    assert repo.tenant_id == "t1"
