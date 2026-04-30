"""Regression-prevention gate: pyproject.toml core dependencies stay GCP-shaped.

Local-backend packages (duckdb, chromadb, sentence-transformers, filelock)
must remain opt-in extras and never sneak into the default install. Existing
GCP packages must remain unconditional.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject_data() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fp:
        return tomllib.load(fp)


def _normalise_pkg(spec: str) -> str:
    """Returns lower-cased package name with no version specifier or extras marker."""
    return re.split(r"[<>=~ \[]", spec.strip())[0].strip().lower()


@pytest.fixture(scope="module")
def default_deps() -> list[str]:
    data = _read_pyproject_data()
    raw_deps = data.get("project", {}).get("dependencies", [])
    assert raw_deps, "pyproject.toml: project.dependencies must not be empty"
    return [_normalise_pkg(d) for d in raw_deps]


@pytest.fixture(scope="module")
def optional_deps_groups() -> dict:
    data = _read_pyproject_data()
    return data.get("project", {}).get("optional-dependencies", {})


def test_default_deps_keep_gcp_packages(default_deps):
    required = {"google-cloud-bigquery", "google-cloud-firestore", "google-cloud-secret-manager"}
    missing = required - set(default_deps)
    assert not missing, (
        f"GCP dependencies removed from default install: {sorted(missing)}. "
        "Local-mode work must keep production GCP path intact."
    )


def test_default_deps_have_no_local_only_packages(default_deps):
    forbidden_prefixes = ("duckdb", "chromadb", "sentence-transformers")
    leaked = [
        d for d in default_deps
        if any(d.startswith(p) for p in forbidden_prefixes)
    ]
    assert not leaked, (
        f"Local-only packages must remain optional extras, but leaked into default deps: {leaked}"
    )


def test_local_extras_group_exists_with_duckdb(optional_deps_groups):
    """The 'local' extras group must exist and pin DuckDB for the local backend."""
    assert "local" in optional_deps_groups, (
        "pyproject.toml must define [project.optional-dependencies].local"
    )
    local = [_normalise_pkg(d) for d in optional_deps_groups["local"]]
    assert "duckdb" in local, "local extras must include duckdb"


def test_setuptools_package_discovery_ignores_local_runtime_dirs():
    data = _read_pyproject_data()
    find = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {})

    assert "arena*" in find.get("include", [])
    assert "data*" in find.get("exclude", [])
    assert "logs*" in find.get("exclude", [])
