"""Storage backend factory.

Selects between the BigQuery-backed default repository (GCP mode) and a
DuckDB-backed local repository.  Local mode is opt-in via ``ARENA_MODE=local``
or ``settings.arena_mode='local'``.

Default behaviour is unchanged: when ``ARENA_MODE`` is unset the factory
constructs ``BigQueryRepository`` with the same inputs the previous direct
construction in ``cli_runtime._repo_or_exit`` used, so production GCP
execution stays on the legacy backend.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arena.config import Settings


logger = logging.getLogger(__name__)


_VALID_MODES = frozenset({"gcp", "local"})


def normalize_arena_mode(value: str | None) -> str:
    """Returns one of ``{'gcp', 'local'}``; falls back to ``'gcp'`` on unknown input."""
    token = str(value or "").strip().lower()
    return token if token in _VALID_MODES else "gcp"


def resolve_arena_mode(settings: "Settings | None" = None) -> str:
    """Resolves the active arena mode preferring settings, then env, then default."""
    candidate: str | None = None
    if settings is not None:
        candidate = getattr(settings, "arena_mode", None)
    if not candidate:
        candidate = os.getenv("ARENA_MODE")
    return normalize_arena_mode(candidate)


def get_repository(settings: "Settings", *, tenant_id: str | None = None) -> Any:
    """Constructs the storage repository for the active arena mode.

    Default ('gcp'):
        Returns ``BigQueryRepository`` — same construction inputs the legacy
        ``cli_runtime._repo_or_exit`` used.

    'local':
        Returns ``LocalRepository`` (DuckDB-backed).  Construction is lazy —
        duckdb itself is only imported the first time the session connects,
        so installing the ``[local]`` extra is what unlocks runtime queries.
    """
    mode = resolve_arena_mode(settings)
    tenant = str(tenant_id or os.getenv("ARENA_TENANT_ID") or "local").strip().lower() or "local"

    if mode == "local":
        from arena.data.local.repository import LocalRepository

        return LocalRepository(tenant_id=tenant, settings=settings)

    from arena.data.bq import BigQueryRepository

    return BigQueryRepository(
        project=settings.google_cloud_project,
        dataset=settings.bq_dataset,
        location=settings.bq_location,
        tenant_id=tenant,
    )
