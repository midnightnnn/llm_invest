"""LocalRepository — DuckDB-backed equivalent of ``BigQueryRepository``.

Current scope: read path plus the minimal local write path required for paper
cycles (memory, order/execution, and sleeve snapshots). Optional surfaces that
are not implemented raise ``AttributeError`` from the ``__getattr__`` fallback
so existing ``hasattr(repo, "...")`` feature detection keeps working.

Wire-up: instantiated by ``arena/data/factory.py:get_repository`` when
``ARENA_MODE=local`` is active.  Construction signature mirrors what the
factory passes (``tenant_id=...``, ``settings=...``).
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import TYPE_CHECKING, Any

from arena.data.local.config_store import LocalConfigStore
from arena.data.local.execution_store import LocalExecutionStore
from arena.data.local.market_store import LocalMarketStore
from arena.data.local.memory_store import LocalMemoryStore
from arena.data.local.session import DuckDBSession, default_db_path
from arena.data.local.sleeve_store import LocalSleeveStore
from arena.models import utc_now

if TYPE_CHECKING:
    from arena.config import Settings


logger = logging.getLogger(__name__)


class LocalRepository:
    """Facade over DuckDB-backed domain stores.

    Mirrors the public surface of ``BigQueryRepository`` for local quickstart:
    ``ensure_dataset`` (no-op), ``ensure_tables`` (DuckDB DDL), and methods
    exposed by the underlying DuckDB-native stores.
    """

    _STORE_ATTRS = (
        "_market_store",
        "_memory_store",
        "_config_store",
        "_execution_store",
        "_sleeve_store",
    )

    def __init__(
        self,
        *,
        tenant_id: str | None = None,
        settings: "Settings | None" = None,
        db_path: str | None = None,
    ) -> None:
        self.settings = settings
        self._session = DuckDBSession(
            db_path if db_path is not None else default_db_path(),
            tenant_id=tenant_id,
        )
        self._market_store = LocalMarketStore(self._session)
        self._memory_store = LocalMemoryStore(self._session)
        self._config_store = LocalConfigStore(self._session)
        self._execution_store = LocalExecutionStore(self._session)
        self._sleeve_store = LocalSleeveStore(self._session, market=self._market_store)

    # ------------------------------------------------------------------
    # Session / infra delegates (parity with BigQueryRepository surface)
    # ------------------------------------------------------------------

    @property
    def session(self) -> DuckDBSession:
        return self._session

    @property
    def client(self):
        """Returns the underlying duckdb connection (parity with bq.client)."""
        return self._session.connect()

    @property
    def project(self) -> str:
        return "local"

    @property
    def dataset(self) -> str:
        return "main"

    @property
    def location(self) -> str:
        return "local"

    @property
    def tenant_id(self) -> str:
        return self._session.tenant_id

    @tenant_id.setter
    def tenant_id(self, value: str) -> None:
        self._session.set_tenant_id(value)

    @property
    def dataset_fqn(self) -> str:
        return self._session.dataset_fqn

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return self._session.resolve_tenant_id(tenant_id)

    def set_tenant_id(self, tenant_id: str | None) -> None:
        self._session.set_tenant_id(tenant_id)

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def ensure_dataset(self) -> None:
        """No-op — DuckDB has no separate dataset namespace, file is the boundary."""
        return None

    def ensure_tables(self) -> None:
        """Idempotently creates every arena table in the local DuckDB file."""
        self._session.ensure_tables()

    # ------------------------------------------------------------------
    # Generic execute / fetch (mirrors BigQueryRepository helpers)
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: Any = None) -> Any:
        return self._session.execute(sql, params)

    def fetch_rows(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        return self._session.fetch_rows(sql, params)

    # ------------------------------------------------------------------
    # Dummy methods for UI/Auth
    # ------------------------------------------------------------------

    def append_runtime_audit_log(
        self,
        *,
        action: str,
        status: str,
        user_email: str | None = None,
        tenant_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Appends one local audit log row using the BigQueryRepository signature."""
        self._session.insert_dict(
            "runtime_audit_logs",
            {
                "created_at": utc_now(),
                "user_email": str(user_email or "").strip().lower() or None,
                "tenant_id": str(tenant_id or self.tenant_id or "").strip().lower() or None,
                "action": str(action or "").strip() or "unknown",
                "status": str(status or "").strip().lower() or "unknown",
                "detail_json": json.dumps(detail or {}, ensure_ascii=False, default=str),
            },
        )

    def list_runtime_user_tenants(self, user_email: str) -> list[dict[str, str]]:
        """Mock access for local mode: always owner of the current tenant."""
        return [{"tenant_id": self.tenant_id or "local", "role": "owner"}]

    def ensure_runtime_user_tenant(self, user_email: str, tenant_id: str, role: str, created_by: str) -> None:
        """No-op for local mode."""
        pass

    def ensure_runtime_access_request_pending(self, user_email: str, user_name: str, google_sub: str) -> None:
        """No-op for local mode."""
        pass

    def all_tenant_run_statuses(self, limit: int = 100) -> list[dict[str, Any]]:
        """Mock access for local mode ops page."""
        return []

    def recent_runtime_audit_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Returns recent local runtime audit logs across tenants."""
        return self._session.fetch_rows(
            """
            SELECT created_at, user_email, tenant_id, action, status, detail_json
            FROM runtime_audit_logs
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            {"limit": max(1, min(int(limit), 500))},
        )

    def append_tenant_run_status(
        self,
        *,
        tenant_id: str,
        run_id: str,
        run_type: str,
        status: str,
        reason_code: str | None = None,
        stage: str | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        recorded_at: datetime | None = None,
        message: str | None = None,
        job_name: str | None = None,
        execution_name: str | None = None,
        log_uri: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Appends one tenant-scoped run status row to local DuckDB."""
        tenant = str(tenant_id or "").strip().lower() or self.tenant_id
        token = str(run_id or "").strip()
        if not token:
            raise ValueError("run_id is required")
        self._session.insert_dict(
            "tenant_run_statuses",
            {
                "tenant_id": tenant,
                "run_id": token,
                "recorded_at": recorded_at or utc_now(),
                "run_type": str(run_type or "").strip().lower() or "unknown",
                "status": str(status or "").strip().lower() or "unknown",
                "reason_code": str(reason_code or "").strip().lower() or None,
                "stage": str(stage or "").strip().lower() or None,
                "started_at": started_at,
                "finished_at": finished_at,
                "message": str(message or "").strip() or None,
                "job_name": str(job_name or "").strip() or None,
                "execution_name": str(execution_name or "").strip() or None,
                "log_uri": str(log_uri or "").strip() or None,
                "detail_json": json.dumps(detail or {}, ensure_ascii=False, default=str),
            },
        )

    # ------------------------------------------------------------------
    # Fallback — delegate to first store that implements the method
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only triggered when normal attribute lookup fails — the explicit
        # delegates above and the @property accessors are checked first.
        for attr in self._STORE_ATTRS:
            try:
                store = object.__getattribute__(self, attr)
            except AttributeError:
                continue
            if hasattr(store, name):
                return getattr(store, name)
        raise AttributeError(
            f"LocalRepository.{name}() is not implemented yet. "
            "The local backend intentionally implements only the quickstart "
            "storage surface so far."
        )
