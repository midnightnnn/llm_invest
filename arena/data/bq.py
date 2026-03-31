"""BigQuery repository facade — compatibility wrapper over domain stores.

Existing code imports ``BigQueryRepository`` and calls methods on it directly.
This module preserves that interface while delegating all work to
:class:`BigQuerySession` and the domain-specific store classes under
``arena.data.bigquery.*``.

New code should prefer importing the individual stores or session directly.
"""

from __future__ import annotations

import logging
from typing import Any

from google.cloud import bigquery

from .bigquery.backtest_store import BacktestStore
from .bigquery.execution_store import ExecutionStore
from .bigquery.ledger_store import LedgerStore
from .bigquery.market_store import MarketStore
from .bigquery.memory_bq_store import MemoryBQStore
from .bigquery.runtime_store import RuntimeStore
from .bigquery.session import BigQuerySession
from .bigquery.sleeve_store import SleeveStore

logger = logging.getLogger(__name__)


class BigQueryRepository:
    """Backwards-compatible facade that owns a :class:`BigQuerySession` and
    domain store instances.

    All domain methods are resolved via ``__getattr__`` to the appropriate
    store.  The underlying stores are also accessible as attributes
    (e.g. ``repo.session``, ``repo._runtime_store``) for new code that wants
    to depend on a narrower interface.
    """

    def __init__(self, project: str, dataset: str, location: str, tenant_id: str | None = None):
        self._session = BigQuerySession(project, dataset, location, tenant_id)

        # Domain stores — new code can use these directly.
        self._ledger_store = LedgerStore(self._session)
        self._market_store = MarketStore(self._session)
        self._memory_bq_store = MemoryBQStore(self._session)
        self._runtime_store = RuntimeStore(self._session)
        self._backtest_store = BacktestStore(self._session)
        self._execution_store = ExecutionStore(self._session, memory_bq_store=self._memory_bq_store)
        self._sleeve_store = SleeveStore(
            self._session,
            ledger=self._ledger_store,
            market=self._market_store,
        )

    # ------------------------------------------------------------------
    # Session / infra delegates
    # ------------------------------------------------------------------

    @property
    def session(self) -> BigQuerySession:
        return self._session

    @property
    def project(self) -> str:
        return self._session.project

    @property
    def dataset(self) -> str:
        return self._session.dataset

    @property
    def location(self) -> str:
        return self._session.location

    @property
    def client(self) -> bigquery.Client:
        return self._session.client

    @property
    def tenant_id(self) -> str:
        return self._session.tenant_id

    @tenant_id.setter
    def tenant_id(self, value: str) -> None:
        self._session.tenant_id = value

    @property
    def dataset_fqn(self) -> str:
        return self._session.dataset_fqn

    @staticmethod
    def _normalize_tenant_id(value: str | None) -> str:
        return BigQuerySession._normalize_tenant_id(value)

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return self._session.resolve_tenant_id(tenant_id)

    def set_tenant_id(self, tenant_id: str | None) -> None:
        self._session.set_tenant_id(tenant_id)

    def ensure_dataset(self) -> None:
        self._session.ensure_dataset()

    def ensure_tables(self) -> None:
        self._session.ensure_tables()

    def _params(self, params: dict[str, Any] | None) -> list[bigquery.QueryParameter]:
        return self._session._params(params)

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> None:
        self._session.execute(sql, params)

    def fetch_rows(self, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self._session.fetch_rows(sql, params)

    # ------------------------------------------------------------------
    # Fallback — delegate any attribute to domain stores
    # ------------------------------------------------------------------

    _STORE_ATTRS = (
        "_memory_bq_store",
        "_runtime_store",
        "_market_store",
        "_ledger_store",
        "_execution_store",
        "_sleeve_store",
        "_backtest_store",
    )

    def __getattr__(self, name: str) -> Any:
        # Only triggered when normal attribute lookup fails (i.e. not on
        # mixin methods or explicit delegates above).
        for attr in self._STORE_ATTRS:
            try:
                store = object.__getattribute__(self, attr)
            except AttributeError:
                continue
            if hasattr(store, name):
                return getattr(store, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute {name!r}")
