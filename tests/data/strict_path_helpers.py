from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from arena.data.bigquery.ledger_store import LedgerStore
from arena.data.bigquery.market_store import MarketStore
from arena.data.bigquery.sleeve_store import SleeveStore
from arena.models import AccountSnapshot, Position


# ---------------------------------------------------------------------------
# Shared fake infrastructure
# ---------------------------------------------------------------------------

class _InsertClient:
    def __init__(self):
        self.payloads: list[dict[str, object]] = []
        self.calls: list[tuple[str, list[dict[str, object]]]] = []

    def insert_rows_json(self, table_id: str, rows: list[dict[str, object]], row_ids=None):
        _ = row_ids
        _ = table_id
        self.calls.append((table_id, list(rows)))
        self.payloads.extend(rows)
        return []


class _FakeSession:
    """Minimal stand-in for BigQuerySession."""

    def __init__(
        self,
        *,
        responses: list[object] | None = None,
        client: object | None = None,
        project: str = "proj",
        dataset: str = "ds",
        tenant_id: str = "local",
    ):
        self.project = project
        self.dataset = dataset
        self.dataset_fqn = f"{project}.{dataset}"
        self._tenant_id = tenant_id
        self._responses = list(responses or [])
        self.calls: list[str] = []
        self.call_pairs: list[tuple[str, dict | None]] = []
        self.client = client or _InsertClient()

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        token = str(tenant_id or self._tenant_id or "").strip().lower()
        return token or "local"

    def fetch_rows(self, sql: str, params=None):
        self.calls.append(sql)
        self.call_pairs.append((sql, params))
        if not self._responses:
            raise AssertionError("unexpected fetch_rows call")
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def execute(self, sql: str, params=None):
        self.calls.append(sql)
        self.call_pairs.append((sql, params))


# ---------------------------------------------------------------------------
# Forecast query helpers (MarketStore)
# ---------------------------------------------------------------------------


class _ForecastSchemaField:
    def __init__(self, name: str):
        self.name = name


class _ForecastSchemaClient:
    def __init__(self, columns: list[str]):
        self._columns = list(columns)

    def get_table(self, table_id: str):
        _ = table_id

        class _T:
            schema = []

        table = _T()
        table.schema = [_ForecastSchemaField(name) for name in self._columns]
        return table


# ---------------------------------------------------------------------------
# Market write helpers
# ---------------------------------------------------------------------------


class _LoadJob:
    def result(self):
        return None


class _MarketWriteClient:
    def __init__(self):
        self.loads: list[tuple[str, list[dict[str, object]]]] = []

    class _DatasetRef:
        def __init__(self, dataset: str):
            self._dataset = dataset

        def table(self, table_name: str) -> str:
            return f"proj.{self._dataset}.{table_name}"

    def get_table(self, table_id: str):
        class _T:
            schema = []

        _ = table_id
        return _T()

    def dataset(self, dataset: str):
        return self._DatasetRef(dataset)

    def load_table_from_file(self, file_obj, table_id: str, job_config=None):
        _ = job_config
        raw = file_obj.read().decode("utf-8").strip()
        rows = [json.loads(line) for line in raw.splitlines()] if raw else []
        self.loads.append((table_id, rows))
        return _LoadJob()


# ---------------------------------------------------------------------------
# Factory helpers — build stores with fake sessions
# ---------------------------------------------------------------------------


def _make_market_store(responses: list[object]) -> MarketStore:
    session = _FakeSession(responses=responses)
    return MarketStore(session)


def _make_forecast_query_store(rows: list[dict], *, columns: list[str]) -> MarketStore:
    session = _FakeSession(
        responses=[list(rows)],
        client=_ForecastSchemaClient(columns),
    )
    return MarketStore(session)


def _make_market_write_store() -> MarketStore:
    client = _MarketWriteClient()
    session = _FakeSession(client=client)
    store = MarketStore(session)
    return store


class _SleeveStoreForBuild(SleeveStore):
    """Used by tests that exercise build_agent_sleeve_snapshot with controlled
    latest_agent_sleeves / latest_close_prices / get_dividend_credits.
    """

    def __init__(self, session, *, fill_result, init_positions_json="[]", ledger=None, market=None):
        super().__init__(session, ledger=ledger, market=market)
        self._fill_result = fill_result
        self._init_positions_json = init_positions_json
        self._latest_close_prices_calls = 0

    def latest_agent_sleeves(self, *, agent_ids, tenant_id=None):
        _ = tenant_id
        agent_id = str(agent_ids[0])
        return {
            agent_id: {
                "initial_cash_krw": 1_000_000.0,
                "initial_positions_json": self._init_positions_json,
                "initialized_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
            }
        }


class _SleeveStoreForInit(SleeveStore):
    """Used by tests that exercise ensure_agent_sleeves / write_account_snapshot."""

    def __init__(self, session, *, snapshot, ledger=None, market=None):
        super().__init__(session, ledger=ledger, market=market)
        self._snapshot = snapshot

    def latest_agent_sleeves(self, *, agent_ids, tenant_id=None):
        _ = (agent_ids, tenant_id)
        return {}

    def latest_account_snapshot(self, *, tenant_id=None):
        _ = tenant_id
        return self._snapshot


class _SleeveStoreForRetarget(SleeveStore):
    """Used by retarget_agent_sleeves_preserve_positions tests."""

    def __init__(self, session, *, snapshots):
        super().__init__(session)
        self._snapshots = snapshots

    def build_agent_sleeve_snapshot(
        self,
        *,
        agent_id,
        sources=None,
        include_simulated=True,
        tenant_id=None,
        as_of_ts=None,
    ):
        _ = (sources, include_simulated, tenant_id, as_of_ts)
        snapshot = self._snapshots.get(
            str(agent_id),
            AccountSnapshot(cash_krw=0.0, total_equity_krw=0.0, positions={}),
        )
        return snapshot, float(snapshot.total_equity_krw), {}


# ---------------------------------------------------------------------------
# LedgerStore subclasses for tests that need ledger behaviours
# ---------------------------------------------------------------------------


class _LedgerStoreForCapitalReplay(LedgerStore):
    """Used by tests that exercise build_agent_sleeve_snapshot with checkpoint
    + capital event replays."""

    def __init__(
        self,
        session,
        *,
        checkpoint: dict[str, object] | None = None,
        capital_events: list[dict[str, object]] | None = None,
        manual_position_adjustments: list[dict[str, object]] | None = None,
        manual_cash_adjustments: list[dict[str, object]] | None = None,
        transfer_events: list[dict[str, object]] | None = None,
    ):
        super().__init__(session)
        self._checkpoint = checkpoint or {}
        self._capital_events = list(capital_events or [])
        self._manual_position_adjustments = list(manual_position_adjustments or [])
        self._manual_cash_adjustments = list(manual_cash_adjustments or [])
        self._transfer_events = list(transfer_events or [])

    def latest_agent_state_checkpoints(self, *, agent_ids, tenant_id=None):
        _ = tenant_id
        if not self._checkpoint:
            return {}
        agent = str(agent_ids[0])
        return {agent: dict(self._checkpoint, agent_id=agent)}

    def capital_events_since(self, *, agent_id, since, tenant_id=None, event_types=None):
        _ = (agent_id, since, tenant_id, event_types)
        return list(self._capital_events)

    def manual_cash_adjustments_since(self, *, agent_id, since, tenant_id=None):
        _ = (agent_id, since, tenant_id)
        return list(self._manual_cash_adjustments)

    def manual_position_adjustments_since(self, *, since, tenant_id=None):
        _ = (since, tenant_id)
        return list(self._manual_position_adjustments)

    def agent_transfer_events_since(self, *, agent_id, since, tenant_id=None):
        _ = (agent_id, since, tenant_id)
        return list(self._transfer_events)


class _MarketStoreForReplay:
    """Minimal market stub that returns empty prices by default."""

    def latest_close_prices_with_currency(self, *, tickers, sources=None, as_of_date=None):
        _ = (tickers, sources, as_of_date)
        return {}


class _SleeveStoreForCapitalReplay(SleeveStore):
    """Used by tests that combine sleeve + ledger for capital replay."""

    def __init__(self, session, *, ledger, market=None):
        super().__init__(session, ledger=ledger, market=market)

    def latest_agent_sleeves(self, *, agent_ids, tenant_id=None):
        _ = (agent_ids, tenant_id)
        return {}

    def get_dividend_credits(self, *, agent_id, since, tenant_id=None):
        _ = (agent_id, since, tenant_id)
        return []


class _SleeveStoreForCapitalRetarget(SleeveStore):
    """Used by retarget_agent_capitals_preserve_positions tests."""

    def __init__(self, session, *, snapshots, ledger):
        super().__init__(session, ledger=ledger)
        self._snapshots = snapshots

    def build_agent_sleeve_snapshot(
        self,
        *,
        agent_id,
        sources=None,
        include_simulated=True,
        tenant_id=None,
        as_of_ts=None,
    ):
        _ = (sources, include_simulated, tenant_id, as_of_ts)
        snapshot = self._snapshots[str(agent_id)]
        return snapshot, float(snapshot.total_equity_krw), {"agent_id": str(agent_id)}


class _LedgerStoreForCapitalRetarget(LedgerStore):
    """Ledger that handles dedup check (existing_event_ids) returning empty + append."""

    def existing_event_ids(self, table_name, event_ids, *, tenant_id=None):
        _ = (table_name, event_ids, tenant_id)
        return set()


class _NavSleeveStore(SleeveStore):
    """Used by NAV tests that track executed SQL."""

    def __init__(self, session):
        super().__init__(session)
        self.executed: list[tuple[str, dict[str, object] | None]] = []
        self.rows: list[dict[str, object]] = []

    class _NavSession:
        """Session that defers to the store's tracking."""

        def __init__(self, store):
            self._store = store
            self.dataset_fqn = "proj.ds"
            self.project = "proj"
            self.dataset = "ds"
            self.client = _InsertClient()

        def resolve_tenant_id(self, tenant_id=None):
            token = str(tenant_id or "").strip().lower()
            return token or "local"

        def fetch_rows(self, sql, params=None):
            self._store.executed.append((sql, dict(params or {})))
            return list(self._store.rows)

        def execute(self, sql, params=None):
            self._store.executed.append((sql, dict(params or {})))

    @classmethod
    def create(cls):
        inst = cls.__new__(cls)
        inst.executed = []
        inst.rows = []
        nav_session = cls._NavSession(inst)
        SleeveStore.__init__(inst, nav_session)
        return inst


class _ActualBasisSleeveStore(SleeveStore):
    """Used by tests that override the lineage origin for actual-capital tracing."""

    def __init__(self, session, *, origin_state, nav_rows=None, ledger=None):
        super().__init__(session, ledger=ledger)
        self._origin_state = dict(origin_state)
        self._nav_rows = list(nav_rows or [])

    def _load_agent_origin_state(self, *, agent_id: str, tenant_id: str | None = None):
        _ = (agent_id, tenant_id)
        return dict(self._origin_state)

    def fetch_agent_nav_history(self, *, tenant_id=None, agent_id=None, agent_ids=None, limit=10000):
        _ = (tenant_id, agent_id, agent_ids, limit)
        return list(self._nav_rows)


# ===================================================================
# Tests — Market Store
# ===================================================================


def _make_capital_replay_store(
    *,
    checkpoint,
    capital_events,
    fills=None,
    manual_position_adjustments=None,
    manual_cash_adjustments=None,
    transfer_events=None,
):
    """Helper to build a SleeveStore + LedgerStore for capital replay tests."""
    # The sleeve session's fetch_rows will be called for execution_reports
    sleeve_session = _FakeSession(responses=[list(fills or [])])
    ledger_session = _FakeSession()
    ledger = _LedgerStoreForCapitalReplay(
        ledger_session,
        checkpoint=checkpoint,
        capital_events=capital_events,
        manual_position_adjustments=manual_position_adjustments,
        manual_cash_adjustments=manual_cash_adjustments,
        transfer_events=transfer_events,
    )
    market = _MarketStoreForReplay()
    store = _SleeveStoreForCapitalReplay(sleeve_session, ledger=ledger, market=market)
    return store
