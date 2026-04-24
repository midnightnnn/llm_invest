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


def test_upsert_instrument_master_persists_ticker_name() -> None:
    session = _FakeSession(client=_InsertClient())
    store = MarketStore(session)

    inserted = store.upsert_instrument_master(
        [
            {
                "instrument_id": "KRX:025860",
                "ticker": "025860",
                "ticker_name": "남해화학",
                "exchange_code": "KRX",
                "currency": "KRW",
                "lot_size": 1,
                "tick_size": 1.0,
                "tradable": True,
                "status": "ACTIVE",
            }
        ]
    )

    assert inserted == 1
    assert session.client.payloads[0]["ticker_name"] == "남해화학"


def test_ticker_name_map_falls_back_to_instrument_master() -> None:
    store = _make_market_store(
        responses=[
            [
                {"ticker": "025860", "ticker_name": None},
                {"ticker": "005930", "ticker_name": "삼성전자"},
            ],
            [
                {"ticker": "025860", "ticker_name": "남해화학"},
            ],
        ]
    )

    out = store.ticker_name_map(tickers=["025860", "005930"], limit=10)

    assert out == {"025860": "남해화학", "005930": "삼성전자"}


def test_rebuild_universe_candidates_skips_rows_without_daily_history_features() -> None:
    session = _FakeSession(
        responses=[
            [
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "MISSING",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:MISSING",
                    "ret_20d": None,
                    "ret_5d": None,
                    "volatility_20d": None,
                    "sentiment_score": 1.0,
                },
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "ZERO",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:ZERO",
                    "ret_20d": 0.0,
                    "ret_5d": 0.0,
                    "volatility_20d": 0.0,
                    "sentiment_score": 0.0,
                },
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "GOOD",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:GOOD",
                    "ret_20d": 0.1,
                    "ret_5d": 0.02,
                    "volatility_20d": 0.12,
                    "sentiment_score": 0.2,
                },
            ]
        ],
        client=_InsertClient(),
    )
    store = MarketStore(session)

    out = store.rebuild_universe_candidates(top_n=10, allowed_tickers=["MISSING", "ZERO", "GOOD"])

    assert out["count"] == 2
    written = {row["ticker"] for row in session.client.payloads}
    assert written == {"ZERO", "GOOD"}


def test_latest_universe_candidate_tickers_scopes_latest_run_by_market() -> None:
    session = _FakeSession(
        responses=[
            [
                {"ticker": "AAPL"},
                {"ticker": "MSFT"},
            ]
        ]
    )
    store = MarketStore(session)

    out = store.latest_universe_candidate_tickers(limit=10, markets=["nasdaq"])

    assert out == ["AAPL", "MSFT"]
    sql, params = session.call_pairs[0]
    assert "FROM scoped" in sql
    assert "IN UNNEST(@markets)" in sql
    assert params["markets"] == ["us"]
    assert params["limit"] == 10


# ---------------------------------------------------------------------------
# SleeveStore subclasses (override methods the tests customise)
# ---------------------------------------------------------------------------


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
        manual_cash_adjustments: list[dict[str, object]] | None = None,
        transfer_events: list[dict[str, object]] | None = None,
    ):
        super().__init__(session)
        self._checkpoint = checkpoint or {}
        self._capital_events = list(capital_events or [])
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


def test_latest_close_prices_propagates_latest_table_failure() -> None:
    store = _make_market_store([RuntimeError("bq down")])

    with pytest.raises(RuntimeError, match="bq down"):
        store.latest_close_prices(tickers=["AAPL"])

    assert len(store.session.calls) == 1
    assert "market_features_latest" in store.session.calls[0]


def test_latest_market_features_does_not_retry_legacy_on_empty_rows() -> None:
    store = _make_market_store([[]])

    rows = store.latest_market_features(tickers=["AAPL"], limit=5)

    assert rows == []
    assert len(store.session.calls) == 1
    assert "market_features_latest" in store.session.calls[0]
    assert "ret_5d IS NOT NULL AND ret_20d IS NOT NULL AND volatility_20d IS NOT NULL" in store.session.calls[0]


def test_latest_missing_daily_feature_tickers_queries_newest_incomplete_snapshots() -> None:
    store = _make_market_store(
        [
            [
                {
                    "as_of_ts": "2026-01-01T00:00:00+00:00",
                    "ticker": "MISS",
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:MISS",
                    "source": "open_trading_us_quote",
                }
            ]
        ]
    )

    rows = store.latest_missing_daily_feature_tickers(sources=["open_trading_us_quote"], limit=5)

    assert rows[0]["ticker"] == "MISS"
    sql, params = store.session.call_pairs[-1]
    assert "market_features_latest" in sql
    assert "ret_5d IS NULL OR ret_20d IS NULL OR volatility_20d IS NULL" in sql
    assert "has_complete_features = 0" in sql
    assert params["sources"] == ["open_trading_us_quote"]


def test_screen_latest_features_does_not_retry_legacy_on_empty_rows() -> None:
    store = _make_market_store([[]])

    rows = store.screen_latest_features(top_n=3)

    assert rows == []
    assert len(store.session.calls) == 1
    assert "market_features_latest" in store.session.calls[0]
    assert "ret_5d IS NOT NULL" in store.session.calls[0]
    assert "ret_20d IS NOT NULL" in store.session.calls[0]
    assert "volatility_20d IS NOT NULL" in store.session.calls[0]


def test_refresh_signal_daily_values_uses_point_in_time_sources() -> None:
    store = _make_market_store([])

    out = store.refresh_signal_daily_values(
        lookback_days=180,
        horizon_days=20,
        sources=["open_trading_us"],
        market="us",
    )

    assert out == 0
    sql, params = store.session.call_pairs[-1]
    assert "signal_daily_values" in sql
    assert "market_features" in sql
    assert "predicted_expected_returns" in sql
    assert "f.run_date <= c.as_of_date" in sql
    assert "fundamentals_derived_daily" in sql
    assert "d.latest_announcement_date <= w.as_of_date" in sql
    assert params["sources"] == ["open_trading_us"]
    assert params["market"] == "us"


def test_latest_opportunity_ranker_scores_reads_fresh_latest_batch() -> None:
    store = _make_market_store([[{"ticker": "AAPL", "recommendation_score": 0.2}]])

    rows = store.latest_opportunity_ranker_scores(
        tickers=["AAPL"],
        profiles=["aggressive"],
        buckets=["momentum"],
        per_profile_limit=2,
        limit=3,
        max_age_hours=12,
    )

    assert rows[0]["ticker"] == "AAPL"
    sql, params = store.session.call_pairs[-1]
    assert "opportunity_ranker_scores_latest" in sql
    assert "latest_batch" in sql
    assert "s.ticker IN UNNEST(@tickers)" in sql
    assert "s.profile IN UNNEST(@profiles)" in sql
    assert "s.bucket IN UNNEST(@buckets)" in sql
    assert "profile_rn <= @per_profile_limit" in sql
    assert "global_rn <= @limit" in sql
    assert "PARTITION BY market" in sql
    assert "s.market = b.market" in sql
    assert params["max_age_hours"] == 12
    assert params["buckets"] == ["momentum"]
    assert params["per_profile_limit"] == 2
    assert params["max_return_rows"] == 19
    assert "markets" not in params


def test_latest_opportunity_ranker_scores_filters_by_market() -> None:
    store = _make_market_store([[{"ticker": "AAPL", "market": "us"}]])

    rows = store.latest_opportunity_ranker_scores(
        markets=["us"],
        limit=5,
        max_age_hours=6,
    )

    assert rows[0]["ticker"] == "AAPL"
    sql, params = store.session.call_pairs[-1]
    assert "s.market IN UNNEST(@markets)" in sql
    assert "market IN UNNEST(@markets)" in sql
    assert params["markets"] == ["us"]


def test_insert_opportunity_ranker_scores_latest_appends_json_rows() -> None:
    store = _make_market_write_store()

    inserted = store.insert_opportunity_ranker_scores_latest(
        [
            {
                "as_of_date": "2026-04-17",
                "computed_at": "2026-04-18T00:00:00+00:00",
                "ranker_version": "ranker",
                "score_source": "learned",
                "ticker": "aapl",
                "recommendation_score": 0.12,
                "feature_json": '{"ret_20d": 0.1}',
                "explanation_json": {"top_features": ["ret_20d"]},
            }
        ]
    )

    assert inserted == 1
    table_id, rows = store.session.client.loads[-1]
    assert table_id == "proj.ds.opportunity_ranker_scores_latest"
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["feature_json"] == {"ret_20d": 0.1}


def test_latest_fundamentals_snapshot_does_not_retry_on_empty_rows() -> None:
    store = _make_market_store([[]])

    rows = store.latest_fundamentals_snapshot(tickers=["AAPL"], limit=5)

    assert rows == []
    assert len(store.session.calls) == 1
    assert "fundamentals_snapshot_latest" in store.session.calls[0]


def test_get_daily_closes_deduplicates_same_timestamp_rows() -> None:
    store = _make_market_store(
        [
            [
                {"as_of_ts": "2026-02-20T00:00:00+00:00", "ticker": "AAPL", "close_price_krw": 100.0},
                {"as_of_ts": "2026-02-20T00:00:00+00:00", "ticker": "AAPL", "close_price_krw": 101.0},
                {"as_of_ts": "2026-02-21T00:00:00+00:00", "ticker": "AAPL", "close_price_krw": 102.0},
            ]
        ]
    )

    rows = store.get_daily_closes(tickers=["AAPL"], lookback_days=10)

    assert rows["AAPL"] == [101.0, 102.0]


def test_get_daily_close_frame_deduplicates_same_day_rows() -> None:
    store = _make_market_store(
        [
            [
                {"d": "2026-02-20", "ticker": "AAPL", "close_price": 100.0},
                {"d": "2026-02-20", "ticker": "AAPL", "close_price": 101.0},
                {"d": "2026-02-21", "ticker": "AAPL", "close_price": 102.0},
                {"d": "2026-02-20", "ticker": "MSFT", "close_price": 200.0},
            ]
        ]
    )

    frame = store.get_daily_close_frame(
        tickers=["AAPL", "MSFT"],
        start=date(2026, 2, 20),
        end=date(2026, 2, 21),
    )

    assert float(frame.loc["2026-02-20", "AAPL"]) == 101.0
    assert float(frame.loc["2026-02-21", "AAPL"]) == 102.0
    assert float(frame.loc["2026-02-20", "MSFT"]) == 200.0


def test_get_daily_close_frame_supports_native_price_field() -> None:
    store = _make_market_store(
        [
            [
                {"d": "2026-02-20", "ticker": "AAPL", "close_price": 100.0},
                {"d": "2026-02-21", "ticker": "AAPL", "close_price": 105.0},
            ]
        ]
    )

    frame = store.get_daily_close_frame(
        tickers=["AAPL"],
        start=date(2026, 2, 20),
        end=date(2026, 2, 21),
        price_field="close_price_native",
    )

    assert "close_price_native AS close_price" in store.session.calls[0]
    assert float(frame.loc["2026-02-21", "AAPL"]) == 105.0


def test_insert_market_features_appends_via_load_job_without_delete() -> None:
    store = _make_market_write_store()

    store.insert_market_features(
        [
            {
                "ticker": "005930",
                "source": "open_trading_kospi",
                "exchange_code": "KRX",
                "instrument_id": "KRX:005930",
                "as_of_ts": "2026-03-07T00:00:00+00:00",
                "close_price_krw": 100.0,
            }
        ]
    )

    assert len(store.session.client.loads) == 1
    table_id, rows = store.session.client.loads[0]
    assert table_id == "proj.ds.market_features"
    assert rows[0]["ticker"] == "005930"
    assert rows[0]["source"] == "open_trading_kospi"
    assert "ingested_at" in rows[0]


def test_insert_market_features_latest_appends_via_load_job_without_delete() -> None:
    store = _make_market_write_store()

    written = store.insert_market_features_latest(
        [
            {
                "ticker": "005930",
                "source": "open_trading_kospi",
                "exchange_code": "KRX",
                "instrument_id": "KRX:005930",
                "as_of_ts": "2026-03-07T00:00:00+00:00",
                "close_price_krw": 100.0,
            }
        ]
    )

    assert written == 1
    assert len(store.session.client.loads) == 1
    table_id, rows = store.session.client.loads[0]
    assert table_id == "proj.ds.market_features_latest"
    assert rows[0]["ticker"] == "005930"
    assert "updated_at" in rows[0]


def test_insert_fundamentals_snapshot_latest_appends_via_load_job_without_delete() -> None:
    store = _make_market_write_store()

    written = store.insert_fundamentals_snapshot_latest(
        [
            {
                "ticker": "AAPL",
                "market": "us",
                "exchange_code": "NASD",
                "instrument_id": "NASD:AAPL",
                "currency": "USD",
                "as_of_ts": "2026-03-07T00:00:00+00:00",
                "per": 28.5,
                "pbr": 7.2,
                "eps": 6.15,
                "bps": 24.8,
                "source": "open_trading_us_price_detail",
            }
        ]
    )

    assert written == 1
    assert len(store.session.client.loads) == 1
    table_id, rows = store.session.client.loads[0]
    assert table_id == "proj.ds.fundamentals_snapshot_latest"
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["market"] == "us"
    assert "updated_at" in rows[0]


def test_replace_predicted_returns_appends_run_batch_without_delete() -> None:
    store = _make_market_write_store()

    written = store.replace_predicted_returns(
        [
            {
                "run_date": "2026-03-14",
                "ticker": "AAPL",
                "exp_return_period": 0.12,
                "forecast_horizon": 20,
                "forecast_model": "ensemble_wmae",
                "is_stacked": True,
            },
            {
                "run_date": "2026-03-14",
                "ticker": "MSFT",
                "exp_return_period": 0.08,
                "forecast_horizon": 20,
                "forecast_model": "ensemble_wmae",
                "is_stacked": True,
            },
        ],
        run_date=date(2026, 3, 14),
    )

    assert written == 2
    assert len(store.session.client.loads) == 1
    table_id, rows = store.session.client.loads[0]
    assert table_id == "proj.ds.predicted_expected_returns"
    assert rows[0]["forecast_run_id"].startswith("fc_")
    assert rows[1]["forecast_run_id"] == rows[0]["forecast_run_id"]


def test_get_predicted_returns_prefers_latest_forecast_batch_when_run_id_exists() -> None:
    store = _make_forecast_query_store(
        [
            {
                "run_date": "2026-03-14",
                "ticker": "AAPL",
                "exp_return_period": 0.12,
                "forecast_horizon": 20,
                "forecast_model": "ensemble_wmae",
                "is_stacked": True,
            }
        ],
        columns=[
            "run_date",
            "forecast_run_id",
            "ticker",
            "exp_return_period",
            "forecast_horizon",
            "forecast_model",
            "is_stacked",
            "created_at",
        ],
    )

    rows = store.get_predicted_returns(tickers=["AAPL"], limit=5, mode="stacked")

    assert rows[0]["ticker"] == "AAPL"
    assert store.session.call_pairs
    sql, params = store.session.call_pairs[-1]
    assert "latest_batch" in sql
    assert "forecast_run_id" in sql
    assert params["tickers"] == ["AAPL"]


# ===================================================================
# Tests — Sleeve Store: build_agent_sleeve_snapshot
# ===================================================================


def test_build_agent_sleeve_snapshot_propagates_execution_history_error() -> None:
    session = _FakeSession(responses=[RuntimeError("execution reports timeout")])
    market = _MarketStoreForReplay()
    store = _SleeveStoreForBuild(session, fill_result=RuntimeError("execution reports timeout"), market=market)

    with pytest.raises(RuntimeError, match="execution reports timeout"):
        store.build_agent_sleeve_snapshot(agent_id="agent-1")


def test_build_agent_sleeve_snapshot_rejects_invalid_initial_positions_json() -> None:
    session = _FakeSession(responses=[[]])
    market = _MarketStoreForReplay()
    store = _SleeveStoreForBuild(session, fill_result=[], init_positions_json="{bad json", market=market)

    with pytest.raises(RuntimeError, match="invalid initial_positions_json"):
        store.build_agent_sleeve_snapshot(agent_id="agent-1")


# ===================================================================
# Tests — Sleeve Store: ensure_agent_sleeves
# ===================================================================


def test_ensure_agent_sleeves_uses_virtual_seed_when_bootstrap_disabled(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)
    snapshot = AccountSnapshot(
        cash_krw=900_000.0,
        total_equity_krw=1_200_000.0,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                exchange_code="NASD",
                instrument_id="NASD:AAPL",
                quantity=6.0,
                avg_price_krw=50_000.0,
                market_price_krw=55_000.0,
            )
        },
    )
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=snapshot)

    store.ensure_agent_sleeves(agent_ids=["gpt", "gemini", "claude"], total_cash_krw=3_000_000.0)

    assert len(session.client.payloads) == 3
    first = session.client.payloads[0]
    assert float(first["initial_cash_krw"]) == 1_000_000.0
    assert str(first["initial_positions_json"]) == "[]"


def test_write_account_snapshot_persists_usd_krw_rate() -> None:
    snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=200_000.0,
        positions={},
        usd_krw_rate=1450.0,
    )
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=snapshot)

    store.write_account_snapshot(store._snapshot)

    assert float(session.client.payloads[0]["usd_krw_rate"]) == pytest.approx(1450.0)


def test_write_account_snapshot_appends_broker_cash_checkpoint() -> None:
    snapshot = AccountSnapshot(
        cash_krw=100_000.0,
        total_equity_krw=200_000.0,
        positions={},
        usd_krw_rate=1450.0,
        cash_foreign=50.0,
        cash_foreign_currency="USD",
    )
    # Ledger store needs a session that handles the dedup check
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids returns empty
    ])
    ledger = LedgerStore(ledger_session)
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=snapshot, ledger=ledger)

    store.write_account_snapshot(store._snapshot)

    assert [table_id for table_id, _ in session.client.calls] == [
        "proj.ds.account_snapshots",
    ]
    # The broker cash event is written via ledger store's session client
    assert [table_id for table_id, _ in ledger_session.client.calls] == [
        "proj.ds.broker_cash_events",
    ]
    _, cash_rows = ledger_session.client.calls[0]
    assert cash_rows[0]["event_type"] == "CASH_CHECKPOINT"
    assert float(cash_rows[0]["amount_krw"]) == pytest.approx(100_000.0)
    assert float(cash_rows[0]["amount_native"]) == pytest.approx(50.0)
    assert cash_rows[0]["currency"] == "USD"


def test_latest_account_snapshot_reads_usd_krw_rate() -> None:
    session = _FakeSession(
        responses=[
            [
                {
                    "snapshot_at": datetime(2026, 2, 21, tzinfo=timezone.utc),
                    "cash_krw": 100_000.0,
                    "total_equity_krw": 200_000.0,
                    "usd_krw_rate": 1450.0,
                }
            ],
            [],  # position rows
        ]
    )
    store = SleeveStore(session)

    snapshot = store.latest_account_snapshot()

    assert snapshot is not None
    assert snapshot.usd_krw_rate == pytest.approx(1450.0)


def test_get_latest_position_tickers_can_union_latest_snapshots_across_tenants() -> None:
    session = _FakeSession(
        responses=[
            [
                {"ticker": "VZ"},
                {"ticker": "005930"},
                {"ticker": "CSX"},
            ]
        ]
    )
    store = SleeveStore(session)

    tickers = store.get_latest_position_tickers(market="us", all_tenants=True)

    assert tickers == ["VZ", "CSX"]
    assert "GROUP BY tenant_id" in session.calls[0]
    assert session.call_pairs[0][1] == {}


def test_account_holdings_at_date_uses_latest_snapshot_before_date() -> None:
    session = _FakeSession(
        responses=[
            [
                {"ticker": "AAPL", "quantity": 2.0},
                {"ticker": "MSFT", "quantity": 1.0},
            ]
        ]
    )
    store = SleeveStore(session)

    holdings = store.account_holdings_at_date(as_of_date=date(2026, 2, 21))

    assert holdings == {"AAPL": pytest.approx(2.0), "MSFT": pytest.approx(1.0)}


def test_account_cash_history_uses_range_filter() -> None:
    start_at = datetime(2026, 2, 21, tzinfo=timezone.utc)
    end_at = datetime(2026, 2, 22, tzinfo=timezone.utc)
    cash_rows = [
        {
            "snapshot_at": start_at,
            "cash_krw": 100_000.0,
            "total_equity_krw": 200_000.0,
            "usd_krw_rate": 1450.0,
            "cash_foreign": 50.0,
            "cash_foreign_currency": "USD",
        }
    ]
    session = _FakeSession(responses=[list(cash_rows)])
    store = SleeveStore(session)

    rows = store.account_cash_history(start_at=start_at, end_at=end_at, tenant_id="midnightnnn")

    assert rows[0]["cash_krw"] == pytest.approx(100_000.0)
    last_sql = session.calls[-1]
    assert "FROM `proj.ds.account_snapshots`" in last_sql
    last_params = session.call_pairs[-1][1]
    assert last_params == {
        "tenant_id": "midnightnnn",
        "start_at": start_at,
        "end_at": end_at,
    }


def test_ensure_agent_sleeves_can_seed_from_account_snapshot(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", "true")
    snapshot = AccountSnapshot(
        cash_krw=900_000.0,
        total_equity_krw=1_200_000.0,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                exchange_code="NASD",
                instrument_id="NASD:AAPL",
                quantity=6.0,
                avg_price_krw=50_000.0,
                market_price_krw=55_000.0,
            )
        },
    )
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=snapshot)

    store.ensure_agent_sleeves(agent_ids=["gpt", "gemini", "claude"], total_cash_krw=3_000_000.0)

    assert len(session.client.payloads) == 3
    first = session.client.payloads[0]
    assert float(first["initial_cash_krw"]) == 300_000.0
    seeded = json.loads(str(first["initial_positions_json"]))
    assert len(seeded) == 1
    assert float(seeded[0]["quantity"]) == pytest.approx(2.0)
    assert float(seeded[0]["avg_price_krw"]) == pytest.approx(50_000.0)


def test_ensure_agent_sleeves_mirrors_seed_into_agent_state_checkpoints(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)
    # Ledger session needs to handle: existing_event_ids (dedup) returning empty
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids for agent_state_checkpoints
    ])
    ledger = LedgerStore(ledger_session)
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=None, ledger=ledger)

    store.ensure_agent_sleeves(agent_ids=["gpt", "gemini"], total_cash_krw=2_000_000.0)

    # sleeve inserts go to the sleeve session's client
    # checkpoint inserts go to the ledger session's client
    sleeve_tables = [table_id for table_id, _ in session.client.calls]
    ledger_tables = [table_id for table_id, _ in ledger_session.client.calls]
    assert sleeve_tables == ["proj.ds.agent_sleeves"]
    assert ledger_tables == ["proj.ds.agent_state_checkpoints"]
    _, checkpoint_rows = ledger_session.client.calls[0]
    assert {row["agent_id"] for row in checkpoint_rows} == {"gpt", "gemini"}
    assert all(row["source"] == "agent_sleeves.ensure" for row in checkpoint_rows)
    assert all(json.loads(str(row["positions_json"])) == [] for row in checkpoint_rows)


def test_ensure_agent_state_checkpoints_writes_checkpoint_only(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)

    class _LedgerForCheckpointEnsure(LedgerStore):
        def latest_agent_state_checkpoints(self, *, agent_ids, tenant_id=None):
            _ = tenant_id
            return {}

        def existing_event_ids(self, table_name, event_ids, *, tenant_id=None):
            _ = (table_name, event_ids, tenant_id)
            return set()

    ledger_session = _FakeSession()
    ledger = _LedgerForCheckpointEnsure(ledger_session)
    session = _FakeSession()
    store = _SleeveStoreForInit(session, snapshot=None, ledger=ledger)

    store.ensure_agent_state_checkpoints(agent_ids=["gpt", "gemini"], total_cash_krw=2_000_000.0)

    assert [table_id for table_id, _ in ledger_session.client.calls] == ["proj.ds.agent_state_checkpoints"]
    _, checkpoint_rows = ledger_session.client.calls[0]
    assert {row["agent_id"] for row in checkpoint_rows} == {"gpt", "gemini"}
    assert all(row["source"] == "agent_state_checkpoints.ensure" for row in checkpoint_rows)
    assert all(float(row["cash_krw"]) == pytest.approx(1_000_000.0) for row in checkpoint_rows)


# ===================================================================
# Tests — Sleeve Store: capital replay via build_agent_sleeve_snapshot
# ===================================================================


def _make_capital_replay_store(
    *,
    checkpoint,
    capital_events,
    fills=None,
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
        manual_cash_adjustments=manual_cash_adjustments,
        transfer_events=transfer_events,
    )
    market = _MarketStoreForReplay()
    store = _SleeveStoreForCapitalReplay(sleeve_session, ledger=ledger, market=market)
    return store


def test_build_agent_sleeve_snapshot_replays_capital_events_from_checkpoint_seed() -> None:
    store = _make_capital_replay_store(
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 1_000_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[
            {
                "event_id": "cap_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 250_000.0,
                "event_type": "INJECTION",
            }
        ],
    )

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(1_250_000.0)
    assert snapshot.total_equity_krw == pytest.approx(1_250_000.0)
    assert baseline == pytest.approx(1_250_000.0)
    assert meta["seed_source"] == "checkpoint_test"
    assert meta["capital_event_count"] == 1
    assert meta["capital_flow_krw"] == pytest.approx(250_000.0)


def test_build_agent_sleeve_snapshot_replays_manual_cash_adjustments() -> None:
    store = _make_capital_replay_store(
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 1_000_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
        manual_cash_adjustments=[
            {
                "event_id": "adj_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "delta_cash_krw": -125_000.0,
            }
        ],
    )

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(875_000.0)
    assert baseline == pytest.approx(875_000.0)
    assert meta["manual_cash_adjustment_count"] == 1
    assert meta["manual_cash_adjustment_krw"] == pytest.approx(-125_000.0)


def test_build_agent_sleeve_snapshot_replays_agent_transfer_events() -> None:
    store = _make_capital_replay_store(
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 100_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
        transfer_events=[
            {
                "event_id": "xfer_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "transfer_type": "POSITION_TRANSFER",
                "from_agent_id": "gemini",
                "to_agent_id": "gpt",
                "ticker": "AAPL",
                "quantity": 1.0,
                "price_krw": 50_000.0,
                "amount_krw": 50_000.0,
            }
        ],
    )

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(50_000.0)
    assert snapshot.total_equity_krw == pytest.approx(100_000.0)
    assert snapshot.positions["AAPL"].quantity == pytest.approx(1.0)
    assert snapshot.positions["AAPL"].avg_price_krw == pytest.approx(50_000.0)
    assert baseline == pytest.approx(100_000.0)
    assert meta["transfer_event_count"] == 1
    assert meta["transfer_cash_krw"] == pytest.approx(-50_000.0)
    assert meta["transfer_equity_krw"] == pytest.approx(0.0)


def test_build_agent_sleeve_snapshot_treats_cash_transfer_as_capital_basis() -> None:
    store = _make_capital_replay_store(
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 100_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
        transfer_events=[
            {
                "event_id": "xfer_cash_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "transfer_type": "CASH_TRANSFER",
                "from_agent_id": "gemini",
                "to_agent_id": "gpt",
                "amount_krw": 250_000.0,
            }
        ],
    )

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(350_000.0)
    assert snapshot.total_equity_krw == pytest.approx(350_000.0)
    assert baseline == pytest.approx(350_000.0)
    assert meta["transfer_event_count"] == 1
    assert meta["transfer_cash_krw"] == pytest.approx(250_000.0)
    assert meta["transfer_equity_krw"] == pytest.approx(250_000.0)


def test_build_agent_sleeve_snapshot_replays_execution_fills_from_checkpoint_seed() -> None:
    store = _make_capital_replay_store(
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 100_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
        fills=[
            {
                "created_at": "2026-03-02T00:00:00+00:00",
                "ticker": "AAPL",
                "exchange_code": "NAS",
                "instrument_id": "AAPL",
                "side": "BUY",
                "filled_qty": 1.0,
                "avg_price_krw": 50_000.0,
                "avg_price_native": 34.0,
                "quote_currency": "USD",
                "fx_rate": 1470.0,
                "status": "FILLED",
            }
        ],
    )

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.cash_krw == pytest.approx(50_000.0)
    assert snapshot.total_equity_krw == pytest.approx(100_000.0)
    assert snapshot.positions["AAPL"].quantity == pytest.approx(1.0)
    assert snapshot.positions["AAPL"].avg_price_krw == pytest.approx(50_000.0)
    assert baseline == pytest.approx(100_000.0)
    assert meta["trade_count_total"] == 1


def test_build_agent_sleeve_snapshot_prefers_latest_instrument_metadata_for_live_positions() -> None:
    class _MarketWithInstrumentMap:
        def latest_close_prices_with_currency(self, *, tickers, sources=None, as_of_date=None):
            _ = sources
            assert tickers == ["CSX"]
            return {
                "CSX": {
                    "close_price_krw": 50_000.0,
                    "close_price_native": 39.3,
                    "quote_currency": "USD",
                    "fx_rate_used": 1272.0,
                }
            }

        def latest_instrument_map(self, tickers):
            assert tickers == ["CSX"]
            return {
                "CSX": {
                    "exchange_code": "NASD",
                    "instrument_id": "NASD:CSX",
                }
            }

    sleeve_session = _FakeSession(responses=[
        [
            {
                "created_at": "2026-03-02T00:00:00+00:00",
                "ticker": "CSX",
                "exchange_code": "NYSE",
                "instrument_id": "NYSE:CSX",
                "side": "BUY",
                "filled_qty": 1.0,
                "avg_price_krw": 40_000.0,
                "avg_price_native": 31.0,
                "quote_currency": "USD",
                "fx_rate": 1290.0,
                "status": "FILLED",
            }
        ],
    ])
    ledger = _LedgerStoreForCapitalReplay(
        _FakeSession(),
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 100_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
    )
    market = _MarketWithInstrumentMap()
    store = _SleeveStoreForCapitalReplay(sleeve_session, ledger=ledger, market=market)

    snapshot, baseline, meta = store.build_agent_sleeve_snapshot(agent_id="gpt")

    assert snapshot.positions["CSX"].exchange_code == "NASD"
    assert snapshot.positions["CSX"].instrument_id == "NASD:CSX"
    assert snapshot.positions["CSX"].market_price_native == pytest.approx(39.3)
    assert baseline == pytest.approx(100_000.0)
    assert meta["trade_count_total"] == 1


def test_agent_holdings_at_date_replays_agent_transfer_events() -> None:
    # agent_holdings_at_date calls:
    # 1. _load_agent_seed_state -> ledger.latest_agent_state_checkpoints -> sleeve.latest_agent_sleeves -> session.fetch_rows (fills) -> ledger.agent_transfer_events_since
    # We need the sleeve session to handle execution_reports fetch (returns []),
    # plus the agent_holdings_at_date fetch (returns []).
    sleeve_session = _FakeSession(responses=[
        [],  # execution_reports fill query
    ])
    ledger = _LedgerStoreForCapitalReplay(
        _FakeSession(),
        checkpoint={
            "event_id": "chk_1",
            "checkpoint_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
            "cash_krw": 100_000.0,
            "positions_json": [],
            "source": "checkpoint_test",
        },
        capital_events=[],
        transfer_events=[
            {
                "event_id": "xfer_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "transfer_type": "POSITION_TRANSFER",
                "from_agent_id": "gemini",
                "to_agent_id": "gpt",
                "ticker": "AAPL",
                "quantity": 1.0,
                "price_krw": 50_000.0,
                "amount_krw": 50_000.0,
            }
        ],
    )

    store = _SleeveStoreForCapitalReplay(sleeve_session, ledger=ledger)

    holdings = store.agent_holdings_at_date(agent_id="gpt", as_of_date=date(2026, 3, 3))

    assert holdings == {"AAPL": pytest.approx(1.0)}


def test_trace_agent_actual_capital_basis_replays_real_cash_events_from_origin() -> None:
    origin_state = {
        "source": "legacy_agent_sleeve",
        "since": datetime(2026, 3, 1, tzinfo=timezone.utc),
        "cash_krw": 1_000_000.0,
        "positions_payload": [],
        "positions_error": None,
    }
    ledger = _LedgerStoreForCapitalReplay(
        _FakeSession(),
        capital_events=[
            {
                "event_id": "cap_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 110_000.0,
                "event_type": "INJECTION",
            }
        ],
        manual_cash_adjustments=[
            {
                "event_id": "adj_1",
                "occurred_at": datetime(2026, 3, 3, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "delta_cash_krw": -10_000.0,
            }
        ],
        transfer_events=[
            {
                "event_id": "xfer_cash_1",
                "occurred_at": datetime(2026, 3, 4, tzinfo=timezone.utc),
                "transfer_type": "CASH_TRANSFER",
                "from_agent_id": "gemini",
                "to_agent_id": "gpt",
                "amount_krw": 25_000.0,
            }
        ],
    )
    store = _ActualBasisSleeveStore(_FakeSession(), origin_state=origin_state, ledger=ledger)

    trace = store.trace_agent_actual_capital_basis(agent_id="gpt")

    assert trace["seed_cash_krw"] == pytest.approx(1_000_000.0)
    assert trace["baseline_equity_krw"] == pytest.approx(1_125_000.0)
    assert trace["capital_flow_krw"] == pytest.approx(110_000.0)
    assert trace["manual_cash_adjustment_krw"] == pytest.approx(-10_000.0)
    assert trace["transfer_equity_krw"] == pytest.approx(25_000.0)


def test_fetch_actual_agent_nav_history_overlays_traced_actual_basis() -> None:
    origin_state = {
        "source": "legacy_agent_sleeve",
        "since": datetime(2026, 3, 1, tzinfo=timezone.utc),
        "cash_krw": 1_000_000.0,
        "positions_payload": [],
        "positions_error": None,
    }
    ledger = _LedgerStoreForCapitalReplay(
        _FakeSession(),
        capital_events=[
            {
                "event_id": "cap_1",
                "occurred_at": datetime(2026, 3, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "amount_krw": 110_000.0,
                "event_type": "INJECTION",
            }
        ],
    )
    store = _ActualBasisSleeveStore(
        _FakeSession(),
        origin_state=origin_state,
        ledger=ledger,
        nav_rows=[
            {
                "nav_date": date(2026, 3, 3),
                "agent_id": "gpt",
                "nav_krw": 1_090_000.0,
                "pnl_krw": 90_000.0,
                "pnl_ratio": 0.09,
            }
        ],
    )

    rows = store.fetch_actual_agent_nav_history(tenant_id="local", agent_ids=["gpt"], limit=10)

    assert rows[0]["baseline_equity_krw"] == pytest.approx(1_110_000.0)
    assert rows[0]["pnl_krw"] == pytest.approx(-20_000.0)
    assert rows[0]["pnl_ratio"] == pytest.approx(-20_000.0 / 1_110_000.0)


# ===================================================================
# Tests — Sleeve Store: NAV
# ===================================================================


def test_fetch_agent_nav_history_prefers_official_rows() -> None:
    store = _NavSleeveStore.create()
    store.rows = [{"nav_date": date(2026, 3, 12), "agent_id": "gpt", "nav_krw": 1_100_000.0, "pnl_krw": 100_000.0, "pnl_ratio": 0.1}]

    rows = store.fetch_agent_nav_history(tenant_id="midnightnnn", agent_ids=["gpt"], limit=10)

    assert rows[0]["agent_id"] == "gpt"
    sql, params = store.executed[0]
    assert "official_nav_daily" in sql
    assert "agent_nav_daily" in sql
    assert params == {"tenant_id": "midnightnnn", "limit": 10, "agent_ids": ["gpt"]}


def test_upsert_agent_nav_daily_mirrors_into_official_nav_daily() -> None:
    store = _NavSleeveStore.create()

    store.upsert_agent_nav_daily(
        nav_date=date(2026, 3, 12),
        agent_id="gpt",
        nav_krw=1_250_000.0,
        baseline_equity_krw=1_000_000.0,
        cash_krw=200_000.0,
        market_value_krw=1_050_000.0,
        capital_flow_krw=150_000.0,
        fx_source="market_features_latest.fx_rate_used",
        valuation_source="agent_sleeve_snapshot",
        tenant_id="midnightnnn",
    )

    assert len(store.executed) == 4
    _, official_params = store.executed[-1]
    assert official_params is not None
    assert official_params["tenant_id"] == "midnightnnn"
    assert official_params["cash_krw"] == pytest.approx(200_000.0)
    assert official_params["market_value_krw"] == pytest.approx(1_050_000.0)
    assert official_params["capital_flow_krw"] == pytest.approx(150_000.0)
    assert official_params["fx_source"] == "market_features_latest.fx_rate_used"
    assert official_params["valuation_source"] == "agent_sleeve_snapshot"


# ===================================================================
# Tests — Sleeve Store: retarget
# ===================================================================


def test_retarget_agent_sleeves_preserve_positions_sets_cash_from_target_gap() -> None:
    session = _FakeSession()
    store = _SleeveStoreForRetarget(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=100_000.0,
                total_equity_krw=400_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000.0,
                        market_price_krw=150_000.0,
                    )
                },
            )
        },
    )

    out = store.retarget_agent_sleeves_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000.0,
    )

    assert len(session.client.payloads) == 1
    first = session.client.payloads[0]
    assert float(first["initial_cash_krw"]) == pytest.approx(200_000.0)
    seeded = json.loads(str(first["initial_positions_json"]))
    assert len(seeded) == 1
    assert float(seeded[0]["quantity"]) == pytest.approx(2.0)
    assert float(seeded[0]["avg_price_krw"]) == pytest.approx(150_000.0)
    assert out["gpt"]["over_target"] is False


def test_retarget_agent_capitals_preserve_positions_appends_capital_events() -> None:
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids for capital_events
        [],  # existing_event_ids for agent_state_checkpoints
    ])
    ledger = _LedgerStoreForCapitalRetarget(ledger_session)
    session = _FakeSession()
    store = _SleeveStoreForCapitalRetarget(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=100_000.0,
                total_equity_krw=400_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000.0,
                        market_price_krw=150_000.0,
                    )
                },
            )
        },
        ledger=ledger,
    )

    out = store.retarget_agent_capitals_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000.0,
        created_by="tester",
    )

    assert len(ledger_session.client.calls) == 2
    table_id, rows = ledger_session.client.calls[0]
    assert table_id == "proj.ds.capital_events"
    assert rows[0]["agent_id"] == "gpt"
    assert float(rows[0]["amount_krw"]) == pytest.approx(100_000.0)
    assert rows[0]["event_type"] == "INJECTION"
    assert out["gpt"]["target_cash_krw"] == pytest.approx(200_000.0)
    assert out["gpt"]["capital_flow_krw"] == pytest.approx(100_000.0)
    assert out["gpt"]["over_target"] is False

    # Checkpoint is also synced after capital event
    cp_table_id, cp_rows = ledger_session.client.calls[1]
    assert cp_table_id == "proj.ds.agent_state_checkpoints"
    assert cp_rows[0]["agent_id"] == "gpt"
    assert float(cp_rows[0]["cash_krw"]) == pytest.approx(200_000.0)
    assert cp_rows[0]["source"] == "capital_events.retarget"


def test_retarget_agent_capitals_preserves_pnl_on_capital_change() -> None:
    """When capital is raised, existing P&L must be preserved on top of new capital."""

    class _PnlAwareStore(_SleeveStoreForCapitalRetarget):
        def __init__(self, session, *, snapshots, baselines, ledger):
            super().__init__(session, snapshots=snapshots, ledger=ledger)
            self._baselines = baselines

        def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True, tenant_id=None, as_of_ts=None):
            _ = (sources, include_simulated, tenant_id, as_of_ts)
            snap = self._snapshots[str(agent_id)]
            baseline = self._baselines[str(agent_id)]
            return snap, baseline, {"agent_id": str(agent_id)}

    # Agent started with 340k capital, now has 60k profit -> equity 400k
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids for capital_events
        [],  # existing_event_ids for agent_state_checkpoints
    ])
    ledger = _LedgerStoreForCapitalRetarget(ledger_session)
    session = _FakeSession()
    store = _PnlAwareStore(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=100_000.0,
                total_equity_krw=400_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000.0,
                        market_price_krw=150_000.0,
                    )
                },
            )
        },
        baselines={"gpt": 340_000.0},
        ledger=ledger,
    )

    out = store.retarget_agent_capitals_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000.0,
        created_by="tester",
    )

    meta = out["gpt"]
    # delta = 500k - 340k = 160k (not 100k like old absolute-target mode)
    assert meta["capital_flow_krw"] == pytest.approx(160_000.0)
    # new cash = 100k + 160k = 260k
    assert meta["target_cash_krw"] == pytest.approx(260_000.0)
    # effective equity = 400k + 160k = 560k = 500k(new capital) + 60k(pnl)
    assert meta["effective_target_equity_krw"] == pytest.approx(560_000.0)
    assert meta["over_target"] is False

    table_id, rows = ledger_session.client.calls[0]
    assert rows[0]["event_type"] == "INJECTION"
    assert float(rows[0]["amount_krw"]) == pytest.approx(160_000.0)


def test_retarget_agent_capitals_clamps_cash_when_withdrawal_exceeds_available() -> None:
    """When capital reduction requires more cash withdrawal than available, clamp to 0."""

    class _PnlAwareStore(_SleeveStoreForCapitalRetarget):
        def __init__(self, session, *, snapshots, baselines, ledger):
            super().__init__(session, snapshots=snapshots, ledger=ledger)
            self._baselines = baselines

        def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True, tenant_id=None, as_of_ts=None):
            _ = (sources, include_simulated, tenant_id, as_of_ts)
            snap = self._snapshots[str(agent_id)]
            baseline = self._baselines[str(agent_id)]
            return snap, baseline, {"agent_id": str(agent_id)}

    # baseline 500k, cash 50k, positions 400k, equity 450k (pnl = -50k)
    # target = 200k -> delta = 200k - 500k = -300k -> new_cash = 50k - 300k = -250k -> clamp
    ledger_session = _FakeSession(responses=[
        [],  # existing_event_ids for capital_events
        [],  # existing_event_ids for agent_state_checkpoints
    ])
    ledger = _LedgerStoreForCapitalRetarget(ledger_session)
    session = _FakeSession()
    store = _PnlAwareStore(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=50_000.0,
                total_equity_krw=450_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=180_000.0,
                        market_price_krw=200_000.0,
                    )
                },
            )
        },
        baselines={"gpt": 500_000.0},
        ledger=ledger,
    )

    out = store.retarget_agent_capitals_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=200_000.0,
        created_by="tester",
    )

    meta = out["gpt"]
    # Clamped: withdraw only available cash (50k), not full 300k
    assert meta["capital_flow_krw"] == pytest.approx(-50_000.0)
    assert meta["target_cash_krw"] == pytest.approx(0.0)
    assert meta["over_target"] is True


def test_retarget_agent_sleeves_preserve_positions_clamps_cash_when_over_target() -> None:
    session = _FakeSession()
    store = _SleeveStoreForRetarget(
        session,
        snapshots={
            "gpt": AccountSnapshot(
                cash_krw=0.0,
                total_equity_krw=600_000.0,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=250_000.0,
                        market_price_krw=300_000.0,
                    )
                },
            )
        },
    )

    out = store.retarget_agent_sleeves_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000.0,
    )

    assert len(session.client.payloads) == 1
    first = session.client.payloads[0]
    assert float(first["initial_cash_krw"]) == 0.0
    assert out["gpt"]["over_target"] is True
