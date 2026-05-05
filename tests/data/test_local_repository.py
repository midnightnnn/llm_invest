"""End-to-end local backend tests for ``LocalRepository``.

Strategy: spin up a real DuckDB file in ``tmp_path``, run ``ensure_tables``
to materialise schema, seed rows directly via raw SQL, and verify each
implemented store method returns the expected shape.

The unimplemented-method path is also asserted so optional-method feature
detection via ``hasattr`` keeps working.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


pytest.importorskip("duckdb")  # entire module skipped without local extras.


from arena.data.local.repository import LocalRepository  # noqa: E402
from arena.agents.investment_chat.drafts import draft_key, load_draft, save_draft  # noqa: E402
from arena.models import ExecutionReport, ExecutionStatus, MemoryEvent, OrderIntent, RiskDecision, Side  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "arena.duckdb")
    r = LocalRepository(tenant_id="tenant-a", db_path=db_path)
    r.ensure_dataset()
    r.ensure_tables()
    yield r
    r.session.close()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Bootstrap surface
# ---------------------------------------------------------------------------


def test_ensure_tables_creates_arena_schema(repo):
    rows = repo.fetch_rows(
        "SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_schema='main'"
    )
    assert int(rows[0]["n"]) > 50  # 54 arena tables today, future-proof.


def test_dataset_fqn_is_empty_for_facade_parity(repo):
    assert repo.dataset_fqn == ""


def test_resolve_tenant_id_normalises(repo):
    assert repo.resolve_tenant_id(None) == "tenant-a"
    assert repo.resolve_tenant_id("Tenant-B") == "tenant-b"


# ---------------------------------------------------------------------------
# ConfigStore (get/set/get_configs)
# ---------------------------------------------------------------------------


def test_set_and_get_config(repo):
    repo.set_config("tenant-a", "risk_policy", "aggressive", updated_by="midnight")
    assert repo.get_config("tenant-a", "risk_policy") == "aggressive"
    assert repo.get_config("tenant-a", "unset_key") is None
    assert repo.get_config("other-tenant", "risk_policy") is None


def test_chat_order_draft_key_is_valid_local_config_key(repo):
    key = draft_key("abc123")
    assert ":" not in key

    save_draft(repo, tenant_id="tenant-a", token="abc123", draft={"status": "draft"})

    assert repo.get_config("tenant-a", key)
    assert load_draft(repo, tenant_id="tenant-a", token="abc123") == {"status": "draft"}


def test_get_configs_returns_latest_per_key(repo):
    repo.set_config("tenant-a", "k1", "old")
    repo.set_config("tenant-a", "k1", "new")
    repo.set_config("tenant-a", "k2", "v2")

    out = repo.get_configs("tenant-a", ["k1", "k2", "k3"])
    assert out == {"k1": "new", "k2": "v2"}


def test_set_config_rejects_blank_inputs(repo):
    with pytest.raises(ValueError):
        repo.set_config("", "k", "v")
    with pytest.raises(ValueError):
        repo.set_config("tenant-a", "", "v")


# ---------------------------------------------------------------------------
# MarketReader
# ---------------------------------------------------------------------------


def _seed_market_features_latest(repo, rows):
    cols = (
        "as_of_ts, ticker, exchange_code, instrument_id, close_price_krw, "
        "close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d, "
        "volatility_20d, sentiment_score, source, updated_at"
    )
    placeholders = ", ".join(["?"] * 14)
    for r in rows:
        repo.execute(
            f"INSERT INTO market_features_latest ({cols}) VALUES ({placeholders})",
            [
                r["as_of_ts"], r["ticker"], r.get("exchange_code"), r.get("instrument_id"),
                r["close_price_krw"], r.get("close_price_native"), r.get("quote_currency", "USD"),
                r.get("fx_rate_used", 1.0), r.get("ret_5d"), r.get("ret_20d"),
                r.get("volatility_20d"), r.get("sentiment_score"), r["source"], r["updated_at"],
            ],
        )


def test_latest_close_prices_returns_only_positive(repo):
    now = _now()
    _seed_market_features_latest(repo, [
        {"as_of_ts": now, "ticker": "AAPL", "close_price_krw": 247000.0, "source": "test", "updated_at": now},
        {"as_of_ts": now, "ticker": "MSFT", "close_price_krw": 601000.0, "source": "test", "updated_at": now},
        {"as_of_ts": now, "ticker": "ZERO", "close_price_krw": 0.0, "source": "test", "updated_at": now},
    ])

    out = repo.latest_close_prices(tickers=["aapl", "MSFT", "ZERO", "MISSING"])
    assert out == {"AAPL": 247000.0, "MSFT": 601000.0}


def test_latest_close_prices_dedups_to_latest_row(repo):
    older = datetime(2026, 4, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 4, 28, tzinfo=timezone.utc)
    _seed_market_features_latest(repo, [
        {"as_of_ts": older, "ticker": "AAPL", "close_price_krw": 100.0, "source": "t", "updated_at": older},
        {"as_of_ts": newer, "ticker": "AAPL", "close_price_krw": 200.0, "source": "t", "updated_at": newer},
    ])
    out = repo.latest_close_prices(tickers=["AAPL"])
    assert out == {"AAPL": 200.0}


def test_latest_close_prices_with_currency_includes_native(repo):
    now = _now()
    _seed_market_features_latest(repo, [
        {
            "as_of_ts": now, "ticker": "AAPL", "close_price_krw": 247000.0,
            "close_price_native": 178.5, "quote_currency": "USD",
            "fx_rate_used": 1383.0, "source": "test", "updated_at": now,
        },
    ])
    out = repo.latest_close_prices_with_currency(tickers=["AAPL"])
    row = out["AAPL"]
    assert row["close_price_krw"] == 247000.0
    assert row["close_price_native"] == 178.5
    assert row["quote_currency"] == "USD"
    assert row["fx_rate_used"] == 1383.0


def test_latest_market_features_returns_full_rows(repo):
    now = _now()
    _seed_market_features_latest(repo, [
        {
            "as_of_ts": now, "ticker": "TSLA", "close_price_krw": 308000.0,
            "ret_5d": -0.024, "ret_20d": -0.011, "volatility_20d": 0.039,
            "sentiment_score": -0.08, "source": "test", "updated_at": now,
        },
    ])
    rows = repo.latest_market_features(tickers=["TSLA"], limit=10)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "TSLA"
    assert rows[0]["ret_5d"] == pytest.approx(-0.024)


def test_ticker_name_map_uses_instrument_master(repo):
    now = _now()
    repo.execute(
        """
        INSERT INTO instrument_master (instrument_id, ticker, ticker_name, exchange_code, currency, updated_at)
        VALUES
          ('NASD:AAPL', 'AAPL', 'Apple Inc.', 'NASD', 'USD', ?),
          ('NASD:MSFT', 'MSFT', 'Microsoft Corp.', 'NASD', 'USD', ?)
        """,
        [now, now],
    )
    out = repo.ticker_name_map(tickers=["AAPL", "MSFT", "MISSING"])
    assert out == {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corp."}


def test_latest_instrument_map_returns_dicts(repo):
    now = _now()
    repo.execute(
        """
        INSERT INTO instrument_master (instrument_id, ticker, ticker_name, exchange_code, currency, updated_at)
        VALUES ('NASD:NVDA', 'NVDA', 'NVIDIA', 'NASD', 'USD', ?)
        """,
        [now],
    )
    out = repo.latest_instrument_map(["NVDA"])
    assert "NVDA" in out
    assert out["NVDA"]["exchange_code"] == "NASD"
    assert out["NVDA"]["currency"] == "USD"


# ---------------------------------------------------------------------------
# MemoryReader (subset)
# ---------------------------------------------------------------------------


def _seed_memory_event(repo, *, event_id: str, agent_id: str, summary: str, ts: datetime):
    repo.execute(
        """
        INSERT INTO agent_memory_events
          (tenant_id, event_id, created_at, agent_id, event_type, summary, trading_mode)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ["tenant-a", event_id, ts, agent_id, "lesson", summary, "paper"],
    )


def test_recent_memory_events_returns_per_agent_in_order(repo):
    t1 = datetime(2026, 4, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 2, tzinfo=timezone.utc)
    _seed_memory_event(repo, event_id="e1", agent_id="gpt", summary="old", ts=t1)
    _seed_memory_event(repo, event_id="e2", agent_id="gpt", summary="new", ts=t2)
    _seed_memory_event(repo, event_id="e3", agent_id="claude", summary="other", ts=t2)

    rows = repo.recent_memory_events("gpt", limit=10)
    summaries = [r["summary"] for r in rows]
    assert summaries == ["new", "old"]


def test_memory_event_by_id_round_trips(repo):
    t = _now()
    _seed_memory_event(repo, event_id="abc", agent_id="gemini", summary="hello", ts=t)
    row = repo.memory_event_by_id(event_id="abc")
    assert row and row["summary"] == "hello"
    assert repo.memory_event_by_id(event_id="missing") is None


def test_memory_events_by_ids_filters_to_known(repo):
    t = _now()
    for eid in ("a", "b", "c"):
        _seed_memory_event(repo, event_id=eid, agent_id="gpt", summary=eid, ts=t)
    rows = repo.memory_events_by_ids(agent_id="gpt", event_ids=["a", "c", "ghost"])
    summaries = sorted(r["summary"] for r in rows)
    assert summaries == ["a", "c"]


def test_memory_events_for_cycle_matches_column_and_payload_cycle(repo):
    t = _now()
    repo.execute(
        """
        INSERT INTO agent_memory_events
          (tenant_id, event_id, created_at, agent_id, event_type, summary, trading_mode, cycle_id, payload_json)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "tenant-a", "cycle-col", t, "gemini", "trade_outcome", "column match", "paper", "cycle-x", "{}",
            "tenant-a", "cycle-payload", t, "gemini", "lesson", "payload match", "paper", None, '{"intent":{"cycle_id":"cycle-x"}}',
            "tenant-a", "other-cycle", t, "gemini", "lesson", "miss", "paper", "cycle-y", "{}",
        ],
    )

    rows = repo.memory_events_for_cycle(
        agent_id="gemini",
        cycle_id="cycle-x",
        event_types=["lesson"],
        limit=10,
    )

    assert [row["event_id"] for row in rows] == ["cycle-payload"]


def test_latest_memory_compaction_cycle_id_uses_latest_matching_cycle(repo):
    older = datetime(2026, 4, 1, tzinfo=timezone.utc)
    newer = datetime(2026, 4, 2, tzinfo=timezone.utc)
    repo.execute(
        """
        INSERT INTO agent_memory_events
          (tenant_id, event_id, created_at, agent_id, event_type, summary, trading_mode, cycle_id, payload_json)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "tenant-a", "old-cycle", older, "gemini", "trade_execution", "old", "paper", "cycle-old", "{}",
            "tenant-a", "new-cycle", newer, "gemini", "thesis_open", "new", "paper", None, '{"cycle_id":"cycle-new"}',
            "tenant-a", "ignored-agent", newer, "gpt", "trade_execution", "ignored", "paper", "cycle-gpt", "{}",
        ],
    )

    cycle_id = repo.latest_memory_compaction_cycle_id(
        agent_ids=["gemini"],
        event_types=["trade_execution", "thesis_open"],
        trading_mode="paper",
    )

    assert cycle_id == "cycle-new"


def test_relation_extraction_pending_sources_returns_source_text(repo):
    t = _now()
    _seed_memory_event(
        repo,
        event_id="rel-source",
        agent_id="gpt",
        summary="AI demand supports NVDA margin recovery.",
        ts=t,
    )

    rows = repo.relation_extraction_pending_sources(
        limit=10,
        source_table="agent_memory_events",
        event_types=["lesson"],
        trading_mode="paper",
        extractor_version="semantic_relation_extractor_v1",
        prompt_version="semantic_relation_prompt_v2",
        ontology_version="semantic_relation_ontology_v1",
        tenant_id="tenant-a",
    )

    assert len(rows) == 1
    assert rows[0]["source_text"] == "AI demand supports NVDA margin recovery."
    assert "text" not in rows[0]


def test_compaction_reflections_for_cycle_returns_existing_reflections(repo):
    t = _now()
    repo.execute(
        """
        INSERT INTO agent_memory_events
          (tenant_id, event_id, created_at, agent_id, event_type, summary, trading_mode, cycle_id, payload_json)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, ?),
          (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "tenant-a", "reflection", t, "gemini", "strategy_reflection", "keep", "paper", "cycle-x", '{"source":"memory_compaction"}',
            "tenant-a", "manual", t, "gemini", "strategy_reflection", "skip", "paper", "cycle-x", '{"source":"manual"}',
        ],
    )

    rows = repo.compaction_reflections_for_cycle(agent_id="gemini", cycle_id="cycle-x", limit=10)

    assert [row["event_id"] for row in rows] == ["reflection"]


# ---------------------------------------------------------------------------
# Research / runtime support
# ---------------------------------------------------------------------------


def test_research_briefings_round_trip_with_filters(repo):
    old = datetime(2026, 4, 1, tzinfo=timezone.utc)
    new = datetime(2026, 4, 2, tzinfo=timezone.utc)
    repo.insert_research_briefings(
        [
            {
                "briefing_id": "brf_global",
                "created_at": old,
                "ticker": "GLOBAL",
                "category": "global_market",
                "headline": "Global",
                "summary": "Macro update",
                "sources": "[]",
                "trading_mode": "paper",
            },
            {
                "briefing_id": "brf_aapl",
                "created_at": new,
                "ticker": "AAPL",
                "category": "held",
                "headline": "Apple",
                "summary": "Ticker update",
                "sources": "[]",
                "trading_mode": "paper",
            },
            {
                "briefing_id": "brf_live",
                "created_at": new,
                "ticker": "MSFT",
                "category": "held",
                "headline": "Live",
                "summary": "Wrong mode",
                "sources": "[]",
                "trading_mode": "live",
            },
        ]
    )

    all_rows = repo.get_research_briefings(limit=10)
    assert [row["briefing_id"] for row in all_rows] == ["brf_aapl", "brf_global"]

    ticker_rows = repo.get_research_briefings(tickers=["aapl"], limit=10)
    assert [row["briefing_id"] for row in ticker_rows] == ["brf_aapl"]

    category_rows = repo.get_research_briefings(categories=["global_market"], limit=10)
    assert [row["briefing_id"] for row in category_rows] == ["brf_global"]

    live_rows = repo.get_research_briefings(trading_mode="live", limit=10)
    assert [row["briefing_id"] for row in live_rows] == ["brf_live"]


def test_append_runtime_audit_log_uses_bigquery_signature(repo):
    repo.append_runtime_audit_log(
        action="agent_cycle",
        status="warning",
        user_email="User@Example.COM",
        tenant_id="Tenant-A",
        detail={"cycle_id": "cycle_1"},
    )

    rows = repo.recent_runtime_audit_logs(limit=5)
    assert len(rows) == 1
    assert rows[0]["user_email"] == "user@example.com"
    assert rows[0]["tenant_id"] == "tenant-a"
    assert rows[0]["action"] == "agent_cycle"
    assert rows[0]["status"] == "warning"
    assert "cycle_1" in rows[0]["detail_json"]


# ---------------------------------------------------------------------------
# Unimplemented surface — fail loudly
# ---------------------------------------------------------------------------


def test_unimplemented_method_raises_attribute_error_for_hasattr(repo):
    with pytest.raises(AttributeError, match="not implemented yet"):
        repo.not_a_local_method()
    assert hasattr(repo, "not_a_local_method") is False


def test_attribute_lookup_traverses_stores_first(repo):
    # ``recent_memory_events`` is implemented on LocalMemoryStore; the
    # __getattr__ fallback must locate it via _STORE_ATTRS instead of raising.
    bound = repo.recent_memory_events  # must not raise
    assert callable(bound)


# ---------------------------------------------------------------------------
# PR4 write path: memory + execution + sleeve replay
# ---------------------------------------------------------------------------


def test_write_memory_event_round_trips(repo):
    event = MemoryEvent(
        agent_id="gpt",
        event_type="manual_note",
        summary="local memory write",
        payload={"ticker": "AAPL"},
        score=0.7,
        trading_mode="paper",
    )

    repo.write_memory_event(event)

    row = repo.memory_event_by_id(event_id=event.event_id)
    assert row is not None
    assert row["summary"] == "local memory write"
    assert row["payload_json"]
    assert row["score"] == pytest.approx(0.7)


def test_update_memory_score_updates_outcome(repo):
    event = MemoryEvent(agent_id="gpt", event_type="manual_note", summary="score me", score=0.4)
    repo.write_memory_event(event)

    repo.update_memory_score(event.event_id, 0.9)

    row = repo.memory_event_by_id(event_id=event.event_id)
    assert row and row["outcome_score"] == pytest.approx(0.9)


def test_execution_write_and_daily_risk_readers(repo):
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2,
        price_krw=100.0,
        rationale="test",
        created_at=datetime(2026, 4, 28, 1, 0, tzinfo=timezone.utc),
    )
    decision = RiskDecision(allowed=True, reason="ok")
    report = ExecutionReport(
        status=ExecutionStatus.SIMULATED,
        order_id="order-1",
        filled_qty=2,
        avg_price_krw=110.0,
        message="paper fill",
        created_at=datetime(2026, 4, 28, 1, 1, tzinfo=timezone.utc),
    )

    repo.write_order_intent(intent, decision)
    repo.write_execution_report(intent, report)

    assert repo.recent_intent_count(datetime(2026, 4, 28, tzinfo=timezone.utc).date(), agent_id="gpt") == 1
    assert repo.recent_turnover_krw(datetime(2026, 4, 28, tzinfo=timezone.utc).date(), agent_id="gpt") == pytest.approx(220.0)
    assert repo.last_trade_time("AAPL", agent_id="gpt") == report.created_at.replace(tzinfo=None)


def test_recent_trade_history_joins_execution_with_intent_metadata(repo):
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.SELL,
        quantity=2,
        price_krw=100.0,
        rationale="사용자와 투자챗봇이 일부 익절을 판단함",
        strategy_refs=["scope:agent_sleeve", "judgment:user+investment_chat"],
        created_at=datetime(2026, 4, 28, 1, 0, tzinfo=timezone.utc),
    )
    decision = RiskDecision(allowed=True, reason="risk ok", policy_hits=["chat_confirmation"])
    report = ExecutionReport(
        status=ExecutionStatus.SIMULATED,
        order_id="order-history-1",
        filled_qty=2,
        avg_price_krw=110.0,
        message="paper fill",
        created_at=datetime(2026, 4, 28, 1, 1, tzinfo=timezone.utc),
    )
    repo.write_order_intent(intent, decision)
    repo.write_execution_report(intent, report)

    repo.set_tenant_id("tenant-b")
    other_intent = intent.model_copy(update={"intent_id": "tenant-b-intent", "rationale": "other tenant"})
    other_report = report.model_copy(update={"order_id": "tenant-b-order"})
    repo.write_order_intent(other_intent, decision)
    repo.write_execution_report(other_intent, other_report)

    rows = repo.recent_trade_history(
        tenant_id="tenant-a",
        ticker="aapl",
        agent_id="gpt",
        scope="agent_sleeve",
        days=3650,
        limit=10,
        statuses=["SIMULATED"],
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["order_id"] == "order-history-1"
    assert row["ticker"] == "AAPL"
    assert row["rationale"] == "사용자와 투자챗봇이 일부 익절을 판단함"
    assert row["risk_reason"] == "risk ok"
    assert row["policy_hits"] == ["chat_confirmation"]
    assert row["strategy_refs"] == ["scope:agent_sleeve", "judgment:user+investment_chat"]


def test_sleeve_snapshot_replays_simulated_execution(repo):
    repo.ensure_agent_state_checkpoints(
        agent_ids=["gpt"],
        total_cash_krw=1_000.0,
        checkpoint_at=datetime(2026, 4, 28, 0, 0, tzinfo=timezone.utc),
    )
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=2,
        price_krw=100.0,
        rationale="test",
        created_at=datetime(2026, 4, 28, 1, 0, tzinfo=timezone.utc),
    )
    report = ExecutionReport(
        status=ExecutionStatus.SIMULATED,
        order_id="order-2",
        filled_qty=2,
        avg_price_krw=100.0,
        message="paper fill",
        created_at=datetime(2026, 4, 28, 1, 1, tzinfo=timezone.utc),
    )
    repo.write_execution_report(intent, report)

    snapshot, baseline, meta = repo.build_agent_sleeve_snapshot(agent_id="gpt")

    assert baseline == pytest.approx(1_000.0)
    assert snapshot.cash_krw == pytest.approx(800.0)
    assert snapshot.positions["AAPL"].quantity == pytest.approx(2.0)
    assert snapshot.total_equity_krw == pytest.approx(1_000.0)
    assert meta["valuation_source"] == "local_sleeve_replay"
