from __future__ import annotations

import json
import math
from datetime import timedelta

import pytest

from arena.config import Settings
from arena.context import ContextBuilder
from arena.memory.policy import normalize_memory_policy
from arena.models import AccountSnapshot, Position, utc_now


class FakeRepo:
    def __init__(self):
        self.calls: list[list[str]] = []
        self.last_universe_limit: int | None = None
        self.universe_rows: list[str] = ["AAPL", "MSFT"]
        self.ticker_name_rows: dict[str, str] = {}
        self.ticker_name_map_calls: list[tuple[list[str], int]] = []
        self.ticker_memory_rows: list[dict] = []
        self.memory_by_id: dict[str, dict] = {}
        self.research_briefings: list[dict] = []
        self.memory_access_rows: list[dict] = []
        self.graph_neighbors_rows: list[dict] = []
        self.relation_candidate_rows: list[dict] = []
        self.relation_candidate_calls: list[dict] = []
        self.candidate_memory_rows: list[dict] = []
        self.active_thesis_rows: dict[str, dict] = {}

    def resolve_tenant_id(self, tenant_id=None):
        _ = tenant_id
        return "local"

    def latest_market_features(self, tickers, limit, sources=None):
        _ = (limit, sources)
        self.calls.append(list(tickers))
        if tickers == ["PLTD"]:
            return []
        if tickers == ["AAPL", "MSFT"]:
            return [{"ticker": "AAPL", "close_price_krw": 1000}]
        return []

    def latest_universe_candidate_tickers(self, *, limit=200):
        self.last_universe_limit = limit
        return list(self.universe_rows[:limit])

    def ticker_name_map(self, *, tickers=None, limit=500):
        tokens = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
        self.ticker_name_map_calls.append((tokens, limit))
        if not tokens:
            return dict(self.ticker_name_rows)
        return {ticker: self.ticker_name_rows[ticker] for ticker in tokens if ticker in self.ticker_name_rows}

    def recent_intent_count(self, day, agent_id=None, include_simulated=True, trading_mode=None):
        _ = (day, agent_id, include_simulated, trading_mode)
        return 0

    def recent_turnover_krw(self, day, agent_id=None, include_simulated=True, trading_mode=None):
        _ = (day, agent_id, include_simulated, trading_mode)
        return 0.0

    def memory_events_by_ids(self, *, agent_id, event_ids, trading_mode="paper", tenant_id=None):
        _ = (agent_id, trading_mode, tenant_id)
        return [self.memory_by_id[eid] for eid in event_ids if eid in self.memory_by_id]

    def get_research_briefings(self, *, tickers=None, categories=None, limit=10, trading_mode="paper", tenant_id=None):
        _ = (tickers, trading_mode, tenant_id)
        rows = list(self.research_briefings)
        if categories:
            allowed = {str(cat).strip().lower() for cat in categories}
            rows = [row for row in rows if str(row.get("category") or "").strip().lower() in allowed]
        return rows[:limit]

    def append_memory_access_events(self, rows, *, tenant_id=None):
        _ = tenant_id
        self.memory_access_rows.extend(list(rows))

    def active_thesis_events(self, *, agent_id: str, tickers: list[str], trading_mode: str = "paper", tenant_id=None):
        _ = (agent_id, trading_mode, tenant_id)
        return [self.active_thesis_rows[ticker] for ticker in tickers if ticker in self.active_thesis_rows]

    def memory_graph_neighbors(
        self,
        *,
        seed_node_ids,
        trading_mode="paper",
        min_confidence=0.0,
        limit=24,
        tenant_id=None,
    ):
        _ = (seed_node_ids, trading_mode, min_confidence, limit, tenant_id)
        return list(self.graph_neighbors_rows[:limit])

    def memory_relation_memory_candidates(
        self,
        *,
        agent_id,
        seed_node_ids,
        trading_mode="paper",
        min_confidence=0.75,
        limit=8,
        tenant_id=None,
    ):
        self.relation_candidate_calls.append(
            {
                "agent_id": agent_id,
                "seed_node_ids": list(seed_node_ids),
                "trading_mode": trading_mode,
                "min_confidence": min_confidence,
                "limit": limit,
                "tenant_id": tenant_id,
            }
        )
        return list(self.relation_candidate_rows[:limit])

    def candidate_memory_events(
        self,
        *,
        agent_id,
        exclude_tickers=None,
        limit=12,
        trading_mode="paper",
        tenant_id=None,
    ):
        _ = (agent_id, trading_mode, tenant_id)
        blocked = {str(t).strip().upper() for t in (exclude_tickers or []) if str(t).strip()}
        rows = []
        for row in self.candidate_memory_rows:
            payload = row.get("payload_json")
            ticker = ""
            if isinstance(payload, str) and payload.strip():
                try:
                    ticker = str(json.loads(payload).get("ticker") or "").strip().upper()
                except Exception:
                    ticker = ""
            if ticker and ticker in blocked:
                continue
            rows.append(row)
        return rows[:limit]


class FakeMemory:
    def __init__(self, recent_rows=None, top_rows=None, vector_store=None):
        self._recent_rows = list(recent_rows or [])
        self._top_rows = list(top_rows or [])
        self.vector_store = vector_store

    def recent(self, agent_id, limit):
        _ = agent_id
        return self._recent_rows[:limit]

    def top(self, agent_id, limit, lookback_days=120):
        _ = (agent_id, lookback_days)
        return self._top_rows[:limit]


class FakeBoard:
    def recent(self, limit):
        return []


class FakeVectorStore:
    def __init__(self, results_by_query=None):
        self.results_by_query = dict(results_by_query or {})

    def search_similar_memories(self, *, query, **kwargs):
        _ = kwargs
        return list(self.results_by_query.get(query, []))


def _settings() -> Settings:
    return Settings(
        google_cloud_project="p",
        bq_dataset="d",
        bq_location="loc",
        agent_ids=["gpt"],
        agent_mode="adk",
        base_currency="KRW",
        sleeve_capital_krw=2_000_000,
        log_level="INFO",
        log_format="rich",
        trading_mode="paper",
        kis_order_endpoint="",
        kis_api_key="",
        kis_api_secret="",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no="",
        kis_account_product_code="01",
        kis_account_key_suffix="",
        kis_env="real",
        kis_target_market="nasdaq",
        kis_overseas_quote_excd="NAS",
        kis_overseas_order_excd="NASD",
        kis_us_natn_cd="840",
        kis_us_tr_mket_cd="01",
        kis_secret_name="KISAPI",
        kis_secret_version="latest",
        kis_http_timeout_seconds=20,
        kis_http_max_retries=0,
        kis_http_backoff_base_seconds=0.1,
        kis_http_backoff_max_seconds=0.1,
        kis_confirm_fills=False,
        kis_confirm_timeout_seconds=25,
        kis_confirm_poll_seconds=2.0,
        usd_krw_rate=1300.0,
        market_sync_history_days=60,
        max_order_krw=350_000,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=0.35,
        min_cash_buffer_ratio=0.10,
        ticker_cooldown_seconds=120,
        max_daily_orders=0,
        estimated_fee_bps=10.0,
        context_max_board_posts=24,
        context_max_memory_events=32,
        context_max_market_rows=64,
        openai_api_key="",
        openai_model="gpt-5.2",
        gemini_api_key="",
        gemini_model="gemini-3-pro-preview",
        llm_timeout_seconds=25,
        research_gemini_model="gemini-2.5-flash",
        default_universe=["AAPL", "MSFT"],
        allow_live_trading=False,
        autonomy_working_set_enabled=True,
        autonomy_tool_default_candidates_enabled=True,
        autonomy_opportunity_context_enabled=True,
    )


def test_context_builder_falls_back_to_default_universe() -> None:
    repo = FakeRepo()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "PLTD": Position(
                ticker="PLTD",
                quantity=1,
                avg_price_krw=10_000,
                market_price_krw=12_000,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["market_features"]
    assert context["market_features"][0]["ticker"] == "AAPL"
    assert repo.calls == [["PLTD"], ["AAPL", "MSFT"]]
    assert context["risk_policy"]["max_order_krw"] == 350_000
    assert context["risk_policy"]["min_cash_buffer_ratio"] == 0.10
    assert context["risk_policy"]["max_daily_orders"] is None
    assert context["risk_policy"]["max_daily_orders_unlimited"] is True
    assert context["risk_policy"]["sleeve_capital_krw"] == 2_000_000
    assert context["order_budget"]["cash_krw"] == 1_000_000
    assert context["order_budget"]["min_cash_required_krw"] == 120_000
    assert context["order_budget"]["max_buy_notional_by_sleeve_krw"] == 880_000
    assert context["order_budget"]["max_buy_notional_krw"] == 350_000
    assert context["order_budget"]["daily_orders_cap"] is None
    assert context["sleeve_state"]["target_sleeve_krw"] == 2_000_000
    assert context["sleeve_state"]["current_equity_krw"] == 1_200_000
    assert "Positions:" not in context["performance_context"]
    assert "Budget " not in context["performance_context"]
    assert "Daily orders" not in context["performance_context"]
    assert "Cash " not in context["performance_context"]


def test_context_builder_normalizes_market_features_from_raw_daily_closes() -> None:
    class RepoWithRawCloses(FakeRepo):
        def __init__(self):
            super().__init__()
            self.close_sources = None

        def latest_market_features(self, tickers, limit, sources=None):
            _ = (limit, sources)
            self.calls.append(list(tickers))
            return [
                {
                    "ticker": "AAPL",
                    "close_price_krw": 1000.0,
                    "ret_5d": 0.0,
                    "ret_20d": 0.0,
                    "volatility_20d": 0.0,
                }
            ]

        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            _ = (tickers, lookback_days)
            self.close_sources = list(sources or [])
            return {"AAPL": [100.0 + idx for idx in range(22)]}

    settings = _settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "nasdaq"
    repo = RepoWithRawCloses()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=10_000,
                market_price_krw=12_000,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    row = context["market_features"][0]
    closes = [100.0 + idx for idx in range(22)]
    assert math.isclose(row["ret_5d"], (closes[-1] / closes[-6]) - 1.0)
    assert math.isclose(row["ret_20d"], (closes[-1] / closes[-21]) - 1.0)
    assert row["volatility_20d"] > 0.0
    assert repo.close_sources == ["open_trading_nasdaq", "open_trading_us"]


def test_context_builder_loads_ticker_names_for_current_positions() -> None:
    repo = FakeRepo()
    repo.ticker_name_rows = {"025860": "남해화학"}
    settings = _settings()
    settings.kis_target_market = "kospi"
    settings.default_universe = []
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "025860": Position(
                ticker="025860",
                quantity=2,
                avg_price_krw=8_000,
                market_price_krw=8_270,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert repo.ticker_name_map_calls
    assert repo.ticker_name_map_calls[-1][0] == ["025860"]
    assert context["ticker_names"]["025860"] == "남해화학"
    assert context["performance"]["positions"][0]["ticker_name"] == "남해화학"
    assert context["sleeve_state"]["sleeve_remaining_krw"] == 880_000
    assert context["sleeve_state"]["over_target"] is False
    assert context["sleeve_state"]["buy_blocked"] is False
    assert "Long-horizon compounding" in context["investment_style_context"]


def test_context_builder_uses_krw_display_when_us_fx_is_unavailable() -> None:
    repo = FakeRepo()
    settings = _settings()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        usd_krw_rate=0.0,
        cash_foreign=500.0,
        cash_foreign_currency="USD",
        positions={},
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["performance"]["display_currency"] == "KRW"
    assert "usd_krw_rate" not in context["order_budget"]
    assert context["order_budget"]["cash_usd"] == pytest.approx(500.0)


def test_context_builder_falls_back_to_runtime_universe_candidates() -> None:
    repo = FakeRepo()
    settings = _settings()
    settings.default_universe = []
    settings.universe_run_top_n = 2
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "PLTD": Position(
                ticker="PLTD",
                quantity=1,
                avg_price_krw=10_000,
                market_price_krw=12_000,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["market_features"]
    assert context["market_features"][0]["ticker"] == "AAPL"
    assert repo.calls == [["PLTD"], ["AAPL", "MSFT"]]
    assert repo.last_universe_limit == 2


def test_context_builder_includes_active_thesis_context_for_holdings() -> None:
    repo = FakeRepo()
    repo.active_thesis_rows["AAPL"] = {
        "event_id": "mem_thesis",
        "event_type": "thesis_update",
        "summary": "AAPL thesis update action=add status=FILLED thesis=AI demand and margin recovery",
        "payload_json": json.dumps(
            {
                "thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_open",
                "ticker": "AAPL",
                "state": "active",
                "thesis_summary": "AI demand and margin recovery",
                "strategy_refs": ["momentum", "earnings_growth"],
            }
        ),
    }
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=2,
                avg_price_krw=100_000,
                market_price_krw=105_000,
            )
        },
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert "Active Thesis:" in context["active_thesis_context"]
    assert "AAPL" in context["active_thesis_context"]
    assert "AI demand and margin recovery" in context["active_thesis_context"]
    assert context["active_theses"][0]["event_type"] == "thesis_update"


def test_context_builder_skips_initial_memory_without_vector_candidates() -> None:
    repo = FakeRepo()
    memory = FakeMemory(
        recent_rows=[
            {
                "event_id": "evt_recent",
                "created_at": "2026-02-18T00:00:00Z",
                "agent_id": "gpt",
                "event_type": "trade_execution",
                "summary": "최근 체결 요약",
                "score": 0.2,
                "payload_json": '{"x":1}',
            }
        ],
        top_rows=[
            {
                "event_id": "evt_top",
                "created_at": "2026-01-10T00:00:00Z",
                "agent_id": "gpt",
                "event_type": "react_tools_summary",
                "summary": "장기적으로 중요했던 도구 사용 패턴",
                "score": 1.0,
                "payload_json": '{"x":3}',
            },
        ],
    )
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"] == []
    assert context["memory_context"] == ""
    assert "Long-horizon compounding" in context["investment_style_context"] or "low turnover" in context["investment_style_context"]


def test_context_builder_skips_vector_rows_without_event_ids() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"summary": "orphan vector row", "score": 0.8, "created_at": "2026-02-22T00:00:00Z"},
                ]
            }
        ),
    )
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"] == []
    assert context["memory_context"] == ""


def test_context_builder_builds_opportunity_query_for_high_cash_state() -> None:
    builder = ContextBuilder(repo=FakeRepo(), memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    snapshot = AccountSnapshot(
        cash_krw=800_000,
        total_equity_krw=1_000_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=200_000,
            )
        },
    )

    query = builder._build_opportunity_memory_query(snapshot)

    assert query is not None
    assert "new entry opportunity" in query
    assert "opportunity cost compare" in query


def test_context_builder_merge_memory_tracks_reserves_opportunity_slots() -> None:
    builder = ContextBuilder(repo=FakeRepo(), memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    primary_rows = [
        {"event_id": f"h{i}", "retrieval_score": 1.0 - (i * 0.01), "importance_score": 0.8}
        for i in range(5)
    ]
    opportunity_rows = [
        {"event_id": "o1", "retrieval_score": 0.10, "importance_score": 0.3},
        {"event_id": "o2", "retrieval_score": 0.09, "importance_score": 0.2},
    ]

    merged = builder._merge_memory_query_tracks(
        primary_rows=primary_rows,
        opportunity_rows=opportunity_rows,
        total_limit=6,
    )

    assert [row["event_id"] for row in merged[:6]] == ["h0", "h1", "h2", "h3", "o1", "o2"]


def test_context_builder_compresses_memories_into_typed_sections() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_reflect", "summary": "Momentum reflection", "score": 0.6, "created_at": "2026-02-20T00:00:00Z"},
                    {"event_id": "evt_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": "2026-02-22T00:00:00Z"},
                    {"event_id": "evt_note", "summary": "AAPL support note", "score": 0.5, "created_at": "2026-02-23T00:00:00Z"},
                ]
            }
        ),
    )
    repo.memory_by_id = {
        "evt_reflect": {
            "event_id": "evt_reflect",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "Momentum chase after earnings spikes usually fades.",
            "importance_score": 0.6,
            "score": 0.6,
            "payload_json": "{}",
        },
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": "2026-02-22T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "outcome_score": 0.8,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
        "evt_note": {
            "event_id": "evt_note",
            "created_at": "2026-02-23T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL held support while semis rolled over.",
            "importance_score": 0.5,
            "score": 0.5,
            "payload_json": '{"source":"manual_note"}',
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert "Neutral Lessons:" in context["memory_context"]
    assert "Momentum chase" in context["memory_context"]
    assert "AAPL BUY" in context["memory_context"]
    assert "status=FILLED" not in context["memory_context"]
    assert "broker=filled" not in context["memory_context"]
    assert "AAPL held support" in context["memory_context"]


def test_context_builder_reserves_candidate_memory_track() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(
        cash_krw=500_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=120_000,
            )
        },
    )
    settings = _settings()
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    market_rows = repo.latest_market_features(["AAPL"], limit=settings.context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=settings.context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    opportunity_query = seed_builder._build_opportunity_memory_query(snapshot)
    assert opportunity_query is not None
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_aapl_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": "2026-02-22T00:00:00Z"},
                    {"event_id": "evt_aapl_note", "summary": "AAPL risk", "score": 0.6, "created_at": "2026-02-23T00:00:00Z"},
                    {"event_id": "evt_aapl_extra", "summary": "AAPL extra", "score": 0.5, "created_at": "2026-02-24T00:00:00Z"},
                ],
                opportunity_query: [
                    {"event_id": "evt_neutral", "summary": "avoid one-day breakouts", "score": 0.6, "created_at": "2026-02-21T00:00:00Z"},
                ],
            }
        ),
    )
    repo.memory_by_id = {
        "evt_aapl_trade": {
            "event_id": "evt_aapl_trade",
            "created_at": "2026-02-22T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
        "evt_aapl_note": {
            "event_id": "evt_aapl_note",
            "created_at": "2026-02-23T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL valuation risk note.",
            "importance_score": 0.6,
            "score": 0.6,
            "payload_json": '{"ticker":"AAPL"}',
        },
        "evt_aapl_extra": {
            "event_id": "evt_aapl_extra",
            "created_at": "2026-02-24T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL duplicate exposure note.",
            "importance_score": 0.5,
            "score": 0.5,
            "payload_json": '{"ticker":"AAPL"}',
        },
        "evt_neutral": {
            "event_id": "evt_neutral",
            "created_at": "2026-02-21T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "Avoid treating one-day breakouts as confirmation.",
            "importance_score": 0.6,
            "score": 0.6,
            "payload_json": "{}",
        },
    }
    repo.candidate_memory_rows = [
        {
            "event_id": "cand_msft",
            "created_at": "2026-02-24T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "candidate_screen_hit",
            "summary": "MSFT candidate_screen_hit: surfaced by screen_market:value; evidence is screen-only.",
            "importance_score": 0.25,
            "score": 0.25,
            "payload_json": '{"source":"candidate_discovery","ticker":"MSFT","evidence_level":"screened_only"}',
        }
    ]
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert "Portfolio Memory:" in context["memory_context"]
    assert "Candidate Memory:" in context["memory_context"]
    assert "Neutral Lessons:" in context["memory_context"]
    assert any(row["event_id"] == "cand_msft" and row["memory_track"] == "candidate" for row in context["memory_events"])
    assert sum(1 for row in context["memory_events"] if "AAPL" in row.get("tickers", [])) <= 2


def test_context_builder_appends_graph_decision_paths_when_enabled() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    settings = _settings()
    settings.memory_policy = normalize_memory_policy({"graph": {"enabled": True, "max_expansion_hops": 1, "max_expanded_nodes": 6}})
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [{"event_id": "evt_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": "2026-02-22T00:00:00Z"}]
            }
        ),
    )
    repo.memory_by_id = {
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": "2026-02-22T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "score": 0.7,
            "graph_node_id": "mem:evt_trade",
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY","intent_id":"intent_1"}}',
        }
    }
    repo.graph_neighbors_rows = [
        {
            "seed_node_id": "mem:evt_trade",
            "direction": "incoming",
            "neighbor_node_id": "intent:intent_1",
            "edge_id": "edge:precedes:intent:intent_1:evt_trade",
            "edge_created_at": "2026-02-22T00:00:00Z",
            "edge_type": "PRECEDES",
            "edge_strength": 0.9,
            "confidence": 1.0,
            "node_created_at": "2026-02-22T00:00:00Z",
            "node_kind": "order_intent",
            "source_table": "agent_order_intents",
            "source_id": "intent_1",
            "agent_id": "gpt",
            "node_trading_mode": "paper",
            "cycle_id": "cycle_1",
            "summary": "BUY AAPL qty=2 rationale=setup",
            "ticker": "AAPL",
        },
        {
            "seed_node_id": "mem:evt_trade",
            "direction": "incoming",
            "neighbor_node_id": "exec:ord_1",
            "edge_id": "edge:resulted_in:exec:ord_1:evt_trade",
            "edge_created_at": "2026-02-22T00:00:01Z",
            "edge_type": "RESULTED_IN",
            "edge_strength": 1.0,
            "confidence": 1.0,
            "node_created_at": "2026-02-22T00:00:01Z",
            "node_kind": "execution_report",
            "source_table": "execution_reports",
            "source_id": "ord_1",
            "agent_id": "gpt",
            "node_trading_mode": "paper",
            "cycle_id": "cycle_1",
            "summary": "FILLED BUY AAPL filled=2.0000 avg=100000",
            "ticker": "AAPL",
        },
    ]
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert "Decision Paths:" not in context["memory_context"]
    assert "preceded by order intent AAPL" not in context["memory_context"]
    assert "resulted from execution report AAPL" not in context["memory_context"]
    assert context["graph_context"].startswith("Decision Paths:")
    assert "preceded by order intent AAPL" in context["graph_context"]
    assert "resulted from execution report AAPL" in context["graph_context"]
    assert "qty=2" not in context["graph_context"]
    assert "filled=2.0000" not in context["graph_context"]
    assert context["graph_events"]
    assert "2026-02-22" in context["memory_context"]


def test_context_builder_ticker_display_uses_context_tags_and_avoids_plain_words() -> None:
    repo = FakeRepo()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())

    reflection = builder._normalize_memory_row(
        {
            "event_id": "evt_reflect",
            "created_at": "2026-03-10T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "Momentum breakouts in bull/low-vol tech regimes work best when breadth confirms.",
            "importance_score": 0.8,
            "score": 0.8,
            "context_tags_json": {"tickers": ["AAPL"], "regimes": ["bull"], "strategies": ["momentum"]},
        }
    )
    note = builder._normalize_memory_row(
        {
            "event_id": "evt_note",
            "created_at": "2026-03-12T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL liquidity looked fake around the open; wait for second push confirmation.",
            "importance_score": 0.5,
            "score": 0.5,
        }
    )

    assert reflection["tickers"] == ["AAPL"]
    assert reflection["canonical_tickers"] == ["AAPL"]
    assert reflection["derived_tickers"] == []
    assert reflection["ticker_source"] == "context_tags"
    assert "IN" not in reflection["tickers"]
    assert "BULL" not in reflection["tickers"]
    assert note["tickers"] == ["AAPL"]
    assert note["canonical_tickers"] == []
    assert note["derived_tickers"] == ["AAPL"]
    assert note["ticker_source"] == "summary_regex"
    assert note["side"] == ""


def test_context_builder_keeps_summary_fallback_derived_but_out_of_ticker_bonus() -> None:
    repo = FakeRepo()
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())

    derived = builder._normalize_memory_row(
        {
            "event_id": "evt_note",
            "created_at": "2026-03-12T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "AAPL liquidity looked fake around the open; wait for confirmation.",
            "importance_score": 0.5,
            "score": 0.5,
        }
    )
    canonical = builder._normalize_memory_row(
        {
            "event_id": "evt_trade",
            "created_at": "2026-03-12T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.5,
            "score": 0.5,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        }
    )

    assert builder._memory_ticker_bonus(derived, {"AAPL"}) == 0.0
    assert builder._memory_ticker_bonus(canonical, {"AAPL"}) > 0.0
    assert "~AAPL" in builder._format_memory_line(derived)
    assert "~BUY" not in builder._format_memory_line(derived)
    canonical_line = builder._format_memory_line(canonical)
    assert "prior entry" in canonical_line
    assert "status=FILLED" not in canonical_line


def test_context_builder_uses_temporal_tiers_when_hierarchy_enabled() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {"hierarchy": {"enabled": True, "working_ttl_hours": 24, "episodic_ttl_days": 60}}
    )
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    now = utc_now()
    working_created = (now - timedelta(hours=3)).isoformat()
    reflection_created = (now - timedelta(days=6)).isoformat()
    trade_created = (now - timedelta(days=4)).isoformat()
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_working", "summary": "tool trace", "score": 0.8, "created_at": working_created},
                    {"event_id": "evt_reflect", "summary": "reflection", "score": 0.6, "created_at": reflection_created},
                    {"event_id": "evt_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": trade_created},
                ]
            }
        ),
    )
    repo.memory_by_id = {
        "evt_working": {
            "event_id": "evt_working",
            "created_at": working_created,
            "agent_id": "gpt",
            "event_type": "react_tools_summary",
            "memory_tier": "working",
            "summary": "Technical signals and screen_market were called repeatedly.",
            "importance_score": 0.8,
            "score": 0.8,
            "payload_json": '{"tool_events":[{"tool":"technical_signals"}]}',
        },
        "evt_reflect": {
            "event_id": "evt_reflect",
            "created_at": reflection_created,
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "memory_tier": "semantic",
            "summary": "Avoid chasing weak breadth breakouts without confirmation.",
            "importance_score": 0.6,
            "score": 0.6,
            "payload_json": "{}",
        },
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": trade_created,
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "memory_tier": "episodic",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "outcome_score": 0.8,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert [row["event_id"] for row in context["memory_events"]] == ["evt_reflect", "evt_trade"]
    assert "Neutral Lessons:" in context["memory_context"]
    assert "Avoid chasing weak breadth" in context["memory_context"]
    assert "AAPL BUY" in context["memory_context"]
    assert "status=FILLED" not in context["memory_context"]
    assert "Past Lessons:" not in context["memory_context"]
    assert "Technical signals and screen_market" not in context["memory_context"]


def test_context_builder_prefers_memories_with_matching_context_tags() -> None:
    class TagRepo(FakeRepo):
        def latest_market_features(self, tickers, limit, sources=None):
            _ = (tickers, limit, sources)
            return [
                {
                    "ticker": "AAPL",
                    "close_price_krw": 1000,
                    "ret_20d": 0.14,
                    "ret_5d": 0.04,
                    "volatility_20d": 0.10,
                }
            ]

    repo = TagRepo()
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=120_000,
            )
        },
    )
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {
            "tagging": {
                "enabled": True,
                "regime_bonus": 0.35,
                "strategy_bonus": 0.25,
                "sector_bonus": 0.15,
            }
        }
    )
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, repo.latest_market_features(["AAPL"], limit=8))
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_miss", "summary": "mismatch", "score": 0.7, "created_at": "2026-02-23T00:00:00Z"},
                    {"event_id": "evt_match", "summary": "match", "score": 0.7, "created_at": "2026-02-23T00:00:00Z"},
                ]
            }
        ),
    )
    repo.memory_by_id = {
        "evt_miss": {
            "event_id": "evt_miss",
            "created_at": "2026-02-23T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
            "primary_regime": "bear",
            "primary_strategy_tag": "mean_reversion",
            "primary_sector": "energy",
            "context_tags_json": '{"regimes":["bear"],"strategies":["mean_reversion"],"sectors":["energy"],"tickers":["AAPL"]}',
        },
        "evt_match": {
            "event_id": "evt_match",
            "created_at": "2026-02-23T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
            "primary_regime": "bull",
            "primary_strategy_tag": "breakout",
            "primary_sector": "tech",
            "context_tags_json": '{"regimes":["bull","low_vol"],"strategies":["momentum","breakout"],"sectors":["tech"],"tickers":["AAPL"]}',
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_match"
    assert context["memory_events"][0]["retrieval_score"] > context["memory_events"][1]["retrieval_score"]


def test_context_builder_prefers_memories_with_stronger_effective_score_when_bonus_enabled() -> None:
    repo = FakeRepo()
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=120_000,
            )
        },
    )
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {
            "forgetting": {"enabled": True},
            "cleanup": {"min_score": 0.30},
            "retrieval": {
                "reranking": {
                    "effective_score_bonus_scale": 0.12,
                    "effective_score_bonus_cap": 0.12,
                }
            },
        }
    )
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    market_rows = repo.latest_market_features(["AAPL"], limit=settings.context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=settings.context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_low", "summary": "low effective", "score": 0.7, "created_at": "2026-02-23T00:00:00Z"},
                    {"event_id": "evt_high", "summary": "high effective", "score": 0.7, "created_at": "2026-02-23T00:00:00Z"},
                ]
            }
        ),
    )
    base_trade = {
        "created_at": "2026-02-23T00:00:00Z",
        "agent_id": "gpt",
        "event_type": "trade_execution",
        "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
        "importance_score": 0.7,
        "score": 0.7,
        "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
    }
    repo.memory_by_id = {
        "evt_low": {
            "event_id": "evt_low",
            **base_trade,
            "effective_score": 0.32,
        },
        "evt_high": {
            "event_id": "evt_high",
            **base_trade,
            "effective_score": 0.90,
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_high"
    assert context["memory_events"][0]["effective_score"] == 0.9
    assert context["memory_events"][0]["retrieval_score"] > context["memory_events"][1]["retrieval_score"]


def test_context_builder_logs_memory_access_when_forgetting_enabled() -> None:
    repo = FakeRepo()
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {"forgetting": {"enabled": True, "access_log_enabled": True}}
    )
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_200_000, positions={})
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, [])
    memory = FakeMemory(
        recent_rows=[],
        vector_store=FakeVectorStore(
            results_by_query={
                queries[0]: [
                    {"event_id": "evt_reflect", "summary": "reflection", "score": 0.6, "created_at": "2026-02-20T00:00:00Z"},
                    {"event_id": "evt_trade", "summary": "trade", "score": 0.7, "created_at": "2026-02-22T00:00:00Z"},
                ]
            }
        ),
    )
    repo.memory_by_id = {
        "evt_reflect": {
            "event_id": "evt_reflect",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "Favor patient entries when breadth is narrowing.",
            "importance_score": 0.6,
            "score": 0.6,
            "payload_json": "{}",
        },
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": "2026-02-22T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.7,
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
    }
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot, cycle_id="cycle_access")

    assert len(context["memory_events"]) == 2
    assert len(repo.memory_access_rows) == 2
    assert all(row["access_type"] == "retrieval" for row in repo.memory_access_rows)
    assert all(row["used_in_prompt"] is True for row in repo.memory_access_rows)
    assert all(row["cycle_id"] == "cycle_access" for row in repo.memory_access_rows)


def test_context_builder_builds_environment_queries_from_research_briefings() -> None:
    repo = FakeRepo()
    repo.research_briefings = [
        {
            "category": "global_market",
            "headline": "Sticky inflation keeps higher-for-longer rates in play",
            "summary": "Bond yields remain elevated and broad risk appetite is fragile.",
        },
        {
            "category": "geopolitical",
            "headline": "Shipping disruptions lift energy and logistics risk",
            "summary": "Geopolitical tension is pushing oil and freight volatility higher.",
        },
    ]
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    snapshot = AccountSnapshot(
        cash_krw=500_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=120_000,
            )
        },
    )

    queries = builder._build_memory_search_queries("gpt", snapshot, [])

    assert any("portfolio state" in query for query in queries)
    assert any("macro regime" in query and "higher-for-longer" in query for query in queries)
    assert any("geopolitical risk" in query and "Shipping disruptions" in query for query in queries)


def test_context_builder_hydrates_vector_hits_and_prefers_ticker_overlap() -> None:
    repo = FakeRepo()
    repo.memory_by_id = {
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=3 status=FILLED policy=ok broker=filled",
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
        "evt_reflect": {
            "event_id": "evt_reflect",
            "created_at": "2026-02-24T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "Avoid averaging down on broken cyclicals.",
            "score": 0.4,
            "payload_json": "{}",
        },
    }
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=120_000,
            )
        },
    )
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    market_rows = repo.latest_market_features(["AAPL"], limit=_settings().context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=_settings().context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    vector_store = FakeVectorStore(
        results_by_query={
            queries[0]: [
                {"event_id": "evt_trade", "summary": "AAPL BUY", "score": 0.7, "created_at": "2026-02-20T00:00:00Z"},
                {"event_id": "evt_reflect", "summary": "cyclicals", "score": 0.4, "created_at": "2026-02-24T00:00:00Z"},
            ]
        }
    )
    memory = FakeMemory(recent_rows=[], vector_store=vector_store)
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_trade"
    assert context["memory_events"][0]["tickers"] == ["AAPL"]
    assert context["memory_events"][0]["canonical_tickers"] == ["AAPL"]
    assert context["memory_events"][0]["derived_tickers"] == []
    assert context["memory_events"][0]["side"] == "BUY"
    assert context["memory_events"][0]["canonical_side"] == "BUY"
    assert context["memory_events"][0]["derived_side"] == ""


def test_context_builder_falls_back_to_raw_vector_rows_when_bq_hydration_fails() -> None:
    class FailingHydrateRepo(FakeRepo):
        def memory_events_by_ids(self, *, agent_id, event_ids, trading_mode="paper", tenant_id=None):
            _ = (agent_id, event_ids, trading_mode, tenant_id)
            raise RuntimeError("bq unavailable")

    repo = FailingHydrateRepo()
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=120_000,
            )
        },
    )
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    market_rows = repo.latest_market_features(["AAPL"], limit=_settings().context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=_settings().context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    vector_store = FakeVectorStore(
        results_by_query={
            queries[0]: [
                {
                    "event_id": "evt_trade",
                    "agent_id": "gpt",
                    "event_type": "trade_execution",
                    "summary": "AAPL BUY qty=3 status=FILLED policy=ok broker=filled",
                    "score": 0.7,
                    "created_at": "2026-02-20T00:00:00Z",
                }
            ]
        }
    )
    memory = FakeMemory(recent_rows=[], vector_store=vector_store)
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_trade"
    assert context["memory_events"][0]["event_type"] == "trade_execution"
    assert context["memory_events"][0]["tickers"] == ["AAPL"]
    assert context["memory_events"][0]["canonical_tickers"] == []
    assert context["memory_events"][0]["derived_tickers"] == ["AAPL"]
    assert context["memory_events"][0]["ticker_source"] == "summary_regex"
    assert context["memory_events"][0]["side"] == "BUY"
    assert context["memory_events"][0]["canonical_side"] == ""
    assert context["memory_events"][0]["derived_side"] == "BUY"
    assert context["memory_events"][0]["side_source"] == "summary_keyword"
    assert "Portfolio Memory:" in context["memory_context"]
    assert "~AAPL" in context["memory_context"]
    assert "~BUY" in context["memory_context"]
    assert "prior entry" in context["memory_context"]
    assert "qty=3" not in context["memory_context"]


def test_context_builder_returns_empty_memory_when_state_query_has_no_vector_hits() -> None:
    repo = FakeRepo()
    repo.memory_by_id = {
        "evt_trade": {
            "event_id": "evt_trade",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=3 status=FILLED policy=ok broker=filled",
            "score": 0.7,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY"}}',
        },
    }
    repo.ticker_memory_rows = [
        {
            "event_id": "evt_bq_only",
            "created_at": "2026-02-24T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "manual_note",
            "summary": "Fallback-only BQ row should not appear when vector works.",
            "score": 0.9,
            "payload_json": "{}",
        }
    ]
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(
                ticker="AAPL",
                quantity=1,
                avg_price_krw=100_000,
                market_price_krw=120_000,
            )
        },
    )
    seed_builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=_settings())
    market_rows = repo.latest_market_features(["AAPL"], limit=_settings().context_max_market_rows)
    if not market_rows:
        market_rows = repo.latest_market_features(["AAPL", "MSFT"], limit=_settings().context_max_market_rows)
    queries = seed_builder._build_memory_search_queries("gpt", snapshot, market_rows)
    vector_store = FakeVectorStore(results_by_query={queries[0]: []})
    memory = FakeMemory(
        recent_rows=[
            {
                "event_id": "evt_recent_only",
                "created_at": "2026-02-25T00:00:00Z",
                "agent_id": "gpt",
                "event_type": "manual_note",
                "summary": "Recent-only fallback row should stay out.",
                "score": 1.0,
                "payload_json": "{}",
            }
        ],
        vector_store=vector_store,
    )
    builder = ContextBuilder(repo=repo, memory=memory, board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)
    assert context["memory_events"] == []
    assert context["memory_context"] == ""


def test_relation_triples_shadow_mode_does_not_affect_retrieval() -> None:
    repo = FakeRepo()
    repo.relation_candidate_rows = [
        {
            "event_id": "evt_relation",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "AAPL relation candidate should stay shadowed.",
            "score": 0.9,
            "importance_score": 0.9,
            "payload_json": "{}",
            "relation_predicate": "contains",
            "relation_object_type": "ticker",
            "relation_object_label": "AAPL",
            "relation_confidence": 0.95,
            "relation_evidence_text": "AAPL relation candidate should stay shadowed.",
        }
    ]
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=1, avg_price_krw=100_000, market_price_krw=120_000)
        },
    )
    builder = ContextBuilder(repo=repo, memory=FakeMemory(vector_store=FakeVectorStore()), board=FakeBoard(), settings=_settings())

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert repo.relation_candidate_calls == []
    assert context["memory_events"] == []
    assert context["relation_context"] == ""


def test_relation_triples_boost_mode_adds_relation_candidates() -> None:
    repo = FakeRepo()
    repo.relation_candidate_rows = [
        {
            "event_id": "evt_relation",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "AAPL relation candidate captures a prior risk lesson.",
            "score": 0.5,
            "importance_score": 0.5,
            "payload_json": "{}",
            "relation_predicate": "contains",
            "relation_object_type": "ticker",
            "relation_object_label": "AAPL",
            "relation_confidence": 0.9,
            "relation_evidence_text": "AAPL relation candidate captures a prior risk lesson.",
        }
    ]
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {"graph": {"semantic_triples": {"mode": "boost", "min_confidence": 0.8, "boost_bonus_base": 0.2}}}
    )
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=1, avg_price_krw=100_000, market_price_krw=120_000)
        },
    )
    builder = ContextBuilder(repo=repo, memory=FakeMemory(vector_store=FakeVectorStore()), board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert repo.relation_candidate_calls
    assert "ticker:AAPL" in repo.relation_candidate_calls[0]["seed_node_ids"]
    assert [row["event_id"] for row in context["memory_events"]] == ["evt_relation"]
    assert context["memory_events"][0]["relation_boost"] == pytest.approx(0.18)
    assert context["relation_context"] == ""


def test_relation_triples_inject_mode_adds_relation_context() -> None:
    repo = FakeRepo()
    repo.relation_candidate_rows = [
        {
            "event_id": "evt_relation",
            "created_at": "2026-02-20T00:00:00Z",
            "agent_id": "gpt",
            "event_type": "strategy_reflection",
            "summary": "AAPL relation candidate captures a prior risk lesson.",
            "score": 0.5,
            "importance_score": 0.5,
            "payload_json": "{}",
            "relation_predicate": "contains",
            "relation_object_type": "ticker",
            "relation_object_label": "AAPL",
            "relation_confidence": 0.9,
            "relation_evidence_text": "AAPL relation candidate captures a prior risk lesson.",
        }
    ]
    settings = _settings()
    settings.memory_policy = normalize_memory_policy(
        {"graph": {"semantic_triples": {"mode": "inject", "max_relation_context_items": 2}}}
    )
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={
            "AAPL": Position(ticker="AAPL", quantity=1, avg_price_krw=100_000, market_price_krw=120_000)
        },
    )
    builder = ContextBuilder(repo=repo, memory=FakeMemory(vector_store=FakeVectorStore()), board=FakeBoard(), settings=settings)

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["memory_events"][0]["event_id"] == "evt_relation"
    assert context["relation_context"].startswith("Relation Hints:")
    assert "contains ticker AAPL" in context["relation_context"]
def test_context_builder_live_mode_raises_when_risk_metrics_unavailable() -> None:
    class FailingRiskRepo(FakeRepo):
        def recent_intent_count(self, day, agent_id=None, include_simulated=True, trading_mode=None):
            _ = (day, agent_id, include_simulated, trading_mode)
            raise RuntimeError("risk metrics unavailable")

    repo = FailingRiskRepo()
    settings = _settings()
    settings.trading_mode = "live"
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_100_000, positions={})

    with pytest.raises(RuntimeError, match="recent_intent_count"):
        builder.build(agent_id="gpt", snapshot=snapshot)


def test_context_builder_no_cap_gap_block_when_nav_exceeds_target() -> None:
    """NAV > target no longer blocks buying — only cash matters."""
    repo = FakeRepo()
    settings = _settings()
    settings.sleeve_capital_krw = 1_000_000
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=1_000_000,
        total_equity_krw=1_200_000,
        positions={},
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["sleeve_state"]["over_target"] is False
    assert context["sleeve_state"]["buy_blocked"] is False
    assert context["order_budget"]["max_buy_notional_by_sleeve_krw"] == 880_000
    assert context["order_budget"]["max_buy_notional_krw"] == 350_000


def test_context_builder_uses_per_agent_capital_for_sleeve_target() -> None:
    repo = FakeRepo()
    settings = _settings()
    settings.sleeve_capital_krw = 1_000_000
    settings.agent_capitals = {"gpt": 500_000, "gemini": 1_500_000}
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(cash_krw=400_000, total_equity_krw=600_000, positions={})

    # gpt target=500_000, equity=600_000 → no longer blocked
    gpt_ctx = builder.build(agent_id="gpt", snapshot=snapshot)
    assert gpt_ctx["sleeve_state"]["target_sleeve_krw"] == 500_000
    assert gpt_ctx["sleeve_state"]["over_target"] is False

    # gemini target=1_500_000, equity=600_000
    gemini_ctx = builder.build(agent_id="gemini", snapshot=snapshot)
    assert gemini_ctx["sleeve_state"]["target_sleeve_krw"] == 1_500_000
    assert gemini_ctx["sleeve_state"]["over_target"] is False

    # unknown agent falls back to sleeve_capital_krw
    unknown_ctx = builder.build(agent_id="unknown_agent", snapshot=snapshot)
    assert unknown_ctx["sleeve_state"]["target_sleeve_krw"] == 1_000_000


def test_context_builder_caps_buy_budget_by_cash_buffer() -> None:
    repo = FakeRepo()
    settings = _settings()
    settings.sleeve_capital_krw = 1_000_000
    builder = ContextBuilder(repo=repo, memory=FakeMemory(), board=FakeBoard(), settings=settings)
    snapshot = AccountSnapshot(
        cash_krw=100_000,
        total_equity_krw=950_000,
        positions={},
    )

    context = builder.build(agent_id="gpt", snapshot=snapshot)

    assert context["sleeve_state"]["sleeve_remaining_krw"] == 5_000
    assert context["order_budget"]["max_buy_notional_by_sleeve_krw"] == 5_000
    assert context["order_budget"]["max_buy_notional_krw"] == 5_000
