from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

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
        self.macro_research_theses: list[dict] = []
        self.macro_research_thesis_calls: list[dict] = []
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
        _ = (trading_mode, tenant_id)
        rows = list(self.research_briefings)
        filters = []
        if tickers:
            allowed_tickers = {str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()}
            filters.append(lambda row: str(row.get("ticker") or "").strip().upper() in allowed_tickers)
        if categories:
            allowed = {str(cat).strip().lower() for cat in categories}
            filters.append(lambda row: str(row.get("category") or "").strip().lower() in allowed)
        if filters:
            rows = [row for row in rows if any(check(row) for check in filters)]
        return rows[:limit]

    def get_macro_research_theses(
        self,
        *,
        source_doc_ids=None,
        themes=None,
        market=None,
        status="active",
        since=None,
        limit=10,
        tenant_id=None,
    ):
        _ = (source_doc_ids, since, tenant_id)
        self.macro_research_thesis_calls.append(
            {
                "themes": themes,
                "market": market,
                "status": status,
                "limit": limit,
            }
        )
        rows = list(self.macro_research_theses)
        clean_market = str(market or "").strip().lower()
        if clean_market and clean_market != "all":
            rows = [
                row for row in rows
                if str(row.get("market") or "").strip().lower() in {clean_market, "all"}
            ]
        if status:
            clean_status = str(status or "").strip().lower()
            rows = [row for row in rows if str(row.get("status") or "").strip().lower() == clean_status]
        if themes:
            allowed = {str(theme).strip().lower() for theme in themes if str(theme).strip()}
            rows = [row for row in rows if str(row.get("theme_key") or "").strip().lower() in allowed]
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
