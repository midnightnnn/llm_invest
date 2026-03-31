from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from arena.board.store import BoardStore
from arena.config import AgentConfig, Settings, merge_agent_risk_settings
from arena.data.bq import BigQueryRepository
from arena.market_sources import (
    KOSPI_MARKETS,
    US_MARKETS,
    live_market_sources_for_markets,
    parse_markets,
)
from arena.memory.forgetting import effective_memory_score
from arena.memory.graph import memory_event_node_id
from arena.memory.policy import (
    get_memory_policy_value,
    memory_event_enabled,
    memory_forgetting_access_curve,
    memory_forgetting_access_log_enabled,
    memory_forgetting_default_decay_factor,
    memory_forgetting_enabled,
    memory_forgetting_min_effective_score,
    memory_forgetting_tier_weight,
    memory_graph_enabled,
    memory_graph_inferred_edge_min_confidence,
    memory_graph_max_expanded_nodes,
    memory_graph_max_expansion_hops,
    memory_hierarchy_enabled,
    memory_hierarchy_episodic_ttl_days,
    memory_hierarchy_working_ttl_hours,
    memory_tagging_enabled,
    memory_tagging_max_tags,
    memory_tagging_regime_bonus,
    memory_tagging_sector_bonus,
    memory_tagging_strategy_bonus,
    memory_vector_search_enabled,
    memory_vector_search_limit,
)
from arena.memory.tags import extract_context_tags, normalize_context_tags, sector_tag_for_ticker
from arena.memory.store import MemoryStore
from arena.memory.thesis import CLOSED_THESIS_EVENT_TYPES
from arena.models import AccountSnapshot, utc_now
from arena.runtime_universe import resolve_runtime_universe

logger = logging.getLogger(__name__)

_TICKER_TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9]{0,5}\b")
_MEMORY_TICKER_STOPWORDS = {
    "BUY",
    "SELL",
    "HOLD",
    "FILLED",
    "SUBMITTED",
    "REJECTED",
    "SIMULATED",
    "STATUS",
    "POLICY",
    "BROKER",
    "KRW",
    "USD",
    "RSI",
    "MACD",
    "ATR",
    "VWAP",
    "OHLC",
    "PNL",
    "NAV",
    "EPS",
    "PE",
    "ETF",
    "IPO",
    "YOY",
    "QOQ",
}


def _safe_json(value: Any) -> Any:
    """Converts nested values into JSON-serializable primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    return value


class ContextBuilder:
    """Builds compact multi-source context payloads for each agent cycle."""

    def __init__(
        self,
        repo: BigQueryRepository,
        memory: MemoryStore,
        board: BoardStore,
        settings: Settings,
    ):
        self.repo = repo
        self.memory = memory
        self.board = board
        self.settings = settings

    _US_MARKETS = US_MARKETS
    _KOSPI_MARKETS = KOSPI_MARKETS

    def _effective_market(self, agent_config: AgentConfig | None = None) -> str:
        """Returns the effective target market for a given agent. Raises if not configured."""
        if agent_config and agent_config.target_market:
            return agent_config.target_market.strip().lower()
        global_market = str(self.settings.kis_target_market or "").strip().lower()
        if not global_market:
            raise ValueError("target_market is not configured. Set target_market in agent config or KIS_TARGET_MARKET globally.")
        return global_market

    def _effective_markets(self, agent_config: AgentConfig | None = None) -> set[str]:
        """Returns the set of effective markets (supports comma-separated multi-market). Raises if not configured."""
        market = self._effective_market(agent_config)  # raises if not configured
        parts = {m.strip().lower() for m in market.split(",") if m.strip()}
        if not parts:
            raise ValueError("target_market is not configured. Set target_market in agent config or KIS_TARGET_MARKET globally.")
        return parts

    def _filter_tickers(self, tickers: list[str], agent_config: AgentConfig | None = None) -> list[str]:
        """Filters tickers to match the configured target market."""
        markets = self._effective_markets(agent_config)
        has_us = bool(markets & self._US_MARKETS)
        has_kospi = bool(markets & self._KOSPI_MARKETS)
        if has_us and has_kospi:
            return [t for t in tickers if t]
        if has_us:
            return [t for t in tickers if t and not t[:1].isdigit()]
        if has_kospi:
            return [t for t in tickers if t.isdigit() and len(t) == 6]
        return [t for t in tickers if t]

    def _market_sources(self, agent_config: AgentConfig | None = None) -> list[str] | None:
        """Returns allowed market_features sources; live runs only trust ETL outputs."""
        if self.settings.trading_mode != "live":
            return None
        result = live_market_sources_for_markets(parse_markets(self._effective_market(agent_config)))
        return result or None

    def _agent_post_authors(self) -> set[str]:
        """Returns normalized agent ids allowed in shared board context."""
        return {str(a).strip() for a in self.settings.agent_ids if str(a).strip()}

    def _trim_text(self, value: object, *, max_len: int) -> str:
        """Returns a bounded string to keep prompt payload compact."""
        text = str(value or "").strip()
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."

    def _append_keyword(self, keywords: list[str], value: object) -> None:
        """Adds one normalized ticker keyword if it is non-empty and unique."""
        token = str(value or "").strip().upper()
        if token and token not in keywords:
            keywords.append(token)

    def _collect_ticker_keywords_from_value(self, value: Any, keywords: list[str], *, depth: int = 0) -> None:
        """Walks compact tool args/results and extracts ticker-like fields."""
        if len(keywords) >= 12 or depth > 3:
            return

        if isinstance(value, dict):
            ticker = value.get("ticker")
            if isinstance(ticker, str):
                self._append_keyword(keywords, ticker)

            tickers = value.get("tickers")
            if isinstance(tickers, list):
                for token in tickers[:8]:
                    self._append_keyword(keywords, token)
                    if len(keywords) >= 12:
                        return

            for key, nested in value.items():
                if key in {"ticker", "tickers"}:
                    continue
                self._collect_ticker_keywords_from_value(nested, keywords, depth=depth + 1)
                if len(keywords) >= 12:
                    return
            return

        if isinstance(value, list):
            for item in value[:8]:
                self._collect_ticker_keywords_from_value(item, keywords, depth=depth + 1)
                if len(keywords) >= 12:
                    return

    def _is_simulated_trade_memory(self, row: dict[str, Any]) -> bool:
        """Returns True for trade_execution memory rows derived from SIMULATED fills."""
        event_type = str(row.get("event_type") or "").strip().lower()
        if event_type != "trade_execution":
            return False

        payload_raw = row.get("payload_json")
        if isinstance(payload_raw, str) and payload_raw.strip():
            try:
                payload = json.loads(payload_raw)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                report = payload.get("report")
                if isinstance(report, dict):
                    status = str(report.get("status") or "").strip().upper()
                    if status == "SIMULATED":
                        return True

        summary = str(row.get("summary") or "").upper()
        return "STATUS=SIMULATED" in summary

    def _coerce_datetime(self, value: Any) -> datetime | None:
        """Converts mixed datetime/string values into timezone-aware datetimes when possible."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def _parse_memory_payload(self, row: dict[str, Any]) -> dict[str, Any]:
        """Parses payload_json into a dict for retrieval-time enrichment."""
        payload_raw = row.get("payload_json")
        if isinstance(payload_raw, dict):
            return payload_raw
        if isinstance(payload_raw, str) and payload_raw.strip():
            try:
                parsed = json.loads(payload_raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _active_thesis_rows(self, *, agent_id: str, focus_tickers: list[str]) -> list[dict[str, Any]]:
        loader = getattr(self.repo, "active_thesis_events", None)
        if not callable(loader) or not focus_tickers:
            return []
        try:
            rows = list(
                loader(
                    agent_id=agent_id,
                    tickers=focus_tickers,
                    trading_mode=self.settings.trading_mode,
                )
            )
        except Exception:
            return []
        return rows

    def _compress_active_thesis_context(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        lines = ["Active Thesis:"]
        for row in rows[:4]:
            payload = self._parse_memory_payload(row)
            ticker = str(payload.get("ticker") or row.get("ticker") or "").strip().upper()
            state = str(payload.get("state") or "").strip().lower()
            thesis_summary = self._trim_text(payload.get("thesis_summary") or row.get("summary"), max_len=140)
            strategy_refs = payload.get("strategy_refs") if isinstance(payload.get("strategy_refs"), list) else []
            refs = ",".join(str(token).strip().lower() for token in strategy_refs[:3] if str(token).strip())
            parts = [part for part in [ticker, state] if part]
            meta = f"[{' | '.join(parts)}] " if parts else ""
            line = f"- {meta}{thesis_summary}"
            if refs:
                line += f" refs={refs}"
            lines.append(line)
        return "\n".join(lines)

    def _extract_memory_tickers(self, row: dict[str, Any]) -> list[str]:
        """Extracts likely ticker symbols from payload first, then summary text."""
        keywords: list[str] = []
        context_tags = row.get("context_tags")
        if isinstance(context_tags, dict):
            for token in context_tags.get("tickers") or []:
                self._append_keyword(keywords, token)
                if len(keywords) >= 4:
                    return keywords[:4]
        payload = self._parse_memory_payload(row)
        if payload:
            self._collect_ticker_keywords_from_value(payload, keywords)
        if not keywords:
            summary = str(row.get("summary") or "")
            for token in _TICKER_TOKEN_RE.findall(summary):
                if token in _MEMORY_TICKER_STOPWORDS:
                    continue
                self._append_keyword(keywords, token)
                if len(keywords) >= 4:
                    break
        return keywords[:4]

    def _extract_memory_side(self, row: dict[str, Any]) -> str:
        """Returns BUY/SELL/HOLD when a memory row clearly encodes one."""
        payload = self._parse_memory_payload(row)
        intent = payload.get("intent") if isinstance(payload, dict) else None
        if isinstance(intent, dict):
            token = str(intent.get("side") or "").strip().upper()
            if token in {"BUY", "SELL", "HOLD"}:
                return token
        summary = str(row.get("summary") or "").upper()
        for token in ("BUY", "SELL", "HOLD"):
            if f" {token} " in f" {summary} ":
                return token
        return ""

    def _normalize_memory_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """Adds retrieval-time metadata used for reranking and prompt compression."""
        normalized = dict(row)
        normalized["event_type"] = str(row.get("event_type") or "event").strip()
        normalized["summary"] = self._trim_text(row.get("summary"), max_len=220)
        normalized["score"] = float(row.get("score") or 0.0)
        normalized["importance_score"] = float(
            row.get("importance_score")
            if row.get("importance_score") is not None
            else row.get("score") or 0.0
        )
        normalized["outcome_score"] = (
            float(row.get("outcome_score"))
            if row.get("outcome_score") is not None
            else None
        )
        created_at = self._coerce_datetime(row.get("created_at"))
        if created_at is not None:
            normalized["created_at"] = created_at
            normalized["created_date"] = created_at.date().isoformat()
            normalized["age_days"] = max(0, (utc_now() - created_at).days)
        else:
            created_date = str(row.get("created_date") or "").strip()
            if created_date:
                normalized["created_date"] = created_date
            try:
                normalized["age_days"] = int(row.get("age_days")) if row.get("age_days") is not None else None
            except (TypeError, ValueError):
                normalized["age_days"] = None
        expires_at = self._coerce_datetime(row.get("expires_at"))
        if expires_at is not None:
            normalized["expires_at"] = expires_at
        raw_context_tags = row.get("context_tags")
        if raw_context_tags is None:
            raw_context_tags = row.get("context_tags_json")
        if isinstance(raw_context_tags, str) and raw_context_tags.strip():
            try:
                raw_context_tags = json.loads(raw_context_tags)
            except Exception:
                raw_context_tags = None
        context_tags = normalize_context_tags(
            raw_context_tags,
            primary_regime=row.get("primary_regime"),
            primary_strategy_tag=row.get("primary_strategy_tag"),
            primary_sector=row.get("primary_sector"),
            max_tags=memory_tagging_max_tags(self.settings.memory_policy),
        )
        if not context_tags and memory_tagging_enabled(self.settings.memory_policy):
            context_tags = extract_context_tags(
                event_type=str(normalized.get("event_type") or ""),
                summary=str(normalized.get("summary") or ""),
                payload=self._parse_memory_payload(row),
                max_tags=memory_tagging_max_tags(self.settings.memory_policy),
            )
        if context_tags:
            normalized["context_tags"] = context_tags
            if not normalized.get("primary_regime"):
                normalized["primary_regime"] = (context_tags.get("regimes") or [None])[0]
            if not normalized.get("primary_strategy_tag"):
                normalized["primary_strategy_tag"] = (context_tags.get("strategies") or [None])[0]
            if not normalized.get("primary_sector"):
                normalized["primary_sector"] = (context_tags.get("sectors") or [None])[0]
        last_accessed_at = self._coerce_datetime(row.get("last_accessed_at"))
        if last_accessed_at is not None:
            normalized["last_accessed_at"] = last_accessed_at
        try:
            if row.get("access_count") is not None:
                normalized["access_count"] = int(row.get("access_count") or 0)
        except (TypeError, ValueError):
            pass
        try:
            if row.get("decay_score") is not None:
                normalized["decay_score"] = float(row.get("decay_score"))
        except (TypeError, ValueError):
            pass
        try:
            if row.get("effective_score") is not None:
                normalized["effective_score"] = float(row.get("effective_score"))
        except (TypeError, ValueError):
            pass
        normalized["memory_tier"] = self._memory_tier_for_row(normalized)
        normalized["tickers"] = self._extract_memory_tickers(normalized)
        normalized["side"] = self._extract_memory_side(normalized)
        return normalized

    def _outcome_decisiveness_bonus(self, outcome_score: Any) -> float:
        """Rewards memories with a clearly known outcome, win or loss."""
        try:
            outcome = float(outcome_score)
        except (TypeError, ValueError):
            return 0.0
        max_bonus = self._memory_policy_float("retrieval.reranking.outcome_bonus_max", 0.18)
        return min(abs(outcome - 0.5) * 0.36, max_bonus)

    def _resolved_effective_score(self, row: dict[str, Any]) -> float | None:
        raw_effective_score = row.get("effective_score")
        try:
            if raw_effective_score is not None:
                return max(0.0, min(float(raw_effective_score), 1.0))
        except (TypeError, ValueError):
            pass
        if not memory_forgetting_enabled(self.settings.memory_policy):
            return None
        _, computed = effective_memory_score(
            row,
            default_decay_factor=memory_forgetting_default_decay_factor(self.settings.memory_policy),
            min_decay_multiplier=memory_forgetting_min_effective_score(self.settings.memory_policy),
            access_curve=memory_forgetting_access_curve(self.settings.memory_policy),
            working_weight=memory_forgetting_tier_weight(self.settings.memory_policy, "working"),
            episodic_weight=memory_forgetting_tier_weight(self.settings.memory_policy, "episodic"),
            semantic_weight=memory_forgetting_tier_weight(self.settings.memory_policy, "semantic"),
            now=utc_now(),
        )
        return computed

    def _memory_effective_score_bonus(self, row: dict[str, Any]) -> float:
        """Adds a bounded lift for memories that survive forgetting with strong effective_score."""
        if not memory_forgetting_enabled(self.settings.memory_policy):
            return 0.0
        scale = self._memory_policy_float("retrieval.reranking.effective_score_bonus_scale", 0.08)
        cap = self._memory_policy_float("retrieval.reranking.effective_score_bonus_cap", 0.08)
        if scale <= 0.0 or cap <= 0.0:
            return 0.0
        effective_score = self._resolved_effective_score(row)
        if effective_score is None:
            return 0.0
        baseline = self._memory_policy_float("cleanup.min_score", 0.30)
        if effective_score <= baseline:
            return 0.0
        normalized = max(0.0, min((effective_score - baseline) / max(1e-6, 1.0 - baseline), 1.0))
        return min(normalized * scale, cap)

    def _memory_policy_float(self, path: str, default: float) -> float:
        value = get_memory_policy_value(self.settings.memory_policy, path, default)
        return float(value) if value is not None else float(default)

    def _memory_policy_int(self, path: str, default: int) -> int:
        value = get_memory_policy_value(self.settings.memory_policy, path, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _memory_tagging_enabled(self) -> bool:
        return memory_tagging_enabled(self.settings.memory_policy)

    def _memory_hierarchy_enabled(self) -> bool:
        return memory_hierarchy_enabled(self.settings.memory_policy)

    def _current_memory_context_tags(
        self,
        *,
        market_rows: list[dict[str, Any]],
        focus_tickers: list[str],
    ) -> dict[str, list[str]]:
        if not self._memory_tagging_enabled():
            return {}

        regimes: list[str] = []
        strategies: list[str] = []
        sectors: list[str] = []

        valid_market_rows = [row for row in (market_rows or []) if str(row.get("ticker") or "").strip()]
        returns: list[float] = []
        volatilities: list[float] = []
        for row in valid_market_rows:
            try:
                returns.append(float(row.get("ret_20d") or 0.0))
            except (TypeError, ValueError):
                pass
            try:
                volatilities.append(float(row.get("volatility_20d") or 0.0))
            except (TypeError, ValueError):
                pass
        if returns:
            avg_ret_20d = sum(returns) / float(len(returns))
            if avg_ret_20d >= 0.05:
                regimes.append("bull")
            elif avg_ret_20d <= -0.03:
                regimes.append("bear")
            else:
                regimes.append("sideways")
        if volatilities:
            avg_vol_20d = sum(volatilities) / float(len(volatilities))
            if avg_vol_20d >= 0.25:
                regimes.append("high_vol")
            elif avg_vol_20d > 0 and avg_vol_20d <= 0.12:
                regimes.append("low_vol")

        row_map = {
            str(row.get("ticker") or "").strip().upper(): row
            for row in valid_market_rows
            if str(row.get("ticker") or "").strip()
        }
        selected_rows = [row_map[token] for token in focus_tickers if token in row_map]
        if not selected_rows:
            selected_rows = valid_market_rows[:4]

        if selected_rows:
            max_ret_20d = 0.0
            max_ret_5d = 0.0
            min_ret_20d = 0.0
            has_values = False
            for row in selected_rows:
                try:
                    ret_20d = float(row.get("ret_20d") or 0.0)
                    ret_5d = float(row.get("ret_5d") or 0.0)
                except (TypeError, ValueError):
                    continue
                if not has_values:
                    max_ret_20d = ret_20d
                    max_ret_5d = ret_5d
                    min_ret_20d = ret_20d
                    has_values = True
                else:
                    max_ret_20d = max(max_ret_20d, ret_20d)
                    max_ret_5d = max(max_ret_5d, ret_5d)
                    min_ret_20d = min(min_ret_20d, ret_20d)
            if has_values:
                if max_ret_20d >= 0.08:
                    strategies.append("momentum")
                if max_ret_20d >= 0.12 and max_ret_5d >= 0.02:
                    strategies.append("breakout")
                if min_ret_20d <= -0.08:
                    strategies.append("mean_reversion")

        for token in focus_tickers[:4]:
            sector = sector_tag_for_ticker(token)
            if sector:
                sectors.append(sector)

        normalized = normalize_context_tags(
            {
                "regimes": regimes,
                "strategies": strategies,
                "sectors": sectors,
            },
            max_tags=memory_tagging_max_tags(self.settings.memory_policy),
        )
        return {
            "regimes": list(normalized.get("regimes") or []),
            "strategies": list(normalized.get("strategies") or []),
            "sectors": list(normalized.get("sectors") or []),
        }

    def _memory_contextual_tag_bonus(
        self,
        row: dict[str, Any],
        *,
        current_context_tags: dict[str, list[str]],
    ) -> float:
        if not self._memory_tagging_enabled():
            return 0.0

        memory_tags = normalize_context_tags(
            row.get("context_tags"),
            primary_regime=row.get("primary_regime"),
            primary_strategy_tag=row.get("primary_strategy_tag"),
            primary_sector=row.get("primary_sector"),
            max_tags=memory_tagging_max_tags(self.settings.memory_policy),
        )
        if not memory_tags:
            return 0.0

        bonus = 0.0
        row_regimes = set(memory_tags.get("regimes") or [])
        row_strategies = set(memory_tags.get("strategies") or [])
        row_sectors = set(memory_tags.get("sectors") or [])
        if row_regimes & set(current_context_tags.get("regimes") or []):
            bonus += memory_tagging_regime_bonus(self.settings.memory_policy)
        if row_strategies & set(current_context_tags.get("strategies") or []):
            bonus += memory_tagging_strategy_bonus(self.settings.memory_policy)
        if row_sectors & set(current_context_tags.get("sectors") or []):
            bonus += memory_tagging_sector_bonus(self.settings.memory_policy)
        return bonus

    def _memory_tier_for_row(self, row: dict[str, Any]) -> str:
        token = str(row.get("memory_tier") or "").strip().lower()
        if token in {"working", "episodic", "semantic"}:
            return token
        if not self._memory_hierarchy_enabled():
            return ""
        event_type = str(row.get("event_type") or "").strip().lower()
        if event_type == "strategy_reflection":
            return "semantic"
        if event_type == "react_tools_summary":
            return "working"
        return "episodic"

    def _memory_tier_bonus(self, memory_tier: str) -> float:
        if not self._memory_hierarchy_enabled():
            return 0.0
        tier = str(memory_tier or "").strip().lower()
        if tier == "semantic":
            return 0.18
        if tier == "episodic":
            return 0.04
        if tier == "working":
            return -0.35
        return 0.0

    def _memory_is_expired(self, row: dict[str, Any]) -> bool:
        if not self._memory_hierarchy_enabled():
            return False
        memory_tier = str(row.get("memory_tier") or "").strip().lower()
        if not memory_tier or memory_tier == "semantic":
            return False
        expires_at = row.get("expires_at")
        if isinstance(expires_at, datetime):
            return expires_at <= utc_now()
        age_days = row.get("age_days")
        try:
            age = int(age_days)
        except (TypeError, ValueError):
            return False
        if memory_tier == "working":
            max_age_days = max(1, math.ceil(memory_hierarchy_working_ttl_hours(self.settings.memory_policy) / 24.0))
            return age >= max_age_days
        if memory_tier == "episodic":
            return age > memory_hierarchy_episodic_ttl_days(self.settings.memory_policy)
        return False

    def _outcome_label(self, outcome_score: Any) -> str:
        """Returns a compact label for prompt compression."""
        try:
            outcome = float(outcome_score)
        except (TypeError, ValueError):
            return ""
        if outcome >= 0.65:
            return "win"
        if outcome <= 0.35:
            return "loss"
        return "neutral"

    def _memory_type_bonus(self, event_type: str) -> float:
        """Returns a small prior to favor actionable lessons over raw logs."""
        token = str(event_type or "").strip().lower()
        if token == "strategy_reflection":
            return self._memory_policy_float("retrieval.reranking.type_bonus_reflection", 0.45)
        if token in CLOSED_THESIS_EVENT_TYPES:
            return 0.34 if token == "thesis_invalidated" else 0.30
        if token == "trade_execution":
            return self._memory_policy_float("retrieval.reranking.type_bonus_trade", 0.28)
        if token == "manual_note":
            return self._memory_policy_float("retrieval.reranking.type_bonus_manual", 0.16)
        if token == "react_tools_summary":
            return self._memory_policy_float("retrieval.reranking.type_bonus_react_tools", -0.12)
        return 0.0

    def _memory_recency_bonus(self, age_days: Any) -> float:
        """Returns a bounded freshness bonus."""
        try:
            age = int(age_days)
        except (TypeError, ValueError):
            return 0.0
        if age <= 3:
            return self._memory_policy_float("retrieval.reranking.recency_bonus_3d", 0.08)
        if age <= 14:
            return self._memory_policy_float("retrieval.reranking.recency_bonus_14d", 0.05)
        if age <= 45:
            return self._memory_policy_float("retrieval.reranking.recency_bonus_45d", 0.02)
        return 0.0

    def _memory_ticker_bonus(self, row: dict[str, Any], active_tickers: set[str]) -> float:
        """Rewards memories that overlap with currently relevant tickers."""
        if not active_tickers:
            return 0.0
        row_tickers = {
            str(token or "").strip().upper()
            for token in (row.get("tickers") or [])
            if str(token or "").strip()
        }
        overlap = row_tickers & active_tickers
        if not overlap:
            return 0.0
        base = self._memory_policy_float("retrieval.reranking.ticker_bonus_base", 0.30)
        step = self._memory_policy_float("retrieval.reranking.ticker_bonus_step", 0.05)
        max_bonus = self._memory_policy_float("retrieval.reranking.ticker_bonus_max", 0.40)
        bonus = base + (step * float(max(len(overlap) - 1, 0)))
        return min(bonus, max_bonus)

    def _hydrate_memory_event_ids(self, agent_id: str, event_ids: list[str]) -> list[dict[str, Any]]:
        """Hydrates vector-hit event ids from BQ when possible."""
        loader = getattr(self.repo, "memory_events_by_ids", None)
        if not callable(loader) or not event_ids:
            return []
        try:
            return list(
                loader(
                    agent_id=agent_id,
                    event_ids=event_ids,
                    trading_mode=self.settings.trading_mode,
                )
            )
        except Exception as exc:
            logger.warning("[yellow]memory hydrate failed[/yellow] agent=%s err=%s", agent_id, str(exc))
            return []

    def _vector_memory_candidates(
        self,
        agent_id: str,
        queries: list[str],
        *,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Fetches vector candidates and returns hydrated rows plus event-level bonuses."""
        if not self.memory.vector_store or not queries or not memory_vector_search_enabled(self.settings.memory_policy):
            return [], {}

        tenant = self.repo.resolve_tenant_id()
        hydrated_bonus: dict[str, float] = {}
        event_ids: list[str] = []
        raw_by_event_id: dict[str, dict[str, Any]] = {}

        per_query_limit = max(2, min(limit, memory_vector_search_limit(self.settings.memory_policy)))
        for q in queries:
            results = self.memory.vector_store.search_similar_memories(
                agent_id=agent_id,
                query=q,
                limit=per_query_limit,
                trading_mode=self.settings.trading_mode,
                tenant_id=tenant,
            )
            for idx, row in enumerate(results):
                eid = str(row.get("event_id") or "").strip()
                bonus = max(0.12, 0.34 - (0.04 * float(idx)))
                if eid:
                    if eid not in hydrated_bonus:
                        event_ids.append(eid)
                    hydrated_bonus[eid] = max(hydrated_bonus.get(eid, 0.0), bonus)
                    raw_by_event_id.setdefault(eid, dict(row))

        hydrated_rows = self._hydrate_memory_event_ids(agent_id, event_ids)
        hydrated_by_event_id = {
            str(row.get("event_id") or "").strip(): self._normalize_memory_row(row)
            for row in hydrated_rows
            if str(row.get("event_id") or "").strip()
        }
        rows: list[dict[str, Any]] = []
        for eid in event_ids:
            if eid in hydrated_by_event_id:
                rows.append(hydrated_by_event_id[eid])
                continue
            raw = raw_by_event_id.get(eid)
            if isinstance(raw, dict):
                rows.append(self._normalize_memory_row(raw))
        return rows, hydrated_bonus

    def _rerank_memory_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        active_tickers: set[str],
        current_context_tags: dict[str, list[str]],
        vector_bonus: dict[str, float],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Ranks memory candidates by actionability rather than raw insertion order."""
        ranked: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in rows:
            row = self._normalize_memory_row(raw)
            key = str(row.get("event_id") or row.get("summary") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)

            event_type = str(row.get("event_type") or "")
            if not memory_event_enabled(self.settings.memory_policy, event_type, True):
                continue
            if self._memory_is_expired(row):
                continue
            if self._memory_hierarchy_enabled() and str(row.get("memory_tier") or "").strip().lower() == "working":
                continue
            retrieval_score = float(row.get("importance_score") or 0.0)
            retrieval_score += self._memory_type_bonus(event_type)
            retrieval_score += self._memory_tier_bonus(str(row.get("memory_tier") or ""))
            retrieval_score += self._memory_recency_bonus(row.get("age_days"))
            retrieval_score += self._memory_ticker_bonus(row, active_tickers)
            retrieval_score += self._memory_contextual_tag_bonus(row, current_context_tags=current_context_tags)
            retrieval_score += self._outcome_decisiveness_bonus(row.get("outcome_score"))
            retrieval_score += self._memory_effective_score_bonus(row)
            retrieval_score += float(vector_bonus.get(str(row.get("event_id") or ""), 0.0))
            if self._is_simulated_trade_memory(row):
                retrieval_score -= 0.22
            row["outcome_label"] = self._outcome_label(row.get("outcome_score"))
            resolved_effective_score = self._resolved_effective_score(row)
            if resolved_effective_score is not None:
                row["effective_score"] = round(resolved_effective_score, 4)
            row["retrieval_score"] = round(retrieval_score, 4)
            ranked.append(row)

        ranked.sort(
            key=lambda r: (
                float(r.get("retrieval_score") or 0.0),
                float(r.get("importance_score") or 0.0),
                -int(r.get("age_days") or 9999) if isinstance(r.get("age_days"), int) else -9999,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _format_memory_line(self, row: dict[str, Any]) -> str:
        """Formats one compact memory line for prompt injection."""
        bits: list[str] = []
        tickers = [str(t).strip().upper() for t in (row.get("tickers") or []) if str(t).strip()]
        if tickers:
            bits.append("/".join(tickers[:2]))
        side = str(row.get("side") or "").strip().upper()
        if side:
            bits.append(side)
        created_date = str(row.get("created_date") or "").strip()
        if created_date:
            bits.append(created_date)
        else:
            age_raw = row.get("age_days")
            if isinstance(age_raw, int):
                bits.append(f"{age_raw}d")
        outcome_label = str(row.get("outcome_label") or "").strip()
        if outcome_label and outcome_label != "neutral":
            bits.append(outcome_label)
        meta = f"[{' | '.join(bits)}] " if bits else ""
        return f"- {meta}{self._trim_text(row.get('summary'), max_len=170)}"

    def _compress_memory_context(self, rows: list[dict[str, Any]]) -> str:
        """Builds a sectioned memory summary instead of a flat log dump."""
        if not rows:
            return ""

        sections: list[str] = []
        used: set[str] = set()
        if self._memory_hierarchy_enabled():
            specs = [
                ("Semantic Lessons", {"semantic"}, 2),
                ("Recent Episodes", {"episodic"}, 3),
            ]
        else:
            specs = [
                ("Past Lessons", {"strategy_reflection"}, 2),
                ("Thesis Outcomes", {"thesis_invalidated", "thesis_realized"}, 2),
                ("Similar Trades", {"trade_execution"}, 2),
                ("Useful Observations", {"manual_note"}, 2),
            ]
        for title, event_types, max_items in specs:
            if self._memory_hierarchy_enabled():
                picked = [
                    row for row in rows
                    if str(row.get("memory_tier") or "") in event_types
                    and str(row.get("event_id") or row.get("summary") or "") not in used
                ][:max_items]
            else:
                picked = [
                    row for row in rows
                    if str(row.get("event_type") or "") in event_types
                    and str(row.get("event_id") or row.get("summary") or "") not in used
                ][:max_items]
            if not picked:
                continue
            sections.append("\n".join([f"{title}:"] + [self._format_memory_line(row) for row in picked]))
            for row in picked:
                used.add(str(row.get("event_id") or row.get("summary") or ""))

        if not sections:
            fallback = [
                row for row in rows
                if str(row.get("event_id") or row.get("summary") or "") not in used
            ][:2]
            if fallback:
                sections.append("\n".join(["Recent Memory:"] + [self._format_memory_line(row) for row in fallback]))

        return "\n".join(sections[:3])

    def _graph_seed_node_ids(self, rows: list[dict[str, Any]]) -> list[str]:
        seeds: list[str] = []
        for row in rows:
            event_id = str(row.get("event_id") or "").strip()
            if not event_id:
                continue
            node_id = str(row.get("graph_node_id") or "").strip() or memory_event_node_id(event_id)
            if node_id and node_id not in seeds:
                seeds.append(node_id)
        return seeds

    def _normalize_graph_neighbor_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        normalized["node_kind"] = str(row.get("node_kind") or "").strip().lower()
        normalized["edge_type"] = str(row.get("edge_type") or "").strip().upper()
        normalized["direction"] = str(row.get("direction") or "").strip().lower()
        normalized["summary"] = self._trim_text(row.get("summary"), max_len=180)
        normalized["ticker"] = str(row.get("ticker") or "").strip().upper()
        node_created_at = self._coerce_datetime(row.get("node_created_at") or row.get("created_at"))
        if node_created_at is not None:
            normalized["node_created_at"] = node_created_at
            normalized["created_date"] = node_created_at.date().isoformat()
        edge_created_at = self._coerce_datetime(row.get("edge_created_at"))
        if edge_created_at is not None:
            normalized["edge_created_at"] = edge_created_at
        detail_json = row.get("detail_json")
        if isinstance(detail_json, str) and detail_json.strip():
            try:
                detail_json = json.loads(detail_json)
            except Exception:
                detail_json = None
        if isinstance(detail_json, dict):
            normalized["detail_json"] = detail_json
        payload_json = row.get("payload_json")
        if isinstance(payload_json, str) and payload_json.strip():
            try:
                payload_json = json.loads(payload_json)
            except Exception:
                payload_json = None
        if isinstance(payload_json, dict):
            normalized["payload_json"] = payload_json
        try:
            if row.get("confidence") is not None:
                normalized["confidence"] = float(row.get("confidence"))
        except (TypeError, ValueError):
            pass
        try:
            if row.get("edge_strength") is not None:
                normalized["edge_strength"] = float(row.get("edge_strength"))
        except (TypeError, ValueError):
            pass
        return normalized

    def _fetch_graph_neighbors(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not memory_graph_enabled(self.settings.memory_policy):
            return []
        loader = getattr(self.repo, "memory_graph_neighbors", None)
        if not callable(loader):
            return []

        max_nodes = memory_graph_max_expanded_nodes(self.settings.memory_policy)
        max_hops = memory_graph_max_expansion_hops(self.settings.memory_policy)
        min_confidence = memory_graph_inferred_edge_min_confidence(self.settings.memory_policy)
        seed_node_ids = self._graph_seed_node_ids(rows[: max(1, min(len(rows), 6))])
        if not seed_node_ids:
            return []

        visited_nodes = set(seed_node_ids)
        seen_pairs: set[tuple[str, str, str, str]] = set()
        expanded: list[dict[str, Any]] = []
        frontier_roots: dict[str, set[str]] = {seed_node_id: {seed_node_id} for seed_node_id in seed_node_ids}

        for hop in range(max_hops):
            if not frontier_roots or len(expanded) >= max_nodes:
                break
            frontier_ids = list(frontier_roots.keys())
            remaining = max_nodes - len(expanded)
            try:
                candidates = list(
                    loader(
                        seed_node_ids=frontier_ids,
                        trading_mode=self.settings.trading_mode,
                        min_confidence=min_confidence,
                        limit=max(remaining * 4, remaining),
                    )
                )
            except Exception as exc:
                logger.warning("[yellow]graph neighbor load failed[/yellow] err=%s", str(exc))
                return expanded

            next_frontier_roots: dict[str, set[str]] = {}
            for raw in candidates:
                neighbor_id = str(raw.get("neighbor_node_id") or "").strip()
                frontier_seed = str(raw.get("seed_node_id") or "").strip()
                if not neighbor_id or not frontier_seed:
                    continue
                roots = frontier_roots.get(frontier_seed) or {frontier_seed}
                edge_type = str(raw.get("edge_type") or "").strip().upper()
                direction = str(raw.get("direction") or "").strip().lower()
                for root_seed in roots:
                    pair = (root_seed, neighbor_id, edge_type, direction)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    item = self._normalize_graph_neighbor_row(raw)
                    item["root_seed_node_id"] = root_seed
                    item["hop"] = hop + 1
                    expanded.append(item)
                    if len(expanded) >= max_nodes:
                        break
                if len(expanded) >= max_nodes:
                    break
                if neighbor_id not in visited_nodes:
                    visited_nodes.add(neighbor_id)
                    next_frontier_roots.setdefault(neighbor_id, set()).update(roots)
            frontier_roots = next_frontier_roots
        expanded = expanded[:max_nodes]
        if expanded:
            logger.info(
                "[cyan]graph expansion[/cyan] seeds=%d expanded=%d hops=%d max_nodes=%d",
                len(seed_node_ids),
                len(expanded),
                max_hops,
                max_nodes,
            )
        return expanded

    def _graph_relation_label(self, row: dict[str, Any]) -> str:
        edge_type = str(row.get("edge_type") or "").strip().upper()
        direction = str(row.get("direction") or "").strip().lower()
        mapping = {
            ("INFORMED_BY", "incoming"): "informed by",
            ("INFORMED_BY", "outgoing"): "informs",
            ("ABSTRACTED_TO", "incoming"): "abstracts",
            ("ABSTRACTED_TO", "outgoing"): "abstracted to",
            ("PRECEDES", "incoming"): "preceded by",
            ("PRECEDES", "outgoing"): "precedes",
            ("RESULTED_IN", "incoming"): "resulted from",
            ("RESULTED_IN", "outgoing"): "resulted in",
            ("EXECUTED_AS", "incoming"): "execution of",
            ("EXECUTED_AS", "outgoing"): "executed as",
            ("FEEDBACK_TO", "incoming"): "feedback from",
            ("FEEDBACK_TO", "outgoing"): "feedback to",
        }
        return mapping.get((edge_type, direction), "references")

    def _graph_neighbor_descriptor(self, row: dict[str, Any]) -> str:
        relation = self._graph_relation_label(row)
        node_kind = str(row.get("node_kind") or "").strip().lower().replace("_", " ")
        summary = self._trim_text(row.get("summary"), max_len=120)
        ticker = str(row.get("ticker") or "").strip().upper()
        prefix = f"{relation} {node_kind}".strip()
        if ticker:
            prefix += f" {ticker}"
        if summary:
            return f"{prefix}: {summary}"
        return prefix

    def _compress_graph_context(
        self,
        seed_rows: list[dict[str, Any]],
        graph_rows: list[dict[str, Any]],
    ) -> str:
        if not graph_rows:
            return ""
        seeds_by_node = {
            (str(row.get("graph_node_id") or "").strip() or memory_event_node_id(str(row.get("event_id") or "").strip())): row
            for row in seed_rows
            if str(row.get("event_id") or "").strip()
        }
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in graph_rows:
            root_seed = str(row.get("root_seed_node_id") or "").strip()
            if root_seed:
                grouped.setdefault(root_seed, []).append(row)

        lines: list[str] = ["Decision Paths:"]
        added = 0
        for seed_node_id in self._graph_seed_node_ids(seed_rows):
            seed = seeds_by_node.get(seed_node_id)
            if not seed:
                continue
            related = grouped.get(seed_node_id) or []
            if not related:
                continue
            descriptors: list[str] = []
            seen: set[str] = set()
            for row in related:
                descriptor = self._graph_neighbor_descriptor(row)
                if descriptor in seen:
                    continue
                seen.add(descriptor)
                descriptors.append(descriptor)
                if len(descriptors) >= 3:
                    break
            if not descriptors:
                continue
            lines.append(
                f"- {self._trim_text(seed.get('summary'), max_len=110)} || " + " ; ".join(descriptors)
            )
            added += 1
            if added >= 3:
                break
        return "\n".join(lines) if added else ""

    def _environment_memory_queries(self) -> list[str]:
        """Builds semantic environment anchors from recent shared research briefings."""
        loader = getattr(self.repo, "get_research_briefings", None)
        if not callable(loader):
            return []
        try:
            rows = list(
                loader(
                    categories=["global_market", "geopolitical"],
                    limit=3,
                    trading_mode=self.settings.trading_mode,
                )
            )
        except Exception as exc:
            logger.warning("[yellow]environment briefing load failed[/yellow] err=%s", str(exc))
            return []

        queries: list[str] = []
        for row in rows:
            category = str(row.get("category") or "").strip().lower()
            headline = self._trim_text(row.get("headline"), max_len=90)
            summary = self._trim_text(row.get("summary"), max_len=150)
            if category == "global_market":
                prefix = "market environment macro regime"
            elif category == "geopolitical":
                prefix = "market environment geopolitical risk"
            else:
                prefix = "market environment"
            query = self._trim_text(
                " ".join(part for part in [prefix, headline, summary] if part),
                max_len=220,
            )
            if len(query) >= 12:
                queries.append(query)
        return [q for q in dict.fromkeys(queries)]

    def _build_memory_search_queries(
        self,
        agent_id: str,
        snapshot: AccountSnapshot,
        market_rows: list[dict[str, Any]],
    ) -> list[str]:
        """Builds a small vector-query set from the current portfolio/market state."""
        total_equity = 0.0
        try:
            total_equity = float(snapshot.total_equity_krw or 0.0)
        except (TypeError, ValueError):
            total_equity = 0.0

        cash_ratio = 0.0
        if total_equity > 0:
            try:
                cash_ratio = max(0.0, min(float(snapshot.cash_krw or 0.0) / total_equity, 1.5))
            except (TypeError, ValueError):
                cash_ratio = 0.0

        holdings: list[tuple[str, float]] = []
        for ticker, pos in snapshot.positions.items():
            try:
                market_value = float(pos.quantity or 0.0) * float(pos.market_price_krw or 0.0)
            except (TypeError, ValueError):
                market_value = 0.0
            holdings.append((str(ticker or "").strip().upper(), market_value))

        holdings = [(ticker, value) for ticker, value in holdings if ticker]
        holdings.sort(key=lambda item: item[1], reverse=True)
        top_tickers = [ticker for ticker, _ in holdings[:4]]

        top1_weight = 0.0
        top3_weight = 0.0
        if total_equity > 0 and holdings:
            top1_weight = max(holdings[0][1], 0.0) / total_equity
            top3_weight = sum(max(value, 0.0) for _, value in holdings[:3]) / total_equity

        if cash_ratio >= 0.55:
            cash_state = "high cash"
        elif cash_ratio <= 0.15:
            cash_state = "fully invested"
        else:
            cash_state = "balanced cash"

        if top1_weight >= 0.35:
            concentration_state = "single-name concentration"
        elif top3_weight >= 0.75:
            concentration_state = "top-heavy portfolio"
        else:
            concentration_state = "diversified portfolio"

        row_map = {
            str(row.get("ticker") or "").strip().upper(): row
            for row in (market_rows or [])
            if str(row.get("ticker") or "").strip()
        }
        winners: list[str] = []
        laggards: list[str] = []
        high_vol: list[str] = []
        for ticker in top_tickers:
            row = row_map.get(ticker) or {}
            try:
                ret_20d = float(row.get("ret_20d") or 0.0)
            except (TypeError, ValueError):
                ret_20d = 0.0
            try:
                vol_20d = float(row.get("volatility_20d") or 0.0)
            except (TypeError, ValueError):
                vol_20d = 0.0
            if ret_20d >= 0.08 and ticker not in winners:
                winners.append(ticker)
            if ret_20d <= -0.05 and ticker not in laggards:
                laggards.append(ticker)
            if vol_20d >= 0.25 and ticker not in high_vol:
                high_vol.append(ticker)

        queries: list[str] = []
        if top_tickers:
            queries.append(
                self._trim_text(
                    (
                        f"portfolio state {cash_state} {concentration_state} "
                        f"holdings {' '.join(top_tickers)} risk management rebalancing position sizing"
                    ),
                    max_len=220,
                )
            )
        else:
            queries.append(
                self._trim_text(
                    f"portfolio state {cash_state} no holdings entry selection risk management",
                    max_len=220,
                )
            )

        environment_queries = self._environment_memory_queries()
        if environment_queries:
            queries.extend(environment_queries[:2])

        secondary_bits: list[str] = []
        if winners:
            secondary_bits.append("winners " + " ".join(winners[:2]))
        if laggards:
            secondary_bits.append("laggards " + " ".join(laggards[:2]))
        if high_vol:
            secondary_bits.append("high volatility " + " ".join(high_vol[:2]))
        if secondary_bits:
            queries.append(
                self._trim_text(
                    "current holdings market state "
                    + " ".join(secondary_bits)
                    + " trim add hold discipline",
                    max_len=220,
                )
            )

        deduped = [q for q in dict.fromkeys([q.strip() for q in queries if q.strip()]) if len(q) >= 12]
        logger.info(
            "[cyan]memory search[/cyan] agent=%s mode=current_state queries=%s",
            agent_id,
            [q[:80] for q in deduped],
        )
        return deduped[:3]

    def _build_opportunity_memory_query(self, snapshot: AccountSnapshot) -> str | None:
        """Builds one neutral opportunity-oriented query to counter holdings-only retrieval."""
        total_equity = 0.0
        try:
            total_equity = float(snapshot.total_equity_krw or 0.0)
        except (TypeError, ValueError):
            total_equity = 0.0

        cash_ratio = 0.0
        if total_equity > 0:
            try:
                cash_ratio = max(0.0, min(float(snapshot.cash_krw or 0.0) / total_equity, 1.5))
            except (TypeError, ValueError):
                cash_ratio = 0.0

        enhanced = bool(getattr(self.settings, "autonomy_opportunity_context_enabled", False))
        if not snapshot.positions or cash_ratio >= 0.30:
            query = (
                "new entry opportunity opportunity cost compare replacement candidate "
                "sector rotation breakout missed winner false breakout entry timing conviction"
                if enhanced
                else "new entry opportunity sector rotation breakout missed winner false breakout entry timing conviction"
            )
        else:
            query = (
                "position exit profit-taking overweight concentration rebalance rotation "
                "replacement candidate opportunity cost compare new opportunity entry"
                if enhanced
                else "position exit profit-taking overweight concentration rebalance rotation new opportunity entry"
            )
        query = self._trim_text(query, max_len=220)
        return query if len(query) >= 12 else None

    @staticmethod
    def _append_unique_memory_rows(
        target: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        seen: set[str],
        limit: int,
    ) -> None:
        """Appends rows while preserving order and preventing duplicate events."""
        if limit <= 0:
            return
        for row in rows:
            key = str(row.get("event_id") or row.get("summary") or "").strip()
            if not key or key in seen:
                continue
            target.append(row)
            seen.add(key)
            if len(target) >= limit:
                return

    def _merge_memory_query_tracks(
        self,
        *,
        primary_rows: list[dict[str, Any]],
        opportunity_rows: list[dict[str, Any]],
        total_limit: int,
    ) -> list[dict[str, Any]]:
        """Merges holdings/opportunity retrieval with a front-of-prompt 4+2 split."""
        cap = max(1, int(total_limit))
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        primary_front = min(4, cap)
        opportunity_front = min(2, max(cap - primary_front, 0))
        self._append_unique_memory_rows(merged, primary_rows, seen=seen, limit=primary_front)
        self._append_unique_memory_rows(
            merged,
            opportunity_rows,
            seen=seen,
            limit=len(merged) + opportunity_front,
        )

        tail_candidates: list[dict[str, Any]] = []
        for row in [*primary_rows, *opportunity_rows]:
            key = str(row.get("event_id") or row.get("summary") or "").strip()
            if not key or key in seen:
                continue
            tail_candidates.append(row)

        tail_candidates.sort(
            key=lambda r: (
                float(r.get("retrieval_score") or 0.0),
                float(r.get("importance_score") or 0.0),
                -int(r.get("age_days") or 9999) if isinstance(r.get("age_days"), int) else -9999,
            ),
            reverse=True,
        )
        self._append_unique_memory_rows(
            merged,
            tail_candidates,
            seen=seen,
            limit=cap,
        )
        return merged[:cap]

    def _fetch_smart_memory_rows(
        self,
        agent_id: str,
        query: str | list[str],
        limit: int,
        *,
        focus_tickers: list[str],
        market_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Builds a strict vector-only candidate pool and reranks before prompt injection."""
        active_tickers = {
            str(t or "").strip().upper()
            for t in focus_tickers
            if str(t or "").strip()
        }
        current_context_tags = self._current_memory_context_tags(
            market_rows=market_rows,
            focus_tickers=[str(t or "").strip().upper() for t in focus_tickers if str(t or "").strip()],
        )

        vector_queries: list[str] = []
        if isinstance(query, str) and query.strip():
            vector_queries.append(query.strip())
        elif isinstance(query, list):
            vector_queries.extend([str(token).strip() for token in query if str(token).strip()][:3])

        if not vector_queries:
            return []

        vector_rows, vector_bonus = self._vector_memory_candidates(
            agent_id=agent_id,
            queries=vector_queries,
            limit=limit,
        )
        if not vector_rows:
            return []

        return self._rerank_memory_rows(
            vector_rows,
            active_tickers=active_tickers,
            current_context_tags=current_context_tags,
            vector_bonus=vector_bonus,
            limit=limit,
        )

    def _log_memory_access_events(
        self,
        *,
        agent_id: str,
        rows: list[dict[str, Any]],
        query: str | list[str],
        cycle_id: str | None = None,
    ) -> None:
        if not memory_forgetting_access_log_enabled(self.settings.memory_policy):
            return
        writer = getattr(self.repo, "append_memory_access_events", None)
        if not callable(writer) or not rows:
            return

        if isinstance(query, list):
            query_text = " || ".join(str(token or "").strip() for token in query if str(token or "").strip())
        else:
            query_text = str(query or "").strip()
        query_text = query_text[:500]
        prompt_keys = {
            str(row.get("event_id") or row.get("summary") or "").strip()
            for row in rows[:6]
            if str(row.get("event_id") or row.get("summary") or "").strip()
        }
        accessed_at = utc_now()
        payload_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            event_id = str(row.get("event_id") or "").strip()
            if not event_id:
                continue
            row_key = str(row.get("event_id") or row.get("summary") or "").strip()
            payload_rows.append(
                {
                    "access_id": f"acc_{uuid4().hex[:12]}",
                    "accessed_at": accessed_at,
                    "event_id": event_id,
                    "agent_id": agent_id,
                    "source_agent_id": str(row.get("agent_id") or "").strip() or None,
                    "trading_mode": self.settings.trading_mode,
                    "access_type": "retrieval",
                    "query_text": query_text or None,
                    "retrieval_score": row.get("retrieval_score"),
                    "used_in_prompt": row_key in prompt_keys,
                    "cycle_id": str(cycle_id or "").strip() or None,
                    "detail_json": {
                        "rank": idx + 1,
                        "event_type": str(row.get("event_type") or "").strip().lower() or None,
                        "memory_tier": str(row.get("memory_tier") or "").strip().lower() or None,
                    },
                }
            )
        if not payload_rows:
            return
        try:
            writer(payload_rows)
            used_in_prompt = sum(1 for row in payload_rows if bool(row.get("used_in_prompt")))
            logger.info(
                "[cyan]memory access log[/cyan] agent=%s cycle_id=%s rows=%d used_in_prompt=%d",
                agent_id,
                str(cycle_id or "").strip() or "-",
                len(payload_rows),
                used_in_prompt,
            )
        except Exception as exc:
            logger.warning(
                "[yellow]memory access log failed[/yellow] agent=%s err=%s",
                agent_id,
                str(exc),
            )

    def build(
        self,
        agent_id: str,
        snapshot: AccountSnapshot,
        *,
        sleeve_baseline_equity_krw: float | None = None,
        sleeve_meta: dict[str, Any] | None = None,
        agent_config: AgentConfig | None = None,
        cycle_id: str | None = None,
    ) -> dict:
        """Assembles market, memory, and board context for one agent."""
        risk_settings = merge_agent_risk_settings(self.settings, agent_config)
        sources = self._market_sources(agent_config)
        focus_tickers = self._filter_tickers(list(snapshot.positions.keys()), agent_config)
        universe = resolve_runtime_universe(
            self.settings,
            self.repo,
            markets=self._effective_market(agent_config),
        )
        market_seed = focus_tickers
        market_rows = self.repo.latest_market_features(
            tickers=market_seed,
            limit=self.settings.context_max_market_rows,
            sources=sources,
        )
        if not market_rows and universe:
            market_rows = self.repo.latest_market_features(
                tickers=universe,
                limit=self.settings.context_max_market_rows,
                sources=sources,
            )
        ticker_name_tokens: list[str] = []
        for token in [*snapshot.positions.keys(), *[row.get("ticker") for row in market_rows if isinstance(row, dict)]]:
            clean = str(token or "").strip().upper()
            if clean and clean not in ticker_name_tokens:
                ticker_name_tokens.append(clean)
        try:
            _ticker_names = (
                self.repo.ticker_name_map(
                    tickers=ticker_name_tokens,
                    limit=max(500, len(ticker_name_tokens) * 4),
                )
                if ticker_name_tokens and hasattr(self.repo, "ticker_name_map")
                else {}
            )
        except Exception:
            _ticker_names = {}
        if _ticker_names:
            enriched_market_rows: list[dict[str, Any]] = []
            for row in market_rows:
                if not isinstance(row, dict):
                    continue
                ticker = str(row.get("ticker") or "").strip().upper()
                ticker_name = str(row.get("ticker_name") or _ticker_names.get(ticker) or "").strip()
                if ticker_name and str(row.get("ticker_name") or "").strip() != ticker_name:
                    enriched = dict(row)
                    enriched["ticker_name"] = ticker_name
                    enriched_market_rows.append(enriched)
                else:
                    enriched_market_rows.append(row)
            market_rows = enriched_market_rows
        memory_query = self._build_memory_search_queries(agent_id, snapshot, market_rows)
        primary_memory_rows = self._fetch_smart_memory_rows(
            agent_id=agent_id,
            query=memory_query,
            limit=self.settings.context_max_memory_events,
            focus_tickers=focus_tickers,
            market_rows=market_rows,
        )
        opportunity_query = self._build_opportunity_memory_query(snapshot)
        opportunity_memory_rows: list[dict[str, Any]] = []
        if opportunity_query:
            opportunity_memory_rows = self._fetch_smart_memory_rows(
                agent_id=agent_id,
                query=opportunity_query,
                limit=self.settings.context_max_memory_events,
                focus_tickers=[],
                market_rows=market_rows,
            )
        memory_rows = self._merge_memory_query_tracks(
            primary_rows=primary_memory_rows,
            opportunity_rows=opportunity_memory_rows,
            total_limit=self.settings.context_max_memory_events,
        )
        active_thesis_rows = self._active_thesis_rows(agent_id=agent_id, focus_tickers=focus_tickers)
        active_thesis_context = self._compress_active_thesis_context(active_thesis_rows)
        logged_queries = list(memory_query)
        if opportunity_query:
            logged_queries.append(opportunity_query)
        self._log_memory_access_events(
            agent_id=agent_id,
            rows=memory_rows,
            query=logged_queries,
            cycle_id=cycle_id,
        )

        board_rows: list[dict[str, Any]] = []
        allowed_authors = self._agent_post_authors()
        if allowed_authors:
            board_rows = [
                row for row in board_rows if str(row.get("agent_id", "")).strip() in allowed_authors
            ]

        # Per-agent performance feedback (virtual sleeve). Keep it compact for prompt injection.
        meta = sleeve_meta or {}
        baseline = 0.0
        if sleeve_baseline_equity_krw is not None:
            try:
                baseline = float(sleeve_baseline_equity_krw)
            except (TypeError, ValueError):
                baseline = 0.0

        nav = 0.0
        try:
            nav = float(snapshot.total_equity_krw)
        except (TypeError, ValueError):
            nav = 0.0

        current_sleeve_initialized_at = meta.get("initialized_at")
        pnl = nav - baseline if baseline > 0 else 0.0
        pnl_ratio = (pnl / baseline) if baseline > 0 else 0.0
        cumulative_pnl_ratio = pnl_ratio
        cumulative_started_at = current_sleeve_initialized_at

        chained_loader = getattr(self.repo, "latest_agent_chained_returns", None)
        if callable(chained_loader):
            try:
                chained_stats = chained_loader(agent_ids=[agent_id]).get(agent_id) or {}
            except Exception as exc:
                logger.warning(
                    "[yellow]latest_agent_chained_returns fallback[/yellow] agent=%s err=%s",
                    agent_id,
                    str(exc),
                )
                chained_stats = {}
            if chained_stats:
                cumulative_pnl_ratio = float(chained_stats.get("return_ratio") or pnl_ratio)
                cumulative_started_at = chained_stats.get("started_at") or current_sleeve_initialized_at
        else:
            chained_stats = {}

        trade_total = int(meta.get("trade_count_total") or 0)
        realized_pnl = 0.0
        try:
            realized_pnl = float(meta.get("realized_pnl_krw") or 0.0)
        except (TypeError, ValueError):
            realized_pnl = 0.0

        win_rate = meta.get("sell_win_rate")
        try:
            win_rate_val = float(win_rate) if win_rate is not None else None
        except (TypeError, ValueError):
            win_rate_val = None

        dividend_income_krw = 0.0
        dividend_count = 0
        try:
            dividend_income_krw = float(meta.get("dividend_income_krw") or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            dividend_count = int(meta.get("dividend_count") or 0)
        except (TypeError, ValueError):
            pass

        # Live values should use the same day boundary as risk checks (UTC date).
        today = utc_now().date()
        try:
            intents_today = int(
                self.repo.recent_intent_count(
                    today,
                    agent_id=agent_id,
                    include_simulated=self.settings.trading_mode != "live",
                    trading_mode=self.settings.trading_mode,
                )
            )
        except Exception as exc:
            if self.settings.trading_mode == "live":
                raise RuntimeError(f"failed to load recent_intent_count for agent={agent_id}") from exc
            logger.warning(
                "[yellow]Recent intent count fallback[/yellow] agent=%s err=%s",
                agent_id,
                str(exc),
            )
            intents_today = 0
        try:
            turnover_today = float(
                self.repo.recent_turnover_krw(
                    today,
                    agent_id=agent_id,
                    include_simulated=self.settings.trading_mode != "live",
                    trading_mode=self.settings.trading_mode,
                )
            )
        except Exception as exc:
            if self.settings.trading_mode == "live":
                raise RuntimeError(f"failed to load recent_turnover_krw for agent={agent_id}") from exc
            logger.warning(
                "[yellow]Recent turnover fallback[/yellow] agent=%s err=%s",
                agent_id,
                str(exc),
            )
            turnover_today = 0.0

        fee_bps = float(risk_settings.estimated_fee_bps)
        est_fees_today = turnover_today * fee_bps / 10_000.0 if turnover_today > 0 and fee_bps > 0 else 0.0

        opened_map = meta.get("position_opened_at") if isinstance(meta.get("position_opened_at"), dict) else {}

        pos_rows: list[dict[str, Any]] = []
        for t, pos in snapshot.positions.items():
            opened_at = opened_map.get(t)
            hold_days = None
            if isinstance(opened_at, datetime):
                hold_days = max(0, (today - opened_at.date()).days)
            avg = float(pos.avg_price_krw or 0.0)
            mkt = float(pos.market_price_krw or 0.0)
            qty = float(pos.quantity or 0.0)
            mv = qty * mkt
            upnl = (mkt - avg) * qty if avg > 0 else 0.0
            upnl_ratio = (mkt / avg - 1.0) if avg > 0 else 0.0
            weight = (mv / nav) if nav > 0 else 0.0
            row_data: dict[str, Any] = {
                "ticker": t,
                "ticker_name": _ticker_names.get(t, ""),
                "quantity": qty,
                "avg_price_krw": avg,
                "market_price_krw": mkt,
                "market_value_krw": mv,
                "weight": weight,
                "unrealized_pnl_krw": upnl,
                "unrealized_pnl_ratio": upnl_ratio,
                "hold_days": hold_days,
                "opened_at": opened_at,
            }
            if pos.quote_currency:
                row_data["quote_currency"] = pos.quote_currency
            if pos.avg_price_native is not None:
                row_data["avg_price_native"] = pos.avg_price_native
            if pos.market_price_native is not None:
                row_data["market_price_native"] = pos.market_price_native
            if pos.fx_rate > 0:
                row_data["fx_rate"] = pos.fx_rate
            pos_rows.append(row_data)

        pos_rows.sort(key=lambda r: float(r.get("market_value_krw") or 0.0), reverse=True)
        hold_days_values = [int(r["hold_days"]) for r in pos_rows if isinstance(r.get("hold_days"), int)]

        # Determine target_market early for currency-aware display
        _target_market = ""
        if agent_config and agent_config.target_market:
            _target_market = agent_config.target_market
        if not _target_market:
            _target_market = self.settings.kis_target_market
        _is_us_market = _target_market.lower().strip() in {"us", "nasdaq", "nyse", "amex"}
        _fx = snapshot.usd_krw_rate if snapshot.usd_krw_rate > 0 else 0.0
        if _fx <= 0:
            _fx = self.settings.usd_krw_rate
            logger.warning("[yellow]context fx using config fallback[/yellow] rate=%.2f", _fx)

        def _fmt_krw(v: float) -> str:
            try:
                return f"{float(v):,.0f}"
            except (TypeError, ValueError):
                return "0"

        def _fmt_usd(v: float) -> str:
            try:
                return f"{float(v):,.2f}"
            except (TypeError, ValueError):
                return "0.00"

        def _fmt_money(krw_val: float) -> str:
            """Format amount in the agent's native currency."""
            if _is_us_market and _fx > 0:
                return f"${_fmt_usd(krw_val / _fx)} USD"
            return f"{_fmt_krw(krw_val)} KRW"

        def _currency_label() -> str:
            return "USD" if _is_us_market else "KRW"

        def _to_display(krw_val: float) -> float:
            """Convert KRW amount to display currency."""
            if _is_us_market and _fx > 0:
                return krw_val / _fx
            return krw_val

        def _fmt_pct(x: float) -> str:
            try:
                return f"{float(x) * 100:+.2f}%"
            except (TypeError, ValueError):
                return "+0.00%"

        perf_lines: list[str] = []
        if baseline > 0 and nav > 0:
            div_part = f" | Dividends {_fmt_money(dividend_income_krw)} ({dividend_count})" if dividend_count > 0 else ""
            base_line = f"NAV {_fmt_money(nav)} (Invested {_fmt_money(baseline)}) | Return {_fmt_pct(pnl_ratio)} | PnL {_fmt_money(pnl)} | Trades {trade_total}{div_part}"
            if chained_stats and abs(cumulative_pnl_ratio - pnl_ratio) > 0.001:
                since_txt = ""
                if isinstance(cumulative_started_at, str) and cumulative_started_at.strip():
                    since_txt = f" since {cumulative_started_at[:10]}"
                base_line += f" | TWR {_fmt_pct(cumulative_pnl_ratio)}{since_txt}"
            perf_lines.append(base_line)
        elif nav > 0:
            perf_lines.append(f"NAV {_fmt_money(nav)} | Trades {trade_total}")

        intents_part = f"{intents_today}"
        if int(risk_settings.max_daily_orders or 0) > 0:
            intents_part = f"{intents_today}/{int(risk_settings.max_daily_orders)}"

        turnover_ratio = (turnover_today / nav) if nav > 0 else 0.0
        fees_part = f"est fees {_fmt_money(est_fees_today)} ({fee_bps:.1f} bps)" if fee_bps > 0 else ""
        win_part = f" | WinRate {win_rate_val*100:.0f}%" if win_rate_val is not None else ""
        perf_lines.append(
            f"Today intents {intents_part} | turnover {_fmt_money(turnover_today)} ({turnover_ratio*100:.0f}%){win_part}" + (f" | {fees_part}" if fees_part else "")
        )

        if pos_rows:
            parts: list[str] = []
            for r in pos_rows:
                t = str(r.get("ticker") or "")
                hd = r.get("hold_days")
                hd_txt = f"{int(hd)}d" if isinstance(hd, int) else "?d"
                parts.append(f"{t} {_fmt_pct(float(r.get('unrealized_pnl_ratio') or 0.0))} ({hd_txt})")
            perf_lines.append("Positions: " + " | ".join(parts))

        performance = {
            "display_currency": _currency_label(),
            "baseline_equity": _to_display(baseline),
            "baseline_equity_krw": baseline,
            "initialized_at": cumulative_started_at,
            "current_sleeve_initialized_at": current_sleeve_initialized_at,
            "nav": _to_display(nav),
            "nav_krw": nav,
            "pnl": _to_display(pnl),
            "pnl_krw": pnl,
            "pnl_ratio": pnl_ratio,
            "cumulative_pnl_ratio": cumulative_pnl_ratio,
            "current_sleeve_pnl": _to_display(pnl),
            "current_sleeve_pnl_krw": pnl,
            "current_sleeve_pnl_ratio": pnl_ratio,
            "trade_count_total": trade_total,
            "realized_pnl": _to_display(realized_pnl),
            "realized_pnl_krw": realized_pnl,
            "sell_win_rate": win_rate_val,
            "today_intents": intents_today,
            "today_turnover": _to_display(turnover_today),
            "today_turnover_krw": turnover_today,
            "today_turnover_ratio": turnover_ratio,
            "estimated_fee_bps": fee_bps,
            "estimated_fees_today": _to_display(est_fees_today),
            "estimated_fees_today_krw": est_fees_today,
            "worst_trade": meta.get("worst_trade"),
            "realized_pnl_by_ticker": meta.get("realized_pnl_by_ticker") if isinstance(meta.get("realized_pnl_by_ticker"), dict) else {},
            "dividend_income_krw": dividend_income_krw,
            "dividend_count": dividend_count,
            "positions": pos_rows,
        }

        max_order_krw = max(float(risk_settings.max_order_krw), 0.0)
        turnover_limit_krw = max(nav * float(risk_settings.max_daily_turnover_ratio), 0.0)
        remaining_turnover_krw = max(turnover_limit_krw - turnover_today, 0.0)
        try:
            cash_krw = float(snapshot.cash_krw)
        except (TypeError, ValueError):
            cash_krw = 0.0
        min_cash_required_krw = max(nav * float(risk_settings.min_cash_buffer_ratio), 0.0)
        max_buy_by_cash_krw = max(cash_krw - min_cash_required_krw, 0.0)
        sleeve_target_krw = max(
            float(self.settings.agent_capitals.get(agent_id, self.settings.sleeve_capital_krw)),
            0.0,
        )
        # "sleeve_remaining_krw" is presented to agents as spendable cash in this cycle.
        # NAV-based cap_gap removed — agents are only constrained by available cash,
        # not by whether NAV exceeds the initial target.  Good performance should not
        # block further buying.
        sleeve_remaining_krw = max_buy_by_cash_krw

        max_daily_orders_raw = int(risk_settings.max_daily_orders)
        max_daily_orders_cap = max_daily_orders_raw if max_daily_orders_raw > 0 else None

        risk_policy = {
            "max_order_krw": max_order_krw,
            "max_daily_turnover_ratio": float(risk_settings.max_daily_turnover_ratio),
            "max_position_ratio": float(risk_settings.max_position_ratio),
            "min_cash_buffer_ratio": float(risk_settings.min_cash_buffer_ratio),
            "ticker_cooldown_seconds": int(risk_settings.ticker_cooldown_seconds),
            "max_daily_orders": max_daily_orders_cap,
            "max_daily_orders_unlimited": max_daily_orders_cap is None,
            "single_share_buy_exception_enabled": True,
            "sleeve_capital_krw": sleeve_target_krw,
        }
        budget_caps = [max_order_krw, remaining_turnover_krw, max_buy_by_cash_krw]
        max_buy_notional_krw = max(0.0, min(budget_caps))
        sleeve_buy_blocked = max_buy_notional_krw <= 1e-9

        order_budget = {
            "display_currency": _currency_label(),
            "cash": _to_display(cash_krw),
            "cash_krw": cash_krw,
            "min_cash_required": _to_display(min_cash_required_krw),
            "min_cash_required_krw": min_cash_required_krw,
            "max_buy_notional_by_cash": _to_display(max_buy_by_cash_krw),
            "max_buy_notional_by_cash_krw": max_buy_by_cash_krw,
            "daily_turnover_limit": _to_display(turnover_limit_krw),
            "remaining_turnover": _to_display(remaining_turnover_krw),
            "remaining_turnover_krw": remaining_turnover_krw,
            "max_order": _to_display(max_order_krw),
            "max_order_krw": max_order_krw,
            "max_buy_notional_by_sleeve": _to_display(sleeve_remaining_krw),
            "max_buy_notional_by_sleeve_krw": sleeve_remaining_krw,
            "max_buy_notional": _to_display(max_buy_notional_krw),
            "max_buy_notional_krw": max_buy_notional_krw,
            "today_intents": intents_today,
            "daily_orders_cap": max_daily_orders_cap,
            "remaining_daily_orders": (
                max(max_daily_orders_raw - intents_today, 0)
                if max_daily_orders_raw > 0
                else None
            ),
        }
        if _is_us_market and _fx > 0:
            order_budget["usd_krw_rate"] = _fx
            order_budget["cash_usd"] = snapshot.cash_foreign if snapshot.cash_foreign > 0 else cash_krw / _fx

        sleeve_state = {
            "display_currency": _currency_label(),
            "target_sleeve": _to_display(sleeve_target_krw),
            "target_sleeve_krw": sleeve_target_krw,
            "current_equity": _to_display(nav),
            "current_equity_krw": nav,
            "sleeve_remaining": _to_display(sleeve_remaining_krw),
            "sleeve_remaining_krw": sleeve_remaining_krw,
            "cap_gap_krw": None,
            "over_target_krw": 0.0,
            "over_target": False,
            "buy_blocked": sleeve_buy_blocked,
        }
        perf_lines.append(
            "Budget "
            f"spendable={_fmt_money(max_buy_notional_krw)} | "
            f"cash_after_buffer={_fmt_money(order_budget['max_buy_notional_by_cash_krw'])} | "
            f"turnover_left={_fmt_money(order_budget['remaining_turnover_krw'])}"
        )
        if sleeve_target_krw > 0:
            perf_lines.append(
                "Sleeve "
                f"target={_fmt_money(sleeve_target_krw)} | "
                f"equity={_fmt_money(nav)}"
            )
        if max_daily_orders_cap is None:
            perf_lines.append("Daily orders cap unlimited")
        else:
            perf_lines.append(
                f"Daily orders left {int(order_budget['remaining_daily_orders'])}/{max_daily_orders_cap}"
            )

        style_lines = [
            "Style: Long-horizon compounding (weeks to months), low turnover preferred unless thesis materially changes."
        ]
        if hold_days_values:
            avg_hold = sum(hold_days_values) / float(len(hold_days_values))
            style_lines.append(f"Current avg hold period {avg_hold:.1f}d across {len(hold_days_values)} positions.")
        if turnover_ratio >= 0.30:
            style_lines.append("Today turnover is elevated; avoid additional rotation unless conviction improved.")
        else:
            style_lines.append("Use incremental adds/trims; avoid reacting to single-point noise.")
        investment_style_context = "\n".join(style_lines)

        graph_rows = self._fetch_graph_neighbors(memory_rows[:6])
        memory_context = self._compress_memory_context(memory_rows[:6])
        graph_context = self._compress_graph_context(memory_rows[:6], graph_rows)
        if graph_context:
            memory_context = f"{memory_context}\n{graph_context}".strip() if memory_context else graph_context
        memory_event_counts: dict[str, int] = {}
        for row in memory_rows[:12]:
            key = str(row.get("event_type") or "").strip().lower() or "unknown"
            memory_event_counts[key] = memory_event_counts.get(key, 0) + 1
        logger.info(
            "[cyan]context summary[/cyan] agent=%s cycle_id=%s memories=%d active_theses=%d graph=%d board=%d types=%s",
            agent_id,
            str(cycle_id or "").strip() or "-",
            len(memory_rows),
            len(active_thesis_rows),
            len(graph_rows),
            len(board_rows),
            memory_event_counts,
        )

        board_context = ""
        if board_rows:
            board_lines = [
                f"{str(r.get('agent_id') or '?')}: {self._trim_text(r.get('title'), max_len=80)}"
                for r in board_rows[:6]
            ]
            board_context = "\n".join(board_lines)

        # Currency/FX context
        fx_info: dict[str, Any] = {}
        if snapshot.usd_krw_rate > 0:
            fx_info["usd_krw_rate"] = snapshot.usd_krw_rate
        if snapshot.cash_foreign > 0 and snapshot.cash_foreign_currency:
            fx_info["cash_foreign"] = snapshot.cash_foreign
            fx_info["cash_foreign_currency"] = snapshot.cash_foreign_currency

        # Determine position currencies
        pos_currencies = {pos.quote_currency for pos in snapshot.positions.values() if pos.quote_currency}
        if pos_currencies:
            fx_info["position_currencies"] = sorted(pos_currencies)

        if fx_info:
            perf_lines.append(
                "FX "
                + " | ".join(f"{k}={v}" for k, v in fx_info.items())
            )
        # Explicit cash breakdown so agents never confuse KRW vs USD
        if _is_us_market:
            usd_cash = snapshot.cash_foreign if snapshot.cash_foreign > 0 else (cash_krw / _fx if _fx > 0 else 0.0)
            perf_lines.append(f"Cash ${_fmt_usd(usd_cash)} USD (={_fmt_krw(cash_krw)} KRW equivalent)")
        else:
            if snapshot.cash_foreign > 0:
                perf_lines.append(f"Cash {_fmt_krw(cash_krw)} KRW | Foreign ${_fmt_usd(snapshot.cash_foreign)} USD (not available for KOSPI)")
            else:
                perf_lines.append(f"Cash {_fmt_krw(cash_krw)} KRW")

        return {
            "agent_id": agent_id,
            "target_market": _target_market,
            "portfolio": snapshot.model_dump(mode="json"),
            "ticker_names": _safe_json(_ticker_names),
            "performance": _safe_json(performance),
            "risk_policy": _safe_json(risk_policy),
            "order_budget": _safe_json(order_budget),
            "sleeve_state": _safe_json(sleeve_state),
            "fx_info": _safe_json(fx_info),
            "performance_context": "\n".join(perf_lines).strip(),
            "active_thesis_context": active_thesis_context,
            "memory_context": memory_context,
            "graph_context": graph_context,
            "board_context": board_context,
            "market_features": _safe_json(market_rows),
            "active_theses": _safe_json(active_thesis_rows),
            "memory_events": _safe_json(memory_rows),
            "graph_events": _safe_json(graph_rows),
            "board_posts": _safe_json(board_rows),
            "investment_style_context": investment_style_context,
            "notes": "Use optional tools for quant analysis and strategy references when needed.",
        }
