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
from arena.memory.candidates import CANDIDATE_MEMORY_EVENT_TYPES
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
    memory_graph_semantic_triples_boost_bonus_base,
    memory_graph_semantic_triples_boost_bonus_cap,
    memory_graph_semantic_triples_boost_enabled,
    memory_graph_semantic_triples_inject_enabled,
    memory_graph_semantic_triples_max_candidates,
    memory_graph_semantic_triples_max_relation_context_items,
    memory_graph_semantic_triples_min_confidence,
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
from arena.memory.relations import semantic_entity_node_id, ticker_node_id
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
_CANDIDATE_MEMORY_EVENT_TYPES = set(CANDIDATE_MEMORY_EVENT_TYPES)
_MEMORY_EXECUTION_NOISE_RE = re.compile(
    r"\b(?:qty|status|policy|broker|order_id|filled|avg|message)=\S+",
    re.IGNORECASE,
)
_MEMORY_WHITESPACE_RE = re.compile(r"\s+")


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

    def _extract_summary_ticker_keywords(self, summary: object) -> list[str]:
        """Best-effort ticker extraction from summary text only."""
        keywords: list[str] = []
        for token in _TICKER_TOKEN_RE.findall(str(summary or "")):
            if token in _MEMORY_TICKER_STOPWORDS:
                continue
            self._append_keyword(keywords, token)
            if len(keywords) >= 4:
                break
        return keywords[:4]

    @staticmethod
    def _summary_side_fallback_allowed(event_type: object) -> bool:
        """Allows summary keyword side inference only for execution-like memory rows."""
        token = str(event_type or "").strip().lower()
        return token in {"trade_execution"}

    @staticmethod
    def _extract_summary_side(summary: object) -> str:
        """Returns a BUY/SELL/HOLD token only when the summary states it explicitly."""
        normalized = f" {str(summary or '').upper()} "
        for token in ("BUY", "SELL", "HOLD"):
            if f" {token} " in normalized:
                return token
        return ""

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

    def _extract_memory_tickers(self, row: dict[str, Any]) -> tuple[list[str], list[str], str]:
        """Extracts canonical tickers first, then summary-derived fallback tickers."""
        canonical: list[str] = []
        sources: list[str] = []
        context_tags = row.get("context_tags")
        if isinstance(context_tags, dict):
            for token in context_tags.get("tickers") or []:
                self._append_keyword(canonical, token)
                if len(canonical) >= 4:
                    break
            if canonical:
                sources.append("context_tags")
        payload = self._parse_memory_payload(row)
        if payload:
            before = list(canonical)
            self._collect_ticker_keywords_from_value(payload, canonical)
            canonical = canonical[:4]
            if canonical and canonical != before:
                sources.append("payload")
            elif canonical and "context_tags" not in sources:
                sources.append("payload")
        if canonical:
            return canonical[:4], [], "+".join(sources)

        derived = self._extract_summary_ticker_keywords(row.get("summary"))
        if derived:
            return [], derived, "summary_regex"
        return [], [], ""

    def _extract_memory_side(self, row: dict[str, Any]) -> tuple[str, str, str]:
        """Returns canonical side first, then summary-derived fallback side."""
        payload = self._parse_memory_payload(row)
        intent = payload.get("intent") if isinstance(payload, dict) else None
        if isinstance(intent, dict):
            token = str(intent.get("side") or "").strip().upper()
            if token in {"BUY", "SELL", "HOLD"}:
                return token, "", "payload"
        if self._summary_side_fallback_allowed(row.get("event_type")):
            token = self._extract_summary_side(row.get("summary"))
            if token:
                return "", token, "summary_keyword"
        return "", "", ""

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
        canonical_tickers, derived_tickers, ticker_source = self._extract_memory_tickers(normalized)
        normalized["canonical_tickers"] = canonical_tickers
        normalized["derived_tickers"] = derived_tickers
        normalized["ticker_source"] = ticker_source
        normalized["tickers"] = canonical_tickers or derived_tickers
        canonical_side, derived_side, side_source = self._extract_memory_side(normalized)
        normalized["canonical_side"] = canonical_side
        normalized["derived_side"] = derived_side
        normalized["side_source"] = side_source
        normalized["side"] = canonical_side or derived_side
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
        if token == "candidate_thesis":
            return 0.34
        if token == "candidate_watchlist":
            return 0.24
        if token == "candidate_rejected":
            return 0.20
        if token == "candidate_screen_hit":
            return 0.12
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
            for token in (row.get("canonical_tickers") or [])
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

    def _relation_seed_node_ids(
        self,
        *,
        focus_tickers: list[str],
        current_context_tags: dict[str, list[str]],
    ) -> list[str]:
        seeds: list[str] = []
        for ticker in focus_tickers[:8]:
            node_id = ticker_node_id(ticker)
            if node_id and node_id not in seeds:
                seeds.append(node_id)
        for entity_type, key in [
            ("regime", "regimes"),
            ("strategy_tag", "strategies"),
            ("sector", "sectors"),
        ]:
            for label in (current_context_tags.get(key) or [])[:4]:
                node_id = semantic_entity_node_id(entity_type, label)
                if node_id and node_id not in seeds:
                    seeds.append(node_id)
        return seeds

    def _relation_trace_text(self, row: dict[str, Any]) -> str:
        predicate = str(row.get("relation_predicate") or "").strip().lower()
        object_type = str(row.get("relation_object_type") or "").strip().lower()
        object_label = self._trim_text(row.get("relation_object_label"), max_len=60)
        evidence = self._trim_text(row.get("relation_evidence_text"), max_len=120)
        bits = [part for part in [predicate, object_type, object_label] if part]
        prefix = " ".join(bits) if bits else "relation match"
        return f"{prefix}: {evidence}" if evidence else prefix

    def _relation_memory_candidates(
        self,
        *,
        agent_id: str,
        focus_tickers: list[str],
        current_context_tags: dict[str, list[str]],
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, list[str]]]:
        if not memory_graph_semantic_triples_boost_enabled(self.settings.memory_policy):
            return [], {}, {}
        loader = getattr(self.repo, "memory_relation_memory_candidates", None)
        if not callable(loader):
            return [], {}, {}
        seed_node_ids = self._relation_seed_node_ids(
            focus_tickers=focus_tickers,
            current_context_tags=current_context_tags,
        )
        if not seed_node_ids:
            return [], {}, {}
        max_candidates = min(
            max(1, int(limit)),
            memory_graph_semantic_triples_max_candidates(self.settings.memory_policy),
        )
        try:
            rows = list(
                loader(
                    agent_id=agent_id,
                    seed_node_ids=seed_node_ids,
                    trading_mode=self.settings.trading_mode,
                    min_confidence=memory_graph_semantic_triples_min_confidence(self.settings.memory_policy),
                    limit=max_candidates,
                )
            )
        except Exception as exc:
            logger.warning("[yellow]relation memory candidates failed[/yellow] agent=%s err=%s", agent_id, str(exc))
            return [], {}, {}

        bonus_base = memory_graph_semantic_triples_boost_bonus_base(self.settings.memory_policy)
        bonus_cap = memory_graph_semantic_triples_boost_bonus_cap(self.settings.memory_policy)
        relation_bonus: dict[str, float] = {}
        relation_traces: dict[str, list[str]] = {}
        normalized_rows: list[dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            row = dict(raw)
            event_id = str(row.get("event_id") or "").strip()
            if not event_id:
                continue
            try:
                confidence = max(0.0, min(float(row.get("relation_confidence") or 0.0), 1.0))
            except (TypeError, ValueError):
                confidence = 0.0
            bonus = min(bonus_cap, bonus_base * confidence)
            relation_bonus[event_id] = max(float(relation_bonus.get(event_id, 0.0)), bonus)
            trace = self._relation_trace_text(row)
            if trace:
                traces = relation_traces.setdefault(event_id, [])
                if trace not in traces:
                    traces.append(trace)
            row["relation_candidate"] = True
            normalized_rows.append(row)
        return normalized_rows, relation_bonus, relation_traces

    def _rerank_memory_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        active_tickers: set[str],
        current_context_tags: dict[str, list[str]],
        vector_bonus: dict[str, float],
        relation_bonus: dict[str, float] | None = None,
        relation_traces: dict[str, list[str]] | None = None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Ranks memory candidates by actionability rather than raw insertion order."""
        ranked: list[dict[str, Any]] = []
        seen: set[str] = set()
        relation_bonus = relation_bonus or {}
        relation_traces = relation_traces or {}
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
            event_id = str(row.get("event_id") or "").strip()
            rel_bonus = float(relation_bonus.get(event_id, 0.0))
            retrieval_score += rel_bonus
            if self._is_simulated_trade_memory(row):
                retrieval_score -= 0.22
            row["outcome_label"] = self._outcome_label(row.get("outcome_score"))
            resolved_effective_score = self._resolved_effective_score(row)
            if resolved_effective_score is not None:
                row["effective_score"] = round(resolved_effective_score, 4)
            if rel_bonus > 0:
                row["relation_boost"] = round(rel_bonus, 4)
                traces = relation_traces.get(event_id) or []
                if traces:
                    row["relation_traces"] = traces[:3]
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

    def _memory_prompt_label(self, row: dict[str, Any]) -> str:
        """Returns the role a memory should play in the prompt."""
        event_type = str(row.get("event_type") or "").strip().lower()
        side = str(row.get("canonical_side") or row.get("derived_side") or row.get("side") or "").strip().upper()
        if event_type == "trade_execution":
            if side == "BUY":
                return "prior entry"
            if side == "SELL":
                return "prior trim"
            if side == "HOLD":
                return "prior hold"
            return "prior trade"
        if event_type == "strategy_reflection":
            return "lesson"
        if event_type == "manual_note":
            return "manual note"
        if event_type == "thesis_invalidated":
            return "invalidated thesis"
        if event_type == "thesis_realized":
            return "realized thesis"
        if event_type == "thesis_update":
            return "thesis update"
        if event_type == "thesis_open":
            return "thesis open"
        if event_type == "candidate_thesis":
            return "candidate thesis"
        if event_type == "candidate_rejected":
            return "rejected candidate"
        if event_type == "candidate_watchlist":
            return "watchlist candidate"
        if event_type == "candidate_screen_hit":
            return "screened candidate"
        if event_type == "react_tools_summary":
            return "prior tool context"
        return event_type.replace("_", " ") if event_type else "memory"

    @staticmethod
    def _strip_memory_execution_noise(summary: str) -> str:
        """Removes broker/order plumbing while preserving the decision rationale."""
        text = _MEMORY_EXECUTION_NOISE_RE.sub("", str(summary or ""))
        text = text.replace("rationale=", "rationale: ")
        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r"([,;:]){2,}", r"\1", text)
        text = _MEMORY_WHITESPACE_RE.sub(" ", text).strip(" -;,.")
        return text

    def _memory_prompt_text(self, row: dict[str, Any]) -> str:
        """Returns the prompt-facing summary without changing stored memory data."""
        event_type = str(row.get("event_type") or "").strip().lower()
        summary = self._trim_text(row.get("summary"), max_len=220)
        if event_type == "trade_execution":
            summary = self._strip_memory_execution_noise(summary)
        return self._trim_text(summary, max_len=150)

    def _memory_prompt_meta_bits(self, row: dict[str, Any]) -> list[str]:
        bits: list[str] = []
        canonical_tickers = [str(t).strip().upper() for t in (row.get("canonical_tickers") or []) if str(t).strip()]
        derived_tickers = [str(t).strip().upper() for t in (row.get("derived_tickers") or []) if str(t).strip()]
        if canonical_tickers:
            bits.append("/".join(canonical_tickers[:2]))
        elif derived_tickers:
            bits.append(f"~{'/'.join(derived_tickers[:2])}")
        label = self._memory_prompt_label(row)
        if label:
            bits.append(label)
        canonical_side = str(row.get("canonical_side") or "").strip().upper()
        derived_side = str(row.get("derived_side") or "").strip().upper()
        if derived_side and not canonical_side:
            bits.append(f"~{derived_side}")
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
        return bits

    def _format_memory_line(self, row: dict[str, Any]) -> str:
        """Formats one deterministic, prompt-facing memory line."""
        bits = self._memory_prompt_meta_bits(row)
        meta = f"[{' | '.join(bits)}] " if bits else ""
        return f"- {meta}{self._memory_prompt_text(row)}"

    def _compress_memory_context(self, rows: list[dict[str, Any]]) -> str:
        """Builds a sectioned memory summary instead of a flat log dump."""
        if not rows:
            return ""

        if any(str(row.get("memory_track") or "").strip() for row in rows):
            sections: list[str] = []
            used: set[str] = set()
            for title, track in (
                ("Portfolio Memory", "portfolio"),
                ("Candidate Memory", "candidate"),
                ("Neutral Lessons", "neutral"),
            ):
                picked = [
                    row for row in rows
                    if str(row.get("memory_track") or "").strip().lower() == track
                    and str(row.get("event_id") or row.get("summary") or "") not in used
                ]
                if not picked:
                    continue
                sections.append("\n".join([f"{title}:"] + [self._format_memory_line(row) for row in picked]))
                for row in picked:
                    used.add(str(row.get("event_id") or row.get("summary") or ""))
            return "\n".join(sections)

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
        summary = self._trim_text(self._strip_memory_execution_noise(str(row.get("summary") or "")), max_len=120)
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
                f"- {self._memory_prompt_text(seed)} || " + " ; ".join(descriptors)
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

    def _build_research_context(self, *, focus_tickers: list[str]) -> tuple[str, list[dict[str, Any]]]:
        """Builds a compact stored-research block for prompt/debug visibility."""
        loader = getattr(self.repo, "get_research_briefings", None)
        if not callable(loader):
            return "", []
        tickers = [str(t or "").strip().upper() for t in focus_tickers if str(t or "").strip()]
        tickers = list(dict.fromkeys(tickers))[:4]
        try:
            rows = list(
                loader(
                    tickers=tickers or None,
                    categories=["global_market", "geopolitical", "sector"],
                    limit=6,
                    trading_mode=self.settings.trading_mode,
                )
            )
        except Exception as exc:
            logger.warning("[yellow]research context load failed[/yellow] err=%s", str(exc))
            return "", []
        lines: list[str] = []
        compact_rows: list[dict[str, Any]] = []
        for row in rows[:6]:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            category = str(row.get("category") or "").strip().lower()
            headline = self._trim_text(row.get("headline"), max_len=90)
            summary = self._trim_text(row.get("summary"), max_len=160)
            label = ticker or category or "research"
            text = " - ".join(part for part in [headline, summary] if part)
            if text:
                lines.append(f"- [{label}] {text}")
            compact_rows.append(
                {
                    "briefing_id": str(row.get("briefing_id") or "").strip(),
                    "created_at": row.get("created_at"),
                    "ticker": ticker or None,
                    "category": category or None,
                    "headline": headline,
                    "summary": summary,
                }
            )
        return "\n".join(lines), compact_rows

    def _compress_relation_context(self, rows: list[dict[str, Any]]) -> str:
        if not memory_graph_semantic_triples_inject_enabled(self.settings.memory_policy):
            return ""
        max_items = memory_graph_semantic_triples_max_relation_context_items(self.settings.memory_policy)
        if max_items <= 0:
            return ""
        lines: list[str] = ["Relation Hints:"]
        seen: set[str] = set()
        for row in rows:
            event_summary = self._trim_text(row.get("summary"), max_len=90)
            for trace in row.get("relation_traces") or []:
                text = self._trim_text(trace, max_len=160)
                if not text or text in seen:
                    continue
                seen.add(text)
                if event_summary:
                    lines.append(f"- {text} | memory: {event_summary}")
                else:
                    lines.append(f"- {text}")
                if len(lines) - 1 >= max_items:
                    break
            if len(lines) - 1 >= max_items:
                break
        return "\n".join(lines) if len(lines) > 1 else ""

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
        relation_rows, relation_bonus, relation_traces = self._relation_memory_candidates(
            agent_id=agent_id,
            focus_tickers=[str(t or "").strip().upper() for t in focus_tickers if str(t or "").strip()],
            current_context_tags=current_context_tags,
            limit=limit,
        )
        if not vector_rows and not relation_rows:
            return []

        return self._rerank_memory_rows(
            [*vector_rows, *relation_rows],
            active_tickers=active_tickers,
            current_context_tags=current_context_tags,
            vector_bonus=vector_bonus,
            relation_bonus=relation_bonus,
            relation_traces=relation_traces,
            limit=limit,
        )

    def _fetch_candidate_memory_rows(
        self,
        *,
        agent_id: str,
        focus_tickers: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        loader = getattr(self.repo, "candidate_memory_events", None)
        if not callable(loader):
            return []
        try:
            rows = loader(
                agent_id=agent_id,
                exclude_tickers=[str(t or "").strip().upper() for t in focus_tickers if str(t or "").strip()],
                limit=max(1, int(limit)),
                trading_mode=self.settings.trading_mode,
            )
        except Exception as exc:
            logger.warning("[yellow]candidate memory load failed[/yellow] agent=%s err=%s", agent_id, str(exc))
            return []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized = self._normalize_memory_row(row)
            if self._memory_is_expired(normalized):
                continue
            normalized["memory_track"] = "candidate"
            out.append(normalized)
        return out

    @staticmethod
    def _memory_row_key(row: dict[str, Any]) -> str:
        return str(row.get("event_id") or row.get("summary") or "").strip()

    @staticmethod
    def _memory_row_tickers(row: dict[str, Any]) -> set[str]:
        tickers: set[str] = set()
        for field in ("canonical_tickers", "derived_tickers", "tickers"):
            value = row.get(field)
            if isinstance(value, list):
                tickers.update(str(token or "").strip().upper() for token in value if str(token or "").strip())
        return {token for token in tickers if token}

    def _memory_balance_bucket(self, row: dict[str, Any], *, held_tickers: set[str]) -> str:
        event_type = str(row.get("event_type") or "").strip().lower()
        if event_type in _CANDIDATE_MEMORY_EVENT_TYPES:
            return "candidate"
        tickers = self._memory_row_tickers(row)
        if tickers & held_tickers:
            return "held"
        if tickers:
            return "nonheld"
        return "neutral"

    def _annotate_memory_track(self, row: dict[str, Any], *, held_tickers: set[str], default_track: str) -> dict[str, Any]:
        annotated = dict(row)
        event_type = str(annotated.get("event_type") or "").strip().lower()
        row_tickers = self._memory_row_tickers(annotated)
        if event_type in _CANDIDATE_MEMORY_EVENT_TYPES:
            track = "candidate"
        elif row_tickers & held_tickers:
            track = "portfolio"
        elif default_track == "portfolio" and (not held_tickers or not row_tickers):
            track = "neutral"
        elif default_track in {"candidate", "neutral", "portfolio"}:
            track = default_track
        else:
            track = "neutral"
        annotated["memory_track"] = track
        annotated["balance_bucket"] = self._memory_balance_bucket(annotated, held_tickers=held_tickers)
        return annotated

    def _append_memory_prompt_rows(
        self,
        target: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        *,
        seen: set[str],
        held_tickers: set[str],
        ticker_counts: dict[str, int],
        event_counts: dict[str, int],
        limit: int,
        track: str,
        same_ticker_cap: int = 2,
        trade_cap: int = 2,
    ) -> None:
        if limit <= 0 or len(target) >= limit:
            return
        for row in rows:
            key = self._memory_row_key(row)
            if not key or key in seen:
                continue
            event_type = str(row.get("event_type") or "").strip().lower()
            if event_type == "trade_execution" and event_counts.get("trade_execution", 0) >= trade_cap:
                continue
            tickers = self._memory_row_tickers(row)
            if tickers and all(ticker_counts.get(ticker, 0) >= same_ticker_cap for ticker in tickers):
                continue
            annotated = self._annotate_memory_track(row, held_tickers=held_tickers, default_track=track)
            target.append(annotated)
            seen.add(key)
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            for ticker in tickers:
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            if len(target) >= limit:
                return

    @staticmethod
    def _cash_ratio(snapshot: AccountSnapshot) -> float:
        try:
            equity = float(snapshot.total_equity_krw or 0.0)
        except (TypeError, ValueError):
            equity = 0.0
        if equity <= 0:
            return 0.0
        try:
            return max(0.0, min(float(snapshot.cash_krw or 0.0) / equity, 1.5))
        except (TypeError, ValueError):
            return 0.0

    def _select_prompt_memory_rows(
        self,
        *,
        portfolio_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        neutral_rows: list[dict[str, Any]],
        snapshot: AccountSnapshot,
        focus_tickers: list[str],
        total_limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Selects prompt-visible memories with track slots and repetition caps."""
        cap = max(1, min(int(total_limit), 6))
        held_tickers = {str(t or "").strip().upper() for t in focus_tickers if str(t or "").strip()}
        candidate_slots = 0
        if candidate_rows:
            candidate_slots = min(2, cap)
            if not held_tickers or self._cash_ratio(snapshot) >= 0.30:
                candidate_slots = min(3, cap)
        neutral_slots = 1 if neutral_rows and cap - candidate_slots > 0 else 0
        portfolio_slots = max(cap - candidate_slots - neutral_slots, 0)
        if portfolio_rows and portfolio_slots <= 0:
            portfolio_slots = 1
            if candidate_slots >= neutral_slots and candidate_slots > 0:
                candidate_slots -= 1
            elif neutral_slots > 0:
                neutral_slots -= 1

        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        ticker_counts: dict[str, int] = {}
        event_counts: dict[str, int] = {}
        self._append_memory_prompt_rows(
            selected,
            portfolio_rows,
            seen=seen,
            held_tickers=held_tickers,
            ticker_counts=ticker_counts,
            event_counts=event_counts,
            limit=portfolio_slots,
            track="portfolio",
        )
        self._append_memory_prompt_rows(
            selected,
            candidate_rows,
            seen=seen,
            held_tickers=held_tickers,
            ticker_counts=ticker_counts,
            event_counts=event_counts,
            limit=len(selected) + candidate_slots,
            track="candidate",
        )
        self._append_memory_prompt_rows(
            selected,
            neutral_rows,
            seen=seen,
            held_tickers=held_tickers,
            ticker_counts=ticker_counts,
            event_counts=event_counts,
            limit=len(selected) + neutral_slots,
            track="neutral",
        )

        tail = [*portfolio_rows, *candidate_rows, *neutral_rows]
        tail.sort(
            key=lambda r: (
                float(r.get("retrieval_score") or 0.0),
                float(r.get("importance_score") or 0.0),
                -int(r.get("age_days") or 9999) if isinstance(r.get("age_days"), int) else -9999,
            ),
            reverse=True,
        )
        self._append_memory_prompt_rows(
            selected,
            tail,
            seen=seen,
            held_tickers=held_tickers,
            ticker_counts=ticker_counts,
            event_counts=event_counts,
            limit=cap,
            track="neutral",
        )
        return selected[:cap]

    def _combine_memory_retrieval_rows(
        self,
        *,
        portfolio_rows: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        neutral_rows: list[dict[str, Any]],
        focus_tickers: list[str],
    ) -> list[dict[str, Any]]:
        held_tickers = {str(t or "").strip().upper() for t in focus_tickers if str(t or "").strip()}
        combined: list[dict[str, Any]] = []
        seen: set[str] = set()
        for track, rows in (
            ("portfolio", portfolio_rows),
            ("candidate", candidate_rows),
            ("neutral", neutral_rows),
        ):
            for row in rows:
                key = self._memory_row_key(row)
                if not key or key in seen:
                    continue
                combined.append(self._annotate_memory_track(row, held_tickers=held_tickers, default_track=track))
                seen.add(key)
        return combined

    def _log_memory_balance_metrics(
        self,
        *,
        agent_id: str,
        cycle_id: str | None,
        retrieved_rows: list[dict[str, Any]],
        prompt_rows: list[dict[str, Any]],
        focus_tickers: list[str],
    ) -> None:
        held_tickers = {str(t or "").strip().upper() for t in focus_tickers if str(t or "").strip()}

        def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
            out = {"held": 0, "candidate": 0, "nonheld": 0, "neutral": 0, "trade_execution": 0}
            for row in rows:
                bucket = self._memory_balance_bucket(row, held_tickers=held_tickers)
                out[bucket] = out.get(bucket, 0) + 1
                if str(row.get("event_type") or "").strip().lower() == "trade_execution":
                    out["trade_execution"] += 1
            return out

        retrieved = _counts(retrieved_rows)
        prompt = _counts(prompt_rows)
        retrieved_total = max(1, len(retrieved_rows))
        prompt_total = max(1, len(prompt_rows))
        logger.info(
            "[cyan]memory balance[/cyan] agent=%s cycle_id=%s retrieved_held=%.2f prompt_held=%.2f prompt_candidate=%.2f prompt_neutral=%.2f trades=%d",
            agent_id,
            str(cycle_id or "").strip() or "-",
            retrieved.get("held", 0) / retrieved_total,
            prompt.get("held", 0) / prompt_total,
            prompt.get("candidate", 0) / prompt_total,
            prompt.get("neutral", 0) / prompt_total,
            prompt.get("trade_execution", 0),
        )

    def _log_memory_access_events(
        self,
        *,
        agent_id: str,
        rows: list[dict[str, Any]],
        query: str | list[str],
        cycle_id: str | None = None,
        prompt_rows: list[dict[str, Any]] | None = None,
        focus_tickers: list[str] | None = None,
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
        visible_rows = prompt_rows if prompt_rows is not None else rows[:6]
        prompt_keys = {
            str(row.get("event_id") or row.get("summary") or "").strip()
            for row in visible_rows
            if str(row.get("event_id") or row.get("summary") or "").strip()
        }
        held_tickers = {
            str(token or "").strip().upper()
            for token in (focus_tickers or [])
            if str(token or "").strip()
        }
        accessed_at = utc_now()
        payload_rows: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            event_id = str(row.get("event_id") or "").strip()
            if not event_id:
                continue
            row_key = str(row.get("event_id") or row.get("summary") or "").strip()
            canonical_tickers = [
                str(token or "").strip().upper()
                for token in (row.get("canonical_tickers") or [])
                if str(token or "").strip()
            ]
            derived_tickers = [
                str(token or "").strip().upper()
                for token in (row.get("derived_tickers") or [])
                if str(token or "").strip()
            ]
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
                        "memory_track": str(row.get("memory_track") or "").strip().lower() or None,
                        "balance_bucket": self._memory_balance_bucket(row, held_tickers=held_tickers),
                        "canonical_tickers": canonical_tickers,
                        "derived_tickers": derived_tickers,
                        "relation_boost": row.get("relation_boost"),
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
        candidate_memory_rows = self._fetch_candidate_memory_rows(
            agent_id=agent_id,
            focus_tickers=focus_tickers,
            limit=max(6, min(self.settings.context_max_memory_events, 12)),
        )
        legacy_memory_rows = self._merge_memory_query_tracks(
            primary_rows=primary_memory_rows,
            opportunity_rows=opportunity_memory_rows,
            total_limit=self.settings.context_max_memory_events,
        )
        memory_rows = self._combine_memory_retrieval_rows(
            portfolio_rows=legacy_memory_rows,
            candidate_rows=candidate_memory_rows,
            neutral_rows=opportunity_memory_rows,
            focus_tickers=focus_tickers,
        )
        prompt_memory_rows = self._select_prompt_memory_rows(
            portfolio_rows=primary_memory_rows,
            candidate_rows=candidate_memory_rows,
            neutral_rows=opportunity_memory_rows,
            snapshot=snapshot,
            focus_tickers=focus_tickers,
            total_limit=min(6, self.settings.context_max_memory_events),
        )
        research_context, research_rows = self._build_research_context(focus_tickers=focus_tickers)
        active_thesis_rows = self._active_thesis_rows(agent_id=agent_id, focus_tickers=focus_tickers)
        active_thesis_context = self._compress_active_thesis_context(active_thesis_rows)
        logged_queries = list(memory_query)
        if opportunity_query:
            logged_queries.append(opportunity_query)
        if candidate_memory_rows:
            logged_queries.append("candidate memory track")
        self._log_memory_balance_metrics(
            agent_id=agent_id,
            cycle_id=cycle_id,
            retrieved_rows=memory_rows,
            prompt_rows=prompt_memory_rows,
            focus_tickers=focus_tickers,
        )
        self._log_memory_access_events(
            agent_id=agent_id,
            rows=memory_rows,
            query=logged_queries,
            cycle_id=cycle_id,
            prompt_rows=prompt_memory_rows,
            focus_tickers=focus_tickers,
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

        # Determine target_market early for currency-aware display
        _target_market = ""
        if agent_config and agent_config.target_market:
            _target_market = agent_config.target_market
        if not _target_market:
            _target_market = self.settings.kis_target_market
        _is_us_market = _target_market.lower().strip() in {"us", "nasdaq", "nyse", "amex"}
        _fx = snapshot.usd_krw_rate if snapshot.usd_krw_rate > 0 else 0.0

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
            return "USD" if _is_us_market and _fx > 0 else "KRW"

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
        if _is_us_market:
            if snapshot.cash_foreign > 0:
                order_budget["cash_usd"] = snapshot.cash_foreign
            elif _fx > 0:
                order_budget["cash_usd"] = cash_krw / _fx

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
        style_lines = [
            "Style: Long-horizon compounding (weeks to months), low turnover preferred unless thesis materially changes."
        ]
        if turnover_ratio >= 0.30:
            style_lines.append("Today turnover is elevated; avoid additional rotation unless conviction improved.")
        else:
            style_lines.append("Use incremental adds/trims; avoid reacting to single-point noise.")
        investment_style_context = "\n".join(style_lines)

        graph_rows = self._fetch_graph_neighbors(prompt_memory_rows)
        memory_context = self._compress_memory_context(prompt_memory_rows)
        graph_context = self._compress_graph_context(prompt_memory_rows, graph_rows)
        relation_context = self._compress_relation_context(prompt_memory_rows)
        memory_event_counts: dict[str, int] = {}
        for row in prompt_memory_rows[:12]:
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
            "market_context": _safe_json(market_rows),
            "research_context": research_context,
            "relation_context": relation_context,
            "memory_context": memory_context,
            "graph_context": graph_context,
            "board_context": board_context,
            "market_features": _safe_json(market_rows),
            "research_briefings": _safe_json(research_rows),
            "active_theses": _safe_json(active_thesis_rows),
            "memory_events": _safe_json(prompt_memory_rows),
            "retrieved_memory_events": _safe_json(memory_rows),
            "graph_events": _safe_json(graph_rows),
            "board_posts": _safe_json(board_rows),
            "investment_style_context": investment_style_context,
            "notes": "Use optional tools for quant analysis and strategy references when needed.",
        }
