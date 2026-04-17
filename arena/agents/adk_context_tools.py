from __future__ import annotations

import inspect
import json
import logging
import os
from datetime import date, datetime
from typing import Any

from arena.config import Settings, research_generation_status
from arena.data.bq import BigQueryRepository
from arena.market_feature_normalization import daily_history_sources
from arena.market_sources import live_market_sources_for_markets
from arena.memory.policy import (
    memory_embed_cache_max,
    memory_peer_lessons_enabled,
    memory_vector_search_enabled,
)
from arena.models import utc_now
from arena.tools.allocation import optimize_hrp

_PEER_LESSON_SOURCES = frozenset({"memory_compaction", "thesis_chain_compaction"})
_PUBLIC_RESEARCH_CATEGORIES = ("global_market", "geopolitical", "sector_trends")
logger = logging.getLogger(__name__)


class _ContextTools:
    """Exposes per-cycle context as callable ADK tools."""

    def __init__(
        self,
        *,
        repo: BigQueryRepository,
        settings: Settings,
        agent_id: str,
        memory_store=None,
        tenant_id: str = "local",
    ):
        self.repo = repo
        self.settings = settings
        self.agent_id = agent_id
        self.tenant_id = str(tenant_id or "").strip().lower() or "local"
        self._context: dict[str, Any] = {}
        self._memory_store = memory_store
        from arena.memory.vector import VectorStore

        self._vector_store = VectorStore(
            project=repo.project,
            location=repo.location,
            embed_cache_max=memory_embed_cache_max(settings.memory_policy),
        )

    def set_context(self, context: dict[str, Any]) -> None:
        """Stores current cycle context for subsequent tool calls."""
        self._context = context

    def _sources(self) -> list[str] | None:
        if self.settings.trading_mode != "live":
            return None
        return live_market_sources_for_markets(self.settings.kis_target_market) or None

    @staticmethod
    def _coerce_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _float_or_none(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _benchmark_period_basis(self, perf: dict[str, Any]) -> tuple[date | None, float | None, str]:
        candidates = (
            ("current_sleeve_pnl_ratio", perf.get("current_sleeve_initialized_at"), perf.get("current_sleeve_pnl_ratio")),
            ("cumulative_pnl_ratio", perf.get("initialized_at"), perf.get("cumulative_pnl_ratio")),
            ("pnl_ratio", perf.get("current_sleeve_initialized_at") or perf.get("initialized_at"), perf.get("pnl_ratio")),
        )
        for metric, raw_start, raw_return in candidates:
            start_dt = self._coerce_datetime(raw_start)
            ret = self._float_or_none(raw_return)
            if start_dt is None or ret is None:
                continue
            return start_dt.date(), ret, metric
        return None, None, ""

    def _benchmark_period_bases(self, perf: dict[str, Any]) -> dict[str, tuple[date, float, str]]:
        bases: dict[str, tuple[date, float, str]] = {}
        current_start = self._coerce_datetime(perf.get("current_sleeve_initialized_at"))
        current_ret = self._float_or_none(perf.get("current_sleeve_pnl_ratio"))
        if current_start is not None and current_ret is not None:
            bases["current_sleeve"] = (current_start.date(), current_ret, "current_sleeve_pnl_ratio")

        cumulative_start = self._coerce_datetime(perf.get("initialized_at"))
        cumulative_ret = self._float_or_none(perf.get("cumulative_pnl_ratio"))
        if cumulative_start is not None and cumulative_ret is not None:
            bases["cumulative"] = (cumulative_start.date(), cumulative_ret, "cumulative_pnl_ratio")

        if not bases:
            fallback_start = self._coerce_datetime(perf.get("current_sleeve_initialized_at") or perf.get("initialized_at"))
            fallback_ret = self._float_or_none(perf.get("pnl_ratio"))
            if fallback_start is not None and fallback_ret is not None:
                bases["portfolio"] = (fallback_start.date(), fallback_ret, "pnl_ratio")
        return bases

    @staticmethod
    def _series_return(series: Any) -> tuple[float | None, int, str | None, str | None]:
        if series is None:
            return None, 0, None, None
        try:
            clean = series.dropna()
        except AttributeError:
            clean = [value for value in series if value is not None]
        if len(clean) < 2:
            return None, int(len(clean)), None, None
        try:
            start_px = float(clean.iloc[0])
            end_px = float(clean.iloc[-1])
            start_date = clean.index[0].date().isoformat()
            end_date = clean.index[-1].date().isoformat()
        except AttributeError:
            start_px = float(clean[0])
            end_px = float(clean[-1])
            start_date = None
            end_date = None
        if start_px <= 0:
            return None, int(len(clean)), start_date, end_date
        return (end_px / start_px) - 1.0, int(len(clean)), start_date, end_date

    @staticmethod
    def _frame_loader_accepts_price_field(frame_loader: Any) -> bool:
        try:
            return "price_field" in inspect.signature(frame_loader).parameters
        except (TypeError, ValueError):
            return False

    def _benchmark_entry(
        self,
        *,
        scope: str,
        bench: str,
        period_start: date,
        agent_ret: float,
        agent_metric: str,
        today: date,
    ) -> dict[str, object] | None:
        lb = 120
        portfolio_start_date = period_start.isoformat()
        bench_ret: float | None = None
        bench_native_ret: float | None = None
        series_len = 0
        benchmark_start_date: str | None = None
        benchmark_end_date: str | None = None
        exact_period_match = False
        benchmark_sources = self._sources()

        frame_loader = getattr(self.repo, "get_daily_close_frame", None)
        if callable(frame_loader):
            frame = frame_loader(
                tickers=[bench],
                start=period_start,
                end=today,
                sources=benchmark_sources,
            )
            if frame is not None and not frame.empty and bench in frame.columns:
                (
                    bench_ret,
                    series_len,
                    benchmark_start_date,
                    benchmark_end_date,
                ) = self._series_return(frame[bench])
                exact_period_match = bench_ret is not None
            if self._frame_loader_accepts_price_field(frame_loader):
                native_frame = frame_loader(
                    tickers=[bench],
                    start=period_start,
                    end=today,
                    sources=benchmark_sources,
                    price_field="close_price_native",
                )
                if native_frame is not None and not native_frame.empty and bench in native_frame.columns:
                    bench_native_ret, _, _, _ = self._series_return(native_frame[bench])

        if bench_ret is None:
            lb = max(30, min(400, (today - period_start).days + 5))
            bench_closes = self.repo.get_daily_closes(
                tickers=[bench],
                lookback_days=lb,
                sources=benchmark_sources,
            )
            series = bench_closes.get(bench, [])
            if series and len(series) >= 2:
                start_px = float(series[0])
                end_px = float(series[-1])
                if start_px > 0:
                    bench_ret = (end_px / start_px) - 1.0
                    series_len = len(series)

        if bench_ret is None:
            return None

        source_basis = (
            "quote_aware"
            if any(str(source).endswith("_quote") for source in (benchmark_sources or []))
            else "daily_only"
        )
        native_currency = (
            "USD"
            if str(self.settings.kis_target_market or "").strip().lower() in {"us", "nasdaq", "nyse", "amex"}
            else "KRW"
        )
        bench_info: dict[str, object] = {
            "ticker": bench,
            "scope": scope,
            "comparison_scope": scope,
            "return": round(bench_ret, 6),
            "return_krw": round(bench_ret, 6),
            "trading_days_in_series": int(series_len),
            "period_alignment": "exact" if exact_period_match else "approximate",
            "currency_basis": "KRW",
            "price_basis": "close_price_krw",
            "source_basis": source_basis,
            "portfolio_start_date": portfolio_start_date,
            "agent_return_metric": agent_metric or "pnl_ratio",
            "agent_return": round(agent_ret, 6),
            "excess_return_vs_benchmark": round(agent_ret - bench_ret, 6),
            "alpha_definition": "simple excess return: agent_return - benchmark return_krw; not risk-adjusted alpha",
        }
        if bench_native_ret is not None:
            bench_info["return_native"] = round(bench_native_ret, 6)
            bench_info["native_currency"] = native_currency
        if benchmark_start_date:
            bench_info["benchmark_start_date"] = benchmark_start_date
        if benchmark_end_date:
            bench_info["benchmark_end_date"] = benchmark_end_date

        notes: list[str] = []
        if exact_period_match:
            if benchmark_start_date and benchmark_start_date != portfolio_start_date:
                notes.append(
                    f"Benchmark uses first available trading day on or after "
                    f"portfolio start ({portfolio_start_date} -> {benchmark_start_date})."
                )
        else:
            notes.append(
                f"Benchmark return is approximate: series covers ~{lb} calendar days "
                f"ending today, which may not align exactly with portfolio start "
                f"({portfolio_start_date}). Compare with caution."
            )
        if source_basis == "quote_aware":
            notes.append(
                "Benchmark uses quote-aware KRW prices to match current sleeve NAV; "
                "native return is provided separately when available."
            )
        else:
            notes.append("Benchmark uses daily-only KRW prices.")
        if scope == "current_sleeve":
            notes.append("Scope is current_sleeve, not cumulative/TWR history.")
        elif scope == "cumulative":
            notes.append("Scope is cumulative/TWR history, not the current sleeve reset period.")
        if notes:
            bench_info["note"] = " ".join(notes)
        return bench_info

    def search_past_experiences(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search your own past trades, lessons, and manual notes. Use this for your personal history only. For other models' distilled lessons, use search_peer_lessons."""
        if (
            not query
            or not query.strip()
            or not self._vector_store
            or not memory_vector_search_enabled(self.settings.memory_policy)
        ):
            return []
        return self._vector_store.search_similar_memories(
            agent_id=self.agent_id,
            query=query,
            limit=max(1, min(int(limit), 10)),
            trading_mode=self.settings.trading_mode,
            tenant_id=self.tenant_id,
        )

    def search_peer_lessons(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search other agents' compacted lessons in the same tenant and mode. Use this for peer takeaways, not your own history."""
        if (
            not query
            or not query.strip()
            or not self._vector_store
            or not memory_vector_search_enabled(self.settings.memory_policy)
            or not memory_peer_lessons_enabled(self.settings.memory_policy)
        ):
            return []
        raw_rows = self._vector_store.search_peer_lessons(
            agent_id=self.agent_id,
            query=query,
            limit=max(1, min(int(limit), 10)),
            trading_mode=self.settings.trading_mode,
            tenant_id=self.tenant_id,
        )
        if not raw_rows:
            return []

        hydrate = getattr(self.repo, "memory_events_by_ids_any_agent", None)
        if not callable(hydrate):
            return raw_rows

        event_ids = [
            str(row.get("event_id") or "").strip()
            for row in raw_rows
            if str(row.get("event_id") or "").strip()
        ]
        hydrated_rows = hydrate(
            event_ids=event_ids,
            trading_mode=self.settings.trading_mode,
            tenant_id=self.tenant_id,
        )
        by_id = {str(row.get("event_id") or "").strip(): row for row in hydrated_rows}

        out: list[dict[str, Any]] = []
        for row in raw_rows:
            event_id = str(row.get("event_id") or "").strip()
            full = by_id.get(event_id)
            if not isinstance(full, dict):
                continue
            payload_raw = full.get("payload_json")
            if isinstance(payload_raw, str) and payload_raw.strip():
                try:
                    payload = json.loads(payload_raw)
                except Exception:
                    payload = {}
            elif isinstance(payload_raw, dict):
                payload = payload_raw
            else:
                payload = {}
            memory_source = str(
                payload.get("source")
                or row.get("memory_source")
                or full.get("memory_source")
                or ""
            ).strip()
            if memory_source and memory_source not in _PEER_LESSON_SOURCES:
                continue
            merged = dict(row)
            merged["author_id"] = str(row.get("agent_id") or full.get("agent_id") or "").strip()
            if memory_source:
                merged["memory_source"] = memory_source
            out.append(merged)
        return out[: max(1, min(int(limit), 10))]

    def get_research_briefing(
        self,
        tickers: list[str] | None = None,
        categories: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Fetches research briefings (global market, geopolitical, sector, held-stock)."""
        max_limit = max(1, min(int(limit), 20))
        clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip()] if tickers else None
        clean_cats = [str(c).strip().lower() for c in categories if str(c).strip()] if categories else None
        rows = list(
            self.repo.get_research_briefings(
                tickers=clean_tickers or None,
                categories=clean_cats or None,
                limit=max_limit,
                trading_mode=self.settings.trading_mode,
                tenant_id=self.tenant_id,
            )
        )
        if len(rows) >= max_limit:
            return rows[:max_limit]

        fallback_tenant = str(
            os.getenv("ARENA_PUBLIC_DEMO_TENANT", "")
            or os.getenv("ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT", "")
            or ""
        ).strip().lower()
        if not fallback_tenant or fallback_tenant == self.tenant_id:
            return rows[:max_limit]
        if clean_tickers:
            return rows[:max_limit]

        fallback_categories = [category for category in (clean_cats or list(_PUBLIC_RESEARCH_CATEGORIES)) if category in _PUBLIC_RESEARCH_CATEGORIES]
        if not fallback_categories:
            return rows[:max_limit]

        status = research_generation_status(self.settings)
        if bool(status.get("can_generate")):
            return rows[:max_limit]

        fallback_rows = list(
            self.repo.get_research_briefings(
                tickers=None,
                categories=fallback_categories,
                limit=max_limit,
                trading_mode=self.settings.trading_mode,
                tenant_id=fallback_tenant,
            )
        )
        if not fallback_rows:
            return rows[:max_limit]

        seen_ids = {
            str(row.get("briefing_id") or "").strip()
            for row in rows
            if str(row.get("briefing_id") or "").strip()
        }
        merged = list(rows)
        for row in fallback_rows:
            briefing_id = str(row.get("briefing_id") or "").strip()
            if briefing_id and briefing_id in seen_ids:
                continue
            annotated = dict(row)
            annotated["source_tenant_id"] = fallback_tenant
            annotated["public_fallback"] = True
            merged.append(annotated)
            if briefing_id:
                seen_ids.add(briefing_id)
            if len(merged) >= max_limit:
                break
        return merged[:max_limit]

    def save_memory(self, summary: str, score: float = 0.5) -> dict[str, str]:
        """Save a short manual note for future retrieval."""
        text = (summary or "").strip()[:600]
        if not text or not self._memory_store:
            return {"status": "skipped", "reason": "empty summary or no memory store"}
        self._memory_store.record_manual_note(
            agent_id=self.agent_id,
            summary=text,
            score=score,
        )
        return {"status": "saved", "event_type": "manual_note", "summary": text[:80]}

    def _portfolio_weights(self) -> tuple[dict[str, float], float, float]:
        """Returns per-ticker market value weights based on current context."""
        portfolio = self._context.get("portfolio") or {}
        if not isinstance(portfolio, dict):
            return {}, 0.0, 0.0
        positions = portfolio.get("positions") or {}
        if not isinstance(positions, dict):
            return {}, 0.0, 0.0

        px_map: dict[str, float] = {}
        for row in self._context.get("market_features", []) or []:
            ticker = str((row or {}).get("ticker") or "").strip().upper()
            if not ticker:
                continue
            try:
                px = float((row or {}).get("close_price_krw") or 0.0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                px_map[ticker] = px

        values: dict[str, float] = {}
        for ticker, pos in positions.items():
            clean_ticker = str(ticker or "").strip().upper()
            if not clean_ticker or not isinstance(pos, dict):
                continue
            try:
                qty = float(pos.get("quantity") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty <= 0:
                continue
            px = px_map.get(clean_ticker, 0.0)
            if px <= 0:
                try:
                    px = float(pos.get("avg_price_krw") or 0.0)
                except (TypeError, ValueError):
                    px = 0.0
            if px <= 0:
                continue
            values[clean_ticker] = qty * px

        stock_mv = float(sum(values.values()))
        try:
            cash = float(portfolio.get("cash_krw") or 0.0)
        except (TypeError, ValueError):
            cash = 0.0
        total = stock_mv + max(cash, 0.0)
        if total <= 0:
            return {}, stock_mv, cash
        weights = {ticker: (value / total) for ticker, value in values.items() if value > 0}
        return weights, stock_mv, cash

    def _load_aligned_returns(
        self,
        tickers: list[str],
        *,
        lookback_days: int,
        min_history: int = 10,
    ) -> tuple[list[str], Any] | tuple[None, str]:
        """Loads aligned close series and converts them to daily returns."""
        try:
            import numpy as np
        except Exception:
            return None, "numpy unavailable"

        closes = self.repo.get_daily_closes(
            tickers=tickers,
            lookback_days=max(int(lookback_days), min_history) + 1,
            sources=daily_history_sources(self._sources()),
        )
        aligned: list[list[float]] = []
        keep: list[str] = []
        min_len: int | None = None
        for ticker in tickers:
            series = closes.get(ticker, [])
            if len(series) < min_history:
                continue
            if min_len is None or len(series) < min_len:
                min_len = len(series)
            keep.append(ticker)
            aligned.append(series)

        if len(keep) < 2 or min_len is None or min_len < min_history:
            return None, "insufficient history for rebalance plan"

        mat = np.stack([np.array(series[-min_len:], dtype=float) for series in aligned], axis=1)
        rets = (mat[1:] / mat[:-1]) - 1.0
        return keep, rets

    def _cap_target_weights(
        self,
        raw_weights: dict[str, float],
        *,
        target_gross: float,
        max_position_ratio: float,
    ) -> tuple[dict[str, float], list[str]]:
        """Scales optimizer weights to gross target and applies per-position caps."""
        gross = max(0.0, min(float(target_gross), 1.0))
        cap = max(0.0, float(max_position_ratio))
        cleaned = {
            str(ticker).strip().upper(): max(float(weight), 0.0)
            for ticker, weight in raw_weights.items()
            if str(ticker).strip()
        }
        total = float(sum(cleaned.values()))
        if gross <= 0.0 or cap <= 0.0 or total <= 0.0:
            return {}, ["no investable gross exposure after constraints"]

        base = {ticker: weight / total for ticker, weight in cleaned.items()}
        target = {ticker: 0.0 for ticker in base}
        active = set(base.keys())
        remaining = gross
        notes: list[str] = []

        if cap * len(base) + 1e-9 < gross:
            notes.append("position caps forced extra cash above minimum buffer")

        while active and remaining > 1e-9:
            active_sum = float(sum(base[ticker] for ticker in active))
            if active_sum <= 0.0:
                break
            proposed = {ticker: remaining * (base[ticker] / active_sum) for ticker in active}
            capped = [ticker for ticker, weight in proposed.items() if weight > cap + 1e-9]
            if not capped:
                for ticker, weight in proposed.items():
                    target[ticker] = weight
                remaining = 0.0
                break
            for ticker in capped:
                target[ticker] = cap
                remaining = max(0.0, remaining - cap)
                active.remove(ticker)

        return (
            {
                ticker: max(0.0, float(weight))
                for ticker, weight in target.items()
                if weight > 1e-9
            },
            notes,
        )

    def _build_hrp_allocation(
        self,
        current_weights: dict[str, float],
        *,
        lookback_days: int = 252,
        delta_threshold: float = 0.005,
    ) -> dict[str, Any]:
        """Builds an HRP risk allocation view for current holdings."""
        tickers = [ticker for ticker in current_weights.keys() if str(ticker).strip()]
        keep, rets_or_reason = self._load_aligned_returns(
            tickers,
            lookback_days=lookback_days,
            min_history=10,
        )
        if keep is None:
            return {
                "status": "skipped",
                "strategy": "hrp",
                "reason": str(rets_or_reason),
            }

        risk_policy = self._context.get("risk_policy") or {}
        try:
            min_cash_buffer_ratio = float(risk_policy.get("min_cash_buffer_ratio") or 0.0)
        except (TypeError, ValueError):
            min_cash_buffer_ratio = 0.0
        try:
            max_position_ratio = float(risk_policy.get("max_position_ratio") or 1.0)
        except (TypeError, ValueError):
            max_position_ratio = 1.0

        target_gross = max(0.0, min(1.0 - min_cash_buffer_ratio, 1.0))
        result = optimize_hrp(keep, rets_or_reason)
        target_weights, notes = self._cap_target_weights(
            result.weights,
            target_gross=target_gross,
            max_position_ratio=max_position_ratio,
        )
        target_cash_weight = max(0.0, 1.0 - float(sum(target_weights.values())))

        hrp_rows: list[dict[str, Any]] = []
        weight_deltas: list[dict[str, Any]] = []
        small_deltas: list[dict[str, Any]] = []
        for ticker in tickers:
            current_weight = float(current_weights.get(ticker, 0.0))
            hrp_weight = float(target_weights.get(ticker, 0.0))
            delta_weight = hrp_weight - current_weight
            relative_to_current = "similar"
            if delta_weight >= delta_threshold:
                relative_to_current = "higher"
            elif delta_weight <= -delta_threshold:
                relative_to_current = "lower"
            row = {
                "ticker": ticker,
                "current_weight": round(current_weight, 6),
                "hrp_weight": round(hrp_weight, 6),
                "delta_weight": round(delta_weight, 6),
                "relative_to_current": relative_to_current,
            }
            hrp_rows.append(row)
            if abs(delta_weight) < delta_threshold:
                small_deltas.append(
                    {
                        "ticker": ticker,
                        "reason": "delta_below_threshold",
                        "delta_weight": round(delta_weight, 6),
                    }
                )
                continue
            if delta_weight > 0:
                weight_deltas.append(
                    {
                        "ticker": ticker,
                        "relative_to_current": "higher",
                        "delta_weight": round(delta_weight, 6),
                        "current_weight": round(current_weight, 4),
                        "hrp_weight": round(hrp_weight, 4),
                    }
                )
            elif current_weight > 0:
                weight_deltas.append(
                    {
                        "ticker": ticker,
                        "relative_to_current": "lower",
                        "delta_weight": round(delta_weight, 6),
                        "current_weight": round(current_weight, 4),
                        "hrp_weight": round(hrp_weight, 4),
                    }
                )

        allocation: dict[str, Any] = {
            "status": "ready",
            "strategy": "hrp",
            "lookback_days": int(lookback_days),
            "hrp_cash_weight": round(target_cash_weight, 6),
            "hrp_concentration_top3": round(
                sum(
                    item["hrp_weight"]
                    for item in sorted(hrp_rows, key=lambda item: item["hrp_weight"], reverse=True)[:3]
                ),
                6,
            ),
            "hrp_hhi": round(sum(float(weight) * float(weight) for weight in target_weights.values()), 6),
            "constraints": {
                "min_cash_buffer_ratio": round(min_cash_buffer_ratio, 6),
                "max_position_ratio": round(max_position_ratio, 6),
            },
            "hrp_weights": sorted(hrp_rows, key=lambda item: float(item["hrp_weight"]), reverse=True),
            "weight_deltas": weight_deltas,
        }
        if notes:
            allocation["notes"] = [str(note) for note in notes if str(note).strip()]
        if small_deltas:
            allocation["small_deltas"] = small_deltas

        try:
            import numpy as np

            if rets_or_reason.shape[0] >= 5:
                target_vec = np.array([target_weights.get(ticker, 0.0) for ticker in keep], dtype=float)
                projected_rets = rets_or_reason @ target_vec
                cum = np.cumprod(1.0 + projected_rets)
                running_max = np.maximum.accumulate(cum)
                allocation["projected_mdd"] = {
                    "days": int(rets_or_reason.shape[0]),
                    "value": round(float((cum / running_max - 1.0).min()), 6),
                }
        except Exception as exc:
            logger.warning(
                "[yellow]projected MDD calculation failed[/yellow] agent=%s err=%s",
                self.agent_id,
                str(exc),
            )

        return allocation

    def portfolio_diagnosis(self, mdd_days: int = 60, top_n: int = 8, benchmark_ticker: str = "") -> dict[str, Any]:
        """Diagnoses current holdings and returns an HRP allocation view."""
        weights, _, _ = self._portfolio_weights()
        if not weights:
            return {"error": "no active positions"}

        vol_map: dict[str, float] = {}
        mf_map: dict[str, dict[str, Any]] = {}
        for row in self._context.get("market_features", []) or []:
            ticker = str((row or {}).get("ticker") or "").strip().upper()
            if not ticker:
                continue
            mf_map[ticker] = row or {}
            try:
                vol = float((row or {}).get("volatility_20d") or 0.0)
            except (TypeError, ValueError):
                vol = 0.0
            if vol > 0:
                vol_map[ticker] = vol

        rc_raw = {ticker: (weight * vol_map.get(ticker, 0.02)) for ticker, weight in weights.items()}
        rc_sum = float(sum(rc_raw.values()))
        risk_contrib = {
            ticker: (value / rc_sum if rc_sum > 0 else 0.0)
            for ticker, value in sorted(rc_raw.items(), key=lambda kv: kv[1], reverse=True)
        }
        top = max(1, min(int(top_n), 20))
        w_sorted = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        hhi = float(sum(weight * weight for _, weight in weights.items()))

        mom = 0.0
        short = 0.0
        vol_weighted = 0.0
        for ticker, weight in weights.items():
            row = mf_map.get(ticker, {})
            try:
                ret_20 = float(row.get("ret_20d") or 0.0)
            except (TypeError, ValueError):
                ret_20 = 0.0
            try:
                ret_5 = float(row.get("ret_5d") or 0.0)
            except (TypeError, ValueError):
                ret_5 = 0.0
            try:
                v20 = float(row.get("volatility_20d") or 0.0)
            except (TypeError, ValueError):
                v20 = 0.0
            mom += weight * ret_20
            short += weight * ret_5
            vol_weighted += weight * v20

        out: dict[str, Any] = {
            "risk_contribution": [
                {"ticker": ticker, "rc": round(rc, 6)} for ticker, rc in list(risk_contrib.items())[:top]
            ],
            "concentration_top3": round(sum(weight for _, weight in w_sorted[:3]), 6),
            "hhi": round(hhi, 6),
            "momentum_20d_weighted": round(mom, 6),
            "momentum_5d_weighted": round(short, 6),
            "volatility_20d_weighted": round(vol_weighted, 6),
        }

        try:
            import numpy as np

            tickers = list(weights.keys())
            closes = self.repo.get_daily_closes(tickers=tickers, lookback_days=int(mdd_days) + 1, sources=None)
            min_len: int | None = None
            aligned: list[list[float]] = []
            keep: list[str] = []
            for ticker in tickers:
                series = closes.get(ticker, [])
                if len(series) < 5:
                    continue
                if min_len is None or len(series) < min_len:
                    min_len = len(series)
                keep.append(ticker)
                aligned.append(series)
            if keep and min_len and min_len >= 5:
                mat = np.stack([np.array(series[-min_len:], dtype=float) for series in aligned], axis=1)
                rets = (mat[1:] / mat[:-1]) - 1.0
                mdd_n = max(5, min(int(mdd_days), rets.shape[0]))
                w_vec = np.array([weights.get(ticker, 0.0) for ticker in keep], dtype=float)
                w_sum = float(w_vec.sum())
                if w_sum > 0:
                    w_vec = w_vec / w_sum
                port_rets = rets[-mdd_n:] @ w_vec
                cum = np.cumprod(1.0 + port_rets)
                running_max = np.maximum.accumulate(cum)
                out["mdd"] = {"days": mdd_n, "value": round(float((cum / running_max - 1.0).min()), 6)}
        except Exception as exc:
            logger.warning(
                "[yellow]portfolio diagnosis MDD calculation failed[/yellow] agent=%s err=%s",
                self.agent_id,
                str(exc),
            )

        perf = self._context.get("performance") or {}
        if isinstance(perf, dict) and perf:
            bench = str(benchmark_ticker or "").strip().upper()
            if not bench:
                if self.settings.kis_target_market == "nasdaq":
                    bench = "QQQ"
                elif self.settings.kis_target_market == "kospi":
                    bench = "069500"
            if bench:
                try:
                    today = utc_now().date()
                    benchmarks: dict[str, dict[str, object]] = {}
                    for scope, (period_start, agent_ret, agent_metric) in self._benchmark_period_bases(perf).items():
                        entry = self._benchmark_entry(
                            scope=scope,
                            bench=bench,
                            period_start=period_start,
                            agent_ret=agent_ret,
                            agent_metric=agent_metric,
                            today=today,
                        )
                        if entry is not None:
                            benchmarks[scope] = entry
                    if benchmarks:
                        out["benchmarks"] = benchmarks
                        out["benchmark"] = (
                            benchmarks.get("current_sleeve")
                            or benchmarks.get("cumulative")
                            or next(iter(benchmarks.values()))
                        )
                except Exception as exc:
                    logger.warning(
                        "[yellow]portfolio benchmark calculation failed[/yellow] agent=%s benchmark=%s err=%s",
                        self.agent_id,
                        bench,
                        str(exc),
                    )

        return out

    # ── Trade performance ────────────────────────────────────────────
    def _unrealized_positions(self, target: str = "") -> list[dict[str, Any]]:
        """Extract current unrealised P&L from context (portfolio or performance)."""
        portfolio = self._context.get("portfolio") or {}
        perf = self._context.get("performance") or {}
        # portfolio["positions"] is dict {ticker: {qty, avg_price_krw, ...}}
        # performance["positions"] is list [{ticker, quantity, avg_price_krw, market_price_krw, ...}]
        unrealized: list[dict[str, Any]] = []

        pos_list: list[dict[str, Any]] = []
        perf_positions = perf.get("positions") if isinstance(perf, dict) else None
        port_positions = portfolio.get("positions") if isinstance(portfolio, dict) else None
        if isinstance(perf_positions, list):
            pos_list = perf_positions
        elif isinstance(port_positions, dict):
            pos_list = [{"ticker": tk, **v} for tk, v in port_positions.items() if isinstance(v, dict)]
        elif isinstance(port_positions, list):
            pos_list = port_positions

        for pos in pos_list:
            tk = str(pos.get("ticker") or "").strip().upper()
            if target and tk != target:
                continue
            qty = float(pos.get("quantity") or pos.get("qty") or 0)
            cost = float(pos.get("avg_price_krw") or 0)
            mkt = float(pos.get("market_price_krw") or 0)
            if qty > 0 and cost > 0 and mkt > 0:
                unrealized.append({
                    "ticker": tk,
                    "qty": int(qty),
                    "cost_krw": round(cost, 0),
                    "current_krw": round(mkt, 0),
                    "return_pct": round((mkt / cost - 1) * 100, 2),
                })
        return unrealized

    def trade_performance(self, lookback_days: int = 90, ticker: str = "") -> dict[str, Any]:
        """Analyses closed round-trip trades and current unrealised P&L."""
        from datetime import timedelta

        target = str(ticker or "").strip().upper()

        since = utc_now() - timedelta(days=max(7, min(int(lookback_days), 365)))
        rows = self.repo.filled_execution_reports_since(
            since=since,
            tenant_id=self.tenant_id,
            include_simulated=True,
        )

        # filter to this agent
        rows = [r for r in rows if str(r.get("agent_id") or "").strip().lower() == self.agent_id]
        if target:
            rows = [r for r in rows if str(r.get("ticker") or "").strip().upper() == target]

        unrealized = self._unrealized_positions(target)

        if not rows:
            return {"period": f"{lookback_days}d", "total_trades": 0, "round_trips": {}, "unrealized": unrealized}

        # ── FIFO round-trip matching ──
        lots: dict[str, list[dict]] = {}
        closed: list[dict[str, Any]] = []

        for r in rows:
            tk = str(r.get("ticker") or "").strip().upper()
            side = str(r.get("side") or "").strip().upper()
            qty = float(r.get("filled_qty") or 0)
            price = float(r.get("avg_price_krw") or 0)
            ts = r.get("created_at")
            if not tk or qty <= 0 or price <= 0:
                continue

            if side == "BUY":
                lots.setdefault(tk, []).append({"qty": qty, "cost": price, "ts": ts, "notional": qty * price})
            elif side == "SELL" and lots.get(tk):
                remaining = qty
                cost_sum = 0.0
                bought_qty = 0.0
                earliest_ts = lots[tk][0]["ts"] if lots[tk] else ts
                while remaining > 0 and lots[tk]:
                    lot = lots[tk][0]
                    take = min(remaining, lot["qty"])
                    cost_sum += take * lot["cost"]
                    bought_qty += take
                    remaining -= take
                    lot["qty"] -= take
                    if lot["qty"] <= 0:
                        lots[tk].pop(0)
                if bought_qty > 0:
                    avg_cost = cost_sum / bought_qty
                    ret_pct = round((price / avg_cost - 1) * 100, 2)
                    held_days = (ts - earliest_ts).days if ts and earliest_ts else 0
                    closed.append({
                        "ticker": tk,
                        "return_pct": ret_pct,
                        "held_days": held_days,
                        "notional": round(cost_sum, 0),
                        "closed_at": ts,
                    })

        # ── Summary ──
        total_trades = len(rows)
        n_closed = len(closed)
        winners = [t for t in closed if t["return_pct"] > 0]
        losers = [t for t in closed if t["return_pct"] <= 0]

        round_trips: dict[str, Any] = {"closed": n_closed}
        if n_closed:
            round_trips["win_rate"] = round(len(winners) / n_closed, 2)
            round_trips["avg_return_pct"] = round(sum(t["return_pct"] for t in closed) / n_closed, 2)
            round_trips["avg_holding_days"] = round(sum(t["held_days"] for t in closed) / n_closed, 1)
            best = max(closed, key=lambda t: t["return_pct"])
            worst = min(closed, key=lambda t: t["return_pct"])
            round_trips["best"] = {"ticker": best["ticker"], "return_pct": best["return_pct"], "held_days": best["held_days"]}
            round_trips["worst"] = {"ticker": worst["ticker"], "return_pct": worst["return_pct"], "held_days": worst["held_days"]}

        # ── Behavioral ──
        behavioral: dict[str, Any] = {}
        if winners:
            behavioral["avg_winner_held_days"] = round(sum(t["held_days"] for t in winners) / len(winners), 1)
        if losers:
            behavioral["avg_loser_held_days"] = round(sum(t["held_days"] for t in losers) / len(losers), 1)
        if winners and losers and behavioral.get("avg_winner_held_days") and behavioral.get("avg_loser_held_days"):
            behavioral["disposition_flag"] = behavioral["avg_loser_held_days"] > behavioral["avg_winner_held_days"] * 1.5

        if len(closed) >= 4:
            notionals = sorted(t["notional"] for t in closed)
            threshold = notionals[len(notionals) * 7 // 10]
            large = [t for t in closed if t["notional"] >= threshold]
            small = [t for t in closed if t["notional"] < threshold]
            if large:
                behavioral["large_position_win_rate"] = round(len([t for t in large if t["return_pct"] > 0]) / len(large), 2)
            if small:
                behavioral["small_position_win_rate"] = round(len([t for t in small if t["return_pct"] > 0]) / len(small), 2)

        # ── Recent streak ──
        recent = sorted(closed, key=lambda t: t["closed_at"] or datetime.min, reverse=True)[:5]
        now = utc_now()
        recent_streak = {
            "last_5": [
                {"ticker": t["ticker"], "return_pct": t["return_pct"], "days_ago": (now - t["closed_at"]).days if t["closed_at"] else 0}
                for t in recent
            ],
        }
        if recent:
            wins = sum(1 for t in recent if t["return_pct"] > 0)
            recent_streak["momentum"] = "winning" if wins >= 3 else "losing" if wins <= 1 else "mixed"

        return {
            "period": f"{lookback_days}d",
            "total_trades": total_trades,
            "round_trips": round_trips,
            "behavioral": behavioral,
            "recent_streak": recent_streak,
            "unrealized": unrealized,
        }
