from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

# Module-level dedup for concurrent agent threads (same process).
_forecast_build_lock = threading.Lock()
_forecast_built_dates: set[str] = set()

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.market_sources import live_market_sources_for_markets
from arena.tools._market_scope import MarketScope, MarketScopeError
from arena.market_feature_normalization import (
    daily_history_sources,
    normalize_market_feature_rows,
    normalize_market_feature_rows_from_closes,
)
from arena.open_trading.client import OpenTradingClient
from arena.open_trading.exchange_codes import (
    normalize_us_quote_exchange,
    target_market_default_us_quote_exchange,
    us_quote_exchange_candidates,
)
from arena.runtime_universe import resolve_runtime_universe

from .allocation import (
    AllocationResult,
    apply_weight_constraints,
    blend_forecast_mu,
    optimize_forecast_sharpe,
    optimize_hrp,
    optimize_max_sharpe,
    recompute_stats,
)
from .screening import build_discovery_rows, momentum_scores
from .sector_map import SECTOR_BY_TICKER

logger = logging.getLogger(__name__)

_RECOMMEND_OPPORTUNITY_MAX_POOL = 500


def _opportunity_selection_limits(
    *,
    top_n: int,
    max_candidates: int | None,
) -> tuple[int, int, str]:
    """Returns global and per-profile rank limits for opportunity selection."""
    try:
        top = int(top_n)
    except (TypeError, ValueError):
        top = 8
    top = max(1, min(top, 20))

    if max_candidates is not None:
        try:
            requested = int(max_candidates)
        except (TypeError, ValueError):
            requested = top
        return max(top, min(max(1, requested), _RECOMMEND_OPPORTUNITY_MAX_POOL)), top, "explicit"

    return top, top, "ranked_union"


def _to_float(value: Any, default: float | None = 0.0) -> float | None:
    """Safely parses mixed API payload values into float."""
    try:
        if value is None:
            return default
        text = str(value).strip().replace(",", "")
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _build_decision_summary(
    *,
    strategy: str,
    orders: list[dict] | None,
    weights: dict[str, float],
) -> dict[str, Any]:
    """Rule-based canonical summary from strategy + orders.

    No LLM-style phrasing: fixed vocabulary (Increase/Reduce/Tilt toward/Keep).
    Outputs headline_code for deterministic test assertions.
    """
    if strategy == "single_name" and weights:
        ticker = next(iter(weights))
        return {
            "headline_code": "single_name",
            "headline": f"Only one usable ticker; allocation set to {ticker}=1.0.",
            "turnover": 0.0,
            "confidence": "low",
        }

    if orders is None:
        return {
            "headline_code": "no_current_portfolio",
            "headline": "No current portfolio context; weights are suggestions only.",
            "turnover": 0.0,
            "confidence": "low",
        }
    if not orders:
        # Portfolio already aligned with target — nothing to trade.
        return {
            "headline_code": "hold",
            "headline": "Keep allocation unchanged; portfolio already aligned with target.",
            "turnover": 0.0,
            "confidence": "low",
        }

    buys = [o for o in orders if o.get("side") == "BUY"]
    sells = [o for o in orders if o.get("side") == "SELL"]
    buy_total = sum(float(o.get("target_weight", 0.0)) - float(o.get("current_weight", 0.0)) for o in buys)
    sell_total = sum(float(o.get("current_weight", 0.0)) - float(o.get("target_weight", 0.0)) for o in sells)
    turnover = round((buy_total + sell_total) / 2.0, 4)

    top_buy = max(
        buys,
        key=lambda o: float(o.get("target_weight", 0.0)) - float(o.get("current_weight", 0.0)),
        default=None,
    )
    top_sell = max(
        sells,
        key=lambda o: float(o.get("current_weight", 0.0)) - float(o.get("target_weight", 0.0)),
        default=None,
    )

    if turnover < 0.03:
        code = "hold"
        headline = "Keep allocation mostly unchanged; expected benefit does not justify turnover."
    elif top_buy and top_sell:
        code = "rotate"
        headline = f"Increase {top_buy['ticker']} and reduce {top_sell['ticker']}."
    elif top_buy:
        code = "accumulate"
        headline = f"Tilt toward {top_buy['ticker']}."
    elif top_sell:
        code = "trim"
        headline = f"Reduce {top_sell['ticker']}."
    else:
        code = "hold"
        headline = "Keep allocation mostly unchanged."

    if turnover >= 0.10:
        confidence = "high"
    elif turnover >= 0.03:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "headline_code": code,
        "headline": headline,
        "turnover": turnover,
        "confidence": confidence,
    }


def _has_any_value(payload: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return True
            continue
        return True
    return False


@dataclass(slots=True)
class QuantTools:
    """BigQuery-backed quant analysis tools."""

    repo: BigQueryRepository
    settings: Settings
    ot_client: OpenTradingClient | None = None
    _cached_universe: list[str] | None = None
    _context: dict | None = None
    _discovery_cache: dict[str, Any] | None = None

    def set_context(self, context: dict) -> None:
        """Stores current cycle context for portfolio-aware rebalance orders."""
        old_market = self._effective_market()
        self._context = context
        # Clear cached universe when agent's effective market changes
        if self._effective_market() != old_market:
            self._cached_universe = None
        self._discovery_cache = None

    def _scope(self) -> MarketScope:
        """Builds a MarketScope from the current cycle context."""
        return MarketScope.from_context(
            self._context,
            fallback=getattr(self.settings, "kis_target_market", None),
        )

    def _effective_market(self) -> str:
        """Returns the calling agent's raw target_market string (comma-list ok)."""
        if self._context:
            tm = str(self._context.get("target_market") or "").strip().lower()
            if tm:
                return tm
        global_market = str(self.settings.kis_target_market or "").strip().lower()
        if not global_market:
            raise MarketScopeError(
                "target_market is not configured for this agent. "
                "Set target_market in agent config."
            )
        return global_market

    def _effective_markets(self) -> set[str]:
        return self._scope().as_set()

    def _has_us_market(self) -> bool:
        return self._scope().has_us

    def _has_kospi_market(self) -> bool:
        return self._scope().has_kospi

    def _us_fundamentals_exchange_candidates(self, requested_exchange: object) -> list[str]:
        requested = str(requested_exchange or "").strip().upper()
        market_default = target_market_default_us_quote_exchange(self._effective_market())
        settings_default = normalize_us_quote_exchange(getattr(self.settings, "kis_overseas_quote_excd", ""))
        return us_quote_exchange_candidates(
            requested,
            market_default,
            settings_default,
            getattr(self.settings, "us_quote_exchanges", []),
        )

    def _fetch_us_fundamental_snapshot(
        self,
        *,
        client: OpenTradingClient,
        ticker: str,
        requested_exchange: object,
    ) -> tuple[str, dict[str, Any]]:
        candidates = self._us_fundamentals_exchange_candidates(requested_exchange)
        fallback_exchange = candidates[0] if candidates else "NAS"
        fallback_raw: dict[str, Any] | None = None
        fallback_raw_exchange = fallback_exchange
        last_exc: Exception | None = None

        for exchange in candidates:
            try:
                raw = client.get_overseas_price_detail(ticker=ticker, excd=exchange)
            except Exception as exc:
                last_exc = exc
                continue
            if _has_any_value(raw, "perx", "pbrx", "epsx", "bpsx"):
                return exchange, raw
            if fallback_raw is None and _has_any_value(raw, "last", "tomv", "curr", "e_ordyn"):
                fallback_raw = raw
                fallback_raw_exchange = exchange

        if fallback_raw is not None:
            return fallback_raw_exchange, fallback_raw
        if last_exc is not None:
            raise last_exc
        return fallback_exchange, {}

    def _portfolio_weights(self) -> tuple[dict[str, float], float, float]:
        """Returns per-ticker market value weights based on current context."""
        if not self._context:
            return {}, 0.0, 0.0
        portfolio = self._context.get("portfolio") or {}
        if not isinstance(portfolio, dict):
            return {}, 0.0, 0.0
        positions = portfolio.get("positions") or {}
        if not isinstance(positions, dict):
            return {}, 0.0, 0.0

        px_map: dict[str, float] = {}
        for row in self._context.get("market_features", []) or []:
            t = str((row or {}).get("ticker") or "").strip().upper()
            if not t:
                continue
            try:
                px = float((row or {}).get("close_price_krw") or 0.0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                px_map[t] = px

        values: dict[str, float] = {}
        for ticker, pos in positions.items():
            t = str(ticker or "").strip().upper()
            if not t or not isinstance(pos, dict):
                continue
            try:
                qty = float(pos.get("quantity") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty <= 0:
                continue
            px = px_map.get(t, 0.0)
            if px <= 0:
                try:
                    px = float(pos.get("avg_price_krw") or 0.0)
                except (TypeError, ValueError):
                    px = 0.0
            if px <= 0:
                continue
            values[t] = qty * px

        stock_mv = float(sum(values.values()))
        try:
            cash = float(portfolio.get("cash_krw") or 0.0)
        except (TypeError, ValueError):
            cash = 0.0
        total = stock_mv + max(cash, 0.0)
        if total <= 0:
            return {}, stock_mv, cash
        weights = {t: (v / total) for t, v in values.items() if v > 0}
        return weights, stock_mv, cash

    def _candidate_tickers_from_context(self) -> list[str]:
        """Returns unresolved discovery candidates injected into the live cycle context."""
        if not self._context:
            return []
        raw = self._context.get("_candidate_tickers") or []
        if not isinstance(raw, list):
            return []
        return [str(t).strip().upper() for t in raw if str(t).strip()]

    def _discovered_candidate_tickers_from_context(self) -> list[str]:
        """Returns the broader discovered candidate basket from the live cycle context."""
        if not self._context:
            return []
        raw = self._context.get("_discovered_candidate_tickers") or []
        if not isinstance(raw, list):
            return []
        return [str(t).strip().upper() for t in raw if str(t).strip()]

    def _working_set_tickers_from_context(self) -> list[str]:
        """Returns self-discovered opportunity tickers visible in the active cycle context."""
        if not bool(getattr(self.settings, "autonomy_working_set_enabled", False)):
            return []
        if not self._context:
            return []
        raw = self._context.get("opportunity_working_set") or []
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            if ticker and ticker not in out:
                out.append(ticker)
        return out

    def _held_tickers_from_context(self) -> list[str]:
        """Returns held tickers from the active portfolio context."""
        if not self._context:
            return []
        portfolio = self._context.get("portfolio") or {}
        if not isinstance(portfolio, dict):
            return []
        positions = portfolio.get("positions") or {}
        if not isinstance(positions, dict):
            return []
        return [str(t).strip().upper() for t in positions if str(t).strip()]

    def _analysis_default_tickers(self, *, limit: int = 10) -> list[str]:
        """Resolves the default analysis basket without forcing exogenous candidates."""
        if not bool(getattr(self.settings, "autonomy_tool_default_candidates_enabled", False)):
            return []
        candidates = (
            self._discovered_candidate_tickers_from_context()
            or self._working_set_tickers_from_context()
            or self._candidate_tickers_from_context()
        )
        if candidates:
            return self._normalize_tickers(candidates, restrict_to_universe=False)[: max(1, min(int(limit), 50))]

        held = self._held_tickers_from_context()
        if held:
            return self._normalize_tickers(held, restrict_to_universe=False)[: max(1, min(int(limit), 50))]

        return self._target_universe()[: max(1, min(int(limit), 50))]

    def _forecast_default_tickers(self) -> list[str]:
        """Builds the default forecast basket from the discovered basket plus holdings."""
        candidates = (
            self._discovered_candidate_tickers_from_context()
            or self._working_set_tickers_from_context()
            or self._candidate_tickers_from_context()
        )
        held = self._held_tickers_from_context()
        combined = self._normalize_tickers(candidates + held)
        return combined[:50]

    def _log_tool_result(self, tool_name: str, rows: list | dict | None, *, key_fields: list[str] | None = None) -> None:
        """Emit a structured TOOL_RESULT log line for later extraction.

        Data is passed via ``extra`` so the JSON formatter emits it as a
        top-level field (``tool_data``) instead of embedding raw JSON inside
        the ``message`` string — which would be stripped by the Rich-tag
        sanitiser regex in ``_JsonFormatter``.
        """
        if rows is None:
            rows = []
        items = rows if isinstance(rows, list) else [rows]
        summary: list[dict] = []
        for r in items:
            if not isinstance(r, dict):
                continue
            if key_fields:
                summary.append({k: r[k] for k in key_fields if k in r})
            else:
                summary.append(r)
        logger.info(
            "TOOL_RESULT %s count=%d",
            tool_name,
            len(summary),
            extra={"tool_name": tool_name, "tool_data": summary},
        )

    def _compact_forecast_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compacts per-model forecast rows into one per ticker.

        Preserves the top-level summary fields agents care about while moving
        model-specific detail into compact ``base_models`` / ``stacked_models``
        lists to avoid repeating ticker/run_date/consensus on every row.
        """
        if not rows:
            return []

        def _sort_key(item: dict[str, Any]) -> tuple[str, int, float]:
            try:
                exp_ret = float(item.get("exp_return_period") or 0.0)
            except (TypeError, ValueError):
                exp_ret = 0.0
            return (
                str(item.get("ticker") or ""),
                0 if item.get("is_stacked") else 1,
                -exp_ret,
            )

        grouped: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
        for row in sorted(rows, key=_sort_key):
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            run_date = str(row.get("run_date") or "").strip()
            horizon = row.get("forecast_horizon")
            try:
                horizon = int(horizon) if horizon is not None else None
            except (TypeError, ValueError):
                horizon = None
            grouped.setdefault((ticker, run_date, horizon), []).append(dict(row))

        compacted: list[dict[str, Any]] = []
        for (_ticker, _run_date, _horizon), items in grouped.items():
            stacked_rows = [r for r in items if bool(r.get("is_stacked"))]
            base_rows = [r for r in items if not bool(r.get("is_stacked"))]

            preferred_primary: dict[str, Any] | None = None
            for model_name in ("ensemble_wmae", "ensemble_avg"):
                preferred_primary = next(
                    (r for r in stacked_rows if str(r.get("forecast_model") or "").strip().lower() == model_name),
                    None,
                )
                if preferred_primary is not None:
                    break
            if preferred_primary is None:
                preferred_primary = stacked_rows[0] if stacked_rows else (items[0] if items else None)
            if preferred_primary is None:
                continue

            entry: dict[str, Any] = {
                "run_date": preferred_primary.get("run_date"),
                "ticker": preferred_primary.get("ticker"),
                "exp_return_period": preferred_primary.get("exp_return_period"),
            }
            for key in (
                "forecast_horizon",
                "forecast_model",
                "is_stacked",
                "forecast_score",
                "prob_up",
                "model_votes_up",
                "model_votes_total",
                "consensus",
            ):
                if preferred_primary.get(key) is not None:
                    entry[key] = preferred_primary.get(key)

            def _mini(rows_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
                out: list[dict[str, Any]] = []
                for r in rows_in:
                    mini = {
                        "forecast_model": r.get("forecast_model"),
                        "exp_return_period": r.get("exp_return_period"),
                    }
                    if r.get("forecast_score") is not None:
                        mini["forecast_score"] = r.get("forecast_score")
                    out.append(mini)
                return out

            if stacked_rows:
                entry["stacked_models"] = _mini(stacked_rows)
            if base_rows:
                entry["base_models"] = _mini(base_rows)
            if base_rows:
                best_base = max(
                    base_rows,
                    key=lambda r: float(r.get("exp_return_period") or 0.0),
                )
                entry["best_base_model"] = str(best_base.get("forecast_model") or "").strip() or None
                entry["best_base_return"] = best_base.get("exp_return_period")
            compacted.append(entry)

        compacted.sort(
            key=lambda r: float(r.get("exp_return_period") or 0.0),
            reverse=True,
        )
        return compacted

    def _sources(self) -> list[str] | None:
        if self.settings.trading_mode != "live":
            return None
        result = live_market_sources_for_markets(self._effective_markets())
        return result or None

    def _target_universe(self) -> list[str]:
        """Returns normalized tickers allowed for this arena run."""
        if self._cached_universe is not None:
            return list(self._cached_universe)
        result = resolve_runtime_universe(
            self.settings,
            self.repo,
            markets=self._effective_markets(),
        )
        self._cached_universe = result
        return list(result)

    def _normalize_tickers(self, tickers: list[str], *, restrict_to_universe: bool = True) -> list[str]:
        tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not restrict_to_universe:
            return tokens

        allowed = set(self._target_universe())
        if not allowed:
            return []
        return [t for t in tokens if t in allowed]

    def _partition_tickers_by_scope(
        self, tickers: list[str] | None
    ) -> tuple[list[str], list[dict[str, str]]]:
        """Splits input tickers into (in_scope, excluded_with_reasons) via MarketScope."""
        try:
            scope = self._scope()
        except MarketScopeError:
            tokens = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
            return list(dict.fromkeys(tokens)), []
        return scope.filter_tickers(tickers or [])

    def _forecast_mode(self, override: str | None = None) -> str:
        default_mode = str(self.settings.forecast_mode or "all").strip().lower() or "all"
        token = str(override or default_mode).strip().lower()
        if not token:
            return default_mode
        if token == "balanced":
            return "all"

        valid_aliases = {
            "all", "both", "full", "raw", "base+stacked", "stacked+base",
            "stacked", "stack", "meta", "lgbm_stack", "ridge_stack", "ensemble_stack",
            "lgbm", "lightgbm", "stacked_lgbm", "meta_lgbm", "stacked_lightgbm",
            "ridge", "stacked_ridge", "meta_ridge",
            "avg", "average", "simple_average", "equal_weight", "ensemble_avg",
            "base", "base_model", "base_models",
        }
        if token not in valid_aliases:
            logger.warning(
                "[yellow]Unknown forecast_mode override ignored[/yellow] override=%s default=%s",
                token,
                default_mode,
            )
            return default_mode
        return token

    def _forecast_auto_build_enabled(self) -> bool:
        token = str(os.getenv("ARENA_FORECAST_AUTO_BUILD", "false")).strip().lower()
        return token in {"1", "true", "yes", "y", "on"}

    def _auto_build_forecasts_if_needed(self) -> bool:
        from arena.forecasting import build_and_store_stacked_forecasts

        today = datetime.now(timezone.utc).date().isoformat()

        with _forecast_build_lock:
            if today in _forecast_built_dates:
                logger.info("[cyan]Forecast auto-build skipped (already built today)[/cyan]")
                return True

            attempts = [
                {"lookback_days": 360, "horizon": 20, "min_series_length": 160, "max_steps": 200},
                {"lookback_days": 240, "horizon": 15, "min_series_length": 90, "max_steps": 120},
                {"lookback_days": 180, "horizon": 10, "min_series_length": 60, "max_steps": 80},
            ]
            last_note = ""
            for idx, cfg in enumerate(attempts, start=1):
                result = build_and_store_stacked_forecasts(self.repo, self.settings, **cfg)
                last_note = str(result.note or "")
                logger.info(
                    "[cyan]Forecast auto-build[/cyan] attempt=%d/%d rows=%d tickers=%d used_neuralforecast=%s note=%s cfg=%s",
                    idx,
                    len(attempts),
                    int(result.rows_written),
                    int(result.tickers_used),
                    str(bool(result.used_neuralforecast)).lower(),
                    last_note,
                    cfg,
                )
                if int(result.rows_written) > 0:
                    _forecast_built_dates.add(today)
                    return True

            logger.warning(
                "[yellow]Forecast auto-build skipped[/yellow] reason=%s",
                last_note or "no rows produced",
            )
            return False

    def _ot(self) -> OpenTradingClient:
        """Lazily creates open-trading API client for valuation/fundamental tools."""
        if self.ot_client is None:
            self.ot_client = OpenTradingClient(self.settings)
        return self.ot_client

    def _load_aligned_returns(
        self, tickers: list[str], *, lookback_days: int
    ) -> tuple[list[str], np.ndarray | None, dict]:
        """Loads aligned close series and returns (keep, rets, quality).

        quality fields:
            status: "ok" | "partial" | "unusable"
            requested_tickers: int
            usable_tickers: int
            excluded: [{ticker, reason}] where reason in {"no_data", "insufficient_history"}
            min_history_days: int (0 when unusable)
        """
        normalized = self._normalize_tickers(tickers)
        requested = len(normalized)

        excluded: list[dict[str, str]] = []
        if requested == 0:
            quality = {
                "status": "unusable",
                "requested_tickers": 0,
                "usable_tickers": 0,
                "excluded": excluded,
                "min_history_days": 0,
            }
            return [], None, quality

        closes = self.repo.get_daily_closes(
            tickers=normalized,
            lookback_days=int(lookback_days) + 1,
            sources=daily_history_sources(self._sources()),
        )

        aligned: list[list[float]] = []
        keep: list[str] = []
        min_len: int | None = None
        for t in normalized:
            series = closes.get(t, [])
            if not series:
                excluded.append({"ticker": t, "reason": "no_data"})
                continue
            if len(series) < 10:
                excluded.append({"ticker": t, "reason": "insufficient_history"})
                continue
            if min_len is None or len(series) < min_len:
                min_len = len(series)
            keep.append(t)
            aligned.append(series)

        usable = len(keep)
        if usable == 0 or min_len is None or min_len < 10:
            quality = {
                "status": "unusable",
                "requested_tickers": requested,
                "usable_tickers": 0,
                "excluded": excluded,
                "min_history_days": 0,
            }
            return [], None, quality

        trimmed = np.stack([np.array(s[-min_len:], dtype=float) for s in aligned], axis=1)
        rets = (trimmed[1:, :] / trimmed[:-1, :]) - 1.0
        quality = {
            "status": "ok" if usable == requested else "partial",
            "requested_tickers": requested,
            "usable_tickers": usable,
            "excluded": excluded,
            "min_history_days": int(min_len),
        }
        return keep, rets, quality

    def _format_allocation(self, tickers: list[str], result, rets=None, mdd_days: int = 60) -> dict:
        weights = {k: round(float(v), 4) for k, v in result.weights.items()}
        out: dict = {
            "tickers": tickers,
            "strategy": getattr(result, "strategy", "unknown"),
            "weights": weights,
            "expected_return_daily": round(float(result.expected_return), 6),
            "volatility_daily": round(float(result.volatility), 6),
            "sharpe_daily": round(float(result.sharpe), 4),
        }

        cur_weights, _, _ = self._portfolio_weights()
        if cur_weights or self._context:
            orders: list[dict] = []
            for t in tickers:
                cur_w = cur_weights.get(t, 0.0)
                tgt_w = weights.get(t, 0.0)
                delta_w = tgt_w - cur_w
                if abs(delta_w) < 0.005:
                    continue
                if delta_w > 0:
                    orders.append({
                        "ticker": t, "side": "BUY",
                        "current_weight": round(cur_w, 4), "target_weight": round(tgt_w, 4),
                    })
                else:
                    sell_ratio = round(abs(delta_w) / cur_w, 4) if cur_w > 0 else 0.0
                    orders.append({
                        "ticker": t, "side": "SELL", "sell_ratio": sell_ratio,
                        "current_weight": round(cur_w, 4), "target_weight": round(tgt_w, 4),
                    })
            # Currently held but not in target → SELL all
            for t, cur_w in cur_weights.items():
                if t not in weights and cur_w > 0.005:
                    orders.append({
                        "ticker": t, "side": "SELL", "sell_ratio": 1.0,
                        "current_weight": round(cur_w, 4), "target_weight": 0.0,
                    })
            out["rebalance_orders"] = orders
        else:
            out["rebalance_suggestions"] = [
                {"ticker": t, "target_weight": weights.get(t, 0.0)} for t in tickers
            ]

        if rets is not None:
            try:
                mdd_n = max(5, min(int(mdd_days), rets.shape[0]))
                w_vec = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
                w_sum = float(w_vec.sum())
                if w_sum > 0:
                    w_vec = w_vec / w_sum
                port_rets = rets[-mdd_n:] @ w_vec
                cum = np.cumprod(1.0 + port_rets)
                running_max = np.maximum.accumulate(cum)
                out["backtest_mdd"] = {"days": mdd_n, "value": round(float((cum / running_max - 1.0).min()), 6)}
            except Exception:
                pass

        out["decision_summary"] = _build_decision_summary(
            strategy=out["strategy"],
            orders=out.get("rebalance_orders"),
            weights=weights,
        )
        return out

    def _discovery_inputs(
        self,
        *,
        windows: list[int],
        vol_adjust: bool,
        include_value: bool,
        min_ret_20d: float | None = None,
        max_volatility: float | None = None,
    ) -> dict[str, Any]:
        universe = self._target_universe()
        sources = self._sources()
        filters_key = (
            tuple(sorted(universe)),
            tuple(sorted(self._effective_markets())),
            tuple(int(w) for w in windows),
            bool(vol_adjust),
            bool(include_value),
            float(min_ret_20d) if min_ret_20d is not None else None,
            float(max_volatility) if max_volatility is not None else None,
            tuple(sources or []),
        )
        if isinstance(self._discovery_cache, dict) and self._discovery_cache.get("key") == filters_key:
            return dict(self._discovery_cache.get("value") or {})

        if not universe:
            payload = {"latest_rows": [], "momentum_rows": [], "fundamentals_rows": []}
            self._discovery_cache = {"key": filters_key, "value": payload}
            return payload

        latest_rows = self.repo.latest_market_features(
            tickers=universe,
            limit=max(50, len(universe)),
            sources=sources,
        )
        latest_tickers = [
            str(row.get("ticker") or "").strip().upper()
            for row in latest_rows
            if isinstance(row, dict) and str(row.get("ticker") or "").strip()
        ]
        closes = self.repo.get_daily_closes(
            tickers=latest_tickers,
            lookback_days=max(windows) + 2,
            sources=daily_history_sources(sources),
        ) if latest_tickers else {}
        latest_rows = normalize_market_feature_rows_from_closes(latest_rows, closes)
        if min_ret_20d is not None:
            latest_rows = [row for row in latest_rows if float(row.get("ret_20d") or 0.0) >= float(min_ret_20d)]
        if max_volatility is not None:
            latest_rows = [row for row in latest_rows if float(row.get("volatility_20d") or 0.0) <= float(max_volatility)]
        momentum_rows = momentum_scores(closes, windows=windows, vol_adjust=vol_adjust) if closes else []

        fundamentals_rows: list[dict[str, Any]] = []
        if include_value:
            loader = getattr(self.repo, "latest_fundamentals_snapshot", None)
            if callable(loader):
                fundamentals_rows = loader(tickers=latest_tickers, limit=max(50, len(latest_tickers)))

        payload = {
            "latest_rows": latest_rows,
            "momentum_rows": momentum_rows,
            "fundamentals_rows": fundamentals_rows,
        }
        self._discovery_cache = {"key": filters_key, "value": payload}
        return payload

    def screen_market(
        self,
        bucket: str | None = None,
        top_n: int = 10,
        *,
        per_bucket: int | None = None,
        windows: list[int] = [20, 60, 126],
        vol_adjust: bool = True,
        sort_by: str | None = None,
        order: str = "desc",
        min_ret_20d: float | None = None,
        max_volatility: float | None = None,
    ) -> list[dict]:
        """Discovers candidates across multiple styles inside the runtime universe.

        `bucket=None` (default) returns a balanced mix across momentum, pullback,
        recovery, defensive, and value. Set `bucket` explicitly to focus on one
        style. If `sort_by` is provided without a bucket, the tool falls back to
        the legacy single-field ranking mode.
        """
        bucket_token = str(bucket or "").strip().lower()
        windows = [int(w) for w in windows if int(w) > 1]
        windows = windows[:6] or [20, 60, 126]

        if sort_by and bucket_token and bucket_token not in {"balanced"}:
            logger.warning(
                "[yellow]screen_market ignoring legacy sort_by because bucket=%s was explicitly requested[/yellow]",
                bucket_token,
            )
            sort_by = None

        if sort_by:
            logger.info(
                "[cyan]TOOL[/cyan] screen_market legacy sort_by=%s order=%s min_ret_20d=%s max_volatility=%s top_n=%d",
                sort_by,
                order,
                str(min_ret_20d),
                str(max_volatility),
                int(top_n),
            )
            out = self.repo.screen_latest_features(
                sort_by=sort_by,
                order=order,
                tickers=self._target_universe(),
                min_ret_20d=min_ret_20d,
                max_volatility=max_volatility,
                top_n=top_n,
                sources=self._sources(),
            )
            self._log_tool_result("screen_market", out, key_fields=["ticker", "ret_20d", "volatility_20d"])
            return out

        include_value = bucket_token in {"", "balanced", "value"}
        logger.info(
            "[cyan]TOOL[/cyan] screen_market bucket=%s top_n=%d per_bucket=%s windows=%s vol_adjust=%s",
            bucket_token or "balanced",
            int(top_n),
            str(per_bucket),
            ",".join(str(w) for w in windows),
            str(bool(vol_adjust)).lower(),
        )
        payload = self._discovery_inputs(
            windows=windows,
            vol_adjust=vol_adjust,
            include_value=include_value,
            min_ret_20d=min_ret_20d,
            max_volatility=max_volatility,
        )
        out = build_discovery_rows(
            payload.get("latest_rows") or [],
            momentum_rows=payload.get("momentum_rows") or [],
            fundamentals_rows=payload.get("fundamentals_rows") or [],
            bucket=bucket_token or "balanced",
            top_n=top_n,
            per_bucket=per_bucket,
            order=order,
        )
        self._log_tool_result("screen_market", out, key_fields=["ticker", "bucket", "score", "ret_20d", "volatility_20d"])
        return out

    def _json_dict(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                import json

                parsed = json.loads(value)
                return dict(parsed) if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _recommend_opportunities_from_learned_rows(
        self,
        learned_rows: list[dict[str, Any]],
        *,
        top_n: int,
        buckets: list[str] | None,
        include_watchlist: bool,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_buckets = {
            str(bucket or "").strip().lower()
            for bucket in (buckets or [])
            if str(bucket or "").strip()
        }
        rows: list[dict[str, Any]] = []
        for raw in learned_rows:
            if not isinstance(raw, dict):
                continue
            ticker = str(raw.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            bucket = str(raw.get("bucket") or "").strip().lower()
            if allowed_buckets and bucket not in allowed_buckets:
                continue
            feature_json = self._json_dict(raw.get("feature_json"))
            explanation_json = self._json_dict(raw.get("explanation_json"))
            score = _to_float(raw.get("recommendation_score"), default=0.0) or 0.0
            pred_excess = _to_float(raw.get("predicted_excess_return_20d"), default=None)
            prob_outperform = _to_float(raw.get("prob_outperform_20d"), default=None)
            pred_drawdown = _to_float(raw.get("predicted_drawdown_20d"), default=None)
            action = str(raw.get("action") or "watchlist").strip().lower()
            confidence = str(raw.get("model_confidence") or "low").strip().lower()
            ranker_version = str(raw.get("ranker_version") or "").strip()
            reasons_for = [
                f"Learned IC ranker score={score:+.4f}",
            ]
            top_contribs = list(explanation_json.get("top_contributions") or [])
            if top_contribs:
                parts = []
                for item in top_contribs[:4]:
                    if isinstance(item, dict):
                        name = str(item.get("signal") or "")
                        contrib = _to_float(item.get("contribution"), default=0.0) or 0.0
                        if name:
                            parts.append(f"{name}({contrib:+.4f})")
                if parts:
                    reasons_for.append("contribs: " + " ".join(parts))
            if prob_outperform is not None:
                reasons_for.append(f"prob_up={float(prob_outperform):.1%}")
            risk_notes: list[str] = []
            blended = _to_float(explanation_json.get("blended_oos_ic_accuracy"), default=None)
            if blended is not None:
                risk_notes.append(f"blended_oos_ic={float(blended):+.3f}")
            scored_count = explanation_json.get("scored_signal_count")
            if scored_count is not None:
                risk_notes.append(f"signals_scored={int(scored_count)}")
            if confidence == "low":
                risk_notes.append("model_confidence=low")
            if explanation_json.get("days_since_regime") and int(explanation_json.get("days_since_regime") or 0) > 3:
                risk_notes.append("regime_stale")
            rows.append(
                {
                    "ticker": ticker,
                    "profile": str(raw.get("profile") or "balanced").strip().lower() or "balanced",
                    "tactical_kind": explanation_json.get("tactical_kind"),
                    "bucket": bucket,
                    "buckets": [bucket] if bucket else [],
                    "recommendation_rank": raw.get("recommendation_rank"),
                    "score": round(float(score), 6),
                    "recommendation_score": round(float(score), 6),
                    "score_source": str(raw.get("score_source") or "learned_ic"),
                    "ranker_version": ranker_version,
                    "predicted_excess_return_20d": pred_excess,
                    "prob_outperform_20d": prob_outperform,
                    "predicted_drawdown_20d": pred_drawdown,
                    "confidence": confidence,
                    "model_confidence": confidence,
                    "action": action,
                    "evidence_level": str(raw.get("evidence_level") or "validated"),
                    "reason": "; ".join(reasons_for),
                    "reason_for": "; ".join(reasons_for),
                    "reason_risk": "; ".join(risk_notes) or "Learned IC ranker output; inspect per-signal IC freshness.",
                    "signal_contributions": top_contribs,
                    "predicted_ic": explanation_json.get("predicted_ic"),
                    "forecast": {
                        "exp_return_period": feature_json.get("forecast_er"),
                        "prob_up": (0.5 + feature_json["forecast_prob"]) if feature_json.get("forecast_prob") is not None else None,
                    },
                    "optimizer_weight": raw.get("optimizer_weight"),
                    "optimizer_raw_weight": raw.get("optimizer_raw_weight"),
                    "validation": {
                        "ranker": "ok",
                        "features": "ok" if feature_json else "missing",
                        "optimizer": "ok" if raw.get("optimizer_weight") is not None else "not_applicable",
                    },
                    "evidence_gaps": [] if confidence != "low" else ["model_confidence_low"],
                    "feature_json": feature_json,
                    "explanation_json": explanation_json,
                }
            )

        rows.sort(
            key=lambda row: (
                int(row.get("recommendation_rank") or 9999),
                -float(row.get("recommendation_score") or 0.0),
                str(row.get("ticker") or ""),
            )
        )
        for idx, row in enumerate(rows, start=1):
            row["recommendation_rank"] = idx

        candidate_actions = {"candidate", "tactical_candidate"}
        watchlist_actions = {"watchlist", "tactical_watchlist"}
        recommendations = [row for row in rows if row.get("action") in candidate_actions]
        if include_watchlist and len(recommendations) < int(top_n):
            recommendations.extend(row for row in rows if row.get("action") in watchlist_actions)
        recommendations = recommendations[: max(1, min(int(top_n), 20))]

        by_profile: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            profile = str(row.get("profile") or "balanced")
            by_profile.setdefault(profile, []).append(row)
        by_profile = {profile: items[: max(1, min(int(top_n), 10))] for profile, items in by_profile.items()}
        first_explanation = self._json_dict(rows[0].get("explanation_json")) if rows else {}
        ranker = {
            "mode": "learned",
            "score_source": str(rows[0].get("score_source") or "learned_ic") if rows else "learned_ic",
            "model_family": first_explanation.get("model_family", "signal_ic_meta_learner"),
            "version": str(rows[0].get("ranker_version") or "") if rows else "",
            "rows": len(rows),
            "blended_oos_ic_accuracy": first_explanation.get("blended_oos_ic_accuracy"),
            "predicted_ic": first_explanation.get("predicted_ic"),
        }
        return {
            "status": "ok" if recommendations else "degraded",
            "recommendations": recommendations,
            "rows": rows,
            "by_profile": by_profile,
            "optimizer": {},
            "ranker": ranker,
            "diagnostics": diagnostics,
        }

    def recommend_opportunities(
        self,
        top_n: int = 8,
        *,
        buckets: list[str] | None = None,
        max_candidates: int | None = None,
        include_watchlist: bool = True,
        max_score_age_hours: int = 30,
    ) -> dict[str, Any]:
        """Returns precomputed signal-IC-weighted opportunities from BigQuery.

        Reads ``opportunity_ranker_scores_latest``; when rows are missing or
        stale, surfaces an explicit ``status='unusable'`` so the agent knows
        shared prep must be rerun. There is no heuristic fallback by design —
        silently substituting a different algorithm would hide failures.
        """
        diagnostics: dict[str, Any] = {
            "pipeline": ["signal_ic_meta_learner", "opportunity_ranker_scores_latest"],
            "max_score_age_hours": max_score_age_hours,
            "warnings": [],
        }
        scope = self._scope()
        market_filter = scope.row_market_filter()
        bucket_tokens = [
            str(bucket or "").strip().lower()
            for bucket in (buckets or [])
            if str(bucket or "").strip()
        ]
        bucket_tokens = list(dict.fromkeys(bucket_tokens))
        global_limit, per_profile_limit, selection_mode = _opportunity_selection_limits(
            top_n=top_n,
            max_candidates=max_candidates,
        )
        diagnostics["selection_scope"] = {
            "mode": selection_mode,
            "requested_max_candidates": max_candidates,
            "global_limit": global_limit,
            "per_profile_limit": per_profile_limit,
            "requested_buckets": bucket_tokens,
            "markets": market_filter,
        }
        learned_rows: list[dict[str, Any]] = []
        loader = getattr(self.repo, "latest_opportunity_ranker_scores", None)
        if callable(loader):
            try:
                learned_rows = loader(
                    limit=global_limit,
                    max_age_hours=max(1, min(int(max_score_age_hours), 24 * 14)),
                    buckets=bucket_tokens or None,
                    markets=market_filter or None,
                    per_profile_limit=per_profile_limit,
                ) or []
            except Exception as exc:
                diagnostics["warnings"].append(f"latest_opportunity_ranker_scores failed: {str(exc)[:160]}")
        else:
            diagnostics["warnings"].append("latest_opportunity_ranker_scores unavailable")
        diagnostics["selection_scope"]["loaded_rows"] = len(learned_rows)

        if learned_rows:
            out = self._recommend_opportunities_from_learned_rows(
                learned_rows,
                top_n=top_n,
                buckets=bucket_tokens,
                include_watchlist=include_watchlist,
                diagnostics=diagnostics,
            )
            self._log_tool_result("recommend_opportunities", out.get("rows") or [], key_fields=["ticker", "profile", "recommendation_score", "confidence", "action"])
            return out

        return {
            "status": "unusable",
            "error": "learned opportunity ranker scores are missing or stale; run build-opportunity-ranker in shared prep",
            "recommendations": [],
            "rows": [],
            "by_profile": {},
            "optimizer": {},
            "ranker": {"score_source": "missing"},
            "diagnostics": diagnostics,
        }


    def optimize_portfolio(
        self,
        tickers: list[str],
        strategy: str = "sharpe",
        lookback_days: int = 252,
        risk_free_rate: float = 0.04,
        mdd_days: int = 60,
        mu_confidence: float = 1.0,
        forecast_mode: str | None = None,
        regime_scale: float = 1.0,
        max_weight: float | None = None,
        min_weight: float | None = None,
        cash_buffer: float | None = None,
    ) -> dict:
        """Runs portfolio optimization with backtest MDD.

        strategy: 'sharpe' (Max-Sharpe Markowitz), 'risk_parity' (HRP), 'forecast' (ML forecast-enhanced).
        regime_scale: 0.5-1.0, scales all weights down for risk-off environments (default 1.0 = no scaling).
        max_weight: per-name cap (e.g. 0.35). Excess redistributed pro-rata to uncapped names.
        min_weight: drops names below threshold (e.g. 0.02), renormalizes remainder.
        cash_buffer: final cash reserve in [0, 1]; equities scaled to (1 - cash_buffer).
        """
        strategy = str(strategy or "sharpe").strip().lower()
        if strategy not in {"sharpe", "risk_parity", "forecast"}:
            return {"error": f"unknown strategy '{strategy}'; choose sharpe, risk_parity, or forecast"}

        in_scope, excluded_scope = self._partition_tickers_by_scope(tickers or [])
        if not in_scope:
            return {
                "status": "unusable",
                "strategy_requested": strategy,
                "error": "all requested tickers are outside the agent market scope",
                "excluded_from_market_scope": excluded_scope,
            }

        logger.info(
            "[cyan]TOOL[/cyan] optimize_portfolio strategy=%s tickers=%d lookback_days=%d excluded_scope=%d",
            strategy,
            len(in_scope),
            int(lookback_days),
            len(excluded_scope),
        )

        keep, rets, quality = self._load_aligned_returns(in_scope, lookback_days=lookback_days)
        if excluded_scope:
            quality["excluded_from_market_scope"] = excluded_scope

        if quality["status"] == "unusable":
            return {
                "status": "unusable",
                "strategy_requested": strategy,
                "data_quality": quality,
                "error": "no usable tickers after data quality filter",
            }

        degraded_reasons: list[str] = []
        forecast_coverage: float | None = None
        blended_mu: np.ndarray | None = None  # set only when forecast optimizer is actually used

        if len(keep) == 1:
            # Single-name graceful path — no optimizer needed.
            ticker = keep[0]
            daily_rets = rets[:, 0] if rets is not None else np.array([], dtype=float)
            exp_ret = float(np.nanmean(daily_rets)) if daily_rets.size else 0.0
            vol = float(np.nanstd(daily_rets, ddof=1)) if daily_rets.size > 1 else 0.0
            sharpe = 0.0
            if vol > 0:
                rf_daily = (1.0 + float(risk_free_rate)) ** (1.0 / 252.0) - 1.0
                sharpe = (exp_ret - rf_daily) / vol
            result = AllocationResult(
                weights={ticker: 1.0},
                expected_return=exp_ret,
                volatility=vol,
                sharpe=sharpe,
                strategy="single_name",
            )
            degraded_reasons.append("single_usable_ticker")
        else:
            if strategy == "forecast":
                mode = self._forecast_mode(forecast_mode)
                table_id = str(self.settings.forecast_table or "").strip() or None
                rows = self.repo.get_predicted_returns(
                    tickers=keep, limit=len(keep), mode=mode, table_id=table_id,
                )
                predicted_mu: dict[str, float] = {}
                if rows:
                    for r in rows:
                        t = str(r.get("ticker", "")).strip().upper()
                        try:
                            predicted_mu[t] = float(r.get("exp_return_period"))
                        except (KeyError, TypeError, ValueError):
                            continue
                forecast_coverage = len(predicted_mu) / float(len(keep))
                if forecast_coverage < 0.5:
                    degraded_reasons.append("forecast_coverage_insufficient")
                    result = optimize_hrp(keep, rets, risk_free_rate=risk_free_rate)
                else:
                    result = optimize_forecast_sharpe(
                        keep, rets, predicted_mu,
                        risk_free_rate=risk_free_rate,
                        mu_confidence=mu_confidence,
                    )
                    blended_mu = blend_forecast_mu(
                        keep, rets, predicted_mu, mu_confidence=mu_confidence,
                    )
            elif strategy == "risk_parity":
                result = optimize_hrp(keep, rets, risk_free_rate=risk_free_rate)
            else:
                result = optimize_max_sharpe(keep, rets, risk_free_rate=risk_free_rate)

        # Apply regime scaling to weights if requested.
        rs = max(0.3, min(float(regime_scale), 1.0))
        if rs < 1.0:
            scaled = {k: v * rs for k, v in result.weights.items()}
            result = AllocationResult(
                weights=scaled,
                expected_return=result.expected_return,
                volatility=result.volatility,
                sharpe=result.sharpe,
                strategy=getattr(result, "strategy", "unknown"),
            )

        # Apply weight constraints (cap / floor / cash buffer) if requested.
        # Only recompute stats when weights actually change — a non-binding
        # constraint (e.g. max_weight=1.0) must preserve the optimizer's stats,
        # which matters especially for forecast strategy (blended mu basis).
        constraints_applied: list[str] = []
        if any(x is not None for x in (max_weight, min_weight, cash_buffer)):
            constrained = apply_weight_constraints(
                result.weights,
                max_weight=max_weight,
                min_weight=min_weight,
                cash_buffer=cash_buffer,
            )
            if constrained != result.weights:
                if max_weight is not None:
                    constraints_applied.append(f"max_weight={float(max_weight):.3f}")
                if min_weight is not None:
                    constraints_applied.append(f"min_weight={float(min_weight):.3f}")
                if cash_buffer is not None:
                    constraints_applied.append(f"cash_buffer={float(cash_buffer):.3f}")

                # Drop columns for tickers removed by min_weight, keep rets + blended_mu aligned.
                dropped_idx = [i for i, t in enumerate(keep) if t not in constrained]
                if dropped_idx and rets is not None:
                    kept_idx = [i for i, t in enumerate(keep) if t in constrained]
                    rets = rets[:, kept_idx]
                    if blended_mu is not None:
                        blended_mu = blended_mu[kept_idx]
                    keep = [keep[i] for i in kept_idx]

                if rets is not None and len(keep) > 0:
                    new_ret, new_vol, new_sharpe = recompute_stats(
                        keep, constrained, rets,
                        risk_free_rate=risk_free_rate,
                        mu_override=blended_mu,
                    )
                else:
                    new_ret, new_vol, new_sharpe = result.expected_return, result.volatility, result.sharpe

                result = AllocationResult(
                    weights=constrained,
                    expected_return=new_ret,
                    volatility=new_vol,
                    sharpe=new_sharpe,
                    strategy=result.strategy,
                )

        out = self._format_allocation(keep, result, rets=rets, mdd_days=mdd_days)
        out["data_quality"] = quality
        if forecast_coverage is not None:
            out["forecast_coverage"] = round(float(forecast_coverage), 4)
        if degraded_reasons:
            out["status"] = "degraded"
            out["degraded_reasons"] = degraded_reasons
        else:
            out["status"] = "ok"
        out["strategy_requested"] = strategy
        if constraints_applied:
            out["constraints_applied"] = constraints_applied
        if rs < 1.0:
            out["regime_scale"] = round(rs, 4)

        evidence_gaps: list[str] = []
        if quality.get("excluded"):
            evidence_gaps.append("some_tickers_excluded")
        if forecast_coverage is not None and forecast_coverage < 1.0:
            evidence_gaps.append("forecast_coverage_partial")
        if evidence_gaps:
            out["evidence_gaps"] = evidence_gaps
        out["validation_notes"] = [
            "Entry timing not evaluated by this tool.",
            "Backtest MDD is target-basket drawdown over the last window, not walk-forward.",
        ]
        self._log_tool_result("optimize_portfolio", [out] if isinstance(out, dict) else out, key_fields=["ticker", "weight", "expected_return"])
        return out

    def forecast_returns(
        self,
        tickers: list[str] | None = None,
        forecast_mode: str | None = None,
    ) -> list[dict]:
        """Loads direction forecasts from 7-model ensemble (NBEATSx, NHITS, PatchTST, iTransformer, Chronos, TimesFM, Lag-Llama).

        Returns prob_up (0~1), model_votes_up/total, consensus label, and exp_return_period.
        If *tickers* is omitted, prefers unresolved discovery candidates plus current holdings
        from the active cycle context before falling back to the broader forecast universe.
        """
        _BQ_LIMIT = 500  # upper bound for BQ query; forecast table has ~50-60 rows
        mode = self._forecast_mode(forecast_mode)
        table_id = str(self.settings.forecast_table or "").strip() or None
        filt: list[str] | None = None
        excluded_scope: list[dict[str, str]] = []
        if tickers is not None:
            in_scope, excluded_scope = self._partition_tickers_by_scope(tickers)
            if excluded_scope:
                logger.warning(
                    "[yellow]forecast_returns dropped out-of-market tickers[/yellow] excluded=%d sample=%s",
                    len(excluded_scope),
                    excluded_scope[:5],
                )
            if not in_scope:
                return []
            filt = self._normalize_tickers(in_scope)
            if not filt:
                return []
        else:
            default_tickers = self._forecast_default_tickers()
            if default_tickers:
                filt = default_tickers
        logger.info(
            "[cyan]TOOL[/cyan] forecast_returns tickers=%s mode=%s excluded_scope=%d",
            str(len(filt)) if filt else "all",
            mode,
            len(excluded_scope),
        )
        rows = self.repo.get_predicted_returns(
            tickers=filt,
            limit=_BQ_LIMIT,
            mode=mode,
            table_id=table_id,
        )
        if not rows and self._forecast_auto_build_enabled() and self._auto_build_forecasts_if_needed():
            rows = self.repo.get_predicted_returns(
                tickers=filt,
                limit=_BQ_LIMIT,
                mode=mode,
                table_id=table_id,
            )
        if not rows:
            logger.warning(
                "[yellow]forecast_returns returned no rows[/yellow] tickers=%s mode=%s",
                str(len(filt)) if filt else "all",
                mode,
            )
            return []
        compact_rows = self._compact_forecast_rows(rows)
        self._log_tool_result("forecast_returns", compact_rows, key_fields=["ticker", "prob_up", "consensus", "model_votes_up", "model_votes_total", "exp_return_period", "forecast_model"])
        return compact_rows

    def _technical_signals_one(self, ticker: str, *, lookback_days: int) -> dict[str, Any]:
        """Calculates RSI/MACD/Bollinger/SMA signals for a single ticker."""
        token = str(ticker or "").strip().upper()
        if not token:
            return {"error": "ticker is required"}

        universe = set(self._target_universe())
        if universe and token not in universe:
            logger.info(
                "[yellow]TOOL technical_signals out-of-universe accepted[/yellow] ticker=%s",
                token,
            )

        lookback = max(60, min(int(lookback_days), 600))
        logger.info("[cyan]TOOL[/cyan] technical_signals ticker=%s lookback_days=%d", token, lookback)

        closes = self.repo.get_daily_closes(
            tickers=[token],
            lookback_days=lookback,
            sources=daily_history_sources(self._sources()),
        )
        series = closes.get(token, [])
        if len(series) < 35:
            return {"error": "insufficient history", "ticker": token, "points": len(series)}

        prices = np.array(series, dtype=float)
        last = float(prices[-1])

        def _ema(values: np.ndarray, span: int) -> np.ndarray:
            alpha = 2.0 / (float(span) + 1.0)
            out = np.empty_like(values, dtype=float)
            out[0] = values[0]
            for i in range(1, len(values)):
                out[i] = (alpha * values[i]) + ((1.0 - alpha) * out[i - 1])
            return out

        delta = np.diff(prices)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = float(np.mean(gains[-14:])) if len(gains) >= 14 else 0.0
        avg_loss = float(np.mean(losses[-14:])) if len(losses) >= 14 else 0.0
        if avg_loss <= 0:
            rsi14 = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi14 = 100.0 - (100.0 / (1.0 + rs))

        ema12 = _ema(prices, 12)
        ema26 = _ema(prices, 26)
        macd_line = ema12 - ema26
        signal_line = _ema(macd_line, 9)
        macd_hist = macd_line - signal_line

        sma20 = float(np.mean(prices[-20:]))
        std20 = float(np.std(prices[-20:]))
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20
        sma50 = float(np.mean(prices[-50:])) if len(prices) >= 50 else None

        macd_cross = "neutral"
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
                macd_cross = "bullish_cross"
            elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
                macd_cross = "bearish_cross"
            elif macd_line[-1] > signal_line[-1]:
                macd_cross = "bullish"
            elif macd_line[-1] < signal_line[-1]:
                macd_cross = "bearish"

        if rsi14 >= 70.0:
            rsi_state = "overbought"
        elif rsi14 <= 30.0:
            rsi_state = "oversold"
        else:
            rsi_state = "neutral"

        if last >= bb_upper:
            bb_state = "near_upper_band"
        elif last <= bb_lower:
            bb_state = "near_lower_band"
        else:
            bb_state = "inside_bands"

        if sma50 is None:
            trend = "unknown"
        elif last >= sma20 >= sma50:
            trend = "uptrend"
        elif last <= sma20 <= sma50:
            trend = "downtrend"
        else:
            trend = "sideways"

        # ── Volume analysis (best-effort) ──
        volume_data: dict[str, Any] | None = None
        try:
            is_kospi = token.isdigit() and len(token) == 6
            client = self._ot()
            if is_kospi:
                vol_rows = client.get_domestic_index_daily(token) if False else []
                # Use daily price API for volume
                from datetime import datetime, timedelta
                end_d = datetime.now().strftime("%Y%m%d")
                start_d = (datetime.now() - timedelta(days=lookback)).strftime("%Y%m%d")
                vol_rows = client.get_domestic_daily_price(ticker=token, start_date=start_d, end_date=end_d, max_pages=2)
                vols = [float(r.get("acml_vol") or 0) for r in vol_rows if float(r.get("acml_vol") or 0) > 0]
            else:
                vol_rows = client.get_overseas_daily_price(ticker=token, max_pages=2)
                vols = [float(r.get("tvol") or r.get("acml_vol") or 0) for r in vol_rows if float(r.get("tvol") or r.get("acml_vol") or 0) > 0]

            if len(vols) >= 5:
                latest_vol = vols[0] if vols[0] > 0 else vols[-1]  # rows newest-first
                avg_20d = float(np.mean(vols[:20])) if len(vols) >= 20 else float(np.mean(vols))
                vol_ratio = round(latest_vol / avg_20d, 4) if avg_20d > 0 else None

                # OBV trend: simplified from last 5 price changes + volume
                obv_changes = 0
                n_obv = min(5, len(prices) - 1, len(vols) - 1)
                for i in range(n_obv):
                    if prices[-(i + 1)] > prices[-(i + 2)]:
                        obv_changes += 1
                    elif prices[-(i + 1)] < prices[-(i + 2)]:
                        obv_changes -= 1
                if obv_changes >= 2:
                    obv_trend = "accumulation"
                elif obv_changes <= -2:
                    obv_trend = "distribution"
                else:
                    obv_trend = "neutral"

                # Price-volume confirmation
                price_up = float(prices[-1]) > float(prices[-2]) if len(prices) >= 2 else False
                vol_up = latest_vol > avg_20d if avg_20d > 0 else False
                if price_up and vol_up:
                    pv_confirm = "confirmed"
                elif price_up and not vol_up:
                    pv_confirm = "divergence"
                elif not price_up and vol_up:
                    pv_confirm = "selling_pressure"
                else:
                    pv_confirm = "neutral"

                volume_data = {
                    "latest": int(latest_vol),
                    "avg_20d": round(avg_20d, 0),
                    "volume_ratio": vol_ratio,
                    "obv_trend": obv_trend,
                    "price_volume_confirm": pv_confirm,
                }
        except Exception as exc:
            logger.debug("Volume fetch failed ticker=%s err=%s", token, str(exc)[:80])

        # ── Investor flow (KOSPI only, best-effort) ──
        investor_flow: dict[str, Any] | None = None
        try:
            if token.isdigit() and len(token) == 6 and self._has_kospi_market():
                from datetime import datetime, timedelta
                end_d = datetime.now().strftime("%Y%m%d")
                start_d = (datetime.now() - timedelta(days=10)).strftime("%Y%m%d")
                flow_rows = self._ot().get_domestic_investor_daily(token, start_date=start_d, end_date=end_d)
                if flow_rows:
                    frgn_sum = sum(float(r.get("frgn_ntby_qty") or 0) for r in flow_rows[:5])
                    orgn_sum = sum(float(r.get("orgn_ntby_qty") or 0) for r in flow_rows[:5])
                    if frgn_sum > 0 and orgn_sum > 0:
                        flow_signal = "smart_money_buy"
                    elif frgn_sum < 0 and orgn_sum < 0:
                        flow_signal = "smart_money_sell"
                    elif frgn_sum > 0 or orgn_sum > 0:
                        flow_signal = "bullish"
                    elif frgn_sum < 0 or orgn_sum < 0:
                        flow_signal = "bearish"
                    else:
                        flow_signal = "neutral"
                    investor_flow = {
                        "foreign_net_buy_5d": int(frgn_sum),
                        "institution_net_buy_5d": int(orgn_sum),
                        "flow_signal": flow_signal,
                    }
        except Exception as exc:
            logger.debug("Investor flow fetch failed ticker=%s err=%s", token, str(exc)[:80])

        # ── Short selling ratio (KOSPI only, best-effort) ──
        short_sale: dict[str, Any] | None = None
        try:
            if token.isdigit() and len(token) == 6 and self._has_kospi_market():
                ss_rows = self._ot().get_domestic_daily_short_sale(ticker=token)
                if ss_rows:
                    recent = ss_rows[:5]
                    ratios = [float(r.get("ssts_vol_rlim") or 0) for r in recent]
                    qtys = [float(r.get("ssts_cntg_qty") or 0) for r in recent]
                    avg_ratio = round(sum(ratios) / len(ratios), 2) if ratios else 0
                    latest_ratio = ratios[0] if ratios else 0
                    latest_qty = int(qtys[0]) if qtys else 0
                    short_sale = {
                        "latest_ratio_pct": latest_ratio,
                        "avg_5d_ratio_pct": avg_ratio,
                        "latest_qty": latest_qty,
                    }
        except Exception as exc:
            logger.debug("Short sale fetch failed ticker=%s err=%s", token, str(exc)[:80])

        result: dict[str, Any] = {
            "ticker": token,
            "price": round(last, 6),
            "rsi_14": round(float(rsi14), 4),
            "rsi_state": rsi_state,
            "macd": {
                "line": round(float(macd_line[-1]), 6),
                "signal": round(float(signal_line[-1]), 6),
                "hist": round(float(macd_hist[-1]), 6),
                "state": macd_cross,
            },
            "moving_averages": {
                "sma_20": round(float(sma20), 6),
                "sma_50": round(float(sma50), 6) if sma50 is not None else None,
                "price_vs_sma20": round(float((last / sma20) - 1.0), 6) if sma20 > 0 else None,
            },
            "bollinger_20_2": {
                "upper": round(float(bb_upper), 6),
                "mid": round(float(sma20), 6),
                "lower": round(float(bb_lower), 6),
                "state": bb_state,
            },
            "trend_state": trend,
            "points": int(len(prices)),
        }
        if volume_data is not None:
            result["volume"] = volume_data
        if investor_flow is not None:
            result["investor_flow"] = investor_flow
        if short_sale is not None:
            result["short_sale"] = short_sale
        return result

    def technical_signals(
        self,
        ticker: str = "",
        *,
        tickers: list[str] | None = None,
        lookback_days: int = 180,
    ) -> dict[str, Any]:
        """Calculates RSI/MACD/Bollinger/SMA signals for one or more tickers."""
        raw_tokens: list[str] = []

        if tickers is not None:
            raw_tokens.extend(self._normalize_tickers(tickers, restrict_to_universe=False))
        elif str(ticker or "").strip():
            raw_tokens.append(str(ticker).strip().upper())
        elif bool(getattr(self.settings, "autonomy_tool_default_candidates_enabled", False)):
            raw_tokens.extend(self._analysis_default_tickers(limit=10))

        raw_tokens = list(dict.fromkeys([t for t in raw_tokens if t]))
        if not raw_tokens:
            return {"error": "ticker or tickers is required"}

        tokens, excluded = self._partition_tickers_by_scope(raw_tokens)
        if not tokens:
            return {
                "error": "all requested tickers are outside the agent market scope",
                "tickers": raw_tokens,
                "excluded_from_market_scope": excluded,
            }

        lookback = max(60, min(int(lookback_days), 600))
        if len(tokens) == 1 and not excluded:
            return self._technical_signals_one(tokens[0], lookback_days=lookback)

        logger.info(
            "[cyan]TOOL[/cyan] technical_signals tickers=%d lookback_days=%d excluded_scope=%d",
            len(tokens),
            lookback,
            len(excluded),
        )
        rows = [self._technical_signals_one(token, lookback_days=lookback) for token in tokens]
        self._log_tool_result("technical_signals", rows, key_fields=["ticker", "rsi_state", "trend_state"])
        result: dict[str, Any] = {
            "tickers": tokens,
            "rows": rows,
            "count": len(rows),
        }
        if excluded:
            result["excluded_from_market_scope"] = excluded
        return result

    def sector_summary(self, period: str = "20d") -> list[dict]:
        """Summarizes average return/volatility by sector from latest features."""
        period_key = str(period).strip().lower()
        field = "ret_20d" if period_key in {"20d", "1m"} else "ret_5d"
        logger.info("[cyan]TOOL[/cyan] sector_summary period=%s field=%s", period_key, field)

        rows = self.repo.screen_latest_features(
            sort_by=field,
            order="desc",
            tickers=self._target_universe(),
            top_n=200,
            sources=self._sources(),
        )
        rows = normalize_market_feature_rows(
            rows,
            repo=self.repo,
            sources=self._sources(),
            lookback_days=22,
        )

        buckets: dict[str, list[dict]] = {}
        for r in rows:
            t = str(r.get("ticker", "")).strip().upper()
            if not t:
                continue
            sector = SECTOR_BY_TICKER.get(t, "Unknown")
            buckets.setdefault(sector, []).append(r)

        out: list[dict] = []
        for sector, items in buckets.items():
            rets = [float(i.get(field) or 0.0) for i in items]
            vols = [float(i.get("volatility_20d") or 0.0) for i in items]
            out.append(
                {
                    "sector": sector,
                    "avg_ret": float(np.mean(rets)) if rets else 0.0,
                    "avg_vol": float(np.mean(vols)) if vols else 0.0,
                    "tickers": [str(i.get("ticker")) for i in items[:5]],
                }
            )

        out.sort(key=lambda x: x.get("avg_ret", 0.0), reverse=True)
        return out

    def get_fundamentals(
        self,
        tickers: list[str] | None = None,
        *,
        excd: str = "NAS",
        max_items: int = 10,
    ) -> dict[str, Any]:
        """Fetches per-ticker valuation metrics. Auto-routes to US (PER/PBR/EPS/BPS) or KOSPI (EPS/BPS/ROE/부채비율) based on agent market."""
        has_us = self._has_us_market()
        has_kospi = self._has_kospi_market()

        if tickers is not None:
            base_tickers = tickers
        elif bool(getattr(self.settings, "autonomy_tool_default_candidates_enabled", False)):
            base_tickers = self._analysis_default_tickers(limit=max_items)
        else:
            base_tickers = []
        requested = self._normalize_tickers(base_tickers, restrict_to_universe=False)
        if not requested:
            return {"error": "no tickers provided", "rows": []}

        in_scope, excluded_scope = self._partition_tickers_by_scope(requested)
        if not in_scope:
            return {
                "error": "all requested tickers are outside the agent market scope",
                "rows": [],
                "excluded_from_market_scope": excluded_scope,
            }

        allowed = set(self._target_universe())
        eligible = [t for t in in_scope if t in allowed] if allowed else in_scope
        excluded = [t for t in in_scope if t not in set(eligible)]
        eligible = eligible[: max(1, min(int(max_items), 50))]

        # Split tickers by market type
        us_tickers = [t for t in eligible if not t[:1].isdigit()] if has_us else []
        kospi_tickers = [t for t in eligible if t.isdigit() and len(t) == 6] if has_kospi else []

        logger.info(
            "[cyan]TOOL[/cyan] get_fundamentals tickers=%d us=%d kospi=%d",
            len(eligible), len(us_tickers), len(kospi_tickers),
        )

        rows: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        client = self._ot()

        # US fundamentals via overseas price detail
        for ticker in us_tickers:
            try:
                exchange, raw = self._fetch_us_fundamental_snapshot(
                    client=client,
                    ticker=ticker,
                    requested_exchange=excd,
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "market": "us",
                        "exchange": exchange,
                        "currency": str(raw.get("curr", "")).strip(),
                        "last": _to_float(raw.get("last"), default=None),
                        "market_cap": _to_float(raw.get("tomv"), default=None),
                        "per": _to_float(raw.get("perx"), default=None),
                        "pbr": _to_float(raw.get("pbrx"), default=None),
                        "eps": _to_float(raw.get("epsx"), default=None),
                        "bps": _to_float(raw.get("bpsx"), default=None),
                        "tradable": str(raw.get("e_ordyn", "")).strip(),
                    }
                )
            except Exception as exc:
                errors.append({"ticker": ticker, "error": str(exc)[:240]})

        # KOSPI fundamentals via domestic financial ratio API
        for ticker in kospi_tickers:
            try:
                ratio_rows = client.get_domestic_financial_ratio(ticker=ticker)
                if not ratio_rows:
                    errors.append({"ticker": ticker, "error": "no financial ratio data"})
                    continue
                latest = ratio_rows[0]
                row_data: dict[str, Any] = {
                    "ticker": ticker,
                    "market": "kospi",
                    "exchange": "KRX",
                    "currency": "KRW",
                    "eps": _to_float(latest.get("eps"), default=None),
                    "bps": _to_float(latest.get("bps"), default=None),
                    "sps": _to_float(latest.get("sps"), default=None),
                    "roe": _to_float(latest.get("roe_val"), default=None),
                    "debt_ratio": _to_float(latest.get("lblt_rate"), default=None),
                    "reserve_ratio": _to_float(latest.get("rsrv_rate"), default=None),
                    "operating_profit_growth": _to_float(latest.get("bsop_prfi_inrt"), default=None),
                    "net_profit_growth": _to_float(latest.get("ntin_inrt"), default=None),
                    "settlement_date": str(latest.get("stac_yymm", "")).strip(),
                }

                # Analyst consensus (best-effort)
                try:
                    opinion_rows = client.get_domestic_invest_opinion(ticker=ticker)
                    if opinion_rows:
                        latest_op = opinion_rows[0]
                        target_price = _to_float(latest_op.get("hts_goal_prc"), default=None)
                        opinion = str(latest_op.get("invt_opnn") or "").strip()
                        prev_close = _to_float(latest_op.get("stck_prdy_clpr"), default=None)
                        gap_pct = _to_float(latest_op.get("nday_dprt"), default=None)
                        if target_price or opinion:
                            consensus: dict[str, Any] = {}
                            if opinion:
                                consensus["opinion"] = opinion
                            if target_price:
                                consensus["target_price"] = target_price
                            if prev_close and target_price and prev_close > 0:
                                consensus["upside_pct"] = round((target_price / prev_close - 1) * 100, 2)
                            if gap_pct is not None:
                                consensus["gap_pct"] = gap_pct
                            consensus["reports"] = len(opinion_rows)
                            row_data["consensus"] = consensus
                except Exception:
                    pass  # best-effort

                rows.append(row_data)
            except Exception as exc:
                errors.append({"ticker": ticker, "error": str(exc)[:240]})

        result: dict[str, Any] = {
            "requested": requested,
            "eligible": eligible,
            "excluded": excluded,
            "rows": rows,
            "errors": errors,
        }
        if excluded_scope:
            result["excluded_from_market_scope"] = excluded_scope
        return result

    def _fetch_fred_latest(self, series_id: str) -> tuple[str, float | None]:
        """Fetches the latest valid observation from FRED API."""
        api_key = getattr(self.settings, "fred_api_key", "")
        if not api_key:
            return "", None
        import requests as _requests
        try:
            resp = _requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": "5",
                },
                timeout=10,
            )
            resp.raise_for_status()
            for obs in resp.json().get("observations", []):
                raw = str(obs.get("value", "")).strip()
                if raw and raw != ".":
                    return str(obs.get("date", "")), float(raw)
        except Exception as exc:
            logger.warning("[yellow]FRED fetch failed[/yellow] series=%s err=%s", series_id, str(exc)[:120])
        return "", None

    def index_snapshot(
        self,
        indices: list[str] | None = None,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """주요 시장지수, 원자재, 채권 수익률 요약을 반환한다. 에이전트의 타겟 마켓에 따라 적절한 지수를 선택한다."""
        scope = self._scope()
        has_us = scope.has_us
        has_kospi = scope.has_kospi

        _VALID_SYMS = set(_INDEX_MAP) | set(_FRED_INDICATORS) | set(_OVERSEAS_COMMODITY_ETFS)

        if indices is None:
            targets: list[str] = []
            for sym, (_, _, group) in _INDEX_MAP.items():
                if group == "global":
                    targets.append(sym)
                elif group == "us" and has_us:
                    targets.append(sym)
                elif group == "kospi" and has_kospi:
                    targets.append(sym)
            _FRED_US_INDEX_SYMS = {"SPX", "COMP", "DJI"}
            for sym in _FRED_INDICATORS:
                if sym in _FRED_US_INDEX_SYMS and not has_us:
                    continue
                targets.append(sym)
            # Gold (via GLD ETF) is a global commodity — include for all agents,
            # same as WTI/US10Y/US30Y from FRED which are market-agnostic macros.
            for sym in _OVERSEAS_COMMODITY_ETFS:
                targets.append(sym)
        else:
            raw = [str(s).strip().upper() for s in indices if str(s).strip()]
            invalid = [s for s in raw if s not in _VALID_SYMS]
            targets = [s for s in raw if s in _VALID_SYMS]
            if invalid:
                logger.warning(
                    "index_snapshot: ignored invalid symbols %s (valid: %s)",
                    invalid, sorted(_VALID_SYMS),
                )
            if not targets:
                return {
                    "indices": [],
                    "errors": [
                        {
                            "symbol": ",".join(invalid),
                            "error": (
                                f"Invalid index symbols. This tool is for market indices only, "
                                f"not individual stock tickers. Valid symbols: {sorted(_VALID_SYMS)}"
                            ),
                        }
                    ],
                    "source": "none",
                }
        targets = targets[:12]

        lookback = max(5, min(int(lookback_days), 120))
        logger.info("[cyan]TOOL[/cyan] index_snapshot indices=%s lookback=%d markets=%s", ",".join(targets), lookback, ",".join(sorted(scope.markets)))

        from datetime import datetime, timedelta

        client = self._ot()
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y%m%d")
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        # Stock indices via KIS API
        for sym in targets:
            if sym in _FRED_INDICATORS or sym in _OVERSEAS_COMMODITY_ETFS:
                continue
            info = _INDEX_MAP.get(sym)
            label = info[1] if info else sym
            try:
                if sym in _DOMESTIC_INDEX_SYMS:
                    # Korean domestic indices use different API + response fields
                    iscd = info[0] if info else sym
                    rows = client.get_domestic_index_daily(iscd, start_date=start_date, end_date=end_date, max_pages=1)
                    close_field = "bstp_nmix_prca"  # 업종지수종가
                else:
                    rows = client.get_overseas_index_daily(sym, start_date=start_date, end_date=end_date, max_pages=1)
                    close_field = "ovrs_nmix_prca"
            except Exception as exc:
                errors.append({"symbol": sym, "error": str(exc)[:200]})
                continue

            if not rows:
                errors.append({"symbol": sym, "error": "no data returned"})
                continue

            closes = []
            for r in rows:
                try:
                    closes.append(float(r.get(close_field) or r.get("clos") or 0))
                except (TypeError, ValueError):
                    continue
            closes = [c for c in closes if c > 0]

            if not closes:
                errors.append({"symbol": sym, "error": "no valid close prices"})
                continue

            last_close = closes[0]
            entry: dict[str, Any] = {"symbol": sym, "name": label, "close": round(last_close, 2), "type": "index"}

            for window, key in [(1, "change_1d"), (5, "return_5d"), (20, "return_20d")]:
                if len(closes) > window:
                    pct = ((closes[0] / closes[window]) - 1.0) * 100.0
                    entry[key] = round(pct, 2)

            results.append(entry)

        # Indices, commodities & bonds via FRED API
        _FRED_INDEX_SYMS = {"SPX", "COMP", "DJI"}
        _FRED_COMMODITY_SYMS = {"WTI"}
        for sym in targets:
            if sym not in _FRED_INDICATORS:
                continue
            series_id, label, unit = _FRED_INDICATORS[sym]
            fred_date, fred_val = self._fetch_fred_latest(series_id)
            if fred_val is not None:
                if sym in _FRED_INDEX_SYMS:
                    entry_type = "index"
                elif sym in _FRED_COMMODITY_SYMS:
                    entry_type = "commodity"
                else:
                    entry_type = "bond_yield"
                results.append({
                    "symbol": sym,
                    "name": label,
                    "value": round(fred_val, 4) if sym not in _FRED_INDEX_SYMS else round(fred_val, 2),
                    "date": fred_date,
                    "unit": unit,
                    "type": entry_type,
                })
            else:
                errors.append({"symbol": sym, "error": "FRED data unavailable"})

        # Overseas commodity ETFs via KIS overseas price API (gold via GLD, etc.)
        for sym in targets:
            if sym not in _OVERSEAS_COMMODITY_ETFS:
                continue
            etf_ticker, label, unit = _OVERSEAS_COMMODITY_ETFS[sym]
            try:
                exchange, raw = self._fetch_us_fundamental_snapshot(
                    client=client,
                    ticker=etf_ticker,
                    requested_exchange=None,
                )
            except Exception as exc:
                errors.append({"symbol": sym, "error": f"{etf_ticker}: {str(exc)[:160]}"})
                continue
            last = _to_float(raw.get("last"), default=None)
            if last is None or last <= 0:
                errors.append({"symbol": sym, "error": f"{etf_ticker}: price unavailable"})
                continue
            results.append({
                "symbol": sym,
                "name": label,
                "proxy_ticker": etf_ticker,
                "exchange": exchange or "",
                "value": round(last, 2),
                "date": str(raw.get("xymd") or "").strip(),
                "unit": unit,
                "type": "commodity",
            })

        return {
            "indices": results,
            "errors": errors if errors else None,
            "source": ("kis_index+fred" if (has_us or has_kospi) else "fred"),
        }


_INDEX_MAP: dict[str, tuple[str, str, str]] = {
    # (api_symbol, display_name, market_group)
    # Korean stock indices (domestic API iscd codes)
    "KOSPI": ("0001", "KOSPI", "kospi"),
    "KOSPI200": ("0028", "KOSPI 200", "kospi"),
    "KOSDAQ": ("1001", "KOSDAQ", "kospi"),
}

# Symbols that use the domestic index API (KIS inquire-daily-indexchartprice)
_DOMESTIC_INDEX_SYMS: set[str] = {"KOSPI", "KOSPI200", "KOSDAQ"}

_FRED_INDICATORS: dict[str, tuple[str, str, str]] = {
    # (fred_series_id, display_name, unit)
    # US stock indices — FRED provides reliable daily data
    "SPX": ("SP500", "S&P 500", "pt"),
    "COMP": ("NASDAQCOM", "NASDAQ Composite", "pt"),
    "DJI": ("DJIA", "Dow Jones Industrial", "pt"),
    # Commodities & bonds
    # FRED gold series (GOLDPMGBD228NLBM / GOLDAMGBD228NLBM) discontinued in 2015.
    # Gold is sourced via GLD ETF through KIS overseas API — see _OVERSEAS_COMMODITY_ETFS.
    "WTI": ("DCOILWTICO", "Crude Oil WTI", "$/bbl"),
    "US10Y": ("DGS10", "US 10Y Treasury Yield", "%"),
    "US30Y": ("DGS30", "US 30Y Treasury Yield", "%"),
}

# US commodity proxies sourced via KIS overseas price API (ETFs that track spot).
# (etf_ticker, display_name, unit)
_OVERSEAS_COMMODITY_ETFS: dict[str, tuple[str, str, str]] = {
    "GOLD": ("GLD", "SPDR Gold Shares (ETF, tracks gold spot)", "$/share"),
}
