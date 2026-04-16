from __future__ import annotations

from typing import Any


def _clip_text(value: Any, *, max_len: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)].rstrip() + "..."


def _compact_rows(
    rows: Any,
    *,
    fields: tuple[str, ...],
    limit: int = 10,
    text_fields: tuple[str, ...] = (),
    max_text: int = 180,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for row in rows[: max(1, min(int(limit), 30))]:
        if not isinstance(row, dict):
            continue
        item: dict[str, Any] = {}
        for field in fields:
            if row.get(field) is None:
                continue
            value = row.get(field)
            if field in text_fields:
                value = _clip_text(value, max_len=max_text)
            item[field] = value
        if item:
            out.append(item)
    return out


def _compact_memory_context_rows(rows: Any) -> list[dict[str, Any]]:
    return _compact_rows(
        rows,
        fields=("event_id", "created_date", "summary", "score", "author_id", "agent_id", "memory_source"),
        limit=3,
        text_fields=("summary",),
        max_text=180,
    )


def _compact_tool_result_for_prompt(
    tool_name: str,
    value: Any,
    *,
    args: dict[str, Any] | None = None,
) -> Any:
    token = str(tool_name or "").strip().lower()
    tool_args = args or {}

    memory_ctx = None
    core = value
    if isinstance(value, dict) and "_memory_context" in value:
        memory_ctx = value.get("_memory_context")
        if set(value.keys()) == {"data", "_memory_context"}:
            core = value.get("data")
        else:
            copied = dict(value)
            copied.pop("_memory_context", None)
            core = copied

    compacted: Any = core

    if token == "screen_market":
        compacted = _compact_rows(
            core,
            fields=(
                "ticker",
                "bucket",
                "bucket_rank",
                "score",
                "reason",
                "reason_for",
                "reason_risk",
                "ret_20d",
                "ret_5d",
                "volatility_20d",
                "sentiment_score",
                "per",
                "pbr",
                "roe",
                "debt_ratio",
                "close_price_krw",
                "evidence_level",
            ),
            limit=12,
            text_fields=("reason", "reason_for", "reason_risk"),
            max_text=140,
        )
    elif token == "forecast_returns":
        rows = _compact_rows(
            core,
            fields=(
                "run_date",
                "ticker",
                "exp_return_period",
                "forecast_horizon",
                "forecast_model",
                "is_stacked",
                "forecast_score",
                "prob_up",
                "model_votes_up",
                "model_votes_total",
                "consensus",
                "best_base_model",
                "best_base_return",
            ),
            limit=12,
        )
        if isinstance(core, list):
            compacted_rows: list[dict[str, Any]] = []
            for src, item in zip(core[:12], rows):
                if not isinstance(src, dict):
                    continue
                stacked = _compact_rows(
                    src.get("stacked_models"),
                    fields=("forecast_model", "exp_return_period", "forecast_score"),
                    limit=3,
                )
                base = _compact_rows(
                    src.get("base_models"),
                    fields=("forecast_model", "exp_return_period", "forecast_score"),
                    limit=3,
                )
                if stacked:
                    item["stacked_models"] = stacked
                if base:
                    item["base_models"] = base
                compacted_rows.append(item)
            compacted = compacted_rows
    elif token == "technical_signals":
        if isinstance(core, dict) and isinstance(core.get("rows"), list):
            rows: list[dict[str, Any]] = []
            for row in core.get("rows", [])[:10]:
                if not isinstance(row, dict):
                    continue
                ma = row.get("moving_averages") or {}
                bb = row.get("bollinger_20_2") or {}
                macd = row.get("macd") or {}
                item: dict[str, Any] = {
                    "ticker": row.get("ticker"),
                    "price": row.get("price"),
                    "rsi_14": row.get("rsi_14"),
                    "rsi_state": row.get("rsi_state"),
                    "macd_state": macd.get("state"),
                    "trend_state": row.get("trend_state"),
                    "price_vs_sma20": ma.get("price_vs_sma20"),
                    "bb_state": bb.get("state"),
                }
                if row.get("investor_flow"):
                    item["investor_flow"] = row["investor_flow"]
                if row.get("short_sale"):
                    item["short_sale"] = row["short_sale"]
                rows.append(item)
            compacted = {
                "tickers": list(core.get("tickers") or [])[:10],
                "count": len(rows),
                "rows": rows,
            }
        elif isinstance(core, dict) and "error" not in core:
            ma = core.get("moving_averages") or {}
            bb = core.get("bollinger_20_2") or {}
            macd = core.get("macd") or {}
            compacted = {
                "ticker": core.get("ticker"),
                "price": core.get("price"),
                "rsi_14": core.get("rsi_14"),
                "rsi_state": core.get("rsi_state"),
                "macd": {
                    "line": macd.get("line"),
                    "signal": macd.get("signal"),
                    "hist": macd.get("hist"),
                    "state": macd.get("state"),
                },
                "moving_averages": {
                    "sma_20": ma.get("sma_20"),
                    "sma_50": ma.get("sma_50"),
                    "price_vs_sma20": ma.get("price_vs_sma20"),
                },
                "bb_state": bb.get("state"),
                "trend_state": core.get("trend_state"),
                "points": core.get("points"),
            }
            if core.get("investor_flow"):
                compacted["investor_flow"] = core["investor_flow"]
            if core.get("short_sale"):
                compacted["short_sale"] = core["short_sale"]
    elif token == "sector_summary":
        rows: list[dict[str, Any]] = []
        if isinstance(core, list):
            for row in core[:10]:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "sector": row.get("sector"),
                        "avg_ret": row.get("avg_ret"),
                        "avg_vol": row.get("avg_vol"),
                        "leaders": list(row.get("tickers") or [])[:3],
                    }
                )
        compacted = rows
    elif token == "get_fundamentals" and isinstance(core, dict):
        rows: list[dict[str, Any]] = []
        for row in list(core.get("rows") or [])[:12]:
            if not isinstance(row, dict):
                continue
            item: dict[str, Any] = {
                "ticker": row.get("ticker"),
                "market": row.get("market"),
            }
            for field in (
                "last",
                "market_cap",
                "per",
                "pbr",
                "eps",
                "bps",
                "roe",
                "debt_ratio",
                "currency",
                "exchange",
                "settlement_date",
                "consensus",
            ):
                if row.get(field) is not None:
                    item[field] = row.get(field)
            rows.append(item)
        errors = _compact_rows(core.get("errors"), fields=("ticker", "error"), limit=5, text_fields=("error",), max_text=140)
        excluded = [str(t).strip().upper() for t in list(core.get("excluded") or []) if str(t).strip()]
        compacted = {
            "requested_count": len(list(core.get("requested") or [])),
            "eligible_count": len(list(core.get("eligible") or [])),
            "excluded_count": len(excluded),
            "rows": rows,
        }
        if excluded:
            compacted["excluded"] = excluded[:5]
        if errors:
            compacted["errors"] = errors
    elif token == "index_snapshot" and isinstance(core, dict):
        rows: list[dict[str, Any]] = []
        for row in list(core.get("indices") or [])[:12]:
            if not isinstance(row, dict):
                continue
            item: dict[str, Any] = {
                "symbol": row.get("symbol"),
                "name": row.get("name"),
                "type": row.get("type"),
            }
            for field in ("close", "value", "unit", "change_1d", "return_5d", "return_20d", "date"):
                if row.get(field) is not None:
                    item[field] = row.get(field)
            rows.append(item)
        compacted = {
            "indices": rows,
            "source": core.get("source"),
        }
        errors = _compact_rows(core.get("errors"), fields=("symbol", "error"), limit=5, text_fields=("error",), max_text=140)
        if errors:
            compacted["errors"] = errors
    elif token == "fetch_reddit_sentiment":
        compacted = _compact_rows(
            core,
            fields=("title", "subreddit", "score", "num_comments", "created", "selftext_snippet"),
            limit=6,
            text_fields=("title", "selftext_snippet"),
            max_text=140,
        )
    elif token == "fetch_sec_filings":
        compacted = _compact_rows(
            core,
            fields=("form_type", "filed_date", "entity", "description"),
            limit=6,
            text_fields=("description",),
            max_text=140,
        )
    elif token == "earnings_calendar" and isinstance(core, dict):
        compacted = {
            "ticker": core.get("ticker"),
            "start_date": core.get("start_date"),
            "days_ahead": core.get("days_ahead"),
            "count": core.get("count"),
            "rows": _compact_rows(
                core.get("rows"),
                fields=("date", "symbol", "name", "time", "eps_forecast"),
                limit=10,
                text_fields=("name",),
                max_text=80,
            ),
        }
        if core.get("error") is not None:
            compacted["error"] = core.get("error")
    elif token == "macro_snapshot" and isinstance(core, dict):
        indicators = core.get("indicators") or {}
        if isinstance(indicators, dict):
            compact_indicators: dict[str, Any] = {}
            for key, item in list(indicators.items())[:12]:
                if isinstance(item, dict):
                    compact_indicators[str(key)] = {
                        "value": item.get("value"),
                        "date": item.get("date"),
                        "unit": item.get("unit"),
                    }
            compacted = {
                "as_of": core.get("as_of"),
                "indicators": compact_indicators,
                "source": core.get("source"),
            }
            if core.get("error") is not None:
                compacted["error"] = core.get("error")
    elif token == "get_research_briefing":
        compacted = _compact_rows(
            core,
            fields=("created_at", "ticker", "category", "headline", "summary"),
            limit=8,
            text_fields=("headline", "summary"),
            max_text=220,
        )
    elif token in {"search_past_experiences", "search_peer_lessons"}:
        compacted = _compact_rows(
            core,
            fields=("event_id", "created_date", "summary", "score", "author_id", "agent_id", "memory_source"),
            limit=5,
            text_fields=("summary",),
            max_text=180,
        )
    elif token == "portfolio_diagnosis" and isinstance(core, dict):
        def _benchmark_for_prompt(raw: object) -> dict[str, Any]:
            benchmark = dict(raw or {})
            if benchmark.get("excess_return_vs_benchmark") is None and benchmark.get("alpha_vs_benchmark") is not None:
                benchmark["excess_return_vs_benchmark"] = benchmark.get("alpha_vs_benchmark")
            benchmark.pop("alpha_vs_benchmark", None)
            return benchmark

        compacted = {
            "risk_contribution": _compact_rows(core.get("risk_contribution"), fields=("ticker", "rc"), limit=5),
            "concentration_top3": core.get("concentration_top3"),
            "hhi": core.get("hhi"),
            "momentum_20d_weighted": core.get("momentum_20d_weighted"),
            "momentum_5d_weighted": core.get("momentum_5d_weighted"),
            "volatility_20d_weighted": core.get("volatility_20d_weighted"),
        }
        if core.get("mdd") is not None:
            compacted["mdd"] = core.get("mdd")
        if isinstance(core.get("benchmarks"), dict):
            compacted["benchmarks"] = {
                str(scope): _benchmark_for_prompt(benchmark)
                for scope, benchmark in dict(core.get("benchmarks") or {}).items()
                if isinstance(benchmark, dict)
            }
        if core.get("benchmark") is not None:
            compacted["benchmark"] = _benchmark_for_prompt(core.get("benchmark"))
        allocation = core.get("hrp_allocation")
        legacy_reference = False
        if not isinstance(allocation, dict):
            allocation = core.get("hrp_reference")
            legacy_reference = isinstance(allocation, dict)
        if not isinstance(allocation, dict):
            allocation = core.get("rebalance_plan")
            legacy_reference = isinstance(allocation, dict)
        if isinstance(allocation, dict):
            raw_weights = allocation.get("hrp_weights")
            if not isinstance(raw_weights, list):
                raw_weights = allocation.get("target_weights")
            compact_weights = _compact_rows(
                raw_weights,
                fields=(
                    "ticker",
                    "current_weight",
                    "hrp_weight",
                    "target_weight",
                    "delta_weight",
                    "relative_to_current",
                    "direction",
                ),
                limit=6,
            )
            for row in compact_weights:
                if row.get("hrp_weight") is None and row.get("target_weight") is not None:
                    row["hrp_weight"] = row.pop("target_weight")
                else:
                    row.pop("target_weight", None)
                if row.get("relative_to_current") is None and row.get("direction") is not None:
                    direction = str(row.pop("direction") or "").strip().lower()
                    row["relative_to_current"] = {
                        "increase": "higher",
                        "decrease": "lower",
                        "maintain": "similar",
                    }.get(direction, direction)
                else:
                    row.pop("direction", None)

            compact_allocation: dict[str, Any] = {
                "status": allocation.get("status"),
                "strategy": allocation.get("strategy"),
                "hrp_cash_weight": allocation.get("hrp_cash_weight", allocation.get("target_cash_weight")),
                "hrp_concentration_top3": allocation.get(
                    "hrp_concentration_top3",
                    allocation.get("target_concentration_top3"),
                ),
                "hrp_hhi": allocation.get("hrp_hhi", allocation.get("target_hhi")),
                "hrp_weights": compact_weights,
            }
            deltas = _compact_rows(
                allocation.get("weight_deltas") or allocation.get("reference_adjustments"),
                fields=(
                    "ticker",
                    "relative_to_current",
                    "direction",
                    "delta_weight",
                    "current_weight",
                    "hrp_weight",
                    "target_weight",
                ),
                limit=8,
            )
            if not deltas and legacy_reference:
                legacy_orders = _compact_rows(
                    allocation.get("rebalance_orders"),
                    fields=("ticker", "side", "current_weight", "target_weight"),
                    limit=8,
                )
                for order in legacy_orders:
                    side = str(order.pop("side", "") or "").strip().upper()
                    relative = "higher" if side == "BUY" else ("lower" if side == "SELL" else "")
                    if relative:
                        order["relative_to_current"] = relative
                    deltas.append(order)
            for row in deltas:
                if row.get("hrp_weight") is None and row.get("target_weight") is not None:
                    row["hrp_weight"] = row.pop("target_weight")
                else:
                    row.pop("target_weight", None)
                if row.get("relative_to_current") is None and row.get("direction") is not None:
                    direction = str(row.pop("direction") or "").strip().lower()
                    row["relative_to_current"] = {
                        "increase": "higher",
                        "decrease": "lower",
                        "maintain": "similar",
                    }.get(direction, direction)
                else:
                    row.pop("direction", None)
            compact_allocation["weight_deltas"] = deltas
            if allocation.get("projected_mdd") is not None:
                compact_allocation["projected_mdd"] = allocation.get("projected_mdd")
            if allocation.get("reason") is not None:
                compact_allocation["reason"] = allocation.get("reason")
            if allocation.get("notes"):
                compact_allocation["notes"] = list(allocation.get("notes") or [])[:3]
            small_deltas = allocation.get("small_deltas") or allocation.get("skipped_adjustments")
            if small_deltas:
                compact_allocation["small_deltas"] = _compact_rows(
                    small_deltas,
                    fields=("ticker", "reason", "delta_weight"),
                    limit=6,
                )
            compacted["hrp_allocation"] = compact_allocation
        if core.get("error") is not None:
            compacted["error"] = core.get("error")
    elif token == "trade_performance" and isinstance(core, dict):
        rt = core.get("round_trips") or {}
        compacted = {
            "period": core.get("period"),
            "total_trades": core.get("total_trades"),
            "round_trips": {
                k: rt[k] for k in ("closed", "win_rate", "avg_return_pct", "avg_holding_days", "best", "worst")
                if k in rt
            },
        }
        bh = core.get("behavioral")
        if bh:
            compacted["behavioral"] = bh
        streak = core.get("recent_streak") or {}
        if streak.get("last_5"):
            compacted["recent_streak"] = streak
        unrealized = list(core.get("unrealized") or [])[:5]
        if unrealized:
            compacted["unrealized"] = unrealized
    elif token == "optimize_portfolio" and isinstance(core, dict):
        compacted = {
            "strategy": core.get("strategy"),
            "expected_return_daily": core.get("expected_return_daily"),
            "volatility_daily": core.get("volatility_daily"),
            "sharpe_daily": core.get("sharpe_daily"),
            "allocations": [],
        }
        weights = core.get("weights") or {}
        if isinstance(weights, dict):
            ordered = sorted(
                (
                    {"ticker": str(t), "target_weight": v}
                    for t, v in weights.items()
                    if str(t).strip()
                ),
                key=lambda item: float(item.get("target_weight") or 0.0),
                reverse=True,
            )
            compacted["allocations"] = ordered[:12]
        orders = _compact_rows(
            core.get("rebalance_orders"),
            fields=("ticker", "side", "size_ratio", "current_weight", "target_weight"),
            limit=12,
        )
        if orders:
            compacted["rebalance_orders"] = orders
        if core.get("backtest_mdd") is not None:
            compacted["backtest_mdd"] = core.get("backtest_mdd")
        if core.get("error") is not None:
            compacted["error"] = core.get("error")

    if memory_ctx:
        compacted_memory = _compact_memory_context_rows(memory_ctx)
        if compacted_memory:
            if isinstance(compacted, dict):
                compacted["_memory_context"] = compacted_memory
            else:
                compacted = {"data": compacted, "_memory_context": compacted_memory}

    return compacted
