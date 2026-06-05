from __future__ import annotations

from typing import Any

from arena.agents.adk_memory_context import model_memory_context_rows


_FORECAST_MODEL_DIRECTION_ALIASES: dict[str, str] = {
    "STRONG_BUY": "MODEL_UP_STRONG",
    "BUY": "MODEL_UP",
    "NEUTRAL": "MODEL_MIXED",
    "SELL": "MODEL_DOWN",
    "STRONG_SELL": "MODEL_DOWN_STRONG",
}


def _normalize_model_direction(value: Any) -> str | None:
    label = str(value or "").strip().upper()
    if not label:
        return None
    return _FORECAST_MODEL_DIRECTION_ALIASES.get(label, label)


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


def _count_items(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, str) and value.strip():
        return 1
    return None


def _requested_count(tool_args: dict[str, Any], core: Any) -> int | None:
    if isinstance(tool_args, dict):
        for key in ("tickers", "indices", "requested"):
            count = _count_items(tool_args.get(key))
            if count is not None:
                return count
        count = _count_items(tool_args.get("ticker"))
        if count is not None:
            return count
    if isinstance(core, dict):
        for key in ("tickers", "indices", "requested"):
            count = _count_items(core.get(key))
            if count is not None:
                return count
        count = _count_items(core.get("ticker"))
        if count is not None:
            return count
    return None


def _compaction_meta(
    *,
    requested_count: int | None,
    returned_count: int,
    visible_count: int,
    visible_limit: int,
) -> dict[str, Any]:
    return {
        "requested_count": requested_count if requested_count is not None else returned_count,
        "returned_count": returned_count,
        "visible_count": visible_count,
        "visible_limit": visible_limit,
        "truncated": returned_count > visible_count,
    }


def _maybe_add_truncation_meta(
    payload: dict[str, Any],
    *,
    requested_count: int | None,
    returned_count: int,
    visible_count: int,
    visible_limit: int,
) -> None:
    if returned_count > visible_count:
        payload["compaction"] = _compaction_meta(
            requested_count=requested_count,
            returned_count=returned_count,
            visible_count=visible_count,
            visible_limit=visible_limit,
        )


def _drop_derived_count(payload: dict[str, Any], *, row_key: str = "rows", count_key: str = "count") -> None:
    rows = payload.get(row_key)
    if isinstance(rows, list) and payload.get(count_key) == len(rows):
        payload.pop(count_key, None)


def _drop_if_row_field_mirror(
    payload: dict[str, Any],
    *,
    list_key: str,
    row_key: str,
    row_field: str,
) -> None:
    rows = payload.get(row_key)
    mirror = payload.get(list_key)
    if not isinstance(rows, list) or not isinstance(mirror, list) or not rows:
        return
    values: list[Any] = []
    for row in rows:
        if not isinstance(row, dict) or row.get(row_field) is None:
            return
        values.append(row.get(row_field))
    if values == mirror:
        payload.pop(list_key, None)


def _compact_memory_context_rows(rows: Any) -> list[dict[str, Any]]:
    return model_memory_context_rows(rows, limit=3)


_MACRO_COMPACT_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("us_policy", ("fed_funds_rate", "sofr")),
    ("us_curve", ("treasury_10y", "treasury_3m", "yield_spread_10y_3m")),
    ("us_inflation", ("cpi_yoy", "core_cpi_yoy", "core_pce_yoy")),
    ("us_growth", ("real_gdp", "industrial_production_yoy")),
    ("us_credit", ("high_yield_oas", "credit_spread_hy_corp", "financial_stress_index")),
    ("us_market", ("sp500", "vix", "wti_crude")),
    ("us_housing", ("mortgage_30y", "case_shiller_home_price_yoy")),
    ("kr_policy", ("bok_base_rate", "kr_treasury_5y", "kr_yield_spread_5y_3y")),
    ("kr_money_credit", ("kr_m2_money_supply", "kr_household_credit", "kr_bank_loan_deposit_spread")),
    ("kr_fx_external", ("usd_krw", "jpy_krw", "kr_current_account", "kr_fx_reserves")),
    ("kr_growth_cycle", ("kr_gdp_growth", "kr_all_industry_production", "kr_leading_cyclical_component")),
    ("kr_inflation", ("kr_cpi", "kr_core_cpi_ex_food_energy", "kr_ppi")),
    ("kr_sentiment", ("kr_consumer_sentiment_index", "kr_economic_sentiment_index")),
    ("kr_housing_commodities", ("kr_house_price_index", "kr_jeonse_price_index", "dubai_oil", "gold_spot")),
)


def _compact_macro_coverage(coverage: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(coverage, dict):
        return out
    for source, item in coverage.items():
        if not isinstance(item, dict):
            continue
        returned = item.get("returned")
        requested = item.get("requested")
        if returned is None or requested is None:
            continue
        out[str(source)] = f"{returned}/{requested}"
    return out


def _compact_macro_indicator(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or item.get("value") is None:
        return None
    out: dict[str, Any] = {"v": item.get("value")}
    if item.get("date") not in {None, ""}:
        out["d"] = item.get("date")
    if item.get("unit") not in {None, ""}:
        out["u"] = item.get("unit")
    if item.get("source") not in {None, ""}:
        out["src"] = item.get("source")
    identifier = item.get("series_id") or item.get("stat_code") or item.get("class_name")
    if identifier not in {None, ""}:
        out["id"] = identifier
    return out


def _compact_macro_evidence(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict) or item.get("k") is None:
        return None
    fields = ("k", "v", "d", "u", "src", "id", "item", "freq", "chg_1m", "chg_3m", "yoy", "z", "pct", "trend", "lag_days")
    out = {field: item.get(field) for field in fields if item.get(field) is not None}
    series = item.get("series")
    if isinstance(series, list) and series:
        out["series"] = [
            {"d": point.get("d"), "v": point.get("v")}
            for point in series[:12]
            if isinstance(point, dict) and point.get("d") is not None and point.get("v") is not None
        ]
    return out or None


def _compact_macro_groups(groups: Any) -> dict[str, Any]:
    if not isinstance(groups, dict):
        return {}
    out: dict[str, Any] = {}
    for group, payload in list(groups.items())[:10]:
        if not isinstance(payload, dict):
            continue
        evidence = [
            compacted
            for compacted in (_compact_macro_evidence(item) for item in (payload.get("evidence") or [])[:8])
            if compacted
        ]
        item: dict[str, Any] = {}
        if payload.get("state") is not None:
            item["state"] = payload.get("state")
        if evidence:
            item["evidence"] = evidence
        if item:
            out[str(group)] = item
    return out


def _compact_macro_snapshot(core: dict[str, Any]) -> dict[str, Any]:
    indicators = core.get("indicators") or {}
    compacted: dict[str, Any] = {
        "as_of": core.get("as_of"),
        "source": core.get("source"),
    }
    coverage = _compact_macro_coverage(core.get("coverage"))
    if coverage:
        compacted["coverage"] = coverage
    if isinstance(core.get("regime_card"), dict):
        for key in ("depth", "data_mode", "focus"):
            if core.get(key) is not None:
                compacted[key] = core.get(key)
        compacted["regime_card"] = core.get("regime_card")
        if isinstance(core.get("market_implications"), dict):
            compacted["market_implications"] = core.get("market_implications")
        groups = _compact_macro_groups(core.get("groups"))
        if groups:
            compacted["groups"] = groups
        movers = _compact_rows(
            core.get("notable_movers"),
            fields=("k", "why"),
            limit=5,
            text_fields=("why",),
            max_text=120,
        )
        if movers:
            compacted["notable_movers"] = movers
        if isinstance(core.get("omitted"), dict):
            compacted["omitted"] = core.get("omitted")
        if core.get("error") is not None:
            compacted["error"] = core.get("error")
        return compacted
    if not isinstance(indicators, dict):
        if core.get("error") is not None:
            compacted["error"] = core.get("error")
        return compacted

    selected_keys: set[str] = set()
    key_indicators: dict[str, dict[str, Any]] = {}
    for bucket, keys in _MACRO_COMPACT_BUCKETS:
        bucket_payload: dict[str, Any] = {}
        for key in keys:
            item = _compact_macro_indicator(indicators.get(key))
            if item is None:
                continue
            bucket_payload[key] = item
            selected_keys.add(key)
        if bucket_payload:
            key_indicators[bucket] = bucket_payload
    if key_indicators:
        compacted["key_indicators"] = key_indicators

    raw_count = len(indicators)
    visible_count = len(selected_keys)
    if raw_count > visible_count:
        compacted["compaction"] = {
            "raw_indicator_count": raw_count,
            "visible_indicator_count": visible_count,
            "omitted_indicator_count": raw_count - visible_count,
        }
    if core.get("error") is not None:
        compacted["error"] = core.get("error")
    return compacted


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

    if token == "recommend_opportunities" and isinstance(core, dict):
        rows = core.get("recommendations") or core.get("rows") or []
        compacted = {
            "status": core.get("status"),
            "recommendations": _compact_rows(
                rows,
                fields=(
                    "ticker",
                    "profile",
                    "tactical_kind",
                    "bucket",
                    "recommendation_rank",
                    "recommendation_score",
                    "score_source",
                    "ranker_version",
                    "score_components",
                    "predicted_excess_return_20d",
                    "prob_outperform_20d",
                    "predicted_drawdown_20d",
                    "confidence",
                    "model_confidence",
                    "action",
                    "reason_for",
                    "reason_risk",
                    "ret_20d",
                    "ret_5d",
                    "volatility_20d",
                    "optimizer_weight",
                    "optimizer_raw_weight",
                    "evidence_level",
                    "signal_contributions",
                ),
                limit=12,
                text_fields=("reason_for", "reason_risk"),
                max_text=160,
            ),
        }
        for row in compacted.get("recommendations") or []:
            if not isinstance(row, dict):
                continue
            if row.get("confidence") is not None and row.get("confidence") == row.get("model_confidence"):
                row.pop("confidence", None)
        optimizer = core.get("optimizer")
        if isinstance(optimizer, dict):
            optimizer_payload = {
                key: optimizer.get(key)
                for key in ("status", "strategy", "strategy_requested", "weights", "raw_weights", "tactical_caps", "forecast_coverage", "degraded_reasons")
                if optimizer.get(key) is not None
            }
            if optimizer_payload:
                compacted["optimizer"] = optimizer_payload
        diagnostics = core.get("diagnostics")
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("score_policy"), dict):
            policy = diagnostics.get("score_policy") or {}
            compacted["score_policy"] = {
                key: policy.get(key)
                for key in ("version", "score_formula")
                if policy.get(key) is not None
            }
        if isinstance(diagnostics, dict) and isinstance(diagnostics.get("selection_scope"), dict):
            scope = diagnostics.get("selection_scope") or {}
            compacted["selection_scope"] = {
                key: scope.get(key)
                for key in (
                    "mode",
                    "requested_max_candidates",
                    "global_limit",
                    "per_profile_limit",
                    "loaded_rows",
                    "requested_buckets",
                    "requested_profiles",
                    "effective_buckets",
                    "effective_profiles",
                    "legacy_profile_bucket_tokens",
                    "loaded_rows_before_filter_fallback",
                )
                if scope.get(key) is not None
            }
        if isinstance(diagnostics, dict) and diagnostics.get("warnings"):
            compacted["warnings"] = list(diagnostics.get("warnings") or [])[:5]
    elif token == "screen_market":
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
                "model_direction",
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
                if item.get("model_direction") is None:
                    model_direction = _normalize_model_direction(src.get("model_direction") or src.get("consensus"))
                    if model_direction is not None:
                        item["model_direction"] = model_direction
                elif isinstance(item.get("model_direction"), str):
                    model_direction = _normalize_model_direction(item.get("model_direction"))
                    if model_direction is not None:
                        item["model_direction"] = model_direction
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
            raw_rows = list(core.get("rows") or [])
            visible_limit = 10
            rows: list[dict[str, Any]] = []
            for row in raw_rows[:visible_limit]:
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
                "tickers": list(core.get("tickers") or [])[:visible_limit],
                "count": len(rows),
                "rows": rows,
            }
            excluded = [str(t).strip().upper() for t in list(core.get("excluded_from_market_scope") or []) if str(t).strip()]
            if excluded:
                compacted["excluded_from_market_scope"] = excluded[:10]
            _maybe_add_truncation_meta(
                compacted,
                requested_count=_requested_count(tool_args, core),
                returned_count=len(raw_rows),
                visible_count=len(rows),
                visible_limit=visible_limit,
            )
            if "compaction" not in compacted:
                _drop_derived_count(compacted)
                _drop_if_row_field_mirror(compacted, list_key="tickers", row_key="rows", row_field="ticker")
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
        if compacted.get("eligible_count") == len(rows):
            compacted.pop("eligible_count", None)
        visible_excluded = compacted.get("excluded") if isinstance(compacted.get("excluded"), list) else []
        if compacted.get("excluded_count") == len(visible_excluded):
            compacted.pop("excluded_count", None)
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
        visible_limit = 6
        if isinstance(core, dict):
            raw_rows = list(core.get("rows") or [])
            rows = _compact_rows(
                raw_rows,
                fields=("ticker", "title", "subreddit", "score", "num_comments", "created", "selftext_snippet"),
                limit=visible_limit,
                text_fields=("title", "selftext_snippet"),
                max_text=140,
            )
            compacted = {
                "tickers": list(core.get("tickers") or []),
                "count": core.get("count", len(raw_rows)),
                "rows": rows,
            }
            _maybe_add_truncation_meta(
                compacted,
                requested_count=_requested_count(tool_args, core),
                returned_count=len(raw_rows),
                visible_count=len(rows),
                visible_limit=visible_limit,
            )
            if "compaction" not in compacted:
                _drop_derived_count(compacted)
                _drop_if_row_field_mirror(compacted, list_key="tickers", row_key="rows", row_field="ticker")
            errors = _compact_rows(core.get("errors"), fields=("ticker", "error"), limit=5, text_fields=("error",), max_text=140)
            if errors:
                compacted["errors"] = errors
        else:
            compacted = _compact_rows(
                core,
                fields=("title", "subreddit", "score", "num_comments", "created", "selftext_snippet"),
                limit=visible_limit,
                text_fields=("title", "selftext_snippet"),
                max_text=140,
            )
    elif token == "fetch_sec_filings":
        visible_limit = 6
        if isinstance(core, dict):
            raw_rows = list(core.get("rows") or [])
            rows = _compact_rows(
                raw_rows,
                fields=("ticker", "form_type", "filed_date", "entity", "description"),
                limit=visible_limit,
                text_fields=("description",),
                max_text=140,
            )
            compacted = {
                "tickers": list(core.get("tickers") or []),
                "filing_type": core.get("filing_type"),
                "count": core.get("count", len(raw_rows)),
                "rows": rows,
            }
            _maybe_add_truncation_meta(
                compacted,
                requested_count=_requested_count(tool_args, core),
                returned_count=len(raw_rows),
                visible_count=len(rows),
                visible_limit=visible_limit,
            )
            if "compaction" not in compacted:
                _drop_derived_count(compacted)
                _drop_if_row_field_mirror(compacted, list_key="tickers", row_key="rows", row_field="ticker")
            errors = _compact_rows(core.get("errors"), fields=("ticker", "error"), limit=5, text_fields=("error",), max_text=140)
            if errors:
                compacted["errors"] = errors
        else:
            compacted = _compact_rows(
                core,
                fields=("form_type", "filed_date", "entity", "description"),
                limit=visible_limit,
                text_fields=("description",),
                max_text=140,
            )
    elif token == "earnings_calendar" and isinstance(core, dict):
        raw_rows = list(core.get("rows") or [])
        visible_limit = 10
        rows = _compact_rows(
            raw_rows,
            fields=("date", "symbol", "name", "time", "eps_forecast"),
            limit=visible_limit,
            text_fields=("name",),
            max_text=80,
        )
        compacted = {
            "ticker": core.get("ticker"),
            "start_date": core.get("start_date"),
            "days_ahead": core.get("days_ahead"),
            "count": core.get("count"),
            "rows": rows,
        }
        if core.get("tickers") is not None:
            compacted["tickers"] = core.get("tickers")
        _maybe_add_truncation_meta(
            compacted,
            requested_count=_requested_count(tool_args, core),
            returned_count=len(raw_rows),
            visible_count=len(rows),
            visible_limit=visible_limit,
        )
        if "compaction" not in compacted:
            _drop_derived_count(compacted)
            _drop_if_row_field_mirror(compacted, list_key="tickers", row_key="rows", row_field="symbol")
        if core.get("error") is not None:
            compacted["error"] = core.get("error")
    elif token == "macro_snapshot" and isinstance(core, dict):
        compacted = _compact_macro_snapshot(core)
    elif token == "get_research_briefing":
        compacted = _compact_rows(
            core,
            fields=(
                "briefing_id",
                "created_at",
                "published_at",
                "source",
                "feed_id",
                "category",
                "market",
                "ticker",
                "headline",
                "summary",
                "detail_json",
                "sources",
                "publisher",
                "title",
                "source_url",
                "snippet",
                "source_doc_id",
                "content_text",
                "content_offset",
                "next_offset",
                "content_hash",
                "text_char_count",
                "fetch_error",
            ),
            limit=8,
        )
    elif token == "get_macro_research_briefing":
        compacted = _compact_rows(
            core,
            fields=(
                "published_at",
                "source",
                "feed_id",
                "doc_type",
                "market",
                "title",
                "headline",
                "summary",
                "market_implication",
                "source_url",
                "themes",
                "source_doc_id",
                "content_text",
                "content_offset",
                "next_offset",
                "content_hash",
                "text_char_count",
                "fetch_error",
            ),
            limit=8,
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
        joint_policy = core.get("joint_policy")
        if isinstance(joint_policy, dict):
            compacted["joint_policy"] = {
                "status": joint_policy.get("status"),
                "score_source": joint_policy.get("score_source"),
                "ranker_version": joint_policy.get("ranker_version"),
                "coverage": joint_policy.get("coverage"),
                "weighted_score": joint_policy.get("weighted_score"),
                "holdings": _compact_rows(
                    joint_policy.get("holdings"),
                    fields=(
                        "ticker",
                        "weight",
                        "score",
                        "rank",
                        "profile",
                        "bucket",
                        "action",
                        "confidence",
                        "top_contributions",
                    ),
                    limit=5,
                ),
            }
            if joint_policy.get("missing_tickers"):
                compacted["joint_policy"]["missing_tickers"] = list(joint_policy.get("missing_tickers") or [])[:5]
        if core.get("mdd") is not None:
            compacted["mdd"] = core.get("mdd")
        if isinstance(core.get("benchmarks"), dict):
            compacted["benchmarks"] = {
                str(scope): _benchmark_for_prompt(benchmark)
                for scope, benchmark in dict(core.get("benchmarks") or {}).items()
                if isinstance(benchmark, dict)
            }
        if core.get("benchmark") is not None:
            primary_benchmark = _benchmark_for_prompt(core.get("benchmark"))
            matched_scope = None
            benchmarks = compacted.get("benchmarks")
            if isinstance(benchmarks, dict):
                for scope, benchmark in benchmarks.items():
                    if benchmark == primary_benchmark:
                        matched_scope = str(scope)
                        break
            if matched_scope:
                compacted["primary_benchmark_scope"] = matched_scope
            else:
                compacted["benchmark"] = primary_benchmark
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
            fields=("ticker", "side", "target_weight", "sell_ratio", "current_weight"),
            limit=12,
        )
        if orders:
            compacted["rebalance_orders"] = orders
        if core.get("backtest_mdd") is not None:
            compacted["backtest_mdd"] = core.get("backtest_mdd")
        if core.get("error") is not None:
            compacted["error"] = core.get("error")
    elif token == "validate_order_draft" and isinstance(core, dict):
        risk = core.get("risk") if isinstance(core.get("risk"), dict) else {}
        intent = core.get("intent") if isinstance(core.get("intent"), dict) else {}
        compacted = {
            "status": core.get("status"),
            "tenant_id": core.get("tenant_id"),
            "scope": core.get("scope"),
            "target_agent_id": core.get("target_agent_id"),
            "judgment_source": core.get("judgment_source"),
            "intent": {
                key: intent.get(key)
                for key in ("ticker", "side", "quantity", "price_krw", "price_native", "quote_currency", "fx_rate", "exchange_code", "instrument_id", "rationale")
                if intent.get(key) is not None
            },
            "risk": {
                key: risk.get(key)
                for key in ("allowed", "reason", "policy_hits")
                if risk.get(key) is not None
            },
            "notional_krw": core.get("notional_krw"),
            "submission_status": core.get("submission_status"),
            "approval_required": core.get("approval_required"),
            "approval_ui": "approval_card" if core.get("approval_required") else None,
        }
        compacted = {key: value for key, value in compacted.items() if value is not None}
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
