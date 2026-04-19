"""Market store — market features, instrument master, universe candidates, forecasts."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import uuid
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from arena.models import utc_now

if TYPE_CHECKING:
    from arena.data.bigquery.session import BigQuerySession

logger = logging.getLogger(__name__)

_OPEN_TRADING_DAILY_SOURCES: set[str] = {
    "open_trading_nasdaq",
    "open_trading_nyse",
    "open_trading_amex",
    "open_trading_us",
    "open_trading_kospi",
}
_DAILY_FILTER_SQL = (
    "(IFNULL(source, '') NOT IN ('open_trading_nasdaq', 'open_trading_nyse', 'open_trading_amex', 'open_trading_us', 'open_trading_kospi') "
    "OR as_of_ts = TIMESTAMP_TRUNC(as_of_ts, DAY))"
)
_DEFAULT_FORECAST_TABLE = "predicted_expected_returns"
_FORECAST_MODEL_COLUMNS: tuple[str, ...] = (
    "forecast_model",
    "forecast_method",
    "model_name",
    "model_id",
    "source_model",
    "ensemble_name",
    "method",
    "model",
)
_FORECAST_STACK_FLAG_COLUMNS: tuple[str, ...] = ("is_stacked", "is_meta_model", "stacked")
_FORECAST_SCORE_COLUMNS: tuple[str, ...] = ("forecast_score", "meta_score", "validation_score", "cv_score")
_FORECAST_MODE_ALIASES: dict[str, tuple[str, ...]] = {
    "all": ("all", "both", "full", "raw", "base+stacked", "stacked+base", "balanced"),
    "stacked": ("stacked", "stack", "meta", "lgbm_stack", "ridge_stack", "ensemble_stack"),
    "lgbm": ("lgbm", "lightgbm", "stacked_lgbm", "lgbm_stack", "meta_lgbm", "stacked_lightgbm"),
    "ridge": ("ridge", "stacked_ridge", "ridge_stack", "meta_ridge"),
    "avg": ("avg", "average", "simple_average", "equal_weight", "ensemble_avg"),
    "base": ("base", "base_model", "base_models"),
}


def _finite_float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


class MarketStore:
    """Market feature / price history repository operations."""

    def __init__(self, session: BigQuerySession) -> None:
        self.session = session

    def _append_json_rows_via_load_job(self, table_id: str, rows: list[dict[str, Any]]) -> int:
        """Appends rows via a BigQuery load job to avoid streaming-buffer DML conflicts."""
        if not rows:
            return 0

        from google.cloud.bigquery import LoadJobConfig, SourceFormat, WriteDisposition

        table = self.session.client.get_table(table_id)
        job_config = LoadJobConfig(
            source_format=SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=WriteDisposition.WRITE_APPEND,
            autodetect=False,
            schema=table.schema,
        )
        ndjson = "\n".join(json.dumps(r) for r in rows).encode("utf-8")
        load_job = self.session.client.load_table_from_file(
            io.BytesIO(ndjson),
            table_id,
            job_config=job_config,
        )
        load_job.result()
        return len(rows)

    def latest_close_prices(
        self,
        *,
        tickers: list[str],
        sources: list[str] | None = None,
    ) -> dict[str, float]:
        """Returns latest close_price_krw per ticker (best-effort)."""
        tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        filters: list[str] = [
            "ticker IN UNNEST(@tickers)",
            "close_price_krw IS NOT NULL",
        ]
        params: dict[str, Any] = {"tickers": tokens}
        if sources:
            filters.append("source IN UNNEST(@sources)")
            params["sources"] = sources

        where = " AND ".join(filters)
        sql = f"""
        WITH latest AS (
          SELECT ticker, close_price_krw, as_of_ts, updated_at,
                 ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY as_of_ts DESC, updated_at DESC) AS rn
          FROM `{self.session.dataset_fqn}.market_features_latest`
          WHERE {where}
        )
        SELECT ticker, close_price_krw
        FROM latest
        WHERE rn = 1
        """
        rows = self.session.fetch_rows(sql, params)

        out: dict[str, float] = {}
        for r in rows:
            t = str(r.get("ticker", "")).strip().upper()
            if not t:
                continue
            try:
                px = float(r.get("close_price_krw") or 0.0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                out[t] = px
        return out

    def latest_close_prices_with_currency(
        self,
        *,
        tickers: list[str],
        sources: list[str] | None = None,
        as_of_date: date | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Returns latest close prices with currency info per ticker.

        When *as_of_date* is given, only prices on or before that date are
        considered — useful for historical NAV recomputation.

        Each value is ``{"close_price_krw": float, "close_price_native": float|None,
        "quote_currency": str, "fx_rate_used": float}``.
        """
        tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        filters: list[str] = [
            "ticker IN UNNEST(@tickers)",
            "close_price_krw IS NOT NULL",
        ]
        params: dict[str, Any] = {"tickers": tokens}
        if sources:
            filters.append("source IN UNNEST(@sources)")
            params["sources"] = sources
        if as_of_date is not None:
            filters.append("DATE(as_of_ts) <= @as_of_date")
            params["as_of_date"] = as_of_date

        where = " AND ".join(filters)
        sql = f"""
        WITH latest AS (
          SELECT ticker, close_price_krw, close_price_native, quote_currency, fx_rate_used,
                 as_of_ts, updated_at,
                 ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY as_of_ts DESC, updated_at DESC) AS rn
          FROM `{self.session.dataset_fqn}.market_features_latest`
          WHERE {where}
        )
        SELECT ticker, close_price_krw, close_price_native, quote_currency, fx_rate_used
        FROM latest
        WHERE rn = 1
        """
        rows = self.session.fetch_rows(sql, params)

        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            t = str(r.get("ticker", "")).strip().upper()
            if not t:
                continue
            try:
                px = float(r.get("close_price_krw") or 0.0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                try:
                    native = float(r.get("close_price_native")) if r.get("close_price_native") is not None else None
                except (TypeError, ValueError):
                    native = None
                out[t] = {
                    "close_price_krw": px,
                    "close_price_native": native,
                    "quote_currency": str(r.get("quote_currency") or ""),
                    "fx_rate_used": float(r.get("fx_rate_used") or 0.0),
                }
        return out

    def latest_market_features(
        self,
        tickers: list[str],
        limit: int,
        sources: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Loads latest market feature rows, optionally filtered by tickers and sources."""
        lim = max(1, min(int(limit), 8_000))
        filters: list[str] = []
        params: dict[str, Any] = {"limit": lim}

        if tickers:
            tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tokens = list(dict.fromkeys(tokens))
            if tokens:
                filters.append("ticker IN UNNEST(@tickers)")
                params["tickers"] = tokens
            else:
                return []

        if sources:
            filters.append("source IN UNNEST(@sources)")
            params["sources"] = sources

        where = "WHERE " + " AND ".join(filters) if filters else ""
        sql = f"""
        WITH latest AS (
          SELECT as_of_ts, ticker, exchange_code, instrument_id, close_price_krw, close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d, volatility_20d, sentiment_score, source, updated_at,
                 ROW_NUMBER() OVER (
                   PARTITION BY ticker
                   ORDER BY
                     CASE
                       WHEN ret_5d IS NOT NULL AND ret_20d IS NOT NULL AND volatility_20d IS NOT NULL THEN 0
                       ELSE 1
                     END,
                     as_of_ts DESC,
                     updated_at DESC
                 ) AS rn
          FROM `{self.session.dataset_fqn}.market_features_latest`
          {where}
        )
        SELECT as_of_ts, ticker, exchange_code, instrument_id, close_price_krw, close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d, volatility_20d, sentiment_score, source
        FROM latest
        WHERE rn = 1
        ORDER BY as_of_ts DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    def latest_missing_daily_feature_tickers(
        self,
        *,
        sources: list[str] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Returns latest tickers whose newest snapshot lacks daily-history features."""
        lim = max(1, min(int(limit), 10_000))
        filters: list[str] = []
        params: dict[str, Any] = {"limit": lim}
        if sources:
            filters.append("source IN UNNEST(@sources)")
            params["sources"] = sources

        where = "WHERE " + " AND ".join(filters) if filters else ""
        sql = f"""
        WITH ranked AS (
          SELECT as_of_ts, ticker, exchange_code, instrument_id, ret_5d, ret_20d, volatility_20d, source, updated_at,
                 ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY as_of_ts DESC, updated_at DESC) AS rn,
                 MAX(
                   CASE
                     WHEN ret_5d IS NOT NULL AND ret_20d IS NOT NULL AND volatility_20d IS NOT NULL THEN 1
                     ELSE 0
                   END
                 ) OVER (PARTITION BY ticker) AS has_complete_features
          FROM `{self.session.dataset_fqn}.market_features_latest`
          {where}
        )
        SELECT as_of_ts, ticker, exchange_code, instrument_id, source
        FROM ranked
        WHERE rn = 1
          AND (ret_5d IS NULL OR ret_20d IS NULL OR volatility_20d IS NULL)
          AND has_complete_features = 0
        ORDER BY as_of_ts DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    def latest_fundamentals_snapshot(
        self,
        *,
        tickers: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Loads the latest fundamental snapshot row per ticker."""
        lim = max(1, min(int(limit), 8_000))
        filters: list[str] = []
        params: dict[str, Any] = {"limit": lim}

        if tickers:
            tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tokens = list(dict.fromkeys(tokens))
            if tokens:
                filters.append("ticker IN UNNEST(@tickers)")
                params["tickers"] = tokens
            else:
                return []

        where = "WHERE " + " AND ".join(filters) if filters else ""
        sql = f"""
        WITH latest AS (
          SELECT as_of_ts, ticker, market, exchange_code, instrument_id, currency, last_native, per, pbr, eps, bps, sps, roe, debt_ratio, reserve_ratio, operating_profit_growth, net_profit_growth, source, updated_at,
                 ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY as_of_ts DESC, updated_at DESC) AS rn
          FROM `{self.session.dataset_fqn}.fundamentals_snapshot_latest`
          {where}
        )
        SELECT as_of_ts, ticker, market, exchange_code, instrument_id, currency, last_native, per, pbr, eps, bps, sps, roe, debt_ratio, reserve_ratio, operating_profit_growth, net_profit_growth, source
        FROM latest
        WHERE rn = 1
        ORDER BY as_of_ts DESC
        LIMIT @limit
        """
        try:
            return self.session.fetch_rows(sql, params)
        except Exception:
            return []

    def latest_feature_dates(self, tickers: list[str], source: str) -> dict[str, date]:
        """Returns max(as_of_date) per ticker for a given source."""
        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not tickers:
            return {}

        sql = f"""
        SELECT ticker, MAX(DATE(as_of_ts)) AS max_d
        FROM `{self.session.dataset_fqn}.market_features`
        WHERE ticker IN UNNEST(@tickers)
          AND source = @source
          AND {_DAILY_FILTER_SQL}
        GROUP BY ticker
        """
        rows = self.session.fetch_rows(sql, {"tickers": tickers, "source": source})
        out: dict[str, date] = {}
        for r in rows:
            t = str(r.get("ticker", "")).strip().upper()
            d = r.get("max_d")
            if t and isinstance(d, date):
                out[t] = d
        return out

    def feature_date_spans(self, tickers: list[str], source: str) -> dict[str, dict[str, Any]]:
        """Returns min/max(as_of_date) and row count per ticker for a given source."""
        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not tickers:
            return {}

        sql = f"""
        SELECT ticker, MIN(DATE(as_of_ts)) AS min_d, MAX(DATE(as_of_ts)) AS max_d, COUNT(*) AS row_count
        FROM `{self.session.dataset_fqn}.market_features`
        WHERE ticker IN UNNEST(@tickers)
          AND source = @source
          AND {_DAILY_FILTER_SQL}
        GROUP BY ticker
        """
        rows = self.session.fetch_rows(sql, {"tickers": tickers, "source": source})
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            t = str(r.get("ticker", "")).strip().upper()
            if not t:
                continue
            row_count = int(r.get("row_count") or 0)
            out[t] = {
                "min_d": r.get("min_d"),
                "max_d": r.get("max_d"),
                "row_count": row_count,
            }
        return out

    def distinct_feature_tickers(
        self,
        *,
        sources: list[str],
        lookback_days: int = 14,
    ) -> list[str]:
        """Returns distinct tickers from market_features for the given sources."""
        if not sources:
            return []
        days = max(1, int(lookback_days))
        sql = f"""
        SELECT DISTINCT ticker
        FROM `{self.session.dataset_fqn}.market_features`
        WHERE source IN UNNEST(@sources)
          AND as_of_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
          AND close_price_krw IS NOT NULL AND close_price_krw > 0
        """
        rows = self.session.fetch_rows(sql, {"sources": sources, "days": days})
        return [str(r["ticker"]).strip().upper() for r in rows if r.get("ticker")]

    def screen_latest_features(
        self,
        *,
        sort_by: str = "ret_20d",
        order: str = "desc",
        tickers: list[str] | None = None,
        min_ret_20d: float | None = None,
        max_volatility: float | None = None,
        top_n: int = 10,
        sources: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Screens the latest row per ticker with optional filters."""
        allowed_fields = {
            "as_of_ts",
            "ticker",
            "close_price_krw",
            "ret_5d",
            "ret_20d",
            "volatility_20d",
            "sentiment_score",
            "source",
        }
        field = sort_by.strip()
        if field not in allowed_fields:
            field = "ret_20d"

        direction = order.strip().lower()
        direction = "asc" if direction == "asc" else "desc"

        filters: list[str] = [
            "ret_5d IS NOT NULL",
            "ret_20d IS NOT NULL",
            "volatility_20d IS NOT NULL",
        ]
        params: dict[str, Any] = {"limit": max(1, min(int(top_n), 500))}

        if tickers is not None:
            tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tokens = list(dict.fromkeys(tokens))
            if not tokens:
                return []
            filters.append("ticker IN UNNEST(@tickers)")
            params["tickers"] = tokens

        if sources:
            filters.append("source IN UNNEST(@sources)")
            params["sources"] = sources

        if min_ret_20d is not None:
            filters.append("ret_20d >= @min_ret_20d")
            params["min_ret_20d"] = float(min_ret_20d)

        if max_volatility is not None:
            filters.append("volatility_20d <= @max_volatility")
            params["max_volatility"] = float(max_volatility)

        where = "WHERE " + " AND ".join(filters) if filters else ""
        sql = f"""
        WITH latest AS (
          SELECT as_of_ts, ticker, exchange_code, instrument_id, close_price_krw, close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d, volatility_20d, sentiment_score, source, updated_at,
                 ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY as_of_ts DESC, updated_at DESC) AS rn
          FROM `{self.session.dataset_fqn}.market_features_latest`
          {where}
        )
        SELECT as_of_ts, ticker, exchange_code, instrument_id, close_price_krw, close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d, volatility_20d, sentiment_score, source
        FROM latest
        WHERE rn = 1
        ORDER BY {field} {direction}
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    def get_daily_closes(
        self,
        *,
        tickers: list[str],
        lookback_days: int,
        sources: list[str] | None = None,
    ) -> dict[str, list[float]]:
        """Loads close_price_krw series per ticker from market_features."""
        tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
        if not tickers:
            return {}

        limit = max(2, min(int(lookback_days), 400))

        filters: list[str] = [
            "ticker IN UNNEST(@tickers)",
            "close_price_krw IS NOT NULL",
            _DAILY_FILTER_SQL,
        ]
        params: dict[str, Any] = {"tickers": tickers, "limit": limit}

        if sources:
            filters.append("source IN UNNEST(@sources)")
            params["sources"] = sources

        where = "WHERE " + " AND ".join(filters)
        sql = f"""
        WITH dedup AS (
          SELECT as_of_ts, ticker, close_price_krw, source, ingested_at,
                 ROW_NUMBER() OVER (
                   PARTITION BY as_of_ts, ticker
                   ORDER BY IFNULL(ingested_at, as_of_ts) DESC, source DESC
                 ) AS rn_key
          FROM `{self.session.dataset_fqn}.market_features`
          {where}
        ), ranked AS (
          SELECT as_of_ts, ticker, close_price_krw,
                 ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY as_of_ts DESC) AS rn
          FROM dedup
          WHERE rn_key = 1
        )
        SELECT as_of_ts, ticker, close_price_krw
        FROM ranked
        WHERE rn <= @limit
        ORDER BY ticker, as_of_ts ASC
        """
        rows = self.session.fetch_rows(sql, params)

        dedup: dict[str, dict[str, float]] = {}
        for r in rows:
            t = str(r.get("ticker", "")).strip().upper()
            px = r.get("close_price_krw")
            ts_raw = r.get("as_of_ts")
            if not t:
                continue
            try:
                val = float(px)
            except (TypeError, ValueError):
                continue
            ts_key = ""
            if isinstance(ts_raw, datetime):
                ts_key = ts_raw.isoformat()
            elif ts_raw is not None:
                ts_key = str(ts_raw)
            if not ts_key:
                continue
            dedup.setdefault(t, {})[ts_key] = val

        out: dict[str, list[float]] = {}
        for t, ts_map in dedup.items():
            ordered = [v for _, v in sorted(ts_map.items(), key=lambda kv: kv[0])]
            out[t] = ordered

        return out

    def get_daily_close_frame(
        self,
        *,
        tickers: list[str],
        start: date,
        end: date,
        sources: list[str] | None = None,
        price_field: str = "close_price_krw",
    ) -> "pd.DataFrame":
        """Loads daily close prices into a DataFrame (index=datetime, columns=ticker)."""
        import pandas as pd

        tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return pd.DataFrame()
        price_column = str(price_field or "close_price_krw").strip()
        if price_column not in {"close_price_krw", "close_price_native"}:
            raise ValueError("price_field must be close_price_krw or close_price_native")

        filters: list[str] = [
            "ticker IN UNNEST(@tickers)",
            f"{price_column} IS NOT NULL",
            "DATE(as_of_ts) >= @start",
            "DATE(as_of_ts) <= @end",
            _DAILY_FILTER_SQL,
        ]
        params: dict[str, Any] = {
            "tickers": tokens,
            "start": start,
            "end": end,
        }

        if sources:
            filters.append("source IN UNNEST(@sources)")
            params["sources"] = sources

        where = " AND ".join(filters)
        sql = f"""
        WITH dedup_source AS (
          SELECT DATE(as_of_ts) AS d, as_of_ts, ticker, {price_column} AS close_price, source, ingested_at,
                 ROW_NUMBER() OVER (
                   PARTITION BY DATE(as_of_ts), ticker, source
                   ORDER BY as_of_ts DESC, IFNULL(ingested_at, as_of_ts) DESC
                 ) AS rn_source
          FROM `{self.session.dataset_fqn}.market_features`
          WHERE {where}
        ), dedup_day AS (
          SELECT d, ticker, close_price, as_of_ts, source, ingested_at,
                 ROW_NUMBER() OVER (
                   PARTITION BY d, ticker
                   ORDER BY as_of_ts DESC, IFNULL(ingested_at, as_of_ts) DESC, source DESC
                 ) AS rn_day
          FROM dedup_source
          WHERE rn_source = 1
        )
        SELECT d, ticker, close_price
        FROM dedup_day
        WHERE rn_day = 1
        ORDER BY d ASC, ticker ASC
        """
        rows = self.session.fetch_rows(sql, params)
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame()

        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["d"] = pd.to_datetime(df["d"])
        # Safety net: keep one row per day/ticker even if upstream query returns duplicates.
        df = df.sort_values(["d", "ticker"]).drop_duplicates(subset=["d", "ticker"], keep="last")
        pivot = df.pivot(index="d", columns="ticker", values="close_price").sort_index()
        return pivot

    def get_predicted_returns(
        self,
        tickers: list[str] | None = None,
        limit: int = 50,
        mode: str = "stacked",
        table_id: str | None = None,
        staleness_days: int | None = None,
    ) -> list[dict[str, Any]]:
        """Loads latest forecast expected annual returns with optional model-mode filtering.

        Args:
            staleness_days: Maximum age in calendar days for forecast data.
                Defaults to ``ARENA_FORECAST_STALENESS_DAYS`` env var or 5.
                Covers weekends + holidays.  Set to 0 to disable the filter.
        """
        limit = max(1, min(int(limit), 500))
        if staleness_days is None:
            try:
                staleness_days = int(os.getenv("ARENA_FORECAST_STALENESS_DAYS", "5"))
            except (TypeError, ValueError):
                staleness_days = 5
        staleness_days = max(0, int(staleness_days))

        filt: list[str] | None = None
        if tickers:
            tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tokens = list(dict.fromkeys(tokens))
            if tokens:
                filt = tokens

        mode_norm = self._normalize_forecast_mode(mode)
        table_ref = self._normalize_forecast_table_id(table_id)
        cols = self._forecast_table_columns(table_ref)
        model_col = next((c for c in _FORECAST_MODEL_COLUMNS if c in cols), None)
        stack_col = next((c for c in _FORECAST_STACK_FLAG_COLUMNS if c in cols), None)
        score_col = next((c for c in _FORECAST_SCORE_COLUMNS if c in cols), None)

        has_period_col = "exp_return_period" in cols
        ret_col = "exp_return_period" if has_period_col else "exp_return_annual"
        select_cols = ["r.run_date", "r.ticker", f"r.{ret_col} AS exp_return_period"]
        if has_period_col and "forecast_horizon" in cols:
            select_cols.append("r.forecast_horizon")
        if model_col:
            select_cols.append(f"CAST(r.{model_col} AS STRING) AS forecast_model")
        if stack_col:
            select_cols.append(f"CAST(r.{stack_col} AS BOOL) AS is_stacked")
        if score_col:
            select_cols.append(f"CAST(r.{score_col} AS FLOAT64) AS forecast_score")
        # Classification columns (optional — graceful if missing)
        for cls_col in ("prob_up", "model_votes_up", "model_votes_total", "consensus"):
            if cls_col in cols:
                select_cols.append(f"r.{cls_col}")

        filters: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if filt:
            filters.append("ticker IN UNNEST(@tickers)")
            params["tickers"] = filt

        if mode_norm == "all":
            pass
        elif mode_norm == "stacked" and stack_col:
            filters.append("IFNULL(CAST(r.{col} AS BOOL), FALSE)".format(col=stack_col))
        elif mode_norm == "base" and stack_col:
            filters.append("NOT IFNULL(CAST(r.{col} AS BOOL), FALSE)".format(col=stack_col))
        elif model_col:
            params["forecast_modes"] = list(self._forecast_mode_aliases(mode_norm))
            filters.append("LOWER(CAST(r.{col} AS STRING)) IN UNNEST(@forecast_modes)".format(col=model_col))
        else:
            logger.warning(
                "[yellow]Forecast query blocked[/yellow] mode=%s table=%s reason=missing model metadata columns",
                mode_norm,
                table_ref,
            )
            return []

        where = "WHERE " + " AND ".join(filters) if filters else ""
        staleness_clause = ""
        if staleness_days > 0:
            staleness_clause = "WHERE run_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @staleness_days DAY)"
            params["staleness_days"] = staleness_days

        if "forecast_run_id" in cols:
            sql = f"""
            WITH latest AS (
              SELECT MAX(run_date) AS run_date
              FROM `{table_ref}`
              {staleness_clause}
            ),
            latest_batch AS (
              SELECT forecast_run_id, created_at
              FROM `{table_ref}`
              WHERE run_date = (SELECT run_date FROM latest)
              QUALIFY ROW_NUMBER() OVER (
                ORDER BY created_at DESC, IFNULL(CAST(forecast_run_id AS STRING), '') DESC
              ) = 1
            )
            SELECT {", ".join(select_cols)}
            FROM `{table_ref}` r
            JOIN latest l USING (run_date)
            JOIN latest_batch b
              ON (
                (b.forecast_run_id IS NOT NULL AND r.forecast_run_id = b.forecast_run_id)
                OR (b.forecast_run_id IS NULL AND r.forecast_run_id IS NULL AND r.created_at = b.created_at)
              )
            {where}
            ORDER BY r.{ret_col} DESC
            LIMIT @limit
            """
        else:
            sql = f"""
            WITH latest AS (
              SELECT MAX(run_date) AS run_date
              FROM `{table_ref}`
              {staleness_clause}
            )
            SELECT {", ".join(select_cols)}
            FROM `{table_ref}` r
            JOIN latest l USING (run_date)
            {where}
            ORDER BY r.{ret_col} DESC
            LIMIT @limit
            """

        try:
            rows = self.session.fetch_rows(sql, params)
        except Exception as exc:
            logger.warning(
                "[yellow]BigQuery predicted_returns query failed[/yellow] table=%s err=%s",
                table_ref,
                str(exc),
            )
            return []

        out: list[dict[str, Any]] = []
        for r in rows:
            run_date = r.get("run_date")
            if isinstance(run_date, (datetime, date)):
                run_date = run_date.isoformat()
            elif run_date is not None:
                run_date = str(run_date)

            ticker = str(r.get("ticker", "")).strip().upper()
            if not ticker:
                continue

            try:
                exp_return_period = float(r.get("exp_return_period"))
            except (TypeError, ValueError):
                continue

            entry: dict[str, Any] = {
                "run_date": run_date,
                "ticker": ticker,
                "exp_return_period": exp_return_period,
            }
            if r.get("forecast_horizon") is not None:
                try:
                    entry["forecast_horizon"] = int(r["forecast_horizon"])
                except (TypeError, ValueError):
                    pass
            out.append(entry)
            if r.get("forecast_model") is not None:
                model_name = str(r.get("forecast_model")).strip()
                if model_name:
                    out[-1]["forecast_model"] = model_name
            if r.get("is_stacked") is not None:
                out[-1]["is_stacked"] = bool(r.get("is_stacked"))
            if r.get("forecast_score") is not None:
                try:
                    out[-1]["forecast_score"] = float(r.get("forecast_score"))
                except (TypeError, ValueError):
                    pass
            # Classification columns
            if r.get("prob_up") is not None:
                try:
                    out[-1]["prob_up"] = round(float(r["prob_up"]), 4)
                except (TypeError, ValueError):
                    pass
            if r.get("model_votes_up") is not None:
                try:
                    out[-1]["model_votes_up"] = int(r["model_votes_up"])
                except (TypeError, ValueError):
                    pass
            if r.get("model_votes_total") is not None:
                try:
                    out[-1]["model_votes_total"] = int(r["model_votes_total"])
                except (TypeError, ValueError):
                    pass
            if r.get("consensus") is not None:
                val = str(r["consensus"]).strip()
                if val:
                    out[-1]["consensus"] = val
        return out

    def refresh_signal_daily_values(
        self,
        *,
        lookback_days: int = 540,
        horizon_days: int = 20,
        sources: list[str] | None = None,
        market: str | None = None,
    ) -> int:
        """Materializes Layer 1 signals + forward labels into signal_daily_values.

        All signals are deterministic functions of market_features, predicted
        forecasts, and fundamentals_derived_daily. Forecast and fundamentals
        joins enforce ``table.effective_date <= base.as_of_date`` to block
        look-ahead.

        Returns 0 because BigQuery INSERT does not report row counts here;
        callers observe the resulting table partition instead.
        """
        horizon = max(5, min(int(horizon_days), 60))
        lookback = max(horizon + 40, min(int(lookback_days), 1500))
        filters = [
            "DATE(as_of_ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)",
            "close_price_krw IS NOT NULL",
            "close_price_krw > 0",
            _DAILY_FILTER_SQL,
        ]
        params: dict[str, Any] = {
            "lookback_days": lookback + horizon + 30,
            "horizon": horizon,
            "market": str(market or "").strip().lower() or None,
        }
        if sources:
            tokens = [str(source or "").strip() for source in sources if str(source or "").strip()]
            if tokens:
                filters.append("source IN UNNEST(@sources)")
                params["sources"] = tokens

        where = "WHERE " + " AND ".join(filters)
        dataset = self.session.dataset_fqn
        sql = f"""
        INSERT INTO `{dataset}.signal_daily_values`
        WITH raw AS (
          SELECT
            DATE(as_of_ts) AS as_of_date,
            as_of_ts,
            ticker,
            exchange_code,
            instrument_id,
            source,
            close_price_krw,
            sentiment_score,
            ingested_at,
            ROW_NUMBER() OVER (
              PARTITION BY DATE(as_of_ts), ticker
              ORDER BY as_of_ts DESC, ingested_at DESC, source DESC
            ) AS rn
          FROM `{dataset}.market_features`
          {where}
        ),
        daily AS (
          SELECT * EXCEPT(rn)
          FROM raw
          WHERE rn = 1
        ),
        price_ctx AS (
          SELECT
            *,
            SAFE_DIVIDE(close_price_krw, LAG(close_price_krw, 1) OVER w) - 1.0 AS daily_return,
            SAFE_DIVIDE(close_price_krw, LAG(close_price_krw, 5) OVER w) - 1.0 AS ret_5d,
            SAFE_DIVIDE(close_price_krw, LAG(close_price_krw, 20) OVER w) - 1.0 AS ret_20d,
            AVG(close_price_krw) OVER (
              PARTITION BY ticker ORDER BY as_of_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) AS sma_20,
            AVG(close_price_krw) OVER (
              PARTITION BY ticker ORDER BY as_of_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
            ) AS sma_60,
            STDDEV_SAMP(close_price_krw) OVER (
              PARTITION BY ticker ORDER BY as_of_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) AS std_px_20,
            SAFE_DIVIDE(LEAD(close_price_krw, @horizon) OVER w, close_price_krw) - 1.0 AS fwd_return_20d,
            SAFE_DIVIDE(MIN(close_price_krw) OVER (
              PARTITION BY ticker ORDER BY as_of_date ROWS BETWEEN 1 FOLLOWING AND @horizon FOLLOWING
            ), close_price_krw) - 1.0 AS fwd_mdd_20d
          FROM daily
          WINDOW w AS (PARTITION BY ticker ORDER BY as_of_date)
        ),
        returns AS (
          SELECT
            *,
            STDDEV_SAMP(daily_return) OVER (
              PARTITION BY ticker ORDER BY as_of_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
            ) AS volatility_20d,
            AVG(GREATEST(daily_return, 0)) OVER (
              PARTITION BY ticker ORDER BY as_of_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
            ) AS rsi_up_14,
            AVG(GREATEST(-daily_return, 0)) OVER (
              PARTITION BY ticker ORDER BY as_of_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
            ) AS rsi_dn_14
          FROM price_ctx
        ),
        complete AS (
          SELECT
            *,
            CASE
              WHEN rsi_up_14 IS NULL OR rsi_dn_14 IS NULL THEN NULL
              WHEN (rsi_up_14 + rsi_dn_14) = 0 THEN 50.0
              ELSE 100.0 * SAFE_DIVIDE(rsi_up_14, rsi_up_14 + rsi_dn_14)
            END AS rsi_14
          FROM returns
          WHERE ret_5d IS NOT NULL
            AND ret_20d IS NOT NULL
            AND volatility_20d IS NOT NULL
        ),
        with_forecast AS (
          SELECT
            c.*,
            f.exp_return_period AS forecast_exp_return,
            f.prob_up AS forecast_prob_up,
            ROW_NUMBER() OVER (
              PARTITION BY c.as_of_date, c.ticker
              ORDER BY f.run_date DESC, f.created_at DESC
            ) AS forecast_rn
          FROM complete c
          LEFT JOIN `{dataset}.predicted_expected_returns` f
            ON f.ticker = c.ticker
           AND f.run_date <= c.as_of_date
           AND IFNULL(f.is_stacked, FALSE)
        ),
        with_fundamentals AS (
          SELECT
            w.* EXCEPT(forecast_rn),
            d.ep AS fund_ep,
            d.bp AS fund_bp,
            d.sp AS fund_sp,
            d.roe AS fund_roe,
            d.revenue_growth_yoy AS fund_rev_growth,
            d.eps_growth_yoy AS fund_eps_growth,
            d.debt_to_equity AS fund_debt_equity,
            ROW_NUMBER() OVER (
              PARTITION BY w.as_of_date, w.ticker
              ORDER BY d.latest_announcement_date DESC, d.created_at DESC
            ) AS fund_rn
          FROM with_forecast w
          LEFT JOIN `{dataset}.fundamentals_derived_daily` d
            ON d.ticker = w.ticker
           AND d.as_of_date <= w.as_of_date
           AND d.latest_announcement_date <= w.as_of_date
          WHERE w.forecast_rn = 1
        ),
        cross_section AS (
          SELECT
            * EXCEPT(fund_rn),
            AVG(ret_20d) OVER (PARTITION BY as_of_date) AS cs_ret20_mean,
            STDDEV_SAMP(ret_20d) OVER (PARTITION BY as_of_date) AS cs_ret20_std,
            AVG(ret_5d) OVER (PARTITION BY as_of_date) AS cs_ret5_mean,
            STDDEV_SAMP(ret_5d) OVER (PARTITION BY as_of_date) AS cs_ret5_std,
            AVG(volatility_20d) OVER (PARTITION BY as_of_date) AS cs_vol_mean,
            STDDEV_SAMP(volatility_20d) OVER (PARTITION BY as_of_date) AS cs_vol_std,
            AVG(IFNULL(sentiment_score, 0.0)) OVER (PARTITION BY as_of_date) AS cs_sent_mean,
            STDDEV_SAMP(IFNULL(sentiment_score, 0.0)) OVER (PARTITION BY as_of_date) AS cs_sent_std,
            AVG(fund_ep) OVER (PARTITION BY as_of_date) AS cs_ep_mean,
            STDDEV_SAMP(fund_ep) OVER (PARTITION BY as_of_date) AS cs_ep_std,
            AVG(fund_bp) OVER (PARTITION BY as_of_date) AS cs_bp_mean,
            STDDEV_SAMP(fund_bp) OVER (PARTITION BY as_of_date) AS cs_bp_std,
            AVG(fund_sp) OVER (PARTITION BY as_of_date) AS cs_sp_mean,
            STDDEV_SAMP(fund_sp) OVER (PARTITION BY as_of_date) AS cs_sp_std,
            AVG(fund_roe) OVER (PARTITION BY as_of_date) AS cs_roe_mean,
            STDDEV_SAMP(fund_roe) OVER (PARTITION BY as_of_date) AS cs_roe_std,
            AVG(fund_rev_growth) OVER (PARTITION BY as_of_date) AS cs_rev_mean,
            STDDEV_SAMP(fund_rev_growth) OVER (PARTITION BY as_of_date) AS cs_rev_std,
            AVG(fund_eps_growth) OVER (PARTITION BY as_of_date) AS cs_epsg_mean,
            STDDEV_SAMP(fund_eps_growth) OVER (PARTITION BY as_of_date) AS cs_epsg_std,
            AVG(fund_debt_equity) OVER (PARTITION BY as_of_date) AS cs_debt_mean,
            STDDEV_SAMP(fund_debt_equity) OVER (PARTITION BY as_of_date) AS cs_debt_std,
            AVG(fwd_return_20d) OVER (PARTITION BY as_of_date) AS fwd_benchmark_return_20d
          FROM with_fundamentals
          WHERE fund_rn = 1 OR fund_rn IS NULL
        ),
        signaled AS (
          SELECT
            *,
            SAFE_DIVIDE(ret_20d - cs_ret20_mean, NULLIF(cs_ret20_std, 0.0)) AS z_ret20,
            SAFE_DIVIDE(ret_5d - cs_ret5_mean, NULLIF(cs_ret5_std, 0.0)) AS z_ret5,
            SAFE_DIVIDE(volatility_20d - cs_vol_mean, NULLIF(cs_vol_std, 0.0)) AS z_vol,
            SAFE_DIVIDE(IFNULL(sentiment_score, 0.0) - cs_sent_mean, NULLIF(cs_sent_std, 0.0)) AS z_sent,
            SAFE_DIVIDE(fund_ep - cs_ep_mean, NULLIF(cs_ep_std, 0.0)) AS z_ep,
            SAFE_DIVIDE(fund_bp - cs_bp_mean, NULLIF(cs_bp_std, 0.0)) AS z_bp,
            SAFE_DIVIDE(fund_sp - cs_sp_mean, NULLIF(cs_sp_std, 0.0)) AS z_sp,
            SAFE_DIVIDE(fund_roe - cs_roe_mean, NULLIF(cs_roe_std, 0.0)) AS z_roe,
            SAFE_DIVIDE(fund_rev_growth - cs_rev_mean, NULLIF(cs_rev_std, 0.0)) AS z_rev,
            SAFE_DIVIDE(fund_eps_growth - cs_epsg_mean, NULLIF(cs_epsg_std, 0.0)) AS z_epsg,
            SAFE_DIVIDE(fund_debt_equity - cs_debt_mean, NULLIF(cs_debt_std, 0.0)) AS z_debt,
            CASE
              WHEN ret_20d > 0 AND ret_5d < 0 THEN 'pullback'
              WHEN ret_20d < 0 AND ret_5d > 0 THEN 'recovery'
              WHEN SAFE_DIVIDE(volatility_20d - cs_vol_mean, NULLIF(cs_vol_std, 0.0)) <= -0.65 THEN 'defensive'
              ELSE 'momentum'
            END AS bucket_calc
          FROM cross_section
        ),
        profiled AS (
          SELECT
            *,
            CASE
              WHEN bucket_calc IN ('momentum', 'recovery') THEN 'aggressive'
              WHEN bucket_calc = 'pullback' THEN 'balanced'
              WHEN bucket_calc = 'defensive' THEN 'defensive'
              ELSE 'balanced'
            END AS profile_calc
          FROM signaled
        )
        SELECT
          as_of_date,
          CURRENT_TIMESTAMP() AS created_at,
          ticker,
          @market AS market,
          exchange_code,
          instrument_id,
          source,
          bucket_calc AS bucket,
          profile_calc AS profile,
          z_ret20 AS signal_momentum_20d,
          CASE
            WHEN z_ret20 IS NULL OR z_ret5 IS NULL THEN NULL
            WHEN z_ret20 > 0 THEN -z_ret5
            ELSE 0.0
          END AS signal_pullback,
          -z_ret5 AS signal_meanrev_5d,
          -z_vol AS signal_lowvol,
          z_sent AS signal_sentiment,
          forecast_exp_return AS signal_forecast_er,
          forecast_prob_up - 0.5 AS signal_forecast_prob,
          CASE
            WHEN rsi_14 IS NULL THEN NULL
            WHEN rsi_14 < 30 THEN 1.0
            WHEN rsi_14 > 70 THEN -1.0
            ELSE 0.0
          END AS signal_rsi_reversal,
          CASE
            WHEN sma_20 IS NULL OR sma_60 IS NULL THEN NULL
            WHEN sma_20 > sma_60 THEN 1.0
            ELSE -1.0
          END AS signal_ma_crossover,
          CASE
            WHEN sma_20 IS NULL OR std_px_20 IS NULL OR std_px_20 = 0 THEN NULL
            ELSE SAFE_DIVIDE(close_price_krw - sma_20, 2.0 * std_px_20)
          END AS signal_bollinger_position,
          z_ep AS signal_ep,
          z_bp AS signal_bp,
          z_sp AS signal_sp,
          z_roe AS signal_roe,
          z_rev AS signal_revenue_growth,
          z_epsg AS signal_eps_growth,
          -z_debt AS signal_low_debt,
          ret_5d,
          ret_20d,
          volatility_20d,
          sentiment_score,
          close_price_krw,
          fwd_return_20d,
          fwd_benchmark_return_20d,
          fwd_return_20d - fwd_benchmark_return_20d AS fwd_excess_return_20d,
          fwd_mdd_20d,
          fwd_return_20d IS NOT NULL
            AND as_of_date <= DATE_SUB(CURRENT_DATE(), INTERVAL @horizon DAY) AS label_ready
        FROM profiled
        WHERE as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)
        """
        self.session.execute(sql, params)
        return 0

    def refresh_signal_daily_ic(
        self,
        *,
        lookback_days: int = 540,
        horizon_days: int = 20,
        market: str | None = None,
    ) -> int:
        """Computes per-signal cross-section IC and Rank-IC time series.

        IC at date t measures how well signal_i at (t - horizon) predicts the
        realized excess return between (t - horizon) and t. Rows with
        ``label_ready = FALSE`` are skipped.
        """
        dataset = self.session.dataset_fqn
        signal_columns = [
            "signal_momentum_20d",
            "signal_pullback",
            "signal_meanrev_5d",
            "signal_lowvol",
            "signal_sentiment",
            "signal_forecast_er",
            "signal_forecast_prob",
            "signal_rsi_reversal",
            "signal_ma_crossover",
            "signal_bollinger_position",
            "signal_ep",
            "signal_bp",
            "signal_sp",
            "signal_roe",
            "signal_revenue_growth",
            "signal_eps_growth",
            "signal_low_debt",
        ]
        params: dict[str, Any] = {
            "lookback_days": max(40, min(int(lookback_days), 1500)),
            "horizon": max(5, min(int(horizon_days), 60)),
            "market": str(market or "").strip().lower() or None,
        }
        unions: list[str] = []
        for col in signal_columns:
            signal_name = col.removeprefix("signal_")
            unions.append(
                f"""
              SELECT
                as_of_date,
                CURRENT_TIMESTAMP() AS created_at,
                '{signal_name}' AS signal_name,
                @horizon AS horizon_days,
                CORR(signal_val, fwd_excess_return_20d) AS ic_20d,
                CORR(signal_rank, return_rank) AS rank_ic_20d,
                COUNT(*) AS sample_size,
                @market AS market
              FROM (
                SELECT
                  as_of_date,
                  {col} AS signal_val,
                  fwd_excess_return_20d,
                  PERCENT_RANK() OVER (PARTITION BY as_of_date ORDER BY {col}) AS signal_rank,
                  PERCENT_RANK() OVER (PARTITION BY as_of_date ORDER BY fwd_excess_return_20d) AS return_rank
                FROM `{dataset}.signal_daily_values`
                WHERE label_ready
                  AND as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)
                  AND {col} IS NOT NULL
                  AND fwd_excess_return_20d IS NOT NULL
                  AND (@market IS NULL OR market = @market)
              )
              GROUP BY as_of_date
                """
            )
        sql = f"INSERT INTO `{dataset}.signal_daily_ic`\n" + "\nUNION ALL\n".join(unions)
        self.session.execute(sql, params)
        return 0

    def refresh_regime_daily_features(
        self,
        *,
        lookback_days: int = 540,
        market: str | None = None,
    ) -> int:
        """Computes per-date regime features from signal_daily_values."""
        dataset = self.session.dataset_fqn
        params = {
            "lookback_days": max(40, min(int(lookback_days), 1500)),
            "market": str(market or "").strip().lower() or None,
        }
        sql = f"""
        INSERT INTO `{dataset}.regime_daily_features`
        SELECT
          as_of_date,
          CURRENT_TIMESTAMP() AS created_at,
          @market AS market,
          APPROX_QUANTILES(volatility_20d, 101)[OFFSET(50)] AS regime_vol_level,
          STDDEV_SAMP(volatility_20d) AS regime_vol_dispersion,
          AVG(ret_20d) AS regime_trend,
          AVG(ret_5d) AS regime_short_reversal,
          STDDEV_SAMP(ret_20d) AS regime_dispersion,
          AVG(IFNULL(sentiment_score, 0.0)) AS regime_sentiment,
          COUNT(*) AS sample_size
        FROM `{dataset}.signal_daily_values`
        WHERE as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)
          AND (@market IS NULL OR market = @market)
        GROUP BY as_of_date
        """
        self.session.execute(sql, params)
        return 0

    def load_signal_daily_ic(
        self,
        *,
        lookback_days: int = 540,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        dataset = self.session.dataset_fqn
        params = {
            "lookback_days": max(40, min(int(lookback_days), 1500)),
            "market": str(market or "").strip().lower() or None,
        }
        sql = f"""
        WITH dedup AS (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY as_of_date, signal_name
            ORDER BY created_at DESC
          ) AS rn
          FROM `{dataset}.signal_daily_ic`
          WHERE as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)
            AND (@market IS NULL OR market = @market OR market IS NULL)
        )
        SELECT * EXCEPT(rn)
        FROM dedup
        WHERE rn = 1
        ORDER BY as_of_date, signal_name
        """
        return self.session.fetch_rows(sql, params)

    def load_regime_daily_features(
        self,
        *,
        lookback_days: int = 540,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        dataset = self.session.dataset_fqn
        params = {
            "lookback_days": max(40, min(int(lookback_days), 1500)),
            "market": str(market or "").strip().lower() or None,
        }
        sql = f"""
        WITH dedup AS (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY as_of_date
            ORDER BY created_at DESC
          ) AS rn
          FROM `{dataset}.regime_daily_features`
          WHERE as_of_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)
            AND (@market IS NULL OR market = @market OR market IS NULL)
        )
        SELECT * EXCEPT(rn)
        FROM dedup
        WHERE rn = 1
        ORDER BY as_of_date
        """
        return self.session.fetch_rows(sql, params)

    def load_signal_scoring_rows(
        self,
        *,
        limit: int = 500,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns latest-date signal rows used for today's scoring."""
        dataset = self.session.dataset_fqn
        params = {
            "limit": max(1, min(int(limit), 5000)),
            "market": str(market or "").strip().lower() or None,
        }
        sql = f"""
        WITH dedup AS (
          SELECT *, ROW_NUMBER() OVER (
            PARTITION BY as_of_date, ticker
            ORDER BY created_at DESC
          ) AS rn
          FROM `{dataset}.signal_daily_values`
          WHERE (@market IS NULL OR market = @market OR market IS NULL)
        ),
        latest AS (
          SELECT MAX(as_of_date) AS as_of_date
          FROM dedup
          WHERE rn = 1
        )
        SELECT d.* EXCEPT(rn)
        FROM dedup d
        JOIN latest l USING (as_of_date)
        WHERE d.rn = 1
        ORDER BY d.signal_momentum_20d DESC NULLS LAST, d.ticker
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    def insert_opportunity_ranker_scores_latest(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        table_id = f"{self.session.dataset_fqn}.opportunity_ranker_scores_latest"
        payloads: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            ticker = str(data.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            data["ticker"] = ticker
            for key in ("feature_json", "explanation_json"):
                value = data.get(key)
                if isinstance(value, str):
                    try:
                        data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        data[key] = {"raw": value}
            payloads.append(data)
        return self._append_json_rows_via_load_job(table_id, payloads)

    def append_opportunity_ranker_run(self, row: dict[str, Any]) -> int:
        if not row:
            return 0
        table_id = f"{self.session.dataset_fqn}.opportunity_ranker_runs"
        data = dict(row)
        for key in ("detail_json", "feature_columns"):
            value = data.get(key)
            if isinstance(value, str):
                try:
                    data[key] = json.loads(value)
                except json.JSONDecodeError:
                    data[key] = {"raw": value} if key == "detail_json" else [value]
        return self._append_json_rows_via_load_job(table_id, [data])

    def insert_fundamentals_history_raw(self, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        table_id = f"{self.session.dataset_fqn}.fundamentals_history_raw"
        payloads: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            ticker = str(data.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            data["ticker"] = ticker
            payloads.append(data)
        return self._append_json_rows_via_load_job(table_id, payloads)

    def append_fundamentals_ingest_run(self, row: dict[str, Any]) -> int:
        if not row:
            return 0
        table_id = f"{self.session.dataset_fqn}.fundamentals_ingest_runs"
        data = dict(row)
        value = data.get("detail_json")
        if isinstance(value, str):
            try:
                data["detail_json"] = json.loads(value)
            except json.JSONDecodeError:
                data["detail_json"] = {"raw": value}
        return self._append_json_rows_via_load_job(table_id, [data])

    def refresh_fundamentals_derived_daily(
        self,
        *,
        lookback_days: int = 600,
        market: str | None = None,
    ) -> int:
        """Materializes PIT-safe fundamentals ratios per (as_of_date, ticker).

        For each market_features date in the lookback window, joins the
        latest announcement whose ``announcement_date`` is on or before
        ``as_of_date`` and computes daily ratios using close price. TTM
        aggregates use the trailing 4 quarters of announcements.
        """
        dataset = self.session.dataset_fqn
        params = {
            "lookback_days": max(40, min(int(lookback_days), 1500)),
            "market": str(market or "").strip().lower() or None,
        }
        sql = f"""
        INSERT INTO `{dataset}.fundamentals_derived_daily`
        WITH price_days AS (
          SELECT
            DATE(as_of_ts) AS as_of_date,
            ticker,
            CASE
              WHEN SAFE_CAST(ticker AS INT64) IS NOT NULL AND LENGTH(ticker) = 6 THEN 'kospi'
              ELSE 'us'
            END AS market,
            close_price_krw,
            ROW_NUMBER() OVER (
              PARTITION BY DATE(as_of_ts), ticker
              ORDER BY as_of_ts DESC, ingested_at DESC
            ) AS rn
          FROM `{dataset}.market_features`
          WHERE DATE(as_of_ts) >= DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)
            AND close_price_krw IS NOT NULL
            AND close_price_krw > 0
            AND (
              @market IS NULL
              OR (@market = 'kospi' AND SAFE_CAST(ticker AS INT64) IS NOT NULL AND LENGTH(ticker) = 6)
              OR (@market = 'us' AND (SAFE_CAST(ticker AS INT64) IS NULL OR LENGTH(ticker) != 6))
            )
            AND {_DAILY_FILTER_SQL}
        ),
        daily_price AS (
          SELECT * EXCEPT(rn) FROM price_days WHERE rn = 1
        ),
        history_dedup AS (
          SELECT
            *,
            ROW_NUMBER() OVER (
              PARTITION BY ticker, fiscal_year, fiscal_quarter
              ORDER BY retrieved_at DESC
            ) AS rn
          FROM `{dataset}.fundamentals_history_raw`
        ),
        history AS (
          SELECT * EXCEPT(rn) FROM history_dedup WHERE rn = 1
        ),
        ttm AS (
          SELECT
            ticker,
            fiscal_year,
            fiscal_quarter,
            fiscal_period_end,
            announcement_date,
            currency,
            net_income,
            revenue,
            gross_profit,
            operating_income,
            ebitda,
            total_equity,
            total_assets,
            total_debt,
            book_value_per_share,
            eps_diluted,
            SUM(net_income) OVER ttm_w AS ni_ttm,
            SUM(revenue) OVER ttm_w AS revenue_ttm,
            SUM(gross_profit) OVER ttm_w AS gross_ttm,
            SUM(operating_income) OVER ttm_w AS op_income_ttm,
            SUM(ebitda) OVER ttm_w AS ebitda_ttm,
            SUM(eps_diluted) OVER ttm_w AS eps_ttm,
            LAG(revenue, 4) OVER lag_w AS revenue_yoy,
            LAG(eps_diluted, 4) OVER lag_w AS eps_yoy
          FROM history
          WINDOW
            ttm_w AS (
              PARTITION BY ticker
              ORDER BY fiscal_period_end
              ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ),
            lag_w AS (
              PARTITION BY ticker
              ORDER BY fiscal_period_end
            )
        ),
        joined AS (
          SELECT
            p.as_of_date,
            p.ticker,
            p.market,
            p.close_price_krw,
            t.announcement_date,
            t.fiscal_period_end,
            t.currency,
            t.eps_ttm,
            t.revenue_ttm,
            t.gross_ttm,
            t.op_income_ttm,
            t.ebitda_ttm,
            t.ni_ttm,
            t.total_equity,
            t.total_assets,
            t.total_debt,
            t.book_value_per_share,
            t.revenue_yoy,
            t.eps_yoy,
            ROW_NUMBER() OVER (
              PARTITION BY p.as_of_date, p.ticker
              ORDER BY t.announcement_date DESC, t.fiscal_period_end DESC
            ) AS rn
          FROM daily_price p
          LEFT JOIN ttm t
            ON t.ticker = p.ticker
           AND t.announcement_date <= p.as_of_date
        )
        SELECT
          as_of_date,
          CURRENT_TIMESTAMP() AS created_at,
          ticker,
          market,
          fiscal_period_end AS latest_fiscal_period_end,
          announcement_date AS latest_announcement_date,
          DATE_DIFF(as_of_date, announcement_date, DAY) AS days_since_announcement,
          close_price_krw AS price_native,
          close_price_krw AS price_krw,
          SAFE_DIVIDE(close_price_krw, NULLIF(eps_ttm, 0)) AS pe,
          SAFE_DIVIDE(close_price_krw, NULLIF(book_value_per_share, 0)) AS pb,
          SAFE_DIVIDE(close_price_krw, NULLIF(SAFE_DIVIDE(revenue_ttm, NULLIF(total_equity, 0)), 0)) AS ps,
          SAFE_DIVIDE(eps_ttm, NULLIF(close_price_krw, 0)) AS ep,
          SAFE_DIVIDE(book_value_per_share, NULLIF(close_price_krw, 0)) AS bp,
          SAFE_DIVIDE(revenue_ttm, NULLIF(close_price_krw * NULLIF(total_equity, 0), 0)) AS sp,
          SAFE_DIVIDE(NULLIF(total_assets, 0) + NULLIF(total_debt, 0) - total_equity, NULLIF(ebitda_ttm, 0)) AS ev_ebitda,
          SAFE_DIVIDE(ni_ttm, NULLIF(total_equity, 0)) AS roe,
          SAFE_DIVIDE(ni_ttm, NULLIF(total_assets, 0)) AS roa,
          SAFE_DIVIDE(gross_ttm, NULLIF(revenue_ttm, 0)) AS gross_margin,
          SAFE_DIVIDE(op_income_ttm, NULLIF(revenue_ttm, 0)) AS operating_margin,
          SAFE_DIVIDE(revenue_ttm, NULLIF(revenue_yoy, 0)) - 1.0 AS revenue_growth_yoy,
          SAFE_DIVIDE(eps_ttm, NULLIF(eps_yoy, 0)) - 1.0 AS eps_growth_yoy,
          SAFE_DIVIDE(total_debt, NULLIF(total_equity, 0)) AS debt_to_equity,
          CASE
            WHEN announcement_date IS NULL THEN 'missing'
            WHEN DATE_DIFF(as_of_date, announcement_date, DAY) > 200 THEN 'stale'
            WHEN DATE_DIFF(as_of_date, announcement_date, DAY) > 120 THEN 'aging'
            ELSE 'fresh'
          END AS coverage_confidence
        FROM joined
        WHERE rn = 1
        """
        self.session.execute(sql, params)
        return 0

    def load_fundamentals_history_raw(
        self,
        *,
        tickers: list[str] | None = None,
        market: str | None = None,
        limit: int = 20000,
    ) -> list[dict[str, Any]]:
        dataset = self.session.dataset_fqn
        params: dict[str, Any] = {
            "limit": max(1, min(int(limit), 200000)),
            "market": str(market or "").strip().lower() or None,
        }
        filters = [
            "(@market IS NULL OR market = @market)",
        ]
        if tickers:
            tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
            if tokens:
                filters.append("ticker IN UNNEST(@tickers)")
                params["tickers"] = list(dict.fromkeys(tokens))
        where = " AND ".join(filters)
        sql = f"""
        WITH dedup AS (
          SELECT *,
                 ROW_NUMBER() OVER (
                   PARTITION BY ticker, fiscal_year, fiscal_quarter
                   ORDER BY retrieved_at DESC
                 ) AS rn
          FROM `{dataset}.fundamentals_history_raw`
          WHERE {where}
        )
        SELECT * EXCEPT(rn)
        FROM dedup
        WHERE rn = 1
        ORDER BY ticker, fiscal_year, fiscal_quarter
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    def latest_opportunity_ranker_scores(
        self,
        *,
        tickers: list[str] | None = None,
        profiles: list[str] | None = None,
        buckets: list[str] | None = None,
        per_profile_limit: int | None = None,
        limit: int = 50,
        max_age_hours: int = 30,
    ) -> list[dict[str, Any]]:
        profile_limit = 0
        if per_profile_limit is not None:
            profile_limit = max(0, min(int(per_profile_limit), 100))
        max_return_rows = min(500, max(1, min(int(limit), 500)) + profile_limit * 8)
        batch_filters = [
            "computed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @max_age_hours HOUR)",
        ]
        row_filters = [
            "s.computed_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @max_age_hours HOUR)",
        ]
        params: dict[str, Any] = {
            "limit": max(1, min(int(limit), 500)),
            "max_age_hours": max(1, min(int(max_age_hours), 24 * 14)),
            "per_profile_limit": profile_limit,
            "max_return_rows": max_return_rows,
        }
        if tickers:
            tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
            if tokens:
                batch_filters.append("ticker IN UNNEST(@tickers)")
                row_filters.append("s.ticker IN UNNEST(@tickers)")
                params["tickers"] = list(dict.fromkeys(tokens))
        if profiles:
            profile_tokens = [str(p).strip().lower() for p in profiles if str(p).strip()]
            if profile_tokens:
                batch_filters.append("profile IN UNNEST(@profiles)")
                row_filters.append("s.profile IN UNNEST(@profiles)")
                params["profiles"] = list(dict.fromkeys(profile_tokens))
        if buckets:
            bucket_tokens = [str(b).strip().lower() for b in buckets if str(b).strip()]
            if bucket_tokens:
                batch_filters.append("bucket IN UNNEST(@buckets)")
                row_filters.append("s.bucket IN UNNEST(@buckets)")
                params["buckets"] = list(dict.fromkeys(bucket_tokens))

        batch_where = "WHERE " + " AND ".join(batch_filters)
        row_where = "WHERE " + " AND ".join(row_filters)
        sql = f"""
        WITH latest_batch AS (
          SELECT ranker_version, computed_at
          FROM `{self.session.dataset_fqn}.opportunity_ranker_scores_latest`
          {batch_where}
          QUALIFY ROW_NUMBER() OVER (
            ORDER BY computed_at DESC, ranker_version DESC
          ) = 1
        ),
        dedup AS (
          SELECT s.*
          FROM `{self.session.dataset_fqn}.opportunity_ranker_scores_latest` s
          JOIN latest_batch b
            ON s.ranker_version = b.ranker_version
           AND s.computed_at = b.computed_at
          {row_where}
          QUALIFY ROW_NUMBER() OVER (
            PARTITION BY s.ticker
            ORDER BY s.recommendation_rank ASC, s.recommendation_score DESC
          ) = 1
        ),
        ranked AS (
          SELECT
            d.*,
            ROW_NUMBER() OVER (
              ORDER BY d.recommendation_rank ASC, d.recommendation_score DESC, d.ticker
            ) AS global_rn,
            ROW_NUMBER() OVER (
              PARTITION BY d.profile
              ORDER BY d.recommendation_rank ASC, d.recommendation_score DESC, d.ticker
            ) AS profile_rn
          FROM dedup d
        )
        SELECT * EXCEPT(global_rn, profile_rn)
        FROM ranked
        WHERE global_rn <= @limit
           OR (@per_profile_limit > 0 AND profile_rn <= @per_profile_limit)
        ORDER BY recommendation_rank ASC, recommendation_score DESC, ticker
        LIMIT @max_return_rows
        """
        return self.session.fetch_rows(sql, params)

    def _normalize_forecast_mode(self, mode: str | None) -> str:
        token = str(mode or "").strip().lower()
        if not token:
            return "stacked"
        if token == "auto":
            return "stacked"
        for key, aliases in _FORECAST_MODE_ALIASES.items():
            if token == key or token in aliases:
                return key
        return token

    def _forecast_mode_aliases(self, mode: str) -> tuple[str, ...]:
        token = self._normalize_forecast_mode(mode)
        return _FORECAST_MODE_ALIASES.get(token, (token,))

    def _normalize_forecast_table_id(self, table_id: str | None) -> str:
        raw = str(table_id or os.getenv("ARENA_FORECAST_TABLE", "")).strip()
        if not raw:
            return f"{self.session.project}.{self.session.dataset}.{_DEFAULT_FORECAST_TABLE}"
        parts = [p for p in raw.split(".") if p]
        if len(parts) == 3:
            return raw
        if len(parts) == 2:
            return f"{self.session.project}.{raw}"
        return f"{self.session.project}.{self.session.dataset}.{raw}"

    def _forecast_table_columns(self, table_id: str) -> set[str]:
        try:
            table = self.session.client.get_table(table_id)
        except Exception:
            return set()
        return {str(f.name).strip().lower() for f in table.schema if str(f.name).strip()}

    def replace_predicted_returns(self, rows: list[dict[str, Any]], *, run_date: date | None = None) -> int:
        """Appends one forecast batch tagged with forecast_run_id into predicted_expected_returns."""
        if not rows:
            return 0

        anchor = run_date

        normalized: list[dict[str, Any]] = []
        created_at = utc_now().isoformat()
        for r in rows:
            ticker = str(r.get("ticker", "")).strip().upper()
            model = str(r.get("forecast_model", "")).strip()
            if not ticker or not model:
                continue

            rd_raw = r.get("run_date")
            rd: date | None = None
            if isinstance(rd_raw, date):
                rd = rd_raw
            elif isinstance(rd_raw, datetime):
                rd = rd_raw.date()
            elif rd_raw is not None and str(rd_raw).strip():
                try:
                    rd = datetime.fromisoformat(str(rd_raw).strip().replace("Z", "+00:00")).date()
                except ValueError:
                    rd = None

            if rd is None:
                rd = anchor or utc_now().date()
            if anchor is None:
                anchor = rd
            if rd != anchor:
                continue

            try:
                exp_return = float(r.get("exp_return_period"))
            except (TypeError, ValueError):
                continue

            horizon_val = int(r.get("forecast_horizon", 20))

            score_raw = r.get("forecast_score")
            score_val: float | None = None
            if score_raw is not None:
                try:
                    score_val = float(score_raw)
                except (TypeError, ValueError):
                    score_val = None

            # Classification fields (optional, from stacked.py)
            prob_up_raw = r.get("prob_up")
            prob_up_val: float | None = None
            if prob_up_raw is not None:
                try:
                    prob_up_val = float(prob_up_raw)
                except (TypeError, ValueError):
                    prob_up_val = None

            normalized.append(
                {
                    "run_date": rd.isoformat(),
                    "ticker": ticker,
                    "exp_return_period": exp_return,
                    "forecast_horizon": horizon_val,
                    "forecast_model": model,
                    "is_stacked": bool(r.get("is_stacked", False)),
                    "forecast_score": score_val,
                    "prob_up": prob_up_val,
                    "model_votes_up": int(r["model_votes_up"]) if r.get("model_votes_up") is not None else None,
                    "model_votes_total": int(r["model_votes_total"]) if r.get("model_votes_total") is not None else None,
                    "consensus": str(r.get("consensus") or "").strip() or None,
                    "created_at": created_at,
                }
            )

        if not normalized:
            return 0

        forecast_run_id = "fc_" + uuid.uuid4().hex[:24]
        for row in normalized:
            row["forecast_run_id"] = forecast_run_id

        import io
        import json

        from google.cloud.bigquery import LoadJobConfig, SourceFormat, WriteDisposition

        table_ref = self.session.client.dataset(self.session.dataset).table("predicted_expected_returns")

        job_config = LoadJobConfig(
            source_format=SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=WriteDisposition.WRITE_APPEND,
            autodetect=False,
            schema=self.session.client.get_table(table_ref).schema,
        )
        ndjson = "\n".join(json.dumps(r) for r in normalized).encode()
        load_job = self.session.client.load_table_from_file(
            io.BytesIO(ndjson),
            table_ref,
            job_config=job_config,
        )
        load_job.result()
        return len(normalized)

    def insert_market_features(self, rows: list[dict[str, Any]]) -> None:
        """Appends market feature rows with ingestion metadata for later read-time dedup."""
        if not rows:
            return

        table_id = f"{self.session.dataset_fqn}.market_features"
        ingested_at = utc_now()

        dedup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in rows:
            data = dict(row)
            ts = data.get("as_of_ts")
            dt: datetime | None = None
            if isinstance(ts, datetime):
                dt = ts
            elif isinstance(ts, str) and ts.strip():
                try:
                    dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
                except ValueError:
                    dt = None
            if dt is None:
                dt = ingested_at

            ticker = str(data.get("ticker", "")).strip().upper()
            source = str(data.get("source", "")).strip()
            exchange_code = str(data.get("exchange_code", "")).strip().upper()
            instrument_id = str(data.get("instrument_id", "")).strip()
            if not ticker:
                continue

            data["as_of_ts"] = dt.isoformat()
            data["ingested_at"] = ingested_at.isoformat()
            data["ticker"] = ticker
            data["source"] = source
            data["exchange_code"] = exchange_code
            data["instrument_id"] = instrument_id

            key_ts = dt.date().isoformat() if source in _OPEN_TRADING_DAILY_SOURCES else dt.replace(microsecond=0).isoformat()
            key = (key_ts, ticker, exchange_code, source)
            dedup[key] = data

        if not dedup:
            return

        # Historical and latest reads already deduplicate by timestamp/date and
        # ingestion time.  Appending via load jobs avoids repeated same-day
        # DELETE DML, which fails against BigQuery streaming buffers during
        # market sync reruns/backfills.
        payloads = list(dedup.values())
        self._append_json_rows_via_load_job(table_id, payloads)

    def insert_market_features_latest(self, rows: list[dict[str, Any]]) -> int:
        """Appends compact latest snapshots, relying on read-time dedup by updated_at/as_of_ts."""
        if not rows:
            return 0

        table_id = f"{self.session.dataset_fqn}.market_features_latest"
        now = utc_now()

        dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            data = dict(row)
            ticker = str(data.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            source = str(data.get("source", "")).strip()
            exchange_code = str(data.get("exchange_code", "")).strip().upper()
            instrument_id = str(data.get("instrument_id", "")).strip()

            ts = data.get("as_of_ts")
            as_of_dt: datetime | None = None
            if isinstance(ts, datetime):
                as_of_dt = ts
            elif isinstance(ts, str) and ts.strip():
                try:
                    as_of_dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
                except ValueError:
                    as_of_dt = None
            if as_of_dt is None:
                as_of_dt = now

            key = (ticker, exchange_code, source)
            prev = dedup.get(key)
            prev_ts = None
            if prev:
                p = prev.get("as_of_ts")
                if isinstance(p, datetime):
                    prev_ts = p
                elif isinstance(p, str) and p.strip():
                    try:
                        prev_ts = datetime.fromisoformat(p.strip().replace("Z", "+00:00"))
                    except ValueError:
                        prev_ts = None
            if prev_ts is not None and prev_ts > as_of_dt:
                continue

            dedup[key] = {
                "as_of_ts": as_of_dt.isoformat(),
                "updated_at": now.isoformat(),
                "ticker": ticker,
                "exchange_code": exchange_code,
                "instrument_id": instrument_id,
                "close_price_krw": data.get("close_price_krw"),
                "close_price_native": data.get("close_price_native"),
                "quote_currency": data.get("quote_currency"),
                "fx_rate_used": data.get("fx_rate_used"),
                "ret_5d": data.get("ret_5d"),
                "ret_20d": data.get("ret_20d"),
                "volatility_20d": data.get("volatility_20d"),
                "sentiment_score": data.get("sentiment_score"),
                "source": source,
            }

        if not dedup:
            return 0

        payloads = list(dedup.values())
        return self._append_json_rows_via_load_job(table_id, payloads)

    def insert_fundamentals_snapshot_latest(self, rows: list[dict[str, Any]]) -> int:
        """Appends compact latest fundamental snapshots with read-time dedup."""
        if not rows:
            return 0

        table_id = f"{self.session.dataset_fqn}.fundamentals_snapshot_latest"
        now = utc_now()

        dedup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in rows:
            data = dict(row)
            ticker = str(data.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            market = str(data.get("market", "")).strip().lower()
            exchange_code = str(data.get("exchange_code", "")).strip().upper()
            source = str(data.get("source", "")).strip()
            instrument_id = str(data.get("instrument_id", "")).strip()

            ts = data.get("as_of_ts")
            as_of_dt: datetime | None = None
            if isinstance(ts, datetime):
                as_of_dt = ts
            elif isinstance(ts, str) and ts.strip():
                try:
                    as_of_dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
                except ValueError:
                    as_of_dt = None
            if as_of_dt is None:
                as_of_dt = now

            key = (ticker, market, exchange_code, source)
            prev = dedup.get(key)
            prev_ts = None
            if prev:
                p = prev.get("as_of_ts")
                if isinstance(p, datetime):
                    prev_ts = p
                elif isinstance(p, str) and p.strip():
                    try:
                        prev_ts = datetime.fromisoformat(p.strip().replace("Z", "+00:00"))
                    except ValueError:
                        prev_ts = None
            if prev_ts is not None and prev_ts > as_of_dt:
                continue

            dedup[key] = {
                "as_of_ts": as_of_dt.isoformat(),
                "updated_at": now.isoformat(),
                "ticker": ticker,
                "market": market,
                "exchange_code": exchange_code,
                "instrument_id": instrument_id,
                "currency": data.get("currency"),
                "last_native": data.get("last_native"),
                "per": data.get("per"),
                "pbr": data.get("pbr"),
                "eps": data.get("eps"),
                "bps": data.get("bps"),
                "sps": data.get("sps"),
                "roe": data.get("roe"),
                "debt_ratio": data.get("debt_ratio"),
                "reserve_ratio": data.get("reserve_ratio"),
                "operating_profit_growth": data.get("operating_profit_growth"),
                "net_profit_growth": data.get("net_profit_growth"),
                "source": source,
            }

        if not dedup:
            return 0

        payloads = list(dedup.values())
        return self._append_json_rows_via_load_job(table_id, payloads)

    def refresh_market_features_latest(
        self,
        *,
        tickers: list[str] | None = None,
        sources: list[str] | None = None,
        lookback_days: int = 14,
    ) -> int:
        """Refreshes latest snapshots from historical market_features rows."""
        filters: list[str] = [
            f"as_of_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {max(1, min(int(lookback_days), 180))} DAY)",
        ]
        params: dict[str, Any] = {}

        if tickers:
            tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tokens = list(dict.fromkeys(tokens))
            if tokens:
                filters.append("ticker IN UNNEST(@tickers)")
                params["tickers"] = tokens
        if sources:
            filters.append("source IN UNNEST(@sources)")
            params["sources"] = sources

        where = "WHERE " + " AND ".join(filters)
        sql = f"""
        WITH dedup AS (
          SELECT as_of_ts, ticker, exchange_code, instrument_id, close_price_krw, close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d, volatility_20d, sentiment_score, source, ingested_at,
                 ROW_NUMBER() OVER (PARTITION BY as_of_ts, ticker, exchange_code, source ORDER BY ingested_at DESC) AS rn_key
          FROM `{self.session.dataset_fqn}.market_features`
          {where}
        ), latest AS (
          SELECT as_of_ts, ticker, exchange_code, instrument_id, close_price_krw, close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d, volatility_20d, sentiment_score, source,
                 ROW_NUMBER() OVER (
                   PARTITION BY ticker, IFNULL(exchange_code, ''), IFNULL(source, '')
                   ORDER BY as_of_ts DESC, ingested_at DESC
                 ) AS rn
          FROM dedup
          WHERE rn_key = 1
        )
        SELECT as_of_ts, ticker, exchange_code, instrument_id, close_price_krw, close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d, volatility_20d, sentiment_score, source
        FROM latest
        WHERE rn = 1
        """
        rows = self.session.fetch_rows(sql, params)
        return self.insert_market_features_latest(rows)

    def upsert_instrument_master(self, rows: list[dict[str, Any]]) -> int:
        """Writes instrument metadata snapshots (append-only, latest query resolves final row)."""
        if not rows:
            return 0
        table_id = f"{self.session.dataset_fqn}.instrument_master"
        now = utc_now()
        payloads: list[dict[str, Any]] = []
        row_ids: list[str] = []

        for row in rows:
            ticker = str(row.get("ticker", "")).strip().upper()
            exchange_code = str(row.get("exchange_code", "")).strip().upper()
            instrument_id = str(row.get("instrument_id", "")).strip()
            if not ticker or not exchange_code:
                continue
            if not instrument_id:
                instrument_id = f"{exchange_code}:{ticker}"
            updated_at = row.get("updated_at")
            if isinstance(updated_at, datetime):
                updated_iso = updated_at.isoformat()
            elif isinstance(updated_at, str) and updated_at.strip():
                updated_iso = updated_at.strip()
            else:
                updated_iso = now.isoformat()

            payload = {
                "instrument_id": instrument_id,
                "ticker": ticker,
                "ticker_name": str(row.get("ticker_name") or "").strip() or None,
                "exchange_code": exchange_code,
                "currency": str(row.get("currency", "")).strip().upper() or None,
                "lot_size": int(row.get("lot_size") or 1),
                "tick_size": float(row["tick_size"]) if row.get("tick_size") is not None else None,
                "tradable": bool(row.get("tradable")) if row.get("tradable") is not None else None,
                "status": str(row.get("status", "")).strip() or None,
                "updated_at": updated_iso,
            }
            payloads.append(payload)
            rid_raw = f"{instrument_id}|{updated_iso}".encode("utf-8")
            row_ids.append("im_" + hashlib.sha1(rid_raw).hexdigest()[:24])

        if not payloads:
            return 0
        errors = self.session.client.insert_rows_json(table_id, payloads, row_ids=row_ids)
        if errors:
            raise RuntimeError(f"instrument_master insert failed: {errors}")
        return len(payloads)

    def latest_instrument_master(
        self,
        *,
        tickers: list[str] | None = None,
        exchange_codes: list[str] | None = None,
        limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Loads latest instrument master rows (one row per instrument_id)."""
        lim = max(1, min(int(limit), 10_000))
        filters: list[str] = []
        params: dict[str, Any] = {"limit": lim}
        if tickers is not None:
            tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tokens = list(dict.fromkeys(tokens))
            if not tokens:
                return []
            filters.append("ticker IN UNNEST(@tickers)")
            params["tickers"] = tokens
        if exchange_codes:
            ex = [str(x).strip().upper() for x in exchange_codes if str(x).strip()]
            if ex:
                filters.append("exchange_code IN UNNEST(@exchange_codes)")
                params["exchange_codes"] = ex

        where = "WHERE " + " AND ".join(filters) if filters else ""
        sql = f"""
        WITH latest AS (
          SELECT instrument_id, ticker, ticker_name, exchange_code, currency, lot_size, tick_size, tradable, status, updated_at,
                 ROW_NUMBER() OVER (PARTITION BY instrument_id ORDER BY updated_at DESC) AS rn
          FROM `{self.session.dataset_fqn}.instrument_master`
          {where}
        )
        SELECT instrument_id, ticker, ticker_name, exchange_code, currency, lot_size, tick_size, tradable, status, updated_at
        FROM latest
        WHERE rn = 1
        ORDER BY updated_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    def latest_instrument_map(self, tickers: list[str]) -> dict[str, dict[str, Any]]:
        """Returns ticker->latest instrument metadata map."""
        rows = self.latest_instrument_master(tickers=tickers, limit=max(200, len(tickers) * 4))
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            t = str(row.get("ticker", "")).strip().upper()
            if not t or t in out:
                continue
            out[t] = row
        return out

    def write_universe_candidates(self, run_id: str, rows: list[dict[str, Any]]) -> int:
        """Writes one candidate run into universe_candidates table."""
        run_id = str(run_id).strip()
        if not run_id or not rows:
            return 0
        table_id = f"{self.session.dataset_fqn}.universe_candidates"
        now = utc_now()
        payloads: list[dict[str, Any]] = []
        row_ids: list[str] = []
        for row in rows:
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            exchange_code = str(row.get("exchange_code", "")).strip().upper()
            instrument_id = str(row.get("instrument_id", "")).strip()
            rank = int(row.get("rank") or (len(payloads) + 1))
            score = float(row.get("score") or 0.0)
            created_at = row.get("created_at") or now
            as_of_ts = row.get("as_of_ts")
            ticker_name = str(row.get("ticker_name") or "").strip() or None
            payload = {
                "run_id": run_id,
                "created_at": created_at.isoformat() if isinstance(created_at, datetime) else str(created_at),
                "as_of_ts": as_of_ts.isoformat() if isinstance(as_of_ts, datetime) else as_of_ts,
                "rank": rank,
                "score": score,
                "instrument_id": instrument_id,
                "ticker": ticker,
                "ticker_name": ticker_name,
                "exchange_code": exchange_code,
                "reasons": str(row.get("reasons", "")).strip(),
            }
            payloads.append(payload)
            rid = f"{run_id}|{rank}|{ticker}|{exchange_code}".encode("utf-8")
            row_ids.append("uc_" + hashlib.sha1(rid).hexdigest()[:24])

        if not payloads:
            return 0
        errors = self.session.client.insert_rows_json(table_id, payloads, row_ids=row_ids)
        if errors:
            raise RuntimeError(f"universe_candidates insert failed: {errors}")
        return len(payloads)

    def latest_universe_candidates(self, *, limit: int = 200) -> list[dict[str, Any]]:
        """Loads the most recent universe candidate run."""
        lim = max(1, min(int(limit), 2_000))
        sql = f"""
        WITH last_run AS (
          SELECT run_id, MAX(created_at) AS max_created
          FROM `{self.session.dataset_fqn}.universe_candidates`
          GROUP BY run_id
          ORDER BY max_created DESC
          LIMIT 1
        )
        SELECT run_id, created_at, as_of_ts, rank, score, instrument_id, ticker, ticker_name, exchange_code, reasons
        FROM `{self.session.dataset_fqn}.universe_candidates`
        WHERE run_id = (SELECT run_id FROM last_run)
        ORDER BY rank ASC
        LIMIT @limit
        """
        try:
            return self.session.fetch_rows(sql, {"limit": lim})
        except Exception:
            return []

    def latest_universe_candidate_tickers(self, *, limit: int = 200) -> list[str]:
        """Returns ticker list from latest universe run."""
        rows = self.latest_universe_candidates(limit=limit)
        out: list[str] = []
        for row in rows:
            t = str(row.get("ticker", "")).strip().upper()
            if t and t not in out:
                out.append(t)
        return out

    def ticker_name_map(self, *, tickers: list[str] | None = None, limit: int = 500) -> dict[str, str]:
        """Returns {ticker: ticker_name} from latest universe run with instrument-master fallback."""
        lim = max(1, int(limit))
        tokens = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        rows = self.latest_universe_candidates(limit=max(lim, len(tokens)))
        out: dict[str, str] = {}
        for row in rows:
            t = str(row.get("ticker", "")).strip().upper()
            name = str(row.get("ticker_name") or "").strip()
            if tokens and t not in tokens:
                continue
            if t and name and t not in out:
                out[t] = name
        instrument_limit = max(200, len(tokens) * 4) if tokens else max(200, lim)
        fallback_rows = self.latest_instrument_master(
            tickers=tokens or None,
            limit=instrument_limit,
        )
        for row in fallback_rows:
            t = str(row.get("ticker", "")).strip().upper()
            name = str(row.get("ticker_name") or "").strip()
            if tokens and t not in tokens:
                continue
            if t and name and t not in out:
                out[t] = name
        return out

    def rebuild_universe_candidates(
        self,
        *,
        top_n: int = 400,
        per_exchange_cap: int = 200,
        sources: list[str] | None = None,
        allowed_tickers: list[str] | None = None,
        ticker_names: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Builds a ranked universe from latest snapshots and writes one run."""
        lim = max(50, min(max(top_n * 6, 600), 8_000))
        rows = self.latest_market_features(
            tickers=allowed_tickers or [],
            limit=lim,
            sources=sources,
        )
        if not rows and allowed_tickers is None:
            rows = self.latest_market_features(
                tickers=[],
                limit=lim,
                sources=sources,
            )
        if not rows:
            return {"run_id": "", "count": 0}

        ranked: list[dict[str, Any]] = []
        for row in rows:
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            ret_20d = _finite_float_or_none(row.get("ret_20d"))
            ret_5d = _finite_float_or_none(row.get("ret_5d"))
            vol = _finite_float_or_none(row.get("volatility_20d"))
            if ret_20d is None or ret_5d is None or vol is None:
                continue
            sentiment = _finite_float_or_none(row.get("sentiment_score"))
            if sentiment is None:
                sentiment = 0.0
            quality = max(0.0, 1.0 - min(max(vol, 0.0), 1.5) / 1.5)
            score = (0.60 * ret_20d) + (0.20 * ret_5d) + (0.15 * sentiment) + (0.05 * quality)

            name_map = ticker_names or {}
            ranked.append(
                {
                    "ticker": ticker,
                    "ticker_name": name_map.get(ticker, ""),
                    "exchange_code": str(row.get("exchange_code", "")).strip().upper(),
                    "instrument_id": str(row.get("instrument_id", "")).strip(),
                    "as_of_ts": row.get("as_of_ts"),
                    "score": float(score),
                    "reasons": (
                        f"ret20={ret_20d:+.4f}, ret5={ret_5d:+.4f}, vol20={vol:.4f}, sentiment={sentiment:+.4f}"
                    ),
                }
            )

        ranked.sort(
            key=lambda r: (
                float(r.get("score") or 0.0),
                str(r.get("as_of_ts") or ""),
            ),
            reverse=True,
        )

        cap = max(1, min(int(per_exchange_cap), max(1, int(top_n))))
        counts: dict[str, int] = {}
        selected: list[dict[str, Any]] = []
        for row in ranked:
            ex = str(row.get("exchange_code", "")).strip().upper()
            counts.setdefault(ex, 0)
            if counts[ex] >= cap:
                continue
            counts[ex] += 1
            selected.append(row)
            if len(selected) >= max(1, int(top_n)):
                break

        if not selected:
            return {"run_id": "", "count": 0}

        now = utc_now()
        run_seed = f"{now.isoformat()}|{len(selected)}|{','.join([s['ticker'] for s in selected[:16]])}"
        run_id = "uv_" + hashlib.sha1(run_seed.encode("utf-8")).hexdigest()[:16]

        payloads: list[dict[str, Any]] = []
        for idx, row in enumerate(selected, start=1):
            payloads.append(
                {
                    "created_at": now,
                    "as_of_ts": row.get("as_of_ts"),
                    "rank": idx,
                    "score": row.get("score"),
                    "instrument_id": row.get("instrument_id"),
                    "ticker": row.get("ticker"),
                    "exchange_code": row.get("exchange_code"),
                    "reasons": row.get("reasons"),
                }
            )
        written = self.write_universe_candidates(run_id, payloads)
        return {
            "run_id": run_id,
            "count": written,
            "exchange_counts": counts,
        }

    def recent_market_count(self) -> int:
        """Returns number of market feature rows from last 24 hours."""
        sql = f"""
        SELECT COUNT(1) AS cnt
        FROM `{self.session.dataset_fqn}.market_features`
        WHERE as_of_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
          AND {_DAILY_FILTER_SQL}
        """
        rows = self.session.fetch_rows(sql)
        return int(rows[0]["cnt"]) if rows else 0

    def market_daily_ticker_coverage(self, *, source: str, day: date) -> int:
        """Returns distinct tickers count for a daily source on a given date."""
        sql = f"""
        SELECT COUNT(DISTINCT ticker) AS cnt
        FROM `{self.session.dataset_fqn}.market_features`
        WHERE source = @source
          AND DATE(as_of_ts) = @day
          AND {_DAILY_FILTER_SQL}
        """
        rows = self.session.fetch_rows(sql, {"source": source, "day": day})
        return int(rows[0].get("cnt") or 0) if rows else 0

    def market_source_distinct_tickers(self, *, source: str) -> int:
        """Returns number of distinct tickers present for a given source."""
        sql = f"""
        SELECT COUNT(DISTINCT ticker) AS cnt
        FROM `{self.session.dataset_fqn}.market_features`
        WHERE source = @source
          AND {_DAILY_FILTER_SQL}
        """
        rows = self.session.fetch_rows(sql, {"source": source})
        return int(rows[0].get("cnt") or 0) if rows else 0
