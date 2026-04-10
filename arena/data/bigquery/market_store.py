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
    ) -> "pd.DataFrame":
        """Loads daily close_price_krw into a DataFrame (index=datetime, columns=ticker)."""
        import pandas as pd

        tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return pd.DataFrame()

        filters: list[str] = [
            "ticker IN UNNEST(@tickers)",
            "close_price_krw IS NOT NULL",
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
          SELECT DATE(as_of_ts) AS d, as_of_ts, ticker, close_price_krw, source, ingested_at,
                 ROW_NUMBER() OVER (
                   PARTITION BY DATE(as_of_ts), ticker, source
                   ORDER BY as_of_ts DESC, IFNULL(ingested_at, as_of_ts) DESC
                 ) AS rn_source
          FROM `{self.session.dataset_fqn}.market_features`
          WHERE {where}
        ), dedup_day AS (
          SELECT d, ticker, close_price_krw, as_of_ts, source, ingested_at,
                 ROW_NUMBER() OVER (
                   PARTITION BY d, ticker
                   ORDER BY as_of_ts DESC, IFNULL(ingested_at, as_of_ts) DESC, source DESC
                 ) AS rn_day
          FROM dedup_source
          WHERE rn_source = 1
        )
        SELECT d, ticker, close_price_krw
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
        pivot = df.pivot(index="d", columns="ticker", values="close_price_krw").sort_index()
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
