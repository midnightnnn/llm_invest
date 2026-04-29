"""Local DuckDB-backed market store.

The SQL here is written in DuckDB-native dialect. BigQuery store SQL is not
translated or reused.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import json
import math
import os
import uuid
from typing import Any

from arena.data.local.session import DuckDBSession
from arena.models import utc_now


_FORECAST_MODE_ALIASES: dict[str, tuple[str, ...]] = {
    "all": ("all", "both", "full", "raw", "base+stacked", "stacked+base", "balanced"),
    "stacked": ("stacked", "stack", "meta", "lgbm_stack", "ridge_stack", "ensemble_stack"),
    "lgbm": ("lgbm", "lightgbm", "stacked_lgbm", "lgbm_stack", "meta_lgbm", "stacked_lightgbm"),
    "ridge": ("ridge", "stacked_ridge", "ridge_stack", "meta_ridge"),
    "avg": ("avg", "average", "simple_average", "equal_weight", "ensemble_avg"),
    "base": ("base", "base_model", "base_models"),
}


_SIGNAL_COLUMNS: tuple[str, ...] = (
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
)

_SIGNAL_DAILY_VALUE_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "created_at",
    "ticker",
    "market",
    "exchange_code",
    "instrument_id",
    "source",
    "bucket",
    "profile",
    *_SIGNAL_COLUMNS,
    "ret_5d",
    "ret_20d",
    "volatility_20d",
    "sentiment_score",
    "close_price_krw",
    "fwd_return_20d",
    "fwd_benchmark_return_20d",
    "fwd_excess_return_20d",
    "fwd_mdd_20d",
    "label_ready",
)


class LocalMarketStore:
    """Market feature read/write access against the DuckDB local backend."""

    def __init__(self, session: DuckDBSession) -> None:
        self.session = session

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value if value is not None else {}, ensure_ascii=False, default=str)

    @staticmethod
    def _date_value(value: Any) -> date | None:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return datetime.strptime(text[:10], "%Y-%m-%d").date()
            except ValueError:
                return None

    @staticmethod
    def _datetime_value(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, datetime.min.time())
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if math.isfinite(out) else None

    @staticmethod
    def _normalize_forecast_mode(mode: str | None) -> str:
        token = str(mode or "").strip().lower()
        if not token or token == "auto":
            return "stacked"
        for key, aliases in _FORECAST_MODE_ALIASES.items():
            if token == key or token in aliases:
                return key
        return token

    @staticmethod
    def _forecast_mode_aliases(mode: str) -> tuple[str, ...]:
        token = LocalMarketStore._normalize_forecast_mode(mode)
        return _FORECAST_MODE_ALIASES.get(token, (token,))

    @staticmethod
    def _market_from_ticker(ticker: str) -> str:
        token = str(ticker or "").strip().upper()
        return "kospi" if token.isdigit() and len(token) == 6 else "us"

    @staticmethod
    def _market_matches(ticker: str, market: str | None) -> bool:
        token = str(market or "").strip().lower()
        if not token:
            return True
        inferred = LocalMarketStore._market_from_ticker(ticker)
        if token in {"nasdaq", "nyse", "amex", "us"}:
            return inferred == "us"
        if token in {"kospi", "kosdaq", "kr", "korea"}:
            return inferred == "kospi"
        return True

    @staticmethod
    def _sources_with_fallback(sources: list[str] | None) -> list[str]:
        return [str(source or "").strip() for source in (sources or []) if str(source or "").strip()]

    # ------------------------------------------------------------------
    # Append/upsert helpers used by local seed/backfill
    # ------------------------------------------------------------------

    @staticmethod
    def _market_feature_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "as_of_ts": row.get("as_of_ts"),
            "ingested_at": row.get("ingested_at") or datetime.utcnow(),
            "ticker": str(row.get("ticker") or "").strip().upper(),
            "exchange_code": str(row.get("exchange_code") or "").strip() or None,
            "instrument_id": str(row.get("instrument_id") or "").strip() or None,
            "close_price_krw": row.get("close_price_krw"),
            "close_price_native": row.get("close_price_native"),
            "quote_currency": str(row.get("quote_currency") or "").strip().upper() or None,
            "fx_rate_used": row.get("fx_rate_used"),
            "ret_5d": row.get("ret_5d"),
            "ret_20d": row.get("ret_20d"),
            "volatility_20d": row.get("volatility_20d"),
            "sentiment_score": row.get("sentiment_score"),
            "source": str(row.get("source") or "local").strip(),
        }

    def insert_market_features(self, rows: list[dict[str, Any]]) -> int:
        payload = [self._market_feature_row(row) for row in rows if str(row.get("ticker") or "").strip()]
        return self.session.insert_dicts("market_features", payload)

    def insert_market_features_latest(self, rows: list[dict[str, Any]]) -> int:
        payload = []
        for row in rows:
            base = self._market_feature_row(row)
            base.pop("ingested_at", None)
            base["updated_at"] = row.get("updated_at") or row.get("ingested_at") or datetime.utcnow()
            payload.append(base)
        payload = [row for row in payload if row["ticker"]]
        return self.session.insert_dicts("market_features_latest", payload)

    def upsert_instrument_master(self, rows: list[dict[str, Any]]) -> int:
        payload = []
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            instrument_id = str(row.get("instrument_id") or "").strip() or ticker
            self.session.execute(
                "DELETE FROM instrument_master WHERE instrument_id = $instrument_id",
                {"instrument_id": instrument_id},
            )
            payload.append(
                {
                    "instrument_id": instrument_id,
                    "ticker": ticker,
                    "ticker_name": str(row.get("ticker_name") or row.get("name") or "").strip() or None,
                    "exchange_code": str(row.get("exchange_code") or "").strip() or "LOCAL",
                    "currency": str(row.get("currency") or row.get("quote_currency") or "").strip().upper() or None,
                    "lot_size": row.get("lot_size") if row.get("lot_size") is not None else 1,
                    "tick_size": row.get("tick_size"),
                    "tradable": bool(row.get("tradable", True)),
                    "status": str(row.get("status") or "ACTIVE").strip().upper(),
                    "updated_at": row.get("updated_at") or datetime.utcnow(),
                }
            )
        return self.session.insert_dicts("instrument_master", payload)

    def latest_feature_dates(self, tickers: list[str], source: str) -> dict[str, date]:
        tokens = [str(t or "").strip().upper() for t in (tickers or []) if str(t or "").strip()]
        if not tokens:
            return {}
        rows = self.session.fetch_rows(
            """
            SELECT ticker, MAX(CAST(as_of_ts AS DATE)) AS latest_date
            FROM market_features
            WHERE ticker IN (SELECT unnest($tickers))
              AND source = $source
            GROUP BY ticker
            """,
            {"tickers": tokens, "source": str(source or "").strip()},
        )
        return {
            str(row.get("ticker") or "").strip().upper(): row["latest_date"]
            for row in rows
            if row.get("ticker") and row.get("latest_date")
        }

    def feature_date_spans(self, tickers: list[str], source: str) -> dict[str, dict[str, Any]]:
        tokens = [str(t or "").strip().upper() for t in (tickers or []) if str(t or "").strip()]
        if not tokens:
            return {}
        rows = self.session.fetch_rows(
            """
            SELECT ticker, MIN(CAST(as_of_ts AS DATE)) AS min_date, MAX(CAST(as_of_ts AS DATE)) AS max_date, COUNT(*) AS row_count
            FROM market_features
            WHERE ticker IN (SELECT unnest($tickers))
              AND source = $source
            GROUP BY ticker
            """,
            {"tickers": tokens, "source": str(source or "").strip()},
        )
        return {
            str(row.get("ticker") or "").strip().upper(): {
                "min_date": row.get("min_date"),
                "max_date": row.get("max_date"),
                "row_count": int(row.get("row_count") or 0),
            }
            for row in rows
            if row.get("ticker")
        }

    def distinct_feature_tickers(
        self,
        *,
        sources: list[str],
        lookback_days: int = 14,
    ) -> list[str]:
        srcs = self._sources_with_fallback(sources)
        params: dict[str, Any] = {"days": max(1, int(lookback_days or 1))}
        clauses = [
            "as_of_ts >= (CURRENT_TIMESTAMP - $days * INTERVAL '1 day')",
            "close_price_krw IS NOT NULL",
            "close_price_krw > 0",
        ]
        if srcs:
            clauses.append("source IN (SELECT unnest($sources))")
            params["sources"] = srcs
        rows = self.session.fetch_rows(
            f"""
            SELECT DISTINCT ticker
            FROM market_features
            WHERE {' AND '.join(clauses)}
            ORDER BY ticker
            """,
            params,
        )
        out = [str(r.get("ticker") or "").strip().upper() for r in rows if str(r.get("ticker") or "").strip()]
        if not out and srcs:
            return self.distinct_feature_tickers(sources=[], lookback_days=lookback_days)
        return out

    def latest_universe_candidate_tickers(
        self,
        *,
        limit: int = 200,
        markets: list[str] | None = None,
    ) -> list[str]:
        """Returns a local universe from latest feature rows.

        Local quickstart does not run the BigQuery universe builder by default,
        so the deterministic demo/backfill market data is the universe source.
        """
        lim = max(1, min(int(limit or 200), 5000))
        market_tokens = [str(m or "").strip().lower() for m in (markets or []) if str(m or "").strip()]
        rows = self.session.fetch_rows(
            """
            WITH latest AS (
              SELECT ticker, ret_20d, volatility_20d, close_price_krw, as_of_ts, ingested_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY as_of_ts DESC, ingested_at DESC
                     ) AS rn
              FROM market_features
              WHERE close_price_krw IS NOT NULL AND close_price_krw > 0
            )
            SELECT ticker
            FROM latest
            WHERE rn = 1
            ORDER BY
              CASE WHEN ret_20d IS NULL THEN 1 ELSE 0 END,
              ret_20d DESC NULLS LAST,
              volatility_20d ASC NULLS LAST,
              ticker
            LIMIT $limit
            """,
            {"limit": lim * 2},
        )
        out: list[str] = []
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            if market_tokens and not any(self._market_matches(ticker, market) for market in market_tokens):
                continue
            if ticker not in out:
                out.append(ticker)
            if len(out) >= lim:
                break
        return out

    def market_daily_ticker_coverage(self, *, source: str, day: date) -> int:
        rows = self.session.fetch_rows(
            """
            SELECT COUNT(DISTINCT ticker) AS cnt
            FROM market_features
            WHERE source = $source
              AND CAST(as_of_ts AS DATE) = $day
              AND close_price_krw IS NOT NULL
              AND close_price_krw > 0
            """,
            {"source": str(source or "").strip(), "day": day},
        )
        count = int(rows[0].get("cnt") or 0) if rows else 0
        if count > 0:
            return count
        # Demo/local data intentionally uses source=local_demo. Treat it as
        # acceptable daily coverage when a live source probe is only being used
        # to decide whether local prep has enough data to proceed.
        rows = self.session.fetch_rows(
            """
            SELECT COUNT(DISTINCT ticker) AS cnt
            FROM market_features
            WHERE CAST(as_of_ts AS DATE) = $day
              AND close_price_krw IS NOT NULL
              AND close_price_krw > 0
            """,
            {"day": day},
        )
        return int(rows[0].get("cnt") or 0) if rows else 0

    def market_source_distinct_tickers(self, *, source: str) -> int:
        rows = self.session.fetch_rows(
            """
            SELECT COUNT(DISTINCT ticker) AS cnt
            FROM market_features
            WHERE source = $source
              AND close_price_krw IS NOT NULL
              AND close_price_krw > 0
            """,
            {"source": str(source or "").strip()},
        )
        count = int(rows[0].get("cnt") or 0) if rows else 0
        if count > 0:
            return count
        rows = self.session.fetch_rows(
            """
            SELECT COUNT(DISTINCT ticker) AS cnt
            FROM market_features
            WHERE close_price_krw IS NOT NULL
              AND close_price_krw > 0
            """
        )
        return int(rows[0].get("cnt") or 0) if rows else 0

    def refresh_market_features_latest(
        self,
        *,
        tickers: list[str] | None = None,
        sources: list[str] | None = None,
        lookback_days: int = 30,
    ) -> int:
        _ = lookback_days
        params: dict[str, Any] = {}
        filters: list[str] = []
        tokens = [str(t or "").strip().upper() for t in (tickers or []) if str(t or "").strip()]
        srcs = [str(s or "").strip() for s in (sources or []) if str(s or "").strip()]
        if tokens:
            filters.append("ticker IN (SELECT unnest($tickers))")
            params["tickers"] = tokens
        if srcs:
            filters.append("source IN (SELECT unnest($sources))")
            params["sources"] = srcs
        where = ("WHERE " + " AND ".join(filters)) if filters else ""
        rows = self.session.fetch_rows(
            f"""
            WITH ranked AS (
              SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker, source
                ORDER BY as_of_ts DESC, ingested_at DESC
              ) AS rn
              FROM market_features
              {where}
            )
            SELECT as_of_ts, ingested_at, ticker, exchange_code, instrument_id, close_price_krw,
                   close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d,
                   volatility_20d, sentiment_score, source
            FROM ranked
            WHERE rn = 1
            """,
            params,
        )
        if tokens:
            self.session.execute(
                "DELETE FROM market_features_latest WHERE ticker IN (SELECT unnest($tickers))",
                {"tickers": tokens},
            )
        elif srcs:
            self.session.execute(
                "DELETE FROM market_features_latest WHERE source IN (SELECT unnest($sources))",
                {"sources": srcs},
            )
        else:
            self.session.execute("DELETE FROM market_features_latest")
        return self.insert_market_features_latest(rows)

    # ------------------------------------------------------------------
    # Latest snapshots
    # ------------------------------------------------------------------

    def latest_close_prices(
        self,
        *,
        tickers: list[str],
        sources: list[str] | None = None,
    ) -> dict[str, float]:
        tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        params: dict[str, Any] = {"tickers": tokens}
        source_clause = ""
        if sources:
            source_clause = "AND source IN (SELECT unnest($sources))"
            params["sources"] = list(sources)

        rows = self.session.fetch_rows(
            f"""
            WITH latest AS (
              SELECT ticker, close_price_krw, as_of_ts, updated_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY as_of_ts DESC, updated_at DESC
                     ) AS rn
              FROM market_features_latest
              WHERE ticker IN (SELECT unnest($tickers))
                AND close_price_krw IS NOT NULL
                {source_clause}
            )
            SELECT ticker, close_price_krw
            FROM latest
            WHERE rn = 1
            """,
            params,
        )

        out: dict[str, float] = {}
        for r in rows:
            ticker = str(r.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            try:
                px = float(r.get("close_price_krw") or 0.0)
            except (TypeError, ValueError):
                px = 0.0
            if px > 0:
                out[ticker] = px
        return out

    def latest_close_prices_with_currency(
        self,
        *,
        tickers: list[str],
        sources: list[str] | None = None,
        as_of_date: date | None = None,
    ) -> dict[str, dict[str, Any]]:
        tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        params: dict[str, Any] = {"tickers": tokens}
        extra_clauses = ""
        if sources:
            extra_clauses += " AND source IN (SELECT unnest($sources))"
            params["sources"] = list(sources)
        if as_of_date is not None:
            extra_clauses += " AND CAST(as_of_ts AS DATE) <= $as_of_date"
            params["as_of_date"] = as_of_date

        rows = self.session.fetch_rows(
            f"""
            WITH latest AS (
              SELECT ticker, close_price_krw, close_price_native,
                     quote_currency, fx_rate_used, as_of_ts, updated_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY as_of_ts DESC, updated_at DESC
                     ) AS rn
              FROM market_features_latest
              WHERE ticker IN (SELECT unnest($tickers))
                AND close_price_krw IS NOT NULL
                {extra_clauses}
            )
            SELECT ticker, close_price_krw, close_price_native,
                   quote_currency, fx_rate_used
            FROM latest
            WHERE rn = 1
            """,
            params,
        )

        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            ticker = str(r.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            try:
                px = float(r.get("close_price_krw") or 0.0)
            except (TypeError, ValueError):
                px = 0.0
            if px <= 0:
                continue
            try:
                native = float(r.get("close_price_native")) if r.get("close_price_native") is not None else None
            except (TypeError, ValueError):
                native = None
            out[ticker] = {
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
        lim = max(1, min(int(limit), 8_000))
        params: dict[str, Any] = {"limit": lim}
        clauses: list[str] = []

        if tickers:
            tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
            tokens = list(dict.fromkeys(tokens))
            if not tokens:
                return []
            clauses.append("ticker IN (SELECT unnest($tickers))")
            params["tickers"] = tokens

        srcs = self._sources_with_fallback(sources)
        if srcs:
            clauses.append("source IN (SELECT unnest($sources))")
            params["sources"] = srcs

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.session.fetch_rows(
            f"""
            WITH latest AS (
              SELECT as_of_ts, ticker, exchange_code, instrument_id,
                     close_price_krw, close_price_native, quote_currency,
                     fx_rate_used, ret_5d, ret_20d, volatility_20d,
                     sentiment_score, source, updated_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY
                         CASE
                           WHEN ret_5d IS NOT NULL
                            AND ret_20d IS NOT NULL
                            AND volatility_20d IS NOT NULL
                           THEN 0 ELSE 1
                         END,
                         as_of_ts DESC,
                         updated_at DESC
                     ) AS rn
              FROM market_features_latest
              {where}
            )
            SELECT as_of_ts, ticker, exchange_code, instrument_id,
                   close_price_krw, close_price_native, quote_currency,
                   fx_rate_used, ret_5d, ret_20d, volatility_20d,
                   sentiment_score, source
            FROM latest
            WHERE rn = 1
            ORDER BY as_of_ts DESC
            LIMIT $limit
            """,
            params,
        )
        if not rows and srcs:
            return self.latest_market_features(tickers=tickers, limit=limit, sources=None)
        return rows

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
        field = str(sort_by or "").strip()
        if field not in allowed_fields:
            field = "ret_20d"
        direction = "asc" if str(order or "").strip().lower() == "asc" else "desc"
        params: dict[str, Any] = {"limit": max(1, min(int(top_n or 10), 500))}
        filters = [
            "ret_5d IS NOT NULL",
            "ret_20d IS NOT NULL",
            "volatility_20d IS NOT NULL",
        ]
        if tickers is not None:
            tokens = [str(t or "").strip().upper() for t in tickers if str(t or "").strip()]
            tokens = list(dict.fromkeys(tokens))
            if not tokens:
                return []
            filters.append("ticker IN (SELECT unnest($tickers))")
            params["tickers"] = tokens
        srcs = self._sources_with_fallback(sources)
        if srcs:
            filters.append("source IN (SELECT unnest($sources))")
            params["sources"] = srcs
        if min_ret_20d is not None:
            filters.append("ret_20d >= $min_ret_20d")
            params["min_ret_20d"] = float(min_ret_20d)
        if max_volatility is not None:
            filters.append("volatility_20d <= $max_volatility")
            params["max_volatility"] = float(max_volatility)
        rows = self.session.fetch_rows(
            f"""
            WITH latest AS (
              SELECT as_of_ts, ticker, exchange_code, instrument_id,
                     close_price_krw, close_price_native, quote_currency, fx_rate_used,
                     ret_5d, ret_20d, volatility_20d, sentiment_score, source, updated_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY as_of_ts DESC, updated_at DESC
                     ) AS rn
              FROM market_features_latest
              WHERE {' AND '.join(filters)}
            )
            SELECT as_of_ts, ticker, exchange_code, instrument_id,
                   close_price_krw, close_price_native, quote_currency, fx_rate_used,
                   ret_5d, ret_20d, volatility_20d, sentiment_score, source
            FROM latest
            WHERE rn = 1
            ORDER BY {field} {direction}
            LIMIT $limit
            """,
            params,
        )
        if not rows and srcs:
            return self.screen_latest_features(
                sort_by=sort_by,
                order=order,
                tickers=tickers,
                min_ret_20d=min_ret_20d,
                max_volatility=max_volatility,
                top_n=top_n,
                sources=None,
            )
        return rows

    def get_daily_closes(
        self,
        *,
        tickers: list[str],
        lookback_days: int,
        sources: list[str] | None = None,
    ) -> dict[str, list[float]]:
        tokens = [str(t or "").strip().upper() for t in tickers if str(t or "").strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}
        limit = max(2, min(int(lookback_days or 1), 400))
        params: dict[str, Any] = {"tickers": tokens, "limit": limit}
        filters = [
            "ticker IN (SELECT unnest($tickers))",
            "close_price_krw IS NOT NULL",
            "close_price_krw > 0",
        ]
        srcs = self._sources_with_fallback(sources)
        if srcs:
            filters.append("source IN (SELECT unnest($sources))")
            params["sources"] = srcs
        rows = self.session.fetch_rows(
            f"""
            WITH dedup AS (
              SELECT CAST(as_of_ts AS DATE) AS d, as_of_ts, ticker, close_price_krw, source, ingested_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY CAST(as_of_ts AS DATE), ticker
                       ORDER BY as_of_ts DESC, ingested_at DESC, source DESC
                     ) AS rn_key
              FROM market_features
              WHERE {' AND '.join(filters)}
            ), ranked AS (
              SELECT d, ticker, close_price_krw,
                     ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY d DESC) AS rn
              FROM dedup
              WHERE rn_key = 1
            )
            SELECT d, ticker, close_price_krw
            FROM ranked
            WHERE rn <= $limit
            ORDER BY ticker, d ASC
            """,
            params,
        )
        if not rows and srcs:
            return self.get_daily_closes(tickers=tokens, lookback_days=lookback_days, sources=None)
        out: dict[str, list[float]] = {}
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            px = self._finite_float(row.get("close_price_krw"))
            if ticker and px is not None and px > 0:
                out.setdefault(ticker, []).append(px)
        return out

    def get_daily_close_frame(
        self,
        *,
        tickers: list[str],
        start: date,
        end: date,
        sources: list[str] | None = None,
        price_field: str = "close_price_krw",
    ):
        import pandas as pd

        tokens = [str(t or "").strip().upper() for t in tickers if str(t or "").strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return pd.DataFrame()
        price_column = str(price_field or "close_price_krw").strip()
        if price_column not in {"close_price_krw", "close_price_native"}:
            raise ValueError("price_field must be close_price_krw or close_price_native")
        params: dict[str, Any] = {"tickers": tokens, "start": start, "end": end}
        filters = [
            "ticker IN (SELECT unnest($tickers))",
            f"{price_column} IS NOT NULL",
            f"{price_column} > 0",
            "CAST(as_of_ts AS DATE) >= $start",
            "CAST(as_of_ts AS DATE) <= $end",
        ]
        srcs = self._sources_with_fallback(sources)
        if srcs:
            filters.append("source IN (SELECT unnest($sources))")
            params["sources"] = srcs
        rows = self.session.fetch_rows(
            f"""
            WITH dedup AS (
              SELECT CAST(as_of_ts AS DATE) AS d, ticker, {price_column} AS close_price,
                     as_of_ts, source, ingested_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY CAST(as_of_ts AS DATE), ticker
                       ORDER BY as_of_ts DESC, ingested_at DESC, source DESC
                     ) AS rn
              FROM market_features
              WHERE {' AND '.join(filters)}
            )
            SELECT d, ticker, close_price
            FROM dedup
            WHERE rn = 1
            ORDER BY d ASC, ticker ASC
            """,
            params,
        )
        if not rows and srcs:
            return self.get_daily_close_frame(
                tickers=tokens,
                start=start,
                end=end,
                sources=None,
                price_field=price_field,
            )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame()
        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["d"] = pd.to_datetime(df["d"])
        df = df.sort_values(["d", "ticker"]).drop_duplicates(subset=["d", "ticker"], keep="last")
        return df.pivot(index="d", columns="ticker", values="close_price").sort_index()

    def replace_predicted_returns(self, rows: list[dict[str, Any]], *, run_date: date | None = None) -> int:
        """Appends one forecast batch into ``predicted_expected_returns``."""
        if not rows:
            return 0
        anchor = run_date
        created_at = utc_now()
        forecast_run_id = "fc_" + uuid.uuid4().hex[:24]
        payload: list[dict[str, Any]] = []
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            model = str(row.get("forecast_model") or "").strip()
            if not ticker or not model:
                continue
            rd = self._date_value(row.get("run_date")) or anchor or created_at.date()
            if anchor is None:
                anchor = rd
            if rd != anchor:
                continue
            exp_return = self._finite_float(row.get("exp_return_period"))
            if exp_return is None:
                continue
            payload.append(
                {
                    "run_date": rd,
                    "forecast_run_id": forecast_run_id,
                    "ticker": ticker,
                    "exp_return_period": exp_return,
                    "forecast_horizon": int(row.get("forecast_horizon") or 20),
                    "forecast_model": model,
                    "is_stacked": bool(row.get("is_stacked", False)),
                    "forecast_score": self._finite_float(row.get("forecast_score")),
                    "prob_up": self._finite_float(row.get("prob_up")),
                    "model_votes_up": int(row["model_votes_up"]) if row.get("model_votes_up") is not None else None,
                    "model_votes_total": int(row["model_votes_total"]) if row.get("model_votes_total") is not None else None,
                    "consensus": str(row.get("consensus") or "").strip() or None,
                    "created_at": created_at,
                }
            )
        return self.session.insert_dicts("predicted_expected_returns", payload)

    def get_predicted_returns(
        self,
        tickers: list[str] | None = None,
        limit: int = 50,
        mode: str = "stacked",
        table_id: str | None = None,
        staleness_days: int | None = None,
    ) -> list[dict[str, Any]]:
        _ = table_id
        lim = max(1, min(int(limit or 50), 500))
        if staleness_days is None:
            try:
                staleness_days = int(os.getenv("ARENA_FORECAST_STALENESS_DAYS", "5") or "5")
            except ValueError:
                staleness_days = 5
        stale = max(0, int(staleness_days or 0))
        params: dict[str, Any] = {"limit": lim}
        batch_filters: list[str] = []
        row_filters: list[str] = []
        if tickers:
            tokens = [str(t or "").strip().upper() for t in tickers if str(t or "").strip()]
            tokens = list(dict.fromkeys(tokens))
            if tokens:
                batch_filters.append("ticker IN (SELECT unnest($tickers))")
                row_filters.append("r.ticker IN (SELECT unnest($tickers))")
                params["tickers"] = tokens
        if stale > 0:
            batch_filters.append("run_date >= $min_run_date")
            row_filters.append("r.run_date >= $min_run_date")
            params["min_run_date"] = utc_now().date() - timedelta(days=stale)

        mode_norm = self._normalize_forecast_mode(mode)
        if mode_norm == "all":
            pass
        elif mode_norm == "stacked":
            batch_filters.append("COALESCE(is_stacked, FALSE)")
            row_filters.append("COALESCE(r.is_stacked, FALSE)")
        elif mode_norm == "base":
            batch_filters.append("NOT COALESCE(is_stacked, FALSE)")
            row_filters.append("NOT COALESCE(r.is_stacked, FALSE)")
        else:
            batch_filters.append("LOWER(forecast_model) IN (SELECT unnest($forecast_modes))")
            row_filters.append("LOWER(r.forecast_model) IN (SELECT unnest($forecast_modes))")
            params["forecast_modes"] = list(self._forecast_mode_aliases(mode_norm))

        batch_where = ("WHERE " + " AND ".join(batch_filters)) if batch_filters else ""
        row_where = ("AND " + " AND ".join(row_filters)) if row_filters else ""
        rows = self.session.fetch_rows(
            f"""
            WITH latest_date AS (
              SELECT MAX(run_date) AS run_date
              FROM predicted_expected_returns
              {batch_where}
            ),
            latest_batch AS (
              SELECT forecast_run_id, created_at
              FROM predicted_expected_returns
              WHERE run_date = (SELECT run_date FROM latest_date)
              ORDER BY created_at DESC, forecast_run_id DESC
              LIMIT 1
            )
            SELECT r.run_date, r.ticker, r.exp_return_period, r.forecast_horizon,
                   r.forecast_model, r.is_stacked, r.forecast_score, r.prob_up,
                   r.model_votes_up, r.model_votes_total, r.consensus
            FROM predicted_expected_returns r
            JOIN latest_date d ON r.run_date = d.run_date
            JOIN latest_batch b
              ON r.forecast_run_id = b.forecast_run_id
             AND r.created_at = b.created_at
            {row_where}
            ORDER BY r.exp_return_period DESC
            LIMIT $limit
            """,
            params,
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            exp_return = self._finite_float(row.get("exp_return_period"))
            if not ticker or exp_return is None:
                continue
            item: dict[str, Any] = {
                "run_date": self._date_value(row.get("run_date")).isoformat() if self._date_value(row.get("run_date")) else str(row.get("run_date")),
                "ticker": ticker,
                "exp_return_period": exp_return,
            }
            for key in ("forecast_model", "consensus"):
                if row.get(key) is not None:
                    item[key] = str(row.get(key) or "")
            for key in ("forecast_horizon", "model_votes_up", "model_votes_total"):
                if row.get(key) is not None:
                    item[key] = int(row[key])
            if row.get("is_stacked") is not None:
                item["is_stacked"] = bool(row.get("is_stacked"))
            for key in ("forecast_score", "prob_up"):
                val = self._finite_float(row.get(key))
                if val is not None:
                    item[key] = val
            out.append(item)
        return out

    @staticmethod
    def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in record.items():
            try:
                import pandas as pd

                if pd.isna(value):
                    value = None
            except Exception:
                pass
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            if str(type(value)).startswith("<class 'pandas.") and hasattr(value, "to_pydatetime"):
                value = value.to_pydatetime()
            out[key] = value
        return out

    @staticmethod
    def _zscore_by_date(frame, column: str):
        import numpy as np

        grouped = frame.groupby("as_of_date")[column]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        return (frame[column] - mean) / std

    def _signal_daily_values_frame(self, df: Any, *, market_key: str | None, created_at: datetime) -> Any:
        """Builds a DuckDB-ready signal_daily_values frame without Python row loops."""
        import numpy as np
        import pandas as pd

        out = pd.DataFrame(index=df.index)
        out["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
        out["created_at"] = created_at.replace(tzinfo=None) if created_at.tzinfo is not None else created_at
        out["ticker"] = df["ticker"].astype(str).str.upper()
        if market_key:
            out["market"] = market_key
        else:
            out["market"] = out["ticker"].map(self._market_from_ticker)

        for col in ("exchange_code", "instrument_id", "source"):
            if col in df:
                out[col] = df[col].astype("object").where(pd.notna(df[col]), None)
            else:
                out[col] = None
        out["bucket"] = df["bucket"].astype(str)
        out["profile"] = df["profile"].astype(str)

        numeric_cols = (
            *_SIGNAL_COLUMNS,
            "ret_5d",
            "ret_20d",
            "volatility_20d",
            "sentiment_score",
            "close_price_krw",
            "fwd_return_20d",
            "fwd_benchmark_return_20d",
            "fwd_excess_return_20d",
            "fwd_mdd_20d",
        )
        for col in numeric_cols:
            if col in df:
                out[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            else:
                out[col] = np.nan
        out["label_ready"] = df["label_ready"].fillna(False).astype(bool)
        return out.loc[:, list(_SIGNAL_DAILY_VALUE_COLUMNS)]

    def refresh_signal_daily_values(
        self,
        *,
        lookback_days: int = 540,
        horizon_days: int = 20,
        sources: list[str] | None = None,
        market: str | None = None,
    ) -> int:
        """Materializes ranker signal rows from local market/forecast data."""
        import numpy as np
        import pandas as pd

        horizon = max(5, min(int(horizon_days or 20), 60))
        lookback = max(horizon + 40, min(int(lookback_days or 540), 1500))
        start = utc_now().date() - timedelta(days=lookback + horizon + 30)
        params: dict[str, Any] = {"start": start}
        filters = [
            "CAST(as_of_ts AS DATE) >= $start",
            "close_price_krw IS NOT NULL",
            "close_price_krw > 0",
        ]
        srcs = self._sources_with_fallback(sources)
        if srcs:
            filters.append("source IN (SELECT unnest($sources))")
            params["sources"] = srcs
        rows = self.session.fetch_rows(
            f"""
            WITH daily AS (
              SELECT CAST(as_of_ts AS DATE) AS as_of_date, as_of_ts, ticker,
                     exchange_code, instrument_id, source, close_price_krw,
                     sentiment_score, ingested_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY CAST(as_of_ts AS DATE), ticker
                       ORDER BY as_of_ts DESC, ingested_at DESC, source DESC
                     ) AS rn
              FROM market_features
              WHERE {' AND '.join(filters)}
            )
            SELECT as_of_date, as_of_ts, ticker, exchange_code, instrument_id,
                   source, close_price_krw, sentiment_score
            FROM daily
            WHERE rn = 1
            ORDER BY ticker, as_of_date
            """,
            params,
        )
        if not rows and srcs:
            return self.refresh_signal_daily_values(
                lookback_days=lookback_days,
                horizon_days=horizon_days,
                sources=None,
                market=market,
            )
        if not rows:
            return 0

        df = pd.DataFrame(rows)
        df["ticker"] = df["ticker"].astype(str).str.upper()
        market_key = str(market or "").strip().lower() or None
        if market_key:
            df = df[df["ticker"].map(lambda t: self._market_matches(t, market_key))]
        if df.empty:
            return 0
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
        df["close_price_krw"] = pd.to_numeric(df["close_price_krw"], errors="coerce")
        df["sentiment_score"] = pd.to_numeric(df.get("sentiment_score"), errors="coerce").fillna(0.0)
        df = df.sort_values(["ticker", "as_of_date"]).reset_index(drop=True)
        grouped = df.groupby("ticker", group_keys=False)
        df["daily_return"] = grouped["close_price_krw"].pct_change()
        df["ret_5d"] = grouped["close_price_krw"].pct_change(5)
        df["ret_20d"] = grouped["close_price_krw"].pct_change(20)
        df["sma_20"] = grouped["close_price_krw"].transform(lambda s: s.rolling(20, min_periods=10).mean())
        df["sma_60"] = grouped["close_price_krw"].transform(lambda s: s.rolling(60, min_periods=30).mean())
        df["std_px_20"] = grouped["close_price_krw"].transform(lambda s: s.rolling(20, min_periods=10).std())
        df["volatility_20d"] = grouped["daily_return"].transform(lambda s: s.rolling(20, min_periods=10).std())
        up = df["daily_return"].clip(lower=0)
        down = (-df["daily_return"]).clip(lower=0)
        df["rsi_up_14"] = up.groupby(df["ticker"]).transform(lambda s: s.rolling(14, min_periods=7).mean())
        df["rsi_dn_14"] = down.groupby(df["ticker"]).transform(lambda s: s.rolling(14, min_periods=7).mean())
        denom = df["rsi_up_14"] + df["rsi_dn_14"]
        df["rsi_14"] = np.where(denom == 0, 50.0, 100.0 * df["rsi_up_14"] / denom)
        df["fwd_return_20d"] = grouped["close_price_krw"].shift(-horizon) / df["close_price_krw"] - 1.0
        df["fwd_min_price"] = grouped["close_price_krw"].transform(
            lambda s: s.shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
        )
        df["fwd_mdd_20d"] = df["fwd_min_price"] / df["close_price_krw"] - 1.0
        df = df.dropna(subset=["ret_5d", "ret_20d", "volatility_20d"]).copy()
        if df.empty:
            return 0

        df["z_ret20"] = self._zscore_by_date(df, "ret_20d")
        df["z_ret5"] = self._zscore_by_date(df, "ret_5d")
        df["z_vol"] = self._zscore_by_date(df, "volatility_20d")
        df["z_sent"] = self._zscore_by_date(df, "sentiment_score")

        forecasts = self.session.fetch_rows(
            """
            SELECT run_date, ticker, exp_return_period, prob_up, created_at
            FROM predicted_expected_returns
            WHERE COALESCE(is_stacked, FALSE)
            ORDER BY ticker, run_date, created_at
            """
        )
        df["signal_forecast_er"] = None
        df["signal_forecast_prob"] = None
        if forecasts:
            fdf = pd.DataFrame(forecasts)
            fdf["ticker"] = fdf["ticker"].astype(str).str.upper()
            fdf["run_date"] = pd.to_datetime(fdf["run_date"]).dt.date
            fdf = fdf.sort_values(["ticker", "run_date", "created_at"]).drop_duplicates(["ticker", "run_date"], keep="last")
            merged_parts = []
            for ticker, part in df.groupby("ticker"):
                fpart = fdf[fdf["ticker"] == ticker]
                if fpart.empty:
                    merged_parts.append(part)
                    continue
                left = part.sort_values("as_of_date").copy()
                right = fpart.sort_values("run_date")[["run_date", "exp_return_period", "prob_up"]].copy()
                left["_asof_ts"] = pd.to_datetime(left["as_of_date"])
                right["_run_ts"] = pd.to_datetime(right["run_date"])
                left["_merge_order"] = range(len(left))
                merged = pd.merge_asof(
                    left,
                    right,
                    left_on="_asof_ts",
                    right_on="_run_ts",
                    direction="backward",
                ).sort_values("_merge_order")
                merged["signal_forecast_er"] = merged["exp_return_period"]
                merged["signal_forecast_prob"] = pd.to_numeric(merged["prob_up"], errors="coerce") - 0.5
                merged_parts.append(
                    merged.drop(
                        columns=[
                            c
                            for c in ("_merge_order", "_asof_ts", "_run_ts", "run_date", "exp_return_period", "prob_up")
                            if c in merged.columns
                        ]
                    )
                )
            df = pd.concat(merged_parts, ignore_index=True)

        df["fwd_benchmark_return_20d"] = df.groupby("as_of_date")["fwd_return_20d"].transform("mean")
        df["fwd_excess_return_20d"] = df["fwd_return_20d"] - df["fwd_benchmark_return_20d"]
        df["signal_momentum_20d"] = df["z_ret20"]
        df["signal_pullback"] = np.where(df["z_ret20"] > 0, -df["z_ret5"], 0.0)
        df["signal_meanrev_5d"] = -df["z_ret5"]
        df["signal_lowvol"] = -df["z_vol"]
        df["signal_sentiment"] = df["z_sent"]
        df["signal_rsi_reversal"] = np.select([df["rsi_14"] < 30, df["rsi_14"] > 70], [1.0, -1.0], default=0.0)
        df["signal_ma_crossover"] = np.where(df["sma_20"] > df["sma_60"], 1.0, -1.0)
        df["signal_bollinger_position"] = (df["close_price_krw"] - df["sma_20"]) / (2.0 * df["std_px_20"].replace(0, np.nan))
        for col in (
            "signal_ep",
            "signal_bp",
            "signal_sp",
            "signal_roe",
            "signal_revenue_growth",
            "signal_eps_growth",
            "signal_low_debt",
        ):
            df[col] = None
        df["bucket"] = np.select(
            [
                (df["ret_20d"] > 0) & (df["ret_5d"] < 0),
                (df["ret_20d"] < 0) & (df["ret_5d"] > 0),
                df["z_vol"] <= -0.65,
            ],
            ["pullback", "recovery", "defensive"],
            default="momentum",
        )
        df["profile"] = np.select(
            [df["bucket"].isin(["momentum", "recovery"]), df["bucket"].eq("pullback"), df["bucket"].eq("defensive")],
            ["aggressive", "balanced", "defensive"],
            default="balanced",
        )
        cutoff = utc_now().date() - timedelta(days=horizon)
        min_keep = utc_now().date() - timedelta(days=lookback)
        df = df[df["as_of_date"] >= min_keep].copy()
        df["label_ready"] = df["fwd_return_20d"].notna() & (df["as_of_date"] <= cutoff)

        self.session.execute("DELETE FROM signal_daily_values WHERE as_of_date >= $min_keep", {"min_keep": min_keep})
        insert_frame = self._signal_daily_values_frame(df, market_key=market_key, created_at=utc_now())
        return self.session.insert_dataframe(
            "signal_daily_values",
            insert_frame,
            columns=_SIGNAL_DAILY_VALUE_COLUMNS,
        )

    def refresh_signal_daily_ic(
        self,
        *,
        lookback_days: int = 540,
        horizon_days: int = 20,
        market: str | None = None,
    ) -> int:
        import pandas as pd

        lookback = max(40, min(int(lookback_days or 540), 1500))
        horizon = max(5, min(int(horizon_days or 20), 60))
        start = utc_now().date() - timedelta(days=lookback)
        market_key = str(market or "").strip().lower() or None
        params: dict[str, Any] = {"start": start}
        where = "as_of_date >= $start AND label_ready"
        if market_key:
            where += " AND (market = $market OR market IS NULL)"
            params["market"] = market_key
        rows = self.session.fetch_rows(f"SELECT * FROM signal_daily_values WHERE {where}", params)
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        out: list[dict[str, Any]] = []
        now = utc_now()
        for as_of_date, day in df.groupby("as_of_date"):
            y = pd.to_numeric(day.get("fwd_excess_return_20d"), errors="coerce")
            for col in _SIGNAL_COLUMNS:
                x = pd.to_numeric(day.get(col), errors="coerce")
                valid = x.notna() & y.notna()
                sample = int(valid.sum())
                if sample < 3 or x[valid].nunique(dropna=True) < 2 or y[valid].nunique(dropna=True) < 2:
                    ic = None
                    rank_ic = None
                else:
                    ic_val = float(x[valid].corr(y[valid]))
                    rank_val = float(x[valid].rank(pct=True).corr(y[valid].rank(pct=True)))
                    ic = ic_val if math.isfinite(ic_val) else None
                    rank_ic = rank_val if math.isfinite(rank_val) else None
                out.append(
                    {
                        "as_of_date": self._date_value(as_of_date),
                        "created_at": now,
                        "signal_name": col.removeprefix("signal_"),
                        "horizon_days": horizon,
                        "ic_20d": ic,
                        "rank_ic_20d": rank_ic,
                        "sample_size": sample,
                        "market": market_key,
                    }
                )
        delete_params: dict[str, Any] = {"start": start}
        delete_sql = "DELETE FROM signal_daily_ic WHERE as_of_date >= $start"
        if market_key:
            delete_sql += " AND (market = $market OR market IS NULL)"
            delete_params["market"] = market_key
        self.session.execute(delete_sql, delete_params)
        return self.session.insert_dicts("signal_daily_ic", out)

    def refresh_regime_daily_features(
        self,
        *,
        lookback_days: int = 540,
        market: str | None = None,
    ) -> int:
        import pandas as pd

        lookback = max(40, min(int(lookback_days or 540), 1500))
        start = utc_now().date() - timedelta(days=lookback)
        market_key = str(market or "").strip().lower() or None
        params: dict[str, Any] = {"start": start}
        where = "as_of_date >= $start"
        if market_key:
            where += " AND (market = $market OR market IS NULL)"
            params["market"] = market_key
        rows = self.session.fetch_rows(f"SELECT * FROM signal_daily_values WHERE {where}", params)
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        now = utc_now()
        out: list[dict[str, Any]] = []
        for as_of_date, day in df.groupby("as_of_date"):
            out.append(
                {
                    "as_of_date": self._date_value(as_of_date),
                    "created_at": now,
                    "market": market_key,
                    "regime_vol_level": self._finite_float(day["volatility_20d"].median()),
                    "regime_vol_dispersion": self._finite_float(day["volatility_20d"].std()),
                    "regime_trend": self._finite_float(day["ret_20d"].mean()),
                    "regime_short_reversal": self._finite_float(day["ret_5d"].mean()),
                    "regime_dispersion": self._finite_float(day["ret_20d"].std()),
                    "regime_sentiment": self._finite_float(day["sentiment_score"].fillna(0.0).mean()),
                    "sample_size": int(len(day)),
                }
            )
        delete_params: dict[str, Any] = {"start": start}
        delete_sql = "DELETE FROM regime_daily_features WHERE as_of_date >= $start"
        if market_key:
            delete_sql += " AND (market = $market OR market IS NULL)"
            delete_params["market"] = market_key
        self.session.execute(delete_sql, delete_params)
        return self.session.insert_dicts("regime_daily_features", out)

    def load_signal_daily_ic(
        self,
        *,
        lookback_days: int = 540,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        start = utc_now().date() - timedelta(days=max(40, min(int(lookback_days or 540), 1500)))
        market_key = str(market or "").strip().lower() or None
        params: dict[str, Any] = {"start": start}
        where = "as_of_date >= $start"
        if market_key:
            where += " AND (market = $market OR market IS NULL)"
            params["market"] = market_key
        return self.session.fetch_rows(
            f"""
            WITH ranked AS (
              SELECT *, ROW_NUMBER() OVER (
                PARTITION BY as_of_date, signal_name
                ORDER BY created_at DESC
              ) AS rn
              FROM signal_daily_ic
              WHERE {where}
            )
            SELECT * EXCLUDE (rn)
            FROM ranked
            WHERE rn = 1
            ORDER BY as_of_date, signal_name
            """,
            params,
        )

    def load_regime_daily_features(
        self,
        *,
        lookback_days: int = 540,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        start = utc_now().date() - timedelta(days=max(40, min(int(lookback_days or 540), 1500)))
        market_key = str(market or "").strip().lower() or None
        params: dict[str, Any] = {"start": start}
        where = "as_of_date >= $start"
        if market_key:
            where += " AND (market = $market OR market IS NULL)"
            params["market"] = market_key
        return self.session.fetch_rows(
            f"""
            WITH ranked AS (
              SELECT *, ROW_NUMBER() OVER (
                PARTITION BY as_of_date
                ORDER BY created_at DESC
              ) AS rn
              FROM regime_daily_features
              WHERE {where}
            )
            SELECT * EXCLUDE (rn)
            FROM ranked
            WHERE rn = 1
            ORDER BY as_of_date
            """,
            params,
        )

    def load_signal_scoring_rows(
        self,
        *,
        limit: int = 500,
        market: str | None = None,
    ) -> list[dict[str, Any]]:
        market_key = str(market or "").strip().lower() or None
        params: dict[str, Any] = {"limit": max(1, min(int(limit or 500), 5000))}
        where = "1 = 1"
        if market_key:
            where += " AND (market = $market OR market IS NULL)"
            params["market"] = market_key
        return self.session.fetch_rows(
            f"""
            WITH ranked AS (
              SELECT *, ROW_NUMBER() OVER (
                PARTITION BY as_of_date, ticker
                ORDER BY created_at DESC
              ) AS rn
              FROM signal_daily_values
              WHERE {where}
            ),
            latest AS (
              SELECT MAX(as_of_date) AS as_of_date
              FROM ranked
              WHERE rn = 1
            )
            SELECT r.* EXCLUDE (rn)
            FROM ranked r
            JOIN latest l ON r.as_of_date = l.as_of_date
            WHERE r.rn = 1
            ORDER BY r.signal_momentum_20d DESC NULLS LAST, r.ticker
            LIMIT $limit
            """,
            params,
        )

    def refresh_fundamentals_derived_daily(
        self,
        *,
        lookback_days: int = 600,
        market: str | None = None,
    ) -> int:
        """No-op local hook for prep parity.

        Local demo/backfill can run without a fundamentals vendor feed. The
        ranker already treats missing fundamentals as nullable signals.
        """
        _ = (lookback_days, market)
        return 0

    def insert_opportunity_ranker_scores_latest(self, rows: list[dict[str, Any]]) -> int:
        payload: list[dict[str, Any]] = []
        for row in rows or []:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            payload.append(
                {
                    "as_of_date": self._date_value(row.get("as_of_date")) or utc_now().date(),
                    "computed_at": self._datetime_value(row.get("computed_at")) or utc_now(),
                    "ranker_version": str(row.get("ranker_version") or "").strip(),
                    "score_source": str(row.get("score_source") or "").strip() or "learned_ic",
                    "ticker": ticker,
                    "market": str(row.get("market") or "").strip().lower() or None,
                    "exchange_code": str(row.get("exchange_code") or "").strip() or None,
                    "instrument_id": str(row.get("instrument_id") or "").strip() or None,
                    "source": str(row.get("source") or "").strip() or None,
                    "profile": str(row.get("profile") or "").strip().lower() or None,
                    "bucket": str(row.get("bucket") or "").strip().lower() or None,
                    "recommendation_rank": int(row["recommendation_rank"]) if row.get("recommendation_rank") is not None else None,
                    "recommendation_score": self._finite_float(row.get("recommendation_score")),
                    "predicted_excess_return_20d": self._finite_float(row.get("predicted_excess_return_20d")),
                    "prob_outperform_20d": self._finite_float(row.get("prob_outperform_20d")),
                    "predicted_drawdown_20d": self._finite_float(row.get("predicted_drawdown_20d")),
                    "model_confidence": str(row.get("model_confidence") or "").strip() or None,
                    "action": str(row.get("action") or "").strip() or None,
                    "evidence_level": str(row.get("evidence_level") or "").strip() or None,
                    "optimizer_weight": self._finite_float(row.get("optimizer_weight")),
                    "optimizer_raw_weight": self._finite_float(row.get("optimizer_raw_weight")),
                    "feature_json": self._json_dumps(row.get("feature_json")),
                    "explanation_json": self._json_dumps(row.get("explanation_json")),
                }
            )
        return self.session.insert_dicts("opportunity_ranker_scores_latest", payload)

    def append_opportunity_ranker_run(self, row: dict[str, Any]) -> int:
        if not row:
            return 0
        payload = {
            "run_id": str(row.get("run_id") or "").strip(),
            "created_at": self._datetime_value(row.get("created_at")) or utc_now(),
            "ranker_version": str(row.get("ranker_version") or "").strip(),
            "status": str(row.get("status") or "").strip().lower() or "unknown",
            "score_source": str(row.get("score_source") or "").strip() or None,
            "training_rows": int(row.get("training_rows") or 0),
            "validation_rows": int(row.get("validation_rows") or 0),
            "scoring_rows": int(row.get("scoring_rows") or 0),
            "oos_ic_20d": self._finite_float(row.get("oos_ic_20d")),
            "oos_hit_rate_20d": self._finite_float(row.get("oos_hit_rate_20d")),
            "feature_columns": self._json_dumps(row.get("feature_columns") or []),
            "detail_json": self._json_dumps(row.get("detail_json")),
        }
        return self.session.insert_dicts("opportunity_ranker_runs", [payload])

    def latest_opportunity_ranker_scores(
        self,
        *,
        tickers: list[str] | None = None,
        profiles: list[str] | None = None,
        buckets: list[str] | None = None,
        markets: list[str] | None = None,
        per_profile_limit: int | None = None,
        limit: int = 50,
        max_age_hours: int = 30,
    ) -> list[dict[str, Any]]:
        profile_limit = max(0, min(int(per_profile_limit or 0), 100))
        params: dict[str, Any] = {
            "limit": max(1, min(int(limit or 50), 500)),
            "max_age_hours": max(1, min(int(max_age_hours or 30), 24 * 14)),
            "per_profile_limit": profile_limit,
            "max_return_rows": min(500, max(1, min(int(limit or 50), 500)) + profile_limit * 8),
        }
        filters = ["computed_at >= (CURRENT_TIMESTAMP - $max_age_hours * INTERVAL '1 hour')"]
        row_filters = ["s.computed_at >= (CURRENT_TIMESTAMP - $max_age_hours * INTERVAL '1 hour')"]
        for name, values, transform in (
            ("tickers", tickers, lambda x: str(x or "").strip().upper()),
            ("profiles", profiles, lambda x: str(x or "").strip().lower()),
            ("buckets", buckets, lambda x: str(x or "").strip().lower()),
            ("markets", markets, lambda x: str(x or "").strip().lower()),
        ):
            tokens = [transform(value) for value in (values or []) if str(value or "").strip()]
            tokens = list(dict.fromkeys(tokens))
            if tokens:
                column = "ticker" if name == "tickers" else name[:-1]
                filters.append(f"{column} IN (SELECT unnest(${name}))")
                row_filters.append(f"s.{column} IN (SELECT unnest(${name}))")
                params[name] = tokens
        return self.session.fetch_rows(
            f"""
            WITH latest_batch AS (
              SELECT market, ranker_version, computed_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY market
                       ORDER BY computed_at DESC, ranker_version DESC
                     ) AS rn
              FROM opportunity_ranker_scores_latest
              WHERE {' AND '.join(filters)}
            ),
            dedup AS (
              SELECT s.*,
                     ROW_NUMBER() OVER (
                       PARTITION BY s.ticker
                       ORDER BY s.recommendation_rank ASC NULLS LAST,
                                s.recommendation_score DESC NULLS LAST
                     ) AS ticker_rn
              FROM opportunity_ranker_scores_latest s
              JOIN latest_batch b
                ON COALESCE(s.market, '') = COALESCE(b.market, '')
               AND s.ranker_version = b.ranker_version
               AND s.computed_at = b.computed_at
               AND b.rn = 1
              WHERE {' AND '.join(row_filters)}
            ),
            ranked AS (
              SELECT d.*,
                     ROW_NUMBER() OVER (
                       ORDER BY d.recommendation_rank ASC NULLS LAST,
                                d.recommendation_score DESC NULLS LAST,
                                d.ticker
                     ) AS global_rn,
                     ROW_NUMBER() OVER (
                       PARTITION BY d.profile
                       ORDER BY d.recommendation_rank ASC NULLS LAST,
                                d.recommendation_score DESC NULLS LAST,
                                d.ticker
                     ) AS profile_rn
              FROM dedup d
              WHERE ticker_rn = 1
            )
            SELECT * EXCLUDE (global_rn, profile_rn, ticker_rn)
            FROM ranked
            WHERE global_rn <= $limit
               OR ($per_profile_limit > 0 AND profile_rn <= $per_profile_limit)
            ORDER BY recommendation_rank ASC NULLS LAST, recommendation_score DESC NULLS LAST, ticker
            LIMIT $max_return_rows
            """,
            params,
        )

    def insert_shared_prep_session(self, row: dict[str, Any]) -> int:
        if not row:
            return 0
        payload = {
            "session_id": str(row.get("session_id") or "").strip(),
            "market": str(row.get("market") or "").strip().lower(),
            "trading_date": self._date_value(row.get("trading_date")) or utc_now().date(),
            "stage": str(row.get("stage") or "").strip().lower(),
            "status": str(row.get("status") or "").strip().lower() or "unknown",
            "forecast_run_id": str(row.get("forecast_run_id") or "").strip() or None,
            "forecast_rows_written": int(row.get("forecast_rows_written") or 0),
            "ranker_run_id": str(row.get("ranker_run_id") or "").strip() or None,
            "ranker_scores_written": int(row.get("ranker_scores_written") or 0),
            "created_at": self._datetime_value(row.get("created_at")) or utc_now(),
            "detail_json": self._json_dumps(row.get("detail_json")),
        }
        return self.session.insert_dicts("shared_prep_sessions", [payload])

    def get_latest_shared_prep_session(
        self,
        *,
        market: str,
        trading_date: Any,
        stage: str,
    ) -> dict[str, Any] | None:
        rows = self.session.fetch_rows(
            """
            SELECT session_id, market, trading_date, stage, status,
                   forecast_run_id, forecast_rows_written, ranker_run_id,
                   ranker_scores_written, created_at, detail_json
            FROM shared_prep_sessions
            WHERE market = $market
              AND trading_date = $trading_date
              AND stage = $stage
            ORDER BY created_at DESC
            LIMIT 1
            """,
            {
                "market": str(market or "").strip().lower(),
                "trading_date": self._date_value(trading_date) or trading_date,
                "stage": str(stage or "").strip().lower(),
            },
        )
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Instrument metadata
    # ------------------------------------------------------------------

    def latest_instrument_map(self, tickers: list[str]) -> dict[str, dict[str, Any]]:
        tokens = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}
        rows = self.session.fetch_rows(
            """
            WITH ranked AS (
              SELECT ticker, exchange_code, instrument_id, ticker_name,
                     currency, lot_size, tick_size, tradable, status, updated_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY updated_at DESC NULLS LAST
                     ) AS rn
              FROM instrument_master
              WHERE ticker IN (SELECT unnest($tickers))
            )
            SELECT ticker, exchange_code, instrument_id, ticker_name,
                   currency, lot_size, tick_size, tradable, status, updated_at
            FROM ranked
            WHERE rn = 1
            """,
            {"tickers": tokens},
        )
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            if ticker and ticker not in out:
                out[ticker] = row
        return out

    def ticker_name_map(
        self,
        *,
        tickers: list[str] | None = None,
        limit: int = 500,
    ) -> dict[str, str]:
        lim = max(1, int(limit))
        tokens = [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))

        params: dict[str, Any] = {"limit": lim}
        clauses: list[str] = ["ticker_name IS NOT NULL", "TRIM(ticker_name) != ''"]
        if tokens:
            clauses.append("ticker IN (SELECT unnest($tickers))")
            params["tickers"] = tokens

        rows = self.session.fetch_rows(
            f"""
            WITH ranked AS (
              SELECT ticker, ticker_name,
                     ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY updated_at DESC NULLS LAST
                     ) AS rn
              FROM instrument_master
              WHERE {' AND '.join(clauses)}
            )
            SELECT ticker, ticker_name
            FROM ranked
            WHERE rn = 1
            LIMIT $limit
            """,
            params,
        )
        out: dict[str, str] = {}
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            name = str(row.get("ticker_name") or "").strip()
            if ticker and name and ticker not in out:
                out[ticker] = name
        return out
