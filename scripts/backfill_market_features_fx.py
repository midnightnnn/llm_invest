"""Diagnose and repair tainted close_price_krw / fx_rate_used rows in market_features.

The sync pipeline previously fell back to a hard-coded default FX rate
(typically 1300.0) for `open_trading_us*` EOD rows whenever
`_ensure_usd_krw_daily_fx()` returned an empty map. That wrote KRW prices
computed from a fabricated FX into the warehouse, which in turn poisoned
cross-day benchmark returns (see portfolio_diagnosis "SPY 14.4%" incident).

This script:
  1. Reports how many rows carry the suspect fx_rate_used=1300.0 for US EOD
     sources while close_price_native is set.
  2. For each such day/ticker, looks up the matching `*_quote` row's real
     fx_rate_used and rewrites close_price_krw + fx_rate_used on the EOD
     row to match.

Usage:
    python scripts/backfill_market_features_fx.py --dry-run
    python scripts/backfill_market_features_fx.py --start 2026-03-06 --apply
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime

from google.cloud import bigquery

logger = logging.getLogger("backfill_market_features_fx")

_US_EOD_SOURCES = (
    "open_trading_us",
    "open_trading_nasdaq",
    "open_trading_nyse",
    "open_trading_amex",
)
_SUSPECT_FX = 1300.0


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--dataset", default=os.environ.get("BQ_DATASET", "llm_arena"))
    parser.add_argument("--location", default=os.environ.get("BQ_LOCATION", "asia-northeast3"))
    parser.add_argument("--start", type=_parse_date, default=None, help="inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=_parse_date, default=None, help="inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--ticker", default=None, help="restrict to single ticker for testing")
    parser.add_argument("--apply", action="store_true", help="run the UPDATE (otherwise dry-run diagnostics only)")
    parser.add_argument("--dry-run", action="store_true", help="alias for omitting --apply")
    return parser.parse_args()


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _partition_filter(alias: str, start: date | None, end: date | None) -> tuple[str, dict[str, object]]:
    clauses: list[str] = []
    params: dict[str, object] = {}
    if start is not None:
        clauses.append(f"DATE({alias}.as_of_ts) >= @start")
        params["start"] = start
    if end is not None:
        clauses.append(f"DATE({alias}.as_of_ts) <= @end")
        params["end"] = end
    return (" AND " + " AND ".join(clauses) if clauses else ""), params


def _scalar_params(values: dict[str, object]) -> list[bigquery.ScalarQueryParameter]:
    params: list[bigquery.ScalarQueryParameter] = []
    for name, value in values.items():
        if isinstance(value, date):
            params.append(bigquery.ScalarQueryParameter(name, "DATE", value))
        elif isinstance(value, float):
            params.append(bigquery.ScalarQueryParameter(name, "FLOAT64", value))
        elif isinstance(value, int):
            params.append(bigquery.ScalarQueryParameter(name, "INT64", value))
        else:
            params.append(bigquery.ScalarQueryParameter(name, "STRING", str(value)))
    return params


def _array_param(name: str, values: list[str], element_type: str = "STRING") -> bigquery.ArrayQueryParameter:
    return bigquery.ArrayQueryParameter(name, element_type, list(values))


def _run(client: bigquery.Client, sql: str, params: list[bigquery.QueryParameter]) -> list[bigquery.Row]:
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    return list(job.result())


def _diagnose(client: bigquery.Client, dataset_fqn: str, args: argparse.Namespace) -> int:
    partition_clause, extra_params = _partition_filter("t", args.start, args.end)
    ticker_clause = ""
    if args.ticker:
        ticker_clause = " AND t.ticker = @ticker"
        extra_params["ticker"] = args.ticker.upper()

    sql = f"""
    SELECT
      COUNT(*) AS tainted_rows,
      COUNT(DISTINCT t.ticker) AS tainted_tickers,
      MIN(DATE(t.as_of_ts)) AS min_d,
      MAX(DATE(t.as_of_ts)) AS max_d
    FROM `{dataset_fqn}.market_features` AS t
    WHERE t.source IN UNNEST(@eod_sources)
      AND t.fx_rate_used = @suspect_fx
      AND t.close_price_native IS NOT NULL
      {partition_clause}{ticker_clause}
    """
    params = [
        _array_param("eod_sources", list(_US_EOD_SOURCES)),
        bigquery.ScalarQueryParameter("suspect_fx", "FLOAT64", _SUSPECT_FX),
        *_scalar_params(extra_params),
    ]
    rows = _run(client, sql, params)
    if not rows:
        logger.info("no diagnostic rows returned")
        return 0

    summary = rows[0]
    tainted = int(summary["tainted_rows"] or 0)
    tickers = int(summary["tainted_tickers"] or 0)
    min_d = summary["min_d"]
    max_d = summary["max_d"]
    logger.info(
        "tainted EOD rows: %d (tickers=%d, range=%s..%s, suspect_fx=%.2f)",
        tainted,
        tickers,
        min_d,
        max_d,
        _SUSPECT_FX,
    )

    # Also report how many of those have a matching quote row we can borrow from.
    sql_pairable = f"""
    WITH tainted AS (
      SELECT DATE(t.as_of_ts) AS d, t.ticker
      FROM `{dataset_fqn}.market_features` AS t
      WHERE t.source IN UNNEST(@eod_sources)
        AND t.fx_rate_used = @suspect_fx
        AND t.close_price_native IS NOT NULL
        {partition_clause}{ticker_clause}
      GROUP BY d, ticker
    ),
    quotes AS (
      SELECT DATE(q.as_of_ts) AS d, q.ticker, ANY_VALUE(q.fx_rate_used) AS real_fx
      FROM `{dataset_fqn}.market_features` AS q
      WHERE ENDS_WITH(q.source, '_quote')
        AND q.fx_rate_used IS NOT NULL
        AND q.fx_rate_used > 0
        AND q.fx_rate_used != @suspect_fx
      GROUP BY d, ticker
    )
    SELECT
      (SELECT COUNT(*) FROM tainted) AS tainted_day_tickers,
      (SELECT COUNT(*) FROM tainted t JOIN quotes q USING (d, ticker)) AS pairable_day_tickers
    """
    pair_rows = _run(client, sql_pairable, params)
    if pair_rows:
        pr = pair_rows[0]
        logger.info(
            "pairable via same-day quote fx: %d / %d day-ticker combos",
            int(pr["pairable_day_tickers"] or 0),
            int(pr["tainted_day_tickers"] or 0),
        )
    return tainted


def _apply(client: bigquery.Client, dataset_fqn: str, args: argparse.Namespace) -> int:
    partition_clause, extra_params = _partition_filter("t", args.start, args.end)
    ticker_clause = ""
    if args.ticker:
        ticker_clause = " AND t.ticker = @ticker"
        extra_params["ticker"] = args.ticker.upper()

    sql = f"""
    MERGE `{dataset_fqn}.market_features` AS t
    USING (
      SELECT DATE(q.as_of_ts) AS d, q.ticker, ANY_VALUE(q.fx_rate_used) AS real_fx
      FROM `{dataset_fqn}.market_features` AS q
      WHERE ENDS_WITH(q.source, '_quote')
        AND q.fx_rate_used IS NOT NULL
        AND q.fx_rate_used > 0
        AND q.fx_rate_used != @suspect_fx
      GROUP BY d, ticker
    ) AS src
    ON DATE(t.as_of_ts) = src.d
       AND t.ticker = src.ticker
       AND t.source IN UNNEST(@eod_sources)
       AND t.fx_rate_used = @suspect_fx
       AND t.close_price_native IS NOT NULL
       {partition_clause}{ticker_clause}
    WHEN MATCHED THEN
      UPDATE SET
        fx_rate_used = src.real_fx,
        close_price_krw = t.close_price_native * src.real_fx
    """
    params = [
        _array_param("eod_sources", list(_US_EOD_SOURCES)),
        bigquery.ScalarQueryParameter("suspect_fx", "FLOAT64", _SUSPECT_FX),
        *_scalar_params(extra_params),
    ]
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    job.result()
    affected = int(job.num_dml_affected_rows or 0)
    logger.info("MERGE updated %d rows", affected)
    return affected


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _args()
    if not args.project:
        print("GOOGLE_CLOUD_PROJECT not set (or pass --project)", file=sys.stderr)
        return 2
    dataset_fqn = f"{args.project}.{args.dataset}"
    client = bigquery.Client(project=args.project, location=args.location)

    tainted = _diagnose(client, dataset_fqn, args)

    if args.dry_run or not args.apply:
        logger.info("dry-run complete; pass --apply to run MERGE")
        return 0

    if tainted == 0:
        logger.info("nothing to repair")
        return 0

    _apply(client, dataset_fqn, args)
    _diagnose(client, dataset_fqn, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
