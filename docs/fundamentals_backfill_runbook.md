# Fundamentals Backfill Runbook

Operational steps to populate `fundamentals_history_raw` with point-in-time
quarterly fundamentals for the dual-market universe. Run this before the first
shared prep that should include fundamental signals in the IC meta-learner.

## 0. Prerequisites

- `llm-arena` CLI installed and BigQuery credentials configured
  (`GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`)
- `llm-arena init-bq` already run so the new tables exist:
  `fundamentals_history_raw`, `fundamentals_derived_daily`,
  `fundamentals_ingest_runs`
- KR backfill needs the normal KIS credentials (already required by the rest of
  the pipeline)
- US backfill defaults to SEC EDGAR CompanyFacts and needs only a declared
  User-Agent (`SEC_USER_AGENT` or `ARENA_OPERATOR_EMAILS`). FMP remains optional
  via `--source fmp`.

## 1. KR backfill — KIS finance endpoints (≈ 30 min)

The KIS `/uapi/domestic-stock/v1/finance/*` endpoints return multi-period
quarterly rows. Announcement dates are inferred
(`fiscal_period_end + 45 days`) and tagged `announcement_date_source =
'kis_heuristic'`.

```bash
# Use the current universe (default; no --tickers flag needed)
llm-arena fundamentals-backfill-kr --period quarter

# Or supply an explicit ticker list
llm-arena fundamentals-backfill-kr --tickers 005930,000660,035420

# Or a file (one ticker per line)
llm-arena fundamentals-backfill-kr --tickers-file kospi_universe.txt
```

Throughput is gated by the KIS rate limit (~20 req/sec). A 2,000-ticker run
completes in roughly 15-30 minutes.

## 2. US backfill — SEC EDGAR CompanyFacts (default)

1. Declare SEC automated access identity. SEC does not require an API key, but
   requests must include a User-Agent with contact information.

   ```bash
   export SEC_USER_AGENT="LLM Arena you@example.com"
   ```

2. Run the backfill. The ingestor downloads SEC ticker-CIK mapping and each
   ticker's CompanyFacts JSON, normalizes common `us-gaap` concepts, uses SEC
   `filed` as the PIT announcement date, and writes rows tagged
   `source='sec_companyfacts'`.

   ```bash
   llm-arena fundamentals-backfill-us --source sec --sleep-seconds 0.15
   ```

   To rotate ticker chunks or retry a subset:

   ```bash
   llm-arena fundamentals-backfill-us --tickers-file us_chunk_$(date +%j).txt
   ```

3. Optional FMP fallback. FMP uses vendor-normalized statements but requires a
   key and may restrict historical depth on free plans.

   ```bash
   export FMP_API_KEY=<your_key>
   llm-arena fundamentals-backfill-us --source fmp --period quarter --limit 4
   ```

## 3. Verify coverage

```bash
llm-arena fundamentals-coverage
```

Expected output (rough guideline):

```
market=kospi years=2020~2026 tickers=1800+ rows=35000+ periods=24
market=us    years=2019~2026 tickers=400+  rows=12000+ periods=28
```

If `market=us` is under-represented, resume the SEC backfill; coverage
accumulates across runs because writes are append-only + deduped by
`(ticker, fiscal_year, fiscal_quarter)` at read time.

## 4. Populate the derived daily table

```bash
llm-arena refresh-fundamentals-derived --lookback-days 600
```

This rebuilds `fundamentals_derived_daily` (PE/PB/EP/BP/SP/ROE/growth/
debt-to-equity) for the lookback window, joining
`fundamentals_history_raw` with `market_features` closing prices under the
PIT constraint `announcement_date <= as_of_date`. Safe to rerun — the table
is append-only and the signal refresh dedups by `(as_of_date, ticker)` with
newest `created_at` winning.

## 5. Train the ranker on fundamentals

After derived daily is fresh, run the signal/IC/regime refresh + ranker build.
The signal SQL joins `fundamentals_derived_daily` as an optional PIT source, so
the new signals only contribute once this prep step has run at least once.

```bash
llm-arena build-opportunity-ranker \
  --lookback-days 540 \
  --horizon 20 \
  --min-ic-dates 60 \
  --min-valid-signals 3
```

The shared-prep and run-pipeline jobs call this automatically; running it
manually is useful for the first post-backfill bootstrap or for debugging.

## 6. Ongoing maintenance

- **KR**: run `llm-arena fundamentals-backfill-kr` daily (it's fast); only new
  periods appear as reports are released.
- **US**: run SEC CompanyFacts weekly for the live universe. Keep FMP as an
  optional enrichment/fallback source when a paid tier is available.
- Coverage report (`llm-arena fundamentals-coverage`) as a smoke test before
  large ranker changes.
- `opportunity_ranker_runs.detail_json.per_signal_train_rows` reveals which
  fundamental signals still have insufficient IC history — investigate before
  trusting their predicted weights.

## Rollback

If corrupt rows are inserted (e.g. schema mismatch after an upstream API
change), use BigQuery's table time travel to restore a safe snapshot, or
filter them out at read time by narrowing
`fundamentals_history_raw.retrieved_at`. The ingest runs table
(`fundamentals_ingest_runs`) records every attempt with start/end timestamps,
making it easy to identify which batch to discard.
