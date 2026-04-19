"""KIS-based historical fundamentals ingestor for KR stocks.

KIS's domestic finance endpoints return multi-period rows keyed by
``stac_yymm`` (fiscal period end). The ingestor composes five endpoints
per ticker, merges by period, and writes PIT-safe rows into
``fundamentals_history_raw``.

Announcement dates are not surfaced by the API, so we use the legally
conservative heuristic ``announcement_date = fiscal_period_end + 45 days``
and tag ``announcement_date_source = 'kis_heuristic'``. DART enrichment is
optional and handled by a separate module.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from arena.open_trading.client import OpenTradingClient

logger = logging.getLogger(__name__)

_KIS_ANNOUNCEMENT_LAG_DAYS: dict[int, int] = {
    1: 45,
    2: 45,
    3: 45,
    4: 90,
}


@dataclass(frozen=True, slots=True)
class KISFundamentalsIngestResult:
    run_id: str
    status: str
    tickers_attempted: int
    tickers_succeeded: int
    quarters_inserted: int
    error_note: str = ""


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _parse_stac_yymm(value: Any) -> tuple[int, int, date] | None:
    """Parses ``stac_yymm`` like '202409' → (year, quarter, fiscal_end_date)."""
    text = str(value or "").strip()
    if len(text) < 6 or not text[:6].isdigit():
        return None
    year = int(text[:4])
    month = int(text[4:6])
    if month not in (3, 6, 9, 12):
        return None
    quarter_map = {3: 1, 6: 2, 9: 3, 12: 4}
    quarter = quarter_map[month]
    end_month_day = {3: 31, 6: 30, 9: 30, 12: 31}[month]
    fiscal_end = date(year, month, end_month_day)
    return year, quarter, fiscal_end


def _infer_announcement(fiscal_end: date, quarter: int) -> date:
    lag = _KIS_ANNOUNCEMENT_LAG_DAYS.get(int(quarter), 45)
    return fiscal_end + timedelta(days=lag)


class KISFundamentalsIngestor:
    """Fetches and writes historical KR fundamentals via KIS finance APIs."""

    SLEEP_BETWEEN_CALLS = 0.12  # KIS real-env roughly allows 20 req/sec

    def __init__(
        self,
        *,
        client: OpenTradingClient,
        repo: Any,
        div_cls_code: str = "1",
        announcement_source: str = "kis_heuristic",
    ) -> None:
        self.client = client
        self.repo = repo
        self.div_cls_code = str(div_cls_code or "1").strip() or "1"
        self.announcement_source = announcement_source

    def _fetch_all_statements(self, ticker: str) -> dict[str, list[dict[str, Any]]]:
        """Returns ``{endpoint: rows}`` for one ticker, skipping endpoints that fail."""
        fetchers = {
            "balance_sheet": self.client.get_domestic_balance_sheet,
            "income_statement": self.client.get_domestic_income_statement,
            "financial_ratio": self.client.get_domestic_financial_ratio,
            "growth_ratio": self.client.get_domestic_growth_ratio,
            "other_major_ratios": self.client.get_domestic_other_major_ratios,
        }
        out: dict[str, list[dict[str, Any]]] = {}
        for name, fn in fetchers.items():
            try:
                out[name] = fn(ticker=ticker, div_cls_code=self.div_cls_code) or []
            except Exception as exc:  # pragma: no cover - transient API errors
                logger.warning("KIS %s fetch failed ticker=%s err=%s", name, ticker, str(exc)[:120])
                out[name] = []
            time.sleep(self.SLEEP_BETWEEN_CALLS)
        return out

    def _merge_period_rows(
        self,
        ticker: str,
        bundles: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Merges per-endpoint rows keyed by ``stac_yymm`` into a single record per period."""
        merged: dict[str, dict[str, Any]] = {}
        for endpoint, rows in bundles.items():
            for row in rows:
                period = str(row.get("stac_yymm") or "").strip()
                if not period:
                    continue
                bucket = merged.setdefault(period, {})
                for key, value in row.items():
                    if key == "stac_yymm" or value in ("", None):
                        continue
                    # prefix endpoint for disambiguation of same-named fields
                    bucket[f"{endpoint}:{key}"] = value
                bucket["stac_yymm"] = period
        records: list[dict[str, Any]] = []
        retrieved_at = datetime.now(timezone.utc)
        for period, fields in merged.items():
            parsed = _parse_stac_yymm(period)
            if not parsed:
                continue
            year, quarter, fiscal_end = parsed
            revenue = _finite_float(fields.get("income_statement:sale_account"))
            gross_profit = _finite_float(fields.get("income_statement:sale_totl_prfi"))
            op_income = _finite_float(fields.get("income_statement:bsop_prti"))
            net_income = _finite_float(fields.get("income_statement:thtr_ntin"))
            total_assets = _finite_float(fields.get("balance_sheet:total_aset"))
            total_equity = _finite_float(fields.get("balance_sheet:total_cptl"))
            total_debt = _finite_float(fields.get("balance_sheet:total_lblt"))
            eps = _finite_float(fields.get("financial_ratio:eps"))
            bps = _finite_float(fields.get("financial_ratio:bps"))
            revenue_growth = _finite_float(fields.get("growth_ratio:grs"))
            operating_growth = _finite_float(fields.get("growth_ratio:bsop_prfi_inrt"))
            equity_growth = _finite_float(fields.get("growth_ratio:equt_inrt"))
            assets_growth = _finite_float(fields.get("growth_ratio:totl_aset_inrt"))
            ebitda = _finite_float(fields.get("other_major_ratios:ebitda"))
            ev_ebitda = _finite_float(fields.get("other_major_ratios:ev_ebitda"))
            payout_ratio = _finite_float(fields.get("other_major_ratios:payout_rate"))
            announcement_date = _infer_announcement(fiscal_end, quarter)
            records.append(
                {
                    "ticker": ticker,
                    "market": "kospi",
                    "fiscal_year": year,
                    "fiscal_quarter": quarter,
                    "fiscal_period_end": fiscal_end.isoformat(),
                    "announcement_date": announcement_date.isoformat(),
                    "announcement_date_source": self.announcement_source,
                    "currency": "KRW",
                    "revenue": revenue,
                    "gross_profit": gross_profit,
                    "operating_income": op_income,
                    "net_income": net_income,
                    "eps_basic": eps,
                    "eps_diluted": eps,
                    "total_assets": total_assets,
                    "total_equity": total_equity,
                    "total_debt": total_debt,
                    "book_value_per_share": bps,
                    "operating_cashflow": None,
                    "free_cashflow": None,
                    "ebitda": ebitda,
                    "ev_ebitda": ev_ebitda,
                    "payout_ratio": payout_ratio,
                    "revenue_growth_yoy": (revenue_growth / 100.0) if revenue_growth is not None else None,
                    "operating_income_growth_yoy": (operating_growth / 100.0) if operating_growth is not None else None,
                    "equity_growth_yoy": (equity_growth / 100.0) if equity_growth is not None else None,
                    "total_assets_growth_yoy": (assets_growth / 100.0) if assets_growth is not None else None,
                    "source": "kis_finance",
                    "retrieved_at": retrieved_at.isoformat(),
                    "restated": False,
                }
            )
        records.sort(key=lambda r: (r["fiscal_year"], r["fiscal_quarter"]))
        return records

    def ingest_ticker(self, ticker: str) -> int:
        bundles = self._fetch_all_statements(ticker=ticker.strip().upper())
        records = self._merge_period_rows(ticker=ticker.strip().upper(), bundles=bundles)
        if not records:
            return 0
        return int(self.repo.insert_fundamentals_history_raw(records) or 0)

    def run(
        self,
        *,
        tickers: list[str],
        market: str = "kospi",
    ) -> KISFundamentalsIngestResult:
        run_id = "fundamentals_kis_" + uuid.uuid4().hex[:20]
        started_at = datetime.now(timezone.utc)
        attempted = 0
        succeeded = 0
        quarters_total = 0
        errors: list[str] = []
        for ticker in tickers:
            normalized = str(ticker or "").strip().upper()
            if not normalized:
                continue
            attempted += 1
            try:
                inserted = self.ingest_ticker(normalized)
                if inserted:
                    succeeded += 1
                    quarters_total += inserted
            except Exception as exc:
                errors.append(f"{normalized}:{str(exc)[:80]}")
                logger.exception("KIS fundamentals ingest failed ticker=%s", normalized)
        finished_at = datetime.now(timezone.utc)
        status = "ok" if succeeded > 0 else ("partial" if attempted else "failed")
        error_note = "; ".join(errors[:10])
        append = getattr(self.repo, "append_fundamentals_ingest_run", None)
        if callable(append):
            try:
                append(
                    {
                        "run_id": run_id,
                        "source": "kis_finance",
                        "market": market,
                        "started_at": started_at.isoformat(),
                        "finished_at": finished_at.isoformat(),
                        "status": status,
                        "tickers_attempted": attempted,
                        "tickers_succeeded": succeeded,
                        "quarters_inserted": quarters_total,
                        "error_note": error_note[:1000] if error_note else None,
                        "detail_json": {
                            "div_cls_code": self.div_cls_code,
                            "announcement_source": self.announcement_source,
                            "sample_errors": errors[:5],
                        },
                    }
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("fundamentals_ingest_runs write failed: %s", exc)
        return KISFundamentalsIngestResult(
            run_id=run_id,
            status=status,
            tickers_attempted=attempted,
            tickers_succeeded=succeeded,
            quarters_inserted=quarters_total,
            error_note=error_note,
        )
