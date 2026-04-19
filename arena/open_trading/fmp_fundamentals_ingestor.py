"""Financial Modeling Prep (FMP) based US fundamentals ingestor.

FMP's free tier caps at 250 calls/day. Each ticker requires three calls
(income statement, balance sheet, cash flow) so a daily run can refresh
roughly 80 tickers. The ingestor accepts an ``http_fn`` callable so that
tests can inject a stub without installing ``requests``.

Announcement dates use FMP's ``fillingDate`` field when available; fallback
is ``date + 40 days`` heuristic tagged ``fmp_heuristic``.
"""

from __future__ import annotations

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

_FMP_BASE_URL = "https://financialmodelingprep.com/stable"


def _redact_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    safe_query = urllib.parse.urlencode(
        [(key, "<redacted>" if key.lower() == "apikey" else value) for key, value in query]
    )
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, safe_query, parsed.fragment))


@dataclass(frozen=True, slots=True)
class FMPFundamentalsIngestResult:
    run_id: str
    status: str
    tickers_attempted: int
    tickers_succeeded: int
    quarters_inserted: int
    error_note: str = ""


@dataclass(slots=True)
class _TickerBundle:
    ticker: str
    income: list[dict[str, Any]] = field(default_factory=list)
    balance: list[dict[str, Any]] = field(default_factory=list)
    cashflow: list[dict[str, Any]] = field(default_factory=list)


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


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if len(text) < 10:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _default_http_fn(url: str, *, timeout: float = 15.0) -> Any:
    """Default HTTP GET → JSON. Returns ``None`` on 4xx/5xx."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "llm-arena-fmp-ingestor"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read()
    except urllib.error.HTTPError as exc:  # pragma: no cover - external I/O
        logger.warning("FMP HTTPError url=%s status=%s", _redact_url(url), exc.code)
        return None
    except Exception as exc:  # pragma: no cover
        logger.warning("FMP request failed url=%s err=%s", _redact_url(url), str(exc)[:120])
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def _quarter_from_period_end(period_end: date) -> int:
    month = period_end.month
    if month in (1, 2, 3):
        return 1
    if month in (4, 5, 6):
        return 2
    if month in (7, 8, 9):
        return 3
    return 4


class FMPFundamentalsIngestor:
    """Fetches and writes historical US fundamentals via FMP freemium API."""

    SLEEP_BETWEEN_TICKERS = 1.0  # be nice to the free tier

    def __init__(
        self,
        *,
        api_key: str,
        repo: Any,
        period: str = "quarter",
        limit: int = 40,
        http_fn: Callable[..., Any] | None = None,
        base_url: str = _FMP_BASE_URL,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise ValueError("FMP api_key is required")
        self.api_key = str(api_key).strip()
        self.repo = repo
        self.period = "quarter" if period == "quarter" else "annual"
        self.limit = max(4, min(int(limit), 120))
        self.http_fn = http_fn or _default_http_fn
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, ticker: str) -> list[dict[str, Any]]:
        query = urllib.parse.urlencode(
            {"symbol": ticker, "period": self.period, "limit": self.limit, "apikey": self.api_key}
        )
        url = f"{self.base_url}/{path}?{query}"
        data = self.http_fn(url)
        if not isinstance(data, list):
            return []
        return [row for row in data if isinstance(row, dict)]

    def _fetch_bundle(self, ticker: str) -> _TickerBundle:
        return _TickerBundle(
            ticker=ticker,
            income=self._get("income-statement", ticker),
            balance=self._get("balance-sheet-statement", ticker),
            cashflow=self._get("cash-flow-statement", ticker),
        )

    def _merge_bundle(self, bundle: _TickerBundle) -> list[dict[str, Any]]:
        combined: dict[str, dict[str, Any]] = {}
        for source_name, rows in (
            ("income", bundle.income),
            ("balance", bundle.balance),
            ("cashflow", bundle.cashflow),
        ):
            for row in rows:
                period_end_text = str(row.get("date") or "").strip()
                if not period_end_text:
                    continue
                key = period_end_text[:10]
                bucket = combined.setdefault(key, {})
                for column, value in row.items():
                    if value in ("", None):
                        continue
                    bucket[f"{source_name}:{column}"] = value

        retrieved_at = datetime.now(timezone.utc)
        records: list[dict[str, Any]] = []
        for period_key, fields in combined.items():
            period_end = _parse_date(period_key)
            if period_end is None:
                continue
            filing_date = _parse_date(fields.get("income:fillingDate") or fields.get("balance:fillingDate") or fields.get("cashflow:fillingDate"))
            if filing_date is None:
                filing_date = period_end + timedelta(days=40)
                announcement_source = "fmp_heuristic"
            else:
                announcement_source = "fmp_filing"
            quarter = _quarter_from_period_end(period_end)
            year = period_end.year
            currency = str(fields.get("income:reportedCurrency") or fields.get("balance:reportedCurrency") or "USD").strip().upper() or "USD"
            revenue = _finite_float(fields.get("income:revenue"))
            gross_profit = _finite_float(fields.get("income:grossProfit"))
            op_income = _finite_float(fields.get("income:operatingIncome"))
            net_income = _finite_float(fields.get("income:netIncome"))
            eps_basic = _finite_float(fields.get("income:eps"))
            eps_diluted = _finite_float(fields.get("income:epsdiluted") or fields.get("income:epsDiluted"))
            total_assets = _finite_float(fields.get("balance:totalAssets"))
            total_equity = _finite_float(fields.get("balance:totalStockholdersEquity") or fields.get("balance:totalEquity"))
            total_debt = _finite_float(fields.get("balance:totalDebt"))
            book_value_per_share = None
            if total_equity is not None:
                shares = _finite_float(fields.get("income:weightedAverageShsOutDil") or fields.get("income:weightedAverageShsOut"))
                if shares and shares > 0:
                    book_value_per_share = total_equity / shares
            operating_cf = _finite_float(fields.get("cashflow:operatingCashFlow"))
            free_cf = _finite_float(fields.get("cashflow:freeCashFlow"))
            ebitda = _finite_float(fields.get("income:ebitda"))
            records.append(
                {
                    "ticker": bundle.ticker,
                    "market": "us",
                    "fiscal_year": year,
                    "fiscal_quarter": quarter if self.period == "quarter" else 0,
                    "fiscal_period_end": period_end.isoformat(),
                    "announcement_date": filing_date.isoformat(),
                    "announcement_date_source": announcement_source,
                    "currency": currency,
                    "revenue": revenue,
                    "gross_profit": gross_profit,
                    "operating_income": op_income,
                    "net_income": net_income,
                    "eps_basic": eps_basic,
                    "eps_diluted": eps_diluted if eps_diluted is not None else eps_basic,
                    "total_assets": total_assets,
                    "total_equity": total_equity,
                    "total_debt": total_debt,
                    "book_value_per_share": book_value_per_share,
                    "operating_cashflow": operating_cf,
                    "free_cashflow": free_cf,
                    "ebitda": ebitda,
                    "ev_ebitda": None,
                    "payout_ratio": None,
                    "revenue_growth_yoy": None,
                    "operating_income_growth_yoy": None,
                    "equity_growth_yoy": None,
                    "total_assets_growth_yoy": None,
                    "source": "fmp",
                    "retrieved_at": retrieved_at.isoformat(),
                    "restated": False,
                }
            )
        records.sort(key=lambda r: (r["fiscal_year"], r["fiscal_quarter"]))
        return records

    def ingest_ticker(self, ticker: str) -> int:
        bundle = self._fetch_bundle(ticker=ticker.strip().upper())
        records = self._merge_bundle(bundle)
        if not records:
            return 0
        return int(self.repo.insert_fundamentals_history_raw(records) or 0)

    def run(
        self,
        *,
        tickers: list[str],
        market: str = "us",
    ) -> FMPFundamentalsIngestResult:
        run_id = "fundamentals_fmp_" + uuid.uuid4().hex[:20]
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
                logger.exception("FMP fundamentals ingest failed ticker=%s", normalized)
            time.sleep(self.SLEEP_BETWEEN_TICKERS)
        finished_at = datetime.now(timezone.utc)
        status = "ok" if succeeded > 0 else ("partial" if attempted else "failed")
        error_note = "; ".join(errors[:10])
        append = getattr(self.repo, "append_fundamentals_ingest_run", None)
        if callable(append):
            try:
                append(
                    {
                        "run_id": run_id,
                        "source": "fmp",
                        "market": market,
                        "started_at": started_at.isoformat(),
                        "finished_at": finished_at.isoformat(),
                        "status": status,
                        "tickers_attempted": attempted,
                        "tickers_succeeded": succeeded,
                        "quarters_inserted": quarters_total,
                        "error_note": error_note[:1000] if error_note else None,
                        "detail_json": {
                            "period": self.period,
                            "limit": self.limit,
                            "sample_errors": errors[:5],
                        },
                    }
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("fundamentals_ingest_runs write failed: %s", exc)
        return FMPFundamentalsIngestResult(
            run_id=run_id,
            status=status,
            tickers_attempted=attempted,
            tickers_succeeded=succeeded,
            quarters_inserted=quarters_total,
            error_note=error_note,
        )
