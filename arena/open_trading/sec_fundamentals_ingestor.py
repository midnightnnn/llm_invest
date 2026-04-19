"""SEC EDGAR CompanyFacts based US fundamentals ingestor.

SEC's ``data.sec.gov`` APIs do not require an API key, but automated access
must identify itself with a declared User-Agent and stay within fair-access
limits. This ingestor converts CompanyFacts XBRL facts into the existing
``fundamentals_history_raw`` schema so downstream PIT-safe derived ratios and
the opportunity ranker can stay source-agnostic.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}
_QUARTER_FPS = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


@dataclass(frozen=True, slots=True)
class SECFundamentalsIngestResult:
    run_id: str
    status: str
    tickers_attempted: int
    tickers_succeeded: int
    quarters_inserted: int
    error_note: str = ""


@dataclass(frozen=True, slots=True)
class _Fact:
    value: float
    end: date
    filed: date
    fy: int
    fp: str
    form: str
    start: date | None = None
    frame: str | None = None


@dataclass(slots=True)
class _PeriodBucket:
    ticker: str
    fiscal_year: int
    fiscal_quarter: int
    fiscal_period_end: date
    announcement_date: date
    restated: bool = False
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None
    total_assets: float | None = None
    total_equity: float | None = None
    total_debt: float | None = None
    diluted_shares: float | None = None
    operating_cashflow: float | None = None
    capex: float | None = None
    ebitda: float | None = None


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


def _quarter_from_period_end(period_end: date) -> int:
    month = period_end.month
    if month in (1, 2, 3):
        return 1
    if month in (4, 5, 6):
        return 2
    if month in (7, 8, 9):
        return 3
    return 4


def _duration_days(fact: _Fact) -> int | None:
    if fact.start is None:
        return None
    return max(0, (fact.end - fact.start).days + 1)


def _is_quarter_duration(fact: _Fact) -> bool:
    days = _duration_days(fact)
    return days is not None and 45 <= days <= 135


def _is_annual_duration(fact: _Fact) -> bool:
    days = _duration_days(fact)
    return days is not None and 300 <= days <= 430


def _prefer_fact(current: _Fact | None, candidate: _Fact) -> _Fact:
    """Prefer original non-amended filings, then earliest filing date."""
    if current is None:
        return candidate
    cand_amended = candidate.form.endswith("/A")
    curr_amended = current.form.endswith("/A")
    if cand_amended != curr_amended:
        return current if cand_amended else candidate
    if candidate.filed != current.filed:
        return candidate if candidate.filed < current.filed else current
    if _is_quarter_duration(candidate) and not _is_quarter_duration(current):
        return candidate
    return current


def _safe_subtract(total: float | None, parts: list[float | None]) -> float | None:
    if total is None or any(part is None for part in parts):
        return None
    value = float(total) - sum(float(part or 0.0) for part in parts)
    return value if math.isfinite(value) else None


def _default_http_json(url: str, *, user_agent: str, timeout: float = 30.0) -> Any:
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read()
            encoding = str(resp.headers.get("Content-Encoding") or "").lower()
    except urllib.error.HTTPError as exc:  # pragma: no cover - external I/O
        logger.warning("SEC HTTPError url=%s status=%s", url, exc.code)
        return None
    except Exception as exc:  # pragma: no cover
        logger.warning("SEC request failed url=%s err=%s", url, str(exc)[:120])
        return None
    if encoding == "gzip" or body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return None


class SECFundamentalsIngestor:
    """Fetches and writes historical US fundamentals via SEC CompanyFacts."""

    FLOW_TAGS: dict[str, tuple[str, ...]] = {
        "revenue": (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
        ),
        "gross_profit": ("GrossProfit",),
        "operating_income": ("OperatingIncomeLoss",),
        "net_income": ("NetIncomeLoss", "ProfitLoss"),
        "eps_basic": ("EarningsPerShareBasic",),
        "eps_diluted": ("EarningsPerShareDiluted",),
        "operating_cashflow": (
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
        "capex": (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ),
        "ebitda": ("EarningsBeforeInterestTaxesDepreciationAmortization",),
        "diluted_shares": ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    }
    INSTANT_TAGS: dict[str, tuple[str, ...]] = {
        "total_assets": ("Assets",),
        "total_equity": (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
    }
    TOTAL_DEBT_TAGS: tuple[str, ...] = (
        "DebtAndFinanceLeaseObligations",
        "LongTermDebtAndFinanceLeaseObligations",
        "LongTermDebt",
    )
    CURRENT_DEBT_TAGS: tuple[str, ...] = (
        "DebtCurrent",
        "ShortTermBorrowings",
        "ShortTermDebt",
        "CurrentPortionOfLongTermDebt",
        "LongTermDebtCurrent",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
    )
    NONCURRENT_DEBT_TAGS: tuple[str, ...] = (
        "LongTermDebtNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
    )

    def __init__(
        self,
        *,
        repo: Any,
        user_agent: str,
        http_json: Callable[..., Any] | None = None,
        sleep_seconds: float = 0.15,
        write_batch_size: int = 1000,
        ticker_map_url: str = _SEC_COMPANY_TICKERS_URL,
        companyfacts_url_template: str = _SEC_COMPANYFACTS_URL,
    ) -> None:
        clean_user_agent = str(user_agent or "").strip()
        if not clean_user_agent:
            raise ValueError("SEC user_agent is required")
        self.repo = repo
        self.user_agent = clean_user_agent
        self.http_json = http_json or _default_http_json
        self.sleep_seconds = max(0.0, float(sleep_seconds))
        self.write_batch_size = max(1, int(write_batch_size))
        self.ticker_map_url = ticker_map_url
        self.companyfacts_url_template = companyfacts_url_template
        self._ticker_map: dict[str, str] | None = None

    def load_ticker_map(self) -> dict[str, str]:
        if self._ticker_map is not None:
            return dict(self._ticker_map)
        data = self.http_json(self.ticker_map_url, user_agent=self.user_agent)
        mapping: dict[str, str] = {}
        if isinstance(data, dict):
            for item in data.values():
                if not isinstance(item, dict):
                    continue
                ticker = str(item.get("ticker") or "").strip().upper()
                cik_raw = item.get("cik_str")
                if not ticker or cik_raw is None:
                    continue
                try:
                    cik = f"{int(cik_raw):010d}"
                except (TypeError, ValueError):
                    continue
                mapping[ticker] = cik
        self._ticker_map = mapping
        return dict(mapping)

    def _companyfacts(self, cik: str) -> dict[str, Any] | None:
        url = self.companyfacts_url_template.format(cik=cik)
        data = self.http_json(url, user_agent=self.user_agent)
        return data if isinstance(data, dict) else None

    def _facts_for_tags(
        self,
        companyfacts: dict[str, Any],
        tags: tuple[str, ...],
        units: tuple[str, ...],
    ) -> list[_Fact]:
        us_gaap = ((companyfacts.get("facts") or {}).get("us-gaap") or {})
        if not isinstance(us_gaap, dict):
            return []
        output: list[_Fact] = []
        for tag in tags:
            obj = us_gaap.get(tag)
            if not isinstance(obj, dict):
                continue
            unit_map = obj.get("units")
            if not isinstance(unit_map, dict):
                continue
            for unit in units:
                rows = unit_map.get(unit)
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    form = str(row.get("form") or "").strip().upper()
                    if form not in _FORMS:
                        continue
                    end = _parse_date(row.get("end"))
                    filed = _parse_date(row.get("filed"))
                    if end is None or filed is None:
                        continue
                    # CompanyFacts includes comparative prior-period facts in
                    # later filings. For PIT factor history we keep the
                    # original filing window and ignore stale comparative
                    # repeats/restatements that would shift old data forward.
                    filed_delay = (filed - end).days
                    if filed_delay < 0 or filed_delay > 200:
                        continue
                    value = _finite_float(row.get("val"))
                    if value is None:
                        continue
                    fy_raw = row.get("fy")
                    try:
                        fy = int(fy_raw) if fy_raw is not None else end.year
                    except (TypeError, ValueError):
                        fy = end.year
                    if abs(fy - end.year) > 1:
                        continue
                    output.append(
                        _Fact(
                            value=float(value),
                            end=end,
                            filed=filed,
                            fy=fy,
                            fp=str(row.get("fp") or "").strip().upper(),
                            form=form,
                            start=_parse_date(row.get("start")),
                            frame=str(row.get("frame") or "").strip() or None,
                        )
                    )
                if output:
                    return output
        return output

    def _set_flow_metric(
        self,
        buckets: dict[tuple[int, int], _PeriodBucket],
        *,
        ticker: str,
        metric: str,
        facts: list[_Fact],
    ) -> None:
        direct: dict[tuple[int, int], _Fact] = {}
        annual: dict[int, _Fact] = {}
        for fact in facts:
            if fact.fp in _QUARTER_FPS and _is_quarter_duration(fact):
                key = (fact.fy, _QUARTER_FPS[fact.fp])
                direct[key] = _prefer_fact(direct.get(key), fact)
            elif fact.fp == "FY" and _is_annual_duration(fact):
                annual[fact.fy] = _prefer_fact(annual.get(fact.fy), fact)

        for (fy, quarter), fact in direct.items():
            bucket = self._bucket(buckets, ticker=ticker, fact=fact, fy=fy, quarter=quarter)
            setattr(bucket, metric, fact.value)

        # Most filers do not publish a standalone fiscal Q4 fact. Derive it
        # from FY - Q1 - Q2 - Q3 when all components are present.
        for fy, fact in annual.items():
            parts = [getattr(buckets.get((fy, q)), metric, None) for q in (1, 2, 3)]
            q4_value = _safe_subtract(fact.value, parts)
            if q4_value is None:
                continue
            bucket = self._bucket(buckets, ticker=ticker, fact=fact, fy=fy, quarter=4)
            setattr(bucket, metric, q4_value)

    def _set_instant_metric(
        self,
        buckets: dict[tuple[int, int], _PeriodBucket],
        *,
        ticker: str,
        metric: str,
        facts: list[_Fact],
    ) -> None:
        chosen: dict[tuple[int, int], _Fact] = {}
        for fact in facts:
            quarter = _QUARTER_FPS.get(fact.fp) or (4 if fact.fp == "FY" else _quarter_from_period_end(fact.end))
            key = (fact.fy, quarter)
            chosen[key] = _prefer_fact(chosen.get(key), fact)
        for (fy, quarter), fact in chosen.items():
            bucket = self._bucket(buckets, ticker=ticker, fact=fact, fy=fy, quarter=quarter)
            setattr(bucket, metric, fact.value)

    def _set_debt_metric(
        self,
        buckets: dict[tuple[int, int], _PeriodBucket],
        *,
        ticker: str,
        companyfacts: dict[str, Any],
    ) -> None:
        total = self._instant_by_key(companyfacts, self.TOTAL_DEBT_TAGS)
        current = self._instant_by_key(companyfacts, self.CURRENT_DEBT_TAGS)
        noncurrent = self._instant_by_key(companyfacts, self.NONCURRENT_DEBT_TAGS)
        keys = set(total) | set(current) | set(noncurrent)
        for key in keys:
            fact = total.get(key) or current.get(key) or noncurrent.get(key)
            if fact is None:
                continue
            bucket = self._bucket(buckets, ticker=ticker, fact=fact, fy=key[0], quarter=key[1])
            if key in total:
                bucket.total_debt = total[key].value
            else:
                cur = current.get(key)
                noncur = noncurrent.get(key)
                if cur is not None or noncur is not None:
                    bucket.total_debt = float(cur.value if cur is not None else 0.0) + float(noncur.value if noncur is not None else 0.0)

    def _instant_by_key(self, companyfacts: dict[str, Any], tags: tuple[str, ...]) -> dict[tuple[int, int], _Fact]:
        facts = self._facts_for_tags(companyfacts, tags, ("USD",))
        chosen: dict[tuple[int, int], _Fact] = {}
        for fact in facts:
            quarter = _QUARTER_FPS.get(fact.fp) or (4 if fact.fp == "FY" else _quarter_from_period_end(fact.end))
            key = (fact.fy, quarter)
            chosen[key] = _prefer_fact(chosen.get(key), fact)
        return chosen

    def _bucket(
        self,
        buckets: dict[tuple[int, int], _PeriodBucket],
        *,
        ticker: str,
        fact: _Fact,
        fy: int,
        quarter: int,
    ) -> _PeriodBucket:
        key = (fy, quarter)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _PeriodBucket(
                ticker=ticker,
                fiscal_year=fy,
                fiscal_quarter=quarter,
                fiscal_period_end=fact.end,
                announcement_date=fact.filed,
                restated=fact.form.endswith("/A"),
            )
            buckets[key] = bucket
            return bucket
        if fact.filed < bucket.announcement_date:
            bucket.announcement_date = fact.filed
        if fact.end > bucket.fiscal_period_end:
            bucket.fiscal_period_end = fact.end
        bucket.restated = bucket.restated or fact.form.endswith("/A")
        return bucket

    def records_for_ticker(self, ticker: str, *, ticker_map: dict[str, str] | None = None) -> list[dict[str, Any]]:
        normalized = str(ticker or "").strip().upper()
        if not normalized:
            return []
        mapping = ticker_map or self.load_ticker_map()
        cik = mapping.get(normalized)
        if not cik:
            return []
        companyfacts = self._companyfacts(cik)
        if not companyfacts:
            return []

        buckets: dict[tuple[int, int], _PeriodBucket] = {}
        for metric, tags in self.FLOW_TAGS.items():
            units = ("USD/shares",) if metric in {"eps_basic", "eps_diluted"} else ("shares",) if metric == "diluted_shares" else ("USD",)
            facts = self._facts_for_tags(companyfacts, tags, units)
            self._set_flow_metric(buckets, ticker=normalized, metric=metric, facts=facts)
        for metric, tags in self.INSTANT_TAGS.items():
            facts = self._facts_for_tags(companyfacts, tags, ("USD",))
            self._set_instant_metric(buckets, ticker=normalized, metric=metric, facts=facts)
        self._set_debt_metric(buckets, ticker=normalized, companyfacts=companyfacts)

        retrieved_at = datetime.now(timezone.utc).isoformat()
        rows: list[dict[str, Any]] = []
        for bucket in sorted(buckets.values(), key=lambda item: (item.fiscal_year, item.fiscal_quarter)):
            has_core = any(
                value is not None
                for value in (
                    bucket.revenue,
                    bucket.net_income,
                    bucket.eps_diluted,
                    bucket.total_assets,
                    bucket.total_equity,
                )
            )
            if not has_core:
                continue
            free_cashflow = None
            if bucket.operating_cashflow is not None:
                free_cashflow = bucket.operating_cashflow - abs(bucket.capex or 0.0)
            book_value_per_share = None
            if bucket.total_equity is not None and bucket.diluted_shares and bucket.diluted_shares > 0:
                book_value_per_share = bucket.total_equity / bucket.diluted_shares
            rows.append(
                {
                    "ticker": bucket.ticker,
                    "market": "us",
                    "fiscal_year": bucket.fiscal_year,
                    "fiscal_quarter": bucket.fiscal_quarter,
                    "fiscal_period_end": bucket.fiscal_period_end.isoformat(),
                    "announcement_date": bucket.announcement_date.isoformat(),
                    "announcement_date_source": "sec_filed",
                    "currency": "USD",
                    "revenue": bucket.revenue,
                    "gross_profit": bucket.gross_profit,
                    "operating_income": bucket.operating_income,
                    "net_income": bucket.net_income,
                    "eps_basic": bucket.eps_basic,
                    "eps_diluted": bucket.eps_diluted,
                    "total_assets": bucket.total_assets,
                    "total_equity": bucket.total_equity,
                    "total_debt": bucket.total_debt,
                    "book_value_per_share": book_value_per_share,
                    "operating_cashflow": bucket.operating_cashflow,
                    "free_cashflow": free_cashflow,
                    "ebitda": bucket.ebitda,
                    "ev_ebitda": None,
                    "payout_ratio": None,
                    "revenue_growth_yoy": None,
                    "operating_income_growth_yoy": None,
                    "equity_growth_yoy": None,
                    "total_assets_growth_yoy": None,
                    "source": "sec_companyfacts",
                    "retrieved_at": retrieved_at,
                    "restated": bucket.restated,
                }
            )
        return rows

    def ingest_ticker(self, ticker: str) -> int:
        rows = self.records_for_ticker(ticker)
        if not rows:
            return 0
        return int(self.repo.insert_fundamentals_history_raw(rows) or 0)

    def run(self, *, tickers: list[str], market: str = "us") -> SECFundamentalsIngestResult:
        run_id = "fundamentals_sec_" + uuid.uuid4().hex[:20]
        started_at = datetime.now(timezone.utc)
        attempted = 0
        succeeded = 0
        quarters_total = 0
        errors: list[str] = []
        pending: list[dict[str, Any]] = []
        ticker_map = self.load_ticker_map()

        def flush() -> int:
            if not pending:
                return 0
            count = int(self.repo.insert_fundamentals_history_raw(list(pending)) or 0)
            pending.clear()
            return count

        for ticker in tickers:
            normalized = str(ticker or "").strip().upper()
            if not normalized:
                continue
            attempted += 1
            try:
                records = self.records_for_ticker(normalized, ticker_map=ticker_map)
                if records:
                    succeeded += 1
                    quarters_total += len(records)
                    pending.extend(records)
                    if len(pending) >= self.write_batch_size:
                        flush()
            except Exception as exc:
                errors.append(f"{normalized}:{str(exc)[:80]}")
                logger.exception("SEC fundamentals ingest failed ticker=%s", normalized)
            if self.sleep_seconds > 0:
                time.sleep(self.sleep_seconds)
        flush()

        finished_at = datetime.now(timezone.utc)
        status = "ok" if succeeded > 0 else ("partial" if attempted else "failed")
        error_note = "; ".join(errors[:10])
        append = getattr(self.repo, "append_fundamentals_ingest_run", None)
        if callable(append):
            try:
                append(
                    {
                        "run_id": run_id,
                        "source": "sec_companyfacts",
                        "market": market,
                        "started_at": started_at.isoformat(),
                        "finished_at": finished_at.isoformat(),
                        "status": status,
                        "tickers_attempted": attempted,
                        "tickers_succeeded": succeeded,
                        "quarters_inserted": quarters_total,
                        "error_note": error_note[:1000] if error_note else None,
                        "detail_json": {
                            "sleep_seconds": self.sleep_seconds,
                            "write_batch_size": self.write_batch_size,
                            "sample_errors": errors[:5],
                        },
                    }
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("fundamentals_ingest_runs write failed: %s", exc)
        return SECFundamentalsIngestResult(
            run_id=run_id,
            status=status,
            tickers_attempted=attempted,
            tickers_succeeded=succeeded,
            quarters_inserted=quarters_total,
            error_note=error_note,
        )
