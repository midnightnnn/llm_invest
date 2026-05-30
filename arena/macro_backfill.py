from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import logging
import math
import uuid
from typing import Any

import requests

from arena.tools.macro_tools import _FRED_BASE, _FRED_INDICATOR_SPECS

logger = logging.getLogger(__name__)

_FRED_BACKFILL_LIMIT = 100000
_ECOS_STAT_SEARCH_BASE = "https://ecos.bok.or.kr/api/StatisticSearch"


@dataclass(frozen=True, slots=True)
class EcosHistoricalSpec:
    key: str
    stat_code: str
    cycle: str
    item_codes: tuple[str, ...]
    name: str
    group: str
    unit: str
    frequency: str
    market: str = "kr"


ECOS_HISTORICAL_SPECS: tuple[EcosHistoricalSpec, ...] = (
    EcosHistoricalSpec("bok_base_rate", "722Y001", "D", ("0101000",), "BOK base rate", "rates_curve", "%", "daily"),
    EcosHistoricalSpec("call_rate", "817Y002", "D", ("010101000",), "Call rate", "rates_curve", "%", "daily"),
    EcosHistoricalSpec("koribor_3m", "817Y002", "D", ("010150000",), "KORIBOR 3M", "rates_curve", "%", "daily"),
    EcosHistoricalSpec("cd_91d", "817Y002", "D", ("010502000",), "CD 91D", "rates_curve", "%", "daily"),
    EcosHistoricalSpec("kr_msb_1y", "817Y002", "D", ("010400001",), "Monetary stabilization bond 1Y", "rates_curve", "%", "daily"),
    EcosHistoricalSpec("kr_treasury_3y", "817Y002", "D", ("010200000",), "Korea treasury 3Y", "rates_curve", "%", "daily"),
    EcosHistoricalSpec("kr_treasury_5y", "817Y002", "D", ("010200001",), "Korea treasury 5Y", "rates_curve", "%", "daily"),
    EcosHistoricalSpec("kr_corp_bond_3y_aa_minus", "817Y002", "D", ("010300000",), "Korea corporate bond 3Y AA-", "rates_curve", "%", "daily"),
    EcosHistoricalSpec("usd_krw", "731Y003", "D", ("0000003",), "USD/KRW close", "external", "KRW per USD", "daily"),
    EcosHistoricalSpec("jpy_krw", "731Y001", "D", ("0000002",), "JPY/KRW reference", "external", "KRW per 100 JPY", "daily"),
    EcosHistoricalSpec("eur_krw", "731Y001", "D", ("0000003",), "EUR/KRW reference", "external", "KRW per EUR", "daily"),
    EcosHistoricalSpec("cny_krw", "731Y003", "D", ("0000010",), "CNY/KRW close", "external", "KRW per CNY", "daily"),
    EcosHistoricalSpec("kospi_index", "802Y001", "D", ("0001000",), "KOSPI index", "markets", "index", "daily"),
    EcosHistoricalSpec("kosdaq_index", "802Y001", "D", ("0089000",), "KOSDAQ index", "markets", "index", "daily"),
    EcosHistoricalSpec("kospi_trading_value", "802Y001", "D", ("0088000",), "KOSPI trading value", "markets", "KRW 100M", "daily"),
    EcosHistoricalSpec("kr_bond_trading_value", "901Y015", "M", ("1", "2040000"), "Korea bond trading value", "markets", "KRW", "monthly"),
    EcosHistoricalSpec("kr_cpi", "901Y009", "M", ("0",), "Korea CPI", "inflation", "index", "monthly"),
    EcosHistoricalSpec("kr_core_cpi", "901Y010", "M", ("QB",), "Korea core CPI", "inflation", "index", "monthly"),
    EcosHistoricalSpec("kr_living_cpi", "901Y010", "M", ("110",), "Korea living CPI", "inflation", "index", "monthly"),
    EcosHistoricalSpec("kr_ppi", "404Y014", "M", ("*AA",), "Korea PPI", "inflation", "index", "monthly"),
    EcosHistoricalSpec("kr_export_price_index", "402Y014", "M", ("*AA",), "Korea export price index", "inflation", "index", "monthly"),
    EcosHistoricalSpec("kr_import_price_index", "401Y015", "M", ("*AA",), "Korea import price index", "inflation", "index", "monthly"),
    EcosHistoricalSpec("kr_deposit_rate", "121Y002", "M", ("BEABAA2",), "Korea bank deposit rate", "credit_money", "%", "monthly"),
    EcosHistoricalSpec("kr_loan_rate", "121Y006", "M", ("BECBLA01",), "Korea bank loan rate", "credit_money", "%", "monthly"),
    EcosHistoricalSpec("kr_bank_deposits", "104Y013", "M", ("BCB8",), "Korea bank total deposits", "credit_money", "KRW 100M", "monthly"),
    EcosHistoricalSpec("kr_bank_loans", "104Y016", "M", ("BDCA1",), "Korea bank loans", "credit_money", "KRW 100M", "monthly"),
    EcosHistoricalSpec("kr_household_credit", "151Y001", "Q", ("1000000",), "Korea household credit", "credit_money", "KRW 100M", "quarterly"),
    EcosHistoricalSpec("kr_household_loan_delinquency", "901Y054", "M", ("MO3AB",), "Korea household loan delinquency rate", "credit_money", "%", "monthly"),
    EcosHistoricalSpec("kr_m1", "161Y001", "M", ("BBLS00",), "Korea M1", "credit_money", "KRW 100M", "monthly"),
    EcosHistoricalSpec("kr_m2", "161Y005", "M", ("BBHS00",), "Korea M2", "credit_money", "KRW 100M", "monthly"),
    EcosHistoricalSpec("kr_lf", "171Y003", "M", ("LAS0000",), "Korea Lf liquidity", "credit_money", "KRW 100M", "monthly"),
    EcosHistoricalSpec("kr_liquidity_l", "172Y001", "M", ("XS00000",), "Korea L liquidity", "credit_money", "KRW 100M", "monthly"),
    EcosHistoricalSpec("kr_gdp_growth", "200Y102", "Q", ("10111",), "Korea real GDP growth", "growth", "% QoQ", "quarterly"),
    EcosHistoricalSpec("kr_private_consumption_growth", "200Y102", "Q", ("10122",), "Korea private consumption growth", "growth", "% QoQ", "quarterly"),
    EcosHistoricalSpec("kr_facility_investment_growth", "200Y102", "Q", ("10123",), "Korea facility investment growth", "growth", "% QoQ", "quarterly"),
    EcosHistoricalSpec("kr_construction_investment_growth", "200Y102", "Q", ("10124",), "Korea construction investment growth", "growth", "% QoQ", "quarterly"),
    EcosHistoricalSpec("kr_goods_exports_growth", "200Y102", "Q", ("10125",), "Korea goods exports growth", "growth", "% QoQ", "quarterly"),
    EcosHistoricalSpec("kr_nominal_gdp", "200Y107", "Q", ("10601",), "Korea nominal GDP", "growth", "KRW 100M", "quarterly"),
    EcosHistoricalSpec("kr_gross_savings_rate", "200Y102", "Q", ("40101",), "Korea gross savings rate", "growth", "%", "quarterly"),
    EcosHistoricalSpec("kr_domestic_investment_rate", "200Y102", "Q", ("40102",), "Korea domestic investment rate", "growth", "%", "quarterly"),
    EcosHistoricalSpec("kr_trade_to_gni_ratio", "200Y102", "Q", ("501",), "Korea trade to GNI ratio", "growth", "%", "quarterly"),
    EcosHistoricalSpec("kr_all_industry_production", "901Y033", "M", ("A00", "1"), "Korea all industry production index", "activity", "index", "monthly"),
    EcosHistoricalSpec("kr_retail_sales_index", "901Y100", "M", ("G0", "T3"), "Korea retail sales index", "activity", "index", "monthly"),
    EcosHistoricalSpec("kr_facility_investment_index", "901Y066", "M", ("I15A",), "Korea facility investment index", "activity", "index", "monthly"),
    EcosHistoricalSpec("kr_coincident_cyclical_component", "901Y067", "M", ("I16D",), "Korea coincident cyclical component", "activity", "index", "monthly"),
    EcosHistoricalSpec("kr_leading_cyclical_component", "901Y067", "M", ("I16E",), "Korea leading cyclical component", "activity", "index", "monthly"),
    EcosHistoricalSpec("kr_consumer_sentiment_index", "511Y002", "M", ("FME",), "Korea consumer sentiment index", "sentiment", "index", "monthly"),
    EcosHistoricalSpec("kr_economic_sentiment_index", "513Y001", "M", ("E1000",), "Korea economic sentiment index", "sentiment", "index", "monthly"),
    EcosHistoricalSpec("kr_all_industry_bsi_actual", "512Y013", "M", ("99988", "AA"), "Korea all industry BSI actual", "sentiment", "index", "monthly"),
    EcosHistoricalSpec("kr_manufacturing_bsi_actual", "512Y013", "M", ("C0000", "AA"), "Korea manufacturing BSI actual", "sentiment", "index", "monthly"),
    EcosHistoricalSpec("kr_fx_reserves", "732Y001", "M", ("99",), "Korea FX reserves", "external", "USD 100M", "monthly"),
    EcosHistoricalSpec("kr_current_account", "301Y013", "M", ("000000",), "Korea current account", "external", "USD 1M", "monthly"),
    EcosHistoricalSpec("kr_direct_investment_assets", "301Y013", "M", ("BOPF11000000",), "Korea direct investment assets", "external", "USD 1M", "monthly"),
    EcosHistoricalSpec("kr_direct_investment_liabilities", "301Y013", "M", ("BOPF12000000",), "Korea direct investment liabilities", "external", "USD 1M", "monthly"),
    EcosHistoricalSpec("kr_portfolio_investment_assets", "301Y013", "M", ("BOPF21000000",), "Korea portfolio investment assets", "external", "USD 1M", "monthly"),
    EcosHistoricalSpec("kr_portfolio_investment_liabilities", "301Y013", "M", ("BOPF22000000",), "Korea portfolio investment liabilities", "external", "USD 1M", "monthly"),
    EcosHistoricalSpec("kr_export_value_index", "403Y001", "M", ("*AA",), "Korea export value index", "external", "index", "monthly"),
    EcosHistoricalSpec("kr_import_value_index", "403Y003", "M", ("*AA",), "Korea import value index", "external", "index", "monthly"),
    EcosHistoricalSpec("kr_net_barter_terms_trade", "403Y005", "M", ("A",), "Korea net barter terms of trade", "external", "index", "monthly"),
    EcosHistoricalSpec("kr_income_terms_trade", "403Y005", "M", ("B",), "Korea income terms of trade", "external", "index", "monthly"),
    EcosHistoricalSpec("kr_external_debt", "311Y004", "Q", ("A000000",), "Korea external debt", "external", "USD 1M", "quarterly"),
    EcosHistoricalSpec("kr_external_claims", "311Y005", "Q", ("B000000",), "Korea external claims", "external", "USD 1M", "quarterly"),
    EcosHistoricalSpec("kr_house_price_index", "901Y062", "M", ("P63A",), "Korea house price index", "housing", "index", "monthly"),
    EcosHistoricalSpec("kr_jeonse_price_index", "901Y063", "M", ("P64A",), "Korea jeonse price index", "housing", "index", "monthly"),
    EcosHistoricalSpec("kr_land_price_change", "901Y064", "M", ("P65A",), "Korea land price change", "housing", "%", "monthly"),
    EcosHistoricalSpec("kr_card_spending", "601Y003", "M", ("200000",), "Korea card spending", "consumption", "KRW 1M", "monthly"),
    EcosHistoricalSpec("kr_construction_orders", "901Y020", "M", ("I42A",), "Korea construction orders", "construction", "KRW 100M", "monthly"),
    EcosHistoricalSpec("kr_construction_completed", "901Y104", "M", ("I48A",), "Korea construction completed value", "construction", "KRW 100M", "monthly"),
)


@dataclass(frozen=True, slots=True)
class MacroBackfillResult:
    start_date: date | None
    end_date: date
    discovered: int
    inserted: int
    source_counts: dict[str, int]
    dry_run: bool = False


@dataclass(slots=True)
class MacroBackfillService:
    settings: Any
    repo: Any
    session: Any = field(default_factory=requests.Session)
    http_timeout: int = 20
    now: Callable[[], datetime] = field(default_factory=lambda: lambda: datetime.now(timezone.utc))

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
                return date.fromisoformat(text[:10])
            except ValueError:
                return None

    @staticmethod
    def _ecos_date(value: Any) -> date | None:
        text = str(value or "").strip()
        if len(text) == 8 and text.isdigit():
            try:
                return datetime.strptime(text, "%Y%m%d").date()
            except ValueError:
                return None
        if len(text) == 6 and text.isdigit():
            try:
                return datetime.strptime(text, "%Y%m").date()
            except ValueError:
                return None
        if len(text) == 6 and text[:4].isdigit() and text[4:5].upper() == "Q" and text[5:].isdigit():
            year = int(text[:4])
            quarter = int(text[5:])
            if 1 <= quarter <= 4:
                return date(year, (quarter - 1) * 3 + 1, 1)
            return None
        if len(text) == 4 and text.isdigit():
            return date(int(text), 1, 1)
        return MacroBackfillService._date_value(text)

    @staticmethod
    def _ecos_period(value: date, cycle: str) -> str:
        cycle_norm = str(cycle or "").strip().upper()
        if cycle_norm == "D":
            return value.strftime("%Y%m%d")
        if cycle_norm == "M":
            return value.strftime("%Y%m")
        if cycle_norm == "Q":
            return f"{value.year}Q{((value.month - 1) // 3) + 1}"
        if cycle_norm == "A":
            return value.strftime("%Y")
        return value.isoformat()

    @staticmethod
    def _ecos_period_end(value: date, cycle: str) -> date:
        cycle_norm = str(cycle or "").strip().upper()
        if cycle_norm == "M":
            next_month = date(value.year + (1 if value.month == 12 else 0), 1 if value.month == 12 else value.month + 1, 1)
            return next_month - timedelta(days=1)
        if cycle_norm == "Q":
            end_month = ((value.month - 1) // 3 + 1) * 3
            next_month = date(value.year + (1 if end_month == 12 else 0), 1 if end_month == 12 else end_month + 1, 1)
            return next_month - timedelta(days=1)
        if cycle_norm == "A":
            return date(value.year, 12, 31)
        return value

    @staticmethod
    def _float_value(value: Any) -> float | None:
        try:
            parsed = float(str(value).strip().replace(",", ""))
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def _market_feature_start(self) -> date | None:
        loader = getattr(self.repo, "earliest_market_feature_date", None)
        if not callable(loader):
            return None
        return self._date_value(loader())

    @staticmethod
    def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[tuple[str, str, date, str, str], dict[str, Any]] = {}
        for row in rows:
            obs_date = MacroBackfillService._date_value(row.get("observation_date"))
            if obs_date is None:
                continue
            key = (
                str(row.get("source") or "").strip().lower(),
                str(row.get("indicator_key") or "").strip(),
                obs_date,
                str(row.get("source_series_id") or "").strip(),
                str(row.get("source_item_code") or "").strip(),
            )
            deduped[key] = row
        return list(deduped.values())

    def _fred_rows(
        self,
        *,
        start_date: date,
        end_date: date,
        observed_at: datetime,
        ingestion_run_id: str,
    ) -> list[dict[str, Any]]:
        api_key = str(getattr(self.settings, "fred_api_key", "") or "").strip()
        if not api_key:
            return []
        rows: list[dict[str, Any]] = []
        for spec in _FRED_INDICATOR_SPECS:
            params = {
                "series_id": spec.series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "asc",
                "observation_start": start_date.isoformat(),
                "observation_end": end_date.isoformat(),
                "limit": str(_FRED_BACKFILL_LIMIT),
            }
            try:
                resp = self.session.get(_FRED_BASE, params=params, timeout=self.http_timeout)
                resp.raise_for_status()
                observations = resp.json().get("observations", [])
            except Exception as exc:
                logger.warning(
                    "[yellow]FRED macro backfill fetch failed[/yellow] series=%s err=%s",
                    spec.series_id,
                    str(exc)[:120],
                )
                continue
            for obs in observations:
                obs_date = self._date_value(obs.get("date"))
                value = self._float_value(obs.get("value"))
                if obs_date is None or value is None or obs_date < start_date or obs_date > end_date:
                    continue
                rows.append(
                    {
                        "observed_at": observed_at,
                        "as_of_date": obs_date,
                        "source": "fred",
                        "indicator_key": spec.key,
                        "indicator_name": spec.name,
                        "group_name": spec.group,
                        "market": "us",
                        "source_series_id": spec.series_id,
                        "source_item_code": None,
                        "frequency": spec.frequency,
                        "observation_date": obs_date,
                        "value": value,
                        "unit": spec.unit,
                        "is_derived": False,
                        "raw_json": dict(obs),
                        "ingestion_run_id": ingestion_run_id,
                    }
                )
        return rows

    def _ecos_historical_rows(
        self,
        *,
        start_date: date,
        end_date: date,
        observed_at: datetime,
        ingestion_run_id: str,
    ) -> list[dict[str, Any]]:
        api_key = str(getattr(self.settings, "ecos_api_key", "") or "").strip()
        if not api_key:
            return []

        rows: list[dict[str, Any]] = []
        for spec in ECOS_HISTORICAL_SPECS:
            start = self._ecos_period(start_date, spec.cycle)
            end = self._ecos_period(end_date, spec.cycle)
            item_path = "/".join(spec.item_codes)
            url = (
                f"{_ECOS_STAT_SEARCH_BASE}/{api_key}/json/kr/1/100000/"
                f"{spec.stat_code}/{spec.cycle}/{start}/{end}/{item_path}"
            )
            try:
                resp = self.session.get(url, timeout=self.http_timeout)
                resp.raise_for_status()
                payload = resp.json()
                source_rows = payload.get("StatisticSearch", {}).get("row", [])
            except Exception as exc:
                logger.warning(
                    "[yellow]ECOS macro backfill fetch failed[/yellow] key=%s stat=%s err=%s",
                    spec.key,
                    spec.stat_code,
                    str(exc)[:120],
                )
                continue

            for item in source_rows:
                obs_date = self._ecos_date(item.get("TIME"))
                value = self._float_value(item.get("DATA_VALUE"))
                if obs_date is None or value is None:
                    continue
                if self._ecos_period_end(obs_date, spec.cycle) < start_date or obs_date > end_date:
                    continue
                rows.append(
                    {
                        "observed_at": observed_at,
                        "as_of_date": obs_date,
                        "source": "ecos",
                        "indicator_key": spec.key,
                        "indicator_name": spec.name,
                        "group_name": spec.group,
                        "market": spec.market,
                        "source_series_id": spec.stat_code,
                        "source_item_code": item_path,
                        "frequency": spec.frequency,
                        "observation_date": obs_date,
                        "value": value,
                        "unit": spec.unit,
                        "is_derived": False,
                        "raw_json": dict(item),
                        "ingestion_run_id": ingestion_run_id,
                    }
                )
        return rows

    def fetch_observations(self, *, start_date: date, end_date: date) -> list[dict[str, Any]]:
        observed_at = self.now()
        ingestion_run_id = uuid.uuid4().hex
        rows: list[dict[str, Any]] = []
        rows.extend(
            self._fred_rows(
                start_date=start_date,
                end_date=end_date,
                observed_at=observed_at,
                ingestion_run_id=ingestion_run_id,
            )
        )
        rows.extend(
            self._ecos_historical_rows(
                start_date=start_date,
                end_date=end_date,
                observed_at=observed_at,
                ingestion_run_id=ingestion_run_id,
            )
        )
        return self._dedupe_rows(rows)

    def backfill(
        self,
        *,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
        dry_run: bool = False,
        replace: bool = True,
    ) -> MacroBackfillResult:
        resolved_end = self._date_value(end_date) or self.now().date()
        resolved_start = self._date_value(start_date) or self._market_feature_start()
        if resolved_start is None:
            return MacroBackfillResult(
                start_date=None,
                end_date=resolved_end,
                discovered=0,
                inserted=0,
                source_counts={},
                dry_run=dry_run,
            )
        if resolved_start > resolved_end:
            return MacroBackfillResult(
                start_date=resolved_start,
                end_date=resolved_end,
                discovered=0,
                inserted=0,
                source_counts={},
                dry_run=dry_run,
            )

        rows = self.fetch_observations(start_date=resolved_start, end_date=resolved_end)
        source_counts = dict(Counter(str(row.get("source") or "") for row in rows if row.get("source")))
        inserted = 0
        if not dry_run:
            if replace and rows:
                deleter = getattr(self.repo, "delete_macro_indicator_observations", None)
                if callable(deleter):
                    delete_start = min(
                        (self._date_value(row.get("observation_date")) for row in rows),
                        default=resolved_start,
                    ) or resolved_start
                    deleter(
                        start_date=delete_start,
                        end_date=resolved_end,
                        sources=sorted(source_counts),
                    )
            inserted = int(self.repo.insert_macro_indicator_observations(rows))
        return MacroBackfillResult(
            start_date=resolved_start,
            end_date=resolved_end,
            discovered=len(rows),
            inserted=inserted,
            source_counts=source_counts,
            dry_run=dry_run,
        )

    def refresh_incremental(
        self,
        *,
        end_date: date | str | None = None,
        replace_days: int = 120,
        dry_run: bool = False,
    ) -> MacroBackfillResult:
        resolved_end = self._date_value(end_date) or self.now().date()
        sources = ["ecos", "fred"]
        latest_loader = getattr(self.repo, "latest_macro_indicator_observation_date", None)
        latest = self._date_value(latest_loader(sources=sources)) if callable(latest_loader) else None
        if latest is None:
            return self.backfill(end_date=resolved_end, dry_run=dry_run, replace=True)

        window_days = max(0, int(replace_days or 0))
        resolved_start = latest - timedelta(days=window_days)
        market_start = self._market_feature_start()
        if market_start is not None and resolved_start < market_start:
            resolved_start = market_start
        return self.backfill(
            start_date=resolved_start,
            end_date=resolved_end,
            dry_run=dry_run,
            replace=True,
        )
