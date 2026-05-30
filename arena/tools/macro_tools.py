from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import math
from typing import Any, Literal

import requests

from arena.config import Settings
from arena.tools._market_scope import MarketScope

logger = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_ECOS_BASE = "https://ecos.bok.or.kr/api/KeyStatisticList"

_FOCUS_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "fx": ("external",),
    "fx_external": ("external",),
    "external": ("external",),
    "rates": ("rates_curve", "policy_rates", "bank_rates"),
    "rates_curve": ("rates_curve", "policy_rates", "bank_rates"),
    "inflation": ("inflation",),
    "growth": ("growth", "activity", "growth_activity"),
    "growth_cycle": ("growth", "activity", "growth_activity"),
    "activity": ("activity", "growth_activity"),
    "credit": ("credit_money", "liquidity_credit"),
    "credit_money": ("credit_money", "liquidity_credit"),
    "money": ("credit_money", "liquidity_credit", "money"),
    "risk_market": ("markets", "market", "commodities"),
    "market": ("markets", "market", "commodities"),
    "markets": ("markets", "market", "commodities"),
    "sentiment": ("sentiment",),
    "housing": ("housing",),
    "construction": ("construction",),
    "consumption": ("consumption",),
}

_REGIME_GROUP_KEYS: dict[str, str] = {
    "external": "fx_external",
    "rates_curve": "rates_curve",
    "policy_rates": "rates_curve",
    "bank_rates": "rates_curve",
    "inflation": "inflation",
    "growth": "growth_cycle",
    "activity": "growth_cycle",
    "growth_activity": "growth_cycle",
    "credit_money": "credit_money",
    "liquidity_credit": "credit_money",
    "markets": "risk_market",
    "market": "risk_market",
    "commodities": "risk_market",
    "sentiment": "sentiment",
    "housing": "housing",
    "construction": "construction",
    "consumption": "consumption",
}

_MACRO_PRIORITY_KEYS: tuple[str, ...] = (
    "fed_funds_rate",
    "sofr",
    "treasury_10y",
    "treasury_3m",
    "yield_spread_10y_3m",
    "cpi_yoy",
    "core_cpi_yoy",
    "real_gdp",
    "industrial_production_yoy",
    "high_yield_oas",
    "financial_stress_index",
    "sp500",
    "vix",
    "wti_crude",
    "mortgage_30y",
    "bok_base_rate",
    "kr_treasury_3y",
    "kr_treasury_5y",
    "kr_yield_spread_5y_3y",
    "usd_krw",
    "jpy_krw",
    "kr_current_account",
    "kr_fx_reserves",
    "kr_gdp_growth",
    "kr_all_industry_production",
    "kr_leading_cyclical_component",
    "kr_cpi",
    "kr_core_cpi",
    "kr_ppi",
    "kr_m2",
    "kr_household_credit",
    "kr_bank_loan_deposit_spread",
    "kr_consumer_sentiment_index",
    "kr_economic_sentiment_index",
    "kr_house_price_index",
    "kr_jeonse_price_index",
)


@dataclass(frozen=True, slots=True)
class _FredIndicatorSpec:
    key: str
    series_id: str
    name: str
    group: str
    unit: str
    frequency: str
    yoy_key: str = ""
    limit: int = 5


_FRED_INDICATOR_SPECS: tuple[_FredIndicatorSpec, ...] = (
    _FredIndicatorSpec("fed_funds_rate", "DFF", "Effective Federal Funds Rate", "policy_rates", "%", "daily"),
    _FredIndicatorSpec("fed_funds_rate_monthly", "FEDFUNDS", "Federal Funds Effective Rate", "policy_rates", "%", "monthly"),
    _FredIndicatorSpec("sofr", "SOFR", "Secured Overnight Financing Rate", "policy_rates", "%", "daily"),
    _FredIndicatorSpec("iorb", "IORB", "Interest Rate on Reserve Balances", "policy_rates", "%", "daily"),
    _FredIndicatorSpec("treasury_3m", "DGS3MO", "US 3M Treasury Yield", "rates_curve", "%", "daily"),
    _FredIndicatorSpec("treasury_2y", "DGS2", "US 2Y Treasury Yield", "rates_curve", "%", "daily"),
    _FredIndicatorSpec("treasury_10y", "DGS10", "US 10Y Treasury Yield", "rates_curve", "%", "daily"),
    _FredIndicatorSpec("treasury_30y", "DGS30", "US 30Y Treasury Yield", "rates_curve", "%", "daily"),
    _FredIndicatorSpec("cpi_index", "CPIAUCSL", "Consumer Price Index", "inflation", "1982-84=100", "monthly", "cpi_yoy", 16),
    _FredIndicatorSpec("core_cpi_index", "CPILFESL", "Core Consumer Price Index", "inflation", "1982-84=100", "monthly", "core_cpi_yoy", 16),
    _FredIndicatorSpec("pce_price_index", "PCEPI", "PCE Price Index", "inflation", "2017=100", "monthly", "pce_yoy", 16),
    _FredIndicatorSpec("core_pce_price_index", "PCEPILFE", "Core PCE Price Index", "inflation", "2017=100", "monthly", "core_pce_yoy", 16),
    _FredIndicatorSpec("breakeven_5y", "T5YIE", "5Y Breakeven Inflation Rate", "inflation", "%", "daily"),
    _FredIndicatorSpec("breakeven_10y", "T10YIE", "10Y Breakeven Inflation Rate", "inflation", "%", "daily"),
    _FredIndicatorSpec("unemployment_rate", "UNRATE", "Unemployment Rate", "labor", "%", "monthly"),
    _FredIndicatorSpec("nonfarm_payrolls", "PAYEMS", "All Employees, Total Nonfarm", "labor", "thousand persons", "monthly"),
    _FredIndicatorSpec("initial_jobless_claims", "ICSA", "Initial Jobless Claims", "labor", "persons", "weekly"),
    _FredIndicatorSpec("labor_force_participation", "CIVPART", "Labor Force Participation Rate", "labor", "%", "monthly"),
    _FredIndicatorSpec("avg_hourly_earnings", "CES0500000003", "Average Hourly Earnings", "labor", "$/hour", "monthly", "avg_hourly_earnings_yoy", 16),
    _FredIndicatorSpec("real_gdp", "GDPC1", "Real Gross Domestic Product", "growth_activity", "billions chained 2017 dollars", "quarterly", "real_gdp_yoy", 8),
    _FredIndicatorSpec("industrial_production", "INDPRO", "Industrial Production Index", "growth_activity", "2017=100", "monthly", "industrial_production_yoy", 16),
    _FredIndicatorSpec("retail_sales", "RSAFS", "Retail and Food Services Sales", "growth_activity", "millions dollars", "monthly", "retail_sales_yoy", 16),
    _FredIndicatorSpec("durable_goods_orders", "DGORDER", "Durable Goods New Orders", "growth_activity", "millions dollars", "monthly", "durable_goods_orders_yoy", 16),
    _FredIndicatorSpec("m2_money_supply", "M2SL", "M2 Money Supply", "liquidity_credit", "billions dollars", "monthly", "m2_money_supply_yoy", 16),
    _FredIndicatorSpec("fed_balance_sheet", "WALCL", "Federal Reserve Total Assets", "liquidity_credit", "millions dollars", "weekly"),
    _FredIndicatorSpec("overnight_reverse_repo", "RRPONTSYD", "Overnight Reverse Repurchase Agreements", "liquidity_credit", "millions dollars", "daily"),
    _FredIndicatorSpec("financial_stress_index", "STLFSI4", "St. Louis Fed Financial Stress Index", "liquidity_credit", "index", "weekly"),
    _FredIndicatorSpec("high_yield_oas", "BAMLH0A0HYM2", "US High Yield Option-Adjusted Spread", "liquidity_credit", "%", "daily"),
    _FredIndicatorSpec("corporate_oas", "BAMLC0A0CM", "US Corporate Option-Adjusted Spread", "liquidity_credit", "%", "daily"),
    _FredIndicatorSpec("sp500", "SP500", "S&P 500", "market", "pt", "daily"),
    _FredIndicatorSpec("nasdaq_composite", "NASDAQCOM", "NASDAQ Composite", "market", "pt", "daily"),
    _FredIndicatorSpec("dow_jones_industrial", "DJIA", "Dow Jones Industrial Average", "market", "pt", "daily"),
    _FredIndicatorSpec("vix", "VIXCLS", "CBOE Volatility Index", "market", "index", "daily"),
    _FredIndicatorSpec("wti_crude", "DCOILWTICO", "WTI Crude Oil", "commodities", "$/bbl", "daily"),
    _FredIndicatorSpec("trade_weighted_dollar", "DTWEXBGS", "Trade Weighted US Dollar Index", "external", "index", "daily"),
    _FredIndicatorSpec("fred_usd_krw", "DEXKOUS", "USD/KRW Exchange Rate", "external", "KRW per USD", "daily"),
    _FredIndicatorSpec("housing_starts", "HOUST", "Housing Starts", "housing", "thousand units", "monthly"),
    _FredIndicatorSpec("building_permits", "PERMIT", "Building Permits", "housing", "thousand units", "monthly"),
    _FredIndicatorSpec("case_shiller_home_price", "CSUSHPINSA", "S&P CoreLogic Case-Shiller US Home Price Index", "housing", "index", "monthly", "case_shiller_home_price_yoy", 16),
    _FredIndicatorSpec("mortgage_30y", "MORTGAGE30US", "30-Year Fixed Rate Mortgage Average", "housing", "%", "weekly"),
)

# ECOS KeyStatisticList KEYSTAT_NAME -> our indicator key mapping
_ECOS_INDICATORS: dict[str, str] = {
    "한국은행 기준금리": "bok_base_rate",
    "콜금리(익일물)": "call_rate",
    "KORIBOR(3개월)": "koribor_3m",
    "CD수익률(91일)": "cd_rate_91d",
    "통안증권수익률(364일)": "monetary_stabilization_bond_364d",
    "국고채수익률(3년)": "kr_treasury_3y",
    "국고채수익률(5년)": "kr_treasury_5y",
    "회사채수익률(3년,AA-)": "corp_bond_3y_aa",
    "예금은행 수신금리": "kr_bank_deposit_rate",
    "예금은행 대출금리": "kr_bank_loan_rate",
    "예금은행총예금(말잔)": "kr_bank_total_deposits",
    "예금은행대출금(말잔)": "kr_bank_loans",
    "가계신용": "kr_household_credit",
    "가계대출연체율": "kr_household_loan_delinquency_rate",
    "M1(협의통화, 평잔)": "kr_m1_money_supply",
    "M2(광의통화, 평잔)": "kr_m2_money_supply",
    "Lf(평잔)": "kr_lf_liquidity",
    "L(말잔)": "kr_l_liquidity",
    "원/달러 환율(종가)": "usd_krw",
    "원/엔(100엔) 환율(매매기준율)": "jpy_krw",
    "원/유로 환율(매매기준율)": "eur_krw",
    "원/위안 환율(종가)": "cny_krw",
    "주식거래대금(KOSPI)": "kospi_trading_value",
    "투자자예탁금": "investor_deposits",
    "채권거래대금": "bond_trading_value",
    "국고채발행액": "treasury_bond_issuance",
    "소비자물가지수": "kr_cpi",
    "농산물 및 석유류제외 소비자물가지수": "kr_core_cpi_ex_food_energy",
    "생활물가지수": "kr_living_cpi",
    "생산자물가지수": "kr_ppi",
    "수출물가지수": "kr_export_price_index",
    "수입물가지수": "kr_import_price_index",
    "실업률": "kr_unemployment",
    "고용률": "kr_employment",
    "경제활동인구": "kr_labor_force",
    "취업자수": "kr_employed_persons",
    "시간당명목임금지수": "kr_nominal_hourly_wage_index",
    "노동생산성지수": "kr_labor_productivity_index",
    "단위노동비용지수": "kr_unit_labor_cost_index",
    "경제성장률(실질, 계절조정 전기대비)": "kr_gdp_growth",
    "민간소비증감률(실질, 계절조정 전기대비)": "kr_private_consumption_growth",
    "설비투자증감률(실질, 계절조정 전기대비)": "kr_facility_investment_growth",
    "건설투자증감률(실질, 계절조정 전기대비)": "kr_construction_investment_growth",
    "재화의 수출 증감률(실질, 계절조정 전기대비)": "kr_goods_export_growth",
    "GDP(명목, 계절조정)": "kr_nominal_gdp",
    "1인당GNI": "kr_gni_per_capita",
    "총저축률": "kr_gross_savings_rate",
    "국내총투자율": "kr_domestic_investment_rate",
    "수출입의 대 GNI 비율": "kr_trade_to_gni_ratio",
    "전산업생산지수": "kr_all_industry_production",
    "제조업생산지수": "kr_manufacturing_production",
    "제조업출하지수": "kr_manufacturing_shipments",
    "제조업재고지수": "kr_manufacturing_inventory",
    "제조업가동률지수": "kr_manufacturing_capacity_utilization",
    "서비스업생산지수": "kr_services_production",
    "도소매업생산지수": "kr_wholesale_retail_production",
    "소매판매액지수": "kr_retail_sales_index",
    "개인신용카드사용액": "kr_personal_credit_card_spending",
    "자동차판매액지수": "kr_auto_sales_index",
    "설비투자지수": "kr_facility_investment_index",
    "설비용 기계류내수출하지수": "kr_machinery_domestic_shipments",
    "국내기계수주액": "kr_domestic_machinery_orders",
    "건설기성액": "kr_construction_completed_value",
    "건축허가면적": "kr_building_permit_area",
    "건설수주액": "kr_construction_orders",
    "건축착공면적": "kr_building_starts_area",
    "동행지수순환변동치": "kr_coincident_cyclical_component",
    "선행지수순환변동치": "kr_leading_cyclical_component",
    "전산업 기업심리지수실적": "kr_all_industry_bsi_actual",
    "소비자심리지수": "kr_consumer_sentiment_index",
    "제조업업황실적BSI": "kr_manufacturing_bsi_actual",
    "경제심리지수": "kr_economic_sentiment_index",
    "제조업매출액증감률": "kr_manufacturing_sales_growth",
    "제조업매출액세전순이익률": "kr_manufacturing_pretax_profit_margin",
    "제조업부채비율": "kr_manufacturing_debt_ratio",
    "가구당월평균소득": "kr_household_monthly_income",
    "평균소비성향": "kr_average_propensity_to_consume",
    "지니계수": "kr_gini_coefficient",
    "5분위배율": "kr_income_quintile_share_ratio",
    "추계인구": "kr_estimated_population",
    "고령인구비율": "kr_elderly_population_ratio",
    "합계출산율": "kr_total_fertility_rate",
    "경상수지": "kr_current_account",
    "직접투자(자산)": "kr_direct_investment_assets",
    "직접투자(부채)": "kr_direct_investment_liabilities",
    "증권투자(자산)": "kr_portfolio_investment_assets",
    "증권투자(부채)": "kr_portfolio_investment_liabilities",
    "수출금액지수": "kr_export_value_index",
    "수입금액지수": "kr_import_value_index",
    "순상품교역조건지수": "kr_net_barter_terms_of_trade",
    "소득교역조건지수": "kr_income_terms_of_trade",
    "외환보유액": "kr_fx_reserves",
    "대외채무": "kr_external_debt",
    "대외채권": "kr_external_claims",
    "주택매매가격지수": "kr_house_price_index",
    "주택전세가격지수": "kr_jeonse_price_index",
    "지가변동률(전기대비)": "kr_land_price_change",
    "Dubai유(현물)": "dubai_oil",
    "금": "gold_spot",
    "코스피지수": "kospi_index",
    "코스닥지수": "kosdaq_index",
}

_ECOS_GROUPS: dict[str, str] = {
    "시장금리": "policy_rates",
    "여수신금리": "bank_rates",
    "예금/대출금": "credit",
    "통화량": "money",
    "환율": "external",
    "주식": "market",
    "채권": "bond_market",
    "성장률": "growth_activity",
    "소득": "national_accounts",
    "GDP대비 비율": "national_accounts",
    "생산": "growth_activity",
    "소비": "consumption",
    "투자": "investment",
    "경기순환지표": "cycle",
    "심리지표": "sentiment",
    "기업경영지표": "corporate",
    "가계": "household",
    "소득분배지표": "distribution",
    "고용": "labor",
    "노동": "labor",
    "인구": "demographics",
    "국제수지": "external",
    "통관수출입": "trade",
    "대외채권/채무": "external",
    "소비자/생산자 물가": "inflation",
    "수출입 물가": "inflation",
    "부동산 가격": "housing",
    "국제원자재가격": "commodities",
}

_ECOS_UNITS: dict[str, str] = {
    "bok_base_rate": "%",
    "call_rate": "%",
    "kr_treasury_3y": "%",
    "kr_treasury_5y": "%",
    "corp_bond_3y_aa": "%",
    "usd_krw": "KRW",
    "kr_cpi": "2020=100",
    "kr_unemployment": "%",
    "kr_employment": "%",
    "kr_gdp_growth": "%",
    "kospi_index": "pt",
    "kosdaq_index": "pt",
}


@dataclass(slots=True)
class MacroTools:
    """FRED + ECOS 기반 거시경제지표 조회 도구."""

    settings: Settings
    repo: Any | None = None
    http_timeout: int = 12
    _session: requests.Session = field(default_factory=requests.Session, repr=False)
    _context: dict[str, Any] = field(default_factory=dict, repr=False)

    def set_context(self, context: dict[str, Any]) -> None:
        self._context = context

    def _scope(self) -> MarketScope:
        return MarketScope.from_context(
            self._context,
            fallback=getattr(self.settings, "kis_target_market", None),
        )

    def _effective_markets(self) -> set[str]:
        return self._scope().as_set()

    def _has_us_market(self) -> bool:
        return self._scope().has_us

    def _has_kospi_market(self) -> bool:
        return self._scope().has_kospi

    # ── FRED helpers ──

    def _fetch_series(self, series_id: str, *, limit: int = 2) -> list[dict[str, str]]:
        api_key = getattr(self.settings, "fred_api_key", "")
        if not api_key:
            return []
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": str(limit),
        }
        try:
            resp = self._session.get(_FRED_BASE, params=params, timeout=self.http_timeout)
            resp.raise_for_status()
            return resp.json().get("observations", [])
        except Exception as exc:
            logger.warning("[yellow]FRED fetch failed[/yellow] series=%s err=%s", series_id, str(exc)[:120])
            return []

    @staticmethod
    def _parse_value(obs: dict[str, str]) -> tuple[str, float | None]:
        d = obs.get("date", "")
        raw = obs.get("value", "").strip()
        if raw in {"", "."}:
            return d, None
        try:
            return d, float(raw)
        except (TypeError, ValueError):
            return d, None

    def _latest_valid(self, series_id: str, limit: int = 5) -> tuple[str, float | None]:
        observations = self._fetch_series(series_id, limit=limit)
        return self._latest_valid_from_observations(observations)

    def _latest_valid_from_observations(self, observations: list[dict[str, str]]) -> tuple[str, float | None]:
        for obs in observations:
            d, val = self._parse_value(obs)
            if val is not None:
                return d, val
        return "", None

    def _compute_yoy_from_observations(self, observations: list[dict[str, str]]) -> tuple[str, float | None]:
        valid = [(d, v) for obs in observations for d, v in [self._parse_value(obs)] if v is not None]
        if len(valid) < 2:
            return (valid[0][0] if valid else ""), None
        latest_date, latest_val = valid[0]
        try:
            latest_dt = date.fromisoformat(latest_date)
        except ValueError:
            return latest_date, None
        candidates: list[tuple[int, float]] = []
        for d_str, v in valid[1:]:
            try:
                dt = date.fromisoformat(d_str)
            except ValueError:
                continue
            diff_months = (latest_dt.year - dt.year) * 12 + (latest_dt.month - dt.month)
            if 11 <= diff_months <= 13:
                candidates.append((abs(diff_months - 12), v))
        if candidates:
            _, prior_val = sorted(candidates, key=lambda item: item[0])[0]
            yoy = ((latest_val - prior_val) / prior_val) * 100.0
            return latest_date, round(yoy, 2)
        return latest_date, None

    def _compute_cpi_yoy(self) -> tuple[str, float | None]:
        observations = self._fetch_series("CPIAUCSL", limit=14)
        return self._compute_yoy_from_observations(observations)

    @staticmethod
    def _indicator_payload(
        *,
        value: float,
        date_value: str,
        unit: str,
        name: str,
        group: str,
        source: str,
        frequency: str,
        series_id: str = "",
        class_name: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": value,
            "date": date_value,
            "unit": unit,
            "name": name,
            "group": group,
            "source": source,
            "frequency": frequency,
        }
        if series_id:
            payload["series_id"] = series_id
        if class_name:
            payload["class_name"] = class_name
        return payload

    @staticmethod
    def _ecos_frequency(cycle: str) -> str:
        token = str(cycle or "").strip()
        if "Q" in token:
            return "quarterly"
        if len(token) == 8 and token.isdigit():
            return "daily"
        if len(token) == 6 and token.isdigit():
            return "monthly"
        if len(token) == 4 and token.isdigit():
            return "annual"
        return ""

    @staticmethod
    def _group_indicators(indicators: dict[str, Any]) -> dict[str, dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        for key, payload in indicators.items():
            if not isinstance(payload, dict):
                continue
            group = str(payload.get("group") or "").strip()
            if not group:
                continue
            groups.setdefault(group, {})[key] = payload
        return groups

    @staticmethod
    def _add_spread(
        indicators: dict[str, Any],
        *,
        key: str,
        left_key: str,
        right_key: str,
        name: str,
        group: str,
        source: str,
        precision: int = 2,
    ) -> None:
        left = indicators.get(left_key, {}).get("value")
        right = indicators.get(right_key, {}).get("value")
        if left is None or right is None:
            return
        try:
            value = round(float(left) - float(right), precision)
        except (TypeError, ValueError):
            return
        indicators[key] = {
            "value": value,
            "unit": "pp",
            "name": name,
            "group": group,
            "source": source,
            "frequency": "derived",
        }

    def _us_macro(self) -> tuple[dict[str, Any], dict[str, int]]:
        api_key = getattr(self.settings, "fred_api_key", "")
        if not api_key:
            return {}, {"requested": 0, "returned": 0}
        indicators: dict[str, Any] = {}
        returned = 0
        for spec in _FRED_INDICATOR_SPECS:
            observations = self._fetch_series(spec.series_id, limit=spec.limit)
            obs_date, val = self._latest_valid_from_observations(observations)
            if val is None:
                continue
            returned += 1
            indicators[spec.key] = self._indicator_payload(
                value=val,
                date_value=obs_date,
                unit=spec.unit,
                name=spec.name,
                group=spec.group,
                source="fred",
                frequency=spec.frequency,
                series_id=spec.series_id,
            )
            if spec.yoy_key:
                yoy_date, yoy_val = self._compute_yoy_from_observations(observations)
                if yoy_val is not None:
                    indicators[spec.yoy_key] = self._indicator_payload(
                        value=yoy_val,
                        date_value=yoy_date,
                        unit="%",
                        name=f"{spec.name} YoY",
                        group=spec.group,
                        source="fred",
                        frequency="derived",
                        series_id=spec.series_id,
                    )

        self._add_spread(
            indicators,
            key="yield_spread_10y_2y",
            left_key="treasury_10y",
            right_key="treasury_2y",
            name="US 10Y minus 2Y Treasury Yield Spread",
            group="rates_curve",
            source="fred",
        )
        self._add_spread(
            indicators,
            key="yield_spread_10y_3m",
            left_key="treasury_10y",
            right_key="treasury_3m",
            name="US 10Y minus 3M Treasury Yield Spread",
            group="rates_curve",
            source="fred",
        )
        self._add_spread(
            indicators,
            key="credit_spread_hy_corp",
            left_key="high_yield_oas",
            right_key="corporate_oas",
            name="High Yield OAS minus Corporate OAS",
            group="liquidity_credit",
            source="fred",
        )
        return indicators, {"requested": len(_FRED_INDICATOR_SPECS), "returned": returned}

    # ── ECOS helpers ──

    def _fetch_ecos_key_stats(self) -> list[dict[str, Any]]:
        api_key = getattr(self.settings, "ecos_api_key", "")
        if not api_key:
            return []
        url = f"{_ECOS_BASE}/{api_key}/json/kr/1/101/"
        try:
            resp = self._session.get(url, timeout=self.http_timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("KeyStatisticList", {}).get("row", [])
        except Exception as exc:
            logger.warning("[yellow]ECOS fetch failed[/yellow] err=%s", str(exc)[:120])
            return []

    def _kr_macro(self) -> tuple[dict[str, Any], dict[str, int]]:
        rows = self._fetch_ecos_key_stats()
        if not rows:
            return {}, {"requested": 0, "returned": 0}
        indicators: dict[str, Any] = {}
        returned = 0
        for row in rows:
            name = str(row.get("KEYSTAT_NAME") or "").strip()
            key = _ECOS_INDICATORS.get(name)
            if not key:
                continue
            raw_val = str(row.get("DATA_VALUE") or "").strip()
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                continue
            cycle = str(row.get("CYCLE") or "")
            class_name = str(row.get("CLASS_NAME") or "").strip()
            unit = _ECOS_UNITS.get(key, str(row.get("UNIT_NAME") or "").strip())
            group = _ECOS_GROUPS.get(class_name, "korea_macro")
            indicators[key] = self._indicator_payload(
                value=val,
                date_value=cycle,
                unit=unit,
                name=name,
                group=group,
                source="ecos",
                frequency=self._ecos_frequency(cycle),
                class_name=class_name,
            )
            returned += 1

        self._add_spread(
            indicators,
            key="kr_yield_spread_5y_3y",
            left_key="kr_treasury_5y",
            right_key="kr_treasury_3y",
            name="Korea 5Y minus 3Y Treasury Yield Spread",
            group="policy_rates",
            source="ecos",
            precision=3,
        )
        self._add_spread(
            indicators,
            key="kr_credit_spread_corp_aa_3y",
            left_key="corp_bond_3y_aa",
            right_key="kr_treasury_3y",
            name="Korea AA- Corporate 3Y minus Treasury 3Y Spread",
            group="policy_rates",
            source="ecos",
            precision=3,
        )
        self._add_spread(
            indicators,
            key="kr_bank_loan_deposit_spread",
            left_key="kr_bank_loan_rate",
            right_key="kr_bank_deposit_rate",
            name="Korea Bank Loan minus Deposit Rate Spread",
            group="bank_rates",
            source="ecos",
            precision=2,
        )

        return indicators, {"requested": len(rows), "returned": returned}

    @staticmethod
    def _date_value(value: Any) -> date | None:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value or "").strip()
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
    def _float_value(value: Any) -> float | None:
        try:
            parsed = float(str(value).strip().replace(",", ""))
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _round_metric(value: float | None, digits: int = 4) -> float | None:
        if value is None or not math.isfinite(value):
            return None
        return round(float(value), digits)

    @staticmethod
    def _normalize_depth(depth: str | None) -> str:
        token = str(depth or "brief").strip().lower()
        return token if token in {"brief", "standard", "full"} else "brief"

    @staticmethod
    def _normalize_tokens(values: list[str] | tuple[str, ...] | str | None) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        return [str(value or "").strip() for value in values if str(value or "").strip()]

    @classmethod
    def _focus_groups(cls, focus: list[str] | tuple[str, ...] | str | None) -> set[str] | None:
        tokens = cls._normalize_tokens(focus)
        if not tokens:
            return None
        groups: set[str] = set()
        for token in tokens:
            key = token.strip().lower()
            groups.update(_FOCUS_GROUP_ALIASES.get(key, (key,)))
        return groups or None

    def _history_sources(self) -> list[str]:
        sources: list[str] = []
        if self._has_us_market():
            sources.append("fred")
        if self._has_kospi_market():
            sources.append("ecos")
        return sources or ["fred", "ecos"]

    def _history_markets(self) -> list[str]:
        markets: list[str] = []
        if self._has_us_market():
            markets.append("us")
        if self._has_kospi_market():
            markets.append("kr")
        return markets

    def _fetch_history_rows(
        self,
        *,
        indicators: list[str],
        lookback_days: int,
    ) -> list[dict[str, Any]]:
        loader = getattr(self.repo, "macro_indicator_observation_history", None) if self.repo is not None else None
        if not callable(loader):
            return []
        days = max(30, min(int(lookback_days or 540), 3650))
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)
        try:
            rows = loader(
                sources=self._history_sources(),
                markets=self._history_markets() or None,
                indicator_keys=indicators or None,
                start_date=start_date,
                end_date=end_date,
                lookback_days=days,
                limit=50000,
            )
        except Exception as exc:
            logger.warning("[yellow]macro history fetch failed[/yellow] err=%s", str(exc)[:120])
            return []
        return [row for row in rows if isinstance(row, dict)]

    @staticmethod
    def _prior_point(points: list[tuple[date, float, dict[str, Any]]], latest_date: date, days: int) -> tuple[date, float, dict[str, Any]] | None:
        cutoff = latest_date - timedelta(days=days)
        candidates = [point for point in points if point[0] <= cutoff]
        return candidates[-1] if candidates else None

    def _history_metric(
        self,
        key: str,
        rows: list[dict[str, Any]],
        *,
        as_of_date: date,
        include_series: bool,
        max_points: int,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        points: list[tuple[date, float, dict[str, Any]]] = []
        for row in rows:
            obs_date = self._date_value(row.get("observation_date"))
            value = self._float_value(row.get("value"))
            if obs_date is None or value is None:
                continue
            points.append((obs_date, value, row))
        if not points:
            return None
        points.sort(key=lambda item: item[0])
        latest_date, latest_value, latest_row = points[-1]
        values = [point[1] for point in points]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stdev = math.sqrt(variance)
        zscore = (latest_value - mean) / stdev if stdev > 0 else None
        percentile = None
        if len(values) > 1:
            below_or_equal = sum(1 for value in values if value <= latest_value)
            percentile = ((below_or_equal - 1) / (len(values) - 1)) * 100.0

        p1m = self._prior_point(points, latest_date, 30)
        p3m = self._prior_point(points, latest_date, 90)
        pyoy = self._prior_point(points, latest_date, 365)
        chg_1m = latest_value - p1m[1] if p1m else None
        chg_3m = latest_value - p3m[1] if p3m else None
        yoy = ((latest_value - pyoy[1]) / pyoy[1]) * 100.0 if pyoy and pyoy[1] else None
        trend_basis = chg_3m if chg_3m is not None else chg_1m
        trend = "flat"
        if trend_basis is not None and abs(trend_basis) > 1e-9:
            trend = "up" if trend_basis > 0 else "down"

        source = str(latest_row.get("source") or "").strip().lower()
        group = str(latest_row.get("group_name") or "").strip() or "macro"
        unit = str(latest_row.get("unit") or "").strip()
        identifier = str(latest_row.get("source_series_id") or "").strip()
        item_code = str(latest_row.get("source_item_code") or "").strip()
        metric: dict[str, Any] = {
            "k": key,
            "v": self._round_metric(latest_value),
            "d": latest_date.isoformat(),
            "src": source,
            "group": group,
            "freq": str(latest_row.get("frequency") or "").strip().lower(),
            "lag_days": max(0, (as_of_date - latest_date).days),
        }
        if unit:
            metric["u"] = unit
        if identifier:
            metric["id"] = identifier
        if item_code:
            metric["item"] = item_code
        for out_key, value in (
            ("chg_1m", chg_1m),
            ("chg_3m", chg_3m),
            ("yoy", yoy),
            ("z", zscore),
            ("pct", percentile),
        ):
            rounded = self._round_metric(value, 2 if out_key in {"pct", "yoy"} else 4)
            if rounded is not None:
                metric[out_key] = rounded
        metric["trend"] = trend
        if include_series:
            point_limit = max(1, min(int(max_points or 24), 120))
            metric["series"] = [
                {"d": point[0].isoformat(), "v": self._round_metric(point[1])}
                for point in points[-point_limit:]
            ]

        indicator = self._indicator_payload(
            value=float(latest_value),
            date_value=latest_date.isoformat(),
            unit=unit,
            name=str(latest_row.get("indicator_name") or key).strip(),
            group=group,
            source=source,
            frequency=str(latest_row.get("frequency") or "").strip().lower(),
            series_id=identifier,
        )
        if item_code:
            indicator["source_item_code"] = item_code
        for out_key in ("chg_1m", "chg_3m", "yoy", "z", "pct", "trend", "lag_days"):
            if out_key in metric:
                indicator[out_key] = metric[out_key]
        return metric, indicator

    @staticmethod
    def _metric_by_key(metrics: list[dict[str, Any]], *keys: str) -> dict[str, Any] | None:
        key_set = set(keys)
        return next((metric for metric in metrics if metric.get("k") in key_set), None)

    def _group_state(self, group: str, metrics: list[dict[str, Any]]) -> str:
        group_norm = str(group or "").strip().lower()
        if group_norm == "external":
            usd = self._metric_by_key(metrics, "usd_krw", "fred_usd_krw")
            if usd and (float(usd.get("pct") or 0) >= 80 or float(usd.get("v") or 0) >= 1350):
                return "pressure_high"
            if usd and usd.get("trend") == "up":
                return "pressure_rising"
            return "neutral"
        if group_norm in {"rates_curve", "policy_rates", "bank_rates"}:
            if any(float(metric.get("chg_3m") or 0) < -0.1 for metric in metrics):
                return "easing"
            if any(float(metric.get("v") or 0) >= 4.0 for metric in metrics):
                return "restrictive"
            return "neutral"
        if group_norm == "inflation":
            if any(float(metric.get("yoy") or 0) >= 3.0 for metric in metrics):
                return "elevated"
            if any(float(metric.get("chg_3m") or 0) < 0 for metric in metrics):
                return "cooling"
            return "neutral"
        if group_norm in {"growth", "activity", "growth_activity"}:
            if any(float(metric.get("yoy") or metric.get("chg_3m") or 0) < 0 for metric in metrics):
                return "slowing"
            if any(float(metric.get("yoy") or metric.get("chg_3m") or 0) > 0 for metric in metrics):
                return "improving"
            return "neutral"
        if group_norm in {"markets", "market", "commodities"}:
            vix = self._metric_by_key(metrics, "vix")
            if vix and float(vix.get("v") or 0) >= 25:
                return "risk_off"
            if any(float(metric.get("chg_3m") or 0) > 0 for metric in metrics):
                return "risk_on"
            return "neutral"
        if group_norm in {"credit_money", "liquidity_credit"}:
            if any(("spread" in str(metric.get("k") or "") or "delinquency" in str(metric.get("k") or "")) and float(metric.get("z") or 0) >= 1 for metric in metrics):
                return "tight"
            return "neutral"
        return "neutral"

    @staticmethod
    def _metric_rank(metric: dict[str, Any]) -> tuple[int, float]:
        key = str(metric.get("k") or "")
        priority = _MACRO_PRIORITY_KEYS.index(key) if key in _MACRO_PRIORITY_KEYS else len(_MACRO_PRIORITY_KEYS)
        score = max(abs(float(metric.get("z") or 0)), abs(float(metric.get("chg_3m") or 0)))
        return priority, -score

    def _select_metric_keys(
        self,
        metrics: dict[str, dict[str, Any]],
        *,
        indicator_keys: list[str],
        focus_groups: set[str] | None,
        max_indicators: int,
    ) -> tuple[list[str], int]:
        available = [
            metric
            for metric in metrics.values()
            if focus_groups is None or str(metric.get("group") or "").strip() in focus_groups
        ]
        if indicator_keys:
            selected = [key for key in indicator_keys if key in metrics and (focus_groups is None or metrics[key].get("group") in focus_groups)]
        else:
            limit = max(1, min(int(max_indicators or 30), 120))
            priority_keys = [key for key in _MACRO_PRIORITY_KEYS if any(metric.get("k") == key for metric in available)]
            remaining = [
                str(metric.get("k"))
                for metric in sorted(available, key=self._metric_rank)
                if str(metric.get("k")) not in priority_keys
            ]
            selected = [*priority_keys, *remaining][:limit]
        return selected, max(0, len(available) - len(selected))

    @staticmethod
    def _strip_internal_metric_fields(metric: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in metric.items() if key not in {"group"} and value is not None}

    def _historical_macro_snapshot(
        self,
        *,
        depth: str,
        focus: list[str] | tuple[str, ...] | str | None,
        indicators: list[str] | tuple[str, ...] | str | None,
        lookback_days: int,
        max_indicators: int,
        include_series: bool,
        max_points: int,
    ) -> dict[str, Any] | None:
        depth_norm = self._normalize_depth(depth)
        indicator_keys = self._normalize_tokens(indicators)
        rows = self._fetch_history_rows(indicators=[], lookback_days=lookback_days)
        if not rows:
            return None

        rows_by_key: dict[str, list[dict[str, Any]]] = {}
        as_of_candidates: list[date] = []
        for row in rows:
            key = str(row.get("indicator_key") or "").strip()
            obs_date = self._date_value(row.get("observation_date"))
            if not key or obs_date is None:
                continue
            rows_by_key.setdefault(key, []).append(row)
            as_of_candidates.append(obs_date)
        if not rows_by_key or not as_of_candidates:
            return None
        as_of_date = max(as_of_candidates)
        series_requested = bool(include_series) and depth_norm == "full"
        metrics: dict[str, dict[str, Any]] = {}
        indicator_payloads: dict[str, dict[str, Any]] = {}
        for key, key_rows in rows_by_key.items():
            metric_pair = self._history_metric(
                key,
                key_rows,
                as_of_date=as_of_date,
                include_series=series_requested,
                max_points=max_points,
            )
            if metric_pair is None:
                continue
            metric, indicator = metric_pair
            metrics[key] = metric
            indicator_payloads[key] = indicator
        focus_groups = self._focus_groups(focus)
        selected_keys, omitted_count = self._select_metric_keys(
            metrics,
            indicator_keys=indicator_keys,
            focus_groups=focus_groups,
            max_indicators=max_indicators,
        )
        if not selected_keys:
            return {
                "as_of": as_of_date.isoformat(),
                "source": "+".join(self._history_sources()),
                "depth": depth_norm,
                "data_mode": "historical",
                "error": "No macro history matched requested focus/indicators.",
            }

        selected_metrics = [metrics[key] for key in selected_keys if key in metrics]
        group_order: list[str] = []
        grouped_metrics: dict[str, list[dict[str, Any]]] = {}
        for metric in selected_metrics:
            group = str(metric.get("group") or "macro")
            if group not in grouped_metrics:
                grouped_metrics[group] = []
                group_order.append(group)
            grouped_metrics[group].append(metric)

        groups_payload: dict[str, dict[str, Any]] = {}
        regime_card: dict[str, str] = {}
        for group in group_order:
            group_metrics = grouped_metrics[group]
            state = self._group_state(group, group_metrics)
            groups_payload[group] = {
                "state": state,
                "evidence": [self._strip_internal_metric_fields(metric) for metric in group_metrics],
            }
            regime_key = _REGIME_GROUP_KEYS.get(group, group)
            if regime_card.get(regime_key) in {None, "neutral"}:
                regime_card[regime_key] = state

        notable = []
        for metric in sorted(selected_metrics, key=lambda item: max(abs(float(item.get("z") or 0)), abs(float(item.get("chg_3m") or 0))), reverse=True)[:5]:
            why_parts = []
            if abs(float(metric.get("z") or 0)) >= 1:
                why_parts.append("unusual percentile/z-score")
            if abs(float(metric.get("chg_3m") or 0)) > 0:
                why_parts.append(f"3m change {metric.get('chg_3m')}")
            notable.append({"k": metric.get("k"), "why": " and ".join(why_parts) or str(metric.get("trend") or "latest")})

        source_counts: dict[str, int] = {}
        for metric in selected_metrics:
            source = str(metric.get("src") or "").strip()
            if source:
                source_counts[source] = source_counts.get(source, 0) + 1
        coverage = {source: {"requested": count, "returned": count} for source, count in sorted(source_counts.items())}

        usd = metrics.get("usd_krw") or metrics.get("fred_usd_krw")
        rates_state = regime_card.get("rates_curve", "neutral")
        growth_state = regime_card.get("growth_cycle", "neutral")
        fx_state = regime_card.get("fx_external", "neutral")
        market_implications = {
            "equity_beta": "neutral_negative" if growth_state == "slowing" or fx_state in {"pressure_high", "pressure_rising"} else "neutral",
            "duration": "neutral_positive" if rates_state == "easing" else ("negative" if rates_state == "restrictive" else "neutral"),
            "usd_exposure": "positive_but_expensive" if usd and (float(usd.get("pct") or 0) >= 75 or float(usd.get("v") or 0) >= 1350) else "neutral",
            "kr_cyclicals": "selective" if growth_state == "slowing" else "neutral",
        }

        payload: dict[str, Any] = {
            "as_of": as_of_date.isoformat(),
            "source": "+".join(sorted(source_counts)) or "+".join(self._history_sources()),
            "depth": depth_norm,
            "data_mode": "historical",
            "coverage": coverage,
            "regime_card": regime_card,
            "market_implications": market_implications,
            "groups": groups_payload,
            "notable_movers": notable,
            "indicators": {key: indicator_payloads[key] for key in selected_keys if key in indicator_payloads},
        }
        focus_tokens = self._normalize_tokens(focus)
        if focus_tokens:
            payload["focus"] = focus_tokens
        if omitted_count > 0:
            payload["omitted"] = {
                "indicator_count": omitted_count,
                "reason": "available via focus/depth/indicators drilldown",
            }
        return payload

    # ── Public API ──

    def macro_snapshot(
        self,
        depth: Literal["brief", "standard", "full"] = "brief",
        focus: list[str] | tuple[str, ...] | str | None = None,
        indicators: list[str] | tuple[str, ...] | str | None = None,
        lookback_days: int = 540,
        max_indicators: int = 30,
        include_series: bool = False,
        max_points: int = 24,
    ) -> dict[str, Any]:
        """마켓에 따라 US(FRED) / KR(ECOS) 거시경제지표를 일괄 조회합니다."""
        logger.info("[cyan]TOOL[/cyan] macro_snapshot")

        historical = self._historical_macro_snapshot(
            depth=depth,
            focus=focus,
            indicators=indicators,
            lookback_days=lookback_days,
            max_indicators=max_indicators,
            include_series=include_series,
            max_points=max_points,
        )
        if historical is not None:
            return historical

        has_us = self._has_us_market()
        has_kr = self._has_kospi_market()

        indicators: dict[str, Any] = {}
        sources: list[str] = []
        coverage: dict[str, Any] = {}

        if has_us:
            us, fred_coverage = self._us_macro()
            coverage["fred"] = fred_coverage
            if us:
                indicators.update(us)
                sources.append("fred")
            elif not getattr(self.settings, "fred_api_key", ""):
                indicators["us_error"] = "FRED_API_KEY is not configured"

        if has_kr:
            kr, ecos_coverage = self._kr_macro()
            coverage["ecos"] = ecos_coverage
            if kr:
                indicators.update(kr)
                sources.append("ecos")
            elif not getattr(self.settings, "ecos_api_key", ""):
                indicators["kr_error"] = "ECOS_API_KEY is not configured"

        if not indicators:
            return {"error": "No macro data available. Check API keys (FRED/ECOS)."}

        return {
            "as_of": datetime.now(timezone.utc).date().isoformat(),
            "indicators": indicators,
            "groups": self._group_indicators(indicators),
            "coverage": coverage,
            "source": "+".join(sources),
        }
