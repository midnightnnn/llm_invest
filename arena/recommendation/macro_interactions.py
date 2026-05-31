"""Macro factor snapshots for conditional joint-policy coefficients."""

from __future__ import annotations

import bisect
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

MACRO_FACTOR_NAMES: tuple[str, ...] = (
    "liquidity",
    "rate_pressure",
    "curve",
    "credit_stress",
    "fx_pressure",
    "growth",
    "inflation",
    "risk_stress",
)

MACRO_CONDITIONING_PAIRS: tuple[tuple[str, str], ...] = (
    ("momentum_20d", "liquidity"),
    ("momentum_20d", "growth"),
    ("momentum_20d", "fx_pressure"),
    ("pullback", "risk_stress"),
    ("lowvol", "rate_pressure"),
    ("lowvol", "credit_stress"),
    ("lowvol", "risk_stress"),
    ("forecast_er", "liquidity"),
    ("forecast_er", "rate_pressure"),
    ("forecast_prob", "risk_stress"),
    ("ma_crossover", "liquidity"),
    ("ep", "inflation"),
    ("bp", "inflation"),
    ("sp", "inflation"),
    ("revenue_growth", "growth"),
    ("eps_growth", "growth"),
    ("low_debt", "credit_stress"),
)

MACRO_CONDITIONING_INDICATOR_KEYS: tuple[str, ...] = (
    "fed_funds_rate",
    "sofr",
    "treasury_3m",
    "treasury_2y",
    "treasury_10y",
    "treasury_30y",
    "cpi_index",
    "core_cpi_index",
    "pce_price_index",
    "core_pce_price_index",
    "breakeven_5y",
    "breakeven_10y",
    "real_gdp",
    "industrial_production",
    "retail_sales",
    "durable_goods_orders",
    "m2_money_supply",
    "fed_balance_sheet",
    "overnight_reverse_repo",
    "financial_stress_index",
    "high_yield_oas",
    "corporate_oas",
    "vix",
    "trade_weighted_dollar",
    "fred_usd_krw",
    "bok_base_rate",
    "call_rate",
    "koribor_3m",
    "cd_91d",
    "kr_treasury_3y",
    "kr_treasury_5y",
    "kr_corp_bond_3y_aa_minus",
    "usd_krw",
    "kr_cpi",
    "kr_core_cpi",
    "kr_ppi",
    "kr_m1",
    "kr_m2",
    "kr_lf",
    "kr_liquidity_l",
    "kr_household_credit",
    "kr_household_loan_delinquency",
    "kr_gdp_growth",
    "kr_all_industry_production",
    "kr_retail_sales_index",
    "kr_facility_investment_index",
    "kr_leading_cyclical_component",
    "kr_consumer_sentiment_index",
    "kr_economic_sentiment_index",
)


@dataclass(frozen=True, slots=True)
class _MacroPoint:
    observation_date: date
    effective_date: date
    value: float


@dataclass(frozen=True, slots=True)
class _MacroSeries:
    key: str
    points: tuple[_MacroPoint, ...]
    effective_dates: tuple[date, ...]
    observation_dates: tuple[date, ...]

    def latest(self, as_of: date) -> _MacroPoint | None:
        idx = bisect.bisect_right(self.effective_dates, as_of) - 1
        if idx < 0:
            return None
        return self.points[idx]

    def point_at_or_before_observation(self, cutoff: date) -> _MacroPoint | None:
        idx = bisect.bisect_right(self.observation_dates, cutoff) - 1
        if idx < 0:
            return None
        return self.points[idx]

    def z_score(self, as_of: date, *, window_days: int = 730, min_points: int = 6) -> float | None:
        latest = self.latest(as_of)
        if latest is None:
            return None
        idx = bisect.bisect_right(self.effective_dates, as_of)
        start_obs = latest.observation_date - timedelta(days=max(1, int(window_days)))
        values = [
            point.value
            for point in self.points[:idx]
            if point.observation_date >= start_obs and math.isfinite(point.value)
        ]
        if len(values) < int(min_points):
            return None
        center = _median(values)
        scale = _std(values)
        if scale <= 1e-9:
            return None
        return _clamp((latest.value - center) / scale, -3.0, 3.0)

    def pct_change(self, as_of: date, *, days: int) -> float | None:
        latest = self.latest(as_of)
        if latest is None:
            return None
        prior = self.point_at_or_before_observation(latest.observation_date - timedelta(days=max(1, int(days))))
        if prior is None or abs(prior.value) <= 1e-12:
            return None
        value = (latest.value / prior.value - 1.0) * 100.0
        return value if math.isfinite(value) else None


@dataclass(frozen=True, slots=True)
class MacroFactorFrame:
    factors_by_key: dict[tuple[date, str], dict[str, float | None]]
    diagnostics: dict[str, Any]
    default_market: str

    def factors_for(self, as_of: date, market: str | None = None) -> dict[str, float | None]:
        market_token = str(market or self.default_market or "").strip().lower()
        values = self.factors_by_key.get((as_of, market_token))
        if values is None and market_token:
            values = self.factors_by_key.get((as_of, self.default_market))
        if values is None:
            values = {}
        return {name: values.get(name) for name in MACRO_FACTOR_NAMES}


def _finite_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _date_key(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return date.min
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return date.min


def _frequency_lag_days(value: Any) -> int:
    token = str(value or "").strip().lower()
    if token.startswith("d"):
        return 1
    if token.startswith("w"):
        return 7
    if token.startswith("m"):
        return 31
    if token.startswith("q"):
        return 90
    if token.startswith("a") or token.startswith("y"):
        return 365
    return 31


def _clamp(value: float | None, lo: float = -3.0, hi: float = 3.0) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return max(float(lo), min(float(hi), float(value)))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return math.sqrt(max(0.0, variance))


def _avg(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return _clamp(sum(clean) / float(len(clean)), -3.0, 3.0)


def _build_series(rows: list[dict[str, Any]]) -> dict[str, _MacroSeries]:
    grouped: dict[str, dict[date, _MacroPoint]] = {}
    for row in rows:
        key = str(row.get("indicator_key") or "").strip()
        value = _finite_float(row.get("value"))
        obs_date = _date_key(row.get("observation_date"))
        if not key or value is None or obs_date == date.min:
            continue
        lag_days = _frequency_lag_days(row.get("frequency"))
        point = _MacroPoint(
            observation_date=obs_date,
            effective_date=obs_date + timedelta(days=lag_days),
            value=float(value),
        )
        grouped.setdefault(key, {})[obs_date] = point

    out: dict[str, _MacroSeries] = {}
    for key, by_date in grouped.items():
        points = tuple(sorted(by_date.values(), key=lambda point: (point.effective_date, point.observation_date)))
        out[key] = _MacroSeries(
            key=key,
            points=points,
            effective_dates=tuple(point.effective_date for point in points),
            observation_dates=tuple(point.observation_date for point in points),
        )
    return out


def _z(series: dict[str, _MacroSeries], key: str, as_of: date, *, min_points: int = 6) -> float | None:
    item = series.get(key)
    if item is None:
        return None
    return item.z_score(as_of, min_points=min_points)


def _yoy(series: dict[str, _MacroSeries], key: str, as_of: date) -> float | None:
    item = series.get(key)
    if item is None:
        return None
    value = item.pct_change(as_of, days=365)
    if value is None:
        return None
    return float(value)


def _spread(
    series: dict[str, _MacroSeries],
    left_key: str,
    right_key: str,
    as_of: date,
    *,
    scale: float = 2.0,
) -> float | None:
    left = series.get(left_key).latest(as_of) if series.get(left_key) is not None else None
    right = series.get(right_key).latest(as_of) if series.get(right_key) is not None else None
    if left is None or right is None:
        return None
    if abs(float(scale)) <= 1e-12:
        return None
    return _clamp((left.value - right.value) / float(scale), -3.0, 3.0)


def _real_money_yoy(
    series: dict[str, _MacroSeries],
    money_keys: tuple[str, ...],
    cpi_keys: tuple[str, ...],
    as_of: date,
) -> float | None:
    money_yoy = next((value for key in money_keys if (value := _yoy(series, key, as_of)) is not None), None)
    cpi_yoy = next((value for key in cpi_keys if (value := _yoy(series, key, as_of)) is not None), None)
    if money_yoy is None:
        return None
    real_yoy = money_yoy - (cpi_yoy or 0.0)
    return _clamp(real_yoy / 10.0, -3.0, 3.0)


def _macro_factors_for_market(
    series: dict[str, _MacroSeries],
    *,
    as_of: date,
    market: str,
) -> dict[str, float | None]:
    market_token = str(market or "").strip().lower()
    is_kr = market_token in {"kr", "kospi", "kosdaq", "korea"}

    us_liquidity = _avg(
        [
            _real_money_yoy(series, ("m2_money_supply",), ("cpi_index", "pce_price_index"), as_of),
            _z(series, "fed_balance_sheet", as_of),
            -_z(series, "overnight_reverse_repo", as_of) if _z(series, "overnight_reverse_repo", as_of) is not None else None,
        ]
    )
    kr_liquidity = _avg(
        [
            _real_money_yoy(series, ("kr_m2", "kr_m1", "kr_lf"), ("kr_cpi", "kr_core_cpi"), as_of),
            _z(series, "kr_household_credit", as_of, min_points=4),
            _z(series, "kr_liquidity_l", as_of, min_points=4),
        ]
    )
    liquidity = _avg([kr_liquidity, us_liquidity]) if is_kr else us_liquidity

    us_rate = _avg(
        [
            _z(series, "fed_funds_rate", as_of),
            _z(series, "sofr", as_of),
            _z(series, "treasury_3m", as_of),
            _z(series, "treasury_10y", as_of),
        ]
    )
    kr_rate = _avg(
        [
            _z(series, "bok_base_rate", as_of),
            _z(series, "call_rate", as_of),
            _z(series, "koribor_3m", as_of),
            _z(series, "cd_91d", as_of),
            _z(series, "kr_treasury_3y", as_of),
        ]
    )
    rate_pressure = _avg([kr_rate, us_rate]) if is_kr else us_rate

    curve = _avg(
        [
            _spread(series, "treasury_10y", "treasury_3m", as_of),
            _spread(series, "kr_treasury_5y", "kr_treasury_3y", as_of, scale=1.0) if is_kr else None,
        ]
    )
    credit_stress = _avg(
        [
            _z(series, "high_yield_oas", as_of),
            _z(series, "corporate_oas", as_of),
            _z(series, "financial_stress_index", as_of),
            _z(series, "kr_corp_bond_3y_aa_minus", as_of) if is_kr else None,
            _z(series, "kr_household_loan_delinquency", as_of, min_points=4) if is_kr else None,
        ]
    )
    fx_pressure = _avg(
        [
            _z(series, "usd_krw", as_of) if is_kr else None,
            _z(series, "fred_usd_krw", as_of) if is_kr else None,
            _z(series, "trade_weighted_dollar", as_of),
        ]
    )
    growth = _avg(
        [
            _clamp((_yoy(series, "industrial_production", as_of) or 0.0) / 10.0, -3.0, 3.0)
            if _yoy(series, "industrial_production", as_of) is not None
            else None,
            _clamp((_yoy(series, "retail_sales", as_of) or 0.0) / 10.0, -3.0, 3.0)
            if _yoy(series, "retail_sales", as_of) is not None
            else None,
            _clamp((_yoy(series, "real_gdp", as_of) or 0.0) / 5.0, -3.0, 3.0)
            if _yoy(series, "real_gdp", as_of) is not None
            else None,
            _z(series, "kr_leading_cyclical_component", as_of, min_points=4) if is_kr else None,
            _z(series, "kr_all_industry_production", as_of, min_points=4) if is_kr else None,
            _z(series, "kr_retail_sales_index", as_of, min_points=4) if is_kr else None,
            _clamp((_finite_latest(series, "kr_gdp_growth", as_of) or 0.0) / 2.0, -3.0, 3.0)
            if is_kr and _finite_latest(series, "kr_gdp_growth", as_of) is not None
            else None,
        ]
    )
    inflation = _avg(
        [
            _clamp((_yoy(series, "cpi_index", as_of) or 0.0) / 5.0, -3.0, 3.0)
            if _yoy(series, "cpi_index", as_of) is not None
            else None,
            _clamp((_yoy(series, "core_cpi_index", as_of) or 0.0) / 5.0, -3.0, 3.0)
            if _yoy(series, "core_cpi_index", as_of) is not None
            else None,
            _z(series, "breakeven_5y", as_of),
            _clamp((_yoy(series, "kr_cpi", as_of) or 0.0) / 5.0, -3.0, 3.0)
            if is_kr and _yoy(series, "kr_cpi", as_of) is not None
            else None,
            _clamp((_yoy(series, "kr_core_cpi", as_of) or 0.0) / 5.0, -3.0, 3.0)
            if is_kr and _yoy(series, "kr_core_cpi", as_of) is not None
            else None,
        ]
    )
    risk_stress = _avg([_z(series, "vix", as_of), credit_stress])

    return {
        "liquidity": liquidity,
        "rate_pressure": rate_pressure,
        "curve": curve,
        "credit_stress": credit_stress,
        "fx_pressure": fx_pressure,
        "growth": growth,
        "inflation": inflation,
        "risk_stress": risk_stress,
    }


def _finite_latest(series: dict[str, _MacroSeries], key: str, as_of: date) -> float | None:
    item = series.get(key)
    if item is None:
        return None
    point = item.latest(as_of)
    return point.value if point is not None and math.isfinite(point.value) else None


def _load_macro_rows(
    repo: Any,
    *,
    rows: list[dict[str, Any]],
    lookback_days: int,
    market: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loader = getattr(repo, "macro_indicator_observation_history", None)
    if not callable(loader) or not rows:
        return [], {"enabled": callable(loader), "macro_rows_loaded": 0}

    dates = [_date_key(row.get("as_of_date")) for row in rows]
    valid_dates = [d for d in dates if d != date.min]
    if not valid_dates:
        return [], {"enabled": True, "macro_rows_loaded": 0}

    start_date = min(valid_dates) - timedelta(days=max(430, int(lookback_days or 0)))
    end_date = max(valid_dates)
    try:
        macro_rows = list(
            loader(
                sources=["fred", "ecos"],
                indicator_keys=list(MACRO_CONDITIONING_INDICATOR_KEYS),
                start_date=start_date,
                end_date=end_date,
                limit=None,
            )
            or []
        )
    except TypeError:
        macro_rows = list(
            loader(
                indicator_keys=list(MACRO_CONDITIONING_INDICATOR_KEYS),
                start_date=start_date,
                end_date=end_date,
                limit=None,
            )
            or []
        )
    except Exception:
        logger.warning(
            "[yellow]Macro interaction history load failed[/yellow] rows=%d market=%s",
            len(rows),
            market,
            exc_info=True,
        )
        return [], {
            "enabled": True,
            "macro_rows_loaded": 0,
            "error": "macro history load failed",
        }
    return macro_rows, {
        "enabled": True,
        "macro_rows_loaded": len(macro_rows),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }


def macro_factor_frame_for_rows(
    repo: Any,
    rows: list[dict[str, Any]],
    *,
    lookback_days: int,
    market: str,
) -> MacroFactorFrame:
    """Builds point-in-time macro factor snapshots for the row dates."""
    market_token = str(market or "").strip().lower()
    if not rows:
        return MacroFactorFrame(
            factors_by_key={},
            diagnostics={"enabled": True, "input_rows": 0, "macro_rows_loaded": 0, "factor_dates": 0},
            default_market=market_token,
        )

    macro_rows, diagnostics = _load_macro_rows(repo, rows=rows, lookback_days=lookback_days, market=market_token)
    diagnostics["input_rows"] = len(rows)
    if not macro_rows:
        diagnostics.update({"series_loaded": 0, "factor_dates": 0, "factor_hit_counts": {}})
        return MacroFactorFrame(factors_by_key={}, diagnostics=diagnostics, default_market=market_token)

    series = _build_series(macro_rows)
    factors_by_key: dict[tuple[date, str], dict[str, float | None]] = {}
    factor_hits: dict[str, int] = {}
    for row in rows:
        as_of = _date_key(row.get("as_of_date"))
        if as_of == date.min:
            continue
        row_market = str(row.get("market") or market_token or "").strip().lower()
        cache_key = (as_of, row_market)
        if cache_key in factors_by_key:
            continue
        factors = _macro_factors_for_market(series, as_of=as_of, market=row_market)
        factors_by_key[cache_key] = factors
        for name, value in factors.items():
            if value is not None:
                factor_hits[name] = factor_hits.get(name, 0) + 1

    diagnostics.update(
        {
            "series_loaded": len(series),
            "factor_dates": len(factors_by_key),
            "factor_hit_counts": dict(sorted(factor_hits.items())),
            "factor_names": list(MACRO_FACTOR_NAMES),
            "conditioning_pair_count": len(MACRO_CONDITIONING_PAIRS),
        }
    )
    return MacroFactorFrame(
        factors_by_key=factors_by_key,
        diagnostics=diagnostics,
        default_market=market_token,
    )


__all__ = [
    "MACRO_CONDITIONING_PAIRS",
    "MACRO_CONDITIONING_INDICATOR_KEYS",
    "MACRO_FACTOR_NAMES",
    "MacroFactorFrame",
    "macro_factor_frame_for_rows",
]
