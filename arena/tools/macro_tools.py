from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import requests

from arena.config import Settings

logger = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_ECOS_BASE = "https://ecos.bok.or.kr/api/KeyStatisticList"

_US_MARKETS: set[str] = {"nasdaq", "nyse", "amex", "us"}

# ECOS KeyStatisticList KEYSTAT_NAME -> our indicator key mapping
_ECOS_INDICATORS: dict[str, str] = {
    "한국은행 기준금리": "bok_base_rate",
    "콜금리(익일물)": "call_rate",
    "국고채수익률(3년)": "kr_treasury_3y",
    "국고채수익률(5년)": "kr_treasury_5y",
    "회사채수익률(3년,AA-)": "corp_bond_3y_aa",
    "원/달러 환율(종가)": "usd_krw",
    "소비자물가지수": "kr_cpi",
    "실업률": "kr_unemployment",
    "고용률": "kr_employment",
    "경제성장률(실질, 계절조정 전기대비)": "kr_gdp_growth",
    "코스피지수": "kospi_index",
    "코스닥지수": "kosdaq_index",
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
    http_timeout: int = 12
    _session: requests.Session = field(default_factory=requests.Session, repr=False)
    _context: dict[str, Any] = field(default_factory=dict, repr=False)

    def set_context(self, context: dict[str, Any]) -> None:
        self._context = context

    def _effective_markets(self) -> set[str]:
        if self._context:
            tm = str(self._context.get("target_market") or "").strip().lower()
            if tm:
                parts = {m.strip() for m in tm.split(",") if m.strip()}
                if parts:
                    return parts
        global_market = str(self.settings.kis_target_market or "").strip().lower()
        if not global_market:
            raise ValueError("target_market is not configured for this agent.")
        return {m.strip() for m in global_market.split(",") if m.strip()}

    def _has_us_market(self) -> bool:
        return bool(self._effective_markets() & _US_MARKETS)

    def _has_kospi_market(self) -> bool:
        return "kospi" in self._effective_markets()

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
        for obs in observations:
            d, val = self._parse_value(obs)
            if val is not None:
                return d, val
        return "", None

    def _compute_cpi_yoy(self) -> tuple[str, float | None]:
        observations = self._fetch_series("CPIAUCSL", limit=14)
        valid = [(d, v) for obs in observations for d, v in [self._parse_value(obs)] if v is not None]
        if len(valid) < 2:
            return (valid[0][0] if valid else ""), None
        latest_date, latest_val = valid[0]
        try:
            latest_dt = date.fromisoformat(latest_date)
        except ValueError:
            return latest_date, None
        for d_str, v in valid[1:]:
            try:
                dt = date.fromisoformat(d_str)
            except ValueError:
                continue
            diff_months = (latest_dt.year - dt.year) * 12 + (latest_dt.month - dt.month)
            if 11 <= diff_months <= 13:
                yoy = ((latest_val - v) / v) * 100.0
                return latest_date, round(yoy, 2)
        return latest_date, None

    def _us_macro(self) -> dict[str, Any]:
        api_key = getattr(self.settings, "fred_api_key", "")
        if not api_key:
            return {}
        indicators: dict[str, Any] = {}
        ff_date, ff_val = self._latest_valid("DFF")
        if ff_val is not None:
            indicators["fed_funds_rate"] = {"value": ff_val, "date": ff_date, "unit": "%"}
        cpi_date, cpi_yoy = self._compute_cpi_yoy()
        if cpi_yoy is not None:
            indicators["cpi_yoy"] = {"value": cpi_yoy, "date": cpi_date, "unit": "%"}
        ur_date, ur_val = self._latest_valid("UNRATE")
        if ur_val is not None:
            indicators["unemployment_rate"] = {"value": ur_val, "date": ur_date, "unit": "%"}
        t10_date, t10_val = self._latest_valid("DGS10")
        if t10_val is not None:
            indicators["treasury_10y"] = {"value": t10_val, "date": t10_date, "unit": "%"}
        t2_date, t2_val = self._latest_valid("DGS2")
        if t2_val is not None:
            indicators["treasury_2y"] = {"value": t2_val, "date": t2_date, "unit": "%"}
        if t10_val is not None and t2_val is not None:
            indicators["yield_spread_10y_2y"] = {"value": round(t10_val - t2_val, 2), "unit": "pp"}
        return indicators

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

    def _kr_macro(self) -> dict[str, Any]:
        rows = self._fetch_ecos_key_stats()
        if not rows:
            return {}
        indicators: dict[str, Any] = {}
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
            unit = _ECOS_UNITS.get(key, str(row.get("UNIT_NAME") or ""))
            indicators[key] = {"value": val, "date": cycle, "unit": unit}

        # Compute KR yield spread (5Y - 3Y)
        t5 = indicators.get("kr_treasury_5y", {}).get("value")
        t3 = indicators.get("kr_treasury_3y", {}).get("value")
        if t5 is not None and t3 is not None:
            indicators["kr_yield_spread_5y_3y"] = {"value": round(t5 - t3, 3), "unit": "pp"}

        return indicators

    # ── Public API ──

    def macro_snapshot(self) -> dict[str, Any]:
        """마켓에 따라 US(FRED) / KR(ECOS) 거시경제지표를 일괄 조회합니다."""
        logger.info("[cyan]TOOL[/cyan] macro_snapshot")

        has_us = self._has_us_market()
        has_kr = self._has_kospi_market()

        indicators: dict[str, Any] = {}
        sources: list[str] = []

        if has_us:
            us = self._us_macro()
            if us:
                indicators.update(us)
                sources.append("fred")
            elif not getattr(self.settings, "fred_api_key", ""):
                indicators["us_error"] = "FRED_API_KEY is not configured"

        if has_kr:
            kr = self._kr_macro()
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
            "source": "+".join(sources),
        }
