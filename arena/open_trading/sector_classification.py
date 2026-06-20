from __future__ import annotations

import re
from typing import Any

import requests


SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"

KIS_INDUSTRY_SECTOR_BY_CODE: dict[str, str] = {
    "0005": "Consumer Staples",
    "0006": "Consumer Discretionary",
    "0007": "Materials",
    "0008": "Materials",
    "0009": "Health Care",
    "0010": "Materials",
    "0011": "Materials",
    "0012": "Industrials",
    "0013": "Technology",
    "0014": "Health Care",
    "0015": "Industrials",
    "0024": "Financials",
    "0025": "Financials",
}

SEC_OWNER_ORG_SECTOR_BY_CODE: dict[str, str] = {
    "04": "Financials",
    "06": "Technology",
}

_SEC_LABEL_SECTORS: tuple[tuple[str, str], ...] = (
    ("technology", "Technology"),
    ("finance", "Financials"),
    ("financial", "Financials"),
    ("life sciences", "Health Care"),
    ("health care", "Health Care"),
    ("healthcare", "Health Care"),
    ("energy", "Energy"),
    ("real estate", "Real Estate"),
    ("utilities", "Utilities"),
    ("manufacturing", "Industrials"),
    ("transportation", "Industrials"),
    ("consumer", "Consumer Discretionary"),
    ("trade", "Consumer Discretionary"),
)


def _clean_code(value: Any) -> str:
    return re.sub(r"\D", "", str(value or "").strip())


def sector_from_kis_industry_code(code: Any) -> str | None:
    token = _clean_code(code).zfill(4)[-4:]
    if not token or token == "0000":
        return None
    return KIS_INDUSTRY_SECTOR_BY_CODE.get(token)


def kis_industry_name(code: Any) -> str | None:
    token = _clean_code(code).zfill(4)[-4:]
    if not token:
        return None
    return f"KIS industry {token}"


def sector_from_sec_owner_org(owner_org: Any) -> str | None:
    text = str(owner_org or "").strip()
    if not text:
        return None
    lowered = text.lower()
    for needle, sector in _SEC_LABEL_SECTORS:
        if needle in lowered:
            return sector
    match = re.match(r"^(\d{2})\b", text)
    if match:
        sector = SEC_OWNER_ORG_SECTOR_BY_CODE.get(match.group(1))
        if sector:
            return sector
    return None


def _sec_owner_org_code(owner_org: Any) -> str | None:
    text = str(owner_org or "").strip()
    if not text:
        return None
    match = re.match(r"^(\d{2})\b", text)
    return match.group(1) if match else None


def _sec_owner_org_label(owner_org: Any) -> str | None:
    text = str(owner_org or "").strip()
    if not text:
        return None
    return re.sub(r"^\d{2}\s+", "", text).strip() or None


class SECClassificationClient:
    """Fetches public SEC company classification hints from EDGAR submissions."""

    def __init__(
        self,
        *,
        user_agent: str,
        session: requests.Session | None = None,
        timeout: int = 10,
        ticker_map_url: str = SEC_COMPANY_TICKERS_URL,
        submissions_url_template: str = SEC_SUBMISSIONS_URL_TEMPLATE,
    ) -> None:
        clean_user_agent = str(user_agent or "").strip()
        if not clean_user_agent:
            raise ValueError("SEC user_agent is required")
        self.user_agent = clean_user_agent
        self.session = session or requests.Session()
        self.timeout = max(1, int(timeout))
        self.ticker_map_url = ticker_map_url
        self.submissions_url_template = submissions_url_template
        self._ticker_map: dict[str, str] | None = None

    def _get_json(self, url: str) -> dict[str, Any]:
        response = self.session.get(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def load_ticker_map(self) -> dict[str, str]:
        if self._ticker_map is not None:
            return dict(self._ticker_map)
        data = self._get_json(self.ticker_map_url)
        mapping: dict[str, str] = {}
        for item in data.values():
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker") or "").strip().upper()
            cik_raw = item.get("cik_str")
            if not ticker or cik_raw is None:
                continue
            try:
                mapping[ticker] = f"{int(cik_raw):010d}"
            except (TypeError, ValueError):
                continue
        self._ticker_map = mapping
        return dict(mapping)

    def classify_ticker(self, ticker: Any) -> dict[str, Any]:
        token = str(ticker or "").strip().upper()
        if not token:
            return {}
        cik = self.load_ticker_map().get(token)
        if not cik:
            return {}
        data = self._get_json(self.submissions_url_template.format(cik=cik))
        owner_org = str(data.get("ownerOrg") or "").strip()
        sic = str(data.get("sic") or "").strip()
        sic_description = str(data.get("sicDescription") or "").strip()
        sector = sector_from_sec_owner_org(owner_org)
        industry_code = _sec_owner_org_code(owner_org) or sic or None
        industry_name = _sec_owner_org_label(owner_org) or sic_description or None
        return {
            "sector": sector,
            "industry_code": industry_code,
            "industry_name": industry_name,
            "classification_source": "sec_edgar" if (owner_org or sic or sic_description) else None,
        }
