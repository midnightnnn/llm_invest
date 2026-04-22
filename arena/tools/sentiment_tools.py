from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote_plus

import requests

from arena.config import Settings
from arena.open_trading.client import OpenTradingClient
from arena.tools._market_scope import MarketScope

logger = logging.getLogger(__name__)

_REDDIT_SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
_REDDIT_USER_AGENT = "llm-arena-bot/1.0"

_SEC_EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
_SEC_SUBMISSIONS = "https://data.sec.gov/submissions"
_SEC_USER_AGENT = "LLMArena support@example.com"
_VIX_HISTORY_CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
_VIX_HISTORY_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
_VKOSPI_INDEX_CODE = "1164"
_VKOSPI_SCALE = 100.0  # KIS returns VKOSPI * 100


def _safe_get(url: str, *, headers: dict | None = None, timeout: int = 10) -> requests.Response | None:
    """HTTP GET with silent failure."""
    try:
        resp = requests.get(url, headers=headers or {}, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as exc:
        logger.warning("[yellow]HTTP fetch failed[/yellow] url=%s err=%s", url[:80], str(exc)[:120])
        return None


@dataclass(slots=True)
class SentimentTools:
    """External unstructured data tools for LLM-driven sentiment analysis."""

    settings: Settings
    http_timeout: int = 10
    ot_client: OpenTradingClient | None = None
    _session: requests.Session = field(default_factory=requests.Session, repr=False)
    _context: dict | None = field(default=None, repr=False)

    def set_context(self, context: dict) -> None:
        """Stores current cycle context for market-aware tool filtering."""
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

    def _analysis_default_ticker(self) -> str:
        """Resolves one self-discovered or portfolio ticker for single-name analysis tools."""
        if not bool(getattr(self.settings, "autonomy_tool_default_candidates_enabled", False)):
            return ""
        if self._context:
            raw = self._context.get("opportunity_working_set") or []
            if isinstance(raw, list):
                for row in raw:
                    if not isinstance(row, dict):
                        continue
                    ticker = str(row.get("ticker") or "").strip().upper()
                    if ticker:
                        return ticker
            raw_candidates = self._context.get("_candidate_tickers") or []
            if isinstance(raw_candidates, list):
                for ticker in raw_candidates:
                    token = str(ticker or "").strip().upper()
                    if token:
                        return token
            portfolio = self._context.get("portfolio") or {}
            if isinstance(portfolio, dict):
                positions = portfolio.get("positions") or {}
                if isinstance(positions, dict):
                    for ticker in positions:
                        token = str(ticker or "").strip().upper()
                        if token:
                            return token
        return ""

    def _ot(self) -> OpenTradingClient:
        """Lazily creates open-trading API client."""
        if self.ot_client is None:
            self.ot_client = OpenTradingClient(self.settings)
        return self.ot_client

    def _fetch_vkospi_history(self, lookback_days: int) -> list[tuple[date, float]]:
        """Fetches VKOSPI daily history from KIS domestic index API."""
        try:
            rows = self._ot().get_domestic_index_daily(
                _VKOSPI_INDEX_CODE,
                max_pages=max(1, lookback_days // 30),
            )
        except Exception as exc:
            logger.warning("[yellow]VKOSPI fetch failed[/yellow] err=%s", str(exc)[:120])
            return []

        series: list[tuple[date, float]] = []
        for row in rows:
            d_raw = str(row.get("stck_bsop_date") or "").strip()
            close_raw = row.get("bstp_nmix_prpr")
            if not d_raw or close_raw is None:
                continue
            try:
                d = datetime.strptime(d_raw, "%Y%m%d").date()
            except Exception:
                continue
            try:
                close = float(close_raw) / _VKOSPI_SCALE
            except (TypeError, ValueError):
                continue
            if close > 0:
                series.append((d, close))

        series.sort(key=lambda x: x[0])
        return series

    def fetch_reddit_sentiment(self, ticker: str, max_posts: int = 10) -> list[dict[str, Any]]:
        """Fetches recent Reddit posts mentioning a ticker from finance subreddits."""
        if not self._has_us_market():
            return [{"error": "fetch_reddit_sentiment is only available for US market agents."}]
        ticker = str(ticker).strip().upper()
        if not ticker:
            return []
        max_posts = max(1, min(int(max_posts), 25))

        logger.info("[cyan]TOOL[/cyan] fetch_reddit_sentiment ticker=%s max_posts=%d", ticker, max_posts)

        posts: list[dict[str, Any]] = []
        headers = {"User-Agent": _REDDIT_USER_AGENT}

        for sub in _REDDIT_SUBREDDITS:
            if len(posts) >= max_posts:
                break

            url = f"https://www.reddit.com/r/{sub}/search.json?q={quote_plus(ticker)}&sort=new&restrict_sr=on&limit=10"
            resp = _safe_get(url, headers=headers, timeout=self.http_timeout)
            if resp is None:
                continue

            try:
                data = resp.json()
            except Exception:
                continue

            for child in (data.get("data", {}).get("children", []))[:max_posts]:
                post = child.get("data", {})
                title = str(post.get("title", "")).strip()
                if not title:
                    continue

                created_utc = post.get("created_utc", 0)
                try:
                    created_str = datetime.fromtimestamp(float(created_utc), tz=timezone.utc).isoformat()
                except Exception:
                    created_str = ""

                posts.append({
                    "title": title,
                    "score": int(post.get("score", 0)),
                    "num_comments": int(post.get("num_comments", 0)),
                    "subreddit": sub,
                    "created": created_str,
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "selftext_snippet": str(post.get("selftext", ""))[:300],
                })

                if len(posts) >= max_posts:
                    break

        return posts

    def fetch_sec_filings(
        self,
        ticker: str = "",
        filing_type: str = "10-K",
        max_items: int = 5,
    ) -> list[dict[str, Any]]:
        """Fetches recent SEC filings for a ticker from EDGAR submissions API."""
        if not self._has_us_market():
            return [{"error": "fetch_sec_filings is only available for US market agents."}]
        ticker = str(ticker or "").strip().upper() or self._analysis_default_ticker()
        if not ticker:
            return []
        max_items = max(1, min(int(max_items), 15))
        filing_type = re.sub(r"[^A-Za-z0-9\-/]", "", str(filing_type).strip().upper()) or "10-K"

        logger.info(
            "[cyan]TOOL[/cyan] fetch_sec_filings ticker=%s type=%s max_items=%d",
            ticker, filing_type, max_items,
        )

        headers = {"User-Agent": _SEC_USER_AGENT, "Accept": "application/json"}

        # Resolve ticker → CIK via SEC company_tickers.json
        cik = ""
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = _safe_get(tickers_url, headers=headers, timeout=self.http_timeout)
        if resp:
            try:
                for entry in resp.json().values():
                    if str(entry.get("ticker", "")).upper() == ticker:
                        cik = str(entry.get("cik_str", ""))
                        break
            except Exception:
                pass

        if not cik:
            return [{"error": f"CIK not found for ticker {ticker}"}]

        # Fetch filings from EDGAR submissions API
        padded = cik.zfill(10)
        sub_url = f"{_SEC_SUBMISSIONS}/CIK{padded}.json"
        resp = _safe_get(sub_url, headers=headers, timeout=self.http_timeout)
        if resp is None:
            return [{"error": f"EDGAR submissions API unavailable for CIK {cik}"}]

        try:
            data = resp.json()
        except Exception:
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])
        entity = str(data.get("name", ticker))

        filings: list[dict[str, Any]] = []
        want = filing_type.upper()
        for i, form in enumerate(forms):
            if len(filings) >= max_items:
                break
            if str(form).upper() != want:
                continue
            acc_raw = str(accessions[i] if i < len(accessions) else "")
            acc_path = acc_raw.replace("-", "")
            doc = str(primary_docs[i]) if i < len(primary_docs) else ""
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_path}/{doc}" if acc_path and doc else ""
            filings.append({
                "form_type": form,
                "filed_date": str(dates[i]) if i < len(dates) else "",
                "entity": entity,
                "description": str(descriptions[i]) if i < len(descriptions) else "",
                "url": filing_url,
            })

        return filings

    def _nasdaq_headers(self) -> dict[str, str]:
        return {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/market-activity/earnings",
        }

    def _parse_vix_history(self, text: str) -> list[tuple[date, float]]:
        rows: list[tuple[date, float]] = []
        raw = str(text or "").strip()
        if not raw:
            return rows

        for line in raw.splitlines()[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            d_raw = parts[0]
            close_raw = parts[-1]
            try:
                if "/" in d_raw:
                    d = datetime.strptime(d_raw, "%m/%d/%Y").date()
                else:
                    d = date.fromisoformat(d_raw)
            except Exception:
                continue
            try:
                close = float(close_raw)
            except (TypeError, ValueError):
                continue
            if close > 0:
                rows.append((d, close))
        return rows

    def _compute_fear_greed(
        self,
        series: list[tuple[date, float]],
        lookback: int,
        *,
        index_name: str,
        source: str,
    ) -> dict[str, Any]:
        """Shared percentile-based fear/greed computation for any volatility index."""
        tail = series[-min(lookback, len(series)) :]
        vals = [v for _, v in tail if v > 0]
        if not vals:
            return {"error": f"{index_name} history parse failed"}

        as_of, last_val = tail[-1]
        below_or_equal = sum(1 for v in vals if v <= last_val)
        percentile = (below_or_equal / float(len(vals))) * 100.0
        score = max(0.0, min(100.0, 100.0 - percentile))
        if score <= 20:
            regime = "Extreme Fear"
        elif score <= 40:
            regime = "Fear"
        elif score <= 60:
            regime = "Neutral"
        elif score <= 80:
            regime = "Greed"
        else:
            regime = "Extreme Greed"

        return {
            "as_of": as_of.isoformat(),
            "fear_greed_score": round(score, 2),
            "regime": regime,
            "volatility_close": round(float(last_val), 4),
            "volatility_index": index_name,
            "volatility_percentile": round(percentile, 2),
            "lookback_days": len(vals),
            "method": "volatility_percentile_proxy",
            "source": source,
        }

    def _compute_breadth_score(self) -> dict[str, Any] | None:
        """Market breadth sub-component (advance/decline ratio)."""
        try:
            client = self._ot()
            if self._has_kospi_market():
                rows = client.get_domestic_market_breadth()
                if not rows:
                    return None
                advance = sum(1 for r in rows if float(r.get("prdy_vrss", 0) or 0) > 0)
                decline = sum(1 for r in rows if float(r.get("prdy_vrss", 0) or 0) < 0)
            else:
                up_rows = client.get_overseas_updown_ranking(gubn="1")
                dn_rows = client.get_overseas_updown_ranking(gubn="0")
                advance = len(up_rows)
                decline = len(dn_rows)
            total = advance + decline
            if total == 0:
                return None
            ad_ratio = advance / total
            score = round(ad_ratio * 100.0, 2)
            return {"advance": advance, "decline": decline, "ad_ratio": round(ad_ratio, 4), "score": score}
        except Exception as exc:
            logger.debug("Breadth score failed: %s", str(exc)[:80])
            return None

    def _compute_momentum_score(self) -> dict[str, Any] | None:
        """Index 20d return as momentum sub-component."""
        try:
            client = self._ot()
            if self._has_kospi_market():
                rows = client.get_domestic_index_daily("0001", max_pages=1)
            else:
                rows = client.get_domestic_index_daily("SPX", max_pages=1)
                if not rows:
                    return None
            if len(rows) < 2:
                return None
            newest = float(rows[0].get("bstp_nmix_prpr") or rows[0].get("stck_clpr") or 0)
            oldest = float(rows[-1].get("bstp_nmix_prpr") or rows[-1].get("stck_clpr") or 0)
            if oldest <= 0:
                return None
            ret_20d = (newest / oldest) - 1.0
            # Map return to 0-100 score: -10% -> 0, 0% -> 50, +10% -> 100
            score = max(0.0, min(100.0, 50.0 + (ret_20d * 500.0)))
            return {"index_return_20d": round(ret_20d, 6), "score": round(score, 2)}
        except Exception as exc:
            logger.debug("Momentum score failed: %s", str(exc)[:80])
            return None

    def _compute_flow_score(self) -> dict[str, Any] | None:
        """KOSPI institutional flow sub-component."""
        if not self._has_kospi_market():
            return None
        try:
            rows = self._ot().get_domestic_foreign_institution_total()
            if not rows:
                return None
            # Aggregate net buy/sell from top rows
            frgn_net = sum(float(r.get("frgn_ntby_tr_pbmn") or r.get("ntby_qty") or 0) for r in rows[:10])
            orgn_net = sum(float(r.get("orgn_ntby_tr_pbmn") or r.get("ntby_qty") or 0) for r in rows[:10])
            combined = frgn_net + orgn_net
            # Positive = bullish, negative = bearish → map to 0-100
            score = max(0.0, min(100.0, 50.0 + (combined / max(abs(combined), 1.0)) * 50.0)) if combined != 0 else 50.0
            return {"foreign_net": frgn_net, "institution_net": orgn_net, "score": round(score, 2)}
        except Exception as exc:
            logger.debug("Flow score failed: %s", str(exc)[:80])
            return None

    def fear_greed_index(self, lookback_days: int = 252) -> dict[str, Any]:
        """Builds a composite market regime indicator from multiple signals.

        KOSPI → VKOSPI + breadth + momentum + institutional flow
        US    → VIX + breadth + momentum
        Score: 0 = extreme fear / risk-off, 100 = extreme greed / risk-on.
        """
        lookback = max(30, min(int(lookback_days), 1500))
        markets = self._effective_markets()
        logger.info("[cyan]TOOL[/cyan] fear_greed_index lookback_days=%d markets=%s", lookback, ",".join(sorted(markets)))

        # ── Core volatility score ──
        vol_result: dict[str, Any] | None = None
        if markets & {"kospi", "kosdaq"}:
            series = self._fetch_vkospi_history(lookback)
            if series:
                vol_result = self._compute_fear_greed(series, lookback, index_name="VKOSPI", source="kis_vkospi")

        if vol_result is None:
            source = ""
            series_vix: list[tuple[date, float]] = []
            for url, src in [(_VIX_HISTORY_CBOE, "cboe_vix"), (_VIX_HISTORY_FRED, "fred_vix")]:
                resp = _safe_get(url, timeout=self.http_timeout)
                if resp is None:
                    continue
                series_vix = self._parse_vix_history(resp.text)
                if series_vix:
                    source = src
                    break
            if series_vix:
                vol_result = self._compute_fear_greed(series_vix, lookback, index_name="VIX", source=source)

        if vol_result is None:
            return {"error": "volatility index history unavailable"}

        # ── Sub-component scores (best-effort) ──
        sub_components: dict[str, Any] = {}
        vol_score = float(vol_result.get("fear_greed_score", 50.0))
        sub_components["volatility"] = {"score": vol_score, "weight": 0.35}

        breadth = self._compute_breadth_score()
        if breadth is not None:
            sub_components["breadth"] = {"score": breadth["score"], "weight": 0.25, "advance": breadth["advance"], "decline": breadth["decline"]}

        momentum = self._compute_momentum_score()
        if momentum is not None:
            sub_components["momentum"] = {"score": momentum["score"], "weight": 0.20, "index_return_20d": momentum["index_return_20d"]}

        flow = self._compute_flow_score()
        if flow is not None:
            sub_components["institutional_flow"] = {"score": flow["score"], "weight": 0.10, "foreign_net": flow["foreign_net"], "institution_net": flow["institution_net"]}

        # ── Compute weighted composite ──
        total_weight = sum(c["weight"] for c in sub_components.values())
        if total_weight > 0:
            regime_score = sum(c["score"] * c["weight"] for c in sub_components.values()) / total_weight
        else:
            regime_score = vol_score
        regime_score = round(max(0.0, min(100.0, regime_score)), 2)

        if regime_score >= 60:
            regime_label = "risk_on"
        elif regime_score >= 40:
            regime_label = "neutral"
        else:
            regime_label = "risk_off"

        # ── Backward-compatible output + new fields ──
        vol_result["regime_score"] = regime_score
        vol_result["regime_label"] = regime_label
        vol_result["sub_components"] = sub_components
        return vol_result

    def _earnings_us(self, ticker: str, days_ahead: int, limit: int) -> dict[str, Any]:
        """US earnings from Nasdaq calendar API."""
        token = str(ticker or "").strip().upper()
        start = datetime.now(timezone.utc).date()
        rows: list[dict[str, Any]] = []
        scanned = 0

        for offset in range(days_ahead + 1):
            d = start + timedelta(days=offset)
            url = f"https://api.nasdaq.com/api/calendar/earnings?date={d.isoformat()}"
            resp = _safe_get(url, headers=self._nasdaq_headers(), timeout=self.http_timeout)
            scanned += 1
            if resp is None:
                continue
            try:
                payload = resp.json()
            except Exception:
                continue

            day_rows = (((payload or {}).get("data") or {}).get("rows") or [])
            if not isinstance(day_rows, list):
                continue

            for item in day_rows:
                if not isinstance(item, dict):
                    continue
                sym = str(item.get("symbol") or "").strip().upper()
                if not sym:
                    continue
                if token and sym != token:
                    continue
                rows.append({
                    "date": d.isoformat(),
                    "symbol": sym,
                    "name": str(item.get("name") or "").strip(),
                    "event_type": "earnings",
                    "time": str(item.get("time") or "").strip(),
                    "eps_forecast": str(item.get("epsForecast") or "").strip(),
                    "num_estimates": str(item.get("noOfEsts") or "").strip(),
                    "last_year_report_date": str(item.get("lastYearRptDt") or "").strip(),
                    "last_year_eps": str(item.get("lastYearEPS") or "").strip(),
                })
                if len(rows) >= limit:
                    break
            if len(rows) >= limit:
                break
            if token and rows:
                break

        return {
            "ticker": token or None,
            "market": "us",
            "start_date": start.isoformat(),
            "days_ahead": days_ahead,
            "days_scanned": scanned,
            "count": len(rows),
            "rows": rows[:limit],
            "source": "nasdaq_calendar_api",
        }

    def _earnings_kospi(self, ticker: str, days_ahead: int, limit: int) -> dict[str, Any]:
        """KOSPI events: dividends + earnings estimates from KIS API."""
        token = str(ticker or "").strip().upper()
        start = datetime.now(timezone.utc).date()
        end = start + timedelta(days=days_ahead)
        rows: list[dict[str, Any]] = []

        # Dividends via KSD
        try:
            client = self._ot()
            div_rows = client.get_domestic_ksdinfo_dividend(
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                sht_cd=token,
            )
            for item in div_rows:
                sym = str(item.get("sht_cd") or "").strip()
                if not sym:
                    continue
                if token and sym != token:
                    continue
                rows.append({
                    "date": str(item.get("record_date") or item.get("bsop_date") or "").strip(),
                    "symbol": sym,
                    "name": str(item.get("isin_name") or item.get("prdt_name") or "").strip(),
                    "event_type": "dividend",
                    "dividend_per_share": str(item.get("per_sto_divi_amt") or "").strip(),
                })
                if len(rows) >= limit:
                    break
        except Exception as exc:
            logger.debug("KOSPI dividend calendar failed: %s", str(exc)[:80])

        # Earnings estimates for specific ticker
        if token and len(rows) < limit:
            try:
                est_rows = self._ot().get_domestic_estimate_perform(token)
                for item in est_rows:
                    rows.append({
                        "date": str(item.get("stlm_dt") or "").strip(),
                        "symbol": token,
                        "name": "",
                        "event_type": "earnings_estimate",
                        "eps_estimate": str(item.get("eps") or "").strip(),
                        "sales_estimate": str(item.get("sale_account") or "").strip(),
                    })
                    if len(rows) >= limit:
                        break
            except Exception as exc:
                logger.debug("KOSPI estimate perform failed: %s", str(exc)[:80])

        return {
            "ticker": token or None,
            "market": "kospi",
            "start_date": start.isoformat(),
            "days_ahead": days_ahead,
            "count": len(rows),
            "rows": rows[:limit],
            "source": "kis_ksdinfo",
        }

    def earnings_calendar(
        self,
        ticker: str = "",
        days_ahead: int = 14,
        limit: int = 30,
    ) -> dict[str, Any]:
        """Fetches upcoming earnings/dividend events. KOSPI uses KIS API, US uses Nasdaq calendar."""
        days = max(1, min(int(days_ahead), 45))
        lim = max(1, min(int(limit), 200))
        logger.info(
            "[cyan]TOOL[/cyan] earnings_calendar ticker=%s days_ahead=%d limit=%d",
            str(ticker or "").strip().upper() or "(all)",
            days,
            lim,
        )

        if self._has_kospi_market():
            return self._earnings_kospi(ticker, days, lim)
        if self._has_us_market():
            return self._earnings_us(ticker, days, lim)
        return {"error": "no supported market configured", "rows": []}
