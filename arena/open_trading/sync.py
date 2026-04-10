from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import stdev
from typing import Any

import pandas as pd

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.models import AccountSnapshot, Position, utc_now

from .client import OpenTradingClient
from .exchange_codes import (
    normalize_us_order_exchange,
    order_to_quote_exchange,
    quote_to_order_exchange,
    target_market_default_us_order_exchange,
    us_order_exchange_candidates,
    us_quote_exchange_candidates,
)

logger = logging.getLogger(__name__)

_US_TARGET_TO_QUOTE_EXCHANGE: dict[str, str] = {
    "nasdaq": "NAS",
    "nyse": "NYS",
    "amex": "AMS",
    "us": "NAS",
}
_US_TARGET_TO_BALANCE_MARKET_CODE: dict[str, str] = {
    "NASD": "01",
    "NYSE": "02",
    "AMEX": "03",
}


@dataclass(slots=True)
class MarketSyncResult:
    """Summarizes one market sync execution."""

    inserted_rows: int
    attempted_tickers: int
    failed_tickers: list[str]


def _to_float(value: object, default: float = 0.0) -> float:
    """Converts mixed API values to float with safe fallback."""
    try:
        if value is None:
            return default
        text = str(value).strip().replace(",", "")
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _finite_float_or_none(value: object) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _has_daily_feature_metrics(row: dict[str, object]) -> bool:
    """True when quote enrichment has daily-history based feature metrics."""
    return all(
        _finite_float_or_none(row.get(key)) is not None
        for key in ("ret_5d", "ret_20d", "volatility_20d")
    )


def _pick_str(row: dict[str, object], keys: list[str]) -> str:
    """Returns the first non-empty string value for keys."""
    for key in keys:
        if key in row:
            value = str(row.get(key, "")).strip()
            if value:
                return value
    return ""


def _window_return(closes: list[float], window: int) -> float | None:
    """Computes trailing return for the provided lookback window."""
    if len(closes) <= window:
        return None
    base = closes[-(window + 1)]
    if base <= 0:
        return None
    return (closes[-1] / base) - 1.0


def _volatility_20d(closes: list[float]) -> float | None:
    """Computes rolling 20-day return volatility (daily std dev)."""
    if len(closes) <= 20:
        return None
    daily_returns: list[float] = []
    for idx in range(1, len(closes)):
        prev = closes[idx - 1]
        now = closes[idx]
        if prev > 0:
            daily_returns.append((now / prev) - 1.0)
    if len(daily_returns) < 20:
        return None
    sample = daily_returns[-20:]
    return float(stdev(sample))


def _quote_to_order_exchange(exchange_code: str) -> str:
    """Converts quote exchange code to KIS overseas order exchange code."""
    return quote_to_order_exchange(exchange_code)


def _order_to_quote_exchange(exchange_code: str) -> str:
    """Converts order exchange code to quote exchange code."""
    return order_to_quote_exchange(exchange_code)


def _us_quote_has_payload(payload: dict[str, object]) -> bool:
    """Returns True when a KIS overseas quote payload looks populated."""
    last = _to_float(payload.get("last"), default=0.0)
    if last > 0:
        return True
    rsym = str(payload.get("rsym") or "").strip().upper()
    return bool(rsym)


def _probe_us_quote_exchange_with_client(
    *,
    client: OpenTradingClient,
    settings: Settings,
    ticker: str,
    preferred_excd: object = "",
) -> tuple[str, dict[str, Any]]:
    """Finds the first KIS quote exchange that returns a populated snapshot."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return "", {}
    for excd in us_quote_exchange_candidates(preferred_excd, settings.us_quote_exchanges):
        try:
            quote = client.get_overseas_price(ticker=symbol, excd=excd)
        except Exception:
            continue
        if _us_quote_has_payload(quote):
            return excd, quote
    return "", {}


def _extract_us_tick_size(detail: dict[str, object], last_price_usd: float) -> float | None:
    """Extracts tick size from KIS detail payload, with a conservative fallback."""
    for key in ("tick_size", "tick", "hoga_unit", "ovrs_hoga_unit", "pric_unit"):
        if key in detail:
            val = _to_float(detail.get(key), default=0.0)
            if val > 0:
                return float(val)
    if last_price_usd <= 0:
        return None
    if last_price_usd >= 1.0:
        return 0.01
    return 0.0001


class MarketDataSyncService:
    """Loads latest market data from open-trading API into BigQuery features."""

    def __init__(self, settings: Settings, repo: BigQueryRepository, client: OpenTradingClient | None = None):
        self.settings = settings
        self.repo = repo
        self.client = client or OpenTradingClient(settings)
        self._usd_krw_daily_fx: dict[date, float] = {}
        self._usd_krw_daily_fx_range: tuple[date, date] | None = None
        self._usd_krw_latest_fx: float | None = None
        self._known_us_quote_exchange_cache: dict[str, str] = {}

    _US_MARKETS: set[str] = {"nasdaq", "nyse", "amex", "us"}

    def _parsed_markets(self) -> set[str]:
        return {m.strip() for m in str(self.settings.kis_target_market or "").split(",") if m.strip()}

    def _has_us_market(self) -> bool:
        return bool(self._parsed_markets() & self._US_MARKETS)

    def _has_kospi_market(self) -> bool:
        return "kospi" in self._parsed_markets()

    def _is_us_market(self) -> bool:
        """Legacy compat: True only if ALL markets are US (no kospi)."""
        markets = self._parsed_markets()
        return bool(markets & self._US_MARKETS) and "kospi" not in markets

    def _daily_source(self, market: str = "") -> str:
        """Returns source tag for a specific market. If not given, uses first parsed market."""
        if not market:
            markets = self._parsed_markets()
            market = "us" if markets & self._US_MARKETS else next(iter(markets), "us")
        if market in self._US_MARKETS:
            return "open_trading_us"
        return f"open_trading_{market}"

    def _quote_source(self, market: str = "") -> str:
        if not market:
            markets = self._parsed_markets()
            market = "us" if markets & self._US_MARKETS else next(iter(markets), "us")
        if market in self._US_MARKETS:
            return "open_trading_us_quote"
        return f"open_trading_{market}_quote"

    @staticmethod
    def _is_kospi_ticker(ticker: str) -> bool:
        return ticker.isdigit() and len(ticker) == 6

    def _all_sources(self) -> list[str]:
        """Returns all daily+quote source tags for configured markets."""
        sources: list[str] = []
        if self._has_us_market():
            sources.extend(["open_trading_us", "open_trading_us_quote"])
        if self._has_kospi_market():
            sources.extend(["open_trading_kospi", "open_trading_kospi_quote"])
        return sources or [self._daily_source(), self._quote_source()]

    def _discover_us_symbols(self) -> list[dict[str, str]]:
        """Discovers US symbols via KIS search API sorted by market cap.

        The KIS screening API (HHDFS76410000) returns ~100 unique tickers per
        call.  To reach the per-exchange cap we issue multiple calls with
        non-overlapping price bands so the result sets are disjoint.
        """
        cap = max(1, int(self.settings.universe_per_exchange_cap))
        exchanges = [("NAS", "NAS"), ("NYS", "NYS")]
        # Price bands (USD) — disjoint ranges to maximise unique tickers.
        _PRICE_BANDS: list[tuple[float | None, float | None]] = [
            (None, None),       # unconstrained (top ~100 by default)
            (100.0, 99999.0),   # mid-to-large cap
            (30.0, 99.99),      # mid cap
            (10.0, 29.99),      # small-to-mid cap
        ]
        all_rows: list[dict[str, str]] = []
        existing: set[str] = set()

        def add_symbol(ticker: object, quote_excd: str = "") -> None:
            token = str(ticker or "").strip().upper()
            if not token or token in existing or token[:1].isdigit():
                return
            normalized_excd = str(quote_excd or "").strip().upper()
            if not normalized_excd:
                normalized_excd = self._known_us_quote_exchange(token)
            if not normalized_excd:
                normalized_excd, _ = self._probe_us_quote_exchange(ticker=token)
            all_rows.append({"ticker": token, "quote_excd": normalized_excd or "NAS"})
            existing.add(token)

        for excd, quote_excd in exchanges:
            excd_seen: set[str] = set()
            excd_raw_total = 0
            for price_min, price_max in _PRICE_BANDS:
                if len(excd_seen) >= cap:
                    break
                try:
                    raw = self.client.search_overseas_stocks(
                        excd=excd,
                        price_min=price_min,
                        price_max=price_max,
                        max_pages=20,
                    )
                except Exception as exc:
                    logger.warning(
                        "[yellow]US discovery band failed[/yellow] excd=%s band=(%s,%s) err=%s",
                        excd, str(price_min), str(price_max), str(exc),
                    )
                    raw = []
                excd_raw_total += len(raw)
                # Sort by market cap descending
                raw.sort(
                    key=lambda r: float(str(r.get("valx") or "0").replace(",", "") or "0"),
                        reverse=True,
                    )
                for row in raw:
                    if len(excd_seen) >= cap:
                        break
                    ticker = str(row.get("symb") or row.get("rsym") or "").strip().upper()
                    if not ticker or ticker in excd_seen or ticker in existing:
                        continue
                    add_symbol(ticker, quote_excd=quote_excd)
                    excd_seen.add(ticker)
            logger.info(
                "[cyan]US discovery[/cyan] excd=%s api_rows=%d unique=%d cap=%d",
                excd, excd_raw_total, len(excd_seen), cap,
            )

        # Always include benchmark ETFs
        for bench in ("SPY", "QQQ", "DIA"):
            add_symbol(bench, quote_excd="NAS")

        # Always include currently held tickers
        if hasattr(self.repo, "get_all_held_tickers"):
            try:
                held = self.repo.get_all_held_tickers()
                for t in held:
                    add_symbol(t)
            except Exception:
                pass

        return all_rows

    def _discover_kospi_symbols(self) -> list[dict[str, str]]:
        """Discovers KOSPI symbols via KIS domestic ranking APIs."""
        cap = max(1, int(self.settings.universe_per_exchange_cap))
        all_rows: list[dict[str, str]] = []
        seen: set[str] = set()
        # ticker → Korean name mapping (populated from ranking API responses)
        self._kospi_ticker_names: dict[str, str] = getattr(self, "_kospi_ticker_names", {})

        def add_symbol(ticker: object, name: str = "") -> None:
            token = str(ticker or "").strip().upper()
            if not self._is_kospi_ticker(token):
                return
            if name:
                self._kospi_ticker_names[token] = name
            if token in seen:
                return
            all_rows.append({"ticker": token, "quote_excd": "KRX"})
            seen.add(token)

        # Explicit KR tickers in default_universe should always be honored.
        for ticker in self.settings.default_universe:
            add_symbol(ticker)

        # Always keep the KOSPI 200 ETF benchmark available.
        add_symbol("069500")

        # Always include currently held domestic tickers.
        if hasattr(self.repo, "get_all_held_tickers"):
            try:
                for ticker in self.repo.get_all_held_tickers():
                    add_symbol(ticker)
            except Exception:
                pass

        discovery_specs: list[tuple[str, Any]] = [
            ("market_cap", self.client.get_domestic_market_cap_ranking),
            ("top_interest", self.client.get_domestic_top_interest_stock),
            ("volume_rank", self.client.get_domestic_volume_rank),
        ]
        for label, loader in discovery_specs:
            try:
                raw = loader(market_scope="0001")
            except Exception as exc:
                logger.warning("[yellow]KOSPI discovery failed[/yellow] source=%s err=%s", label, str(exc))
                continue
            before = len(seen)
            for row in raw:
                sym = row.get("mksc_shrn_iscd") or row.get("stck_shrn_iscd") or row.get("pdno")
                name = str(row.get("hts_kor_isnm") or row.get("kor_isnm") or "").strip()
                add_symbol(sym, name)
                if len(seen) >= cap:
                    break
            logger.info(
                "[cyan]KOSPI discovery[/cyan] source=%s api_rows=%d unique_added=%d cap=%d",
                label,
                len(raw),
                max(0, len(seen) - before),
                cap,
            )
            if len(seen) >= cap:
                break

        return all_rows

    def _target_symbols(self) -> list[dict[str, str]]:
        """Selects target symbols with exchange hints. Supports combined markets."""
        has_us = self._has_us_market()
        has_kospi = self._has_kospi_market()
        result: list[dict[str, str]] = []
        seen: set[str] = set()

        if has_kospi:
            kospi_symbols = self._discover_kospi_symbols()
            for symbol in kospi_symbols:
                t = symbol["ticker"]
                if t not in seen:
                    result.append(symbol)
                    seen.add(t)

        if has_us:
            us_symbols = self._discover_us_symbols()
            for s in us_symbols:
                t = s["ticker"]
                if t not in seen:
                    result.append(s)
                    seen.add(t)

        if not result:
            # Fallback
            target = str(self.settings.kis_target_market or "us").strip()
            tickers = [str(t).strip().upper() for t in self.settings.default_universe if str(t).strip() and not str(t).strip()[:1].isdigit()]
            fixed_excd = _US_TARGET_TO_QUOTE_EXCHANGE.get(target, "")
            for t in list(dict.fromkeys(tickers)):
                result.append({"ticker": t, "quote_excd": fixed_excd})

        return result

    def _include_missing_daily_feature_symbols(self, symbols: list[dict[str, str]]) -> list[dict[str, str]]:
        """Adds existing latest tickers with missing daily features to the daily sync target set."""
        loader = getattr(self.repo, "latest_missing_daily_feature_tickers", None)
        if not callable(loader):
            return symbols

        limit = max(200, min(max(int(self.settings.universe_per_exchange_cap) * 4, 200), 2_000))
        try:
            rows = loader(sources=self._all_sources(), limit=limit)
        except Exception as exc:
            logger.warning("[yellow]Missing daily feature scan skipped[/yellow] err=%s", str(exc))
            return symbols
        if not rows:
            return symbols

        out = list(symbols)
        seen = {str(row.get("ticker") or "").strip().upper() for row in out if str(row.get("ticker") or "").strip()}
        added = 0
        for row in rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker or ticker in seen:
                continue
            if self._is_kospi_ticker(ticker):
                if not self._has_kospi_market():
                    continue
                out.append({"ticker": ticker, "quote_excd": "KRX"})
            else:
                if not self._has_us_market() or ticker[:1].isdigit():
                    continue
                exchange_code = str(row.get("exchange_code") or "").strip().upper()
                quote_excd = _order_to_quote_exchange(exchange_code)
                if not quote_excd:
                    quote_excd = self._known_us_quote_exchange(ticker)
                if not quote_excd:
                    quote_excd = str(self.settings.kis_overseas_quote_excd or "NAS").strip().upper() or "NAS"
                out.append({"ticker": ticker, "quote_excd": quote_excd})
            seen.add(ticker)
            added += 1

        if added:
            logger.info("[cyan]Missing daily feature backfill targets added[/cyan] count=%d", added)
        return out

    @staticmethod
    def _parse_chart_date(token: object) -> datetime | None:
        """Parses YYYYMMDD tokens used across KIS daily chart APIs."""
        raw = str(token or "").strip()
        if raw.isdigit() and len(raw) == 8:
            try:
                return datetime.strptime(raw, "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_chart_series(
        candles: list[dict[str, object]],
        *,
        close_keys: tuple[str, ...],
    ) -> list[tuple[datetime, float]]:
        """Extracts ordered (date, close) pairs from mixed KIS daily chart payloads."""
        if not candles:
            return []

        frame = pd.DataFrame(candles)
        if frame.empty:
            return []

        date_col = next((col for col in ("xymd", "stck_bsop_date") if col in frame.columns), "")
        close_col = next((col for col in close_keys if col in frame.columns), "")
        if not date_col or not close_col:
            return []

        frame = frame[[date_col, close_col]].copy()
        frame = frame.dropna(subset=[date_col, close_col])
        if frame.empty:
            return []

        frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
        frame = frame.dropna(subset=[close_col]).sort_values(date_col)
        out: list[tuple[datetime, float]] = []
        for token, px in frame[[date_col, close_col]].itertuples(index=False, name=None):
            parsed = MarketDataSyncService._parse_chart_date(token)
            if parsed is None:
                continue
            out.append((parsed, float(px)))
        return out

    def _build_feature_rows(
        self,
        *,
        ticker: str,
        candles: list[dict[str, object]],
        close_keys: tuple[str, ...],
        quote_currency: str,
        source: str,
        since_date: date | None,
        exchange_code: str,
        instrument_id: str,
        fx_by_date: dict[date, float] | None = None,
        default_fx: float = 1.0,
    ) -> list[dict[str, object]]:
        """Builds normalized market_features rows from daily candle history."""
        series = self._extract_chart_series(candles, close_keys=close_keys)
        if not series:
            return []

        rows: list[dict[str, object]] = []
        krw_closes: list[float] = []
        fx_rate = float(default_fx) if quote_currency != "KRW" else 1.0

        for idx, (as_of_ts, native_close) in enumerate(series):
            if quote_currency != "KRW":
                mapped_fx = float((fx_by_date or {}).get(as_of_ts.date()) or 0.0)
                if mapped_fx > 0:
                    fx_rate = mapped_fx
                if fx_rate <= 0:
                    continue
                close_price_krw = float(native_close * fx_rate)
                fx_used = float(fx_rate)
            else:
                close_price_krw = float(native_close)
                fx_used = 1.0

            krw_closes.append(close_price_krw)
            if since_date is not None and as_of_ts.date() < since_date:
                continue

            sub = krw_closes[: idx + 1]
            ret_5d = _window_return(sub, 5)
            ret_20d = _window_return(sub, 20)
            vol_20d = _volatility_20d(sub)

            sentiment = 0.0
            if idx > 0 and krw_closes[idx - 1] > 0:
                daily_ret = (krw_closes[idx] / krw_closes[idx - 1]) - 1.0
                sentiment = max(-1.0, min(1.0, float(daily_ret) * 10.0))

            rows.append(
                {
                    "as_of_ts": as_of_ts,
                    "ticker": ticker,
                    "exchange_code": exchange_code,
                    "instrument_id": instrument_id,
                    "close_price_krw": float(close_price_krw),
                    "close_price_native": float(native_close),
                    "quote_currency": quote_currency,
                    "fx_rate_used": float(fx_used),
                    "ret_5d": float(ret_5d) if ret_5d is not None else None,
                    "ret_20d": float(ret_20d) if ret_20d is not None else None,
                    "volatility_20d": float(vol_20d) if vol_20d is not None else None,
                    "sentiment_score": float(sentiment),
                    "source": source,
                }
            )

        return rows

    def _ensure_usd_krw_daily_fx(self, candles: list[dict[str, object]]) -> dict[date, float]:
        """Loads USD/KRW daily FX rows once per sync window when configured."""
        symbol = str(self.settings.usd_krw_fx_symbol or "").strip().upper()
        if not symbol:
            return {}

        series = self._extract_chart_series(
            candles,
            close_keys=("clos", "ovrs_nmix_prpr"),
        )
        if not series:
            return {}

        start_date = series[0][0].date()
        end_date = series[-1][0].date()
        if self._usd_krw_daily_fx_range is not None:
            cached_start, cached_end = self._usd_krw_daily_fx_range
            if cached_start <= start_date and cached_end >= end_date and self._usd_krw_daily_fx:
                return self._usd_krw_daily_fx

        try:
            fx_rows = self.client.get_usd_krw_daily_chart(
                symbol=symbol,
                start_date=start_date.strftime("%Y%m%d"),
                end_date=end_date.strftime("%Y%m%d"),
                market_div_code=self.settings.usd_krw_fx_market_div_code,
                period="D",
                max_pages=8,
            )
        except Exception as exc:
            logger.warning("[yellow]USD/KRW daily FX fetch failed[/yellow] symbol=%s err=%s", symbol, str(exc))
            return self._usd_krw_daily_fx

        fx_series = self._extract_chart_series(
            fx_rows,
            close_keys=("ovrs_nmix_prpr", "clos"),
        )
        if not fx_series:
            return self._usd_krw_daily_fx

        self._usd_krw_daily_fx = {ts.date(): float(px) for ts, px in fx_series if float(px) > 0}
        self._usd_krw_daily_fx_range = (fx_series[0][0].date(), fx_series[-1][0].date())
        return self._usd_krw_daily_fx

    def _latest_usd_krw_fx_rate(self) -> float:
        """Returns the freshest available USD/KRW rate for intraday quote sync."""
        if self._usd_krw_latest_fx is not None and self._usd_krw_latest_fx > 0:
            return float(self._usd_krw_latest_fx)

        symbol = str(self.settings.usd_krw_fx_symbol or "").strip().upper()
        if not symbol:
            raise RuntimeError("USD/KRW FX symbol not configured")

        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=7)).strftime("%Y%m%d")
        end = today.strftime("%Y%m%d")
        try:
            fx_rows = self.client.get_usd_krw_daily_chart(
                symbol=symbol,
                start_date=start,
                end_date=end,
                market_div_code=self.settings.usd_krw_fx_market_div_code,
                period="D",
                max_pages=4,
            )
            fx_series = self._extract_chart_series(
                fx_rows,
                close_keys=("ovrs_nmix_prpr", "clos"),
            )
        except Exception as exc:
            logger.warning("[yellow]USD/KRW latest FX fetch failed[/yellow] symbol=%s err=%s", symbol, str(exc))
            raise RuntimeError(f"USD/KRW latest FX fetch failed for symbol={symbol}") from exc

        if not fx_series:
            raise RuntimeError(f"USD/KRW latest FX unavailable for symbol={symbol}")

        self._usd_krw_latest_fx = float(fx_series[-1][1])
        return float(self._usd_krw_latest_fx)

    def _quote_exchange_candidates(self, preferred_excd: str = "") -> list[str]:
        """Returns quote exchange candidates for US tickers in retry order."""
        return us_quote_exchange_candidates(preferred_excd, self.settings.us_quote_exchanges)

    def _prime_known_us_quote_exchanges(self, tickers: list[str]) -> None:
        """Preloads stored exchange hints in one repo call to avoid per-ticker BQ scans."""
        loader = getattr(self.repo, "latest_instrument_map", None)
        if not callable(loader):
            return

        pending: list[str] = []
        for ticker in tickers:
            token = str(ticker or "").strip().upper()
            if token and token not in self._known_us_quote_exchange_cache and token not in pending:
                pending.append(token)
        if not pending:
            return

        try:
            instrument_map = loader(pending)
        except Exception:
            return

        for token in pending:
            row = instrument_map.get(token) or {}
            self._known_us_quote_exchange_cache[token] = _order_to_quote_exchange(row.get("exchange_code") or "")

    def _known_us_quote_exchange(self, ticker: str) -> str:
        """Returns the latest stored KIS quote exchange hint for a ticker."""
        token = str(ticker or "").strip().upper()
        if not token:
            return ""
        self._prime_known_us_quote_exchanges([token])
        return self._known_us_quote_exchange_cache.get(token, "")

    def _probe_us_quote_exchange(self, *, ticker: str, preferred_excd: str = "") -> tuple[str, dict[str, Any]]:
        """Finds the first KIS quote exchange that returns a populated snapshot."""
        return _probe_us_quote_exchange_with_client(
            client=self.client,
            settings=self.settings,
            ticker=ticker,
            preferred_excd=[preferred_excd, self._known_us_quote_exchange(ticker)],
        )

    def _sync_us_daily(
        self,
        *,
        ticker: str,
        preferred_excd: str,
        source: str,
        since_date: date | None,
    ) -> tuple[list[dict[str, object]], dict[str, Any]]:
        """Builds US daily rows with exchange fallback probing."""
        last_exc: Exception | None = None
        resolved_excd, _ = self._probe_us_quote_exchange(ticker=ticker, preferred_excd=preferred_excd)
        candidates: list[str] = []
        for excd in [resolved_excd, *self._quote_exchange_candidates(preferred_excd)]:
            token = str(excd or "").strip().upper()
            if token and token not in candidates:
                candidates.append(token)

        for excd in candidates:
            try:
                candles = self.client.get_overseas_daily_price(
                    ticker=ticker,
                    excd=excd,
                    bymd="",
                    gubn="0",
                    modp="1",
                )
            except Exception as exc:
                last_exc = exc
                continue
            if not candles:
                continue

            order_excd = _quote_to_order_exchange(excd)
            if not order_excd:
                continue
            instrument_id = f"{order_excd}:{ticker}"
            fx_by_date = self._ensure_usd_krw_daily_fx(candles)
            fallback_fx = 0.0
            if fx_by_date:
                for _, rate in sorted(fx_by_date.items()):
                    if float(rate) > 0:
                        fallback_fx = float(rate)
                        break
            else:
                fallback_fx = self._latest_usd_krw_fx_rate()
            rows = self._build_feature_rows(
                ticker=ticker,
                candles=candles,
                close_keys=("clos", "ovrs_nmix_prpr"),
                quote_currency="USD",
                source=source,
                since_date=since_date,
                exchange_code=order_excd,
                instrument_id=instrument_id,
                fx_by_date=fx_by_date,
                default_fx=fallback_fx,
            )
            instrument_row = {
                "instrument_id": instrument_id,
                "ticker": ticker,
                "exchange_code": order_excd,
                "currency": "USD",
                "lot_size": 1,
                "tick_size": None,
                "tradable": True,
                "status": "ACTIVE",
                "updated_at": utc_now(),
            }
            return rows, instrument_row

        if last_exc is not None:
            raise last_exc
        raise ValueError("daily price returned empty rows")

    def _sync_kospi(
        self,
        ticker: str,
        since_date: date | None,
        source: str,
    ) -> tuple[list[dict[str, object]], dict[str, Any]]:
        """Builds KOSPI feature rows from domestic daily price history."""
        end_date = datetime.now(timezone.utc).strftime("%Y%m%d")
        start_date = (
            datetime.now(timezone.utc) - timedelta(days=max(self.settings.market_sync_history_days, 400))
        ).strftime("%Y%m%d")
        candles = self.client.get_domestic_daily_price(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            market_div_code="J",
            period_div_code="D",
            org_adj_prc="1",
        )
        rows = self._build_feature_rows(
            ticker=ticker,
            candles=candles,
            close_keys=("stck_clpr",),
            quote_currency="KRW",
            source=source,
            since_date=since_date,
            exchange_code="KRX",
            instrument_id=f"KRX:{ticker}",
            default_fx=1.0,
        )
        instrument_row = {
            "instrument_id": f"KRX:{ticker}",
            "ticker": ticker,
            "ticker_name": str(getattr(self, "_kospi_ticker_names", {}).get(ticker) or "").strip() or None,
            "exchange_code": "KRX",
            "currency": "KRW",
            "lot_size": 1,
            "tick_size": 1.0,
            "tradable": True,
            "status": "ACTIVE",
            "updated_at": utc_now(),
        }
        return rows, instrument_row

    def _sync_us_quote(self, *, ticker: str, preferred_excd: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fetches one US live quote with exchange fallback probing."""
        last_exc: Exception | None = None
        resolved_excd, resolved_quote = self._probe_us_quote_exchange(ticker=ticker, preferred_excd=preferred_excd)
        candidates: list[str] = []
        for excd in [resolved_excd, *self._quote_exchange_candidates(preferred_excd)]:
            token = str(excd or "").strip().upper()
            if token and token not in candidates:
                candidates.append(token)

        for excd in candidates:
            quote = resolved_quote if excd == resolved_excd and resolved_quote else {}
            if not quote:
                try:
                    quote = self.client.get_overseas_price(ticker=ticker, excd=excd)
                except Exception as exc:
                    last_exc = exc
                    continue
            if not _us_quote_has_payload(quote):
                continue

            last = _to_float(quote.get("last"), default=0.0)
            if last <= 0:
                continue
            rate = _to_float(quote.get("rate"), default=0.0)
            order_excd = _quote_to_order_exchange(excd)
            if not order_excd:
                continue
            instrument_id = f"{order_excd}:{ticker}"

            detail: dict[str, Any] = {}
            try:
                detail = self.client.get_overseas_price_detail(ticker=ticker, excd=excd)
            except Exception:
                detail = {}
            fx_rate = _to_float(detail.get("t_rate"), default=0.0)
            if fx_rate <= 0:
                fx_rate = self._latest_usd_krw_fx_rate()
            else:
                self._usd_krw_latest_fx = float(fx_rate)
            tick_size = _extract_us_tick_size(detail, last)
            tradable_raw = str(detail.get("e_ordyn") or quote.get("e_ordyn") or "").strip().upper()
            tradable = None
            if tradable_raw in {"Y", "N"}:
                tradable = tradable_raw == "Y"

            quote_meta = {
                "last_usd": float(last),
                "rate_pct": float(rate),
                "fx_rate": float(fx_rate),
                "exchange_code": order_excd,
                "instrument_id": instrument_id,
            }
            instrument_row = {
                "instrument_id": instrument_id,
                "ticker": ticker,
                "exchange_code": order_excd,
                "currency": str(detail.get("curr") or "USD").strip().upper() or "USD",
                "lot_size": 1,
                "tick_size": tick_size,
                "tradable": tradable,
                "status": "ACTIVE",
                "updated_at": utc_now(),
            }
            return quote_meta, instrument_row

        if last_exc is not None:
            raise last_exc
        raise ValueError("quote returned empty last")

    def _refresh_latest_and_universe(
        self,
        *,
        rows: list[dict[str, object]],
        instrument_rows: list[dict[str, Any]],
        tickers: list[str],
        symbols: list[dict[str, str]] | None = None,
    ) -> None:
        """Best-effort refresh for latest snapshots and candidate universe."""
        if rows and hasattr(self.repo, "insert_market_features_latest"):
            try:
                self.repo.insert_market_features_latest(rows)
            except Exception as exc:
                logger.warning("[yellow]market_features_latest refresh skipped[/yellow] err=%s", str(exc))
        elif hasattr(self.repo, "refresh_market_features_latest"):
            try:
                self.repo.refresh_market_features_latest(
                    tickers=tickers,
                    sources=self._all_sources(),
                    lookback_days=30,
                )
            except Exception as exc:
                logger.warning("[yellow]market_features_latest backfill skipped[/yellow] err=%s", str(exc))

        if instrument_rows and hasattr(self.repo, "upsert_instrument_master"):
            try:
                self.repo.upsert_instrument_master(instrument_rows)
            except Exception as exc:
                logger.warning("[yellow]instrument_master upsert skipped[/yellow] err=%s", str(exc))

        quote_exchange_map = {
            str(symbol.get("ticker") or "").strip().upper(): str(symbol.get("quote_excd") or "").strip().upper()
            for symbol in (symbols or [])
            if isinstance(symbol, dict) and str(symbol.get("ticker") or "").strip()
        }
        if hasattr(self.repo, "insert_fundamentals_snapshot_latest"):
            try:
                self._refresh_fundamentals_snapshot(
                    tickers=tickers,
                    quote_exchange_map=quote_exchange_map,
                )
            except Exception as exc:
                logger.warning("[yellow]fundamentals snapshot refresh skipped[/yellow] err=%s", str(exc))

        if hasattr(self.repo, "rebuild_universe_candidates"):
            try:
                self.repo.rebuild_universe_candidates(
                    top_n=max(1, int(self.settings.universe_run_top_n)),
                    per_exchange_cap=max(1, int(self.settings.universe_per_exchange_cap)),
                    sources=self._all_sources(),
                    allowed_tickers=tickers,
                    ticker_names=getattr(self, "_kospi_ticker_names", {}),
                )
            except Exception as exc:
                logger.warning("[yellow]universe_candidates rebuild skipped[/yellow] err=%s", str(exc))

    def _refresh_fundamentals_snapshot(
        self,
        *,
        tickers: list[str],
        quote_exchange_map: dict[str, str] | None = None,
    ) -> None:
        """Best-effort refresh for fundamental snapshot rows used by discovery buckets."""
        tokens = [str(t).strip().upper() for t in tickers if str(t).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return

        market_rows: list[dict[str, Any]] = []
        if hasattr(self.repo, "latest_market_features"):
            try:
                market_rows = self.repo.latest_market_features(
                    tickers=tokens,
                    limit=max(50, len(tokens)),
                    sources=self._all_sources(),
                )
            except Exception:
                market_rows = []
        market_map = {
            str(row.get("ticker") or "").strip().upper(): row
            for row in market_rows
            if isinstance(row, dict) and str(row.get("ticker") or "").strip()
        }

        def _opt_float(value: object) -> float | None:
            parsed = _to_float(value, default=0.0)
            return parsed if parsed != 0.0 else None

        snapshot_rows: list[dict[str, Any]] = []
        for ticker in tokens:
            market_row = market_map.get(ticker) or {}
            as_of_ts = market_row.get("as_of_ts") or utc_now()
            exchange_code = str(market_row.get("exchange_code") or "").strip().upper()
            instrument_id = str(market_row.get("instrument_id") or "").strip()
            if self._is_kospi_ticker(ticker):
                ratio_rows = self.client.get_domestic_financial_ratio(ticker=ticker)
                if not ratio_rows:
                    continue
                latest = ratio_rows[0]
                last_native = _to_float(market_row.get("close_price_native"), default=0.0)
                if last_native <= 0:
                    last_native = _to_float(market_row.get("close_price_krw"), default=0.0)
                eps = _opt_float(latest.get("eps"))
                bps = _opt_float(latest.get("bps"))
                snapshot_rows.append(
                    {
                        "as_of_ts": as_of_ts,
                        "ticker": ticker,
                        "market": "kospi",
                        "exchange_code": exchange_code or "KRX",
                        "instrument_id": instrument_id or f"KRX:{ticker}",
                        "currency": "KRW",
                        "last_native": last_native if last_native > 0 else None,
                        "per": (last_native / eps) if last_native > 0 and eps and eps > 0 else None,
                        "pbr": (last_native / bps) if last_native > 0 and bps and bps > 0 else None,
                        "eps": eps,
                        "bps": bps,
                        "sps": _opt_float(latest.get("sps")),
                        "roe": _opt_float(latest.get("roe_val")),
                        "debt_ratio": _opt_float(latest.get("lblt_rate")),
                        "reserve_ratio": _opt_float(latest.get("rsrv_rate")),
                        "operating_profit_growth": _opt_float(latest.get("bsop_prfi_inrt")),
                        "net_profit_growth": _opt_float(latest.get("ntin_inrt")),
                        "source": "open_trading_kospi_financial_ratio",
                    }
                )
                continue

            preferred_excd = str((quote_exchange_map or {}).get(ticker) or "").strip().upper()
            if not preferred_excd:
                preferred_excd = self._known_us_quote_exchange(ticker)
            if not preferred_excd:
                preferred_excd = str(self.settings.kis_overseas_quote_excd or "NAS").strip().upper()
            detail = self.client.get_overseas_price_detail(ticker=ticker, excd=preferred_excd)
            price_last = _opt_float(detail.get("last"))
            snapshot_rows.append(
                {
                    "as_of_ts": as_of_ts,
                    "ticker": ticker,
                    "market": "us",
                    "exchange_code": exchange_code or _quote_to_order_exchange(preferred_excd) or preferred_excd,
                    "instrument_id": instrument_id or f"{_quote_to_order_exchange(preferred_excd) or preferred_excd}:{ticker}",
                    "currency": str(detail.get("curr") or "USD").strip().upper() or "USD",
                    "last_native": price_last,
                    "per": _opt_float(detail.get("perx")),
                    "pbr": _opt_float(detail.get("pbrx")),
                    "eps": _opt_float(detail.get("epsx")),
                    "bps": _opt_float(detail.get("bpsx")),
                    "source": "open_trading_us_price_detail",
                }
            )

        if snapshot_rows:
            self.repo.insert_fundamentals_snapshot_latest(snapshot_rows)

    def _should_force_kospi_backfill(
        self,
        *,
        ticker: str,
        span: dict[str, Any] | None,
    ) -> bool:
        """Returns whether KOSPI history should be fully backfilled again."""
        if not span:
            return False
        min_d = span.get("min_d")
        target_start = (
            datetime.now(timezone.utc).date()
            - timedelta(days=max(int(self.settings.market_sync_history_days), 400))
        )
        if not isinstance(min_d, date):
            return True
        return min_d > target_start

    def _should_force_us_backfill(
        self,
        *,
        ticker: str,
        span: dict[str, Any] | None,
    ) -> bool:
        """Returns whether US history should be fully backfilled again."""
        if not span:
            return False
        row_count = int(span.get("row_count") or 0)
        if row_count < 21:
            return True
        min_d = span.get("min_d")
        target_start = (
            datetime.now(timezone.utc).date()
            - timedelta(days=max(int(self.settings.market_sync_history_days), 400))
        )
        if not isinstance(min_d, date):
            return True
        return min_d > target_start

    def sync_market_features(self) -> MarketSyncResult:
        """Fetches market data and writes feature rows into BigQuery."""
        symbols = self._include_missing_daily_feature_symbols(self._target_symbols())
        if not symbols:
            logger.warning("[yellow]No tickers selected for market sync[/yellow] target=%s", self.settings.kis_target_market)
            return MarketSyncResult(inserted_rows=0, attempted_tickers=0, failed_tickers=[])

        tickers = [s["ticker"] for s in symbols]
        self._prime_known_us_quote_exchanges(tickers)
        latest_dates: dict[str, Any] = {}
        spans_by_source: dict[str, dict[str, dict[str, Any]]] = {}
        try:
            # Query per-market sources and merge latest dates
            for src in self._all_sources():
                if src.endswith("_quote"):
                    continue
                part = self.repo.latest_feature_dates(tickers, src)
                for t, d in part.items():
                    if t not in latest_dates or (d and (not latest_dates[t] or d > latest_dates[t])):
                        latest_dates[t] = d
                if hasattr(self.repo, "feature_date_spans"):
                    spans_by_source[src] = self.repo.feature_date_spans(tickers, src)
        except Exception as exc:
            latest_dates = {}
            spans_by_source = {}
            logger.warning("[yellow]Latest feature date lookup skipped[/yellow] err=%s", str(exc))

        rows: list[dict[str, object]] = []
        failures: list[str] = []
        instrument_rows: list[dict[str, Any]] = []

        for symbol in symbols:
            ticker = str(symbol.get("ticker") or "").strip().upper()
            preferred_excd = str(symbol.get("quote_excd") or "").strip().upper()
            if not ticker:
                continue
            since_date = latest_dates.get(ticker)
            try:
                if self._is_kospi_ticker(ticker):
                    src = self._daily_source("kospi")
                    if self._should_force_kospi_backfill(
                        ticker=ticker,
                        span=(spans_by_source.get(src) or {}).get(ticker),
                    ):
                        since_date = None
                        logger.info(
                            "[cyan]KOSPI history backfill triggered[/cyan] ticker=%s source=%s",
                            ticker,
                            src,
                        )
                    new_rows, instrument_row = self._sync_kospi(
                        ticker=ticker,
                        since_date=since_date,
                        source=src,
                    )
                else:
                    src = self._daily_source("us")
                    if self._should_force_us_backfill(
                        ticker=ticker,
                        span=(spans_by_source.get(src) or {}).get(ticker),
                    ):
                        since_date = None
                        logger.info(
                            "[cyan]US history backfill triggered[/cyan] ticker=%s source=%s",
                            ticker,
                            src,
                        )
                    new_rows, instrument_row = self._sync_us_daily(
                        ticker=ticker,
                        preferred_excd=preferred_excd,
                        source=src,
                        since_date=since_date,
                    )

                if new_rows:
                    rows.extend(new_rows)
                else:
                    logger.info("[cyan]No new market rows[/cyan] ticker=%s", ticker)
                instrument_rows.append(instrument_row)
            except Exception as exc:
                failures.append(ticker)
                logger.error("[red]Market sync failed[/red] ticker=%s err=%s", ticker, str(exc))

        inserted = 0
        if rows:
            try:
                self.repo.insert_market_features(rows)
                inserted = len(rows)
                self._refresh_latest_and_universe(rows=rows, instrument_rows=instrument_rows, tickers=tickers, symbols=symbols)
                logger.info(
                    "[green]Market sync done[/green] inserted_rows=%d tickers=%d failed=%d",
                    inserted,
                    len(symbols),
                    len(failures),
                    extra={
                        "event": "market_sync_done",
                        "target_market": self.settings.kis_target_market,
                        "inserted_rows": inserted,
                        "attempted_tickers": len(symbols),
                        "failed_tickers": len(failures),
                    },
                )
            except Exception as exc:
                inserted = 0
                failures = sorted(set(failures) | {str(row.get("ticker") or "") for row in rows if row.get("ticker")})
                logger.error(
                    "[red]Market sync write failed[/red] err=%s",
                    str(exc),
                    extra={
                        "event": "market_sync_write_failed",
                        "target_market": self.settings.kis_target_market,
                        "attempted_tickers": len(symbols),
                        "failed_tickers": len(failures),
                    },
                )
        else:
            self._refresh_latest_and_universe(rows=[], instrument_rows=instrument_rows, tickers=tickers, symbols=symbols)
            logger.info(
                "[cyan]Market sync produced zero new rows[/cyan] tickers=%d",
                len(symbols),
                extra={
                    "event": "market_sync_noop",
                    "target_market": self.settings.kis_target_market,
                    "attempted_tickers": len(symbols),
                },
            )

        return MarketSyncResult(
            inserted_rows=inserted,
            attempted_tickers=len(symbols),
            failed_tickers=failures,
        )

    def sync_market_quotes(self) -> MarketSyncResult:
        """Fetches intraday quotes and writes them as hourly feature rows."""
        symbols = self._target_symbols()
        if not symbols:
            logger.warning("[yellow]No tickers selected for quote sync[/yellow] target=%s", self.settings.kis_target_market)
            return MarketSyncResult(inserted_rows=0, attempted_tickers=0, failed_tickers=[])

        now = datetime.now(timezone.utc)
        as_of_ts = now.replace(minute=0, second=0, microsecond=0)
        tickers = [s["ticker"] for s in symbols]
        self._prime_known_us_quote_exchanges(tickers)

        daily_map: dict[tuple[str, str], dict[str, object]] = {}
        try:
            if hasattr(self.repo, "latest_market_features"):
                daily_rows = self.repo.latest_market_features(
                    tickers=tickers,
                    limit=max(200, len(tickers) * 3),
                    sources=[s for s in self._all_sources() if not s.endswith("_quote")],
                )
                for row in daily_rows:
                    t = str(row.get("ticker", "") or "").strip().upper()
                    ex = str(row.get("exchange_code", "") or "").strip().upper()
                    if t:
                        daily_map[(t, ex)] = dict(row)
                        daily_map.setdefault((t, ""), dict(row))
        except Exception as exc:
            logger.warning("[yellow]Daily feature lookup skipped[/yellow] err=%s", str(exc))

        rows: list[dict[str, object]] = []
        failures: list[str] = []
        instrument_rows: list[dict[str, Any]] = []

        for symbol in symbols:
            ticker = str(symbol.get("ticker") or "").strip().upper()
            preferred_excd = str(symbol.get("quote_excd") or "").strip().upper()
            if not ticker:
                continue
            try:
                if self._is_kospi_ticker(ticker):
                    quote = self.client.get_domestic_price(ticker=ticker, market_div_code="J")
                    last = _to_float(quote.get("stck_prpr"), default=0.0)
                    rate = _to_float(quote.get("prdy_ctrt"), default=0.0)
                    if last <= 0:
                        raise ValueError("quote returned empty last")
                    base = daily_map.get((ticker, "KRX"), daily_map.get((ticker, ""), {}))
                    if not _has_daily_feature_metrics(base):
                        raise ValueError("daily history features unavailable for quote snapshot")
                    rows.append(
                        {
                            "as_of_ts": as_of_ts,
                            "ticker": ticker,
                            "exchange_code": "KRX",
                            "instrument_id": f"KRX:{ticker}",
                            "close_price_krw": float(last),
                            "close_price_native": float(last),
                            "quote_currency": "KRW",
                            "fx_rate_used": 1.0,
                            "ret_5d": base.get("ret_5d"),
                            "ret_20d": base.get("ret_20d"),
                            "volatility_20d": base.get("volatility_20d"),
                            "sentiment_score": float(max(-1.0, min(1.0, rate / 100.0))),
                            "source": self._quote_source("kospi"),
                        }
                    )
                    instrument_rows.append(
                        {
                            "instrument_id": f"KRX:{ticker}",
                            "ticker": ticker,
                            "ticker_name": str(getattr(self, "_kospi_ticker_names", {}).get(ticker) or "").strip() or None,
                            "exchange_code": "KRX",
                            "currency": "KRW",
                            "lot_size": 1,
                            "tick_size": 1.0,
                            "tradable": None,
                            "status": "ACTIVE",
                            "updated_at": utc_now(),
                        }
                    )
                else:
                    quote_meta, instrument_row = self._sync_us_quote(ticker=ticker, preferred_excd=preferred_excd)
                    exchange_code = str(quote_meta.get("exchange_code") or "").strip().upper()
                    base = daily_map.get((ticker, exchange_code), daily_map.get((ticker, ""), {}))
                    if not _has_daily_feature_metrics(base):
                        raise ValueError("daily history features unavailable for quote snapshot")
                    rows.append(
                        {
                            "as_of_ts": as_of_ts,
                            "ticker": ticker,
                            "exchange_code": exchange_code,
                            "instrument_id": str(quote_meta.get("instrument_id") or ""),
                            "close_price_krw": float(quote_meta["last_usd"] * float(quote_meta["fx_rate"])),
                            "close_price_native": float(quote_meta["last_usd"]),
                            "quote_currency": "USD",
                            "fx_rate_used": float(quote_meta["fx_rate"]),
                            "ret_5d": base.get("ret_5d"),
                            "ret_20d": base.get("ret_20d"),
                            "volatility_20d": base.get("volatility_20d"),
                            "sentiment_score": float(max(-1.0, min(1.0, float(quote_meta["rate_pct"]) / 100.0))),
                            "source": self._quote_source("us"),
                        }
                    )
                    instrument_rows.append(instrument_row)
            except Exception as exc:
                failures.append(ticker)
                logger.warning("[yellow]Quote sync failed[/yellow] ticker=%s err=%s", ticker, str(exc))

        inserted = 0
        if rows:
            try:
                self.repo.insert_market_features(rows)
                inserted = len(rows)
                self._refresh_latest_and_universe(rows=rows, instrument_rows=instrument_rows, tickers=tickers, symbols=symbols)
                logger.info(
                    "[green]Quote sync done[/green] inserted_rows=%d tickers=%d failed=%d",
                    inserted,
                    len(symbols),
                    len(failures),
                    extra={
                        "event": "quote_sync_done",
                        "target_market": self.settings.kis_target_market,
                        "inserted_rows": inserted,
                        "attempted_tickers": len(symbols),
                        "failed_tickers": len(failures),
                    },
                )
            except Exception as exc:
                inserted = 0
                logger.error(
                    "[red]Quote sync write failed[/red] err=%s",
                    str(exc),
                    extra={
                        "event": "quote_sync_write_failed",
                        "target_market": self.settings.kis_target_market,
                        "attempted_tickers": len(symbols),
                        "failed_tickers": len(failures),
                    },
                )
        else:
            self._refresh_latest_and_universe(rows=[], instrument_rows=instrument_rows, tickers=tickers, symbols=symbols)
            logger.info(
                "[cyan]Quote sync produced zero rows[/cyan] tickers=%d",
                len(symbols),
                extra={
                    "event": "quote_sync_noop",
                    "target_market": self.settings.kis_target_market,
                    "attempted_tickers": len(symbols),
                },
            )

        return MarketSyncResult(
            inserted_rows=inserted,
            attempted_tickers=len(symbols),
            failed_tickers=failures,
        )


class AccountSyncService:
    """Loads live account snapshot and persists it into BigQuery."""
    _US_MARKETS: set[str] = {"nasdaq", "nyse", "amex", "us"}

    def __init__(self, settings: Settings, repo: BigQueryRepository, client: OpenTradingClient | None = None):
        self.settings = settings
        self.repo = repo
        self.client = client or OpenTradingClient(settings)
        self._usd_krw_latest_fx: float | None = None

    def _latest_usd_krw_fx_rate(self) -> float:
        """Returns the freshest available USD/KRW rate for account valuation."""
        if self._usd_krw_latest_fx is not None and self._usd_krw_latest_fx > 0:
            return float(self._usd_krw_latest_fx)

        symbol = str(self.settings.usd_krw_fx_symbol or "").strip().upper()
        if not symbol:
            raise RuntimeError("USD/KRW FX symbol not configured")

        today = datetime.now(timezone.utc).date()
        start = (today - timedelta(days=7)).strftime("%Y%m%d")
        end = today.strftime("%Y%m%d")
        try:
            fx_rows = self.client.get_usd_krw_daily_chart(
                symbol=symbol,
                start_date=start,
                end_date=end,
                market_div_code=self.settings.usd_krw_fx_market_div_code,
                period="D",
                max_pages=4,
            )
            fx_series = MarketDataSyncService._extract_chart_series(
                fx_rows,
                close_keys=("ovrs_nmix_prpr", "clos"),
            )
        except Exception as exc:
            logger.warning("[yellow]USD/KRW latest FX fetch failed[/yellow] symbol=%s err=%s", symbol, str(exc))
            raise RuntimeError(f"USD/KRW latest FX fetch failed for symbol={symbol}") from exc

        if not fx_series:
            raise RuntimeError(f"USD/KRW latest FX unavailable for symbol={symbol}")

        self._usd_krw_latest_fx = float(fx_series[-1][1])
        return float(self._usd_krw_latest_fx)

    def _resolve_overseas_order_exchange(
        self,
        *,
        ticker: str,
        raw_exchange: str,
        known_order_exchange: str = "",
    ) -> str:
        order_excd = normalize_us_order_exchange(raw_exchange)
        if order_excd:
            return order_excd
        order_excd = normalize_us_order_exchange(known_order_exchange)
        if order_excd:
            return order_excd
        quote_excd, _ = _probe_us_quote_exchange_with_client(
            client=self.client,
            settings=self.settings,
            ticker=ticker,
        )
        order_excd = _quote_to_order_exchange(quote_excd)
        if order_excd:
            return order_excd
        return target_market_default_us_order_exchange(self.settings.kis_target_market)

    def _position_from_overseas_row(
        self,
        row: dict[str, object],
        *,
        known_order_exchange: str = "",
    ) -> Position | None:
        """Builds one Position object from overseas balance response row."""
        ticker = str(row.get("pdno", "")).strip().upper()
        if not ticker:
            return None

        settled_qty = _to_float(row.get("cblc_qty13"), default=0.0)
        today_buy_qty = _to_float(row.get("thdt_buy_ccld_qty1"), default=0.0)
        today_sell_qty = _to_float(row.get("thdt_sll_ccld_qty1"), default=0.0)
        current_qty = _to_float(row.get("ccld_qty_smtl1"), default=0.0)
        orderable_qty = _to_float(row.get("ord_psbl_qty1"), default=0.0)

        # KIS present-balance exposes both carry-over quantity and current quantity.
        # For sleeve/account reconciliation we need the current visible holding,
        # including today's filled buys/sells, not just the settled carry-over.
        quantity = current_qty
        if quantity <= 0:
            quantity = orderable_qty
        if quantity <= 0:
            quantity = settled_qty + today_buy_qty - today_sell_qty
        if quantity <= 0:
            quantity = settled_qty
        if quantity <= 0:
            return None

        fx = _to_float(row.get("bass_exrt"), default=0.0)
        if fx <= 0:
            fx = self._latest_usd_krw_fx_rate()
            logger.warning(
                "[yellow]position fx_rate fallback[/yellow] ticker=%s rate=%.2f source=latest_fx",
                ticker,
                fx,
            )

        avg_price_ccy = _to_float(row.get("avg_unpr3"), default=0.0)
        market_price_ccy = _to_float(row.get("ovrs_now_pric1"), default=avg_price_ccy)

        raw_ex = str(row.get("ovrs_excg_cd") or row.get("excg_cd") or "").strip().upper()
        order_excd = self._resolve_overseas_order_exchange(
            ticker=ticker,
            raw_exchange=raw_ex,
            known_order_exchange=known_order_exchange,
        )

        tr_crcy = str(row.get("tr_crcy_cd") or "").strip().upper() or "USD"

        return Position(
            ticker=ticker,
            exchange_code=order_excd,
            instrument_id=f"{order_excd}:{ticker}" if order_excd else "",
            quantity=quantity,
            avg_price_krw=max(avg_price_ccy * fx, 0.0),
            market_price_krw=max(market_price_ccy * fx, 0.0),
            avg_price_native=avg_price_ccy if avg_price_ccy > 0 else None,
            market_price_native=market_price_ccy if market_price_ccy > 0 else None,
            quote_currency=tr_crcy,
            fx_rate=fx,
        )

    def _position_from_domestic_row(self, row: dict[str, object]) -> Position | None:
        """Builds one Position object from domestic balance response row."""
        ticker = str(row.get("pdno", "")).strip().upper()
        if not ticker:
            return None

        quantity = _to_float(row.get("hldg_qty"), default=0.0)
        if quantity <= 0:
            return None

        avg_price = _to_float(row.get("pchs_avg_pric"), default=0.0)
        market_price = _to_float(row.get("prpr"), default=avg_price)

        return Position(
            ticker=ticker,
            exchange_code="KRX",
            instrument_id=f"KRX:{ticker}",
            quantity=quantity,
            avg_price_krw=max(avg_price, 0.0),
            market_price_krw=max(market_price, 0.0),
            avg_price_native=avg_price if avg_price > 0 else None,
            market_price_native=market_price if market_price > 0 else None,
            quote_currency="KRW",
            fx_rate=1.0,
        )

    def _sync_overseas(self) -> AccountSnapshot:
        """Loads overseas account and builds snapshot."""
        balance_market_codes = self._us_balance_market_codes()
        positions_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        rows_by_market_code: dict[str, list[dict[str, Any]]] = {}
        summary_by_market_code: dict[str, list[dict[str, Any]]] = {}
        if balance_market_codes:
            for market_code in balance_market_codes:
                rows1, _, rows3 = self.client.get_overseas_present_balance(tr_mket_cd=market_code)
                positions_rows.extend(rows1)
                summary_rows.extend(rows3)
                rows_by_market_code[market_code] = [dict(row) for row in rows1 if isinstance(row, dict)]
                summary_by_market_code[market_code] = [dict(row) for row in rows3 if isinstance(row, dict)]
        else:
            positions_rows, _, summary_rows = self.client.get_overseas_present_balance()

        tickers = [
            str((row or {}).get("pdno") or "").strip().upper()
            for row in positions_rows
            if isinstance(row, dict) and str((row or {}).get("pdno") or "").strip()
        ]
        instrument_map: dict[str, dict[str, Any]] = {}
        loader = getattr(self.repo, "latest_instrument_map", None)
        if callable(loader) and tickers:
            try:
                instrument_map = loader(tickers)
            except Exception as exc:
                logger.warning(
                    "[yellow]instrument_map load failed[/yellow] tickers=%d err=%s",
                    len(tickers),
                    str(exc),
                )
                instrument_map = {}

        positions: dict[str, Position] = {}
        live_fx = 0.0
        for row in positions_rows:
            ticker = str((row or {}).get("pdno") or "").strip().upper()
            known_order_exchange = ""
            if ticker:
                known_order_exchange = normalize_us_order_exchange((instrument_map.get(ticker) or {}).get("exchange_code") or "")
            position = self._position_from_overseas_row(row, known_order_exchange=known_order_exchange)
            if not position:
                continue
            positions[position.ticker] = position
            # Extract live USD/KRW rate from the first row that has it.
            if live_fx <= 0:
                live_fx = _to_float(row.get("bass_exrt"), default=0.0)

        summary_candidates = [dict(row) for row in summary_rows if isinstance(row, dict)]
        cash_foreign = 0.0
        for summary in summary_candidates:
            cash_foreign = max(cash_foreign, _to_float(summary.get("frcr_use_psbl_amt"), default=0.0))
        if live_fx <= 0:
            live_fx = self._latest_usd_krw_fx_rate()
            logger.warning(
                "[yellow]overseas balance fx_rate fallback[/yellow] rate=%.2f source=latest_fx (API bass_exrt missing)",
                live_fx,
            )

        cash_krw = 0.0
        for summary in summary_candidates:
            cash_krw = max(
                cash_krw,
                _to_float(summary.get("tot_dncl_amt"), default=0.0),
                _to_float(summary.get("wdrw_psbl_tot_amt"), default=0.0),
            )
        if cash_krw <= 0:
            cash_krw = cash_foreign * live_fx

        total_equity_krw = cash_krw + sum(pos.market_value_krw() for pos in positions.values())
        if len(rows_by_market_code) > 1:
            anchor_market_code = ""
            anchor_total_equity = 0.0
            for market_code, market_summaries in summary_by_market_code.items():
                market_total = max(
                    (_to_float(summary.get("tot_asst_amt"), default=0.0) for summary in market_summaries),
                    default=0.0,
                )
                if market_total > anchor_total_equity:
                    anchor_market_code = market_code
                    anchor_total_equity = market_total
            if anchor_market_code and anchor_total_equity > 0:
                anchor_tickers = {
                    str((row or {}).get("pdno") or "").strip().upper()
                    for row in rows_by_market_code.get(anchor_market_code, [])
                    if str((row or {}).get("pdno") or "").strip()
                }
                extra_market_value = sum(
                    pos.market_value_krw()
                    for ticker, pos in positions.items()
                    if ticker not in anchor_tickers
                )
                total_equity_krw = anchor_total_equity + extra_market_value

        usd_krw = live_fx
        return AccountSnapshot(
            cash_krw=max(cash_krw, 0.0),
            total_equity_krw=max(total_equity_krw, 0.0),
            positions=positions,
            usd_krw_rate=usd_krw,
            cash_foreign=max(cash_foreign, 0.0),
            cash_foreign_currency="USD",
        )

    def _us_balance_market_codes(self) -> list[str]:
        markets = self._parsed_markets()
        exchanges: list[str] = []
        if "us" in markets or len(markets & self._US_MARKETS) > 1:
            exchanges.extend(us_order_exchange_candidates())
        else:
            target_default = target_market_default_us_order_exchange(",".join(sorted(markets)))
            if target_default:
                exchanges.append(target_default)
        if not exchanges:
            fallback = normalize_us_order_exchange(self.settings.kis_overseas_order_excd)
            if fallback:
                exchanges.append(fallback)

        out: list[str] = []
        for exchange_code in exchanges:
            market_code = _US_TARGET_TO_BALANCE_MARKET_CODE.get(str(exchange_code or "").strip().upper())
            if market_code and market_code not in out:
                out.append(market_code)

        fallback_code = str(self.settings.kis_us_tr_mket_cd or "").strip()
        if fallback_code and fallback_code not in out:
            out.append(fallback_code)
        return out

    def _sync_domestic(self) -> AccountSnapshot:
        """Loads domestic account and builds snapshot."""
        positions_rows, summary_rows = self.client.get_domestic_balance(inqr_dvsn="02")

        positions: dict[str, Position] = {}
        for row in positions_rows:
            position = self._position_from_domestic_row(row)
            if not position:
                continue
            positions[position.ticker] = position

        summary = summary_rows[0] if summary_rows else {}

        # Use actual orderable cash (주문가능현금) instead of dnca_tot_amt
        dnca = _to_float(summary.get("dnca_tot_amt"), default=0.0)
        try:
            cash_krw = float(self.client.get_domestic_orderable_cash())
        except Exception as exc:
            err = str(exc).strip() or exc.__class__.__name__
            raise RuntimeError(f"domestic orderable cash query failed: {err}") from exc
        if dnca > 0 and cash_krw != dnca:
            logger.info(
                "[yellow]Cash divergence[/yellow] dnca_tot_amt=%.0f orderable=%.0f diff=%.0f",
                dnca, cash_krw, dnca - cash_krw,
            )
        if cash_krw < 0:
            raise RuntimeError(f"domestic orderable cash must be non-negative, got {cash_krw}")

        total_equity_krw = _to_float(summary.get("tot_evlu_amt"), default=0.0)
        if total_equity_krw <= 0:
            total_equity_krw = cash_krw + sum(pos.market_value_krw() for pos in positions.values())

        return AccountSnapshot(
            cash_krw=max(cash_krw, 0.0),
            total_equity_krw=max(total_equity_krw, 0.0),
            positions=positions,
        )

    def _parsed_markets(self) -> set[str]:
        raw = (self.settings.kis_target_market or "").lower().strip()
        return {t.strip() for t in raw.split(",") if t.strip()}

    def _has_us_market(self) -> bool:
        return bool(self._parsed_markets() & {"nasdaq", "nyse", "amex", "us"})

    def _has_kospi_market(self) -> bool:
        return bool(self._parsed_markets() & {"kospi", "kosdaq"})

    def sync_account_snapshot(self) -> AccountSnapshot:
        """Fetches account balance and writes latest snapshot."""
        has_us = self._has_us_market()
        has_kr = self._has_kospi_market()

        if has_us and has_kr:
            # Combined: merge both snapshots
            us_snap = self._sync_overseas()
            kr_snap = self._sync_domestic()
            merged_positions = {**us_snap.positions, **kr_snap.positions}
            snapshot = AccountSnapshot(
                cash_krw=us_snap.cash_krw + kr_snap.cash_krw,
                cash_foreign=us_snap.cash_foreign,
                cash_foreign_currency=us_snap.cash_foreign_currency,
                total_equity_krw=us_snap.total_equity_krw + kr_snap.total_equity_krw,
                positions=merged_positions,
            )
        elif has_us:
            snapshot = self._sync_overseas()
        elif has_kr:
            snapshot = self._sync_domestic()
        else:
            raise ValueError(f"unsupported target market for account sync: {self.settings.kis_target_market}")

        self.repo.write_account_snapshot(snapshot)
        logger.info(
            "[green]Account sync done[/green] cash=%.0f equity=%.0f positions=%d",
            snapshot.cash_krw,
            snapshot.total_equity_krw,
            len(snapshot.positions),
            extra={
                "event": "account_sync_done",
                "target_market": self.settings.kis_target_market,
                "cash_krw": snapshot.cash_krw,
                "total_equity_krw": snapshot.total_equity_krw,
                "positions": len(snapshot.positions),
            },
        )
        return snapshot


@dataclass(slots=True)
class BrokerTradeSyncResult:
    """Summarizes one broker trade ledger sync execution."""

    inserted_events: int
    scanned_rows: int
    skipped_existing: int
    failed_scopes: list[str]


class BrokerTradeSyncService:
    """Loads raw broker trade rows and stores them as idempotent ledger events."""

    _US_MARKETS: set[str] = {"nasdaq", "nyse", "amex", "us"}

    def __init__(
        self,
        settings: Settings,
        repo: BigQueryRepository,
        client: OpenTradingClient | None = None,
    ):
        self.settings = settings
        self.repo = repo
        self.client = client or OpenTradingClient(settings)

    def _fallback_fx_rate(self) -> float:
        """Returns the latest account snapshot FX rate when available."""
        loader = getattr(self.repo, "latest_account_snapshot", None)
        if callable(loader):
            try:
                snap = loader()
                if snap is not None and float(getattr(snap, "usd_krw_rate", 0.0) or 0.0) > 0:
                    return float(snap.usd_krw_rate)
            except Exception as exc:
                logger.warning("[yellow]broker_trade snapshot fx lookup failed[/yellow] err=%s", str(exc))
        return 0.0

    def _parsed_markets(self) -> set[str]:
        return {m.strip() for m in str(self.settings.kis_target_market or "").split(",") if m.strip()}

    def _has_us_market(self) -> bool:
        return bool(self._parsed_markets() & self._US_MARKETS)

    def _has_kospi_market(self) -> bool:
        return "kospi" in self._parsed_markets()

    def _us_exchange_candidates(self) -> list[str]:
        markets = self._parsed_markets()
        out: list[str] = []
        if "us" in markets or len(markets & self._US_MARKETS) > 1:
            out.extend(us_order_exchange_candidates())
        else:
            target_default = target_market_default_us_order_exchange(",".join(sorted(markets)))
            if target_default:
                out.append(target_default)
        if not out:
            fallback = normalize_us_order_exchange(self.settings.kis_overseas_order_excd)
            if fallback:
                out.append(fallback)
        if not out:
            out.extend(us_order_exchange_candidates())
        deduped: list[str] = []
        for token in out:
            cleaned = str(token or "").strip().upper()
            if cleaned and cleaned not in deduped:
                deduped.append(cleaned)
        return deduped

    @staticmethod
    def _parse_side(value: object) -> str:
        token = str(value or "").strip().upper()
        if token in {"02", "BUY", "B", "매수"}:
            return "BUY"
        if token in {"01", "SELL", "S", "매도"}:
            return "SELL"
        if "BUY" in token or "매수" in token:
            return "BUY"
        if "SELL" in token or "매도" in token:
            return "SELL"
        return token or "UNKNOWN"

    @staticmethod
    def _parse_date_token(value: object) -> str:
        token = str(value or "").strip().replace("-", "")
        return token if token.isdigit() and len(token) == 8 else ""

    @staticmethod
    def _parse_time_token(value: object) -> str:
        token = str(value or "").strip().replace(":", "")
        if token.isdigit() and len(token) == 6:
            return token
        if token.isdigit() and len(token) == 4:
            return f"{token}00"
        return ""

    @classmethod
    def _parse_occurred_at(cls, row: dict[str, object]) -> datetime:
        date_token = cls._parse_date_token(
            _pick_str(row, ["ord_dt", "ORD_DT", "ccld_dt", "CCLD_DT", "trad_dt", "TRAD_DT"])
        )
        if not date_token:
            return utc_now()
        time_token = cls._parse_time_token(
            _pick_str(row, ["ord_tmd", "ORD_TMD", "ord_tmd1", "ORD_TMD1", "ccld_tmd", "CCLD_TMD", "trad_tmd", "TRAD_TMD"])
        )
        try:
            if time_token:
                return datetime.strptime(f"{date_token}{time_token}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            return datetime.strptime(date_token, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return utc_now()

    @staticmethod
    def _filled_qty(row: dict[str, object]) -> float:
        return max(
            _to_float(row.get("ft_ccld_qty"), default=0.0),
            _to_float(row.get("CCLD_QTY"), default=0.0),
            _to_float(row.get("ccld_qty"), default=0.0),
            _to_float(row.get("tot_ccld_qty"), default=0.0),
            _to_float(row.get("TOT_CCLD_QTY"), default=0.0),
        )

    @staticmethod
    def _avg_price(row: dict[str, object]) -> float:
        return max(
            _to_float(row.get("ft_ccld_unpr3"), default=0.0),
            _to_float(row.get("CCLD_UNPR"), default=0.0),
            _to_float(row.get("ccld_unpr"), default=0.0),
            _to_float(row.get("avg_pric"), default=0.0),
            _to_float(row.get("avg_unpr"), default=0.0),
            _to_float(row.get("AVG_UNPR"), default=0.0),
        )

    @staticmethod
    def _stable_event_id(scope: str, row: dict[str, object]) -> str:
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        return f"{scope}_{hashlib.md5(payload.encode('utf-8')).hexdigest()}"

    def _normalize_overseas_row(
        self,
        row: dict[str, object],
        *,
        exchange_code: str,
    ) -> dict[str, Any] | None:
        qty = self._filled_qty(row)
        if qty <= 0:
            return None
        price_native = self._avg_price(row)
        fx_rate = max(
            _to_float(row.get("bass_exrt"), default=0.0),
            _to_float(row.get("exrt"), default=0.0),
        )
        # fx_rate is set later from erlm_exrt in sync_broker_trade_events;
        # if still 0 after that step, the row is skipped.
        occurred_at = self._parse_occurred_at(row)
        scope = f"us:{exchange_code}"
        return {
            "event_id": self._stable_event_id(scope, row),
            "occurred_at": occurred_at,
            "broker_order_id": _pick_str(row, ["odno", "ODNO", "ord_no", "ORD_NO", "orgn_odno", "ORGN_ODNO"]),
            "broker_fill_id": _pick_str(row, ["exec_seq", "EXEC_SEQ", "ccld_seq", "CCLD_SEQ"]),
            "account_id": str(self.settings.kis_account_no or "").strip() or None,
            "ticker": _pick_str(row, ["pdno", "PDNO", "symb", "SYMB"]).upper(),
            "exchange_code": exchange_code,
            "instrument_id": (
                f"{exchange_code}:{_pick_str(row, ['pdno', 'PDNO', 'symb', 'SYMB']).strip().upper()}"
                if _pick_str(row, ["pdno", "PDNO", "symb", "SYMB"]).strip()
                else None
            ),
            "side": self._parse_side(_pick_str(row, ["sll_buy_dvsn", "SLL_BUY_DVSN", "sll_buy_dvsn_cd", "SLL_BUY_DVSN_CD"])),
            "quantity": qty,
            "price_krw": max(price_native * fx_rate, 0.0),
            "price_native": price_native if price_native > 0 else None,
            "quote_currency": "USD",
            "fx_rate": fx_rate if fx_rate > 0 else None,
            "fee_krw": None,
            "status": "FILLED",
            "source": "kis_inquire_overseas_ccnl",
            "raw_payload_json": row,
        }

    def _normalize_domestic_row(self, row: dict[str, object]) -> dict[str, Any] | None:
        qty = self._filled_qty(row)
        if qty <= 0:
            return None
        price_krw = self._avg_price(row)
        occurred_at = self._parse_occurred_at(row)
        ticker = _pick_str(row, ["pdno", "PDNO", "mksc_shrn_iscd", "MKSC_SHRN_ISCD"]).upper()
        return {
            "event_id": self._stable_event_id("kospi", row),
            "occurred_at": occurred_at,
            "broker_order_id": _pick_str(row, ["odno", "ODNO", "ord_no", "ORD_NO"]),
            "broker_fill_id": _pick_str(row, ["exec_seq", "EXEC_SEQ", "ccld_seq", "CCLD_SEQ"]),
            "account_id": str(self.settings.kis_account_no or "").strip() or None,
            "ticker": ticker,
            "exchange_code": "KRX",
            "instrument_id": f"KRX:{ticker}" if ticker else None,
            "side": self._parse_side(_pick_str(row, ["sll_buy_dvsn_cd", "SLL_BUY_DVSN_CD", "sll_buy_dvsn", "SLL_BUY_DVSN"])),
            "quantity": qty,
            "price_krw": max(price_krw, 0.0),
            "price_native": price_krw if price_krw > 0 else None,
            "quote_currency": "KRW",
            "fx_rate": 1.0,
            "fee_krw": None,
            "status": "FILLED",
            "source": "kis_inquire_domestic_daily_ccld",
            "raw_payload_json": row,
        }

    def sync_broker_trade_events(self, *, days: int = 7) -> BrokerTradeSyncResult:
        """Fetches raw broker execution rows and stores normalized trade events."""
        scanned_rows = 0
        failed_scopes: list[str] = []
        normalized_rows: list[dict[str, Any]] = []

        if self._has_us_market():
            # Build erlm_exrt lookup from period-trans API (date+ticker+side → fx)
            erlm_fx_map: dict[str, float] = {}
            try:
                now = datetime.now(timezone.utc)
                start_dt = (now - timedelta(days=max(int(days), 1))).strftime("%Y%m%d")
                end_dt = now.strftime("%Y%m%d")
                for excg in self._us_exchange_candidates():
                    try:
                        pt_rows, _ = self.client.inquire_overseas_period_trans(
                            start_date=start_dt,
                            end_date=end_dt,
                            exchange_code=excg,
                        )
                        for pt in pt_rows:
                            fx = _to_float(pt.get("erlm_exrt"), default=0.0)
                            if fx <= 0:
                                continue
                            # Key: trad_dt + pdno + sll_buy_dvsn_cd
                            key = (
                                str(pt.get("trad_dt") or "").strip()
                                + ":" + str(pt.get("pdno") or "").strip().upper()
                                + ":" + str(pt.get("sll_buy_dvsn_cd") or "").strip()
                            )
                            erlm_fx_map[key] = fx
                    except Exception as exc:
                        logger.warning("[yellow]period_trans fx fetch skipped[/yellow] excg=%s err=%s", excg, str(exc))
            except Exception as exc:
                logger.warning("[yellow]period_trans fx fetch failed[/yellow] err=%s", str(exc))

            for exchange_code in self._us_exchange_candidates():
                try:
                    rows = self.client.inquire_overseas_ccnl(days=max(int(days), 1), exchange_code=exchange_code)
                except Exception as exc:
                    failed_scopes.append(f"us:{exchange_code}")
                    logger.warning(
                        "[yellow]Broker trade sync skipped[/yellow] scope=us:%s err=%s",
                        exchange_code,
                        str(exc),
                    )
                    continue
                scanned_rows += len(rows)
                for row in rows:
                    normalized = self._normalize_overseas_row(row, exchange_code=exchange_code)
                    if not normalized:
                        continue
                    # Match erlm_exrt from period_trans by date + ticker + side
                    ord_dt = str(row.get("ord_dt") or row.get("ORD_DT") or "").strip()
                    pdno = str(row.get("pdno") or row.get("PDNO") or "").strip().upper()
                    side_cd = str(row.get("sll_buy_dvsn_cd") or row.get("SLL_BUY_DVSN_CD") or "").strip()
                    key = f"{ord_dt}:{pdno}:{side_cd}"
                    erlm_fx = erlm_fx_map.get(key, 0.0)
                    if erlm_fx > 0:
                        normalized["fx_rate"] = erlm_fx
                        native = normalized.get("price_native")
                        if native and float(native) > 0:
                            normalized["price_krw"] = float(native) * erlm_fx
                    # Skip rows without a reliable FX rate
                    if not normalized.get("fx_rate") or float(normalized["fx_rate"]) <= 0:
                        logger.warning(
                            "[yellow]Broker trade skipped (no FX)[/yellow] ticker=%s date=%s — will retry next sync",
                            normalized.get("ticker"),
                            ord_dt,
                        )
                        continue
                    normalized_rows.append(normalized)

        if self._has_kospi_market():
            now = datetime.now(timezone.utc)
            start_date = (now - timedelta(days=max(int(days), 1))).strftime("%Y%m%d")
            end_date = now.strftime("%Y%m%d")
            try:
                rows = self.client.inquire_domestic_daily_ccld(start_date=start_date, end_date=end_date)
            except Exception as exc:
                failed_scopes.append("kospi")
                logger.warning(
                    "[yellow]Broker trade sync skipped[/yellow] scope=kospi err=%s",
                    str(exc),
                )
                rows = []
            scanned_rows += len(rows)
            for row in rows:
                normalized = self._normalize_domestic_row(row)
                if normalized:
                    normalized_rows.append(normalized)

        deduped_by_id: dict[str, dict[str, Any]] = {}
        for row in normalized_rows:
            event_id = str(row.get("event_id") or "").strip()
            if event_id and event_id not in deduped_by_id:
                deduped_by_id[event_id] = row
        deduped_rows = list(deduped_by_id.values())
        existing = self.repo.existing_event_ids(
            "broker_trade_events",
            [str(row.get("event_id") or "") for row in deduped_rows],
        )
        new_rows = [row for row in deduped_rows if str(row.get("event_id") or "") not in existing]
        self.repo.append_broker_trade_events(new_rows)

        return BrokerTradeSyncResult(
            inserted_events=len(new_rows),
            scanned_rows=scanned_rows,
            skipped_existing=max(len(deduped_rows) - len(new_rows), 0),
            failed_scopes=failed_scopes,
        )


@dataclass(slots=True)
class BrokerCashSyncResult:
    """Summarizes one broker cash ledger sync execution."""

    inserted_events: int
    scanned_rows: int
    skipped_existing: int
    failed_scopes: list[str]


class BrokerCashSyncService(BrokerTradeSyncService):
    """Builds signed cash-flow ledger rows from broker execution history."""

    @staticmethod
    def _signed_trade_notional(side: str, amount: float) -> float:
        token = str(side or "").strip().upper()
        gross = abs(float(amount or 0.0))
        if token == "BUY":
            return -gross
        if token == "SELL":
            return gross
        return 0.0

    def _normalize_overseas_cash_row(
        self,
        row: dict[str, object],
        *,
        exchange_code: str,
    ) -> dict[str, Any] | None:
        qty = self._filled_qty(row)
        if qty <= 0:
            return None
        side = self._parse_side(_pick_str(row, ["sll_buy_dvsn", "SLL_BUY_DVSN", "sll_buy_dvsn_cd", "SLL_BUY_DVSN_CD"]))
        price_native = self._avg_price(row)
        api_fx = max(
            _to_float(row.get("bass_exrt"), default=0.0),
            _to_float(row.get("exrt"), default=0.0),
        )
        fx_rate = api_fx if api_fx > 0 else self._fallback_fx_rate()
        if fx_rate <= 0:
            logger.warning(
                "[yellow]Broker cash skipped (no FX)[/yellow] exchange=%s date=%s",
                exchange_code,
                _pick_str(row, ["ord_dt", "ORD_DT", "trad_dt", "TRAD_DT"]),
            )
            return None
        gross_native = qty * max(price_native, 0.0)
        signed_native = self._signed_trade_notional(side, gross_native)
        if abs(signed_native) <= 1e-9:
            return None
        occurred_at = self._parse_occurred_at(row)
        scope = f"cash:us:{exchange_code}"
        return {
            "event_id": self._stable_event_id(scope, row),
            "occurred_at": occurred_at,
            "account_id": str(self.settings.kis_account_no or "").strip() or None,
            "currency": "USD",
            "amount_native": signed_native,
            "amount_krw": signed_native * fx_rate,
            "fx_rate": fx_rate if fx_rate > 0 else None,
            "event_type": "TRADE_SETTLEMENT",
            "source": "kis_inquire_overseas_ccnl",
            "raw_payload_json": row,
        }

    def _normalize_domestic_cash_row(self, row: dict[str, object]) -> dict[str, Any] | None:
        qty = self._filled_qty(row)
        if qty <= 0:
            return None
        side = self._parse_side(_pick_str(row, ["sll_buy_dvsn_cd", "SLL_BUY_DVSN_CD", "sll_buy_dvsn", "SLL_BUY_DVSN"]))
        price_krw = self._avg_price(row)
        gross_krw = qty * max(price_krw, 0.0)
        signed_krw = self._signed_trade_notional(side, gross_krw)
        if abs(signed_krw) <= 1e-9:
            return None
        occurred_at = self._parse_occurred_at(row)
        return {
            "event_id": self._stable_event_id("cash:kospi", row),
            "occurred_at": occurred_at,
            "account_id": str(self.settings.kis_account_no or "").strip() or None,
            "currency": "KRW",
            "amount_native": signed_krw,
            "amount_krw": signed_krw,
            "fx_rate": 1.0,
            "event_type": "TRADE_SETTLEMENT",
            "source": "kis_inquire_domestic_daily_ccld",
            "raw_payload_json": row,
        }

    @staticmethod
    def _cash_event(
        *,
        event_id: str,
        occurred_at: datetime,
        currency: str,
        amount_native: float | None,
        amount_krw: float,
        fx_rate: float | None,
        event_type: str,
        source: str,
        raw_payload_json: dict[str, Any],
    ) -> dict[str, Any] | None:
        native = float(amount_native) if amount_native is not None else None
        krw = float(amount_krw or 0.0)
        if abs(krw) <= 1e-9 and (native is None or abs(native) <= 1e-9):
            return None
        return {
            "event_id": event_id,
            "occurred_at": occurred_at,
            "account_id": None,
            "currency": str(currency or "").strip().upper() or "KRW",
            "amount_native": native,
            "amount_krw": krw,
            "fx_rate": float(fx_rate) if fx_rate is not None and float(fx_rate) > 0 else None,
            "event_type": str(event_type or "").strip().upper() or "UNKNOWN",
            "source": str(source or "").strip() or None,
            "raw_payload_json": raw_payload_json,
        }

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if isinstance(value, datetime):
            return value
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _normalize_overseas_fee_rows_from_period_trans(
        self,
        row: dict[str, object],
        *,
        exchange_code: str,
    ) -> list[dict[str, Any]]:
        occurred_at = self._parse_occurred_at(row)
        api_fx = max(
            _to_float(row.get("erlm_exrt"), default=0.0),
            _to_float(row.get("bass_exrt"), default=0.0),
        )
        fx_rate = api_fx if api_fx > 0 else self._fallback_fx_rate()
        out: list[dict[str, Any]] = []

        fee_krw = (
            _to_float(row.get("dmst_wcrc_fee"), default=0.0)
            + _to_float(row.get("ovrs_wcrc_fee"), default=0.0)
        )
        if fee_krw > 0:
            payload = {"kind": "fee_krw", "exchange_code": exchange_code, "row": row}
            event = self._cash_event(
                event_id=self._stable_event_id(f"cashfee:us:{exchange_code}:krw", payload),
                occurred_at=occurred_at,
                currency="KRW",
                amount_native=-abs(fee_krw),
                amount_krw=-abs(fee_krw),
                fx_rate=1.0,
                event_type="BROKER_FEE",
                source="kis_inquire_overseas_period_trans",
                raw_payload_json=payload,
            )
            if event:
                out.append(event)

        fee_usd = (
            _to_float(row.get("dmst_frcr_fee1"), default=0.0)
            + _to_float(row.get("frcr_fee1"), default=0.0)
        )
        if fee_usd > 0:
            if fx_rate <= 0:
                logger.warning(
                    "[yellow]Broker cash fee skipped (no FX)[/yellow] exchange=%s date=%s",
                    exchange_code,
                    _pick_str(row, ["trad_dt", "TRAD_DT", "ord_dt", "ORD_DT"]),
                )
                return out
            payload = {"kind": "fee_usd", "exchange_code": exchange_code, "row": row}
            event = self._cash_event(
                event_id=self._stable_event_id(f"cashfee:us:{exchange_code}:usd", payload),
                occurred_at=occurred_at,
                currency="USD",
                amount_native=-abs(fee_usd),
                amount_krw=-abs(fee_usd) * fx_rate,
                fx_rate=fx_rate,
                event_type="BROKER_FEE",
                source="kis_inquire_overseas_period_trans",
                raw_payload_json=payload,
            )
            if event:
                out.append(event)

        return out

    def _normalize_domestic_profit_fee_tax_rows(self, row: dict[str, object]) -> list[dict[str, Any]]:
        date_token = self._parse_date_token(_pick_str(row, ["trad_dt", "TRAD_DT", "ord_dt", "ORD_DT"]))
        if not date_token:
            return []
        try:
            occurred_at = datetime.strptime(date_token, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return []

        metrics = [
            ("BROKER_FEE", _to_float(row.get("fee"), default=0.0), "fee"),
            ("BROKER_TAX", _to_float(row.get("tl_tax"), default=0.0), "tax"),
            ("BROKER_INTEREST", _to_float(row.get("loan_int"), default=0.0), "loan_int"),
        ]
        out: list[dict[str, Any]] = []
        for event_type, amount, kind in metrics:
            if amount <= 0:
                continue
            payload = {"kind": kind, "row": row}
            event = self._cash_event(
                event_id=self._stable_event_id(f"cashprofit:kospi:{kind}", payload),
                occurred_at=occurred_at,
                currency="KRW",
                amount_native=-abs(amount),
                amount_krw=-abs(amount),
                fx_rate=1.0,
                event_type=event_type,
                source="kis_inquire_domestic_period_profit",
                raw_payload_json=payload,
            )
            if event:
                out.append(event)
        return out

    @staticmethod
    def _cash_day_key(occurred_at: datetime) -> date:
        return occurred_at.date()

    def _known_cash_flows_by_day(
        self,
        *,
        rows: list[dict[str, Any]],
    ) -> dict[tuple[date, str], float]:
        out: dict[tuple[date, str], float] = {}
        for row in rows:
            event_type = str(row.get("event_type") or "").strip().upper()
            if event_type in {"CASH_CHECKPOINT", "DEPOSIT", "WITHDRAWAL"}:
                continue
            occurred_at = self._parse_datetime(row.get("occurred_at"))
            if occurred_at is None:
                continue
            currency = str(row.get("currency") or "").strip().upper() or "KRW"
            if currency == "USD":
                amount = _to_float(row.get("amount_native"), default=0.0)
            else:
                amount = _to_float(row.get("amount_krw"), default=0.0)
            if abs(amount) <= 1e-9:
                continue
            key = (self._cash_day_key(occurred_at), currency)
            out[key] = out.get(key, 0.0) + amount
        return out

    def _derive_residual_cash_events_from_snapshots(
        self,
        *,
        days: int,
        normalized_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        history_loader = getattr(self.repo, "account_cash_history", None)
        if not callable(history_loader):
            return []
        existing_loader = getattr(self.repo, "broker_cash_events_since", None)

        start_at = utc_now() - timedelta(days=max(int(days), 1) + 2)
        history_rows = history_loader(start_at=start_at)
        if len(history_rows) < 2:
            return []

        all_known_rows: list[dict[str, Any]] = list(normalized_rows)
        if callable(existing_loader):
            try:
                all_known_rows.extend(existing_loader(since=start_at))
            except Exception as exc:
                logger.warning("[yellow]Existing broker cash load skipped[/yellow] err=%s", str(exc))

        known_flow_by_day = self._known_cash_flows_by_day(rows=all_known_rows)

        latest_by_day: dict[date, dict[str, Any]] = {}
        for row in history_rows:
            snapshot_at = self._parse_datetime(row.get("snapshot_at"))
            if snapshot_at is None:
                continue
            day = self._cash_day_key(snapshot_at)
            current = latest_by_day.get(day)
            if current is None or snapshot_at > current["snapshot_at"]:
                latest_by_day[day] = {
                    "snapshot_at": snapshot_at,
                    "cash_krw": _to_float(row.get("cash_krw"), default=0.0),
                    "cash_foreign": _to_float(row.get("cash_foreign"), default=0.0),
                    "usd_krw_rate": _to_float(row.get("usd_krw_rate"), default=0.0),
                    "cash_foreign_currency": str(row.get("cash_foreign_currency") or "USD").strip().upper() or "USD",
                }

        ordered_days = sorted(latest_by_day)
        if len(ordered_days) < 2:
            return []
        last_complete_day = ordered_days[-1]
        out: list[dict[str, Any]] = []

        for prev_day, curr_day in zip(ordered_days[:-1], ordered_days[1:]):
            if curr_day == last_complete_day:
                continue
            prev = latest_by_day[prev_day]
            curr = latest_by_day[curr_day]

            prev_fx = float(prev["usd_krw_rate"] or 0.0)
            curr_fx = float(curr["usd_krw_rate"] or 0.0)
            prev_usd = float(prev["cash_foreign"])
            curr_usd = float(curr["cash_foreign"])
            if (prev_usd > 0 or curr_usd > 0) and (prev_fx <= 0 or curr_fx <= 0):
                logger.warning(
                    "[yellow]Broker cash residual inference skipped (missing FX)[/yellow] prev_day=%s curr_day=%s",
                    prev_day.isoformat(),
                    curr_day.isoformat(),
                )
                continue
            prev_domestic = float(prev["cash_krw"]) - (prev_usd * prev_fx)
            curr_domestic = float(curr["cash_krw"]) - (curr_usd * curr_fx)

            usd_residual = (curr_usd - prev_usd) - known_flow_by_day.get((curr_day, "USD"), 0.0)
            if abs(usd_residual) > 1e-6:
                usd_payload = {
                    "inferred": True,
                    "inference_reason": "account_cash_history_residual",
                    "currency": "USD",
                    "previous_snapshot_at": prev["snapshot_at"].isoformat(),
                    "current_snapshot_at": curr["snapshot_at"].isoformat(),
                    "previous_cash_native": prev_usd,
                    "current_cash_native": curr_usd,
                    "known_flow_native": known_flow_by_day.get((curr_day, "USD"), 0.0),
                    "residual_native": usd_residual,
                }
                event = self._cash_event(
                    event_id=self._stable_event_id(f"cashresidual:{curr_day.isoformat()}:USD", usd_payload),
                    occurred_at=curr["snapshot_at"],
                    currency="USD",
                    amount_native=usd_residual,
                    amount_krw=usd_residual * curr_fx,
                    fx_rate=curr_fx,
                    event_type="DEPOSIT" if usd_residual > 0 else "WITHDRAWAL",
                    source="account_cash_history_residual",
                    raw_payload_json=usd_payload,
                )
                if event:
                    out.append(event)

            krw_residual = (curr_domestic - prev_domestic) - known_flow_by_day.get((curr_day, "KRW"), 0.0)
            if abs(krw_residual) > 1_000.0:
                krw_payload = {
                    "inferred": True,
                    "inference_reason": "account_cash_history_residual",
                    "currency": "KRW",
                    "previous_snapshot_at": prev["snapshot_at"].isoformat(),
                    "current_snapshot_at": curr["snapshot_at"].isoformat(),
                    "previous_cash_native": prev_domestic,
                    "current_cash_native": curr_domestic,
                    "known_flow_native": known_flow_by_day.get((curr_day, "KRW"), 0.0),
                    "residual_native": krw_residual,
                }
                event = self._cash_event(
                    event_id=self._stable_event_id(f"cashresidual:{curr_day.isoformat()}:KRW", krw_payload),
                    occurred_at=curr["snapshot_at"],
                    currency="KRW",
                    amount_native=krw_residual,
                    amount_krw=krw_residual,
                    fx_rate=1.0,
                    event_type="DEPOSIT" if krw_residual > 0 else "WITHDRAWAL",
                    source="account_cash_history_residual",
                    raw_payload_json=krw_payload,
                )
                if event:
                    out.append(event)

        return out

    def sync_broker_cash_events(self, *, days: int = 7) -> BrokerCashSyncResult:
        """Fetches broker execution history and stores signed cash ledger events."""
        scanned_rows = 0
        failed_scopes: list[str] = []
        normalized_rows: list[dict[str, Any]] = []

        if self._has_us_market():
            for exchange_code in self._us_exchange_candidates():
                try:
                    rows = self.client.inquire_overseas_ccnl(days=max(int(days), 1), exchange_code=exchange_code)
                except Exception as exc:
                    failed_scopes.append(f"us:{exchange_code}")
                    logger.warning(
                        "[yellow]Broker cash sync skipped[/yellow] scope=us:%s err=%s",
                        exchange_code,
                        str(exc),
                    )
                    continue
                scanned_rows += len(rows)
                for row in rows:
                    normalized = self._normalize_overseas_cash_row(row, exchange_code=exchange_code)
                    if normalized:
                        normalized_rows.append(normalized)
                try:
                    fee_rows, _ = self.client.inquire_overseas_period_trans(
                        start_date=(datetime.now(timezone.utc) - timedelta(days=max(int(days), 1))).strftime("%Y%m%d"),
                        end_date=datetime.now(timezone.utc).strftime("%Y%m%d"),
                        exchange_code=exchange_code,
                    )
                except Exception as exc:
                    failed_scopes.append(f"us:{exchange_code}:fees")
                    logger.warning(
                        "[yellow]Broker cash fee sync skipped[/yellow] scope=us:%s err=%s",
                        exchange_code,
                        str(exc),
                    )
                    fee_rows = []
                scanned_rows += len(fee_rows)
                for row in fee_rows:
                    normalized_rows.extend(self._normalize_overseas_fee_rows_from_period_trans(row, exchange_code=exchange_code))

        if self._has_kospi_market():
            now = datetime.now(timezone.utc)
            start_date = (now - timedelta(days=max(int(days), 1))).strftime("%Y%m%d")
            end_date = now.strftime("%Y%m%d")
            try:
                rows = self.client.inquire_domestic_daily_ccld(start_date=start_date, end_date=end_date)
            except Exception as exc:
                failed_scopes.append("kospi")
                logger.warning(
                    "[yellow]Broker cash sync skipped[/yellow] scope=kospi err=%s",
                    str(exc),
                )
                rows = []
            scanned_rows += len(rows)
            for row in rows:
                normalized = self._normalize_domestic_cash_row(row)
                if normalized:
                    normalized_rows.append(normalized)
            try:
                profit_rows, _ = self.client.inquire_domestic_period_profit(
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as exc:
                failed_scopes.append("kospi:fees")
                logger.warning(
                    "[yellow]Broker cash fee/tax sync skipped[/yellow] scope=kospi err=%s",
                    str(exc),
                )
                profit_rows = []
            scanned_rows += len(profit_rows)
            for row in profit_rows:
                normalized_rows.extend(self._normalize_domestic_profit_fee_tax_rows(row))

        normalized_rows.extend(self._derive_residual_cash_events_from_snapshots(days=days, normalized_rows=normalized_rows))

        deduped_by_id: dict[str, dict[str, Any]] = {}
        for row in normalized_rows:
            event_id = str(row.get("event_id") or "").strip()
            if event_id and event_id not in deduped_by_id:
                deduped_by_id[event_id] = row
        deduped_rows = list(deduped_by_id.values())
        existing = self.repo.existing_event_ids(
            "broker_cash_events",
            [str(row.get("event_id") or "") for row in deduped_rows],
        )
        new_rows = [row for row in deduped_rows if str(row.get("event_id") or "") not in existing]
        self.repo.append_broker_cash_events(new_rows)

        return BrokerCashSyncResult(
            inserted_events=len(new_rows),
            scanned_rows=scanned_rows,
            skipped_existing=max(len(deduped_rows) - len(new_rows), 0),
            failed_scopes=failed_scopes,
        )


@dataclass(slots=True)
class DividendSyncResult:
    """Summarizes one dividend sync execution."""

    tickers_checked: int
    dividends_found: int
    events_inserted: int
    skipped_duplicate: int
    broker_cash_events_inserted: int = 0


class DividendSyncService:
    """Discovers dividends (US overseas + KOSPI domestic) and attributes them to agent sleeves."""

    # Korean dividend withholding is 15.4% (소득세 14% + 지방소득세 1.4%)
    _KR_WITHHOLDING_RATE = 0.154

    def __init__(
        self,
        settings: Settings,
        repo: BigQueryRepository,
        client: OpenTradingClient | None = None,
    ):
        self.settings = settings
        self.repo = repo
        self.client = client or OpenTradingClient(settings)

    @staticmethod
    def _is_kospi_ticker(ticker: str) -> bool:
        t = ticker.strip()
        return len(t) == 6 and t.isdigit()

    def _quote_exchange_candidates(self, preferred_excd: str = "") -> list[str]:
        """Returns exchange candidates for API calls."""
        out: list[str] = []
        if preferred_excd.strip():
            out.append(preferred_excd.strip().upper())
        for token in self.settings.us_quote_exchanges:
            t = str(token).strip().upper()
            if t:
                out.append(t)
        out.extend(["NAS", "NYS", "AMS"])
        dedup: list[str] = []
        for token in out:
            if token and token not in dedup:
                dedup.append(token)
        return dedup

    @staticmethod
    def _parse_dividend_per_share(row: dict[str, Any]) -> float:
        """Extracts per-share dividend amount from KIS period_rights row."""
        candidates = [
            "divi_amt",
            "divd_amt",
            "per_shr_divi_amt",
            "each_stk_divi_amt",
            "sht_divi_amt",
            "per_sto_divi_amt",  # KSD domestic field (현금배당금)
            "amt",
        ]
        for key in candidates:
            raw = row.get(key)
            if raw is not None:
                val = _to_float(raw, default=0.0)
                if val > 0:
                    return val
        return 0.0

    @staticmethod
    def _parse_date_field(row: dict[str, Any], *keys: str) -> date | None:
        """Parses YYYYMMDD or YYYY-MM-DD from multiple candidate keys."""
        for key in keys:
            raw = str(row.get(key) or "").strip().replace("-", "")
            if raw.isdigit() and len(raw) == 8:
                try:
                    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))
                except ValueError:
                    continue
        return None

    def _fetch_overseas_dividends(
        self, ticker: str, start_date: str, end_date: str,
    ) -> list[dict[str, Any]]:
        """Fetches overseas (US) dividend rows via period_rights API."""
        for excd in self._quote_exchange_candidates():
            try:
                rows = self.client.get_overseas_period_rights(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    excd=excd,
                )
                if rows:
                    return rows
            except Exception as exc:
                logger.debug("period_rights failed ticker=%s excd=%s err=%s", ticker, excd, str(exc))
                continue
        return []

    def _fetch_domestic_dividends(
        self, ticker: str, start_date: str, end_date: str,
    ) -> list[dict[str, Any]]:
        """Fetches domestic (KOSPI/KOSDAQ) dividend rows via KSD API."""
        try:
            rows = self.client.get_domestic_ksdinfo_dividend(
                start_date=start_date,
                end_date=end_date,
                sht_cd=ticker,
                gb1="0",
            )
            return rows
        except Exception as exc:
            logger.debug("ksdinfo_dividend failed ticker=%s err=%s", ticker, str(exc))
            return []

    def _broker_dividend_cash_event(
        self,
        *,
        tenant: str,
        ticker: str,
        rights_row: dict[str, Any],
        ex_dt: date,
        pay_dt: date | None,
        is_domestic: bool,
        per_share: float,
        total_shares: float,
        usd_krw: float,
        withholding: float,
        quantity_source: str,
    ) -> dict[str, Any] | None:
        """Builds one broker-level cash event for a dividend credit."""
        if total_shares <= 0 or per_share <= 0:
            return None

        event_day = pay_dt or ex_dt
        occurred_at = datetime.combine(event_day, datetime.min.time(), tzinfo=timezone.utc)
        event_id = f"cashdiv_{tenant}_{ticker}_{ex_dt.strftime('%Y%m%d')}_{event_day.strftime('%Y%m%d')}"

        if is_domestic:
            gross_krw = total_shares * per_share
            net_krw = gross_krw * (1.0 - withholding)
            amount_native = net_krw
            currency = "KRW"
            fx_rate = 1.0
        else:
            gross_usd = total_shares * per_share
            net_usd = gross_usd * (1.0 - withholding)
            net_krw = net_usd * usd_krw
            amount_native = net_usd
            currency = "USD"
            fx_rate = usd_krw

        return {
            "event_id": event_id,
            "occurred_at": occurred_at,
            "account_id": str(self.settings.kis_account_no or "").strip() or None,
            "currency": currency,
            "amount_native": float(amount_native),
            "amount_krw": float(net_krw),
            "fx_rate": float(fx_rate) if fx_rate > 0 else None,
            "event_type": "DIVIDEND_CREDIT",
            "source": "kis_ksdinfo_dividend" if is_domestic else "kis_period_rights",
            "raw_payload_json": {
                "ticker": ticker,
                "ex_date": ex_dt.isoformat(),
                "pay_date": pay_dt.isoformat() if pay_dt else None,
                "per_share": float(per_share),
                "withholding_rate": float(withholding),
                "shares_held_total": float(total_shares),
                "shares_source": quantity_source,
                "rights_row": rights_row,
            },
        }

    def sync_dividends(
        self,
        *,
        agent_ids: list[str] | None = None,
        lookback_days: int | None = None,
        usd_krw_override: float | None = None,
    ) -> DividendSyncResult:
        """Runs the full dividend discovery + attribution pipeline.

        Routes each ticker to the correct API:
        - 6-digit numeric tickers → domestic KSD dividend API (KRW)
        - Others → overseas period_rights API (USD → KRW)
        """
        lookback = lookback_days or self.settings.dividend_lookback_days
        us_withholding = self.settings.dividend_withholding_rate_us
        usd_krw = max(usd_krw_override or self.settings.usd_krw_rate, 1.0)
        tenant = self.repo.resolve_tenant_id()
        agents = agent_ids or self.settings.agent_ids

        held_tickers = self.repo.get_all_held_tickers()
        if not held_tickers:
            logger.info("[cyan]Dividend sync[/cyan] no held tickers; skipping")
            return DividendSyncResult(tickers_checked=0, dividends_found=0, events_inserted=0, skipped_duplicate=0)

        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=lookback)
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        tickers_checked = 0
        dividends_found = 0
        events_inserted = 0
        skipped_duplicate = 0
        broker_cash_events_inserted = 0

        for ticker in held_tickers:
            ticker = str(ticker).strip().upper()
            if not ticker:
                continue
            tickers_checked += 1

            is_domestic = self._is_kospi_ticker(ticker)

            # Fetch dividend data from correct API
            if is_domestic:
                rights_rows = self._fetch_domestic_dividends(ticker, start_str, end_str)
            else:
                rights_rows = self._fetch_overseas_dividends(ticker, start_str, end_str)

            if not rights_rows:
                continue

            for rrow in rights_rows:
                per_share = self._parse_dividend_per_share(rrow)
                if per_share <= 0:
                    continue

                if is_domestic:
                    ex_dt = self._parse_date_field(
                        rrow, "record_date", "bass_dt", "stdr_dt", "ex_date",
                    )
                else:
                    ex_dt = self._parse_date_field(
                        rrow, "ex_date", "ex_dt", "exdt", "bass_dt", "stdr_dt",
                    )
                if ex_dt is None:
                    continue

                dividends_found += 1
                record_dt = self._parse_date_field(rrow, "rcrd_dt", "record_dt", "record_date")
                if is_domestic:
                    pay_dt = self._parse_date_field(rrow, "divi_pay_dt", "pay_dt", "pay_date", "dvdn_pay_dt")
                else:
                    pay_dt = self._parse_date_field(rrow, "pay_dt", "pay_date", "dvdn_pay_dt")

                # Per-agent attribution
                candidate_event_ids: list[str] = []
                for aid in agents:
                    eid = f"div_{tenant}_{aid}_{ticker}_{ex_dt.strftime('%Y%m%d')}"
                    candidate_event_ids.append(eid)

                existing_ids = self.repo.dividend_event_exists(
                    event_ids=candidate_event_ids,
                    tenant_id=tenant,
                )

                insert_rows: list[dict[str, Any]] = []
                now_iso = utc_now().isoformat()
                withholding = self._KR_WITHHOLDING_RATE if is_domestic else us_withholding
                for aid in agents:
                    eid = f"div_{tenant}_{aid}_{ticker}_{ex_dt.strftime('%Y%m%d')}"
                    if eid in existing_ids:
                        skipped_duplicate += 1
                        continue

                    holdings = self.repo.agent_holdings_at_date(
                        agent_id=aid,
                        as_of_date=ex_dt,
                        tenant_id=tenant,
                    )
                    shares = holdings.get(ticker, 0.0)
                    if shares <= 0:
                        continue

                    if is_domestic:
                        # Domestic: per_share is already KRW
                        gross_krw = shares * per_share
                        net_krw = gross_krw * (1.0 - withholding)
                        gross_usd = gross_krw / usd_krw if usd_krw > 0 else 0.0
                        net_usd = net_krw / usd_krw if usd_krw > 0 else 0.0
                    else:
                        # Overseas: per_share is USD
                        gross_usd = shares * per_share
                        net_usd = gross_usd * (1.0 - withholding)
                        net_krw = net_usd * usd_krw

                    insert_rows.append({
                        "tenant_id": tenant,
                        "event_id": eid,
                        "created_at": now_iso,
                        "agent_id": aid,
                        "ticker": ticker,
                        "exchange_code": "KRX" if is_domestic else "",
                        "ex_date": ex_dt.isoformat(),
                        "record_date": record_dt.isoformat() if record_dt else None,
                        "pay_date": pay_dt.isoformat() if pay_dt else None,
                        "shares_held": float(shares),
                        "gross_per_share_usd": float(per_share if not is_domestic else per_share / usd_krw),
                        "gross_amount_usd": float(gross_usd),
                        "withholding_rate": float(withholding),
                        "net_amount_usd": float(net_usd),
                        "usd_krw_rate": float(usd_krw),
                        "net_amount_krw": float(net_krw),
                    })

                if insert_rows:
                    try:
                        self.repo.insert_dividend_events(insert_rows)
                        events_inserted += len(insert_rows)
                    except Exception as exc:
                        logger.error(
                            "[red]Dividend event insert failed[/red] ticker=%s err=%s",
                            ticker, str(exc),
                        )
                        continue

                append_cash_events = getattr(self.repo, "append_broker_cash_events", None)
                existing_event_ids = getattr(self.repo, "existing_event_ids", None)
                if callable(append_cash_events) and callable(existing_event_ids):
                    total_shares = 0.0
                    quantity_source = "account_snapshot"
                    account_holdings_at_date = getattr(self.repo, "account_holdings_at_date", None)
                    if callable(account_holdings_at_date):
                        try:
                            total_shares = float(
                                account_holdings_at_date(as_of_date=ex_dt, ticker=ticker, tenant_id=tenant).get(ticker, 0.0)
                                or 0.0
                            )
                        except Exception as exc:
                            logger.warning(
                                "[yellow]Dividend broker holding lookup skipped[/yellow] ticker=%s err=%s",
                                ticker,
                                str(exc),
                            )
                    if total_shares <= 0:
                        total_shares = sum(float(row.get("shares_held") or 0.0) for row in insert_rows)
                        quantity_source = "agent_holdings_fallback"

                    cash_row = self._broker_dividend_cash_event(
                        tenant=tenant,
                        ticker=ticker,
                        rights_row=dict(rrow),
                        ex_dt=ex_dt,
                        pay_dt=pay_dt,
                        is_domestic=is_domestic,
                        per_share=per_share,
                        total_shares=total_shares,
                        usd_krw=usd_krw,
                        withholding=withholding,
                        quantity_source=quantity_source,
                    )
                    if cash_row is not None:
                        cash_event_id = str(cash_row.get("event_id") or "")
                        existing = existing_event_ids("broker_cash_events", [cash_event_id], tenant_id=tenant)
                        if cash_event_id not in existing:
                            try:
                                append_cash_events([cash_row], tenant_id=tenant)
                                broker_cash_events_inserted += 1
                            except Exception as exc:
                                logger.error(
                                    "[red]Dividend cash event insert failed[/red] ticker=%s err=%s",
                                    ticker,
                                    str(exc),
                                )

        result = DividendSyncResult(
            tickers_checked=tickers_checked,
            dividends_found=dividends_found,
            events_inserted=events_inserted,
            skipped_duplicate=skipped_duplicate,
            broker_cash_events_inserted=broker_cash_events_inserted,
        )
        logger.info(
            "[green]Dividend sync done[/green] tickers=%d found=%d inserted=%d skipped=%d cash=%d",
            result.tickers_checked,
            result.dividends_found,
            result.events_inserted,
            result.skipped_duplicate,
            result.broker_cash_events_inserted,
        )
        return result
