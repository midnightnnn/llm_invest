from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from arena.config import Settings
from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, Side, utc_now
from arena.open_trading.client import OpenTradingClient
from arena.open_trading.exchange_codes import (
    instrument_id_us_order_exchange,
    normalize_us_order_exchange as _normalize_us_order_exchange_token,
    order_to_quote_exchange as _order_to_quote_exchange,
    parse_target_markets,
    target_market_default_us_order_exchange,
    us_order_exchange_candidates,
)

logger = logging.getLogger(__name__)


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


def _pick_str(row: dict[str, object], keys: list[str]) -> str:
    """Returns the first non-empty string value for keys."""
    for key in keys:
        if key in row:
            value = str(row.get(key, "")).strip()
            if value:
                return value
    return ""


def _normalize_us_order_exchange(exchange_code: str, default_exchange: str) -> str:
    """Normalizes US exchange code into KIS overseas order code."""
    token = _normalize_us_order_exchange_token(exchange_code)
    if token:
        return token
    fallback = _normalize_us_order_exchange_token(default_exchange)
    if fallback:
        return fallback
    raise ValueError(
        "unable to resolve US order exchange "
        f"(exchange_code={str(exchange_code or '').strip().upper() or '-'} "
        f"default_exchange={str(default_exchange or '').strip().upper() or '-'})"
    )


def _us_reconcile_exchange_candidates(exchange_code: str, default_exchange: str) -> list[str]:
    """Builds a stable US exchange candidate list for fill-reconcile lookups."""
    candidates: list[str] = []
    for normalized in us_order_exchange_candidates(exchange_code, default_exchange):
        if normalized not in candidates:
            candidates.append(normalized)
    return candidates or list(us_order_exchange_candidates())


def _infer_us_tick_size(local_price: float) -> float:
    """Returns a conservative US tick size used when explicit metadata is unavailable."""
    px = max(float(local_price), 0.0)
    if px >= 1.0:
        return 0.01
    return 0.0001


def _infer_krx_tick_size(local_price: float) -> float:
    """Returns fallback KRX tick size from the current domestic price band."""
    px = max(float(local_price), 0.0)
    if px < 2_000:
        return 1.0
    if px < 5_000:
        return 5.0
    if px < 20_000:
        return 10.0
    if px < 50_000:
        return 50.0
    if px < 200_000:
        return 100.0
    if px < 500_000:
        return 500.0
    return 1_000.0


def _round_to_tick(price: float, tick: float, side: Side) -> float:
    """Rounds price to tradable tick. BUY rounds up, SELL rounds down."""
    px = max(float(price), 0.0)
    tk = max(float(tick), 0.0001)
    steps = px / tk
    if side == Side.BUY:
        rounded = math.ceil(steps) * tk
    else:
        rounded = math.floor(steps) * tk
    return max(round(rounded + 1e-12, 6), tk)


def _runtime_tenant() -> str:
    """Returns the current runtime tenant label for order logs."""
    return (os.getenv("ARENA_TENANT_ID", "") or "").strip().lower() or "-"


class KISOpenTradingBroker:
    """Places live orders directly through Korea Investment open-trading API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenTradingClient(settings=settings)

    def _default_us_order_exchange(self) -> str:
        target_default = target_market_default_us_order_exchange(self.settings.kis_target_market)
        if target_default:
            return target_default
        return ""

    def _configured_markets(self) -> list[str]:
        return parse_target_markets(self.settings.kis_target_market)

    def _allows_us_market(self) -> bool:
        return any(market in {"nasdaq", "nyse", "amex", "us"} for market in self._configured_markets())

    def _allows_kospi_market(self) -> bool:
        return any(market in {"kospi", "kosdaq"} for market in self._configured_markets())

    @staticmethod
    def _is_kospi_ticker(ticker: str) -> bool:
        token = str(ticker or "").strip().upper()
        return token.isdigit() and len(token) == 6

    def _live_slippage_bps(self, intent: OrderIntent) -> float:
        """Returns dynamic slippage buffer in bps based on order notional."""
        base = max(float(self.settings.live_slippage_bps_base), 0.0)
        impact = max(float(self.settings.live_slippage_bps_impact), 0.0)
        cap = max(float(self.settings.live_slippage_bps_max), base)
        scale = max(float(intent.notional_krw), 0.0) / 1_000_000.0
        dynamic = base + (math.sqrt(scale) * impact if scale > 0 else 0.0)
        return min(dynamic, cap)

    def _resolved_fx_rate(self, intent: OrderIntent | None = None, fx_rate: float | None = None) -> float:
        """Returns the FX rate to use for US order conversion.

        Raises ValueError if no live FX rate is available — orders must not
        proceed with a stale config default.
        """
        if fx_rate is not None and float(fx_rate) > 0:
            return float(fx_rate)
        if intent is not None and float(intent.fx_rate or 0.0) > 0:
            return float(intent.fx_rate)
        raise ValueError(
            "No live USD/KRW rate available for order. "
            "Ensure account snapshot has usd_krw_rate before placing US orders."
        )

    def _infer_market(
        self,
        *,
        ticker: str = "",
        exchange_code: str = "",
        instrument_id: str = "",
        quote_currency: str = "",
    ) -> str:
        """Infers one concrete order market from the runtime config plus order identity."""
        if self._allows_us_market():
            us_exchange = _normalize_us_order_exchange_token(exchange_code) or instrument_id_us_order_exchange(instrument_id)
            if us_exchange:
                return "us"
        if self._allows_kospi_market() and str(exchange_code or "").strip().upper() == "KRX":
            return "kospi"

        token = str(ticker or "").strip().upper()
        if token:
            if self._is_kospi_ticker(token) and self._allows_kospi_market():
                return "kospi"
            if not token[:1].isdigit() and self._allows_us_market():
                return "us"

        currency = str(quote_currency or "").strip().upper()
        if currency == "USD" and self._allows_us_market():
            return "us"
        if currency == "KRW" and self._allows_kospi_market():
            return "kospi"

        if self._allows_us_market() and not self._allows_kospi_market():
            return "us"
        if self._allows_kospi_market() and not self._allows_us_market():
            return "kospi"
        raise ValueError(
            "unable to infer order market "
            f"(target_market={self.settings.kis_target_market} ticker={token or '-'} "
            f"exchange_code={str(exchange_code or '').strip().upper() or '-'} "
            f"instrument_id={str(instrument_id or '').strip() or '-'})"
        )

    def _to_order_payload(
        self,
        intent: OrderIntent,
        *,
        fx_rate: float | None = None,
    ) -> tuple[str, int, float, float, float, str, float]:
        """Converts normalized intent into market-specific payload with slippage-adjusted limit."""
        qty = int(math.floor(intent.quantity))
        if qty <= 0:
            raise ValueError("quantity rounded down to zero")

        bps = self._live_slippage_bps(intent)
        if intent.side == Side.BUY:
            adjusted_price_krw = float(intent.price_krw) * (1.0 + bps / 10_000.0)
        else:
            adjusted_price_krw = float(intent.price_krw) * max(0.01, 1.0 - bps / 10_000.0)

        market = self._infer_market(
            ticker=intent.ticker,
            exchange_code=intent.exchange_code or "",
            instrument_id=intent.instrument_id or "",
            quote_currency=intent.quote_currency or "",
        )

        if market == "us":
            fx = self._resolved_fx_rate(intent, fx_rate)
            local_raw = max(adjusted_price_krw / fx, 0.0001)
            local_tick = _infer_us_tick_size(local_raw)
            local_limit = _round_to_tick(local_raw, local_tick, intent.side)
            exchange_hint = intent.exchange_code or instrument_id_us_order_exchange(intent.instrument_id)
            order_exchange = _normalize_us_order_exchange(exchange_hint, self._default_us_order_exchange())
            return "us", qty, local_limit, local_limit * fx, bps, order_exchange, fx

        if market == "kospi":
            local_limit = _round_to_tick(adjusted_price_krw, _infer_krx_tick_size(adjusted_price_krw), intent.side)
            return "kospi", qty, local_limit, local_limit, bps, "KRX", 1.0

        raise ValueError(f"unsupported market for open-trading broker: {self.settings.kis_target_market}")

    def _fetch_live_price(self, *, ticker: str, market: str, exchange_code: str) -> float:
        """Best-effort live price fetch in native currency. Returns 0.0 on failure."""
        try:
            if market == "us":
                excd = _order_to_quote_exchange(exchange_code) or "NAS"
                quote = self.client.get_overseas_price(ticker=ticker, excd=excd)
                return _to_float(quote.get("last"), default=0.0)
            else:
                quote = self.client.get_domestic_price(ticker=ticker)
                return _to_float(quote.get("stck_prpr"), default=0.0)
        except Exception as exc:
            logger.debug("Live price fetch failed ticker=%s err=%s", ticker, exc)
            return 0.0

    def _query_fill_once(
        self,
        *,
        market: str,
        order_id: str,
        ticker: str,
        qty: int,
        fallback_price_krw: float,
        message: str,
        exchange_code: str = "",
        fx_rate: float | None = None,
    ) -> ExecutionReport | None:
        """Queries KIS once and returns FILLED report if execution is observed."""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if market == "kospi":
            rows = self.client.inquire_domestic_daily_ccld(
                start_date=today,
                end_date=today,
                pdno=ticker,
                odno=order_id,
            )
        else:
            rows = []
            last_exc: Exception | None = None
            for excd in _us_reconcile_exchange_candidates(exchange_code, self._default_us_order_exchange()):
                try:
                    scanned = self.client.inquire_overseas_ccnl(
                        days=7,
                        pdno=ticker,
                        exchange_code=excd,
                    )
                except Exception as exc:
                    last_exc = exc
                    continue
                if scanned:
                    rows.extend(scanned)
            if not rows and last_exc is not None:
                raise last_exc

        best_filled = 0.0
        best_avg_ccy = 0.0

        if not rows:
            logger.debug("[reconcile] no rows from KIS API for ticker=%s order_id=%s", ticker, order_id)

        for row in rows:
            odno = _pick_str(row, ["odno", "ODNO", "ord_no", "ORD_NO", "orgn_odno", "ORGN_ODNO"])
            if odno and odno != order_id:
                continue

            filled = max(
                _to_float(row.get("ft_ccld_qty"), default=0.0),
                _to_float(row.get("CCLD_QTY"), default=0.0),
                _to_float(row.get("ccld_qty"), default=0.0),
                _to_float(row.get("tot_ccld_qty"), default=0.0),
                _to_float(row.get("TOT_CCLD_QTY"), default=0.0),
            )
            if filled <= 0:
                continue

            avg_ccy = max(
                _to_float(row.get("ft_ccld_unpr3"), default=0.0),
                _to_float(row.get("CCLD_UNPR"), default=0.0),
                _to_float(row.get("ccld_unpr"), default=0.0),
                _to_float(row.get("avg_pric"), default=0.0),
                _to_float(row.get("avg_unpr"), default=0.0),
                _to_float(row.get("AVG_UNPR"), default=0.0),
            )

            if filled > best_filled:
                best_filled = filled
                best_avg_ccy = avg_ccy

        if best_filled <= 0:
            return None

        avg_price_krw = float(fallback_price_krw)
        if best_avg_ccy > 0:
            if market == "us":
                avg_price_krw = best_avg_ccy * self._resolved_fx_rate(fx_rate=fx_rate)
            else:
                avg_price_krw = best_avg_ccy

        return ExecutionReport(
            status=ExecutionStatus.FILLED,
            order_id=order_id,
            filled_qty=min(float(qty), float(best_filled)),
            avg_price_krw=float(avg_price_krw),
            avg_price_native=float(best_avg_ccy) if best_avg_ccy > 0 else None,
            quote_currency="USD" if market == "us" else "KRW",
            fx_rate=self._resolved_fx_rate(fx_rate=fx_rate) if market == "us" else 1.0,
            message=message,
            created_at=utc_now(),
        )

    def _confirm_fill(
        self,
        *,
        market: str,
        order_id: str,
        intent: OrderIntent,
        qty: int,
        fallback_price_krw: float,
        exchange_code: str = "",
        fx_rate: float | None = None,
    ) -> ExecutionReport | None:
        """Best-effort fill confirmation; returns FILLED report when detected."""
        if not self.settings.kis_confirm_fills:
            return None

        timeout_s = max(1, int(self.settings.kis_confirm_timeout_seconds))
        poll_s = max(0.5, float(self.settings.kis_confirm_poll_seconds))
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            try:
                confirmed = self._query_fill_once(
                    market=market,
                    order_id=order_id,
                    ticker=intent.ticker,
                    qty=qty,
                    fallback_price_krw=fallback_price_krw,
                    message="confirmed",
                    exchange_code=exchange_code,
                    fx_rate=fx_rate,
                )
                if confirmed is not None:
                    return confirmed
            except Exception as exc:
                logger.warning(
                    "[yellow]Fill confirm skipped[/yellow] ordno=%s err=%s",
                    order_id,
                    str(exc),
                )

            time.sleep(poll_s)

        return None

    def reconcile_submitted(
        self,
        *,
        order_id: str,
        ticker: str,
        exchange_code: str = "",
        side: str,
        requested_qty: float,
        fallback_price_krw: float,
        fx_rate: float | None = None,
    ) -> ExecutionReport | None:
        """Attempts one-shot reconciliation for prior SUBMITTED orders."""
        _ = side  # kept for interface completeness/logging extension.
        try:
            market = self._infer_market(
                ticker=ticker,
                exchange_code=exchange_code,
            )
        except ValueError:
            return None
        qty = max(1, int(math.floor(float(requested_qty))))
        return self._query_fill_once(
            market=market,
            order_id=str(order_id),
            ticker=str(ticker).strip().upper(),
            qty=qty,
            fallback_price_krw=max(float(fallback_price_krw), 1.0),
            message="reconciled",
            exchange_code=str(exchange_code or ""),
            fx_rate=fx_rate,
        )

    def place_order(self, intent: OrderIntent, *, fx_rate: float | None = None) -> ExecutionReport:
        """Submits one order and maps response into execution report."""
        try:
            market, qty, limit_price, limit_price_krw, slippage_bps, order_exchange, resolved_fx = self._to_order_payload(
                intent,
                fx_rate=fx_rate,
            )
        except Exception as exc:
            return ExecutionReport(
                status=ExecutionStatus.REJECTED,
                order_id=f"reject_{uuid4().hex[:10]}",
                filled_qty=0.0,
                avg_price_krw=0.0,
                avg_price_native=None,
                quote_currency=intent.quote_currency,
                fx_rate=intent.fx_rate,
                message=str(exc),
                created_at=utc_now(),
            )

        # Re-anchor limit price to live quote when stale close price would miss the fill.
        live_native = self._fetch_live_price(ticker=intent.ticker, market=market, exchange_code=order_exchange)
        if live_native > 0:
            stale_limit = limit_price
            bps = slippage_bps
            if intent.side == Side.BUY and live_native > limit_price:
                adjusted = live_native * (1.0 + bps / 10_000.0)
                if market == "us":
                    limit_price = _round_to_tick(adjusted, _infer_us_tick_size(adjusted), Side.BUY)
                    limit_price_krw = limit_price * resolved_fx
                else:
                    limit_price = _round_to_tick(adjusted, _infer_krx_tick_size(adjusted), Side.BUY)
                    limit_price_krw = limit_price
                logger.info(
                    "[yellow]LIVE ADJUST BUY[/yellow] tenant=%s ticker=%s stale=%.4f live=%.4f new=%.4f",
                    _runtime_tenant(),
                    intent.ticker,
                    stale_limit,
                    live_native,
                    limit_price,
                )
            elif intent.side == Side.SELL and live_native < limit_price:
                adjusted = live_native * max(0.01, 1.0 - bps / 10_000.0)
                if market == "us":
                    limit_price = _round_to_tick(adjusted, _infer_us_tick_size(adjusted), Side.SELL)
                    limit_price_krw = limit_price * resolved_fx
                else:
                    limit_price = _round_to_tick(adjusted, _infer_krx_tick_size(adjusted), Side.SELL)
                    limit_price_krw = limit_price
                logger.info(
                    "[yellow]LIVE ADJUST SELL[/yellow] tenant=%s ticker=%s stale=%.4f live=%.4f new=%.4f",
                    _runtime_tenant(),
                    intent.ticker,
                    stale_limit,
                    live_native,
                    limit_price,
                )

        try:
            if market == "us":
                result = self.client.place_overseas_order(
                    ticker=intent.ticker,
                    side=intent.side.value.lower(),
                    quantity=qty,
                    limit_price=limit_price,
                    exchange_code=order_exchange,
                    ord_dvsn="00",
                )
            else:
                result = self.client.place_domestic_cash_order(
                    ticker=intent.ticker,
                    side=intent.side.value.lower(),
                    quantity=qty,
                    limit_price=limit_price,
                    market_code="KRX",
                    ord_dvsn="00",
                )

            output = result.get("output", {})
            order_id = str(
                output.get("ODNO")
                or output.get("odno")
                or output.get("ORD_NO")
                or output.get("ord_no")
                or f"kis_{uuid4().hex[:10]}"
            )
            message = str(result.get("msg1") or "accepted")

            logger.info(
                "[green]LIVE ORDER[/green] tenant=%s market=%s exchange=%s ticker=%s side=%s qty=%d ordno=%s slippage_bps=%.2f",
                _runtime_tenant(),
                market,
                order_exchange,
                intent.ticker,
                intent.side.value,
                qty,
                order_id,
                slippage_bps,
            )

            report = ExecutionReport(
                status=ExecutionStatus.SUBMITTED,
                order_id=order_id,
                filled_qty=0.0,
                avg_price_krw=float(limit_price_krw),
                avg_price_native=float(limit_price),
                quote_currency="USD" if market == "us" else "KRW",
                fx_rate=resolved_fx,
                message=message,
                created_at=utc_now(),
            )

            confirmed = self._confirm_fill(
                market=market,
                order_id=order_id,
                intent=intent,
                qty=qty,
                fallback_price_krw=float(limit_price_krw),
                exchange_code=order_exchange,
                fx_rate=resolved_fx,
            )
            return confirmed or report
        except Exception as exc:
            logger.exception("[red]KIS live order failed[/red] tenant=%s intent=%s", _runtime_tenant(), intent.intent_id)
            return ExecutionReport(
                status=ExecutionStatus.ERROR,
                order_id=f"err_{uuid4().hex[:10]}",
                filled_qty=0.0,
                avg_price_krw=0.0,
                avg_price_native=None,
                quote_currency=intent.quote_currency,
                fx_rate=intent.fx_rate if intent.fx_rate > 0 else (resolved_fx if 'resolved_fx' in locals() else 0.0),
                message=str(exc),
                created_at=utc_now(),
            )
