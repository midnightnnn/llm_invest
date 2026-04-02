from __future__ import annotations

import logging
import math
from typing import Any

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.market_sources import live_market_sources_for_markets
from arena.models import ExecutionReport, OrderIntent, Side
from arena.open_trading.exchange_codes import (
    instrument_id_us_order_exchange,
    normalize_us_order_exchange,
    parse_target_markets,
    target_market_default_us_order_exchange,
)

logger = logging.getLogger(__name__)

_US_TARGET_MARKETS = {"nasdaq", "nyse", "amex", "us"}
_KR_TARGET_MARKETS = {"kospi", "kosdaq"}


def latest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keeps one latest market row per ticker for fast lookup."""
    ranked = sorted(rows, key=lambda r: str((r or {}).get("as_of_ts") or ""), reverse=True)
    latest: dict[str, dict[str, Any]] = {}
    for row in ranked:
        ticker = str((row or {}).get("ticker", "")).strip().upper()
        if ticker and ticker not in latest:
            latest[ticker] = row
    return list(latest.values())


def market_row_by_ticker(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Builds ticker-to-latest-row lookup map."""
    out: dict[str, dict[str, Any]] = {}
    for row in latest_rows(rows):
        ticker = str((row or {}).get("ticker", "")).strip().upper()
        if ticker and ticker not in out:
            out[ticker] = row
    return out


def live_market_feature_sources(settings: Settings) -> list[str] | None:
    """Returns live BigQuery source priority for the current target market."""
    if settings.trading_mode != "live":
        return None
    return live_market_sources_for_markets(settings.kis_target_market) or None


def fetch_market_row_from_bq(
    repo: BigQueryRepository,
    settings: Settings,
    ticker: str,
) -> dict[str, Any] | None:
    """Fetches latest market row from BigQuery when not present in context."""
    try:
        rows = repo.latest_market_features(
            tickers=[ticker],
            limit=6,
            sources=live_market_feature_sources(settings),
        )
    except Exception as exc:
        logger.warning("[yellow]Price lookup failed[/yellow] ticker=%s err=%s", ticker, str(exc))
        return None

    for row in rows:
        try:
            price_krw = float(row.get("close_price_krw") or 0.0)
        except (TypeError, ValueError):
            price_krw = 0.0
        try:
            native_price = float(row.get("close_price_native") or 0.0)
        except (TypeError, ValueError):
            native_price = 0.0
        if price_krw > 0 or native_price > 0:
            return row
    return None


def _display_ticker(ticker: str, ticker_names: dict[str, str] | None = None) -> str:
    """Formats KOSPI tickers with explicit names when the context provides them."""
    token = str(ticker or "").strip().upper()
    if not token:
        return ""
    if token.isdigit() and len(token) == 6 and ticker_names:
        name = str(ticker_names.get(token) or "").strip()
        if name:
            return f"{name}({token})"
    return token


def _configured_markets(settings: Settings) -> set[str]:
    return set(parse_target_markets(settings.kis_target_market))


def _allows_us_market(settings: Settings) -> bool:
    return bool(_configured_markets(settings) & _US_TARGET_MARKETS)


def _allows_kr_market(settings: Settings) -> bool:
    return bool(_configured_markets(settings) & _KR_TARGET_MARKETS)


def _is_korean_ticker(ticker: str) -> bool:
    token = str(ticker or "").strip().upper()
    return token.isdigit() and len(token) == 6


def _infer_market_from_identity(
    settings: Settings,
    *,
    ticker: str,
    exchange_code: str = "",
    instrument_id: str = "",
    quote_currency: str = "",
) -> str:
    if _allows_us_market(settings):
        us_exchange = normalize_us_order_exchange(exchange_code) or instrument_id_us_order_exchange(instrument_id)
        if us_exchange:
            return "us"
    if _allows_kr_market(settings) and str(exchange_code or "").strip().upper() == "KRX":
        return "kospi"

    token = str(ticker or "").strip().upper()
    if token:
        if _is_korean_ticker(token) and _allows_kr_market(settings):
            return "kospi"
        if not token[:1].isdigit() and _allows_us_market(settings):
            return "us"

    currency = str(quote_currency or "").strip().upper()
    if currency == "USD" and _allows_us_market(settings):
        return "us"
    if currency == "KRW" and _allows_kr_market(settings):
        return "kospi"
    return ""


def resolve_order_price(
    settings: Settings,
    *,
    market_row: dict[str, Any] | None,
    portfolio: dict[str, Any],
) -> tuple[float, float | None, str, float]:
    """Resolves KRW/native execution price using live FX when available."""
    row = market_row or {}
    quote_currency = str(row.get("quote_currency") or "").strip().upper()
    if not quote_currency:
        inferred_market = _infer_market_from_identity(
            settings,
            ticker=str(row.get("ticker") or ""),
            exchange_code=str(row.get("exchange_code") or ""),
            instrument_id=str(row.get("instrument_id") or ""),
        )
        if not inferred_market:
            if _allows_us_market(settings) and not _allows_kr_market(settings):
                inferred_market = "us"
            elif _allows_kr_market(settings) and not _allows_us_market(settings):
                inferred_market = "kospi"
        if inferred_market == "us":
            quote_currency = "USD"
        elif inferred_market == "kospi":
            quote_currency = "KRW"

    try:
        stored_price_krw = float(row.get("close_price_krw") or 0.0)
    except (TypeError, ValueError):
        stored_price_krw = 0.0
    try:
        native_price = float(row.get("close_price_native") or 0.0)
    except (TypeError, ValueError):
        native_price = 0.0
    try:
        stored_fx = float(row.get("fx_rate_used") or 0.0)
    except (TypeError, ValueError):
        stored_fx = 0.0
    try:
        live_fx = float(portfolio.get("usd_krw_rate") or 0.0)
    except (TypeError, ValueError):
        live_fx = 0.0

    if quote_currency == "KRW":
        if native_price <= 0 and stored_price_krw > 0:
            native_price = stored_price_krw
        return max(stored_price_krw, native_price, 0.0), native_price or None, "KRW", 1.0

    fx_rate = live_fx if live_fx > 0 else stored_fx if stored_fx > 0 else 0.0
    if native_price > 0:
        if fx_rate > 0:
            price_krw = native_price * fx_rate
            return float(price_krw), float(native_price), quote_currency, float(fx_rate)
        if stored_price_krw > 0:
            return float(stored_price_krw), float(native_price), quote_currency, 0.0
        return 0.0, float(native_price), quote_currency, 0.0

    if stored_price_krw > 0:
        inferred_native = (stored_price_krw / fx_rate) if fx_rate > 0 else None
        return float(stored_price_krw), inferred_native, quote_currency, float(fx_rate)

    return 0.0, None, quote_currency, float(fx_rate)


def format_orders_summary(
    intents: list[OrderIntent],
    raw_orders: list[dict[str, Any]],
    *,
    ticker_names: dict[str, str] | None = None,
) -> str:
    """Formats confirmed orders into a concise summary for board generation."""
    if not intents and not raw_orders:
        return "이번 사이클: 주문 없음 (HOLD)"
    lines: list[str] = []
    for intent in intents:
        display_ticker = _display_ticker(intent.ticker, ticker_names)
        lines.append(
            f"- {display_ticker} {intent.side.value} {intent.quantity}주 "
            f"@₩{intent.price_krw:,.0f} (근거: {intent.rationale[:100]})"
        )
    for order in raw_orders:
        side = str(order.get("side", "")).upper()
        ticker = str(order.get("ticker", "")).upper()
        if side == "HOLD" and ticker:
            lines.append(
                f"- {_display_ticker(ticker, ticker_names)} HOLD "
                f"(근거: {str(order.get('rationale', ''))[:100]})"
            )
    return "이번 사이클 주문 내역:\n" + "\n".join(lines) if lines else "이번 사이클: 주문 없음 (HOLD)"


def format_execution_summary(
    intents: list[OrderIntent],
    reports: list[ExecutionReport],
    *,
    ticker_names: dict[str, str] | None = None,
) -> str:
    """Formats actual execution outcomes for fact-grounded board writing."""
    if not intents:
        return "이번 사이클 실제 실행 결과: 주문 없음 (HOLD)"

    lines: list[str] = []
    padded_reports: list[ExecutionReport | None] = list(reports[: len(intents)])
    if len(padded_reports) < len(intents):
        padded_reports.extend([None] * (len(intents) - len(padded_reports)))

    for intent, report in zip(intents, padded_reports):
        display_ticker = _display_ticker(intent.ticker, ticker_names)
        qty_text = f"{intent.quantity:g}주"
        if report is None:
            lines.append(f"- {display_ticker} {intent.side.value} {qty_text} 결과 미확인")
            continue

        status = report.status.value
        if status in {"FILLED", "SIMULATED"}:
            filled_qty = max(float(report.filled_qty or 0.0), 0.0)
            avg_price = float(report.avg_price_krw or intent.price_krw or 0.0)
            lines.append(f"- {display_ticker} {intent.side.value} {filled_qty:g}주 {status} @₩{avg_price:,.0f}")
            continue
        if status == "SUBMITTED":
            lines.append(f"- {display_ticker} {intent.side.value} {qty_text} SUBMITTED (주문번호: {report.order_id})")
            continue

        reason = str(report.message or "").strip()
        if reason:
            lines.append(f"- {display_ticker} {intent.side.value} {qty_text} {status} (사유: {reason[:120]})")
        else:
            lines.append(f"- {display_ticker} {intent.side.value} {qty_text} {status}")

    return "이번 사이클 실제 실행 결과:\n" + "\n".join(lines)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return default


def _resolve_exchange_identity(
    settings: Settings,
    *,
    ticker: str,
    order: dict[str, Any],
    market_row: dict[str, Any] | None,
    position: dict[str, Any],
) -> tuple[str, str]:
    exchange_code = str(
        order.get("exchange_code")
        or (market_row or {}).get("exchange_code")
        or (position.get("exchange_code") if isinstance(position, dict) else "")
        or ""
    ).strip().upper()
    instrument_id = str(
        order.get("instrument_id")
        or (market_row or {}).get("instrument_id")
        or (position.get("instrument_id") if isinstance(position, dict) else "")
        or ""
    ).strip()
    inferred_market = _infer_market_from_identity(
        settings,
        ticker=ticker,
        exchange_code=exchange_code,
        instrument_id=instrument_id,
        quote_currency=str((market_row or {}).get("quote_currency") or ""),
    )
    if inferred_market == "us" and not exchange_code:
        exchange_code = normalize_us_order_exchange(instrument_id_us_order_exchange(instrument_id))
    if inferred_market == "us" and not exchange_code:
        exchange_code = target_market_default_us_order_exchange(settings.kis_target_market)
    if inferred_market == "kospi" and not exchange_code:
        exchange_code = "KRX"
    if not instrument_id and exchange_code:
        instrument_id = f"{exchange_code}:{ticker}"
    return exchange_code, instrument_id


def _resolve_order_quantity(
    settings: Settings,
    *,
    side_raw: str,
    size_ratio: float,
    price: float,
    sleeve_equity: float,
    holdings: dict[str, Any],
    ticker: str,
    order_budget: dict[str, Any],
    market_row: dict[str, Any] | None = None,
) -> float:
    position = holdings.get(ticker, {}) if isinstance(holdings, dict) else {}
    hold_qty = _safe_float(position.get("quantity") if isinstance(position, dict) else 0.0)

    if side_raw == "SELL":
        target_qty = max(hold_qty * size_ratio, 0.0)
        if hold_qty > 0:
            target_qty = min(target_qty, hold_qty)
        if settings.trading_mode == "live":
            # Ceil keeps small live sell intents from collapsing to zero shares.
            return float(int(min(math.ceil(target_qty), max(hold_qty, 0.0))))
        return round(max(target_qty, 0.0001), 4)

    budget = min(sleeve_equity * size_ratio, settings.max_order_krw * 0.95)
    max_buy_notional = _safe_float(order_budget.get("max_buy_notional_krw"))
    if max_buy_notional > 0:
        budget = min(budget, max_buy_notional * 0.98)

    current_value = max(hold_qty, 0.0) * float(price)
    max_position_value = max(float(settings.max_position_ratio) * float(sleeve_equity), 0.0)
    max_add_by_position = max(max_position_value - current_value, 0.0)
    budget = min(budget, max_add_by_position * 0.98)

    if budget <= 0:
        return 0.0

    # Volatility-based size cap: reduce budget for high-volatility tickers
    vol_20d = _safe_float((market_row or {}).get("volatility_20d"))
    if vol_20d > 0.03:  # >3% daily vol triggers cap
        vol_cap = max(0.5, min(1.0, 0.03 / vol_20d))
        budget = budget * vol_cap

    raw_qty = max(budget / price, 0.0)
    if settings.trading_mode == "live":
        return float(int(math.floor(raw_qty)))
    return round(max(raw_qty, 0.0001), 4)


def build_order_intents(
    *,
    repo: BigQueryRepository,
    settings: Settings,
    agent_id: str,
    sleeve_capital_krw: float,
    cycle_id: str,
    context: dict[str, Any],
    orders: list[dict[str, Any]],
    row_map: dict[str, dict[str, Any]],
    feedback_events: list[dict[str, Any]] | None = None,
) -> tuple[list[OrderIntent], set[str]]:
    """Converts model order payloads into validated execution intents."""
    intents: list[OrderIntent] = []
    tickers_mentioned: set[str] = set()

    portfolio = context.get("portfolio", {}) or {}
    if not isinstance(portfolio, dict):
        portfolio = {}
    holdings = portfolio.get("positions", {})
    if not isinstance(holdings, dict):
        holdings = {}
    order_budget = context.get("order_budget", {})
    if not isinstance(order_budget, dict):
        order_budget = {}

    sleeve_equity = _safe_float(portfolio.get("total_equity_krw"))
    if sleeve_equity <= 0:
        sleeve_equity = float(sleeve_capital_krw)

    for order in orders:
        if not isinstance(order, dict):
            continue

        side_raw = str(order.get("side", "HOLD")).strip().upper()
        ticker = str(order.get("ticker", "")).strip().upper()
        size_ratio = max(0.0, min(_safe_float(order.get("size_ratio")), 1.0))
        rationale = str(order.get("rationale", ""))[:1200]
        strategy_refs = order.get("strategy_refs", [])
        if not isinstance(strategy_refs, list):
            strategy_refs = []

        if ticker:
            tickers_mentioned.add(ticker)
        if side_raw not in {"BUY", "SELL"} or not ticker or size_ratio <= 0:
            continue

        market_row = row_map.get(ticker)
        if market_row is None:
            market_row = fetch_market_row_from_bq(repo, settings, ticker)
            if market_row is not None:
                row_map[ticker] = market_row

        price, native_price, quote_currency, fx_rate = resolve_order_price(
            settings,
            market_row=market_row,
            portfolio=portfolio,
        )
        if price <= 0:
            logger.warning("[yellow]ADK skipped intent[/yellow] agent=%s ticker=%s reason=no_price", agent_id, ticker)
            if feedback_events is not None:
                feedback_events.append({"ticker": ticker, "side": side_raw, "status": "skipped", "reason": "no_price"})
            continue

        if side_raw == "SELL" and ticker not in holdings:
            logger.warning(
                "[yellow]ADK skipped intent[/yellow] agent=%s ticker=%s reason=no_holdings_for_sell",
                agent_id,
                ticker,
            )
            if feedback_events is not None:
                feedback_events.append({"ticker": ticker, "side": side_raw, "status": "skipped", "reason": "no_holdings_for_sell"})
            continue

        quantity = _resolve_order_quantity(
            settings,
            side_raw=side_raw,
            size_ratio=size_ratio,
            price=price,
            sleeve_equity=sleeve_equity,
            holdings=holdings,
            ticker=ticker,
            order_budget=order_budget,
            market_row=market_row,
        )
        if quantity < 1 and settings.trading_mode == "live":
            logger.warning(
                "[yellow]ADK skipped intent[/yellow] agent=%s ticker=%s reason=live_qty_under_1",
                agent_id,
                ticker,
            )
            if feedback_events is not None:
                feedback_events.append({"ticker": ticker, "side": side_raw, "status": "skipped", "reason": "live_qty_under_1"})
            continue
        if quantity <= 0:
            logger.warning(
                "[yellow]ADK skipped intent[/yellow] agent=%s ticker=%s reason=no_budget",
                agent_id,
                ticker,
            )
            if feedback_events is not None:
                feedback_events.append({"ticker": ticker, "side": side_raw, "status": "skipped", "reason": "no_budget"})
            continue

        position = holdings.get(ticker, {}) if isinstance(holdings, dict) else {}
        exchange_code, instrument_id = _resolve_exchange_identity(
            settings,
            ticker=ticker,
            order=order,
            market_row=market_row,
            position=position if isinstance(position, dict) else {},
        )
        inferred_market = _infer_market_from_identity(
            settings,
            ticker=ticker,
            exchange_code=exchange_code,
            instrument_id=instrument_id,
            quote_currency=quote_currency,
        )
        if inferred_market == "us" and not exchange_code:
            logger.warning(
                "[yellow]ADK skipped intent[/yellow] agent=%s ticker=%s reason=unresolved_exchange",
                agent_id,
                ticker,
            )
            if feedback_events is not None:
                feedback_events.append({"ticker": ticker, "side": side_raw, "status": "skipped", "reason": "unresolved_exchange"})
            continue

        intents.append(
            OrderIntent(
                agent_id=agent_id,
                ticker=ticker,
                trading_mode=settings.trading_mode,
                exchange_code=exchange_code,
                instrument_id=instrument_id,
                side=Side(side_raw),
                quantity=quantity,
                price_krw=price,
                price_native=native_price,
                quote_currency=quote_currency,
                fx_rate=fx_rate,
                rationale=rationale,
                strategy_refs=[str(ref) for ref in strategy_refs][:6],
                cycle_id=cycle_id,
            )
        )
        if feedback_events is not None:
            feedback_events.append({"ticker": ticker, "side": side_raw, "status": "intent_built"})

    return intents, tickers_mentioned
