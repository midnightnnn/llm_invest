from __future__ import annotations

import logging
from uuid import uuid4

import requests

from arena.config import Settings
from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, utc_now

logger = logging.getLogger(__name__)


class PaperBroker:
    """Simulates immediate fills for testing and paper trading."""

    def place_order(self, intent: OrderIntent, *, fx_rate: float | None = None) -> ExecutionReport:
        """Returns a simulated fill at the requested price."""
        _ = fx_rate
        order_id = f"paper_{uuid4().hex[:10]}"
        logger.info(
            "[green]SIM ORDER[/green] agent=%s ticker=%s side=%s qty=%.4f px=%.2f",
            intent.agent_id,
            intent.ticker,
            intent.side.value,
            intent.quantity,
            intent.price_krw,
        )
        return ExecutionReport(
            status=ExecutionStatus.SIMULATED,
            order_id=order_id,
            filled_qty=intent.quantity,
            avg_price_krw=intent.price_krw,
            avg_price_native=intent.price_native,
            quote_currency=intent.quote_currency,
            fx_rate=intent.fx_rate,
            message="paper fill",
            created_at=utc_now(),
        )


class KISHttpBroker:
    """Sends live orders to a user-provided HTTP execution endpoint."""

    def __init__(self, settings: Settings):
        self.endpoint = settings.kis_order_endpoint
        self.api_key = settings.kis_api_key
        self.api_secret = settings.kis_api_secret
        self.account_no = settings.kis_account_no
        if not self.endpoint:
            raise ValueError("KIS_ORDER_ENDPOINT is required for live trading")

    def place_order(self, intent: OrderIntent, *, fx_rate: float | None = None) -> ExecutionReport:
        """Posts a normalized order payload and parses the broker response."""
        _ = fx_rate
        payload = {
            "ticker": intent.ticker,
            "side": intent.side.value,
            "quantity": intent.quantity,
            "price_krw": intent.price_krw,
            "account_no": self.account_no,
            "client_order_id": intent.intent_id,
        }
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
            "X-API-SECRET": self.api_secret,
        }
        try:
            response = requests.post(self.endpoint, json=payload, headers=headers, timeout=12)
            response.raise_for_status()
            body = response.json()
            status = ExecutionStatus.FILLED if body.get("status") in {"FILLED", "DONE"} else ExecutionStatus.ERROR
            return ExecutionReport(
                status=status,
                order_id=str(body.get("order_id", f"kis_{uuid4().hex[:10]}")),
                filled_qty=float(body.get("filled_qty", 0.0)),
                avg_price_krw=float(body.get("avg_price_krw", intent.price_krw)),
                avg_price_native=float(body.get("avg_price_native", intent.price_native or 0.0)) or None,
                quote_currency=str(body.get("quote_currency", intent.quote_currency or "")).strip().upper(),
                fx_rate=float(body.get("fx_rate", intent.fx_rate or 0.0)),
                message=str(body.get("message", "ok")),
                created_at=utc_now(),
            )
        except Exception as exc:
            logger.exception("[red]LIVE ORDER FAILED[/red] intent=%s", intent.intent_id)
            return ExecutionReport(
                status=ExecutionStatus.ERROR,
                order_id=f"err_{uuid4().hex[:10]}",
                filled_qty=0.0,
                avg_price_krw=0.0,
                avg_price_native=None,
                quote_currency=intent.quote_currency,
                fx_rate=intent.fx_rate,
                message=str(exc),
                created_at=utc_now(),
            )
