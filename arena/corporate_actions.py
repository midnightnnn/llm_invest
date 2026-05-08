from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


_ACTIVE_STATUSES = {"", "active", "planned", "pending"}


def _bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _float_value(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio_values(entry: dict[str, Any]) -> tuple[float | None, float | None]:
    numerator = _float_value(
        entry.get("ratio_numerator", entry.get("numerator", entry.get("new_shares", entry.get("new"))))
    )
    denominator = _float_value(
        entry.get("ratio_denominator", entry.get("denominator", entry.get("old_shares", entry.get("old"))))
    )
    if numerator is not None and denominator is not None:
        return numerator, denominator

    raw_ratio = str(entry.get("ratio") or "").strip()
    if not raw_ratio:
        return numerator, denominator
    for sep in (":", "/"):
        if sep in raw_ratio:
            left, right = raw_ratio.split(sep, 1)
            return _float_value(left), _float_value(right)
    parsed = _float_value(raw_ratio)
    if parsed is None:
        return numerator, denominator
    return parsed, 1.0


def _date_value(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _as_date(value: datetime | date | None) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _raw_action_entries(value: Any) -> list[dict[str, Any]]:
    raw = value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return []

    if isinstance(raw, dict):
        for key in ("actions", "corporate_actions", "items"):
            items = raw.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        if "ticker" in raw:
            return [raw]
        out: list[dict[str, Any]] = []
        for ticker, item in raw.items():
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row.setdefault("ticker", ticker)
            out.append(row)
        return out

    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    return []


def normalize_planned_corporate_actions(value: Any) -> list[dict[str, Any]]:
    """Returns sanitized planned corporate actions from runtime config payloads."""
    out: list[dict[str, Any]] = []
    for entry in _raw_action_entries(value):
        ticker = str(entry.get("ticker") or "").strip().upper()
        numerator, denominator = _ratio_values(entry)
        if not ticker or numerator is None or denominator is None or numerator <= 0 or denominator <= 0:
            continue

        normalized: dict[str, Any] = {
            "ticker": ticker,
            "action_type": str(entry.get("action_type") or entry.get("type") or "corporate_action").strip().lower(),
            "ratio_numerator": int(numerator) if float(numerator).is_integer() else numerator,
            "ratio_denominator": int(denominator) if float(denominator).is_integer() else denominator,
        }
        effective_date = _date_value(entry.get("effective_date") or entry.get("ex_date"))
        if effective_date is not None:
            normalized["effective_date"] = effective_date.isoformat()
        status = str(entry.get("status") or "").strip().lower()
        if status:
            normalized["status"] = status
        if "cash_in_lieu" in entry:
            normalized["cash_in_lieu"] = _bool_value(entry.get("cash_in_lieu"))
        if "block_trading" in entry:
            normalized["block_trading"] = _bool_value(entry.get("block_trading"), True)
        tolerance = _float_value(entry.get("quantity_tolerance", entry.get("tolerance")))
        if tolerance is not None and tolerance >= 0:
            normalized["quantity_tolerance"] = tolerance
        note = str(entry.get("note") or entry.get("reason") or "").strip()
        if note:
            normalized["note"] = note
        out.append(normalized)
    return out


@dataclass(frozen=True)
class PlannedCorporateAction:
    ticker: str
    action_type: str
    ratio_numerator: float
    ratio_denominator: float
    effective_date: date | None = None
    status: str = ""
    cash_in_lieu: bool = False
    block_trading: bool = True
    quantity_tolerance: float = 1e-6
    note: str = ""

    @property
    def ratio(self) -> float:
        return self.ratio_numerator / self.ratio_denominator

    def is_active(self) -> bool:
        return self.status.strip().lower() in _ACTIVE_STATUSES

    def is_effective_on(self, as_of: datetime | date | None) -> bool:
        if self.effective_date is None:
            return True
        current_date = _as_date(as_of)
        if current_date is None:
            return False
        return current_date >= self.effective_date


def planned_corporate_actions(value: Any) -> list[PlannedCorporateAction]:
    """Parses configured planned corporate actions into typed values."""
    out: list[PlannedCorporateAction] = []
    for entry in normalize_planned_corporate_actions(value):
        out.append(
            PlannedCorporateAction(
                ticker=str(entry["ticker"]),
                action_type=str(entry["action_type"]),
                ratio_numerator=float(entry["ratio_numerator"]),
                ratio_denominator=float(entry["ratio_denominator"]),
                effective_date=_date_value(entry.get("effective_date")),
                status=str(entry.get("status") or ""),
                cash_in_lieu=_bool_value(entry.get("cash_in_lieu")),
                block_trading=_bool_value(entry.get("block_trading"), True),
                quantity_tolerance=max(float(entry.get("quantity_tolerance", 1e-6) or 0.0), 0.0),
                note=str(entry.get("note") or ""),
            )
        )
    return out


def active_action_for_ticker(
    value: Any,
    ticker: str,
    *,
    as_of: datetime | date | None = None,
    require_effective: bool = False,
    require_trading_block: bool = False,
) -> PlannedCorporateAction | None:
    token = str(ticker or "").strip().upper()
    if not token:
        return None
    for action in planned_corporate_actions(value):
        if action.ticker != token or not action.is_active():
            continue
        if require_trading_block and not action.block_trading:
            continue
        if require_effective and not action.is_effective_on(as_of):
            continue
        return action
    return None


def corporate_action_adjustment_candidate(
    value: Any,
    *,
    ticker: str,
    ledger_quantity: float,
    broker_quantity: float,
    as_of: datetime | date | None,
) -> dict[str, Any] | None:
    action = active_action_for_ticker(value, ticker, as_of=as_of, require_effective=True)
    if action is None:
        return None
    ledger_qty = float(ledger_quantity or 0.0)
    broker_qty = float(broker_quantity or 0.0)
    if ledger_qty <= 0 or broker_qty < 0:
        return None

    expected_qty = ledger_qty * action.ratio
    tolerance = max(action.quantity_tolerance, 1e-9)
    candidates: list[tuple[str, float]] = [("theoretical", expected_qty)]
    floor_qty = math.floor(expected_qty + 1e-12)
    fractional_qty = expected_qty - floor_qty
    if action.cash_in_lieu and fractional_qty > tolerance:
        candidates.append(("cash_in_lieu_floor", float(floor_qty)))

    for basis, candidate_qty in candidates:
        if abs(broker_qty - candidate_qty) > tolerance:
            continue
        return {
            "action_type": action.action_type,
            "ticker": action.ticker,
            "ratio_numerator": action.ratio_numerator,
            "ratio_denominator": action.ratio_denominator,
            "effective_date": action.effective_date.isoformat() if action.effective_date else "",
            "cash_in_lieu": action.cash_in_lieu,
            "ledger_quantity": ledger_qty,
            "broker_quantity": broker_qty,
            "expected_post_quantity": expected_qty,
            "matched_broker_quantity": broker_qty,
            "matched_basis": basis,
            "suggested_delta_quantity": broker_qty - ledger_qty,
            "fractional_quantity": fractional_qty,
            "quantity_tolerance": tolerance,
            "manual_adjustment_reason": f"corporate_action {action.action_type} {action.ticker}",
        }
    return None
