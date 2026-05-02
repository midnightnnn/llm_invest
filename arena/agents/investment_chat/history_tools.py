from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from arena.agents.investment_chat.constants import AGENT_ID
from arena.agents.investment_chat.context import normalize_tenant
from arena.agents.investment_chat.utils import safe_float
from arena.tools.registry import ToolEntry


_DEFAULT_STATUSES = ["FILLED", "SIMULATED", "SUBMITTED"]
_ALLOWED_STATUSES = {"FILLED", "SIMULATED", "SUBMITTED", "ERROR", "REJECTED"}


def _normalize_scope(scope: str | None) -> str:
    token = str(scope or "all").strip().lower()
    if token in {"", "all", "any", "전체"}:
        return "all"
    if token in {"account", "total", "total_account", "total-account", "portfolio", "계좌"}:
        return "account"
    if token in {"agent", "sleeve", "agent_sleeve", "agent-sleeve", "에이전트", "슬리브"}:
        return "agent_sleeve"
    return ""


def _normalize_statuses(statuses: object) -> list[str]:
    if isinstance(statuses, (list, tuple, set)):
        raw_tokens = [str(token) for token in statuses]
    else:
        raw = str(statuses or "").strip()
        raw_tokens = raw.replace("|", ",").replace(";", ",").split(",") if raw else []
    out: list[str] = []
    for raw_token in raw_tokens:
        token = raw_token.strip().upper()
        if token in _ALLOWED_STATUSES and token not in out:
            out.append(token)
    return out or list(_DEFAULT_STATUSES)


def _list_value(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
        return [text]
    return [value]


def _datetime_value(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value or "")


def _ref_value(refs: list[Any], prefix: str) -> str:
    for ref in refs:
        text = str(ref or "")
        if text.startswith(prefix):
            return text.split(":", 1)[1].strip()
    return ""


def _format_trade(row: dict[str, Any]) -> dict[str, Any]:
    strategy_refs = _list_value(row.get("strategy_refs"))
    policy_hits = _list_value(row.get("policy_hits"))
    agent_id = str(row.get("agent_id") or "").strip().lower()
    scope = _ref_value(strategy_refs, "scope:")
    if scope not in {"account", "agent_sleeve"}:
        scope = "account" if agent_id == AGENT_ID else "agent_sleeve"
    judgment_source = _ref_value(strategy_refs, "judgment:")

    filled_qty = safe_float(row.get("filled_qty"))
    avg_price_krw = safe_float(row.get("avg_price_krw"))
    return {
        "occurred_at": _datetime_value(row.get("created_at")),
        "order_id": str(row.get("order_id") or ""),
        "intent_id": str(row.get("intent_id") or ""),
        "trading_mode": str(row.get("trading_mode") or ""),
        "agent_id": agent_id,
        "scope": scope,
        "judgment_source": judgment_source,
        "ticker": str(row.get("ticker") or "").strip().upper(),
        "exchange_code": str(row.get("exchange_code") or ""),
        "instrument_id": str(row.get("instrument_id") or ""),
        "side": str(row.get("side") or "").strip().upper(),
        "requested_qty": safe_float(row.get("requested_qty")),
        "filled_qty": filled_qty,
        "avg_price_krw": avg_price_krw,
        "avg_price_native": row.get("avg_price_native"),
        "quote_currency": str(row.get("quote_currency") or ""),
        "fx_rate": safe_float(row.get("fx_rate")),
        "notional_krw": abs(filled_qty * avg_price_krw),
        "status": str(row.get("status") or ""),
        "message": str(row.get("message") or ""),
        "rationale": str(row.get("rationale") or ""),
        "risk_reason": str(row.get("risk_reason") or ""),
        "policy_hits": policy_hits,
        "strategy_refs": strategy_refs,
    }


def build_history_tool_entries(*, repo: Any, tenant_id: str) -> list[ToolEntry]:
    tenant = normalize_tenant(tenant_id)

    def get_trade_history(
        ticker: str = "",
        agent_id: str = "",
        scope: str = "all",
        days: int = 365,
        limit: int = 50,
        statuses: str = "",
    ) -> dict[str, Any]:
        """Reads exact persisted order/execution history with rationale metadata."""
        reader = getattr(repo, "recent_trade_history", None)
        if not callable(reader):
            return {"status": "unavailable", "tenant_id": tenant, "error": "trade history reader is unavailable"}

        scope_token = _normalize_scope(scope)
        if not scope_token:
            return {"status": "error", "tenant_id": tenant, "error": "scope must be all, account, or agent_sleeve"}
        ticker_token = str(ticker or "").strip().upper()
        agent_token = str(agent_id or "").strip().lower()
        if agent_token in {"all", "전체", "*"}:
            agent_token = ""
        days_int = max(1, min(int(safe_float(days, 365)), 3650))
        limit_int = max(1, min(int(safe_float(limit, 50)), 100))
        status_tokens = _normalize_statuses(statuses)
        try:
            rows = reader(
                tenant_id=tenant,
                ticker=ticker_token,
                agent_id=agent_token,
                scope=scope_token,
                days=days_int,
                limit=limit_int,
                statuses=status_tokens,
            )
        except Exception as exc:
            return {"status": "error", "tenant_id": tenant, "error": str(exc)}

        trades = [_format_trade(dict(row or {})) for row in rows or []]
        return {
            "status": "ok",
            "tenant_id": tenant,
            "filters": {
                "ticker": ticker_token,
                "agent_id": agent_token,
                "scope": scope_token,
                "days": days_int,
                "limit": limit_int,
                "statuses": status_tokens,
            },
            "count": len(trades),
            "trades": trades,
        }

    return [
        ToolEntry(
            tool_id="get_trade_history",
            name="get_trade_history",
            description=(
                "Reads exact persisted trade history for this Arena tenant, including execution status, "
                "filled quantity, price, rationale, risk reason, and strategy refs. Read-only."
            ),
            category="account",
            callable=get_trade_history,
            tier="core",
            label_ko="거래 이력",
            sort_order=4,
        )
    ]
