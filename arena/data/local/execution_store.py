"""Local DuckDB-backed execution store.

This is a small DuckDB-native counterpart to the BigQuery execution store. It
covers the paths needed by risk checks and paper/local agent cycles:
order-intent writes, execution-report upserts, daily counters, cooldown lookups,
and filled-report replay.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from arena.data.local.session import DuckDBSession
from arena.models import ExecutionReport, OrderIntent, RiskDecision


class LocalExecutionStore:
    """Order intent and execution report operations for local mode."""

    def __init__(self, session: DuckDBSession) -> None:
        self.session = session

    def _tenant_token(self, tenant_id: str | None = None) -> str:
        return self.session.resolve_tenant_id(tenant_id)

    @staticmethod
    def _active_statuses(*, include_simulated: bool = True, include_submitted: bool = True) -> list[str]:
        statuses = ["FILLED"]
        if include_submitted:
            statuses.append("SUBMITTED")
        if include_simulated:
            statuses.append("SIMULATED")
        return statuses

    @staticmethod
    def _normalize_trading_mode_token(trading_mode: str | None) -> str:
        token = str(trading_mode or "").strip().lower()
        return token if token in {"paper", "live"} else ""

    @staticmethod
    def _mode_expr() -> str:
        return (
            "COALESCE(trading_mode, CASE "
            "WHEN status = 'SIMULATED' THEN 'paper' "
            "WHEN status IN ('FILLED', 'SUBMITTED', 'ERROR') THEN 'live' "
            "ELSE 'paper' END)"
        )

    def recent_turnover_krw(
        self,
        day: date,
        *,
        agent_id: str | None = None,
        include_simulated: bool = True,
        include_submitted: bool = True,
        trading_mode: str | None = None,
        tenant_id: str | None = None,
    ) -> float:
        tenant = self._tenant_token(tenant_id)
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "day": day,
            "statuses": self._active_statuses(
                include_simulated=include_simulated,
                include_submitted=include_submitted,
            ),
        }
        filters = [
            "tenant_id = $tenant_id",
            "CAST(created_at AS DATE) = $day",
            "status IN (SELECT unnest($statuses))",
        ]
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._mode_expr()} = $trading_mode")
            params["trading_mode"] = mode
        if agent_id:
            filters.append("agent_id = $agent_id")
            params["agent_id"] = str(agent_id)
        rows = self.session.fetch_rows(
            f"""
            SELECT COALESCE(SUM(ABS(
              (CASE WHEN status = 'SUBMITTED' THEN requested_qty ELSE filled_qty END) * avg_price_krw
            )), 0.0) AS turnover
            FROM execution_reports
            WHERE {' AND '.join(filters)}
            """,
            params,
        )
        return float(rows[0]["turnover"]) if rows else 0.0

    def recent_intent_count(
        self,
        day: date,
        *,
        agent_id: str | None = None,
        include_simulated: bool = True,
        include_submitted: bool = True,
        trading_mode: str | None = None,
        tenant_id: str | None = None,
    ) -> int:
        tenant = self._tenant_token(tenant_id)
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "day": day,
            "statuses": self._active_statuses(
                include_simulated=include_simulated,
                include_submitted=include_submitted,
            ),
        }
        filters = [
            "tenant_id = $tenant_id",
            "CAST(created_at AS DATE) = $day",
            "status IN (SELECT unnest($statuses))",
        ]
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._mode_expr()} = $trading_mode")
            params["trading_mode"] = mode
        if agent_id:
            filters.append("agent_id = $agent_id")
            params["agent_id"] = str(agent_id)
        rows = self.session.fetch_rows(
            f"""
            SELECT COUNT(DISTINCT intent_id) AS cnt
            FROM execution_reports
            WHERE {' AND '.join(filters)}
            """,
            params,
        )
        return int(rows[0].get("cnt") or 0) if rows else 0

    def last_trade_time(
        self,
        ticker: str,
        *,
        agent_id: str | None = None,
        exchange_code: str | None = None,
        instrument_id: str | None = None,
        include_simulated: bool = True,
        include_submitted: bool = True,
        trading_mode: str | None = None,
        tenant_id: str | None = None,
    ) -> datetime | None:
        tenant = self._tenant_token(tenant_id)
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "ticker": str(ticker or "").strip().upper(),
            "statuses": self._active_statuses(
                include_simulated=include_simulated,
                include_submitted=include_submitted,
            ),
        }
        filters = [
            "tenant_id = $tenant_id",
            "ticker = $ticker",
            "status IN (SELECT unnest($statuses))",
        ]
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._mode_expr()} = $trading_mode")
            params["trading_mode"] = mode
        if agent_id:
            filters.append("agent_id = $agent_id")
            params["agent_id"] = str(agent_id)
        if instrument_id:
            filters.append("instrument_id = $instrument_id")
            params["instrument_id"] = str(instrument_id)
        elif exchange_code:
            filters.append("COALESCE(exchange_code, '') = $exchange_code")
            params["exchange_code"] = str(exchange_code)
        rows = self.session.fetch_rows(
            f"""
            SELECT created_at
            FROM execution_reports
            WHERE {' AND '.join(filters)}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            params,
        )
        return rows[0]["created_at"] if rows else None

    def write_order_intent(self, intent: OrderIntent, decision: RiskDecision) -> None:
        tenant = self._tenant_token()
        self.session.insert_dict(
            "agent_order_intents",
            {
                "tenant_id": tenant,
                "intent_id": intent.intent_id,
                "cycle_id": str(intent.cycle_id or "").strip() or None,
                "llm_call_id": str(intent.llm_call_id or "").strip() or None,
                "created_at": intent.created_at,
                "trading_mode": self._normalize_trading_mode_token(intent.trading_mode) or "paper",
                "agent_id": intent.agent_id,
                "ticker": str(intent.ticker or "").strip().upper(),
                "exchange_code": intent.exchange_code or None,
                "instrument_id": intent.instrument_id or None,
                "side": intent.side.value,
                "quantity": float(intent.quantity),
                "price_krw": float(intent.price_krw),
                "price_native": intent.price_native,
                "quote_currency": intent.quote_currency or None,
                "fx_rate": intent.fx_rate if intent.fx_rate > 0 else None,
                "notional_krw": float(intent.notional_krw),
                "rationale": intent.rationale,
                "strategy_refs": list(intent.strategy_refs or []),
                "allowed": bool(decision.allowed),
                "risk_reason": decision.reason,
                "policy_hits": list(decision.policy_hits or []),
            },
        )

    def write_execution_report(self, intent: OrderIntent, report: ExecutionReport) -> None:
        tenant = self._tenant_token()
        row = {
            "tenant_id": tenant,
            "order_id": report.order_id,
            "intent_id": intent.intent_id,
            "cycle_id": str(intent.cycle_id or "").strip() or None,
            "created_at": report.created_at,
            "trading_mode": self._normalize_trading_mode_token(intent.trading_mode) or "paper",
            "agent_id": intent.agent_id,
            "ticker": str(intent.ticker or "").strip().upper(),
            "exchange_code": intent.exchange_code or None,
            "instrument_id": intent.instrument_id or None,
            "side": intent.side.value,
            "requested_qty": float(intent.quantity),
            "filled_qty": float(report.filled_qty),
            "avg_price_krw": float(report.avg_price_krw),
            "avg_price_native": report.avg_price_native,
            "quote_currency": report.quote_currency or intent.quote_currency or None,
            "fx_rate": report.fx_rate if report.fx_rate > 0 else (intent.fx_rate if intent.fx_rate > 0 else None),
            "status": report.status.value,
            "message": report.message,
        }
        self.session.execute(
            """
            DELETE FROM execution_reports
            WHERE tenant_id = $tenant_id AND order_id = $order_id AND intent_id = $intent_id
            """,
            {"tenant_id": tenant, "order_id": report.order_id, "intent_id": intent.intent_id},
        )
        self.session.insert_dict("execution_reports", row)

    def recent_submitted_reports(
        self,
        *,
        limit: int = 200,
        lookback_hours: int = 336,
        trading_mode: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self._tenant_token()
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "lookback_hours": max(1, min(int(lookback_hours), 24 * 14)),
            "limit": max(1, min(int(limit), 1000)),
        }
        filters = [
            "tenant_id = $tenant_id",
            "status = 'SUBMITTED'",
            "created_at >= current_timestamp - ($lookback_hours * INTERVAL '1 hour')",
        ]
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._mode_expr()} = $trading_mode")
            params["trading_mode"] = mode
        return self.session.fetch_rows(
            f"""
            SELECT order_id, intent_id, created_at, trading_mode, agent_id, ticker,
                   exchange_code, instrument_id, side, requested_qty, filled_qty,
                   avg_price_krw, avg_price_native, quote_currency, fx_rate, status, message
            FROM execution_reports
            WHERE {' AND '.join(filters)}
            ORDER BY created_at DESC
            LIMIT $limit
            """,
            params,
        )

    def filled_execution_reports_since(
        self,
        *,
        since: datetime,
        trading_mode: str | None = None,
        tenant_id: str | None = None,
        include_simulated: bool = False,
    ) -> list[dict[str, Any]]:
        tenant = self._tenant_token(tenant_id)
        statuses = ["FILLED"]
        if include_simulated:
            statuses.append("SIMULATED")
        params: dict[str, Any] = {"tenant_id": tenant, "since": since, "statuses": statuses}
        filters = [
            "tenant_id = $tenant_id",
            "status IN (SELECT unnest($statuses))",
            "created_at >= $since",
        ]
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._mode_expr()} = $trading_mode")
            params["trading_mode"] = mode
        return self.session.fetch_rows(
            f"""
            SELECT order_id, intent_id, created_at, trading_mode, agent_id, ticker,
                   exchange_code, instrument_id, side, requested_qty, filled_qty,
                   avg_price_krw, avg_price_native, quote_currency, fx_rate, status, message
            FROM execution_reports
            WHERE {' AND '.join(filters)}
            ORDER BY created_at ASC, order_id ASC, intent_id ASC
            """,
            params,
        )
