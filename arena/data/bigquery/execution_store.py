"""Execution store — order intents, execution reports, turnover tracking."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from arena.memory.graph import (
    build_execution_report_graph_node,
    build_intent_execution_edge,
    build_order_intent_graph_node,
)
from arena.models import ExecutionReport, OrderIntent, RiskDecision

if TYPE_CHECKING:
    from arena.data.bigquery.session import BigQuerySession


class ExecutionStore:
    """Order intent/execution store operations."""

    def __init__(self, session: BigQuerySession, *, memory_bq_store: Any | None = None) -> None:
        self.session = session
        self._memory_bq_store = memory_bq_store

    def _tenant_token(self, tenant_id: str | None = None) -> str:
        return self.session.resolve_tenant_id(tenant_id)

    @staticmethod
    def _active_statuses(*, include_simulated: bool = True, include_submitted: bool = True) -> list[str]:
        """Builds execution status set used by daily counters and cooldown checks."""
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
    def _execution_trading_mode_expr() -> str:
        # Legacy rows may not have trading_mode populated yet.
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
        """Returns same-day traded notional for active statuses (optionally per agent)."""
        tenant = self._tenant_token(tenant_id)
        statuses = self._active_statuses(
            include_simulated=include_simulated,
            include_submitted=include_submitted,
        )
        filters: list[str] = ["tenant_id = @tenant_id", "DATE(created_at) = @day", "status IN UNNEST(@statuses)"]
        params: dict[str, object] = {"tenant_id": tenant, "day": day}
        params["statuses"] = statuses
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._execution_trading_mode_expr()} = @trading_mode")
            params["trading_mode"] = mode
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = str(agent_id)

        where = " AND ".join(filters)
        sql = f"""
        SELECT COALESCE(SUM(ABS((CASE WHEN status = 'SUBMITTED' THEN requested_qty ELSE filled_qty END) * avg_price_krw)), 0.0) AS turnover
        FROM `{self.session.dataset_fqn}.execution_reports`
        WHERE {where}
        """
        rows = self.session.fetch_rows(sql, params)
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
        """Returns same-day intent count from execution rows (optionally per agent)."""
        tenant = self._tenant_token(tenant_id)
        statuses = self._active_statuses(
            include_simulated=include_simulated,
            include_submitted=include_submitted,
        )
        filters: list[str] = ["tenant_id = @tenant_id", "DATE(created_at) = @day", "status IN UNNEST(@statuses)"]
        params: dict[str, object] = {"tenant_id": tenant, "day": day}
        params["statuses"] = statuses
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._execution_trading_mode_expr()} = @trading_mode")
            params["trading_mode"] = mode
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = str(agent_id)

        where = " AND ".join(filters)
        sql = f"""
        SELECT COUNT(DISTINCT intent_id) AS cnt
        FROM `{self.session.dataset_fqn}.execution_reports`
        WHERE {where}
        """
        rows = self.session.fetch_rows(sql, params)
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
        """Returns timestamp of latest successful execution by ticker (optionally per agent)."""
        tenant = self._tenant_token(tenant_id)
        statuses = self._active_statuses(
            include_simulated=include_simulated,
            include_submitted=include_submitted,
        )
        filters: list[str] = ["tenant_id = @tenant_id", "ticker = @ticker", "status IN UNNEST(@statuses)"]
        params: dict[str, object] = {"tenant_id": tenant, "ticker": ticker}
        params["statuses"] = statuses
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._execution_trading_mode_expr()} = @trading_mode")
            params["trading_mode"] = mode
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = str(agent_id)
        if instrument_id:
            filters.append("instrument_id = @instrument_id")
            params["instrument_id"] = str(instrument_id)
        elif exchange_code:
            filters.append("COALESCE(exchange_code, '') = @exchange_code")
            params["exchange_code"] = str(exchange_code)

        where = " AND ".join(filters)
        sql = f"""
        SELECT created_at
        FROM `{self.session.dataset_fqn}.execution_reports`
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT 1
        """
        rows = self.session.fetch_rows(sql, params)
        return rows[0]["created_at"] if rows else None

    def write_order_intent(self, intent: OrderIntent, decision: RiskDecision) -> None:
        """Persists each order intent and its risk decision."""
        tenant = self._tenant_token()
        sql = f"""
        INSERT INTO `{self.session.dataset_fqn}.agent_order_intents`
        (tenant_id, intent_id, cycle_id, llm_call_id, created_at, trading_mode, agent_id, ticker, exchange_code, instrument_id, side, quantity, price_krw, price_native, quote_currency, fx_rate, notional_krw, rationale, strategy_refs, allowed, risk_reason, policy_hits)
        VALUES
        (@tenant_id, @intent_id, @cycle_id, @llm_call_id, @created_at, @trading_mode, @agent_id, @ticker, @exchange_code, @instrument_id, @side, @quantity, @price_krw, @price_native, @quote_currency, @fx_rate, @notional_krw, @rationale, @strategy_refs, @allowed, @risk_reason, @policy_hits)
        """
        self.session.execute(
            sql,
            {
                "tenant_id": tenant,
                "intent_id": intent.intent_id,
                "cycle_id": str(intent.cycle_id or "").strip() or None,
                "llm_call_id": str(intent.llm_call_id or "").strip() or None,
                "created_at": intent.created_at,
                "trading_mode": self._normalize_trading_mode_token(intent.trading_mode) or "paper",
                "agent_id": intent.agent_id,
                "ticker": intent.ticker,
                "exchange_code": intent.exchange_code,
                "instrument_id": intent.instrument_id,
                "side": intent.side.value,
                "quantity": intent.quantity,
                "price_krw": intent.price_krw,
                "price_native": intent.price_native,
                "quote_currency": intent.quote_currency or None,
                "fx_rate": intent.fx_rate if intent.fx_rate > 0 else None,
                "notional_krw": intent.notional_krw,
                "rationale": intent.rationale,
                "strategy_refs": intent.strategy_refs,
                "allowed": decision.allowed,
                "risk_reason": decision.reason,
                "policy_hits": decision.policy_hits,
            },
        )
        upsert_nodes = getattr(self._memory_bq_store, "upsert_memory_graph_nodes", None) if self._memory_bq_store else None
        if callable(upsert_nodes):
            upsert_nodes(
                [
                    build_order_intent_graph_node(
                        intent,
                        decision.model_dump(mode="json"),
                    )
                ],
                tenant_id=tenant,
            )

    def write_execution_report(self, intent: OrderIntent, report: ExecutionReport) -> None:
        """Persists execution outcomes from the gateway (idempotent by order_id+intent_id)."""
        tenant = self._tenant_token()
        sql = f"""
        MERGE `{self.session.dataset_fqn}.execution_reports` AS t
        USING (
          SELECT
            @tenant_id AS tenant_id,
            @order_id AS order_id,
            @intent_id AS intent_id,
            @cycle_id AS cycle_id,
            @created_at AS created_at,
            @trading_mode AS trading_mode,
            @agent_id AS agent_id,
            @ticker AS ticker,
            @exchange_code AS exchange_code,
            @instrument_id AS instrument_id,
            @side AS side,
            @requested_qty AS requested_qty,
            @filled_qty AS filled_qty,
            @avg_price_krw AS avg_price_krw,
            @avg_price_native AS avg_price_native,
            @quote_currency AS quote_currency,
            @fx_rate AS fx_rate,
            @status AS status,
            @message AS message
        ) AS s
        ON t.tenant_id = s.tenant_id AND t.order_id = s.order_id AND t.intent_id = s.intent_id
        WHEN MATCHED THEN UPDATE SET
          tenant_id = s.tenant_id,
          cycle_id = s.cycle_id,
          created_at = s.created_at,
          trading_mode = s.trading_mode,
          agent_id = s.agent_id,
          ticker = s.ticker,
          exchange_code = s.exchange_code,
          instrument_id = s.instrument_id,
          side = s.side,
          requested_qty = s.requested_qty,
          filled_qty = s.filled_qty,
          avg_price_krw = s.avg_price_krw,
          avg_price_native = s.avg_price_native,
          quote_currency = s.quote_currency,
          fx_rate = s.fx_rate,
          status = s.status,
          message = s.message
        WHEN NOT MATCHED THEN INSERT
          (tenant_id, order_id, intent_id, cycle_id, created_at, trading_mode, agent_id, ticker, exchange_code, instrument_id, side, requested_qty, filled_qty, avg_price_krw, avg_price_native, quote_currency, fx_rate, status, message)
          VALUES
          (s.tenant_id, s.order_id, s.intent_id, s.cycle_id, s.created_at, s.trading_mode, s.agent_id, s.ticker, s.exchange_code, s.instrument_id, s.side, s.requested_qty, s.filled_qty, s.avg_price_krw, s.avg_price_native, s.quote_currency, s.fx_rate, s.status, s.message)
        """
        self.session.execute(
            sql,
            {
                "tenant_id": tenant,
                "order_id": report.order_id,
                "intent_id": intent.intent_id,
                "cycle_id": intent.cycle_id or None,
                "created_at": report.created_at,
                "trading_mode": self._normalize_trading_mode_token(intent.trading_mode) or "paper",
                "agent_id": intent.agent_id,
                "ticker": intent.ticker,
                "exchange_code": intent.exchange_code,
                "instrument_id": intent.instrument_id,
                "side": intent.side.value,
                "requested_qty": intent.quantity,
                "filled_qty": report.filled_qty,
                "avg_price_krw": report.avg_price_krw,
                "avg_price_native": report.avg_price_native,
                "quote_currency": report.quote_currency or intent.quote_currency or None,
                "fx_rate": report.fx_rate if report.fx_rate > 0 else (intent.fx_rate if intent.fx_rate > 0 else None),
                "status": report.status.value,
                "message": report.message,
            },
        )
        upsert_nodes = getattr(self._memory_bq_store, "upsert_memory_graph_nodes", None) if self._memory_bq_store else None
        upsert_edges = getattr(self._memory_bq_store, "upsert_memory_graph_edges", None) if self._memory_bq_store else None
        if callable(upsert_nodes):
            upsert_nodes([build_execution_report_graph_node(intent, report)], tenant_id=tenant)
        if callable(upsert_edges):
            upsert_edges([build_intent_execution_edge(intent, report)], tenant_id=tenant)

    def recent_submitted_reports(
        self,
        *,
        limit: int = 200,
        lookback_hours: int = 336,
        trading_mode: str | None = None,
    ) -> list[dict]:
        """Returns recent SUBMITTED execution rows that can be reconciled."""
        tenant = self._tenant_token()
        lim = max(1, min(int(limit), 1000))
        hrs = max(1, min(int(lookback_hours), 24 * 14))
        filters = [
            "tenant_id = @tenant_id",
            "status = 'SUBMITTED'",
            "created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @lookback_hours HOUR)",
        ]
        params: dict[str, object] = {"tenant_id": tenant, "lookback_hours": hrs, "limit": lim}
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._execution_trading_mode_expr()} = @trading_mode")
            params["trading_mode"] = mode
        where = " AND ".join(filters)
        sql = f"""
        SELECT
          order_id, intent_id, created_at, trading_mode, agent_id, ticker, exchange_code, instrument_id, side,
          requested_qty, filled_qty, avg_price_krw, avg_price_native, quote_currency, fx_rate, status, message
        FROM `{self.session.dataset_fqn}.execution_reports`
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    def filled_execution_reports_since(
        self,
        *,
        since: datetime,
        trading_mode: str | None = None,
        tenant_id: str | None = None,
        include_simulated: bool = False,
    ) -> list[dict[str, Any]]:
        """Returns FILLED (and optionally SIMULATED) execution rows on or after a timestamp."""
        tenant = self._tenant_token(tenant_id)
        statuses = ["FILLED"]
        if include_simulated:
            statuses.append("SIMULATED")
        filters = ["tenant_id = @tenant_id", "status IN UNNEST(@statuses)", "created_at >= @since"]
        params: dict[str, object] = {"tenant_id": tenant, "since": since, "statuses": statuses}
        mode = self._normalize_trading_mode_token(trading_mode)
        if mode:
            filters.append(f"{self._execution_trading_mode_expr()} = @trading_mode")
            params["trading_mode"] = mode
        where = " AND ".join(filters)
        sql = f"""
        SELECT
          order_id, intent_id, created_at, trading_mode, agent_id, ticker, exchange_code, instrument_id, side,
          requested_qty, filled_qty, avg_price_krw, avg_price_native, quote_currency, fx_rate, status, message
        FROM `{self.session.dataset_fqn}.execution_reports`
        WHERE {where}
        ORDER BY created_at ASC, order_id ASC, intent_id ASC
        """
        return self.session.fetch_rows(sql, params)
