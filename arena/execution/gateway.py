from __future__ import annotations

import logging
import os
from uuid import uuid4

from arena.broker.base import BrokerClient
from arena.config import AgentConfig, merge_agent_risk_settings
from arena.data.bq import BigQueryRepository
from arena.memory.store import MemoryStore
from arena.models import AccountSnapshot, ExecutionReport, ExecutionStatus, OrderIntent, RiskDecision, utc_now
from arena.risk import RiskEngine

logger = logging.getLogger(__name__)


def _runtime_tenant() -> str:
    """Returns the current runtime tenant label when repo resolution is unavailable."""
    return (os.getenv("ARENA_TENANT_ID", "") or "").strip().lower() or "-"


class ExecutionGateway:
    """Centralizes policy checks and order submission for all agents."""

    def __init__(
        self,
        repo: BigQueryRepository,
        risk_engine: RiskEngine,
        broker: BrokerClient,
        memory_store: MemoryStore,
        agent_configs: dict[str, AgentConfig] | None = None,
    ):
        self.repo = repo
        self.risk_engine = risk_engine
        self.broker = broker
        self.memory_store = memory_store
        self._agent_configs = agent_configs or {}

    def _tenant_label(self) -> str:
        """Resolves tenant label for order logs."""
        resolve = getattr(self.repo, "resolve_tenant_id", None)
        if callable(resolve):
            try:
                tenant = str(resolve() or "").strip().lower()
            except Exception:
                tenant = ""
            if tenant:
                return tenant
        return _runtime_tenant()

    def process(self, intent: OrderIntent, snapshot: AccountSnapshot) -> ExecutionReport:
        """Processes one intent through risk validation and broker execution."""
        if not str(intent.trading_mode or "").strip():
            intent.trading_mode = self.risk_engine.settings.trading_mode
        now = utc_now()
        include_simulated = self.risk_engine.settings.trading_mode != "live"
        daily_turnover = self.repo.recent_turnover_krw(
            now.date(),
            agent_id=intent.agent_id,
            include_simulated=include_simulated,
            trading_mode=intent.trading_mode,
        )
        daily_orders = self.repo.recent_intent_count(
            now.date(),
            agent_id=intent.agent_id,
            include_simulated=include_simulated,
            trading_mode=intent.trading_mode,
        )
        last_trade_at = self.repo.last_trade_time(
            intent.ticker,
            agent_id=intent.agent_id,
            exchange_code=intent.exchange_code or None,
            instrument_id=intent.instrument_id or None,
            include_simulated=include_simulated,
            trading_mode=intent.trading_mode,
        )
        # Per-agent risk: if agent has risk_overrides, create a temporary RiskEngine
        ac = self._agent_configs.get(intent.agent_id)
        if ac and ac.risk_overrides:
            merged_settings = merge_agent_risk_settings(self.risk_engine.settings, ac)
            risk_engine = RiskEngine(settings=merged_settings)
        else:
            risk_engine = self.risk_engine
        decision = risk_engine.evaluate(
            intent=intent,
            snapshot=snapshot,
            daily_turnover_krw=daily_turnover,
            daily_order_count=daily_orders,
            last_trade_at=last_trade_at,
            now=now,
        )
        self.repo.write_order_intent(intent, decision)
        tenant = self._tenant_label()

        if not decision.allowed:
            logger.warning(
                "[yellow]ORDER REJECTED[/yellow] tenant=%s intent=%s agent=%s ticker=%s hits=%s",
                tenant,
                intent.intent_id,
                intent.agent_id,
                intent.ticker,
                ",".join(decision.policy_hits),
                extra={
                    "event": "order_rejected",
                    "tenant_id": tenant,
                    "intent_id": intent.intent_id,
                    "agent_id": intent.agent_id,
                    "ticker": intent.ticker,
                    "side": intent.side.value,
                    "notional_krw": intent.notional_krw,
                    "policy_hits": list(decision.policy_hits),
                },
            )
            report = ExecutionReport(
                status=ExecutionStatus.REJECTED,
                order_id=f"reject_{uuid4().hex[:10]}",
                filled_qty=0.0,
                avg_price_krw=0.0,
                message=decision.reason,
                created_at=now,
            )
            self.repo.write_execution_report(intent, report)
            self.memory_store.record_execution(intent=intent, decision=decision, report=report)
            record_thesis_lifecycle = getattr(self.memory_store, "record_thesis_lifecycle", None)
            if callable(record_thesis_lifecycle):
                record_thesis_lifecycle(
                    intent=intent,
                    decision=decision,
                    report=report,
                    snapshot_before=snapshot,
                )
            return report

        report = self.broker.place_order(
            intent,
            fx_rate=snapshot.usd_krw_rate if snapshot.usd_krw_rate > 0 else None,
        )
        self.repo.write_execution_report(intent, report)
        self.memory_store.record_execution(intent=intent, decision=decision, report=report)
        record_thesis_lifecycle = getattr(self.memory_store, "record_thesis_lifecycle", None)
        if callable(record_thesis_lifecycle):
            record_thesis_lifecycle(
                intent=intent,
                decision=decision,
                report=report,
                snapshot_before=snapshot,
            )

        if report.status in {ExecutionStatus.FILLED, ExecutionStatus.SIMULATED}:
            logger.info(
                "[green]ORDER EXECUTED[/green] tenant=%s intent=%s order=%s status=%s",
                tenant,
                intent.intent_id,
                report.order_id,
                report.status.value,
                extra={
                    "event": "order_executed",
                    "tenant_id": tenant,
                    "intent_id": intent.intent_id,
                    "agent_id": intent.agent_id,
                    "ticker": intent.ticker,
                    "side": intent.side.value,
                    "order_id": report.order_id,
                    "status": report.status.value,
                    "filled_qty": report.filled_qty,
                    "avg_price_krw": report.avg_price_krw,
                },
            )
        elif report.status == ExecutionStatus.SUBMITTED:
            logger.info(
                "[cyan]ORDER SUBMITTED[/cyan] tenant=%s intent=%s order=%s",
                tenant,
                intent.intent_id,
                report.order_id,
                extra={
                    "event": "order_submitted",
                    "tenant_id": tenant,
                    "intent_id": intent.intent_id,
                    "agent_id": intent.agent_id,
                    "ticker": intent.ticker,
                    "side": intent.side.value,
                    "order_id": report.order_id,
                    "status": report.status.value,
                },
            )
        else:
            logger.error(
                "[red]ORDER ERROR[/red] tenant=%s intent=%s order=%s msg=%s",
                tenant,
                intent.intent_id,
                report.order_id,
                report.message,
                extra={
                    "event": "order_error",
                    "tenant_id": tenant,
                    "intent_id": intent.intent_id,
                    "agent_id": intent.agent_id,
                    "ticker": intent.ticker,
                    "side": intent.side.value,
                    "order_id": report.order_id,
                    "status": report.status.value,
                    "error_message": report.message,
                },
            )

        return report

    def reconcile_submitted_orders(self, *, limit: int = 200, lookback_hours: int = 336) -> int:
        """Re-checks recent SUBMITTED orders and upgrades status when fills are confirmed."""
        reconcile = getattr(self.broker, "reconcile_submitted", None)
        if not callable(reconcile):
            return 0

        reconcile_trading_mode = str(
            getattr(getattr(self.risk_engine, "settings", None), "trading_mode", "")
            or "paper"
        ).strip() or "paper"
        rows = self.repo.recent_submitted_reports(
            limit=limit,
            lookback_hours=lookback_hours,
            trading_mode=reconcile_trading_mode,
        )
        if not rows:
            logger.debug("[reconcile] no SUBMITTED orders within %dh", lookback_hours)
            return 0

        logger.info("[cyan]Reconcile scan[/cyan] submitted=%d lookback_hours=%d", len(rows), lookback_hours)

        snapshot_fx: float | None = None
        latest_snapshot = getattr(self.repo, "latest_account_snapshot", None)
        if callable(latest_snapshot):
            try:
                snapshot = latest_snapshot()
            except Exception:
                snapshot = None
            if snapshot is not None and float(getattr(snapshot, "usd_krw_rate", 0.0) or 0.0) > 0:
                snapshot_fx = float(snapshot.usd_krw_rate)

        updated = 0
        for row in rows:
            try:
                report = reconcile(
                    order_id=str(row.get("order_id") or ""),
                    ticker=str(row.get("ticker") or ""),
                    exchange_code=str(row.get("exchange_code") or ""),
                    side=str(row.get("side") or ""),
                    requested_qty=float(row.get("requested_qty") or 0.0),
                    fallback_price_krw=float(row.get("avg_price_krw") or 0.0),
                    fx_rate=float(row.get("fx_rate") or 0.0) or snapshot_fx,
                )
            except Exception as exc:
                logger.warning(
                    "[yellow]Submitted reconcile failed[/yellow] order=%s err=%s",
                    str(row.get("order_id") or ""),
                    str(exc),
                )
                continue

            if report is None:
                continue

            status = report.status.value
            if status not in {"FILLED", "SIMULATED", "REJECTED", "ERROR"}:
                continue

            req_qty = float(row.get("requested_qty") or 0.0)
            if req_qty <= 0:
                continue
            side = str(row.get("side") or "BUY").strip().upper()
            if side not in {"BUY", "SELL"}:
                side = "BUY"
            intent = OrderIntent(
                agent_id=str(row.get("agent_id") or ""),
                ticker=str(row.get("ticker") or ""),
                trading_mode=reconcile_trading_mode,
                exchange_code=str(row.get("exchange_code") or ""),
                instrument_id=str(row.get("instrument_id") or ""),
                side=side,
                quantity=req_qty,
                price_krw=max(float(row.get("avg_price_krw") or 0.0), 1.0),
                price_native=float(row.get("avg_price_native") or 0.0) or None,
                quote_currency=str(row.get("quote_currency") or "").strip().upper(),
                fx_rate=float(row.get("fx_rate") or 0.0),
                rationale="reconcile_submitted",
                strategy_refs=[],
                created_at=row.get("created_at") or utc_now(),
                intent_id=str(row.get("intent_id") or f"intent_{uuid4().hex[:12]}"),
            )
            report.order_id = str(row.get("order_id") or report.order_id)
            self.repo.write_execution_report(intent, report)
            if self.memory_store is not None:
                try:
                    self.memory_store.record_execution(
                        intent=intent,
                        decision=RiskDecision(allowed=True, reason="reconciled", policy_hits=[]),
                        report=report,
                    )
                    record_thesis_lifecycle = getattr(self.memory_store, "record_thesis_lifecycle", None)
                    if callable(record_thesis_lifecycle):
                        record_thesis_lifecycle(
                            intent=intent,
                            decision=RiskDecision(allowed=True, reason="reconciled", policy_hits=[]),
                            report=report,
                            snapshot_before=None,
                        )
                except Exception as exc:
                    logger.warning(
                        "[yellow]Reconcile memory sync failed[/yellow] order=%s err=%s",
                        report.order_id,
                        str(exc),
                    )
            updated += 1

        if updated > 0:
            logger.info(
                "[cyan]Submitted reconciled[/cyan] updated=%d scanned=%d",
                updated,
                len(rows),
            )
        return updated
