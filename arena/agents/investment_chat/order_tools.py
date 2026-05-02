from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from arena.agents.investment_chat.audit import append_chat_audit
from arena.agents.investment_chat.constants import AGENT_ID
from arena.agents.investment_chat.context import normalize_tenant
from arena.agents.investment_chat.drafts import approval_token, load_draft, save_draft
from arena.agents.investment_chat.locks import tenant_lock
from arena.agents.investment_chat.memory import build_execution_memory, record_chat_decision_memory
from arena.agents.investment_chat.scope import (
    chat_actor_email,
    chat_strategy_refs,
    normalize_order_scope,
    snapshot_for_order_scope,
)
from arena.agents.investment_chat.utils import model_dump, repo_metric, repo_tenant_scope, safe_float, utc_iso
from arena.broker.open_trading import KISOpenTradingBroker
from arena.broker.paper import KISHttpBroker, PaperBroker
from arena.config import Settings, merge_agent_risk_settings
from arena.execution.gateway import ExecutionGateway
from arena.models import ExecutionStatus, OrderIntent, Side
from arena.risk import RiskEngine
from arena.tools.registry import ToolEntry


def _parse_audit_detail(row: dict[str, Any]) -> dict[str, Any]:
    detail = row.get("detail")
    if isinstance(detail, dict):
        return detail
    raw = row.get("detail_json")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def build_order_tool_entries(*, repo: Any, settings: Settings, tenant_id: str) -> list[ToolEntry]:
    tenant = normalize_tenant(tenant_id)

    def _approval_status_row(token: str, draft: dict[str, Any]) -> dict[str, Any]:
        intent = draft.get("intent") if isinstance(draft.get("intent"), dict) else {}
        risk = draft.get("risk") if isinstance(draft.get("risk"), dict) else {}
        report = draft.get("execution_report") if isinstance(draft.get("execution_report"), dict) else {}
        message = str(
            draft.get("error")
            or draft.get("message")
            or report.get("message")
            or risk.get("reason")
            or ""
        ).strip()
        return {
            "approval_token": token,
            "status": str(draft.get("status") or "").strip().lower(),
            "created_at": draft.get("created_at") or "",
            "submitted_at": draft.get("submitted_at") or "",
            "expires_at": draft.get("expires_at") or "",
            "scope": draft.get("scope") or "account",
            "target_agent_id": draft.get("target_agent_id") or "",
            "ticker": intent.get("ticker") or "",
            "side": intent.get("side") or "",
            "quantity": intent.get("quantity") or 0,
            "notional_krw": draft.get("notional_krw") or 0,
            "risk_allowed": risk.get("allowed"),
            "policy_hits": risk.get("policy_hits") or [],
            "order_id": report.get("order_id") or "",
            "execution_status": report.get("status") or "",
            "message": message,
            "judgment_source": draft.get("judgment_source") or "user+investment_chat",
        }

    def get_order_approval_status(approval_token: str = "", limit: int = 5) -> dict[str, Any]:
        """Reads recent investment-chat approval card results, including broker errors."""
        explicit = str(approval_token or "").strip()
        tokens: list[str] = []
        if explicit:
            tokens = [explicit]
        else:
            loader = getattr(repo, "recent_runtime_audit_logs", None)
            if callable(loader):
                try:
                    audit_rows = loader(limit=max(20, min(200, int(limit or 5) * 20))) or []
                except TypeError:
                    audit_rows = loader(max(20, min(200, int(limit or 5) * 20))) or []
                except Exception:
                    audit_rows = []
                seen: set[str] = set()
                for row in audit_rows:
                    if not isinstance(row, dict):
                        continue
                    if str(row.get("tenant_id") or "").strip().lower() not in {"", tenant}:
                        continue
                    if str(row.get("action") or "").strip() != "chat_order_validate":
                        continue
                    detail = _parse_audit_detail(row)
                    token = str(detail.get("approval_token") or "").strip()
                    if token and token not in seen:
                        seen.add(token)
                        tokens.append(token)
                    if len(tokens) >= max(1, min(int(limit or 5), 20)):
                        break
        orders: list[dict[str, Any]] = []
        for token in tokens:
            draft = load_draft(repo, tenant_id=tenant, token=token)
            if isinstance(draft, dict):
                orders.append(_approval_status_row(token, draft))
        return {
            "status": "ok",
            "tenant_id": tenant,
            "count": len(orders),
            "orders": orders,
        }

    def validate_order_draft(
        ticker: str,
        side: str,
        quantity: float,
        price_krw: float,
        rationale: str,
        agent_id: str = AGENT_ID,
        scope: str = "account",
        exchange_code: str = "",
        instrument_id: str = "",
        price_native: float | None = None,
        quote_currency: str = "",
        fx_rate: float = 0.0,
    ) -> dict[str, Any]:
        """Validates a proposed order against Arena risk policy without submitting it."""
        ticker_token = str(ticker or "").strip().upper()
        side_token = str(side or "").strip().upper()
        if side_token not in {"BUY", "SELL"}:
            return {"status": "error", "error": "side must be BUY or SELL"}
        if not ticker_token:
            return {"status": "error", "error": "ticker is required"}
        if safe_float(quantity) <= 0 or safe_float(price_krw) <= 0:
            return {"status": "error", "error": "quantity and price_krw must be positive"}
        scope_token = normalize_order_scope(scope)
        if not scope_token:
            return {"status": "error", "error": "scope must be account or agent_sleeve"}
        target_agent = str(agent_id or AGENT_ID).strip().lower() or AGENT_ID
        if scope_token == "account":
            target_agent = AGENT_ID
        elif target_agent == AGENT_ID:
            return {
                "status": "error",
                "error": "agent_id must name the target batch agent when scope='agent_sleeve'",
            }
        snapshot, snapshot_meta = snapshot_for_order_scope(
            repo=repo,
            settings=settings,
            tenant_id=tenant,
            scope=scope_token,
            agent_id=target_agent,
        )
        if snapshot is None:
            return {
                "status": "missing_account_snapshot",
                "tenant_id": tenant,
                "scope": scope_token,
                "agent_id": target_agent,
                "error": (
                    "No account snapshot is stored; order validation needs current cash and holdings."
                    if scope_token == "account"
                    else "No agent sleeve snapshot is available for this target agent."
                ),
                "metadata": snapshot_meta,
            }
        user_email = chat_actor_email()
        strategy_refs = chat_strategy_refs(scope=scope_token, agent_id=target_agent, user_email=user_email)

        intent = OrderIntent(
            agent_id=target_agent,
            ticker=ticker_token,
            trading_mode=str(getattr(settings, "trading_mode", "") or "paper").strip().lower() or "paper",
            exchange_code=str(exchange_code or "").strip().upper(),
            instrument_id=str(instrument_id or "").strip(),
            side=Side(side_token),
            quantity=float(quantity),
            price_krw=float(price_krw),
            price_native=price_native,
            quote_currency=str(quote_currency or "").strip().upper(),
            fx_rate=float(fx_rate or 0.0),
            rationale=str(rationale or "").strip() or "chat_order_draft",
            strategy_refs=strategy_refs,
        )
        include_simulated = settings.trading_mode.strip().lower() != "live"
        now = datetime.now(timezone.utc)
        agent_config = (settings.agent_configs or {}).get(intent.agent_id)
        risk_settings = (
            merge_agent_risk_settings(settings, agent_config)
            if agent_config and agent_config.risk_overrides
            else settings
        )
        decision = RiskEngine(settings=risk_settings).evaluate(
            intent=intent,
            snapshot=snapshot,
            daily_turnover_krw=float(
                repo_metric(
                    repo,
                    "recent_turnover_krw",
                    0.0,
                    tenant_id=tenant,
                    day=now.date(),
                    agent_id=intent.agent_id,
                    include_simulated=include_simulated,
                    trading_mode=intent.trading_mode,
                )
                or 0.0
            ),
            daily_order_count=int(
                repo_metric(
                    repo,
                    "recent_intent_count",
                    0,
                    tenant_id=tenant,
                    day=now.date(),
                    agent_id=intent.agent_id,
                    include_simulated=include_simulated,
                    trading_mode=intent.trading_mode,
                )
                or 0
            ),
            last_trade_at=repo_metric(
                repo,
                "last_trade_time",
                None,
                tenant_id=tenant,
                ticker=intent.ticker,
                agent_id=intent.agent_id,
                exchange_code=intent.exchange_code or None,
                instrument_id=intent.instrument_id or None,
                include_simulated=include_simulated,
                trading_mode=intent.trading_mode,
            ),
            now=now,
        )
        fingerprint_payload = {
            "tenant_id": tenant,
            "agent_id": intent.agent_id,
            "ticker": intent.ticker,
            "trading_mode": intent.trading_mode,
            "exchange_code": intent.exchange_code,
            "instrument_id": intent.instrument_id,
            "side": intent.side.value,
            "quantity": float(intent.quantity),
            "price_krw": float(intent.price_krw),
            "price_native": intent.price_native,
            "quote_currency": intent.quote_currency,
            "fx_rate": intent.fx_rate,
            "rationale": intent.rationale,
            "strategy_refs": list(intent.strategy_refs or []),
            "scope": scope_token,
        }
        order_fingerprint = approval_token(fingerprint_payload)
        token = approval_token({**fingerprint_payload, "draft_nonce": uuid4().hex})
        intent.intent_id = f"chat_{token}"
        expires_at = now + timedelta(minutes=max(1, int(os.getenv("ARENA_CHAT_ORDER_DRAFT_TTL_MINUTES", "15") or "15")))
        draft = {
            "approval_token": token,
            "status": "draft" if decision.allowed else "risk_rejected",
            "created_at": utc_iso(now),
            "expires_at": utc_iso(expires_at),
            "intent": model_dump(intent),
            "risk": model_dump(decision),
            "notional_krw": intent.notional_krw,
            "order_fingerprint": order_fingerprint,
            "required_confirmation": f"CONFIRM {token}",
            "scope": scope_token,
            "target_agent_id": intent.agent_id,
            "judgment_source": "user+investment_chat",
            "approved_by": user_email,
            "snapshot_meta": snapshot_meta,
        }
        save_draft(repo, tenant_id=tenant, token=token, draft=draft)
        append_chat_audit(
            repo,
            tenant_id=tenant,
            action="chat_order_validate",
            status="ok" if decision.allowed else "blocked",
            detail={
                "approval_token": token,
                "ticker": intent.ticker,
                "side": intent.side.value,
                "quantity": intent.quantity,
                "notional_krw": intent.notional_krw,
                "risk_allowed": decision.allowed,
                "policy_hits": list(decision.policy_hits or []),
                "scope": scope_token,
                "target_agent_id": intent.agent_id,
                "judgment_source": "user+investment_chat",
            },
            user_email=user_email,
        )
        return {
            "status": "ok",
            "tenant_id": tenant,
            "scope": scope_token,
            "target_agent_id": intent.agent_id,
            "judgment_source": "user+investment_chat",
            "approved_by": user_email,
            "approval_token": token,
            "intent": model_dump(intent),
            "risk": model_dump(decision),
            "notional_krw": intent.notional_krw,
            "submission_status": "not_submitted",
            "approval_required": True,
            "required_confirmation": f"CONFIRM {token}",
            "expires_at": utc_iso(expires_at),
            "snapshot_meta": snapshot_meta,
        }

    def submit_approved_order(approval_token: str, confirmation_text: str) -> dict[str, Any]:
        """Submits a previously validated order draft from a backend/UI approval bridge."""
        token = str(approval_token or "").strip()
        required = f"CONFIRM {token}"
        if not token:
            return {"status": "blocked", "error": "approval_token is required", "required_confirmation": required}
        if str(confirmation_text or "").strip() != required:
            return {
                "status": "blocked",
                "error": "confirmation_text must exactly match the required confirmation phrase",
                "required_confirmation": required,
            }
        draft = load_draft(repo, tenant_id=tenant, token=token)
        if not draft:
            return {"status": "missing", "error": "approval token not found or expired", "approval_token": token}
        if str(draft.get("status") or "").strip().lower() == "submitted":
            return {
                "status": "already_submitted",
                "approval_token": token,
                "execution_report": draft.get("execution_report") or {},
                "intent": draft.get("intent") or {},
            }
        if str(draft.get("status") or "").strip().lower() != "draft":
            return {
                "status": "blocked",
                "approval_token": token,
                "error": f"draft is not submittable: status={draft.get('status')}",
                "risk": draft.get("risk") or {},
            }
        try:
            expires_at = datetime.fromisoformat(str(draft.get("expires_at") or "").replace("Z", "+00:00"))
        except ValueError:
            expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        if expires_at < datetime.now(timezone.utc):
            draft["status"] = "expired"
            save_draft(repo, tenant_id=tenant, token=token, draft=draft)
            return {"status": "expired", "approval_token": token, "error": "approval token expired"}

        intent = OrderIntent.model_validate(draft.get("intent") or {})
        scope_token = normalize_order_scope(str(draft.get("scope") or "account"))
        if not scope_token:
            return {"status": "blocked", "approval_token": token, "error": "draft has invalid order scope"}
        user_email = str(draft.get("approved_by") or chat_actor_email() or "").strip().lower()
        broker = PaperBroker()
        if str(getattr(settings, "trading_mode", "") or "").strip().lower() == "live":
            if not bool(getattr(settings, "allow_live_trading", False)):
                return {"status": "blocked", "approval_token": token, "error": "live trading is not enabled for this runtime"}
            broker = KISHttpBroker(settings=settings) if getattr(settings, "kis_order_endpoint", "") else KISOpenTradingBroker(settings=settings)
        try:
            memory_store = build_execution_memory(repo, settings)
        except Exception as exc:
            append_chat_audit(
                repo,
                tenant_id=tenant,
                action="chat_order_submit",
                status="error",
                detail={
                    "approval_token": token,
                    "intent_id": intent.intent_id,
                    "scope": scope_token,
                    "target_agent_id": intent.agent_id,
                    "stage": "memory_store_init",
                    "error": str(exc)[:500],
                },
                user_email=user_email,
            )
            return {
                "status": "blocked",
                "approval_token": token,
                "error": "execution memory store is unavailable; order was not submitted",
            }
        gateway = ExecutionGateway(
            repo=repo,
            risk_engine=RiskEngine(settings=settings),
            broker=broker,
            memory_store=memory_store,
            agent_configs=getattr(settings, "agent_configs", {}) or {},
        )
        snapshot, snapshot_meta = snapshot_for_order_scope(
            repo=repo,
            settings=settings,
            tenant_id=tenant,
            scope=scope_token,
            agent_id=intent.agent_id,
        )
        if snapshot is None:
            return {
                "status": "missing_account_snapshot",
                "approval_token": token,
                "scope": scope_token,
                "target_agent_id": intent.agent_id,
                "error": "No account snapshot is stored" if scope_token == "account" else "No agent sleeve snapshot is available",
                "metadata": snapshot_meta,
            }
        write_lock = tenant_lock(tenant)
        with write_lock, repo_tenant_scope(repo, tenant):
            report = gateway.process(intent, snapshot)
        submitted_statuses = {ExecutionStatus.FILLED, ExecutionStatus.SIMULATED, ExecutionStatus.SUBMITTED}
        response_status = "submitted" if report.status in submitted_statuses else report.status.value.lower()
        audit_status = "ok" if response_status == "submitted" else "error"
        memory_warnings: list[str] = []
        if report.status in submitted_statuses:
            try:
                with write_lock, repo_tenant_scope(repo, tenant):
                    warning = record_chat_decision_memory(
                        memory_store=memory_store,
                        intent=intent,
                        report=report,
                        approval_token=token,
                        tenant_id=tenant,
                        scope=scope_token,
                        user_email=user_email,
                    )
                if warning:
                    memory_warnings.append(warning)
            except Exception as exc:
                memory_warnings.append("record_chat_decision")
                append_chat_audit(
                    repo,
                    tenant_id=tenant,
                    action="chat_order_memory_sync",
                    status="warning",
                    detail={
                        "approval_token": token,
                        "intent_id": intent.intent_id,
                        "scope": scope_token,
                        "target_agent_id": intent.agent_id,
                        "stage": "record_chat_decision",
                        "error": str(exc)[:500],
                    },
                    user_email=user_email,
                )
        draft["status"] = response_status
        draft["submitted_at"] = utc_iso()
        draft["execution_report"] = model_dump(report)
        draft["memory_warnings"] = memory_warnings
        save_draft(repo, tenant_id=tenant, token=token, draft=draft)
        append_chat_audit(
            repo,
            tenant_id=tenant,
            action="chat_order_submit",
            status=audit_status,
            detail={
                "approval_token": token,
                "intent_id": intent.intent_id,
                "ticker": intent.ticker,
                "side": intent.side.value,
                "quantity": intent.quantity,
                "report_status": report.status.value,
                "order_id": report.order_id,
                "scope": scope_token,
                "target_agent_id": intent.agent_id,
                "judgment_source": "user+investment_chat",
                "memory_warnings": memory_warnings,
            },
            user_email=user_email,
        )
        return {
            "status": response_status,
            "approval_token": token,
            "scope": scope_token,
            "target_agent_id": intent.agent_id,
            "judgment_source": "user+investment_chat",
            "intent": model_dump(intent),
            "execution_report": model_dump(report),
            "memory_warnings": memory_warnings,
            "message": str(report.message or ""),
            **({"error": str(report.message or "")} if response_status != "submitted" else {}),
        }

    return [
        ToolEntry(
            tool_id="validate_order_draft",
            name="validate_order_draft",
            description="Builds and risk-checks a proposed order draft without submitting it. Always returns submission_status='not_submitted'.",
            category="execution",
            callable=validate_order_draft,
            tier="core",
            label_ko="주문 초안 검증",
            sort_order=4,
        ),
        ToolEntry(
            tool_id="get_order_approval_status",
            name="get_order_approval_status",
            description="Reads recent investment-chat approval card/button results, including broker rejection messages. Use when the user asks what happened after approval.",
            category="execution",
            callable=get_order_approval_status,
            tier="core",
            label_ko="승인 결과 조회",
            sort_order=5,
        ),
        ToolEntry(
            tool_id="submit_approved_order",
            name="submit_approved_order",
            description="Internal backend/UI approval bridge for submitting a stored order draft. Do not ask the user to type CONFIRM in chat; the web approval card supplies confirmation.",
            category="execution",
            callable=submit_approved_order,
            tier="core",
            label_ko="승인 주문 제출",
            sort_order=6,
        ),
    ]
