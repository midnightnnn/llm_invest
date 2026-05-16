from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional
from uuid import uuid4

from google.adk.tools.tool_context import ToolContext

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
from arena.logging_utils import failure_extra
from arena.models import ExecutionStatus, OrderIntent, Side
from arena.risk import RiskEngine
from arena.tools.registry import ToolEntry

logger = logging.getLogger(__name__)

OrderSide = Literal["BUY", "SELL"]
OrderScope = Literal["account", "agent_sleeve"]
MAX_BATCH_ORDERS = 20


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


def _confirmation_state_key(tool_context: ToolContext) -> str:
    function_call_id = str(getattr(tool_context, "function_call_id", "") or "").strip()
    return f"investment_chat.order_confirmation.{function_call_id or 'unknown'}"


def _batch_confirmation_state_key(tool_context: ToolContext) -> str:
    function_call_id = str(getattr(tool_context, "function_call_id", "") or "").strip()
    return f"investment_chat.order_batch_confirmation.{function_call_id or 'unknown'}"


def _confirmation_payload(draft: dict[str, Any]) -> dict[str, Any]:
    intent = draft.get("intent") if isinstance(draft.get("intent"), dict) else {}
    risk = draft.get("risk") if isinstance(draft.get("risk"), dict) else {}
    return {
        "action": "submit_order",
        "ticker": intent.get("ticker") or "",
        "side": intent.get("side") or "",
        "quantity": intent.get("quantity") or 0,
        "notional_krw": draft.get("notional_krw") or 0,
        "scope": draft.get("scope") or "account",
        "target_agent_id": draft.get("target_agent_id") or "",
        "risk_allowed": risk.get("allowed"),
        "policy_hits": risk.get("policy_hits") or [],
        "judgment_source": draft.get("judgment_source") or "user+investment_chat",
    }


def _batch_order_summary(token: str, draft: dict[str, Any]) -> dict[str, Any]:
    payload = _confirmation_payload(draft)
    payload["approval_token"] = token
    payload["status"] = str(draft.get("status") or "").strip().lower()
    payload["submittable"] = payload["status"] == "draft" and bool(payload.get("risk_allowed"))
    payload["batch_index"] = draft.get("batch_index")
    return payload


def _batch_confirmation_payload(batch_token: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    submittable = [row for row in rows if row.get("submittable")]
    return {
        "action": "submit_order_batch",
        "batch_token": batch_token,
        "order_count": len(rows),
        "submittable_count": len(submittable),
        "blocked_count": len(rows) - len(submittable),
        "notional_krw": sum(float(row.get("notional_krw") or 0.0) for row in submittable),
        "orders": [
            {
                key: row.get(key)
                for key in (
                    "ticker",
                    "side",
                    "quantity",
                    "notional_krw",
                    "scope",
                    "target_agent_id",
                    "risk_allowed",
                    "policy_hits",
                    "submittable",
                )
            }
            for row in rows
        ],
        "judgment_source": "user+investment_chat",
    }


def _confirmation_hint(draft: dict[str, Any]) -> str:
    payload = _confirmation_payload(draft)
    ticker = str(payload.get("ticker") or "").strip().upper()
    side = str(payload.get("side") or "").strip().upper()
    side_label = "매수" if side == "BUY" else "매도" if side == "SELL" else side
    quantity = payload.get("quantity") or 0
    notional = payload.get("notional_krw") or 0
    scope = str(payload.get("scope") or "account")
    scope_label = "전체 계좌" if scope == "account" else "에이전트 sleeve"
    return (
        f"{scope_label} 주문을 제출할까요? {ticker} {side_label} {float(quantity):g}주, "
        f"예상 금액 {float(notional):,.0f}원. "
        "ADK Web 확인창에서 Confirmed 체크박스를 체크한 뒤 Submit을 눌러야 승인됩니다."
    )


def _batch_confirmation_hint(rows: list[dict[str, Any]]) -> str:
    submittable = [row for row in rows if row.get("submittable")]
    blocked_count = max(0, len(rows) - len(submittable))
    total_notional = sum(float(row.get("notional_krw") or 0.0) for row in submittable)
    preview = ", ".join(
        f"{str(row.get('ticker') or '').upper()} {str(row.get('side') or '').upper()} {float(row.get('quantity') or 0):g}주"
        for row in submittable[:5]
    )
    suffix = f" 외 {len(submittable) - 5}건" if len(submittable) > 5 else ""
    blocked_note = f" 리스크/입력 문제로 {blocked_count}건은 제출되지 않습니다." if blocked_count else ""
    return (
        f"총 {len(submittable)}건의 주문을 일괄 제출할까요? {preview}{suffix}, "
        f"예상 합계 {total_notional:,.0f}원.{blocked_note} "
        "ADK Web 확인창에서 Confirmed 체크박스를 체크한 뒤 Submit을 눌러야 승인됩니다."
    )


def _batch_confirmation_token(tokens: list[str]) -> str:
    return approval_token({"approval_tokens": tokens})


def _batch_required_confirmation(tokens: list[str]) -> str:
    return f"CONFIRM_BATCH {_batch_confirmation_token(tokens)}"


def _batch_submit_status(results: list[dict[str, Any]]) -> str:
    if not results:
        return "blocked"
    submitted_statuses = {"submitted", "already_submitted"}
    submitted_count = sum(
        1 for result in results if str(result.get("status") or "").strip().lower() in submitted_statuses
    )
    if submitted_count == len(results):
        return "submitted"
    if submitted_count > 0:
        return "partial"
    first_status = str(results[0].get("status") or "").strip().lower()
    return first_status or "blocked"


def _build_order_tool_entries(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    include_internal_bridge: bool,
) -> list[ToolEntry]:
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

    def _build_batch_drafts(
        orders: list[dict[str, Any]],
        *,
        approval_channel: str = "",
    ) -> dict[str, Any]:
        if not isinstance(orders, list) or not orders:
            return {"status": "error", "error": "orders must be a non-empty list"}
        if len(orders) > MAX_BATCH_ORDERS:
            return {
                "status": "error",
                "error": f"orders cannot contain more than {MAX_BATCH_ORDERS} items",
                "max_orders": MAX_BATCH_ORDERS,
            }

        results: list[dict[str, Any]] = []
        tokens: list[str] = []
        for index, raw_order in enumerate(orders):
            if not isinstance(raw_order, dict):
                results.append(
                    {
                        "status": "error",
                        "batch_index": index,
                        "error": "each order must be an object",
                    }
                )
                continue
            price_native_raw = raw_order.get("price_native")
            price_native = (
                None
                if price_native_raw is None or str(price_native_raw).strip() == ""
                else safe_float(price_native_raw)
            )
            result = validate_order_draft(
                ticker=str(raw_order.get("ticker") or ""),
                side=str(raw_order.get("side") or "").upper(),  # type: ignore[arg-type]
                quantity=safe_float(raw_order.get("quantity")),
                price_krw=safe_float(raw_order.get("price_krw")),
                rationale=str(raw_order.get("rationale") or ""),
                agent_id=str(raw_order.get("agent_id") or AGENT_ID),
                scope=str(raw_order.get("scope") or "account"),  # type: ignore[arg-type]
                exchange_code=str(raw_order.get("exchange_code") or ""),
                instrument_id=str(raw_order.get("instrument_id") or ""),
                price_native=price_native,
                quote_currency=str(raw_order.get("quote_currency") or ""),
                fx_rate=safe_float(raw_order.get("fx_rate")),
            )
            result["batch_index"] = index
            token = str(result.get("approval_token") or "").strip()
            if token:
                tokens.append(token)
            results.append(result)

        batch_token = approval_token(
            {
                "tenant_id": tenant,
                "approval_tokens": tokens,
                "batch_nonce": uuid4().hex,
            }
        )
        rows: list[dict[str, Any]] = []
        for result in results:
            token = str(result.get("approval_token") or "").strip()
            if not token:
                rows.append(
                    {
                        "approval_token": "",
                        "status": str(result.get("status") or "error").strip().lower(),
                        "submittable": False,
                        "batch_index": result.get("batch_index"),
                        "error": result.get("error") or "",
                    }
                )
                continue
            draft = load_draft(repo, tenant_id=tenant, token=token)
            if isinstance(draft, dict):
                draft["batch_token"] = batch_token
                draft["batch_index"] = result.get("batch_index")
                draft["batch_count"] = len(results)
                if approval_channel:
                    draft["approval_channel"] = approval_channel
                save_draft(repo, tenant_id=tenant, token=token, draft=draft)
                rows.append(_batch_order_summary(token, draft))
            else:
                rows.append(
                    {
                        "approval_token": token,
                        "status": "missing",
                        "submittable": False,
                        "batch_index": result.get("batch_index"),
                    }
                )
        submittable_tokens = [str(row.get("approval_token") or "") for row in rows if row.get("submittable")]
        return {
            "status": "ok",
            "tenant_id": tenant,
            "batch_token": batch_token,
            "order_count": len(results),
            "submittable_count": len(submittable_tokens),
            "blocked_count": len(rows) - len(submittable_tokens),
            "approval_tokens": [token for token in tokens if token],
            "submittable_approval_tokens": submittable_tokens,
            "notional_krw": sum(float(row.get("notional_krw") or 0.0) for row in rows if row.get("submittable")),
            "orders": rows,
            "submission_status": "not_submitted",
            "approval_required": bool(submittable_tokens),
        }

    def validate_order_draft(
        ticker: str,
        side: OrderSide,
        quantity: float,
        price_krw: float,
        rationale: str,
        agent_id: str = AGENT_ID,
        scope: OrderScope = "account",
        exchange_code: str = "",
        instrument_id: str = "",
        price_native: Optional[float] = None,
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
        rationale_text = str(rationale or "").strip()
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
            rationale=rationale_text or "chat_order_draft",
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

    def validate_order_batch_draft(orders: list[dict[str, Any]]) -> dict[str, Any]:
        """Validates multiple proposed order drafts without submitting them."""
        return _build_batch_drafts(orders)

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
            logger.warning(
                "[yellow]Investment chat order memory store init failed[/yellow] tenant=%s intent=%s err=%s",
                tenant,
                intent.intent_id,
                str(exc),
                extra=failure_extra(
                    "chat_order_memory_store_init_failed",
                    exc,
                    tenant_id=tenant,
                    intent_id=intent.intent_id,
                    scope=scope_token,
                    target_agent_id=intent.agent_id,
                ),
                exc_info=True,
            )
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
                logger.warning(
                    "[yellow]Investment chat order memory sync failed[/yellow] tenant=%s intent=%s order=%s err=%s",
                    tenant,
                    intent.intent_id,
                    report.order_id,
                    str(exc),
                    extra=failure_extra(
                        "chat_order_memory_sync_failed",
                        exc,
                        tenant_id=tenant,
                        intent_id=intent.intent_id,
                        order_id=report.order_id,
                        scope=scope_token,
                        target_agent_id=intent.agent_id,
                    ),
                    exc_info=True,
                )
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

    def submit_approved_order_batch(approval_tokens: list[str], confirmation_text: str) -> dict[str, Any]:
        """Submits multiple previously validated order drafts from a backend/UI approval bridge."""
        tokens: list[str] = []
        seen: set[str] = set()
        for raw_token in approval_tokens or []:
            token = str(raw_token or "").strip()
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
        required = _batch_required_confirmation(tokens)
        if not tokens:
            return {"status": "blocked", "error": "approval_tokens is required", "required_confirmation": required}
        if str(confirmation_text or "").strip() != required:
            return {
                "status": "blocked",
                "error": "confirmation_text must exactly match the required batch confirmation phrase",
                "required_confirmation": required,
            }

        results = [
            submit_approved_order(approval_token=token, confirmation_text=f"CONFIRM {token}")
            for token in tokens
        ]
        submitted_statuses = {"submitted", "already_submitted"}
        submitted_count = sum(
            1 for result in results if str(result.get("status") or "").strip().lower() in submitted_statuses
        )
        status = _batch_submit_status(results)
        user_email = chat_actor_email()
        append_chat_audit(
            repo,
            tenant_id=tenant,
            action="chat_order_batch_submit",
            status="ok" if status in {"submitted", "partial"} else "error",
            detail={
                "approval_tokens": tokens,
                "submitted_count": submitted_count,
                "failed_count": len(results) - submitted_count,
                "result_status": status,
            },
            user_email=user_email,
        )
        return {
            "status": status,
            "tenant_id": tenant,
            "order_count": len(tokens),
            "submitted_count": submitted_count,
            "failed_count": len(results) - submitted_count,
            "orders": results,
        }

    def submit_order_with_confirmation(
        ticker: str,
        side: OrderSide,
        quantity: float,
        price_krw: float,
        rationale: str,
        tool_context: ToolContext,
        agent_id: str = AGENT_ID,
        scope: OrderScope = "account",
        exchange_code: str = "",
        instrument_id: str = "",
        price_native: Optional[float] = None,
        quote_currency: str = "",
        fx_rate: float = 0.0,
    ) -> dict[str, Any]:
        """Validates an order and uses ADK tool confirmation before submitting it."""
        state_key = _confirmation_state_key(tool_context)
        confirmation = getattr(tool_context, "tool_confirmation", None)
        state = getattr(tool_context, "state", None)
        if confirmation is not None:
            token = ""
            if state is not None:
                token = str(state.get(state_key) or "").strip()
            if not bool(getattr(confirmation, "confirmed", False)):
                confirmation_payload = getattr(confirmation, "payload", None)
                reason = "confirmed_checkbox_unchecked" if confirmation_payload is not None else "not_confirmed"
                if token:
                    draft = load_draft(repo, tenant_id=tenant, token=token)
                    if isinstance(draft, dict) and str(draft.get("status") or "").strip().lower() == "draft":
                        draft["status"] = "rejected"
                        draft["rejected_at"] = utc_iso()
                        draft["rejection_reason"] = reason
                        save_draft(repo, tenant_id=tenant, token=token, draft=draft)
                    append_chat_audit(
                        repo,
                        tenant_id=tenant,
                        action="chat_order_submit",
                        status="blocked",
                        detail={
                            "approval_token": token,
                            "stage": "adk_tool_confirmation",
                            "reason": reason,
                        },
                        user_email=chat_actor_email(),
                    )
                return {
                    "status": "rejected",
                    "tenant_id": tenant,
                    "reason": reason,
                    "submission_status": "not_submitted",
                    "message": (
                        "ADK Web에서 confirmed=false가 반환되어 주문을 제출하지 않았습니다. "
                        "승인하려면 ADK Web 확인창에서 Confirmed 체크박스를 체크한 뒤 Submit을 눌러야 합니다."
                        if reason == "confirmed_checkbox_unchecked"
                        else "ADK Web에서 주문이 확인되지 않아 제출하지 않았습니다."
                    ),
                }
            if not token:
                return {
                    "status": "blocked",
                    "tenant_id": tenant,
                    "error": "ADK tool confirmation state was missing; order was not submitted.",
                }
            return submit_approved_order(approval_token=token, confirmation_text=f"CONFIRM {token}")

        draft_result = validate_order_draft(
            ticker=ticker,
            side=side,
            quantity=quantity,
            price_krw=price_krw,
            rationale=rationale,
            agent_id=agent_id,
            scope=scope,
            exchange_code=exchange_code,
            instrument_id=instrument_id,
            price_native=price_native,
            quote_currency=quote_currency,
            fx_rate=fx_rate,
        )
        if str(draft_result.get("status") or "").strip().lower() != "ok":
            return draft_result
        risk = draft_result.get("risk") if isinstance(draft_result.get("risk"), dict) else {}
        if not bool(risk.get("allowed")):
            return draft_result
        token = str(draft_result.get("approval_token") or "").strip()
        draft = load_draft(repo, tenant_id=tenant, token=token)
        if not isinstance(draft, dict):
            return {
                "status": "blocked",
                "tenant_id": tenant,
                "error": "validated order draft could not be loaded; order was not submitted.",
            }
        draft["approval_channel"] = "adk_tool_confirmation"
        save_draft(repo, tenant_id=tenant, token=token, draft=draft)
        if state is not None:
            state[state_key] = token
        tool_context.request_confirmation(
            hint=_confirmation_hint(draft),
            payload=_confirmation_payload(draft),
        )
        return {
            "status": "waiting_for_confirmation",
            "tenant_id": tenant,
            "scope": draft.get("scope") or "account",
            "target_agent_id": draft.get("target_agent_id") or "",
            "judgment_source": draft.get("judgment_source") or "user+investment_chat",
            "intent": draft.get("intent") or {},
            "risk": draft.get("risk") or {},
            "notional_krw": draft.get("notional_krw") or 0,
            "submission_status": "not_submitted",
            "approval_required": True,
            "approval_ui": "adk_tool_confirmation",
            "message": "ADK tool confirmation is required before this order can be submitted.",
        }

    def submit_order_batch_with_confirmation(
        orders: list[dict[str, Any]],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Validates multiple orders, asks for one ADK confirmation, then submits approved drafts."""
        state_key = _batch_confirmation_state_key(tool_context)
        confirmation = getattr(tool_context, "tool_confirmation", None)
        state = getattr(tool_context, "state", None)
        if confirmation is not None:
            stored = state.get(state_key) if state is not None else None
            approval_tokens: list[str] = []
            if isinstance(stored, dict):
                raw_tokens = stored.get("approval_tokens")
                if isinstance(raw_tokens, list):
                    approval_tokens = [str(token or "").strip() for token in raw_tokens if str(token or "").strip()]
            elif isinstance(stored, list):
                approval_tokens = [str(token or "").strip() for token in stored if str(token or "").strip()]

            if not bool(getattr(confirmation, "confirmed", False)):
                confirmation_payload = getattr(confirmation, "payload", None)
                reason = "confirmed_checkbox_unchecked" if confirmation_payload is not None else "not_confirmed"
                for token in approval_tokens:
                    draft = load_draft(repo, tenant_id=tenant, token=token)
                    if isinstance(draft, dict) and str(draft.get("status") or "").strip().lower() == "draft":
                        draft["status"] = "rejected"
                        draft["rejected_at"] = utc_iso()
                        draft["rejection_reason"] = reason
                        save_draft(repo, tenant_id=tenant, token=token, draft=draft)
                append_chat_audit(
                    repo,
                    tenant_id=tenant,
                    action="chat_order_batch_submit",
                    status="blocked",
                    detail={
                        "approval_tokens": approval_tokens,
                        "stage": "adk_tool_confirmation",
                        "reason": reason,
                    },
                    user_email=chat_actor_email(),
                )
                return {
                    "status": "rejected",
                    "tenant_id": tenant,
                    "order_count": len(approval_tokens),
                    "reason": reason,
                    "submission_status": "not_submitted",
                    "message": (
                        "ADK Web에서 confirmed=false가 반환되어 주문 묶음을 제출하지 않았습니다. "
                        "승인하려면 ADK Web 확인창에서 Confirmed 체크박스를 체크한 뒤 Submit을 눌러야 합니다."
                        if reason == "confirmed_checkbox_unchecked"
                        else "ADK Web에서 주문 묶음이 확인되지 않아 제출하지 않았습니다."
                    ),
                }
            if not approval_tokens:
                return {
                    "status": "blocked",
                    "tenant_id": tenant,
                    "error": "ADK tool confirmation state was missing; batch orders were not submitted.",
                }
            return submit_approved_order_batch(
                approval_tokens=approval_tokens,
                confirmation_text=_batch_required_confirmation(approval_tokens),
            )

        draft_batch = _build_batch_drafts(orders, approval_channel="adk_tool_confirmation")
        if str(draft_batch.get("status") or "").strip().lower() != "ok":
            return draft_batch
        submittable_tokens = [
            str(token or "").strip()
            for token in draft_batch.get("submittable_approval_tokens", [])
            if str(token or "").strip()
        ]
        rows = draft_batch.get("orders") if isinstance(draft_batch.get("orders"), list) else []
        if not submittable_tokens:
            return {
                **draft_batch,
                "status": "blocked",
                "error": "No orders passed validation for submission.",
                "approval_required": False,
            }
        if state is not None:
            state[state_key] = {
                "batch_token": draft_batch.get("batch_token") or "",
                "approval_tokens": submittable_tokens,
            }
        tool_context.request_confirmation(
            hint=_batch_confirmation_hint(rows),
            payload=_batch_confirmation_payload(str(draft_batch.get("batch_token") or ""), rows),
        )
        return {
            **draft_batch,
            "status": "waiting_for_confirmation",
            "approval_required": True,
            "approval_ui": "adk_tool_confirmation",
            "message": "One ADK tool confirmation is required before this order batch can be submitted.",
        }

    entries = [
        ToolEntry(
            tool_id="submit_order_batch_with_confirmation",
            name="submit_order_batch_with_confirmation",
            description=(
                "Preferred order submission tool when the user asks for two or more buy/sell orders in one turn. "
                "Pass every order in the orders array using keys ticker, side, quantity, price_krw, rationale, "
                "and optional agent_id, scope, exchange_code, instrument_id, price_native, quote_currency, fx_rate. "
                "The tool validates all orders, asks for one ADK confirmation for the whole batch, and submits only "
                "after the user approves. Write each rationale as an ontology-friendly investment memo with explicit "
                "ticker names, catalyst/risk/thesis/outcome terms, account or sleeve context, and why the side/size "
                "is appropriate."
            ),
            category="execution",
            callable=submit_order_batch_with_confirmation,
            tier="core",
            label_ko="ADK 승인 주문 일괄 제출",
            sort_order=2,
        ),
        ToolEntry(
            tool_id="submit_order_with_confirmation",
            name="submit_order_with_confirmation",
            description=(
                "Preferred order submission tool for investment chat. Validates the proposed order, asks for ADK "
                "tool confirmation, and submits only after the user approves in the ADK confirmation dialog. "
                "Write rationale as an ontology-friendly investment memo with explicit ticker names, clear "
                "catalyst/risk/thesis/outcome terms, user intent, evidence or account/sleeve context, and why the "
                "side/size is appropriate."
            ),
            category="execution",
            callable=submit_order_with_confirmation,
            tier="core",
            label_ko="ADK 승인 주문 제출",
            sort_order=3,
        ),
        ToolEntry(
            tool_id="validate_order_batch_draft",
            name="validate_order_batch_draft",
            description=(
                "Builds and risk-checks multiple proposed order drafts without submitting them. Use this for "
                "multi-order draft/risk-check requests where the user did not ask to submit. Always returns "
                "submission_status='not_submitted'."
            ),
            category="execution",
            callable=validate_order_batch_draft,
            tier="core",
            label_ko="주문 초안 일괄 검증",
            sort_order=4,
        ),
        ToolEntry(
            tool_id="validate_order_draft",
            name="validate_order_draft",
            description=(
                "Builds and risk-checks a proposed order draft without submitting it. Always returns "
                "submission_status='not_submitted'. Write rationale as an ontology-friendly investment memo with "
                "explicit ticker names, clear catalyst/risk/thesis/outcome terms, user intent, evidence or "
                "account/sleeve context, and why the side/size is appropriate."
            ),
            category="execution",
            callable=validate_order_draft,
            tier="core",
            label_ko="주문 초안 검증",
            sort_order=5,
        ),
        ToolEntry(
            tool_id="get_order_approval_status",
            name="get_order_approval_status",
            description="Reads recent investment-chat approval card/button results, including broker rejection messages. Use when the user asks what happened after approval.",
            category="execution",
            callable=get_order_approval_status,
            tier="core",
            label_ko="승인 결과 조회",
            sort_order=6,
        ),
    ]
    if include_internal_bridge:
        entries.extend(
            [
                ToolEntry(
                    tool_id="submit_approved_order",
                    name="submit_approved_order",
                    description="Internal backend/UI approval bridge for submitting a stored order draft. Not exposed to the LLM tool registry.",
                    category="execution",
                    callable=submit_approved_order,
                    tier="internal",
                    label_ko="승인 주문 제출",
                    sort_order=7,
                ),
                ToolEntry(
                    tool_id="submit_approved_order_batch",
                    name="submit_approved_order_batch",
                    description="Internal backend/UI approval bridge for submitting multiple stored order drafts. Not exposed to the LLM tool registry.",
                    category="execution",
                    callable=submit_approved_order_batch,
                    tier="internal",
                    label_ko="승인 주문 일괄 제출",
                    sort_order=8,
                ),
            ]
        )
    return entries


def build_order_tool_entries(*, repo: Any, settings: Settings, tenant_id: str) -> list[ToolEntry]:
    return _build_order_tool_entries(
        repo=repo,
        settings=settings,
        tenant_id=tenant_id,
        include_internal_bridge=False,
    )


def build_order_bridge_tool_entries(*, repo: Any, settings: Settings, tenant_id: str) -> list[ToolEntry]:
    return [
        entry
        for entry in _build_order_tool_entries(
        repo=repo,
        settings=settings,
        tenant_id=tenant_id,
        include_internal_bridge=True,
        )
        if entry.tier == "internal"
    ]
