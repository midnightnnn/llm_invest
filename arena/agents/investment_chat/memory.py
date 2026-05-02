from __future__ import annotations

from typing import Any

from arena.config import Settings
from arena.memory.store import MemoryStore
from arena.models import OrderIntent

from arena.agents.investment_chat.utils import model_dump


def build_execution_memory(repo: Any, settings: Settings) -> MemoryStore:
    return MemoryStore(
        repo=repo,
        trading_mode=str(getattr(settings, "trading_mode", "") or "paper").strip().lower() or "paper",
        memory_policy=getattr(settings, "memory_policy", {}) or {},
    )


def record_chat_decision_memory(
    *,
    memory_store: Any,
    intent: OrderIntent,
    report: Any,
    approval_token: str,
    tenant_id: str,
    scope: str,
    user_email: str,
) -> str | None:
    recorder = getattr(memory_store, "record_reflection", None)
    if not callable(recorder):
        return "record_reflection_unavailable"
    summary = (
        f"사용자+투자챗봇 판단: {intent.ticker} {intent.side.value} qty={intent.quantity:.4f} "
        f"scope={scope} target_agent={intent.agent_id}. 이유: {str(intent.rationale or '').strip()}"
    ).strip()
    payload = {
        "source": "investment_chat_order_decision",
        "judgment_source": "user+investment_chat",
        "tenant_id": tenant_id,
        "scope": scope,
        "target_agent_id": intent.agent_id,
        "approved_by": user_email,
        "approval_token": approval_token,
        "intent": intent.model_dump(mode="json"),
        "report": model_dump(report),
        "memory_tier": "semantic",
    }
    recorder(
        intent.agent_id,
        summary,
        score=0.72,
        payload=payload,
        semantic_key=f"chat_order_decision:{intent.intent_id}",
    )
    return None
