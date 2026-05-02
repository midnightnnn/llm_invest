from __future__ import annotations

from contextvars import ContextVar

REQUEST_TENANT: ContextVar[str] = ContextVar("arena_investment_chat_tenant", default="")
REQUEST_USER_EMAIL: ContextVar[str] = ContextVar("arena_investment_chat_user_email", default="")
REQUEST_PROVIDER: ContextVar[str] = ContextVar("arena_investment_chat_provider", default="")
REQUEST_MODEL: ContextVar[str] = ContextVar("arena_investment_chat_model", default="")


def normalize_tenant(value: str | None) -> str:
    return str(value or "").strip().lower() or "local"
