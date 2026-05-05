from __future__ import annotations

import logging
import re
from secrets import token_hex

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from arena.logging_utils import failure_extra

logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def _is_api_error_boundary_path(path: str) -> bool:
    clean = str(path or "")
    return clean == "/api" or clean.startswith("/api/") or clean.startswith("/investment-chat/order-drafts")


def _request_id(request: Request) -> str:
    raw = str(request.headers.get("X-Request-ID") or "").strip()
    if raw:
        clean = _REQUEST_ID_RE.sub("-", raw)[:96].strip("-")
        if clean:
            return clean
    return f"req_{token_hex(12)}"


def _request_tenant(request: Request) -> str:
    try:
        tenant = str(request.query_params.get("tenant_id") or "").strip().lower()
    except Exception:
        tenant = ""
    if tenant:
        return tenant
    try:
        session = getattr(request, "session", None)
        if isinstance(session, dict):
            tenant = str(session.get("investment_chat_tenant_id") or "").strip().lower()
    except Exception:
        tenant = ""
    return tenant or "-"


class ApiErrorBoundaryMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path") or "")
        if not _is_api_error_boundary_path(path):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        request_id = _request_id(request)
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        response_started = False

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
                MutableHeaders(scope=message).setdefault("X-Request-ID", request_id)
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception as exc:
            if response_started:
                raise
            tenant = _request_tenant(request)
            logger.exception(
                "[red]UI API request failed[/red] request_id=%s method=%s path=%s tenant=%s err=%s",
                request_id,
                request.method,
                path,
                tenant,
                str(exc),
                extra=failure_extra(
                    "ui_api_request_failed",
                    exc,
                    request_id=request_id,
                    method=request.method,
                    path=path,
                    tenant_id=tenant,
                    status_code=500,
                ),
            )
            response = JSONResponse(
                {
                    "status": "error",
                    "error": "internal_server_error",
                    "request_id": request_id,
                },
                status_code=500,
                headers={
                    "Cache-Control": "no-store",
                    "X-Request-ID": request_id,
                },
            )
            await response(scope, receive, send)


def register_api_error_middleware(app: FastAPI) -> None:
    """Adds JSON 500 logging for API routes while leaving HTML routes alone."""
    app.add_middleware(ApiErrorBoundaryMiddleware)
