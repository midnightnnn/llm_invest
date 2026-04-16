from __future__ import annotations

import html
import secrets
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from arena.data.bq import BigQueryRepository
from arena.ui.templating import render_ui_template


@dataclass(frozen=True)
class AuthRouteDeps:
    repo: BigQueryRepository
    auth_enabled: bool
    google_client_id: str
    google_client_secret: str
    redirect_uri: str
    viewer_roles: set[str]
    tailwind_layout: Callable[..., str]
    html_response: Callable[..., Response]
    current_user: Callable[[Request], dict[str, Any] | None]
    access_rows_for_user: Callable[[str], list[dict[str, str]]]
    tenant_list_for_roles: Callable[[list[dict[str, str]], set[str]], list[str]]
    latest_access_request: Callable[[str], dict[str, Any] | None]
    ensure_access_request_pending: Callable[[dict[str, Any] | None], dict[str, Any] | None]
    ensure_user_access: Callable[[dict[str, Any] | None], Any]
    exchange_google_code_for_user: Callable[[str], dict[str, str]]
    fmt_ts: Callable[[object], str]


def register_auth_routes(app: FastAPI, *, deps: AuthRouteDeps) -> None:
    def _auth_notice_html(
        *,
        eyebrow: str,
        eyebrow_text_class: str,
        border_class: str,
        title: str,
        paragraphs: list[dict[str, str]],
        actions: list[dict[str, str]],
        note: str = "",
    ) -> str:
        return render_ui_template(
            "auth_notice.jinja2",
            eyebrow=eyebrow,
            eyebrow_text_class=eyebrow_text_class,
            border_class=border_class,
            title=title,
            paragraphs=paragraphs,
            note=note,
            actions=actions,
        )

    @app.get("/auth/google/login")
    def auth_google_login(request: Request) -> Response:
        if not deps.auth_enabled:
            return RedirectResponse(url="/settings", status_code=302)
        if not deps.google_client_id or not deps.google_client_secret:
            body = render_ui_template(
                "inline_notice.jinja2",
                container_classes="mt-6 reveal rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800",
                message="Google OAuth env missing: set GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET",
            )
            return HTMLResponse(
                deps.tailwind_layout("Settings", body, active="settings"),
                status_code=500,
                headers={"Cache-Control": "no-store"},
            )
        state = secrets.token_urlsafe(24)
        request.session["oauth_state"] = state
        query = urlencode(
            {
                "client_id": deps.google_client_id,
                "redirect_uri": deps.redirect_uri,
                "response_type": "code",
                "scope": "openid email profile",
                "access_type": "offline",
                "prompt": "select_account",
                "state": state,
            }
        )
        return RedirectResponse(url=f"https://accounts.google.com/o/oauth2/v2/auth?{query}", status_code=302)

    @app.get("/auth/google/callback")
    def auth_google_callback(
        request: Request,
        code: str = Query(default=""),
        state: str = Query(default=""),
    ) -> RedirectResponse:
        if not deps.auth_enabled:
            return RedirectResponse(url="/settings", status_code=302)
        saved_state = str(request.session.get("oauth_state") or "")
        request.session.pop("oauth_state", None)
        if not code or not state or not saved_state or state != saved_state:
            deps.repo.append_runtime_audit_log(action="auth_google_callback", status="denied", detail={"reason": "invalid_state"})
            return RedirectResponse(url="/settings?ok=0&msg=invalid%20oauth%20state", status_code=302)
        try:
            user = deps.exchange_google_code_for_user(code)
            request.session["user"] = user
            access_result = deps.ensure_user_access(user)
            access_rows = list(getattr(access_result, "access_rows", []))
            viewer_tenants = deps.tenant_list_for_roles(access_rows, deps.viewer_roles)
            next_path = str(request.session.pop("next_path", "") or "/")
            if str(getattr(access_result, "blocked_status", "") or "").strip().lower() == "rejected":
                request.session["next_path"] = next_path
                deps.repo.append_runtime_audit_log(
                    action="auth_google_login",
                    status="denied",
                    user_email=user["email"],
                    detail={"name": user.get("name"), "reason": "rejected"},
                )
                return RedirectResponse(url="/auth/pending", status_code=302)
            if not viewer_tenants:
                deps.repo.append_runtime_audit_log(
                    action="auth_google_login",
                    status="error",
                    user_email=user["email"],
                    detail={"name": user.get("name"), "reason": "tenant_provision_failed"},
                )
                return RedirectResponse(url="/settings?ok=0&msg=tenant%20provisioning%20failed", status_code=302)
            if bool(getattr(access_result, "created_tenant", False)) and next_path in {"", "/"}:
                next_path = f"/settings?tenant_id={viewer_tenants[0]}"
            deps.repo.append_runtime_audit_log(
                action="auth_google_login",
                status="ok",
                user_email=user["email"],
                tenant_id=viewer_tenants[0],
                detail={
                    "name": user.get("name"),
                    "auto_provisioned": bool(getattr(access_result, "created_tenant", False)),
                },
            )
            return RedirectResponse(url=next_path, status_code=302)
        except Exception as exc:
            deps.repo.append_runtime_audit_log(
                action="auth_google_login",
                status="error",
                detail={"error": str(exc)},
            )
            return RedirectResponse(url="/settings?ok=0&msg=google%20login%20failed", status_code=302)

    @app.get("/auth/logout")
    def auth_logout(request: Request) -> RedirectResponse:
        user = deps.current_user(request)
        if user:
            deps.repo.append_runtime_audit_log(action="auth_logout", status="ok", user_email=str(user.get("email") or ""))
        request.session.clear()
        return RedirectResponse(url="/settings", status_code=302)

    @app.get("/auth/pending", response_class=HTMLResponse)
    def auth_pending(request: Request) -> Response:
        if not deps.auth_enabled:
            return RedirectResponse(url="/", status_code=302)
        user = deps.current_user(request)
        if not user:
            request.session["next_path"] = str(request.session.get("next_path") or "/")
            return RedirectResponse(url="/auth/google/login", status_code=302)
        user_email = str(user.get("email") or "").strip().lower()
        access_rows = deps.access_rows_for_user(user_email)
        viewer_tenants = deps.tenant_list_for_roles(access_rows, deps.viewer_roles)
        if viewer_tenants:
            next_path = str(request.session.pop("next_path", "") or "/")
            return RedirectResponse(url=next_path, status_code=302)
        request_row = deps.latest_access_request(user_email) or deps.ensure_access_request_pending(user) or {}
        status = str(request_row.get("status") or "pending").strip().lower() or "pending"
        requested_at = deps.fmt_ts(request_row.get("requested_at") or "")
        note = html.escape(str(request_row.get("note") or "").strip())
        if status == "rejected":
            body = _auth_notice_html(
                eyebrow="Access Review",
                eyebrow_text_class="text-rose-500",
                border_class="border-rose-200",
                title="접근 요청이 거절되었습니다",
                paragraphs=[
                    {
                        "classes": "mt-4 text-sm leading-7 text-ink-700",
                        "html": (
                            f'계정 <span class="font-semibold text-ink-900">{html.escape(user_email)}</span> '
                            "은 현재 승인되지 않았습니다."
                        ),
                    }
                ],
                note=note,
                actions=[
                    {
                        "href": "/auth/logout",
                        "label": "Logout",
                        "classes": "rounded-xl border border-ink-300 bg-white px-4 py-2 text-sm font-semibold text-ink-700 hover:bg-ink-50",
                    }
                ],
            )
            return deps.html_response(deps.tailwind_layout("Access Denied", body, active=""), max_age=0)
        body = _auth_notice_html(
            eyebrow="Access Pending",
            eyebrow_text_class="text-amber-500",
            border_class="border-amber-200",
            title="승인 대기 중입니다",
            paragraphs=[
                {
                    "classes": "mt-4 text-sm leading-7 text-ink-700",
                    "html": (
                        "구글 로그인은 완료되었습니다. "
                        f'<span class="font-semibold text-ink-900">{html.escape(user_email)}</span> '
                        "계정은 아직 UI 접근 승인이 나지 않았습니다."
                    ),
                },
                {
                    "classes": "mt-3 text-sm text-ink-500",
                    "html": f"요청 시각: {html.escape(requested_at or '-')}",
                },
                {
                    "classes": "mt-2 text-sm text-ink-500",
                    "html": "승인 후 이 페이지를 새로고침하거나 다시 로그인하면 바로 입장할 수 있습니다.",
                },
            ],
            actions=[
                {
                    "href": "/auth/logout",
                    "label": "Logout",
                    "classes": "rounded-xl border border-ink-300 bg-white px-4 py-2 text-sm font-semibold text-ink-700 hover:bg-ink-50",
                },
                {
                    "href": "/auth/pending",
                    "label": "Refresh",
                    "classes": "rounded-xl bg-ink-900 px-4 py-2 text-sm font-semibold text-white hover:bg-ink-700",
                },
            ],
        )
        return deps.html_response(deps.tailwind_layout("Access Pending", body, active=""), max_age=0)

    @app.get("/auth/forbidden", response_class=HTMLResponse)
    def auth_forbidden(request: Request) -> Response:
        if not deps.auth_enabled:
            return RedirectResponse(url="/", status_code=302)
        user = deps.current_user(request)
        if not user:
            return RedirectResponse(url="/auth/google/login", status_code=302)
        body = _auth_notice_html(
            eyebrow="Viewer Account",
            eyebrow_text_class="text-sky-500",
            border_class="border-sky-200",
            title="관리자 권한이 없습니다",
            paragraphs=[
                {
                    "classes": "mt-4 text-sm leading-7 text-ink-700",
                    "html": (
                        f'<span class="font-semibold text-ink-900">{html.escape(str(user.get("email") or ""))}</span> '
                        "계정은 조회 권한만 승인되었습니다. 설정 또는 관리자 페이지는 owner/admin 계정으로만 접근할 수 있습니다."
                    ),
                }
            ],
            actions=[
                {
                    "href": "/board",
                    "label": "Board",
                    "classes": "rounded-xl bg-ink-900 px-4 py-2 text-sm font-semibold text-white hover:bg-ink-700",
                },
                {
                    "href": "/auth/logout",
                    "label": "Logout",
                    "classes": "rounded-xl border border-ink-300 bg-white px-4 py-2 text-sm font-semibold text-ink-700 hover:bg-ink-50",
                },
            ],
        )
        return deps.html_response(deps.tailwind_layout("Forbidden", body, active=""), max_age=0)
