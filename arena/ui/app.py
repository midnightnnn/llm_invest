from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

_KST = ZoneInfo("Asia/Seoul")
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.sessions import SessionMiddleware

from arena.config import (
    Settings,
    distribution_allows_paper_kis_credentials,
    distribution_allows_real_kis_credentials,
    distribution_uses_broker_credentials,
    normalize_distribution_mode,
)
from arena.data.bq import BigQueryRepository
from arena.market_sources import live_market_sources_for_markets, parse_markets
from arena.security import CredentialStore
from arena.ui.access import access_rows_for_user as _access_rows_for_user_raw
from arena.ui.access import build_operator_emails as _build_operator_emails
from arena.ui.access import default_tenant_for_email as _default_tenant_for_email_raw
from arena.ui.access import is_operator as _is_operator_raw
from arena.ui.access import safe_tenant as _safe_tenant_raw
from arena.ui.access import tenant_list_for_roles as _tenant_list_for_roles_raw
from arena.ui.http import html_response as _html_response
from arena.ui.http import json_response as _json_response
from arena.ui.layout import md_block as _md_block
from arena.ui.layout import tailwind_layout as _tailwind_layout_raw
from arena.ui.memory import register_memory_routes
from arena.ui.provisioning import UIProvisioner
from arena.ui.app_support import (
    RUN_REASON_LABELS as _RUN_REASON_LABELS,
    RUN_STAGE_LABELS as _RUN_STAGE_LABELS,
    RUN_STATUS_META as _RUN_STATUS_META,
    agent_logo_svg as _agent_logo_svg,
    default_prompt_template as _default_prompt_template,
    float_env as _float_env,
    fmt_ts as _fmt_ts,
    provider_api_key_help_html as _provider_api_key_help_html,
    to_date as _to_date,
)
from arena.ui.run_status import build_run_status_helpers
from arena.ui.routes.auth import AuthRouteDeps, register_auth_routes
from arena.ui.routes.board import register_board_routes
from arena.ui.routes.nav import register_nav_routes
from arena.ui.routes.ops import OpsRouteDeps, register_ops_routes
from arena.ui.routes.overview import register_overview_routes
from arena.ui.routes.settings_page import (
    SettingsPageRouteDeps,
    register_settings_page_routes,
)
from arena.ui.routes.settings_admin import (
    AdminSettingsRouteDeps,
    register_admin_settings_routes,
)
from arena.ui.routes.sleeves import SleeveRouteDeps, register_sleeve_routes
from arena.ui.routes.trades import register_trades_routes
from arena.ui.routes.viewer import ViewerRouteDeps
from arena.ui.viewer_analytics import (
    chained_index,
    drawdown,
    max_drawdown,
    metric_card,
    total_return,
)
from arena.ui.viewer_data import build_viewer_data_helpers
from arena.ui.runtime import (
    UIRuntime,
    _parse_json_array,
    _parse_json_object,
    _safe_float,
    _safe_int,
    _to_bool_token,
)

logger = logging.getLogger(__name__)


def _build_app(*, repo: BigQueryRepository, settings: Settings) -> FastAPI:
    # --- Thread pool for parallel BigQuery queries ---
    _executor = ThreadPoolExecutor(max_workers=8)

    credential_store_error = ""
    try:
        credential_store = CredentialStore(project=settings.google_cloud_project, repo=repo)
    except Exception as exc:
        credential_store = None
        credential_store_error = str(exc)

    runtime = UIRuntime(
        repo=repo,
        settings=settings,
        executor=_executor,
        default_prompt_template=_default_prompt_template,
        credential_store=credential_store,
    )

    @asynccontextmanager
    async def _app_lifespan(_app: FastAPI):
        _ = _app
        try:
            yield
        finally:
            runtime.clear_cache()
            _executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="LLM ARENA", docs_url=None, redoc_url=None, lifespan=_app_lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=500)

    auth_enabled = str(os.getenv("ARENA_UI_AUTH_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    settings_enabled = str(os.getenv("ARENA_UI_SETTINGS_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}

    _operator_emails = _build_operator_emails(
        os.getenv("ARENA_OPERATOR_EMAILS", ""),
        auth_enabled=auth_enabled,
    )

    def _is_operator(user: dict[str, Any] | None) -> bool:
        return _is_operator_raw(user, _operator_emails)
    google_client_id = str(os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")).strip()
    google_client_secret = str(os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "")).strip()
    redirect_uri = str(os.getenv("ARENA_UI_GOOGLE_REDIRECT_URI", "http://127.0.0.1:8080/auth/google/callback")).strip()
    session_secret = str(os.getenv("ARENA_UI_SESSION_SECRET", "dev-only-change-me")).strip()
    if auth_enabled and session_secret == "dev-only-change-me":
        logger.warning("[red]ARENA_UI_SESSION_SECRET is default — set a strong secret for production[/red]")
    app.add_middleware(SessionMiddleware, secret_key=session_secret, same_site="lax", https_only=False)

    def _cached_fetch(cache_key: str, fetch_fn, *args, **kwargs):
        return runtime.cached_fetch(cache_key, fetch_fn, *args, **kwargs)

    def _invalidate_tenant_cache(tenant: str, *scopes: str) -> None:
        runtime.invalidate_tenant_cache(tenant, *scopes)

    def _invalidate_tenant_runtime_cache(tenant: str) -> None:
        runtime.invalidate_tenant_runtime_cache(tenant)

    def _safe_tenant(value: str) -> str:
        return _safe_tenant_raw(value)

    def _default_tenant_for_email(email: str) -> str:
        return _default_tenant_for_email_raw(email)

    def _current_user(request: Request) -> dict[str, Any] | None:
        if not auth_enabled:
            return {"email": "local@localhost", "name": "Local User", "sub": "local"}
        user = request.session.get("user")
        if isinstance(user, dict) and user.get("email"):
            return user
        return None

    def _require_user(request: Request, next_path: str) -> dict[str, Any] | None:
        user = _current_user(request)
        if user:
            return user
        request.session["next_path"] = next_path
        return None

    def _exchange_google_code_for_user(code: str) -> dict[str, str]:
        token_res = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": google_client_id,
                "client_secret": google_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
            timeout=20,
        )
        token_res.raise_for_status()
        token_body = token_res.json()
        raw_id_token = str(token_body.get("id_token") or "").strip()
        if not raw_id_token:
            raise RuntimeError("missing id_token from Google OAuth response")
        info = google_id_token.verify_oauth2_token(raw_id_token, google_requests.Request(), google_client_id)
        email = str(info.get("email") or "").strip().lower()
        if not email:
            raise RuntimeError("Google account email missing")
        return {
            "email": email,
            "name": str(info.get("name") or email),
            "sub": str(info.get("sub") or ""),
        }

    _VIEWER_ROLES = {"viewer", "admin", "owner"}
    _ADMIN_ROLES = {"admin", "owner"}
    provisioner = UIProvisioner(repo=repo)

    def _access_rows_for_user(user_email: str) -> list[dict[str, str]]:
        return _access_rows_for_user_raw(repo, user_email)

    def _tenant_list_for_roles(access_rows: list[dict[str, str]], allowed_roles: set[str]) -> list[str]:
        return _tenant_list_for_roles_raw(access_rows, allowed_roles)

    def _ensure_access_request_pending(user: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(user, dict):
            return None
        saver = getattr(repo, "ensure_runtime_access_request_pending", None)
        if not callable(saver):
            return None
        try:
            return saver(
                user_email=str(user.get("email") or "").strip().lower(),
                user_name=str(user.get("name") or "").strip(),
                google_sub=str(user.get("sub") or "").strip(),
            )
        except Exception as exc:
            logger.warning("[yellow]runtime access request save skipped[/yellow] err=%s", str(exc))
            return None

    def _latest_access_request(user_email: str) -> dict[str, Any] | None:
        loader = getattr(repo, "latest_runtime_access_request", None)
        if not callable(loader):
            return None
        try:
            return loader(user_email=str(user_email or "").strip().lower())
        except Exception:
            return None

    def _pending_redirect(request: Request, *, next_path: str, record_pending: bool = True) -> RedirectResponse:
        request.session["next_path"] = next_path
        if record_pending:
            _ensure_access_request_pending(_current_user(request))
        return RedirectResponse(url="/auth/pending", status_code=302)

    def _forbidden_redirect(request: Request, *, next_path: str) -> RedirectResponse:
        request.session["next_path"] = next_path
        return RedirectResponse(url="/auth/forbidden", status_code=302)

    def _resolve_admin_context(
        request: Request,
        *,
        requested_tenant: str,
        next_path: str,
    ) -> tuple[dict[str, Any] | None, str, str, list[str], RedirectResponse | None]:
        user = _require_user(request, next_path) if auth_enabled else _current_user(request)
        if auth_enabled and not user:
            return None, "", "local", [], RedirectResponse(url="/auth/google/login", status_code=302)

        user_email = str((user or {}).get("email") or "").strip().lower()
        access_result = provisioner.ensure_user_access(user) if auth_enabled else None
        access_rows = access_result.access_rows if auth_enabled and access_result is not None else []
        if auth_enabled and access_result is not None and access_result.blocked_status == "rejected":
            return user, user_email, "local", [], _pending_redirect(
                request,
                next_path=next_path,
                record_pending=False,
            )
        viewer_tenants = _tenant_list_for_roles(access_rows, _VIEWER_ROLES) if auth_enabled else []
        if auth_enabled and not viewer_tenants:
            return user, user_email, "local", [], _pending_redirect(request, next_path=next_path)
        allowed_tenants = _tenant_list_for_roles(access_rows, _ADMIN_ROLES) if auth_enabled else []
        if auth_enabled and not allowed_tenants:
            fallback = viewer_tenants[0] if viewer_tenants else "local"
            return user, user_email, fallback, [], _forbidden_redirect(request, next_path=next_path)

        requested = str(requested_tenant or "").strip().lower()
        if auth_enabled:
            tenant = requested if requested and requested in allowed_tenants else (allowed_tenants[0] if allowed_tenants else "local")
        else:
            tenant = requested or "local"
        return user, user_email, tenant, allowed_tenants, None

    def _tenant_access_denied(requested_tenant: str, resolved_tenant: str) -> bool:
        requested = str(requested_tenant or "").strip().lower()
        if not auth_enabled:
            return False
        if not requested:
            return False
        return requested != str(resolved_tenant or "").strip().lower()

    def _scoped_agent_ids_for_tenant(tenant: str) -> list[str]:
        tenant_settings = _settings_for_tenant(tenant)
        parsed = [str(x).strip().lower() for x in tenant_settings.agent_ids if str(x).strip()]
        if parsed:
            return parsed
        return [str(x).strip().lower() for x in settings.agent_ids if str(x).strip()]

    def _is_live_mode(active_settings: Settings | None = None) -> bool:
        runtime_settings = active_settings or settings
        trading_mode = str(runtime_settings.trading_mode or "").strip().lower()
        distribution_mode = normalize_distribution_mode(getattr(runtime_settings, "distribution_mode", "private"))
        return trading_mode == "live" and distribution_mode != "simulated_only"

    def _live_market_sources(active_settings: Settings | None = None) -> list[str] | None:
        runtime_settings = active_settings or settings
        return live_market_sources_for_markets(parse_markets(runtime_settings.kis_target_market)) or None

    def _resolve_viewer_context(
        request: Request,
        *,
        requested_tenant: str,
        next_path: str,
    ) -> tuple[str, list[str], dict[str, Any] | None, RedirectResponse | None]:
        user = _require_user(request, next_path) if auth_enabled else _current_user(request)
        if auth_enabled and not user:
            return "local", [], None, RedirectResponse(url="/auth/google/login", status_code=302)
        user_email = str((user or {}).get("email") or "").strip().lower()
        access_result = provisioner.ensure_user_access(user) if auth_enabled else None
        access_rows = access_result.access_rows if auth_enabled and access_result is not None else []
        if auth_enabled and access_result is not None and access_result.blocked_status == "rejected":
            return "local", [], user, _pending_redirect(request, next_path=next_path, record_pending=False)
        allowed_tenants = _tenant_list_for_roles(access_rows, _VIEWER_ROLES) if auth_enabled else []
        if auth_enabled and not allowed_tenants:
            return "local", [], user, _pending_redirect(request, next_path=next_path)
        requested = str(requested_tenant or "").strip().lower()
        if auth_enabled:
            tenant = requested if requested and requested in allowed_tenants else (allowed_tenants[0] if allowed_tenants else "local")
        else:
            tenant = requested or "local"
        scoped_agent_ids = _scoped_agent_ids_for_tenant(tenant)
        if not scoped_agent_ids:
            scoped_agent_ids = [str(x).strip().lower() for x in settings.agent_ids if str(x).strip()]
        return tenant, scoped_agent_ids, user, None

    def _settings_for_tenant(tenant: str) -> Settings:
        return runtime.settings_for_tenant(tenant)

    def _get_default_registry(tenant: str):
        return runtime.get_default_registry(tenant)

    def _current_admin_view_model(tenant: str, *, _latest_creds: dict[str, Any] | None = None) -> dict[str, Any]:
        return runtime.current_admin_view_model(tenant, latest_creds=_latest_creds)

    run_status_helpers = build_run_status_helpers(
        repo=repo,
        cached_fetch=_cached_fetch,
        parse_json_object=_parse_json_object,
        safe_float=_safe_float,
        run_status_meta=_RUN_STATUS_META,
        run_reason_labels=_RUN_REASON_LABELS,
        run_stage_labels=_RUN_STAGE_LABELS,
    )
    _latest_tenant_status_payload = run_status_helpers.latest_tenant_status_payload
    _header_status_kwargs = run_status_helpers.header_status_kwargs

    def _tailwind_layout(*args, tenant: str = "local", user: dict[str, Any] | None = None, **kwargs) -> str:
        kwargs.update(_header_status_kwargs(tenant))
        kwargs.setdefault("tenant", tenant)
        if str(tenant).strip().lower() == "midnightnnn" or (user and _is_operator(user)):
            kwargs.setdefault("extra_nav_items", [("/ops", "Ops", "ops")])
        return _tailwind_layout_raw(*args, **kwargs)

    viewer_data_helpers = build_viewer_data_helpers(
        repo=repo,
        settings=settings,
        executor=_executor,
        settings_for_tenant=_settings_for_tenant,
        is_live_mode=_is_live_mode,
        live_market_sources=_live_market_sources,
        agent_logo_svg=_agent_logo_svg,
        safe_float=_safe_float,
        safe_int=_safe_int,
        to_date=_to_date,
    )

    register_ops_routes(
        app,
        deps=OpsRouteDeps(
            repo=repo,
            executor=_executor,
            cached_fetch=_cached_fetch,
            current_user=_current_user,
            is_operator=_is_operator,
            tailwind_layout=_tailwind_layout,
            html_response=_html_response,
            json_response=_json_response,
            metric_card=metric_card,
            run_status_meta=_RUN_STATUS_META,
            credential_store=credential_store,
        ),
    )

    register_memory_routes(
        app,
        repo=repo,
        settings=settings,
        settings_enabled=settings_enabled,
        resolve_admin_context=_resolve_admin_context,
        cached_fetch=_cached_fetch,
        invalidate_tenant_cache=_invalidate_tenant_cache,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    register_auth_routes(
        app,
        deps=AuthRouteDeps(
            repo=repo,
            auth_enabled=auth_enabled,
            google_client_id=google_client_id,
            google_client_secret=google_client_secret,
            redirect_uri=redirect_uri,
            viewer_roles=_VIEWER_ROLES,
            tailwind_layout=_tailwind_layout,
            html_response=_html_response,
            current_user=_current_user,
            access_rows_for_user=_access_rows_for_user,
            tenant_list_for_roles=_tenant_list_for_roles,
            latest_access_request=_latest_access_request,
            ensure_access_request_pending=_ensure_access_request_pending,
            ensure_user_access=provisioner.ensure_user_access,
            exchange_google_code_for_user=_exchange_google_code_for_user,
            fmt_ts=_fmt_ts,
        ),
    )

    viewer_route_deps = ViewerRouteDeps(
        repo=repo,
        executor=_executor,
        auth_enabled=auth_enabled,
        kst=_KST,
        resolve_viewer_context=_resolve_viewer_context,
        cached_fetch=_cached_fetch,
        current_user=_current_user,
        tailwind_layout=_tailwind_layout,
        html_response=_html_response,
        json_response=_json_response,
        get_default_registry=_get_default_registry,
        settings_for_tenant=_settings_for_tenant,
        latest_tenant_status_payload=_latest_tenant_status_payload,
        fetch_board=viewer_data_helpers.fetch_board,
        fetch_tool_events_for_post=viewer_data_helpers.fetch_tool_events_for_post,
        fetch_theses_for_board_post=viewer_data_helpers.fetch_theses_for_board_post,
        fetch_nav=viewer_data_helpers.fetch_nav,
        fetch_token_usage_summary=viewer_data_helpers.fetch_token_usage_summary,
        fetch_trade_count_summary=viewer_data_helpers.fetch_trade_count_summary,
        fetch_token_usage_daily=viewer_data_helpers.fetch_token_usage_daily,
        fetch_trade_count_daily=viewer_data_helpers.fetch_trade_count_daily,
        fetch_trades=viewer_data_helpers.fetch_trades,
        fetch_trades_for_board_post=viewer_data_helpers.fetch_trades_for_board_post,
        fetch_sleeve_snapshot_cards=viewer_data_helpers.fetch_sleeve_snapshot_cards,
        default_benchmark=viewer_data_helpers.default_benchmark,
        is_live_mode=_is_live_mode,
        metric_card=metric_card,
        fmt_ts=_fmt_ts,
        md_block=_md_block,
        safe_float=_safe_float,
        safe_int=_safe_int,
        to_date=_to_date,
        chained_index=chained_index,
        drawdown=drawdown,
        total_return=total_return,
        max_drawdown=max_drawdown,
    )
    register_overview_routes(app, deps=viewer_route_deps)
    register_board_routes(app, deps=viewer_route_deps)
    register_nav_routes(app, deps=viewer_route_deps)
    register_trades_routes(app, deps=viewer_route_deps)

    register_settings_page_routes(
        app,
        deps=SettingsPageRouteDeps(
            repo=repo,
            settings=settings,
            settings_enabled=settings_enabled,
            auth_enabled=auth_enabled,
            credential_store=credential_store,
            credential_store_error=credential_store_error,
            executor=_executor,
            cached_fetch=_cached_fetch,
            current_admin_view_model=_current_admin_view_model,
            current_user=_current_user,
            fmt_ts=_fmt_ts,
            html_response=_html_response,
            provider_api_key_help_html=_provider_api_key_help_html,
            resolve_admin_context=_resolve_admin_context,
            is_live_mode=_is_live_mode,
            invalidate_tenant_runtime_cache=_invalidate_tenant_runtime_cache,
            settings_for_tenant=_settings_for_tenant,
            tailwind_layout=_tailwind_layout,
            tenant_access_denied=_tenant_access_denied,
        ),
    )

    def _settings_redirect(tenant_id: str, *, ok: bool, msg: str, tab: str = "") -> RedirectResponse:
        params: dict[str, str] = {
            "tenant_id": str(tenant_id or "local").strip().lower() or "local",
            "ok": "1" if ok else "0",
            "msg": msg,
        }
        if tab:
            params["tab"] = tab
        query = urlencode(params)
        return RedirectResponse(url=f"/settings?{query}", status_code=303)

    register_sleeve_routes(
        app,
        deps=SleeveRouteDeps(
            repo=repo,
            settings_enabled=settings_enabled,
            resolve_admin_context=_resolve_admin_context,
            resolve_viewer_context=_resolve_viewer_context,
            tenant_access_denied=_tenant_access_denied,
            current_admin_view_model=_current_admin_view_model,
            settings_redirect=_settings_redirect,
            invalidate_tenant_cache=_invalidate_tenant_cache,
            settings_for_tenant=_settings_for_tenant,
            scoped_agent_ids_for_tenant=_scoped_agent_ids_for_tenant,
            is_live_mode=_is_live_mode,
            live_market_sources=_live_market_sources,
            cached_fetch=_cached_fetch,
            fetch_sleeves=viewer_data_helpers.fetch_sleeves,
            fetch_sleeve_snapshot_cards=viewer_data_helpers.fetch_sleeve_snapshot_cards,
            tailwind_layout=_tailwind_layout,
            html_response=_html_response,
            json_response=_json_response,
            fmt_ts=_fmt_ts,
            float_env=_float_env,
        ),
    )

    register_admin_settings_routes(
        app,
        deps=AdminSettingsRouteDeps(
            repo=repo,
            settings_enabled=settings_enabled,
            credential_store=credential_store,
            resolve_admin_context=_resolve_admin_context,
            tenant_access_denied=_tenant_access_denied,
            current_admin_view_model=_current_admin_view_model,
            settings_for_tenant=_settings_for_tenant,
            get_default_registry=_get_default_registry,
            invalidate_tenant_cache=_invalidate_tenant_cache,
            invalidate_tenant_runtime_cache=_invalidate_tenant_runtime_cache,
            settings_redirect=_settings_redirect,
            parse_json_array=_parse_json_array,
            safe_float=_safe_float,
            to_bool_token=_to_bool_token,
            is_live_mode=_is_live_mode,
            live_market_sources=_live_market_sources,
        ),
    )

    return app


def serve_ui(
    *,
    repo: BigQueryRepository,
    settings: Settings,
    host: str = "0.0.0.0",
    port: int | None = None,
) -> None:
    """Starts FastAPI-based inspector UI server."""
    resolved_port = int(os.getenv("PORT", str(port or 8080)))
    app = _build_app(repo=repo, settings=settings)
    logger.info("[green]UI server starting (FastAPI)[/green] host=%s port=%d", host, resolved_port)
    uvicorn.run(app, host=host, port=resolved_port, log_level="info")
