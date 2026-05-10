from __future__ import annotations

import html
import json
import time
from urllib.parse import urlencode

from fastapi import FastAPI, Request


# Bumped each process start so the ADK iframe URL changes whenever the server
# is restarted, defeating the browser disk cache that otherwise serves a stale
# index.html (with the previous arena-mobile-overrides block).
_IFRAME_CACHE_BUST = str(int(time.time()))
from fastapi.responses import JSONResponse

from arena.agents.investment_chat.context import REQUEST_USER_EMAIL, normalize_tenant
from arena.agents.investment_chat.config_tools import (
    build_config_bridge_tool_entries,
    load_chat_agent_config,
    recent_config_drafts,
)
from arena.agents.investment_chat.drafts import load_draft
from arena.agents.investment_chat.order_tools import build_order_bridge_tool_entries
from arena.agents.investment_chat.selection import (
    normalize_chat_model_selection,
    normalize_stored_advisor_model_selection,
    tenant_default_chat_selection,
)
from arena.providers.registry import canonical_provider, default_model_for_provider, list_adk_provider_specs
from arena.ui.investment_chat_adk import _chat_app_name
from arena.ui.investment_chat_providers import tenant_available_provider_specs
from arena.ui.routes.viewer import ViewerRouteDeps
from arena.ui.templating import render_ui_template


def _next_path(tenant_id: str = "", provider: str = "", model: str = "") -> str:
    query = urlencode(
        {
            key: value
            for key, value in {
                "tenant_id": tenant_id,
                "provider": provider,
                "model": model,
            }.items()
            if str(value or "").strip()
        }
    )
    return f"/investment-chat?{query}" if query else "/investment-chat"


def _adk_iframe_src(tenant_id: str, provider: str, model: str) -> str:
    provider = canonical_provider(provider) or str(provider or "").strip().lower()
    model = normalize_chat_model_selection(provider, model)
    params = {
        key: value
        for key, value in {
            "tenant_id": tenant_id,
            "provider": provider,
            "model": model,
        }.items()
        if str(value or "").strip()
    }
    params["_v"] = _IFRAME_CACHE_BUST
    return f"/investment-chat/adk/dev-ui/?{urlencode(params)}"


def _provider_options(repo=None, tenant_id: str = "") -> list[dict[str, str]]:
    specs = list(list_adk_provider_specs())
    if repo is not None and str(tenant_id or "").strip():
        specs, _credential_scoped = tenant_available_provider_specs(repo, tenant_id=tenant_id)
    return [
        {"value": spec.provider_id, "label": spec.label}
        for spec in specs
    ]


def _parse_audit_detail(row: dict) -> dict:
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


def _chat_quantity(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _order_result_chat_delivery_text(result: dict) -> str:
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    ticker = str(intent.get("ticker") or "").strip().upper()
    side = str(intent.get("side") or "").strip().upper()
    quantity = _chat_quantity(intent.get("quantity"))
    order_bits = [token for token in [ticker, side, f"{quantity}주" if quantity else ""] if token]
    if order_bits:
        return f"방금 {' '.join(order_bits)} 주문 승인 결과를 확인해서 알려줘."
    return "방금 주문 승인 결과를 확인해서 알려줘."


def _recent_order_drafts(repo, *, tenant_id: str, limit: int = 5) -> list[dict]:
    loader = getattr(repo, "recent_runtime_audit_logs", None)
    if not callable(loader):
        return []
    try:
        audit_rows = loader(limit=max(20, min(200, int(limit or 5) * 20))) or []
    except TypeError:
        audit_rows = loader(max(20, min(200, int(limit or 5) * 20))) or []
    except Exception:
        return []

    tenant = str(tenant_id or "").strip().lower() or "local"
    out: list[dict] = []
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
        if not token or token in seen:
            continue
        seen.add(token)
        draft = load_draft(repo, tenant_id=tenant, token=token)
        if not isinstance(draft, dict):
            continue
        if str(draft.get("approval_channel") or "").strip().lower() == "adk_tool_confirmation":
            continue
        status = str(draft.get("status") or "").strip().lower()
        risk = draft.get("risk") if isinstance(draft.get("risk"), dict) else {}
        intent = draft.get("intent") if isinstance(draft.get("intent"), dict) else {}
        out.append(
            {
                "approval_token": token,
                "status": status,
                "submittable": status == "draft",
                "expires_at": draft.get("expires_at") or "",
                "notional_krw": draft.get("notional_krw") or 0,
                "scope": draft.get("scope") or "account",
                "target_agent_id": draft.get("target_agent_id") or "",
                "intent": intent,
                "risk": risk,
            }
        )
        if len(out) >= max(1, min(int(limit or 5), 20)):
            break
    return out


def register_investment_chat_routes(app: FastAPI, *, deps: ViewerRouteDeps) -> None:
    @app.get("/investment-chat")
    def investment_chat(request: Request, tenant_id: str = "", provider: str = "", model: str = ""):
        tenant, _agent_ids, user, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=_next_path(tenant_id, provider, model),
        )
        if redirect is not None:
            return redirect
        provider_options = _provider_options(deps.repo, tenant)
        chat_config = load_chat_agent_config(deps.repo, tenant_id=tenant)
        tenant_settings = deps.settings_for_tenant(tenant)
        requested_provider = canonical_provider(provider) or str(provider or "").strip().lower()
        provider_token = requested_provider
        valid_providers = {item["value"] for item in provider_options}
        tenant_default_provider, tenant_default_model = tenant_default_chat_selection(
            tenant_settings,
            allowed_providers=valid_providers,
        )
        try:
            session_tenant = normalize_tenant(str(request.session.get("investment_chat_tenant_id") or ""))
        except Exception:
            session_tenant = ""
        if provider_token not in valid_providers:
            stored_provider = canonical_provider(chat_config.get("provider")) or str(
                chat_config.get("provider") or ""
            ).strip().lower()
            provider_token = stored_provider if stored_provider in valid_providers else ""
        if provider_token not in valid_providers and session_tenant == tenant:
            try:
                provider_token = canonical_provider(request.session.get("investment_chat_provider")) or str(
                    request.session.get("investment_chat_provider") or ""
                ).strip().lower()
            except Exception:
                provider_token = ""
        if provider_token not in valid_providers:
            provider_token = tenant_default_provider if tenant_default_provider in valid_providers else ""
        if provider_token not in valid_providers:
            provider_token = str(provider_options[0]["value"]) if provider_options else ""
        default_model = default_model_for_provider(tenant_settings, provider_token) if provider_token else ""
        if provider_token and provider_token == tenant_default_provider and tenant_default_model:
            default_model = tenant_default_model
        default_model = normalize_chat_model_selection(provider_token, default_model)
        model_token = str(model or "").strip() if provider_token and requested_provider == provider_token else ""
        model_token = normalize_chat_model_selection(provider_token, model_token)
        stored_provider = canonical_provider(chat_config.get("provider")) or str(chat_config.get("provider") or "").strip().lower()
        if not model_token and stored_provider == provider_token:
            model_token = normalize_stored_advisor_model_selection(
                provider_token,
                chat_config.get("model"),
                advisor_default_model=default_model,
                chat_config=chat_config,
            )
        if not model_token and session_tenant == tenant:
            try:
                session_provider = canonical_provider(request.session.get("investment_chat_provider")) or str(
                    request.session.get("investment_chat_provider") or ""
                ).strip().lower()
                if session_provider == provider_token:
                    model_token = normalize_stored_advisor_model_selection(
                        provider_token,
                        request.session.get("investment_chat_model"),
                        advisor_default_model=default_model,
                        chat_config=chat_config,
                    )
            except Exception:
                model_token = ""
        if not model_token and provider_token == tenant_default_provider:
            model_token = tenant_default_model
        if not model_token:
            model_token = default_model
        model_token = normalize_chat_model_selection(provider_token, model_token)
        try:
            request.session["investment_chat_tenant_id"] = tenant
            request.session["investment_chat_provider"] = provider_token
            request.session["investment_chat_model"] = model_token
        except Exception:
            pass

        chat_session_app_name = _chat_app_name(tenant, provider_token, model_token) if provider_token and model_token else ""
        chat_session_user_id = "user"
        body = render_ui_template(
            "investment_chat_body.jinja2",
            iframe_src=_adk_iframe_src(tenant, provider_token, model_token) if provider_token and model_token else "",
            chat_session_app_name=chat_session_app_name,
            chat_session_user_id=chat_session_user_id,
            tenant=html.escape(tenant),
        )
        return deps.html_response(
            deps.tailwind_layout(
                "투자챗봇",
                body,
                active="investment_chat",
                tenant=tenant,
                user=user,
                chat_session_app_name=chat_session_app_name,
                chat_session_user_id=chat_session_user_id,
                max_width_class="max-w-none",
                hide_page_header=True,
                main_class="flex-1 min-w-0 w-full p-0 box-border",
            ),
            max_age=0,
        )

    @app.get("/investment-chat/order-drafts")
    def investment_chat_order_drafts(request: Request, tenant_id: str = "", limit: int = 5):
        tenant, _agent_ids, _user, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=_next_path(tenant_id),
        )
        if redirect is not None:
            return JSONResponse({"status": "blocked", "error": "authentication required"}, status_code=401)
        return {"status": "ok", "tenant_id": tenant, "drafts": _recent_order_drafts(deps.repo, tenant_id=tenant, limit=limit)}

    @app.get("/investment-chat/config-drafts")
    def investment_chat_config_drafts(request: Request, tenant_id: str = "", limit: int = 5):
        tenant, _agent_ids, _user, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=_next_path(tenant_id),
        )
        if redirect is not None:
            return JSONResponse({"status": "blocked", "error": "authentication required"}, status_code=401)
        return {
            "status": "ok",
            "tenant_id": tenant,
            "drafts": recent_config_drafts(deps.repo, tenant_id=tenant, limit=limit),
        }

    @app.post("/investment-chat/config-drafts/{approval_token}/apply")
    def investment_chat_apply_config_draft(request: Request, approval_token: str, tenant_id: str = ""):
        tenant, _agent_ids, user, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=_next_path(tenant_id),
        )
        if redirect is not None:
            return JSONResponse({"status": "blocked", "error": "authentication required"}, status_code=401)
        token = str(approval_token or "").strip()
        if not token:
            return JSONResponse({"status": "blocked", "error": "approval_token is required"}, status_code=400)
        user_token = REQUEST_USER_EMAIL.set(str(getattr(user, "email", "") or "").strip().lower())
        try:
            tenant_settings = deps.settings_for_tenant(tenant)
            bridge_entries = build_config_bridge_tool_entries(
                repo=deps.repo,
                settings=tenant_settings,
                tenant_id=tenant,
            )
            apply_tool = bridge_entries[0].callable if bridge_entries else None
            if not callable(apply_tool):
                return JSONResponse({"status": "error", "error": "config approval bridge is unavailable"}, status_code=500)
            result = apply_tool(approval_token=token, confirmation_text=f"CONFIRM {token}")
            if result.get("status") in {"applied", "already_applied"}:
                deps.invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")
            if result.get("status") == "applied":
                result["chat_delivery_text"] = "방금 설정 변경 승인 결과를 확인해서 알려줘."
            elif result.get("status") == "already_applied":
                result["chat_delivery_text"] = "방금 설정 변경은 이미 적용된 상태야. 현재 적용 결과를 확인해서 알려줘."
            else:
                result["chat_delivery_text"] = "방금 설정 변경 승인이 실패했어. 실패 이유를 확인해서 알려줘."
            return JSONResponse(result, status_code=200 if result.get("status") != "error" else 500)
        finally:
            REQUEST_USER_EMAIL.reset(user_token)

    @app.post("/investment-chat/order-drafts/{approval_token}/submit")
    def investment_chat_submit_order_draft(request: Request, approval_token: str, tenant_id: str = ""):
        tenant, _agent_ids, user, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=_next_path(tenant_id),
        )
        if redirect is not None:
            return JSONResponse({"status": "blocked", "error": "authentication required"}, status_code=401)
        settings = deps.settings_for_tenant(tenant)
        bridge_entries = build_order_bridge_tool_entries(repo=deps.repo, settings=settings, tenant_id=tenant)
        entry = next((item for item in bridge_entries if item.name == "submit_approved_order"), None)
        if entry is None or not callable(entry.callable):
            return JSONResponse({"status": "error", "error": "order approval bridge unavailable"}, status_code=500)
        user_email = str((user or {}).get("email") or "").strip().lower()
        token = str(approval_token or "").strip()
        email_token = REQUEST_USER_EMAIL.set(user_email)
        try:
            result = entry.callable(approval_token=token, confirmation_text=f"CONFIRM {token}")
        finally:
            REQUEST_USER_EMAIL.reset(email_token)
        if isinstance(result, dict):
            result = dict(result)
            result["chat_delivery_text"] = _order_result_chat_delivery_text(result)
        return result
