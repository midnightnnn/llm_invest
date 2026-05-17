from __future__ import annotations

import html
import json
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, Form, Request


# Bumped each process start so the ADK iframe URL changes whenever the server
# is restarted, defeating the browser disk cache that otherwise serves a stale
# index.html (with the previous arena-mobile-overrides block).
_IFRAME_CACHE_BUST = str(int(time.time()))
from fastapi.responses import JSONResponse, RedirectResponse

from arena.config import (
    distribution_allows_paper_kis_credentials,
    distribution_allows_real_kis_credentials,
    distribution_uses_broker_credentials,
    normalize_distribution_mode,
)
from arena.agents.investment_chat.audit import append_chat_audit
from arena.agents.investment_chat.context import REQUEST_USER_EMAIL, normalize_tenant
from arena.agents.investment_chat.config_tools import (
    build_config_bridge_tool_entries,
    load_chat_agent_config,
    recent_config_drafts,
)
from arena.agents.investment_chat.credential_tools import (
    CREDENTIAL_CHANGE_APPLY_ACTION,
    load_credential_draft,
    recent_credential_drafts,
    save_credential_draft,
)
from arena.agents.investment_chat.drafts import approval_token, load_draft
from arena.agents.investment_chat.order_tools import build_order_bridge_tool_entries
from arena.agents.investment_chat.selection import (
    chat_model_routing_config,
    normalize_chat_model_selection,
    normalize_stored_advisor_model_selection,
    tenant_default_chat_selection,
)
from arena.providers.registry import canonical_provider, default_model_for_provider, list_adk_provider_specs
from arena.providers.model_discovery import (
    ModelDiscoveryError,
    discover_model_options_with_api_key,
    model_option_sets,
    save_model_options_catalog,
)
from arena.ui.account_onboarding import sync_account_snapshot_after_kis_save
from arena.ui.investment_chat_adk import _chat_app_name
from arena.ui.investment_chat_providers import tenant_available_provider_specs
from arena.ui.routes.viewer import ViewerRouteDeps
from arena.ui.templating import render_ui_template

_MODEL_KEY_HELP_URLS = {
    "gpt": "https://platform.openai.com/api-keys",
    "gemini": "https://aistudio.google.com/apikey",
    "claude": "https://console.anthropic.com/settings/keys",
}
_KIS_HELP_URL = "https://apiportal.koreainvestment.com"


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


def _model_key_provider_options(tenant_settings, current_provider: str = "") -> list[dict[str, Any]]:
    _ = tenant_settings
    current = canonical_provider(current_provider) or str(current_provider or "").strip().lower()
    options: list[dict[str, Any]] = []
    for spec in list_adk_provider_specs():
        if not spec.api_key_setting:
            continue
        options.append(
            {
                "value": spec.provider_id,
                "label": spec.label,
                "default_model": "",
                "default_cheap_model": "",
                "advisor_models": [],
                "aux_models": [],
                "selected": spec.provider_id == current,
            }
        )
    if options and not any(bool(item.get("selected")) for item in options):
        options[0]["selected"] = True
    return options


def _selected_model_key_option(options: list[dict[str, Any]]) -> dict[str, Any]:
    return next((item for item in options if item.get("selected")), options[0] if options else {})


def _merge_chat_model_routing(existing: dict, *, router_model: str, utility_model: str) -> dict:
    routing = dict(existing.get("model_routing") or {}) if isinstance(existing.get("model_routing"), dict) else {}
    routing["router_model"] = router_model
    routing["utility_model"] = utility_model
    return routing


def _upsert_provider_agent_config(
    repo,
    *,
    tenant: str,
    tenant_settings,
    provider: str,
    model: str,
    updated_by: str,
) -> None:
    getter = getattr(repo, "get_config", None)
    setter = getattr(repo, "set_config", None)
    if not callable(getter) or not callable(setter):
        return
    try:
        raw = getter(tenant, "agents_config")
    except Exception:
        raw = ""
    try:
        parsed = json.loads(str(raw or ""))
    except Exception:
        parsed = []
    entries = [dict(entry) for entry in parsed if isinstance(entry, dict)] if isinstance(parsed, list) else []
    provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
    target_id = provider_token
    matched = False
    for entry in entries:
        entry_provider = canonical_provider(entry.get("provider")) or str(entry.get("provider") or "").strip().lower()
        entry_id = str(entry.get("id") or "").strip().lower()
        if entry_provider == provider_token or entry_id == target_id:
            entry["id"] = entry_id or target_id
            entry["provider"] = provider_token
            entry["model"] = model
            if not entry.get("capital_krw"):
                entry["capital_krw"] = float(getattr(tenant_settings, "sleeve_capital_krw", 0.0) or 0.0)
            if not entry.get("target_market"):
                entry["target_market"] = str(getattr(tenant_settings, "kis_target_market", "") or "us").strip().lower()
            matched = True
            break
    if not matched:
        entries.append(
            {
                "id": target_id,
                "provider": provider_token,
                "model": model,
                "capital_krw": float(getattr(tenant_settings, "sleeve_capital_krw", 0.0) or 0.0),
                "target_market": str(getattr(tenant_settings, "kis_target_market", "") or "us").strip().lower(),
            }
        )
    setter(tenant, "agents_config", json.dumps(entries, ensure_ascii=False), updated_by=updated_by)


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
    orders = result.get("orders")
    if isinstance(orders, list) and orders:
        count = len(orders)
        submitted = int(result.get("submitted_count") or 0)
        if result.get("status") in {"submitted", "already_submitted"}:
            return f"방금 주문 {count}건 일괄 승인 결과를 확인해서 알려줘."
        if submitted:
            return f"방금 주문 {count}건 중 {submitted}건 승인 결과를 확인해서 알려줘."
        return "방금 주문 일괄 승인이 실패했어. 실패 이유를 확인해서 알려줘."
    intent = result.get("intent") if isinstance(result.get("intent"), dict) else {}
    ticker = str(intent.get("ticker") or "").strip().upper()
    side = str(intent.get("side") or "").strip().upper()
    quantity = _chat_quantity(intent.get("quantity"))
    order_bits = [token for token in [ticker, side, f"{quantity}주" if quantity else ""] if token]
    if order_bits:
        return f"방금 {' '.join(order_bits)} 주문 승인 결과를 확인해서 알려줘."
    return "방금 주문 승인 결과를 확인해서 알려줘."


def _parse_utc_datetime(value: str):
    token = str(value or "").strip()
    if not token:
        return None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _credential_result_chat_delivery_text(result: dict) -> str:
    action = str(result.get("action") or "").strip().lower()
    kind = str(result.get("credential_kind") or "").strip().lower()
    provider = str(result.get("provider_label") or result.get("provider") or "").strip()
    if result.get("status") in {"applied", "deleted"}:
        verb = "삭제" if action == "delete" else "저장"
        if kind == "kis_account":
            env = "모의투자" if str(result.get("env") or "").strip().lower() == "demo" else "실전투자"
            return f"방금 KIS {env} API key {verb} 결과를 확인해서 알려줘."
        return f"방금 {provider} LLM API key {verb} 결과를 확인해서 알려줘."
    if kind == "kis_account":
        return "방금 KIS API key 변경이 실패했어. 실패 이유를 확인해서 알려줘."
    return f"방금 {provider} LLM API key 변경이 실패했어. 실패 이유를 확인해서 알려줘."


def _normalize_kis_env(value: str | None, fallback: str = "demo") -> str:
    token = str(value or "").strip().lower()
    if token in {"", "demo", "paper", "mock", "vps", "virtual", "sandbox", "모의", "모의투자"}:
        return "demo" if not token else "demo"
    if token in {"real", "live", "prod", "production", "실전", "실전투자"}:
        return "real"
    fallback_token = str(fallback or "").strip().lower()
    return "real" if fallback_token == "real" else "demo"


def _split_kis_account_no(value: str, prdt_cd: str = "01") -> tuple[str, str]:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    product = "".join(ch for ch in str(prdt_cd or "") if ch.isdigit())[:2] or "01"
    if len(digits) < 8:
        return "", product
    return digits[:8], digits[8:10] or product


def _kis_account_key(account: dict) -> tuple[str, str]:
    cano, prdt_cd = _split_kis_account_no(
        str(account.get("account_no") or account.get("cano") or ""),
        str(account.get("prdt_cd") or "01"),
    )
    return cano, prdt_cd


def _existing_kis_account_rows(credential_store, *, tenant_id: str) -> list[dict[str, str]]:
    loader = getattr(credential_store, "list_kis_accounts_meta", None)
    if not callable(loader):
        return []
    try:
        raw_rows = loader(tenant_id=tenant_id) or []
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    for row in raw_rows:
        if not isinstance(row, dict):
            continue
        cano, prdt_cd = _kis_account_key(row)
        if not cano:
            continue
        rows.append(
            {
                "env": _normalize_kis_env(str(row.get("env") or "demo")),
                "cano": cano,
                "prdt_cd": prdt_cd,
            }
        )
    return rows


def _recent_order_drafts(repo, *, tenant_id: str, limit: int = 20) -> list[dict]:
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
                "created_at": draft.get("created_at") or "",
                "expires_at": draft.get("expires_at") or "",
                "notional_krw": draft.get("notional_krw") or 0,
                "scope": draft.get("scope") or "account",
                "target_agent_id": draft.get("target_agent_id") or "",
                "batch_token": draft.get("batch_token") or "",
                "batch_index": draft.get("batch_index"),
                "batch_count": draft.get("batch_count"),
                "intent": intent,
                "risk": risk,
            }
        )
        if len(out) >= max(1, min(int(limit or 5), 20)):
            break
    return out


def register_investment_chat_routes(app: FastAPI, *, deps: ViewerRouteDeps) -> None:
    credential_store = deps.credential_store
    credential_store_error = deps.credential_store_error

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
        model_key_options = _model_key_provider_options(tenant_settings, provider_token)
        selected_key_option = _selected_model_key_option(model_key_options)
        selected_key_provider = str(selected_key_option.get("value") or "")
        selected_key_model = (
            model_token
            if selected_key_provider == provider_token
            else str(selected_key_option.get("default_model") or "")
        )
        if not selected_key_model:
            selected_key_model = str(selected_key_option.get("default_model") or "")
        selected_key_cheap_model = str(selected_key_option.get("default_cheap_model") or selected_key_model)
        if selected_key_provider == provider_token:
            routing_config = chat_model_routing_config(chat_config)
            selected_key_cheap_model = str(routing_config.get("router_model") or "").strip() or selected_key_cheap_model
        body = render_ui_template(
            "investment_chat_body.jinja2",
            iframe_src=_adk_iframe_src(tenant, provider_token, model_token) if provider_token and model_token else "",
            chat_session_app_name=chat_session_app_name,
            chat_session_user_id=chat_session_user_id,
            tenant=html.escape(tenant),
            llm_key_provider_options=model_key_options,
            llm_key_provider=selected_key_provider,
            llm_key_model=selected_key_model,
            llm_key_cheap_model=selected_key_cheap_model,
            llm_key_model_presets={str(item["value"]): list(item.get("advisor_models") or []) for item in model_key_options},
            llm_key_cheap_model_presets={str(item["value"]): list(item.get("aux_models") or []) for item in model_key_options},
            llm_key_help_urls={
                str(item["value"]): _MODEL_KEY_HELP_URLS.get(str(item["value"]), "")
                for item in model_key_options
            },
            kis_key_help_url=_KIS_HELP_URL,
            llm_key_store_available=credential_store is not None,
            llm_key_store_error=str(credential_store_error or ""),
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

    @app.post("/investment-chat/model-key/options")
    def investment_chat_model_key_options(
        request: Request,
        tenant_id: str = Form(default="local"),
        provider: str = Form(default=""),
        api_key: str = Form(default=""),
    ):
        tenant, _agent_ids, user, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=_next_path(tenant_id),
        )
        if redirect is not None:
            return JSONResponse({"error": "authentication required"}, status_code=401)
        tenant_settings = deps.settings_for_tenant(tenant)
        allowed = {str(item["value"]) for item in _model_key_provider_options(tenant_settings, provider)}
        provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
        if provider_token not in allowed:
            return JSONResponse({"error": "invalid provider"}, status_code=400)
        api_key_token = str(api_key or "").strip()
        if not api_key_token:
            return JSONResponse({"error": "api_key is required"}, status_code=400)
        try:
            options = discover_model_options_with_api_key(provider_token, api_key_token)
        except ModelDiscoveryError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        updated_by = str((user or {}).get("email") or "").strip() or "investment_chat"
        save_model_options_catalog(deps.repo, tenant_id=tenant, options=options, updated_by=updated_by)
        return JSONResponse({"status": "ok", **options})

    @app.post("/investment-chat/model-key")
    def investment_chat_model_key(
        request: Request,
        tenant_id: str = Form(default="local"),
        provider: str = Form(default=""),
        model: str = Form(default=""),
        cheap_model: str = Form(default=""),
        api_key: str = Form(default=""),
    ):
        tenant, _agent_ids, user, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=_next_path(tenant_id),
        )
        if redirect is not None:
            return redirect
        if credential_store is None:
            return JSONResponse(
                {"error": f"credential store unavailable: {credential_store_error or 'unknown error'}"},
                status_code=503,
            )

        tenant_settings = deps.settings_for_tenant(tenant)
        allowed = {str(item["value"]) for item in _model_key_provider_options(tenant_settings, provider)}
        provider_token = canonical_provider(provider) or str(provider or "").strip().lower()
        if provider_token not in allowed:
            return JSONResponse({"error": "invalid provider"}, status_code=400)

        api_key_token = str(api_key or "").strip()
        if not api_key_token:
            return JSONResponse({"error": "api_key is required"}, status_code=400)

        try:
            options = discover_model_options_with_api_key(provider_token, api_key_token)
        except ModelDiscoveryError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        advisor_options, router_options, utility_options = model_option_sets(options)
        model_token = normalize_chat_model_selection(provider_token, model)
        if not model_token or model_token not in advisor_options:
            return JSONResponse({"error": "model is required"}, status_code=400)
        cheap_model_token = normalize_chat_model_selection(provider_token, cheap_model)
        if not cheap_model_token or cheap_model_token not in router_options or cheap_model_token not in utility_options:
            return JSONResponse({"error": "cheap_model is required"}, status_code=400)

        user_email = str((user or {}).get("email") or "").strip()
        updated_by = user_email or "investment_chat"
        credential_store.save_model_keys(
            tenant_id=tenant,
            updated_by=updated_by,
            providers={provider_token: {"api_key": api_key_token, "model": model_token}},
        )

        existing = load_chat_agent_config(deps.repo, tenant_id=tenant)
        merged = dict(existing)
        merged["provider"] = provider_token
        merged["model"] = model_token
        merged["model_routing"] = _merge_chat_model_routing(
            merged,
            router_model=cheap_model_token,
            utility_model=cheap_model_token,
        )
        deps.repo.set_config(
            tenant,
            "investment_chat_config",
            json.dumps(merged, ensure_ascii=False),
            updated_by=updated_by,
        )
        save_model_options_catalog(deps.repo, tenant_id=tenant, options=options, updated_by=updated_by)
        _upsert_provider_agent_config(
            deps.repo,
            tenant=tenant,
            tenant_settings=tenant_settings,
            provider=provider_token,
            model=model_token,
            updated_by=updated_by,
        )
        try:
            deps.invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")
        except TypeError:
            deps.invalidate_tenant_cache(tenant)

        query = urlencode({"tenant_id": tenant, "provider": provider_token, "model": model_token})
        return RedirectResponse(url=f"/investment-chat?{query}", status_code=303)

    @app.get("/investment-chat/order-drafts")
    def investment_chat_order_drafts(request: Request, tenant_id: str = "", limit: int = 20):
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

    @app.get("/investment-chat/credential-drafts")
    def investment_chat_credential_drafts(request: Request, tenant_id: str = "", limit: int = 5):
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
            "drafts": recent_credential_drafts(deps.repo, tenant_id=tenant, limit=limit),
        }

    @app.post("/investment-chat/credential-drafts/{approval_token}/apply")
    def investment_chat_apply_credential_draft(
        request: Request,
        approval_token: str,
        tenant_id: str = Form(default="local"),
        api_key: str = Form(default=""),
        model: str = Form(default=""),
        cheap_model: str = Form(default=""),
        kis_env: str = Form(default=""),
        kis_account_no: str = Form(default=""),
        kis_app_key: str = Form(default=""),
        kis_app_secret: str = Form(default=""),
        kis_paper_app_key: str = Form(default=""),
        kis_paper_app_secret: str = Form(default=""),
        confirmed: str = Form(default=""),
    ):
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
        if credential_store is None:
            return JSONResponse(
                {"status": "error", "error": f"credential store unavailable: {credential_store_error or 'unknown error'}"},
                status_code=503,
            )
        draft = load_credential_draft(deps.repo, tenant_id=tenant, token=token)
        if not isinstance(draft, dict):
            return JSONResponse({"status": "missing", "approval_token": token, "error": "approval token not found"}, status_code=404)
        if str(draft.get("status") or "").strip().lower() not in {"", "draft"}:
            return JSONResponse(
                {"status": draft.get("status") or "blocked", "approval_token": token, "error": "draft is not pending"},
                status_code=409,
            )
        expires_at = _parse_utc_datetime(str(draft.get("expires_at") or ""))
        if expires_at is not None and expires_at < datetime.now(timezone.utc):
            draft["status"] = "expired"
            save_credential_draft(deps.repo, tenant_id=tenant, token=token, draft=draft)
            return JSONResponse({"status": "expired", "approval_token": token, "error": "approval token expired"}, status_code=409)

        tenant_settings = deps.settings_for_tenant(tenant)
        action = str(draft.get("action") or "").strip().lower()
        if action not in {"upsert", "delete"}:
            return JSONResponse({"status": "blocked", "approval_token": token, "error": "invalid credential action"}, status_code=400)

        user_email = str((user or {}).get("email") or "").strip().lower()
        updated_by = user_email or "investment_chat"
        credential_kind = str(draft.get("credential_kind") or "").strip().lower()
        if credential_kind not in {"", "model_key", "kis_account"}:
            return JSONResponse({"status": "blocked", "approval_token": token, "error": "invalid credential kind"}, status_code=400)
        if not credential_kind:
            credential_kind = "model_key" if str(draft.get("provider") or "").strip() else "kis_account"

        provider_token = ""
        provider_label = ""
        model_token = ""
        env_token = ""
        if credential_kind == "kis_account":
            distribution_mode = normalize_distribution_mode(getattr(tenant_settings, "distribution_mode", "private"))
            uses_broker_credentials = distribution_uses_broker_credentials(tenant_settings) or distribution_mode == "simulated_only"
            allow_real_kis_credentials = distribution_allows_real_kis_credentials(tenant_settings)
            allow_paper_kis_credentials = (
                distribution_allows_paper_kis_credentials(tenant_settings) or distribution_mode == "simulated_only"
            )
            if not uses_broker_credentials:
                return JSONResponse(
                    {"status": "blocked", "approval_token": token, "error": "tenant does not use KIS credentials"},
                    status_code=400,
                )
            env_token = _normalize_kis_env(kis_env, str(draft.get("env") or "demo"))
            if env_token == "real" and not allow_real_kis_credentials:
                return JSONResponse(
                    {"status": "blocked", "approval_token": token, "error": "tenant is not approved for real KIS credentials"},
                    status_code=400,
                )
            if env_token == "demo" and not allow_paper_kis_credentials:
                return JSONResponse(
                    {"status": "blocked", "approval_token": token, "error": "tenant is not approved for demo KIS credentials"},
                    status_code=400,
                )

            cano, prdt_cd = _split_kis_account_no(kis_account_no)
            if not cano:
                return JSONResponse({"status": "blocked", "approval_token": token, "error": "kis_account_no is required"}, status_code=400)

            existing_accounts = _existing_kis_account_rows(credential_store, tenant_id=tenant)
            remaining_accounts = [
                row for row in existing_accounts
                if _kis_account_key(row) != (cano, prdt_cd)
            ]
            if action == "delete":
                confirmed_token = str(confirmed or "").strip().lower()
                if confirmed_token not in {"1", "true", "yes", "on", "confirmed"}:
                    return JSONResponse({"status": "blocked", "approval_token": token, "error": "confirmation is required"}, status_code=400)
                credential_store.save_kis_accounts(
                    tenant_id=tenant,
                    updated_by=updated_by,
                    accounts=remaining_accounts,
                    notes="investment chat KIS credential deletion",
                )
                if distribution_mode == "paper_only" and not remaining_accounts:
                    deps.repo.set_config(tenant, "distribution_mode", "simulated_only", updated_by=updated_by)
                draft["status"] = "deleted"
            else:
                app_key_token = str(kis_app_key or "").strip()
                app_secret_token = str(kis_app_secret or "").strip()
                paper_app_key_token = str(kis_paper_app_key or "").strip()
                paper_app_secret_token = str(kis_paper_app_secret or "").strip()
                if env_token == "real" and (not app_key_token or not app_secret_token):
                    return JSONResponse(
                        {"status": "blocked", "approval_token": token, "error": "kis_app_key and kis_app_secret are required"},
                        status_code=400,
                    )
                if env_token == "demo" and (not paper_app_key_token or not paper_app_secret_token):
                    return JSONResponse(
                        {
                            "status": "blocked",
                            "approval_token": token,
                            "error": "kis_paper_app_key and kis_paper_app_secret are required",
                        },
                        status_code=400,
                    )
                account_row = {
                    "env": env_token,
                    "account_no": f"{cano}{prdt_cd}",
                    "app_key": app_key_token,
                    "app_secret": app_secret_token,
                    "paper_app_key": paper_app_key_token,
                    "paper_app_secret": paper_app_secret_token,
                }
                credential_store.save_kis_accounts(
                    tenant_id=tenant,
                    updated_by=updated_by,
                    accounts=[*remaining_accounts, account_row],
                    notes="investment chat KIS credential update",
                )
                if distribution_mode == "simulated_only":
                    deps.repo.set_config(tenant, "distribution_mode", "paper_only", updated_by=updated_by)
                try:
                    deps.invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")
                except TypeError:
                    deps.invalidate_tenant_cache(tenant)
                draft["account_snapshot_sync"] = sync_account_snapshot_after_kis_save(
                    repo=deps.repo,
                    settings=tenant_settings,
                    tenant_id=tenant,
                    settings_for_tenant=deps.settings_for_tenant,
                    updated_by=updated_by,
                )
                draft["status"] = "applied"
            reload_url = _next_path(tenant)
        else:
            allowed = {str(item["value"]) for item in _model_key_provider_options(tenant_settings, draft.get("provider"))}
            provider_token = canonical_provider(draft.get("provider")) or str(draft.get("provider") or "").strip().lower()
            if provider_token not in allowed:
                return JSONResponse({"status": "blocked", "approval_token": token, "error": "invalid provider"}, status_code=400)
            provider_label = str(draft.get("provider_label") or provider_token)
            model_token = normalize_chat_model_selection(provider_token, model or draft.get("model"))
            cheap_model_token = normalize_chat_model_selection(
                provider_token,
                cheap_model or draft.get("cheap_model") or draft.get("router_model"),
            )
            if action == "delete":
                confirmed_token = str(confirmed or "").strip().lower()
                if confirmed_token not in {"1", "true", "yes", "on", "confirmed"}:
                    return JSONResponse({"status": "blocked", "approval_token": token, "error": "confirmation is required"}, status_code=400)
                remover = getattr(credential_store, "remove_model_key", None)
                if not callable(remover):
                    return JSONResponse({"status": "error", "approval_token": token, "error": "credential store cannot delete model keys"}, status_code=500)
                remover(tenant_id=tenant, updated_by=updated_by, provider=provider_token)
                draft["status"] = "deleted"
                reload_url = _next_path(tenant)
            else:
                api_key_token = str(api_key or "").strip()
                if not api_key_token:
                    return JSONResponse({"status": "blocked", "approval_token": token, "error": "api_key is required"}, status_code=400)
                try:
                    options = discover_model_options_with_api_key(provider_token, api_key_token)
                except ModelDiscoveryError as exc:
                    return JSONResponse(
                        {"status": "blocked", "approval_token": token, "error": str(exc)},
                        status_code=400,
                    )
                advisor_options, router_options, utility_options = model_option_sets(options)
                if not model_token or model_token not in advisor_options:
                    return JSONResponse({"status": "blocked", "approval_token": token, "error": "model is required"}, status_code=400)
                if not cheap_model_token or cheap_model_token not in router_options or cheap_model_token not in utility_options:
                    return JSONResponse(
                        {"status": "blocked", "approval_token": token, "error": "cheap_model is required"},
                        status_code=400,
                    )
                credential_store.save_model_keys(
                    tenant_id=tenant,
                    updated_by=updated_by,
                    providers={provider_token: {"api_key": api_key_token, "model": model_token}},
                )
                existing = load_chat_agent_config(deps.repo, tenant_id=tenant)
                merged = dict(existing)
                merged["provider"] = provider_token
                merged["model"] = model_token
                merged["model_routing"] = _merge_chat_model_routing(
                    merged,
                    router_model=cheap_model_token,
                    utility_model=cheap_model_token,
                )
                deps.repo.set_config(
                    tenant,
                    "investment_chat_config",
                    json.dumps(merged, ensure_ascii=False),
                    updated_by=updated_by,
                )
                save_model_options_catalog(deps.repo, tenant_id=tenant, options=options, updated_by=updated_by)
                _upsert_provider_agent_config(
                    deps.repo,
                    tenant=tenant,
                    tenant_settings=tenant_settings,
                    provider=provider_token,
                    model=model_token,
                    updated_by=updated_by,
                )
                draft["status"] = "applied"
                reload_url = _next_path(tenant, provider_token, model_token)

        draft["applied_at"] = datetime.now(timezone.utc).isoformat()
        save_credential_draft(deps.repo, tenant_id=tenant, token=token, draft=draft, updated_by=updated_by)
        try:
            deps.invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")
        except TypeError:
            deps.invalidate_tenant_cache(tenant)
        result = {
            "status": draft["status"],
            "approval_token": token,
            "tenant_id": tenant,
            "action": action,
            "credential_kind": credential_kind,
            "provider": provider_token,
            "provider_label": provider_label or draft.get("provider_label") or provider_token,
            "model": model_token,
            "cheap_model": cheap_model_token if credential_kind == "model_key" else "",
            "env": env_token,
            "reload_url": reload_url,
        }
        result["chat_delivery_text"] = _credential_result_chat_delivery_text(result)
        append_chat_audit(
            deps.repo,
            tenant_id=tenant,
            action=CREDENTIAL_CHANGE_APPLY_ACTION,
            status=draft["status"],
            detail={
                "approval_token": token,
                "credential_kind": credential_kind,
                "action": action,
                "provider": provider_token,
                "model": model_token,
                "cheap_model": cheap_model_token if credential_kind == "model_key" else "",
                "env": env_token,
            },
            user_email=user_email,
        )
        return JSONResponse(result)

    @app.post("/investment-chat/order-drafts/batch-submit")
    async def investment_chat_submit_order_draft_batch(
        request: Request,
        tenant_id: str = "",
    ):
        tenant, _agent_ids, user, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=_next_path(tenant_id),
        )
        if redirect is not None:
            return JSONResponse({"status": "blocked", "error": "authentication required"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        body = payload if isinstance(payload, dict) else {}
        raw_tokens = body.get("approval_tokens")
        if not isinstance(raw_tokens, list):
            return JSONResponse({"status": "blocked", "error": "approval_tokens must be a list"}, status_code=400)
        tokens: list[str] = []
        seen: set[str] = set()
        for raw_token in raw_tokens:
            token = str(raw_token or "").strip()
            if token and token not in seen:
                seen.add(token)
                tokens.append(token)
        if not tokens:
            return JSONResponse({"status": "blocked", "error": "approval_tokens is required"}, status_code=400)
        settings = deps.settings_for_tenant(tenant)
        bridge_entries = build_order_bridge_tool_entries(repo=deps.repo, settings=settings, tenant_id=tenant)
        entry = next((item for item in bridge_entries if item.name == "submit_approved_order_batch"), None)
        if entry is None or not callable(entry.callable):
            return JSONResponse({"status": "error", "error": "order batch approval bridge unavailable"}, status_code=500)
        user_email = str((user or {}).get("email") or "").strip().lower()
        batch_token = str(body.get("batch_token") or "").strip()
        email_token = REQUEST_USER_EMAIL.set(user_email)
        try:
            result = entry.callable(
                approval_tokens=tokens,
                confirmation_text=f"CONFIRM_BATCH {approval_token({'approval_tokens': tokens})}",
            )
        finally:
            REQUEST_USER_EMAIL.reset(email_token)
        if isinstance(result, dict):
            result = dict(result)
            if batch_token:
                result["batch_token"] = batch_token
            result["chat_delivery_text"] = _order_result_chat_delivery_text(result)
        return result

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
