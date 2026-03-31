from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from arena.config import (
    Settings,
    distribution_allows_paper_kis_credentials,
    distribution_allows_real_kis_credentials,
    distribution_uses_broker_credentials,
)
from arena.providers import default_model_for_provider, list_adk_provider_specs
from arena.ui.memory import build_memory_settings_panel
from arena.ui.routes.settings_render import (
    build_agents_panel,
    build_credentials_panel,
    build_mcp_panel,
    build_save_progress_script,
    build_tab_script,
)
from arena.ui.routes.settings_render_capital import build_capital_panel
from arena.ui.templating import render_ui_template


@dataclass(frozen=True)
class SettingsPageRouteDeps:
    repo: Any
    settings: Settings
    settings_enabled: bool
    auth_enabled: bool
    credential_store: Any | None
    credential_store_error: str | None
    executor: Any
    cached_fetch: Callable[..., Any]
    current_admin_view_model: Callable[..., dict[str, Any]]
    current_user: Callable[[Request], dict[str, Any] | None]
    fmt_ts: Callable[[object], str]
    html_response: Callable[..., HTMLResponse]
    provider_api_key_help_html: Callable[[str], str]
    resolve_admin_context: Callable[..., Any]
    is_live_mode: Callable[[Settings | None], bool]
    invalidate_tenant_runtime_cache: Callable[[str], None]
    settings_for_tenant: Callable[[str], Settings]
    tailwind_layout: Callable[..., str]
    tenant_access_denied: Callable[[str, str], bool]


def register_settings_page_routes(app: FastAPI, *, deps: SettingsPageRouteDeps) -> None:
    repo = deps.repo
    settings = deps.settings
    settings_enabled = deps.settings_enabled
    auth_enabled = deps.auth_enabled
    credential_store = deps.credential_store
    credential_store_error = deps.credential_store_error
    _executor = deps.executor
    _cached_fetch = deps.cached_fetch
    _current_admin_view_model = deps.current_admin_view_model
    _current_user = deps.current_user
    _fmt_ts = deps.fmt_ts
    _html_response = deps.html_response
    _provider_api_key_help_html = deps.provider_api_key_help_html
    _resolve_admin_context = deps.resolve_admin_context
    _is_live_mode = deps.is_live_mode
    _invalidate_tenant_runtime_cache = deps.invalidate_tenant_runtime_cache
    _settings_for_tenant = deps.settings_for_tenant
    _tailwind_layout = deps.tailwind_layout
    _tenant_access_denied = deps.tenant_access_denied

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
        ok: str = Query(default="", description="save status"),
        msg: str = Query(default="", description="message"),
        tab: str = Query(default="credentials", description="settings tab"),
    ) -> HTMLResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        user, user_email, tenant, _allowed_tenants, redirect = _resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path="/settings",
        )
        if redirect:
            return redirect

        # --- Parallel BQ fetches for settings page performance ---
        _fut_latest = _executor.submit(
            _cached_fetch,
            f"settings:latest_creds:{tenant}",
            repo.latest_runtime_credentials,
            tenant_id=tenant,
        )
        _fut_recent = _executor.submit(
            _cached_fetch,
            f"settings:recent_creds:{tenant}",
            repo.recent_runtime_credentials,
            limit=32,
        )
        latest = _fut_latest.result() or {}
        all_recent = _fut_recent.result() or []
        history = [
            row
            for row in all_recent
            if str(row.get("tenant_id") or "").strip().lower() == tenant
        ][:8]
        admin_vm = _current_admin_view_model(tenant, _latest_creds=latest)
        allowed_tabs = {"credentials", "agents", "mcp", "memory", "capital"}
        initial_tab = str(tab or "agents").strip().lower()
        if initial_tab == "credentials":
            initial_tab = "agents"
        if initial_tab not in allowed_tabs:
            initial_tab = "agents"

        popup_text = ""
        popup_level = "success"
        if ok:
            popup_text = msg or ("Saved" if ok == "1" else "Failed")
            popup_level = "success" if ok == "1" else "error"

        rows = "".join(
            (
                "<tr class='border-b border-ink-100/50 hover:bg-ink-100/40 transition-colors'>"
                f"<td class='px-3 py-2 text-xs text-ink-600'>{_fmt_ts(r.get('updated_at'))}</td>"
                f"<td class='px-3 py-2 text-sm'>{html.escape(str(r.get('tenant_id') or ''))}</td>"
                f"<td class='px-3 py-2 text-sm'>{html.escape(str(r.get('kis_secret_name') or '-'))}</td>"
                f"<td class='px-3 py-2 text-sm'>{html.escape(str(r.get('model_secret_name') or '-'))}</td>"
                f"<td class='px-3 py-2 text-sm'>{html.escape(str(r.get('kis_account_no_masked') or '-'))}</td>"
                f"<td class='px-3 py-2 text-sm'>{'Y' if bool(r.get('has_openai')) else '-'}</td>"
                f"<td class='px-3 py-2 text-sm'>{'Y' if bool(r.get('has_gemini')) else '-'}</td>"
                f"<td class='px-3 py-2 text-sm'>{'Y' if bool(r.get('has_anthropic')) else '-'}</td>"
                "</tr>"
            )
            for r in history
        )

        tool_entries = admin_vm["tool_entries"]
        risk = admin_vm["risk"]
        prompt_text = str(admin_vm["prompt_text"] or "")
        tenant_settings = admin_vm.get("tenant_settings", settings)
        distribution_mode = str(getattr(tenant_settings, "distribution_mode", "private") or "private").strip().lower()
        allow_real_kis_credentials = distribution_allows_real_kis_credentials(tenant_settings)
        allow_paper_kis_credentials = distribution_allows_paper_kis_credentials(tenant_settings)
        uses_broker_credentials = distribution_uses_broker_credentials(tenant_settings)
        paper_onboarding_enabled = distribution_mode == "simulated_only"
        allow_paper_kis_credentials_ui = allow_paper_kis_credentials or paper_onboarding_enabled
        uses_broker_credentials_ui = uses_broker_credentials or paper_onboarding_enabled
        if distribution_mode == "paper_only":
            credentials_mode_note = "공개용 준비 모드: 실전 키는 숨기고 모의투자 키만 노출합니다."
        elif distribution_mode == "simulated_only":
            credentials_mode_note = "초기 온보딩 모드: 지금은 simulated로 바로 사용할 수 있고, 아래에 KIS demo 키를 저장하면 tenant가 paper 모드로 전환됩니다."
        else:
            credentials_mode_note = ""

        # --- Dynamic KIS account rows ---
        kis_meta = admin_vm.get("kis_accounts_meta") or []
        active_kis_account_no = str(admin_vm.get("active_kis_account_no") or "").strip()
        active_kis_account_no_masked = str(admin_vm.get("active_kis_account_no_masked") or "").strip()
        credentials_parts = build_credentials_panel(
            tenant=tenant,
            credentials_mode_note=credentials_mode_note,
            active_kis_account_no=active_kis_account_no,
            active_kis_account_no_masked=active_kis_account_no_masked,
            kis_meta=kis_meta,
            allow_real_kis_credentials=allow_real_kis_credentials,
            allow_paper_kis_credentials=allow_paper_kis_credentials_ui,
            uses_broker_credentials=uses_broker_credentials_ui,
            rows_html=rows,
        )
        credentials_panel = credentials_parts.panel_html
        kis_env_options_html = credentials_parts.kis_env_options_html
        kis_template_real_fields = credentials_parts.kis_template_real_fields
        kis_template_paper_fields = credentials_parts.kis_template_paper_fields

        # --- Unified Agents panel (agents + models + capital + API keys + provider + per-agent settings) ---
        agents_cfg = admin_vm["agents_config"]
        api_status = admin_vm.get("api_key_status", {})
        tenant_settings = admin_vm.get("tenant_settings", settings)
        provider_specs = list_adk_provider_specs()
        provider_options_html = "".join(
            f'<option value="{html.escape(spec.provider_id)}">{html.escape(spec.provider_id)}</option>'
            for spec in provider_specs
        )
        provider_key_help = {
            spec.provider_id: _provider_api_key_help_html(spec.provider_id)
            for spec in provider_specs
        }
        default_models = {
            spec.provider_id: default_model_for_provider(tenant_settings, spec.provider_id)
            for spec in provider_specs
        }

        # Build configurable tool list for catalog + per-agent toggles
        configurable_tools = [
            {
                "tool_id": str(e.get("tool_id") or ""),
                "category": str(e.get("category") or ""),
                "tier": str(e.get("tier") or ""),
                "description": str(e.get("description") or ""),
                "description_ko": str(e.get("description_ko") or e.get("description") or ""),
                "label_ko": str(e.get("label_ko") or e.get("tool_id") or ""),
                "params": e.get("params") or [],
                "source": e.get("source") or "",
            }
            for e in tool_entries if e.get("configurable")
        ]

        # Risk field definitions for per-agent risk overrides
        risk_fields_meta = [
            ("max_order_krw", "최대 주문 금액", risk["max_order_krw"]),
            ("max_daily_turnover_ratio", "일일 최대 회전율", risk["max_daily_turnover_ratio"]),
            ("max_position_ratio", "최대 종목 비중", risk["max_position_ratio"]),
            ("min_cash_buffer_ratio", "최소 현금 비율", risk["min_cash_buffer_ratio"]),
            ("ticker_cooldown_seconds", "종목 쿨다운", risk["ticker_cooldown_seconds"]),
            ("max_daily_orders", "일일 최대 주문 수", risk["max_daily_orders"]),
            ("estimated_fee_bps", "예상 수수료 (bps)", risk["estimated_fee_bps"]),
        ]

        agents_panel = build_agents_panel(
            agents_cfg=agents_cfg,
            api_status=api_status,
            research_status=admin_vm.get("research_status", {}),
            tenant_settings=tenant_settings,
            prompt_text=prompt_text,
            provider_options_html=provider_options_html,
            provider_key_help=provider_key_help,
            default_models=default_models,
            configurable_tools=configurable_tools,
            risk_fields_meta=risk_fields_meta,
            tenant=tenant,
            user_email=user_email,
            provider_api_key_help_html=_provider_api_key_help_html,
            is_live_mode=_is_live_mode,
            default_capital_krw=int(admin_vm["sleeve_capital_krw"]),
        )

        mcp_panel = build_mcp_panel(
            tenant=tenant,
            mcp_servers=admin_vm["mcp_servers"],
            agents_cfg=agents_cfg,
            configurable_tools=configurable_tools,
        )
        memory_panel = build_memory_settings_panel(
            repo,
            settings,
            tenant_id=tenant,
            cached_fetch=_cached_fetch,
        )

        # --- Capital panel (ECharts visualisations) ---
        capital_agent_ids = [
            str(a.get("agent_id") or "").strip().lower()
            for a in agents_cfg
            if str(a.get("agent_id") or "").strip()
        ] or [str(a).strip().lower() for a in tenant_settings.agent_ids if str(a).strip()]
        capital_per_agent = {
            str(a.get("agent_id") or a.get("id") or "").strip().lower(): int(a.get("capital_krw") or admin_vm["sleeve_capital_krw"])
            for a in agents_cfg
            if str(a.get("agent_id") or a.get("id") or "").strip()
        }
        capital_panel_core = build_capital_panel(
            tenant=tenant,
            agent_ids=capital_agent_ids,
            sleeve_capital_krw=int(admin_vm["sleeve_capital_krw"]),
            agent_capitals=capital_per_agent,
            user_email=user_email or "",
            is_live=_is_live_mode(tenant_settings),
        )
        # Prepend KIS Accounts section to capital panel (after opening div)
        kis_html = credentials_parts.kis_section_html
        marker = 'class="hidden space-y-5">'
        idx = capital_panel_core.find(marker)
        if idx >= 0:
            insert_at = idx + len(marker)
            capital_panel = capital_panel_core[:insert_at] + kis_html + capital_panel_core[insert_at:]
        else:
            capital_panel = capital_panel_core + kis_html

        tab_script = build_tab_script(
            initial_tab=initial_tab,
            uses_broker_credentials=uses_broker_credentials,
            kis_env_options_html=kis_env_options_html,
            kis_template_real_fields=kis_template_real_fields,
            kis_template_paper_fields=kis_template_paper_fields,
        )

        save_progress_script = build_save_progress_script()

        # Map tab → sidebar active key and Korean page title
        _TAB_TO_ACTIVE = {"agents": "agents", "capital": "capital", "mcp": "tools", "memory": "memory"}
        _TAB_TO_TITLE = {
            "agents": "\uc5d0\uc774\uc804\ud2b8",
            "capital": "\uc790\ubcf8\uad00\ub9ac",
            "mcp": "\ub3c4\uad6c\uad00\ub9ac",
            "memory": "\uae30\uc5b5\uad00\ub9ac",
        }
        sidebar_active = _TAB_TO_ACTIVE.get(initial_tab, "agents")
        page_title = _TAB_TO_TITLE.get(initial_tab, "\uc5d0\uc774\uc804\ud2b8")

        body = render_ui_template(
            "settings_body.jinja2",
            popup_text=popup_text,
            popup_level=popup_level,
            credential_store_error=str(credential_store_error or "").strip() if credential_store is None else "",
            auth_enabled=auth_enabled,
            tenant=tenant,
            initial_tab=initial_tab,
            credentials_panel=credentials_panel,
            agents_panel=agents_panel,
            capital_panel=capital_panel,
            mcp_panel=mcp_panel,
            memory_panel=memory_panel,
            tab_script=tab_script,
            save_progress_script=save_progress_script,
        )
        return _html_response(_tailwind_layout(page_title, body, active=sidebar_active, needs_charts=True, needs_echarts=True, tenant=tenant, user=_current_user(request)), max_age=0)

    @app.post("/settings/save", response_class=HTMLResponse)
    def settings_save(
        request: Request,
        tenant_id: str = Form(default="local"),
        updated_by: str = Form(default="ui-admin"),
        kis_accounts_json: str = Form(default="[]"),
        notes: str = Form(default=""),
    ) -> HTMLResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _, user_email, tenant, _, redirect = _resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/settings?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        if _tenant_access_denied(tenant_id, tenant):
            repo.append_runtime_audit_log(
                action="settings_save",
                status="denied",
                user_email=user_email,
                tenant_id=tenant,
                detail={"reason": "tenant_not_allowed"},
            )
            return settings_page(request=request, tenant_id=tenant, ok="0", msg="tenant access denied")

        if credential_store is None:
            body = render_ui_template(
                "inline_notice.jinja2",
                container_classes="mt-6 reveal rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800",
                message=f"Secret Manager init failed: {html.escape(credential_store_error or 'unknown error')}",
            )
            return _html_response(_tailwind_layout("Settings", body, active="settings", tenant=tenant), max_age=0)
        try:
            accounts_raw = json.loads(kis_accounts_json) if kis_accounts_json.strip() else []
            if not isinstance(accounts_raw, list):
                accounts_raw = []
            accounts = [a for a in accounts_raw if isinstance(a, dict)]
            tenant_settings = _settings_for_tenant(tenant)
            distribution_mode = str(getattr(tenant_settings, "distribution_mode", "private") or "private").strip().lower()
            allow_real_kis_credentials = distribution_allows_real_kis_credentials(tenant_settings)
            allow_paper_kis_credentials = distribution_allows_paper_kis_credentials(tenant_settings) or distribution_mode == "simulated_only"
            uses_broker_credentials = distribution_uses_broker_credentials(tenant_settings) or distribution_mode == "simulated_only"
            sanitized_accounts: list[dict[str, Any]] = []
            if uses_broker_credentials:
                for account in accounts:
                    sanitized = dict(account)
                    has_real_key_input = bool(str(sanitized.get("app_key") or "").strip() or str(sanitized.get("app_secret") or "").strip())
                    wants_real_env = str(sanitized.get("env") or "real").strip().lower() == "real"
                    if not allow_real_kis_credentials and (has_real_key_input or wants_real_env):
                        raise ValueError("tenant is not approved for real KIS credentials")
                    if not allow_real_kis_credentials:
                        sanitized["env"] = "demo"
                        sanitized["app_key"] = ""
                        sanitized["app_secret"] = ""
                    if not allow_paper_kis_credentials:
                        sanitized["paper_app_key"] = ""
                        sanitized["paper_app_secret"] = ""
                    sanitized_accounts.append(sanitized)
            accounts = sanitized_accounts

            if distribution_mode == "simulated_only" and not accounts:
                repo.append_runtime_audit_log(
                    action="settings_save",
                    status="ok",
                    user_email=user_email or updated_by,
                    tenant_id=tenant,
                    detail={"kis_secret_name": None, "num_accounts": 0, "distribution_mode": "simulated_only"},
                )
                return settings_page(
                    request=request,
                    tenant_id=tenant,
                    ok="1",
                    msg="Saved: tenant remains in simulated mode until a KIS demo account is connected",
                )

            next_distribution_mode = distribution_mode
            if distribution_mode == "simulated_only" and accounts:
                next_distribution_mode = "paper_only"
            elif distribution_mode == "paper_only" and not accounts:
                next_distribution_mode = "simulated_only"

            if distribution_mode == "paper_only" and not accounts:
                repo.set_config(tenant, "distribution_mode", "simulated_only", user_email or updated_by)
                _invalidate_tenant_runtime_cache(tenant)
                repo.append_runtime_audit_log(
                    action="settings_save",
                    status="ok",
                    user_email=user_email or updated_by,
                    tenant_id=tenant,
                    detail={"kis_secret_name": None, "num_accounts": 0, "distribution_mode": "simulated_only"},
                )
                return settings_page(
                    request=request,
                    tenant_id=tenant,
                    ok="1",
                    msg="Saved: paper credentials removed and tenant returned to simulated mode",
                )

            refs = credential_store.save_kis_accounts(
                tenant_id=tenant,
                updated_by=user_email or updated_by,
                accounts=accounts,
                notes=notes,
            )
            if next_distribution_mode != distribution_mode:
                repo.set_config(tenant, "distribution_mode", next_distribution_mode, user_email or updated_by)
                _invalidate_tenant_runtime_cache(tenant)
            repo.append_runtime_audit_log(
                action="settings_save",
                status="ok",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={
                    "kis_secret_name": refs.kis_secret_name,
                    "num_accounts": len(accounts),
                    "distribution_mode": next_distribution_mode,
                },
            )
            msg = (
                f"Saved: tenant={refs.tenant_id}, "
                f"kis_secret={refs.kis_secret_name}, accounts={len(accounts)}, mode={next_distribution_mode}"
            )
            return settings_page(request=request, tenant_id=tenant, ok="1", msg=msg)
        except Exception as exc:
            repo.append_runtime_audit_log(
                action="settings_save",
                status="error",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"error": str(exc)},
            )
            return settings_page(request=request, tenant_id=tenant, ok="0", msg=str(exc))
