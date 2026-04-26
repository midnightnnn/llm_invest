from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse

from arena.config import Settings
from arena.providers import list_adk_provider_specs, provider_alias_map
from arena.ui.admin_agent_config import (
    AdminAgentConfigStore,
    build_agents_config_save_payload,
    build_single_agent_entry,
    serialize_agents_config_entries,
)
from arena.ui.admin_runtime_ops import (
    AdminRuntimeOps,
    live_market_sources_for_market_value,
)

@dataclass(frozen=True)
class AdminSettingsRouteDeps:
    repo: Any
    settings_enabled: bool
    credential_store: Any | None
    resolve_admin_context: Callable[..., Any]
    tenant_access_denied: Callable[[str, str], bool]
    current_admin_view_model: Callable[[str], dict[str, Any]]
    settings_for_tenant: Callable[[str], Settings]
    get_default_registry: Callable[[str], Any]
    invalidate_tenant_cache: Callable[..., None]
    invalidate_tenant_runtime_cache: Callable[[str], None]
    settings_redirect: Callable[..., RedirectResponse]
    parse_json_array: Callable[[object], list[Any]]
    safe_float: Callable[[object, float], float]
    to_bool_token: Callable[[object, bool], bool]
    is_live_mode: Callable[[Settings | None], bool]
    live_market_sources: Callable[[Settings | None], list[str] | None]


def register_admin_settings_routes(app: FastAPI, *, deps: AdminSettingsRouteDeps) -> None:
    repo = deps.repo
    settings_enabled = deps.settings_enabled
    credential_store = deps.credential_store
    _resolve_admin_context = deps.resolve_admin_context
    _tenant_access_denied = deps.tenant_access_denied
    _current_admin_view_model = deps.current_admin_view_model
    _settings_for_tenant = deps.settings_for_tenant
    _get_default_registry = deps.get_default_registry
    _invalidate_tenant_cache = deps.invalidate_tenant_cache
    _invalidate_tenant_runtime_cache = deps.invalidate_tenant_runtime_cache
    _settings_redirect = deps.settings_redirect
    _parse_json_array = deps.parse_json_array
    _safe_float = deps.safe_float
    _to_bool_token = deps.to_bool_token
    _is_live_mode = deps.is_live_mode
    _live_market_sources = deps.live_market_sources

    config_store = AdminAgentConfigStore(
        repo=repo,
        current_admin_view_model=_current_admin_view_model,
    )
    runtime_ops = AdminRuntimeOps(
        repo=repo,
        is_live_mode=_is_live_mode,
        live_market_sources=_live_market_sources,
        safe_float=_safe_float,
    )

    def _live_sources_for(tenant_settings: Settings, market_value: str) -> list[str] | None:
        return live_market_sources_for_market_value(
            tenant_settings=tenant_settings,
            market_value=market_value,
            is_live_mode=_is_live_mode,
            live_market_sources=_live_market_sources,
        )

    @app.get("/admin/prompt")
    def admin_prompt_get(
        request: Request,
        tenant_id: str = Query(default="local"),
    ):
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _, _, tenant, _, redirect = _resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/settings?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        vm = _current_admin_view_model(tenant)
        return JSONResponse(
            {
                "tenant_id": tenant,
                "core_prompt": vm["core_prompt_text"],
                "system_prompt": vm["prompt_text"],
            }
        )

    @app.post("/admin/prompt")
    def admin_prompt_save(
        request: Request,
        tenant_id: str = Form(default="local"),
        updated_by: str = Form(default="ui-admin"),
        system_prompt: str = Form(default=""),
    ):
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
            return _settings_redirect(tenant, ok=False, msg="tenant access denied")
        prompt = str(system_prompt or "").strip()
        if not prompt:
            return _settings_redirect(tenant, ok=False, msg="user prompt cannot be empty")
        try:
            # core_prompt is global (file-only); not saved per-tenant via UI.
            repo.set_config(tenant, "system_prompt", prompt, updated_by=user_email or updated_by)
            _invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")
            repo.append_runtime_audit_log(
                action="admin_prompt_save",
                status="ok",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"user_length": len(prompt)},
            )
            return _settings_redirect(tenant, ok=True, msg="prompts saved")
        except Exception as exc:
            repo.append_runtime_audit_log(
                action="admin_prompt_save",
                status="error",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"error": str(exc)},
            )
            return _settings_redirect(tenant, ok=False, msg=str(exc))

    @app.get("/admin/agents")
    def admin_agents_get(
        request: Request,
        tenant_id: str = Query(default="local"),
    ):
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _, _, tenant, _, redirect = _resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/settings?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        vm = _current_admin_view_model(tenant)
        return JSONResponse(
            {
                "tenant_id": tenant,
                "agent_ids": vm["agent_ids"],
                "agent_models": vm["agent_models"],
                "agents_config": vm["agents_config"],
                "invalid_runtime_config_keys": vm.get("invalid_runtime_config_keys", []),
                "api_key_status": vm.get("api_key_status", {}),
                "research_status": vm.get("research_status", {}),
            }
        )

    @app.post("/admin/agents")
    def admin_agents_save(
        request: Request,
        tenant_id: str = Form(default="local"),
        updated_by: str = Form(default="ui-admin"),
        agents_config_json: str = Form(default="[]"),
    ):
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
            return _settings_redirect(tenant, ok=False, msg="tenant access denied")
        tenant_settings = _settings_for_tenant(tenant)

        raw_entries = _parse_json_array(agents_config_json)
        if not raw_entries:
            return _settings_redirect(tenant, ok=False, msg="agents_config is empty")

        allowed_providers = {spec.provider_id for spec in list_adk_provider_specs()}
        prov_alias = provider_alias_map()
        payload = build_agents_config_save_payload(
            raw_entries,
            tenant_settings=tenant_settings,
            allowed_providers=allowed_providers,
            provider_aliases=prov_alias,
            safe_float=_safe_float,
        )

        if not payload.entries:
            return _settings_redirect(tenant, ok=False, msg="no valid agents in config")

        agents_config_for_db = serialize_agents_config_entries(
            payload.entries,
            tenant_settings=tenant_settings,
            safe_float=_safe_float,
        )

        try:
            repo.set_config(
                tenant,
                "agents_config",
                json.dumps(agents_config_for_db, ensure_ascii=False),
                updated_by=user_email or updated_by,
            )
            synced_market = config_store.sync_market(
                tenant=tenant,
                entries=agents_config_for_db,
                tenant_settings=tenant_settings,
                updated_by=user_email or updated_by,
            )

            if payload.api_keys and credential_store is not None:
                credential_store.save_model_keys(
                    tenant_id=tenant,
                    updated_by=user_email or updated_by,
                    providers=payload.api_keys,
                )

            sync_summary = runtime_ops.sync_runtime_state(
                tenant=tenant,
                tenant_settings=tenant_settings,
                entries=agents_config_for_db,
                updated_by=user_email or updated_by,
                sources=_live_sources_for(tenant_settings, synced_market),
            )
            _invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")

            repo.append_runtime_audit_log(
                action="admin_agents_save",
                status="ok",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={
                    "agent_ids": payload.agent_ids,
                    "models": payload.models_by_provider,
                    "kis_target_market": synced_market,
                    "agents_config": agents_config_for_db,
                    "api_keys_provided": list(payload.api_keys.keys()),
                    "sync": sync_summary,
                },
            )
            return _settings_redirect(tenant, ok=True, msg="agent settings saved")
        except Exception as exc:
            repo.append_runtime_audit_log(
                action="admin_agents_save",
                status="error",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"error": str(exc)},
            )
            return _settings_redirect(tenant, ok=False, msg=str(exc))

    @app.post("/admin/agents/save-one")
    async def admin_agents_save_one(request: Request):
        if not settings_enabled:
            return JSONResponse({"ok": False, "message": "settings disabled"}, status_code=403)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "message": "invalid JSON body"}, status_code=400)

        requested_tenant = str(body.get("tenant_id") or "local")
        agent_data = body.get("agent", {})
        if not isinstance(agent_data, dict):
            return JSONResponse({"ok": False, "message": "agent.id is required"}, status_code=400)
        aid = str(agent_data.get("id") or agent_data.get("agent_id") or "").strip().lower()
        if not aid:
            return JSONResponse({"ok": False, "message": "agent.id is required"}, status_code=400)

        _, user_email, tenant, _, redirect = _resolve_admin_context(
            request,
            requested_tenant=requested_tenant,
            next_path=f"/settings?tenant_id={requested_tenant}",
        )
        if redirect:
            return JSONResponse({"ok": False, "message": "auth redirect required"}, status_code=403)
        if _tenant_access_denied(requested_tenant, tenant):
            return JSONResponse({"ok": False, "message": "tenant access denied"}, status_code=403)
        tenant_settings = _settings_for_tenant(tenant)

        allowed_providers = {spec.provider_id for spec in list_adk_provider_specs()}
        prov_alias = provider_alias_map()

        try:
            existing_entries, _ = config_store.load_for_update(tenant)
            existing_entry = next(
                (
                    dict(entry)
                    for entry in existing_entries
                    if isinstance(entry, dict) and str(entry.get("id") or "").strip().lower() == aid
                ),
                None,
            )
            aid, provider, new_entry, raw_api_key = build_single_agent_entry(
                agent_data=agent_data,
                existing_entry=existing_entry,
                tenant_settings=tenant_settings,
                allowed_providers=allowed_providers,
                provider_aliases=prov_alias,
                safe_float=_safe_float,
            )

            found = False
            for index, entry in enumerate(existing_entries):
                if isinstance(entry, dict) and str(entry.get("id", "")).strip().lower() == aid:
                    existing_entries[index] = new_entry
                    found = True
                    break
            if not found:
                existing_entries.append(new_entry)

            agents_config_for_db = serialize_agents_config_entries(
                existing_entries,
                tenant_settings=tenant_settings,
                safe_float=_safe_float,
            )

            repo.set_config(
                tenant,
                "agents_config",
                json.dumps(agents_config_for_db, ensure_ascii=False),
                updated_by=user_email or "ui-admin",
            )
            synced_market = config_store.sync_market(
                tenant=tenant,
                entries=agents_config_for_db,
                tenant_settings=tenant_settings,
                updated_by=user_email or "ui-admin",
            )

            if raw_api_key and credential_store is not None:
                key_payload = {"api_key": raw_api_key}
                if new_entry["model"]:
                    key_payload["model"] = new_entry["model"]
                credential_store.save_model_keys(
                    tenant_id=tenant,
                    updated_by=user_email or "ui-admin",
                    providers={provider: key_payload},
                )

            sync_summary = runtime_ops.sync_runtime_state(
                tenant=tenant,
                tenant_settings=tenant_settings,
                entries=agents_config_for_db,
                updated_by=user_email or "ui-admin",
                sources=_live_sources_for(tenant_settings, synced_market),
            )
            _invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")

            repo.append_runtime_audit_log(
                action="admin_agent_save_one",
                status="ok",
                user_email=user_email or "ui-admin",
                tenant_id=tenant,
                detail={
                    "agent_id": aid,
                    "provider": provider,
                    "kis_target_market": synced_market,
                    "updated_fields": sorted(str(key) for key in agent_data.keys()),
                    "sync": sync_summary,
                },
            )
            return JSONResponse({"ok": True, "message": f"Agent '{aid}' saved"})
        except ValueError as exc:
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
        except Exception as exc:
            repo.append_runtime_audit_log(
                action="admin_agent_save_one",
                status="error",
                user_email=user_email or "ui-admin",
                tenant_id=tenant,
                detail={"agent_id": aid, "error": str(exc)},
            )
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    @app.post("/admin/agents/remove-one")
    async def admin_agents_remove_one(request: Request):
        if not settings_enabled:
            return JSONResponse({"ok": False, "message": "settings disabled"}, status_code=403)

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "message": "invalid JSON body"}, status_code=400)

        requested_tenant = str(body.get("tenant_id") or "local")
        aid = str(body.get("agent_id") or "").strip().lower()
        force_remove = bool(body.get("force"))
        if not aid:
            return JSONResponse({"ok": False, "message": "agent_id is required"}, status_code=400)

        _, user_email, tenant, _, redirect = _resolve_admin_context(
            request,
            requested_tenant=requested_tenant,
            next_path=f"/settings?tenant_id={requested_tenant}",
        )
        if redirect:
            return JSONResponse({"ok": False, "message": "auth redirect required"}, status_code=403)
        if _tenant_access_denied(requested_tenant, tenant):
            return JSONResponse({"ok": False, "message": "tenant access denied"}, status_code=403)

        tenant_settings = _settings_for_tenant(tenant)

        try:
            existing_entries, _ = config_store.load_for_update(tenant)
            target_entry = next(
                (
                    dict(entry)
                    for entry in existing_entries
                    if isinstance(entry, dict) and str(entry.get("id") or "").strip().lower() == aid
                ),
                None,
            )
            if target_entry is None:
                return JSONResponse({"ok": False, "message": f"agent '{aid}' not found"}, status_code=404)

            warning_message = runtime_ops.build_remove_warning(
                tenant=tenant,
                tenant_settings=tenant_settings,
                agent_entry=target_entry,
            )
            if warning_message and not force_remove:
                return JSONResponse({"ok": False, "confirm_required": True, "message": warning_message})

            remaining_entries = [
                dict(entry)
                for entry in existing_entries
                if isinstance(entry, dict) and str(entry.get("id") or "").strip().lower() != aid
            ]
            agents_config_for_db = serialize_agents_config_entries(
                remaining_entries,
                tenant_settings=tenant_settings,
                safe_float=_safe_float,
            )

            repo.set_config(
                tenant,
                "agents_config",
                json.dumps(agents_config_for_db, ensure_ascii=False),
                updated_by=user_email or "ui-admin",
            )
            synced_market = config_store.sync_market(
                tenant=tenant,
                entries=agents_config_for_db,
                tenant_settings=tenant_settings,
                updated_by=user_email or "ui-admin",
            )
            _invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")

            remaining_agent_ids = [
                str(entry.get("id") or "").strip().lower()
                for entry in agents_config_for_db
            ]
            repo.append_runtime_audit_log(
                action="admin_agent_remove_one",
                status="ok",
                user_email=user_email or "ui-admin",
                tenant_id=tenant,
                detail={
                    "agent_id": aid,
                    "remaining_agent_ids": remaining_agent_ids,
                    "kis_target_market": synced_market,
                    "forced": force_remove,
                },
            )
            return JSONResponse(
                {
                    "ok": True,
                    "message": f"Agent '{aid}' removed",
                    "remaining_agent_ids": remaining_agent_ids,
                }
            )
        except Exception as exc:
            repo.append_runtime_audit_log(
                action="admin_agent_remove_one",
                status="error",
                user_email=user_email or "ui-admin",
                tenant_id=tenant,
                detail={"agent_id": aid, "forced": force_remove, "error": str(exc)},
            )
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=500)

    @app.get("/admin/risk")
    def admin_risk_get(
        request: Request,
        tenant_id: str = Query(default="local"),
    ):
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _, _, tenant, _, redirect = _resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/settings?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        vm = _current_admin_view_model(tenant)
        return JSONResponse(
            {
                "tenant_id": tenant,
                "risk_policy": vm["risk"],
                "invalid_runtime_config_keys": vm.get("invalid_runtime_config_keys", []),
            }
        )

    @app.post("/admin/risk")
    def admin_risk_save(
        request: Request,
        tenant_id: str = Form(default="local"),
        updated_by: str = Form(default="ui-admin"),
        max_order_krw: str = Form(default=""),
        max_daily_turnover_ratio: str = Form(default=""),
        max_position_ratio: str = Form(default=""),
        min_cash_buffer_ratio: str = Form(default=""),
        ticker_cooldown_seconds: str = Form(default=""),
        max_daily_orders: str = Form(default=""),
        estimated_fee_bps: str = Form(default=""),
    ):
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
            return _settings_redirect(tenant, ok=False, msg="tenant access denied")
        try:
            risk = {
                "max_order_krw": float(max_order_krw),
                "max_daily_turnover_ratio": float(max_daily_turnover_ratio),
                "max_position_ratio": float(max_position_ratio),
                "min_cash_buffer_ratio": float(min_cash_buffer_ratio),
                "ticker_cooldown_seconds": int(float(ticker_cooldown_seconds)),
                "max_daily_orders": int(float(max_daily_orders)),
                "estimated_fee_bps": float(estimated_fee_bps),
            }
        except ValueError:
            return _settings_redirect(tenant, ok=False, msg="risk fields must be numeric")

        try:
            repo.set_config(
                tenant,
                "risk_policy",
                json.dumps(risk, ensure_ascii=False),
                updated_by=user_email or updated_by,
            )
            _invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")
            repo.append_runtime_audit_log(
                action="admin_risk_save",
                status="ok",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail=risk,
            )
            return _settings_redirect(tenant, ok=True, msg="risk_policy saved")
        except Exception as exc:
            repo.append_runtime_audit_log(
                action="admin_risk_save",
                status="error",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"error": str(exc)},
            )
            return _settings_redirect(tenant, ok=False, msg=str(exc))

    @app.get("/admin/tools")
    def admin_tools_get(
        request: Request,
        tenant_id: str = Query(default="local"),
    ):
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _, _, tenant, _, redirect = _resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/settings?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        vm = _current_admin_view_model(tenant)
        return JSONResponse(
            {
                "tenant_id": tenant,
                "disabled_tools": vm["disabled_tools"],
                "tool_entries": vm["tool_entries"],
                "invalid_runtime_config_keys": vm.get("invalid_runtime_config_keys", []),
            }
        )

    @app.post("/admin/tools")
    def admin_tools_save(
        request: Request,
        tenant_id: str = Form(default="local"),
        updated_by: str = Form(default="ui-admin"),
        disabled_tools: list[str] = Form(default=[]),
        disabled_tools_csv: str = Form(default=""),
    ):
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
            return _settings_redirect(tenant, ok=False, msg="tenant access denied")
        registry = _get_default_registry(tenant)
        allowed_tool_ids = {entry.tool_id for entry in registry.list_entries(include_disabled=True)}
        raw_tokens = list(disabled_tools or [])
        if disabled_tools_csv:
            raw_tokens.extend(disabled_tools_csv.split(","))
        disabled = sorted(
            tool_id for token in raw_tokens if (tool_id := str(token).strip()) and tool_id in allowed_tool_ids
        )
        try:
            repo.set_config(
                tenant,
                "disabled_tools",
                json.dumps(disabled, ensure_ascii=False),
                updated_by=user_email or updated_by,
            )
            _invalidate_tenant_cache(tenant, "runtime", "memory")
            repo.append_runtime_audit_log(
                action="admin_tools_save",
                status="ok",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"disabled_tools": disabled},
            )
            return _settings_redirect(tenant, ok=True, msg="disabled_tools saved")
        except Exception as exc:
            repo.append_runtime_audit_log(
                action="admin_tools_save",
                status="error",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"error": str(exc)},
            )
            return _settings_redirect(tenant, ok=False, msg=str(exc))

    @app.get("/admin/tools/mcp")
    def admin_mcp_get(
        request: Request,
        tenant_id: str = Query(default="local"),
    ):
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _, _, tenant, _, redirect = _resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/settings?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        vm = _current_admin_view_model(tenant)
        return JSONResponse(
            {
                "tenant_id": tenant,
                "mcp_servers": vm["mcp_servers"],
                "invalid_runtime_config_keys": vm.get("invalid_runtime_config_keys", []),
            }
        )

    @app.post("/admin/tools/mcp")
    def admin_mcp_save(
        request: Request,
        tenant_id: str = Form(default="local"),
        updated_by: str = Form(default="ui-admin"),
        mcp_servers_json: str = Form(default="[]"),
    ):
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
            return _settings_redirect(tenant, ok=False, msg="tenant access denied")

        raw_servers = _parse_json_array(mcp_servers_json)
        normalized: list[dict[str, Any]] = []
        for row in raw_servers:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            url = str(row.get("url") or "").strip()
            transport = str(row.get("transport") or "sse").strip().lower() or "sse"
            if not name or not url:
                continue
            if transport not in {"sse", "streamable_http"}:
                return _settings_redirect(tenant, ok=False, msg=f"invalid mcp transport: {transport}")
            normalized.append(
                {
                    "name": name,
                    "url": url,
                    "transport": transport,
                    "enabled": _to_bool_token(row.get("enabled"), True),
                }
            )
        try:
            repo.set_config(
                tenant,
                "mcp_servers",
                json.dumps(normalized, ensure_ascii=False),
                updated_by=user_email or updated_by,
            )
            _invalidate_tenant_runtime_cache(tenant)
            repo.append_runtime_audit_log(
                action="admin_mcp_save",
                status="ok",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"count": len(normalized)},
            )
            return _settings_redirect(tenant, ok=True, msg="mcp_servers saved")
        except Exception as exc:
            repo.append_runtime_audit_log(
                action="admin_mcp_save",
                status="error",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"error": str(exc)},
            )
            return _settings_redirect(tenant, ok=False, msg=str(exc))
