from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse

from arena.config import Settings
from arena.market_sources import live_market_sources_for_markets
from arena.providers import canonical_provider, list_adk_provider_specs, provider_alias_map, runtime_row_api_key_status

logger = logging.getLogger(__name__)


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

    def _load_explicit_agents_config_rows(tenant: str) -> tuple[list[dict[str, Any]], bool]:
        raw_loader = getattr(repo, "get_config", None)
        if not callable(raw_loader):
            return [], False
        try:
            raw_value = raw_loader(tenant, "agents_config")
        except Exception:
            raw_value = None
        raw_text = str(raw_value or "").strip()
        if not raw_text:
            return [], False
        try:
            parsed = json.loads(raw_text)
        except Exception:
            return [], False
        if not isinstance(parsed, list):
            return [], False
        return [dict(entry) for entry in parsed if isinstance(entry, dict)], True

    def _load_agent_entries_for_update(tenant: str) -> tuple[list[dict[str, Any]], bool]:
        entries, has_explicit_config = _load_explicit_agents_config_rows(tenant)
        if has_explicit_config:
            return entries, True
        vm = _current_admin_view_model(tenant)
        fallback_entries = [
            dict(entry)
            for entry in list(vm.get("agents_config") or [])
            if isinstance(entry, dict)
        ]
        return fallback_entries, False

    def _serialize_agents_config_entries(
        entries: list[dict[str, Any]],
        *,
        tenant_settings: Settings,
    ) -> list[dict[str, Any]]:
        agents_config_for_db: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            aid = str(entry.get("id") or "").strip().lower()
            if not aid:
                continue
            capital = _safe_float(entry.get("capital_krw"), tenant_settings.sleeve_capital_krw)
            if capital <= 0:
                capital = tenant_settings.sleeve_capital_krw
            db_entry: dict[str, Any] = {
                "id": aid,
                "provider": str(entry.get("provider") or "").strip().lower(),
                "model": str(entry.get("model") or "").strip(),
                "capital_krw": capital,
            }
            if entry.get("target_market"):
                db_entry["target_market"] = str(entry["target_market"]).strip().lower()
            if entry.get("system_prompt"):
                db_entry["system_prompt"] = str(entry["system_prompt"]).strip()
            if entry.get("risk_policy"):
                db_entry["risk_policy"] = entry["risk_policy"]
            if isinstance(entry.get("disabled_tools"), list):
                db_entry["disabled_tools"] = [str(x).strip() for x in entry["disabled_tools"] if str(x).strip()]
            agents_config_for_db.append(db_entry)
        return agents_config_for_db

    def _normalize_market_tokens(raw_market: object) -> list[str]:
        alias = {"kr": "kospi", "korea": "kospi"}
        allowed = {"us", "nasdaq", "nyse", "amex", "kospi", "kosdaq"}
        tokens: list[str] = []
        for token in str(raw_market or "").split(","):
            market = alias.get(str(token).strip().lower(), str(token).strip().lower())
            if not market or market not in allowed or market in tokens:
                continue
            tokens.append(market)
        if "us" in tokens:
            tokens = [token for token in tokens if token == "us" or token not in {"nasdaq", "nyse", "amex"}]
        return tokens

    def _derive_tenant_market_from_agents(entries: list[dict[str, Any]], *, fallback_market: str) -> str:
        tokens: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for market in _normalize_market_tokens(entry.get("target_market")):
                if market not in tokens:
                    tokens.append(market)
        if not tokens:
            tokens = _normalize_market_tokens(fallback_market)
        return ",".join(tokens)

    def _sync_tenant_market_config(
        *,
        tenant: str,
        entries: list[dict[str, Any]],
        tenant_settings: Settings,
        updated_by: str,
    ) -> str:
        current_market = str(getattr(tenant_settings, "kis_target_market", "") or "").strip().lower()
        next_market = _derive_tenant_market_from_agents(entries, fallback_market=current_market)
        if next_market and next_market != current_market:
            repo.set_config(tenant, "kis_target_market", next_market, updated_by)
        return next_market or current_market

    def _live_market_sources_for_market_value(
        *,
        tenant_settings: Settings,
        market_value: str,
    ) -> list[str] | None:
        if not _is_live_mode(tenant_settings):
            return None
        sources = live_market_sources_for_markets(market_value)
        if sources:
            return sources
        return _live_market_sources(tenant_settings)

    def _sync_agent_runtime_state(
        *,
        tenant: str,
        tenant_settings: Settings,
        entries: list[dict[str, Any]],
        updated_by: str,
        sources: list[str] | None,
    ) -> dict[str, Any]:
        parsed_agent_ids = [str(entry.get("id") or "").strip().lower() for entry in entries if str(entry.get("id") or "").strip()]
        sync_summary: dict[str, Any] = {"agents": len(parsed_agent_ids)}
        if not parsed_agent_ids:
            return sync_summary

        capitals = [_safe_float(entry.get("capital_krw"), tenant_settings.sleeve_capital_krw) for entry in entries]
        avg_capital = sum(capitals) / len(capitals) if capitals else tenant_settings.sleeve_capital_krw
        target_capitals = {
            str(entry.get("id") or "").strip().lower(): _safe_float(entry.get("capital_krw"), tenant_settings.sleeve_capital_krw)
            for entry in entries
            if str(entry.get("id") or "").strip()
        }

        tenant_is_live = _is_live_mode(tenant_settings)
        sync_capitals = getattr(repo, "retarget_agent_capitals_preserve_positions", None)
        sync_sleeves = getattr(repo, "retarget_agent_sleeves_preserve_positions", None)
        sync_fn = sync_capitals if callable(sync_capitals) else sync_sleeves
        if callable(sync_fn):
            sync_kwargs = {
                "agent_ids": parsed_agent_ids,
                "target_sleeve_capital_krw": avg_capital,
                "target_capitals": target_capitals,
                "include_simulated": not tenant_is_live,
                "sources": sources,
                "tenant_id": tenant,
            }
            if sync_fn is sync_capitals:
                sync_kwargs["created_by"] = updated_by
            try:
                sync_out = sync_fn(**sync_kwargs)
                over_target_agents = sorted(
                    [aid for aid, meta in dict(sync_out or {}).items() if bool(dict(meta).get("over_target"))]
                )
                sync_summary["over_target_agents"] = over_target_agents
            except Exception as retarget_exc:
                logger.warning(
                    "[yellow]Sleeve retarget after agents save failed[/yellow] tenant=%s err=%s",
                    tenant,
                    str(retarget_exc),
                )
                sync_summary["retarget_error"] = str(retarget_exc)

        build_sleeve = getattr(repo, "build_agent_sleeve_snapshot", None)
        upsert_nav = getattr(repo, "upsert_agent_nav_daily", None)
        if callable(build_sleeve) and callable(upsert_nav):
            nav_date = datetime.now(timezone.utc).date()
            nav_updated = 0
            nav_failed: list[str] = []
            for aid in parsed_agent_ids:
                try:
                    snap, baseline, meta = build_sleeve(
                        agent_id=aid,
                        sources=sources,
                        include_simulated=not tenant_is_live,
                        tenant_id=tenant,
                    )
                    upsert_nav(
                        nav_date=nav_date,
                        agent_id=aid,
                        nav_krw=float(snap.total_equity_krw),
                        baseline_equity_krw=float(baseline),
                        cash_krw=float(snap.cash_krw),
                        market_value_krw=sum(pos.market_value_krw() for pos in snap.positions.values()),
                        capital_flow_krw=float((meta or {}).get("capital_flow_krw") or 0.0)
                        + float((meta or {}).get("manual_cash_adjustment_krw") or 0.0),
                        fx_source=str((meta or {}).get("fx_source") or ""),
                        valuation_source=str((meta or {}).get("valuation_source") or "agent_sleeve_snapshot"),
                        tenant_id=tenant,
                    )
                    nav_updated += 1
                except Exception as nav_exc:
                    nav_failed.append(aid)
                    logger.warning(
                        "[yellow]Agent NAV sync after agents save failed[/yellow] tenant=%s agent=%s err=%s",
                        tenant,
                        aid,
                        str(nav_exc),
                    )
            sync_summary["nav_sync"] = {
                "updated": nav_updated,
                "failed_agents": nav_failed,
                "nav_date": nav_date.isoformat(),
            }
        return sync_summary

    def _build_agent_remove_warning(
        *,
        tenant: str,
        tenant_settings: Settings,
        agent_entry: dict[str, Any],
    ) -> str:
        aid = str(agent_entry.get("id") or "").strip().lower()
        provider = canonical_provider(str(agent_entry.get("provider") or "").strip().lower())
        if not provider:
            provider = provider_alias_map().get(aid, "")

        has_api_key = False
        latest_creds_loader = getattr(repo, "latest_runtime_credentials", None)
        if callable(latest_creds_loader):
            try:
                latest_creds = latest_creds_loader(tenant_id=tenant) or {}
                has_api_key = bool(runtime_row_api_key_status(latest_creds).get(provider))
            except Exception:
                has_api_key = False

        tenant_is_live = _is_live_mode(tenant_settings)
        tenant_live_sources = _live_market_sources(tenant_settings) if tenant_is_live else None
        sleeve_equity = 0.0
        baseline_equity = 0.0
        snapshot_available = False
        build_sleeve = getattr(repo, "build_agent_sleeve_snapshot", None)
        if callable(build_sleeve):
            try:
                snapshot, baseline, _meta = build_sleeve(
                    agent_id=aid,
                    sources=tenant_live_sources,
                    include_simulated=not tenant_is_live,
                    tenant_id=tenant,
                )
                sleeve_equity = float(getattr(snapshot, "total_equity_krw", 0.0) or 0.0)
                baseline_equity = float(baseline or 0.0)
                snapshot_available = True
            except Exception as exc:
                logger.warning(
                    "[yellow]Agent remove warning preflight failed[/yellow] tenant=%s agent=%s err=%s",
                    tenant,
                    aid,
                    str(exc),
                )

        warning_bits: list[str] = []
        if has_api_key:
            warning_bits.append("저장된 API key")

        active_equity = max(sleeve_equity, baseline_equity)
        if active_equity > 0:
            warning_bits.append(f"활성 슬리브 자금/평가금액 ₩{int(round(active_equity)):,}")
        elif not snapshot_available:
            configured_capital = _safe_float(agent_entry.get("capital_krw"), 0.0)
            if configured_capital > 0:
                warning_bits.append(f"설정된 자본금 ₩{int(round(configured_capital)):,}")

        if not warning_bits:
            return ""

        message = f"'{aid}' 제거 전 확인: " + ", ".join(warning_bits) + " 이 있습니다. 정말 제거할까요?"
        if has_api_key:
            message += " 저장된 API key 자체는 삭제되지 않습니다."
        return message

    def _build_single_agent_entry(
        *,
        agent_data: dict[str, Any],
        existing_entry: dict[str, Any] | None,
        tenant_settings: Settings,
        allowed_providers: set[str],
        provider_aliases: dict[str, str],
    ) -> tuple[str, str, dict[str, Any], str]:
        current = dict(existing_entry or {})
        aid = str(agent_data.get("id") or agent_data.get("agent_id") or current.get("id") or "").strip().lower()
        if not aid:
            raise ValueError("agent.id is required")

        provider_raw = agent_data.get("provider") if "provider" in agent_data else current.get("provider")
        provider = canonical_provider(str(provider_raw or "").strip().lower())
        if not provider:
            provider = provider_aliases.get(aid, "")
        if provider not in allowed_providers:
            raise ValueError(f"invalid provider: {provider}")

        model = str(agent_data.get("model") if "model" in agent_data else current.get("model") or "").strip()
        capital_raw = agent_data.get("capital_krw") if "capital_krw" in agent_data else current.get("capital_krw")
        capital = _safe_float(capital_raw, tenant_settings.sleeve_capital_krw)
        if capital <= 0:
            capital = _safe_float(current.get("capital_krw"), tenant_settings.sleeve_capital_krw)
        if capital <= 0:
            capital = tenant_settings.sleeve_capital_krw

        target_market = (
            str(agent_data.get("target_market") if "target_market" in agent_data else current.get("target_market") or "")
            .strip()
            .lower()
        )
        system_prompt = str(agent_data.get("system_prompt") if "system_prompt" in agent_data else current.get("system_prompt") or "").strip()
        risk_policy = agent_data.get("risk_policy") if "risk_policy" in agent_data else current.get("risk_policy")
        disabled_tools_raw = agent_data.get("disabled_tools") if "disabled_tools" in agent_data else current.get("disabled_tools")

        entry: dict[str, Any] = {
            "id": aid,
            "provider": provider,
            "model": model,
            "capital_krw": capital,
        }
        if target_market:
            entry["target_market"] = target_market
        if system_prompt:
            entry["system_prompt"] = system_prompt
        if isinstance(risk_policy, dict) and risk_policy:
            entry["risk_policy"] = risk_policy
        if isinstance(disabled_tools_raw, list):
            entry["disabled_tools"] = [str(x).strip() for x in disabled_tools_raw if str(x).strip()]

        raw_api_key = str(agent_data.get("api_key") or "").strip()
        return aid, provider, entry, raw_api_key

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
        return JSONResponse({"tenant_id": tenant, "core_prompt": vm["core_prompt_text"], "system_prompt": vm["prompt_text"]})

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
        tenant_is_live = _is_live_mode(tenant_settings)

        raw_entries = _parse_json_array(agents_config_json)
        if not raw_entries:
            return _settings_redirect(tenant, ok=False, msg="agents_config is empty")

        allowed_providers = {spec.provider_id for spec in list_adk_provider_specs()}
        _prov_alias = provider_alias_map()
        entries: list[dict[str, Any]] = []
        api_keys: dict[str, dict[str, str]] = {}
        for item in raw_entries:
            if not isinstance(item, dict):
                continue
            aid = str(item.get("id") or "").strip().lower()
            if not aid:
                continue
            # Resolve provider: explicit field first, then infer from id.
            provider = canonical_provider(str(item.get("provider") or "").strip().lower())
            if not provider:
                provider = _prov_alias.get(aid, "")
            if provider not in allowed_providers:
                continue
            model = str(item.get("model") or "").strip()
            capital = _safe_float(item.get("capital_krw"), tenant_settings.sleeve_capital_krw)
            if capital <= 0:
                capital = tenant_settings.sleeve_capital_krw
            target_market = str(item.get("target_market") or "").strip().lower()
            entry: dict[str, Any] = {"id": aid, "provider": provider, "model": model, "capital_krw": capital}
            if target_market:
                entry["target_market"] = target_market
            if item.get("system_prompt"):
                entry["system_prompt"] = str(item["system_prompt"]).strip()
            if isinstance(item.get("risk_policy"), dict) and item["risk_policy"]:
                entry["risk_policy"] = item["risk_policy"]
            if isinstance(item.get("disabled_tools"), list):
                entry["disabled_tools"] = [str(x).strip() for x in item["disabled_tools"] if str(x).strip()]
            entries.append(entry)
            raw_key = str(item.get("api_key") or "").strip()
            provider_payload = dict(api_keys.get(provider) or {})
            if raw_key:
                provider_payload["api_key"] = raw_key
            if model:
                provider_payload["model"] = model
            if provider_payload:
                api_keys[provider] = provider_payload

        if not entries:
            return _settings_redirect(tenant, ok=False, msg="no valid agents in config")

        parsed_agent_ids = [e["id"] for e in entries]
        models: dict[str, str] = {}
        for entry in entries:
            provider = str(entry.get("provider") or "")
            if entry["model"] and provider not in models:
                models[provider] = entry["model"]

        agents_config_for_db = []
        for entry in entries:
            db_entry: dict[str, Any] = {
                "id": entry["id"],
                "provider": entry.get("provider", ""),
                "model": entry["model"],
                "capital_krw": entry["capital_krw"],
            }
            if entry.get("target_market"):
                db_entry["target_market"] = entry["target_market"]
            if entry.get("system_prompt"):
                db_entry["system_prompt"] = entry["system_prompt"]
            if entry.get("risk_policy"):
                db_entry["risk_policy"] = entry["risk_policy"]
            if entry.get("disabled_tools"):
                db_entry["disabled_tools"] = entry["disabled_tools"]
            agents_config_for_db.append(db_entry)

        try:
            repo.set_config(
                tenant,
                "agents_config",
                json.dumps(agents_config_for_db, ensure_ascii=False),
                updated_by=user_email or updated_by,
            )
            synced_market = _sync_tenant_market_config(
                tenant=tenant,
                entries=agents_config_for_db,
                tenant_settings=tenant_settings,
                updated_by=user_email or updated_by,
            )

            if api_keys and credential_store is not None:
                credential_store.save_model_keys(
                    tenant_id=tenant,
                    updated_by=user_email or updated_by,
                    providers=api_keys,
                )

            sync_summary = _sync_agent_runtime_state(
                tenant=tenant,
                tenant_settings=tenant_settings,
                entries=agents_config_for_db,
                updated_by=user_email or updated_by,
                sources=_live_market_sources_for_market_value(
                    tenant_settings=tenant_settings,
                    market_value=synced_market,
                ),
            )
            _invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")

            repo.append_runtime_audit_log(
                action="admin_agents_save",
                status="ok",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={
                    "agent_ids": parsed_agent_ids,
                    "models": models,
                    "kis_target_market": synced_market,
                    "agents_config": agents_config_for_db,
                    "api_keys_provided": list(api_keys.keys()),
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
        _prov_alias = provider_alias_map()

        try:
            existing_entries, _ = _load_agent_entries_for_update(tenant)
            existing_entry = next(
                (
                    dict(entry)
                    for entry in existing_entries
                    if isinstance(entry, dict) and str(entry.get("id") or "").strip().lower() == aid
                ),
                None,
            )
            aid, provider, new_entry, raw_api_key = _build_single_agent_entry(
                agent_data=agent_data,
                existing_entry=existing_entry,
                tenant_settings=tenant_settings,
                allowed_providers=allowed_providers,
                provider_aliases=_prov_alias,
            )

            found = False
            for index, entry in enumerate(existing_entries):
                if isinstance(entry, dict) and str(entry.get("id", "")).strip().lower() == aid:
                    existing_entries[index] = new_entry
                    found = True
                    break
            if not found:
                existing_entries.append(new_entry)

            agents_config_for_db = _serialize_agents_config_entries(
                existing_entries,
                tenant_settings=tenant_settings,
            )

            repo.set_config(
                tenant,
                "agents_config",
                json.dumps(agents_config_for_db, ensure_ascii=False),
                updated_by=user_email or "ui-admin",
            )
            synced_market = _sync_tenant_market_config(
                tenant=tenant,
                entries=agents_config_for_db,
                tenant_settings=tenant_settings,
                updated_by=user_email or "ui-admin",
            )

            if raw_api_key and credential_store is not None:
                credential_store.save_model_keys(
                    tenant_id=tenant,
                    updated_by=user_email or "ui-admin",
                    providers={provider: {"api_key": raw_api_key, "model": new_entry["model"]} if new_entry["model"] else {"api_key": raw_api_key}},
                )

            sync_summary = _sync_agent_runtime_state(
                tenant=tenant,
                tenant_settings=tenant_settings,
                entries=agents_config_for_db,
                updated_by=user_email or "ui-admin",
                sources=_live_market_sources_for_market_value(
                    tenant_settings=tenant_settings,
                    market_value=synced_market,
                ),
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
            existing_entries, _ = _load_agent_entries_for_update(tenant)
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

            warning_message = _build_agent_remove_warning(
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
            agents_config_for_db = _serialize_agents_config_entries(
                remaining_entries,
                tenant_settings=tenant_settings,
            )

            repo.set_config(
                tenant,
                "agents_config",
                json.dumps(agents_config_for_db, ensure_ascii=False),
                updated_by=user_email or "ui-admin",
            )
            synced_market = _sync_tenant_market_config(
                tenant=tenant,
                entries=agents_config_for_db,
                tenant_settings=tenant_settings,
                updated_by=user_email or "ui-admin",
            )
            _invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")

            repo.append_runtime_audit_log(
                action="admin_agent_remove_one",
                status="ok",
                user_email=user_email or "ui-admin",
                tenant_id=tenant,
                detail={
                    "agent_id": aid,
                    "remaining_agent_ids": [str(entry.get("id") or "").strip().lower() for entry in agents_config_for_db],
                    "kis_target_market": synced_market,
                    "forced": force_remove,
                },
            )
            return JSONResponse(
                {
                    "ok": True,
                    "message": f"Agent '{aid}' removed",
                    "remaining_agent_ids": [str(entry.get("id") or "").strip().lower() for entry in agents_config_for_db],
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
