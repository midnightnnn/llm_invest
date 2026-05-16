from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

from google.adk.tools.tool_context import ToolContext

from arena.agents.investment_chat.audit import append_chat_audit, config_set
from arena.agents.investment_chat.context import normalize_tenant
from arena.agents.investment_chat.drafts import (
    approval_token,
    load_config_draft,
    save_config_draft,
)
from arena.agents.investment_chat.locks import tenant_lock
from arena.agents.investment_chat.market_scope import account_market_override
from arena.agents.investment_chat.scope import chat_actor_email
from arena.agents.investment_chat.utils import latest_account_snapshot, safe_float, sources_for_settings, utc_iso
from arena.config import Settings
from arena.providers import list_adk_provider_specs, provider_alias_map
from arena.providers.model_discovery import load_model_options_catalog
from arena.tools.registry import ToolEntry
from arena.ui.admin_agent_config import (
    AdminAgentConfigStore,
    build_single_agent_entry,
    serialize_agents_config_entries,
)
from arena.ui.admin_runtime_ops import AdminRuntimeOps

logger = logging.getLogger(__name__)

CONFIG_CHANGE_PROPOSE_ACTION = "chat_config_change_propose"
CONFIG_CHANGE_APPLY_ACTION = "chat_config_change_apply"

_TENANT_CONFIG_JSON_KEYS = {"risk_policy", "disabled_tools", "mcp_servers", "memory_policy"}
_TENANT_CONFIG_STRING_KEYS = {
    "system_prompt",
    "memory_compactor_prompt",
}
_TENANT_CONFIG_NUMBER_KEYS = {
    "sleeve_capital_krw",
    "research_max_tickers",
    "research_mover_top_n",
    "research_earnings_lookahead_days",
}
_TENANT_CONFIG_BOOL_KEYS = {"research_enabled"}
_CHAT_AGENT_ALLOWED_FIELDS = {
    "provider",
    "model",
    "disabled_tools",
    "llm_params",
    "model_routing",
    "memory_compaction_model",
}
ConfigChangeAction = Literal["update", "upsert", "add", "remove"]
CapitalAllocationMode = Literal["unchanged", "fixed_krw", "add_krw", "account_percent", "whole_account"]


def _confirmation_state_key(tool_context: ToolContext) -> str:
    function_call_id = str(getattr(tool_context, "function_call_id", "") or "").strip()
    return f"investment_chat.config_confirmation.{function_call_id or 'unknown'}"


def _confirmation_payload(draft: dict[str, Any]) -> dict[str, Any]:
    diffs = draft.get("diffs") if isinstance(draft.get("diffs"), list) else []
    return {
        "action": "apply_config_change",
        "scope": draft.get("scope") or "",
        "config_action": draft.get("action") or "",
        "summary": draft.get("summary") or "",
        "diff_count": len(diffs),
        "diffs": diffs[:20],
    }


def _confirmation_hint(draft: dict[str, Any]) -> str:
    payload = _confirmation_payload(draft)
    scope = str(payload.get("scope") or "").strip().lower()
    scope_label = {
        "agent": "투자 에이전트",
        "chat_agent": "투자챗봇",
        "tenant": "테넌트",
    }.get(scope, "설정")
    summary = str(payload.get("summary") or "").strip()
    return (
        f"{scope_label} 설정 변경을 적용할까요? {summary} "
        "ADK Web 확인창에서 Confirmed 체크박스를 체크한 뒤 Submit을 눌러야 승인됩니다."
    )


def load_chat_agent_config(repo: Any, *, tenant_id: str) -> dict[str, Any]:
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return {}
    try:
        raw = getter(tenant_id, "investment_chat_config")
    except TypeError:
        raw = getter(tenant_id=tenant_id, config_key="investment_chat_config")
    except Exception:
        return {}
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("change_json must be a JSON object")
    return parsed


def _json_object_field(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return parsed


def _audit_detail(row: dict[str, Any]) -> dict[str, Any]:
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


def _settings_agent_entries(settings: Settings) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for aid in [str(x or "").strip().lower() for x in settings.agent_ids if str(x or "").strip()]:
        ac = (settings.agent_configs or {}).get(aid)
        if ac is None:
            continue
        entry: dict[str, Any] = {
            "id": aid,
            "provider": str(ac.provider or "").strip().lower(),
            "model": str(ac.model or "").strip(),
            "capital_krw": safe_float(ac.capital_krw, settings.sleeve_capital_krw),
        }
        if ac.target_market:
            entry["target_market"] = str(ac.target_market).strip().lower()
        if ac.system_prompt:
            entry["system_prompt"] = str(ac.system_prompt).strip()
        if isinstance(ac.risk_overrides, dict) and ac.risk_overrides:
            entry["risk_policy"] = dict(ac.risk_overrides)
        if isinstance(ac.disabled_tools, list):
            entry["disabled_tools"] = [str(x).strip() for x in ac.disabled_tools if str(x).strip()]
        if isinstance(ac.llm_params, dict) and ac.llm_params:
            entry["llm_params"] = dict(ac.llm_params)
        memory_model = str(getattr(ac, "memory_compaction_model", "") or "").strip()
        if memory_model:
            entry["memory_compaction_model"] = memory_model
        entries.append(entry)
    return entries


def _latest_agent_entries_for_apply(repo: Any, *, settings: Settings, tenant: str) -> list[dict[str, Any]]:
    store = _admin_config_store(repo, settings)
    entries, _has_explicit = store.load_for_update(tenant)
    return [dict(entry) for entry in entries if isinstance(entry, dict)]


def _merge_agent_apply_entries(
    *,
    repo: Any,
    settings: Settings,
    tenant: str,
    draft: dict[str, Any],
    draft_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Applies one agent draft onto the latest config to avoid stale full-list overwrites."""
    if str(draft.get("scope") or "").strip().lower() != "agent":
        return draft_entries
    change = draft.get("change") if isinstance(draft.get("change"), dict) else {}
    aid = str(draft.get("agent_id") or change.get("agent_id") or change.get("id") or "").strip().lower()
    action = str(draft.get("action") or change.get("action") or "update").strip().lower()
    if not aid or action not in {"update", "upsert", "add", "remove"}:
        return draft_entries

    latest_entries = _latest_agent_entries_for_apply(repo, settings=settings, tenant=tenant)
    if action == "remove":
        return [
            entry
            for entry in latest_entries
            if str(entry.get("id") or "").strip().lower() != aid
        ]

    replacement = next(
        (
            dict(entry)
            for entry in draft_entries
            if str(entry.get("id") or "").strip().lower() == aid
        ),
        None,
    )
    if replacement is None:
        return draft_entries

    merged = list(latest_entries)
    for index, entry in enumerate(merged):
        if str(entry.get("id") or "").strip().lower() == aid:
            merged[index] = replacement
            break
    else:
        merged.append(replacement)
    return merged


def _admin_config_store(repo: Any, settings: Settings) -> AdminAgentConfigStore:
    def _view_model(_tenant: str) -> dict[str, Any]:
        return {"agents_config": _settings_agent_entries(settings)}

    return AdminAgentConfigStore(repo=repo, current_admin_view_model=_view_model)


def _is_live_mode(settings: Settings | None) -> bool:
    return str(getattr(settings, "trading_mode", "") or "").strip().lower() == "live"


def _live_market_sources(settings: Settings | None) -> list[str] | None:
    return sources_for_settings(settings) if settings is not None else None


def _runtime_ops(repo: Any) -> AdminRuntimeOps:
    return AdminRuntimeOps(
        repo=repo,
        is_live_mode=_is_live_mode,
        live_market_sources=_live_market_sources,
        safe_float=safe_float,
    )


def _resolve_capital_allocation(
    *,
    fields: dict[str, Any],
    repo: Any,
    tenant_id: str,
    fallback_capital: float,
) -> dict[str, Any] | None:
    raw = fields.get("capital_allocation") or fields.get("sleeve_allocation")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("capital_allocation must be an object")
    mode = str(raw.get("mode") or "").strip().lower()
    if mode in {"fixed", "fixed_krw", "amount"}:
        amount = safe_float(raw.get("amount_krw", raw.get("capital_krw")), fallback_capital)
        if amount <= 0:
            raise ValueError("fixed capital allocation requires amount_krw > 0")
    elif mode in {"add", "add_krw", "increment", "increment_krw"}:
        increment = safe_float(raw.get("amount_krw", raw.get("capital_krw")), 0.0)
        if increment <= 0:
            raise ValueError("add_krw capital allocation requires amount_krw > 0")
        amount = safe_float(fallback_capital) + increment
    elif mode in {"percent", "account_percent", "account_ratio"}:
        percent = safe_float(raw.get("percent", raw.get("ratio")), 0.0)
        if percent <= 0 or percent > 100:
            raise ValueError("account_percent allocation requires 0 < percent <= 100")
        snapshot = latest_account_snapshot(
            repo,
            tenant_id=tenant_id,
            market_scope=account_market_override(repo, tenant_id=tenant_id),
        )
        if snapshot is None:
            raise ValueError("account_percent allocation requires a stored account snapshot")
        amount = safe_float(getattr(snapshot, "total_equity_krw", 0.0)) * percent / 100.0
    elif mode in {"whole", "whole_account", "account"}:
        snapshot = latest_account_snapshot(
            repo,
            tenant_id=tenant_id,
            market_scope=account_market_override(repo, tenant_id=tenant_id),
        )
        if snapshot is None:
            raise ValueError("whole_account allocation requires a stored account snapshot")
        amount = safe_float(getattr(snapshot, "total_equity_krw", 0.0))
    else:
        raise ValueError("capital_allocation.mode must be fixed_krw, account_percent, or whole_account")
    if amount <= 0:
        raise ValueError("resolved capital allocation must be > 0")
    fields["capital_krw"] = float(amount)
    return {"mode": mode, "resolved_capital_krw": float(amount), **dict(raw)}


def _entry_summary(before: dict[str, Any] | None, after: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "provider",
        "model",
        "capital_krw",
        "target_market",
        "system_prompt",
        "risk_policy",
        "disabled_tools",
        "llm_params",
        "memory_compaction_model",
    ]
    old = before or {}
    diffs: list[dict[str, Any]] = []
    for key in keys:
        if old.get(key) != after.get(key):
            diffs.append({"field": key, "before": old.get(key), "after": after.get(key)})
    return diffs


def _build_agent_change(
    *,
    repo: Any,
    settings: Settings,
    tenant: str,
    change: dict[str, Any],
) -> dict[str, Any]:
    action = str(change.get("action") or "update").strip().lower()
    aid = str(change.get("agent_id") or change.get("id") or "").strip().lower()
    if not aid:
        raise ValueError("agent_id is required for agent config changes")

    store = _admin_config_store(repo, settings)
    entries, _has_explicit = store.load_for_update(tenant)
    normalized_entries = [dict(entry) for entry in entries if isinstance(entry, dict)]
    existing_entry = next(
        (
            dict(entry)
            for entry in normalized_entries
            if str(entry.get("id") or "").strip().lower() == aid
        ),
        None,
    )
    if action == "remove":
        next_entries = [
            entry
            for entry in normalized_entries
            if str(entry.get("id") or "").strip().lower() != aid
        ]
        summary = f"Remove investment agent '{aid}'"
        return {
            "scope": "agent",
            "action": action,
            "agent_id": aid,
            "summary": summary,
            "diffs": [{"field": "agent", "before": existing_entry, "after": None}],
            "apply": {"kind": "agents_config", "agents_config": next_entries},
        }
    if action not in {"update", "upsert", "add"}:
        raise ValueError("agent config action must be update, upsert, add, or remove")

    fields = change.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("fields object is required for agent update")
    agent_data = {"id": aid, **dict(fields)}
    allocation_meta = _resolve_capital_allocation(
        fields=agent_data,
        repo=repo,
        tenant_id=tenant,
        fallback_capital=safe_float(
            (existing_entry or {}).get("capital_krw"),
            settings.sleeve_capital_krw,
        ),
    )

    allowed_providers = {spec.provider_id for spec in list_adk_provider_specs()}
    _aid, _provider, new_entry, _raw_api_key = build_single_agent_entry(
        agent_data=agent_data,
        existing_entry=existing_entry,
        tenant_settings=settings,
        allowed_providers=allowed_providers,
        provider_aliases=provider_alias_map(),
        safe_float=safe_float,
    )
    if allocation_meta:
        new_entry["capital_allocation"] = allocation_meta

    found = False
    for index, entry in enumerate(normalized_entries):
        if str(entry.get("id") or "").strip().lower() == aid:
            normalized_entries[index] = new_entry
            found = True
            break
    if not found:
        normalized_entries.append(new_entry)
    agents_config = serialize_agents_config_entries(
        normalized_entries,
        tenant_settings=settings,
        safe_float=safe_float,
    )
    summary = f"Update investment agent '{aid}'"
    diffs = _entry_summary(existing_entry, new_entry)
    return {
        "scope": "agent",
        "action": "update",
        "agent_id": aid,
        "summary": summary,
        "diffs": diffs,
        "apply": {"kind": "agents_config", "agents_config": agents_config},
    }


def _normalize_tenant_fields(fields: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    allowed = _TENANT_CONFIG_JSON_KEYS | _TENANT_CONFIG_STRING_KEYS | _TENANT_CONFIG_NUMBER_KEYS | _TENANT_CONFIG_BOOL_KEYS
    for key, value in fields.items():
        token = str(key or "").strip()
        if token not in allowed:
            raise ValueError(f"tenant config key is not allowed: {token}")
        if token in _TENANT_CONFIG_JSON_KEYS:
            if not isinstance(value, (dict, list)):
                raise ValueError(f"{token} must be an object or array")
            normalized[token] = json.dumps(value, ensure_ascii=False)
        elif token in _TENANT_CONFIG_NUMBER_KEYS:
            number = safe_float(value, -1.0)
            if number < 0:
                raise ValueError(f"{token} must be numeric")
            normalized[token] = str(number)
        elif token in _TENANT_CONFIG_BOOL_KEYS:
            normalized[token] = "true" if bool(value) else "false"
        else:
            normalized[token] = str(value or "").strip()
    return normalized


def _build_tenant_change(change: dict[str, Any]) -> dict[str, Any]:
    fields = change.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("fields object is required for tenant config changes")
    values = _normalize_tenant_fields(fields)
    diffs = [{"field": key, "before": None, "after": value} for key, value in sorted(values.items())]
    return {
        "scope": "tenant",
        "action": "update",
        "summary": f"Update tenant settings: {', '.join(sorted(values))}",
        "diffs": diffs,
        "apply": {"kind": "tenant_config", "values": values},
    }


def _build_chat_agent_change(change: dict[str, Any]) -> dict[str, Any]:
    fields = change.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError("fields object is required for chat_agent config changes")
    clean: dict[str, Any] = {}
    for key, value in fields.items():
        token = str(key or "").strip()
        if token not in _CHAT_AGENT_ALLOWED_FIELDS:
            raise ValueError(f"chat_agent field is not allowed: {token}")
        if token in {"disabled_tools"}:
            if not isinstance(value, list):
                raise ValueError(f"{token} must be an array")
            clean[token] = [str(item).strip() for item in value if str(item).strip()]
        elif token in {"llm_params", "model_routing"}:
            if not isinstance(value, dict):
                raise ValueError(f"{token} must be an object")
            clean[token] = dict(value)
        else:
            clean[token] = str(value or "").strip()
    values = {"investment_chat_config": json.dumps(clean, ensure_ascii=False)}
    return {
        "scope": "chat_agent",
        "action": "update",
        "summary": f"Update investment chat agent settings: {', '.join(sorted(fields))}",
        "diffs": [{"field": key, "before": None, "after": value} for key, value in sorted(fields.items())],
        "apply": {"kind": "tenant_config", "values": values},
    }


def _build_config_change(
    *,
    repo: Any,
    settings: Settings,
    tenant: str,
    change: dict[str, Any],
) -> dict[str, Any]:
    scope = str(change.get("scope") or "").strip().lower()
    if scope == "agent":
        return _build_agent_change(repo=repo, settings=settings, tenant=tenant, change=change)
    if scope == "tenant":
        return _build_tenant_change(change)
    if scope == "chat_agent":
        return _build_chat_agent_change(change)
    raise ValueError("scope must be agent, tenant, or chat_agent")


def config_draft_status_row(token: str, draft: dict[str, Any]) -> dict[str, Any]:
    status = str(draft.get("status") or "").strip().lower()
    return {
        "approval_token": token,
        "status": status,
        "submittable": status == "draft",
        "created_at": draft.get("created_at") or "",
        "applied_at": draft.get("applied_at") or "",
        "expires_at": draft.get("expires_at") or "",
        "scope": draft.get("scope") or "",
        "action": draft.get("action") or "",
        "summary": draft.get("summary") or "",
        "diffs": draft.get("diffs") or [],
        "rationale": draft.get("rationale") or "",
        "message": draft.get("message") or draft.get("error") or "",
    }


def recent_config_drafts(repo: Any, *, tenant_id: str, limit: int = 5) -> list[dict[str, Any]]:
    loader = getattr(repo, "recent_runtime_audit_logs", None)
    if not callable(loader):
        return []
    try:
        audit_rows = loader(limit=max(20, min(200, int(limit or 5) * 20))) or []
    except TypeError:
        audit_rows = loader(max(20, min(200, int(limit or 5) * 20))) or []
    except Exception:
        return []

    tenant = normalize_tenant(tenant_id)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in audit_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("tenant_id") or "").strip().lower() not in {"", tenant}:
            continue
        if str(row.get("action") or "").strip() != CONFIG_CHANGE_PROPOSE_ACTION:
            continue
        detail = _audit_detail(row)
        token = str(detail.get("approval_token") or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        draft = load_config_draft(repo, tenant_id=tenant, token=token)
        if isinstance(draft, dict):
            if str(draft.get("approval_channel") or "").strip().lower() == "adk_tool_confirmation":
                continue
            out.append(config_draft_status_row(token, draft))
        if len(out) >= max(1, min(int(limit or 5), 20)):
            break
    return out


def _apply_agents_config(
    *,
    repo: Any,
    settings: Settings,
    tenant: str,
    draft: dict[str, Any],
    updated_by: str,
) -> dict[str, Any]:
    payload = draft.get("apply") if isinstance(draft.get("apply"), dict) else {}
    entries = payload.get("agents_config")
    if not isinstance(entries, list):
        raise ValueError("draft is missing agents_config apply payload")
    draft_entries = [dict(entry) for entry in entries if isinstance(entry, dict)]
    merged_entries = _merge_agent_apply_entries(
        repo=repo,
        settings=settings,
        tenant=tenant,
        draft=draft,
        draft_entries=draft_entries,
    )
    agents_config = serialize_agents_config_entries(
        merged_entries,
        tenant_settings=settings,
        safe_float=safe_float,
    )
    config_set(
        repo,
        tenant,
        "agents_config",
        json.dumps(agents_config, ensure_ascii=False),
        updated_by=updated_by,
    )
    store = _admin_config_store(repo, settings)
    synced_market = store.sync_market(
        tenant=tenant,
        entries=agents_config,
        tenant_settings=settings,
        updated_by=updated_by,
    )
    sync_summary = _runtime_ops(repo).sync_runtime_state(
        tenant=tenant,
        tenant_settings=settings,
        entries=agents_config,
        updated_by=updated_by,
        sources=sources_for_settings(settings),
    )
    return {
        "config_keys": ["agents_config"],
        "agent_ids": [str(entry.get("id") or "").strip().lower() for entry in agents_config],
        "kis_target_market": synced_market,
        "sync": sync_summary,
    }


def _apply_tenant_config(
    *,
    repo: Any,
    tenant: str,
    draft: dict[str, Any],
    updated_by: str,
) -> dict[str, Any]:
    payload = draft.get("apply") if isinstance(draft.get("apply"), dict) else {}
    values = payload.get("values")
    if not isinstance(values, dict):
        raise ValueError("draft is missing tenant config values")
    keys: list[str] = []
    for key, value in values.items():
        token = str(key or "").strip()
        if not token:
            continue
        config_set(repo, tenant, token, str(value or ""), updated_by=updated_by)
        keys.append(token)
    return {"config_keys": keys}


def build_config_tool_entries(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    invalidate_tenant_cache: Callable[..., Any] | None = None,
) -> list[ToolEntry]:
    return _build_config_tool_entries(
        repo=repo,
        settings=settings,
        tenant_id=tenant_id,
        include_internal_bridge=False,
        invalidate_tenant_cache=invalidate_tenant_cache,
    )


def build_config_bridge_tool_entries(*, repo: Any, settings: Settings, tenant_id: str) -> list[ToolEntry]:
    return _build_config_tool_entries(
        repo=repo,
        settings=settings,
        tenant_id=tenant_id,
        include_internal_bridge=True,
        invalidate_tenant_cache=None,
    )[-1:]


def _build_config_tool_entries(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
    include_internal_bridge: bool,
    invalidate_tenant_cache: Callable[..., Any] | None,
) -> list[ToolEntry]:
    tenant = normalize_tenant(tenant_id)

    def _create_config_change_draft(change: dict[str, Any], rationale: str = "") -> dict[str, Any]:
        try:
            built = _build_config_change(repo=repo, settings=settings, tenant=tenant, change=change)
        except Exception as exc:
            return {"status": "error", "tenant_id": tenant, "error": str(exc)}

        now = datetime.now(timezone.utc)
        token = approval_token(
            {
                "tenant_id": tenant,
                "change": change,
                "nonce": uuid4().hex,
            }
        )
        expires_at = now + timedelta(minutes=30)
        user_email = chat_actor_email()
        draft = {
            "approval_token": token,
            "status": "draft",
            "created_at": utc_iso(now),
            "expires_at": utc_iso(expires_at),
            "tenant_id": tenant,
            "scope": built["scope"],
            "action": built["action"],
            "summary": built["summary"],
            "diffs": built["diffs"],
            "rationale": str(rationale or "").strip(),
            "change": change,
            "apply": built["apply"],
            "required_confirmation": f"CONFIRM {token}",
            "approved_by": user_email,
        }
        save_config_draft(repo, tenant_id=tenant, token=token, draft=draft)
        append_chat_audit(
            repo,
            tenant_id=tenant,
            action=CONFIG_CHANGE_PROPOSE_ACTION,
            status="ok",
            detail={
                "approval_token": token,
                "scope": draft["scope"],
                "action": draft["action"],
                "summary": draft["summary"],
            },
            user_email=user_email,
        )
        return {
            "status": "ok",
            "tenant_id": tenant,
            "approval_token": token,
            "approval_required": True,
            "required_confirmation": f"CONFIRM {token}",
            "expires_at": draft["expires_at"],
            "summary": draft["summary"],
            "diffs": draft["diffs"],
            "message": "설정 변경 초안이 생성되었습니다. UI 승인 버튼을 눌러야 적용됩니다.",
        }

    def _propose_config_change_with_confirmation(
        change: dict[str, Any],
        *,
        rationale: str = "",
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        if tool_context is None:
            return _create_config_change_draft(change, rationale=rationale)

        state_key = _confirmation_state_key(tool_context)
        confirmation = getattr(tool_context, "tool_confirmation", None)
        state = getattr(tool_context, "state", None)
        if confirmation is not None:
            token = ""
            if state is not None:
                token = str(state.get(state_key) or "").strip()
            if not bool(getattr(confirmation, "confirmed", False)):
                confirmation_payload = getattr(confirmation, "payload", None)
                reason = "confirmed_checkbox_unchecked" if confirmation_payload is not None else "not_confirmed"
                if token:
                    draft = load_config_draft(repo, tenant_id=tenant, token=token)
                    if isinstance(draft, dict) and str(draft.get("status") or "").strip().lower() == "draft":
                        draft["status"] = "rejected"
                        draft["rejected_at"] = utc_iso()
                        draft["rejection_reason"] = reason
                        save_config_draft(repo, tenant_id=tenant, token=token, draft=draft)
                    append_chat_audit(
                        repo,
                        tenant_id=tenant,
                        action=CONFIG_CHANGE_APPLY_ACTION,
                        status="blocked",
                        detail={
                            "approval_token": token,
                            "stage": "adk_tool_confirmation",
                            "reason": reason,
                        },
                        user_email=chat_actor_email(),
                    )
                return {
                    "status": "rejected",
                    "tenant_id": tenant,
                    "reason": reason,
                    "apply_status": "not_applied",
                    "message": (
                        "ADK Web에서 confirmed=false가 반환되어 설정 변경을 적용하지 않았습니다. "
                        "승인하려면 ADK Web 확인창에서 Confirmed 체크박스를 체크한 뒤 Submit을 눌러야 합니다."
                        if reason == "confirmed_checkbox_unchecked"
                        else "ADK Web에서 설정 변경이 확인되지 않아 적용하지 않았습니다."
                    ),
                }
            if not token:
                return {
                    "status": "blocked",
                    "tenant_id": tenant,
                    "error": "ADK tool confirmation state was missing; config change was not applied.",
                    "apply_status": "not_applied",
                }
            result = apply_approved_config_change(approval_token=token, confirmation_text=f"CONFIRM {token}")
            if callable(invalidate_tenant_cache) and result.get("status") in {"applied", "already_applied"}:
                try:
                    invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")
                except Exception:
                    logger.warning("[yellow]Investment chat config cache invalidation failed[/yellow] tenant=%s", tenant)
            result["approval_ui"] = "adk_tool_confirmation"
            return result

        draft_result = _create_config_change_draft(change, rationale=rationale)
        if str(draft_result.get("status") or "").strip().lower() != "ok":
            return draft_result
        token = str(draft_result.get("approval_token") or "").strip()
        draft = load_config_draft(repo, tenant_id=tenant, token=token)
        if not isinstance(draft, dict):
            return {
                "status": "blocked",
                "tenant_id": tenant,
                "error": "config change draft could not be loaded; config change was not applied.",
                "apply_status": "not_applied",
            }
        draft["approval_channel"] = "adk_tool_confirmation"
        save_config_draft(repo, tenant_id=tenant, token=token, draft=draft)
        if state is not None:
            state[state_key] = token
        tool_context.request_confirmation(
            hint=_confirmation_hint(draft),
            payload=_confirmation_payload(draft),
        )
        return {
            "status": "waiting_for_confirmation",
            "tenant_id": tenant,
            "approval_required": True,
            "approval_ui": "adk_tool_confirmation",
            "apply_status": "not_applied",
            "expires_at": draft.get("expires_at") or "",
            "scope": draft.get("scope") or "",
            "action": draft.get("action") or "",
            "summary": draft.get("summary") or "",
            "diffs": draft.get("diffs") or [],
            "message": "ADK tool confirmation is required before this config change can be applied.",
        }

    def propose_agent_config_change(
        agent_id: str,
        action: ConfigChangeAction = "update",
        provider: str = "",
        model: str = "",
        capital_krw: Optional[float] = None,
        capital_allocation_mode: CapitalAllocationMode = "unchanged",
        capital_allocation_percent: Optional[float] = None,
        capital_allocation_amount_krw: Optional[float] = None,
        target_market: str = "",
        system_prompt: str = "",
        risk_policy_json: str = "",
        disabled_tools: Optional[list[str]] = None,
        llm_params_json: str = "",
        memory_compaction_model: str = "",
        rationale: str = "",
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Creates an investment-agent settings draft and asks for ADK confirmation before it applies."""
        fields: dict[str, Any] = {}
        for key, value in {
            "provider": provider,
            "model": model,
            "target_market": target_market,
            "system_prompt": system_prompt,
            "memory_compaction_model": memory_compaction_model,
        }.items():
            text = str(value or "").strip()
            if text:
                fields[key] = text
        if capital_krw is not None:
            fields["capital_krw"] = float(capital_krw)
        allocation_mode = str(capital_allocation_mode or "").strip().lower()
        if allocation_mode in {"unchanged", "default", "none"}:
            allocation_mode = ""
        if allocation_mode:
            allocation: dict[str, Any] = {"mode": allocation_mode}
            if capital_allocation_percent is not None:
                allocation["percent"] = float(capital_allocation_percent)
            if capital_allocation_amount_krw is not None:
                allocation["amount_krw"] = float(capital_allocation_amount_krw)
            elif capital_krw is not None:
                allocation["amount_krw"] = float(capital_krw)
            fields["capital_allocation"] = allocation
        try:
            risk_policy = _json_object_field(risk_policy_json, field_name="risk_policy_json")
            llm_params = _json_object_field(llm_params_json, field_name="llm_params_json")
        except Exception as exc:
            return {"status": "error", "tenant_id": tenant, "error": str(exc)}
        if risk_policy:
            fields["risk_policy"] = risk_policy
        if llm_params:
            fields["llm_params"] = llm_params
        if disabled_tools is not None:
            fields["disabled_tools"] = [str(tool_id).strip() for tool_id in disabled_tools if str(tool_id).strip()]
        change = {
            "scope": "agent",
            "action": action,
            "agent_id": str(agent_id or "").strip().lower(),
            "fields": fields,
        }
        return _propose_config_change_with_confirmation(change, rationale=rationale, tool_context=tool_context)

    def propose_chat_agent_config_change(
        provider: str = "",
        model: str = "",
        router_model: str = "",
        utility_model: str = "",
        cheap_model: str = "",
        disabled_tools: Optional[list[str]] = None,
        llm_params_json: str = "",
        model_routing_json: str = "",
        memory_compaction_model: str = "",
        rationale: str = "",
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Creates an investment-chat-agent settings draft and asks for ADK confirmation before it applies."""
        fields: dict[str, Any] = {}
        for key, value in {
            "provider": provider,
            "model": model,
            "memory_compaction_model": memory_compaction_model,
        }.items():
            text = str(value or "").strip()
            if text:
                fields[key] = text
        if disabled_tools is not None:
            fields["disabled_tools"] = [str(tool_id).strip() for tool_id in disabled_tools if str(tool_id).strip()]
        try:
            llm_params = _json_object_field(llm_params_json, field_name="llm_params_json")
            model_routing = _json_object_field(model_routing_json, field_name="model_routing_json")
        except Exception as exc:
            return {"status": "error", "tenant_id": tenant, "error": str(exc)}
        shared_cheap_model = str(cheap_model or "").strip()
        router_model_token = str(router_model or shared_cheap_model).strip()
        utility_model_token = str(utility_model or shared_cheap_model).strip()
        if router_model_token:
            model_routing["router_model"] = router_model_token
        if utility_model_token:
            model_routing["utility_model"] = utility_model_token
        if llm_params:
            fields["llm_params"] = llm_params
        if model_routing:
            fields["model_routing"] = model_routing
        return _propose_config_change_with_confirmation(
            {"scope": "chat_agent", "action": "update", "fields": fields},
            rationale=rationale,
            tool_context=tool_context,
        )

    def list_chat_model_options(provider: str = "") -> dict[str, Any]:
        """Returns cached provider model options discovered from the tenant's saved API key."""
        catalog = load_model_options_catalog(repo, tenant_id=tenant)
        providers = catalog.get("providers") if isinstance(catalog.get("providers"), dict) else {}
        chat_config = load_chat_agent_config(repo, tenant_id=tenant)
        routing = chat_config.get("model_routing") if isinstance(chat_config.get("model_routing"), dict) else {}
        current = {
            "provider": provider_alias_map().get(
                str(chat_config.get("provider") or "").strip().lower(),
                str(chat_config.get("provider") or "").strip().lower(),
            ),
            "advisor_model": str(chat_config.get("model") or "").strip(),
            "router_model": str(routing.get("router_model") or routing.get("cheap_model") or "").strip(),
            "utility_model": str(routing.get("utility_model") or routing.get("cheap_model") or "").strip(),
        }
        if not isinstance(providers, dict) or not providers:
            return {
                "status": "missing",
                "tenant_id": tenant,
                "current": current,
                "error": "model options have not been fetched yet; refresh models in settings or the API-key setup flow",
            }
        provider_token = str(provider or "").strip().lower()
        if provider_token:
            provider_token = provider_alias_map().get(provider_token, provider_token)
            item = providers.get(provider_token)
            if not isinstance(item, dict):
                return {
                    "status": "missing",
                    "tenant_id": tenant,
                    "provider": provider_token,
                    "current": current,
                    "error": "cached model options not found for provider",
                }
            return {
                "status": "ok",
                "tenant_id": tenant,
                "provider": provider_token,
                "current": current,
                "advisor_models": list(item.get("advisor_models") or []),
                "router_models": list(item.get("router_models") or []),
                "utility_models": list(item.get("utility_models") or []),
                "fetched_at": str(item.get("fetched_at") or ""),
            }
        return {
            "status": "ok",
            "tenant_id": tenant,
            "current": current,
            "providers": {
                str(key): {
                    "advisor_models": list(value.get("advisor_models") or []),
                    "router_models": list(value.get("router_models") or []),
                    "utility_models": list(value.get("utility_models") or []),
                    "fetched_at": str(value.get("fetched_at") or ""),
                }
                for key, value in providers.items()
                if isinstance(value, dict)
            },
        }

    def propose_tenant_config_change(
        system_prompt: str = "",
        memory_compactor_prompt: str = "",
        sleeve_capital_krw: Optional[float] = None,
        research_max_tickers: Optional[int] = None,
        research_mover_top_n: Optional[int] = None,
        research_earnings_lookahead_days: Optional[int] = None,
        research_enabled: Optional[bool] = None,
        risk_policy_json: str = "",
        disabled_tools: Optional[list[str]] = None,
        mcp_servers_json: str = "",
        memory_policy_json: str = "",
        rationale: str = "",
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Creates a tenant-level settings draft and asks for ADK confirmation before it applies."""
        fields: dict[str, Any] = {}
        for key, value in {
            "system_prompt": system_prompt,
            "memory_compactor_prompt": memory_compactor_prompt,
        }.items():
            text = str(value or "").strip()
            if text:
                fields[key] = text
        for key, value in {
            "sleeve_capital_krw": sleeve_capital_krw,
            "research_max_tickers": research_max_tickers,
            "research_mover_top_n": research_mover_top_n,
            "research_earnings_lookahead_days": research_earnings_lookahead_days,
        }.items():
            if value is not None:
                fields[key] = value
        if research_enabled is not None:
            fields["research_enabled"] = bool(research_enabled)
        if disabled_tools is not None:
            fields["disabled_tools"] = [str(tool_id).strip() for tool_id in disabled_tools if str(tool_id).strip()]
        try:
            for key, raw in {
                "risk_policy": risk_policy_json,
                "mcp_servers": mcp_servers_json,
                "memory_policy": memory_policy_json,
            }.items():
                parsed = _json_object_field(raw, field_name=f"{key}_json")
                if parsed:
                    fields[key] = parsed
        except Exception as exc:
            return {"status": "error", "tenant_id": tenant, "error": str(exc)}
        return _propose_config_change_with_confirmation(
            {"scope": "tenant", "action": "update", "fields": fields},
            rationale=rationale,
            tool_context=tool_context,
        )

    def get_config_change_status(approval_token: str = "", limit: int = 5) -> dict[str, Any]:
        """Reads recent investment-chat settings approval drafts and results."""
        token = str(approval_token or "").strip()
        if token:
            draft = load_config_draft(repo, tenant_id=tenant, token=token)
            drafts = [config_draft_status_row(token, draft)] if isinstance(draft, dict) else []
        else:
            drafts = recent_config_drafts(repo, tenant_id=tenant, limit=limit)
        return {"status": "ok", "tenant_id": tenant, "count": len(drafts), "drafts": drafts}

    def apply_approved_config_change(approval_token: str, confirmation_text: str) -> dict[str, Any]:
        """Internal backend/UI bridge for applying a stored config draft after explicit approval."""
        token = str(approval_token or "").strip()
        required = f"CONFIRM {token}"
        if not token:
            return {"status": "blocked", "error": "approval_token is required", "required_confirmation": required}
        if str(confirmation_text or "").strip() != required:
            return {
                "status": "blocked",
                "error": "confirmation_text must exactly match the required confirmation phrase",
                "required_confirmation": required,
            }
        draft = load_config_draft(repo, tenant_id=tenant, token=token)
        if not isinstance(draft, dict):
            return {"status": "missing", "approval_token": token, "error": "approval token not found or expired"}
        current_status = str(draft.get("status") or "").strip().lower()
        if current_status == "applied":
            return {
                "status": "already_applied",
                "approval_token": token,
                "summary": draft.get("summary") or "",
                "apply_result": draft.get("apply_result") or {},
            }
        if current_status != "draft":
            return {
                "status": "blocked",
                "approval_token": token,
                "error": f"draft is not applicable: status={draft.get('status')}",
            }
        try:
            expires_at = datetime.fromisoformat(str(draft.get("expires_at") or "").replace("Z", "+00:00"))
        except ValueError:
            expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        if expires_at < datetime.now(timezone.utc):
            draft["status"] = "expired"
            save_config_draft(repo, tenant_id=tenant, token=token, draft=draft)
            return {"status": "expired", "approval_token": token, "error": "approval token expired"}

        user_email = str(draft.get("approved_by") or chat_actor_email() or "").strip().lower()
        updated_by = user_email or "investment_chat"
        payload = draft.get("apply") if isinstance(draft.get("apply"), dict) else {}
        kind = str(payload.get("kind") or "").strip()
        try:
            with tenant_lock(tenant):
                if kind == "agents_config":
                    apply_result = _apply_agents_config(
                        repo=repo,
                        settings=settings,
                        tenant=tenant,
                        draft=draft,
                        updated_by=updated_by,
                    )
                elif kind == "tenant_config":
                    apply_result = _apply_tenant_config(
                        repo=repo,
                        tenant=tenant,
                        draft=draft,
                        updated_by=updated_by,
                    )
                else:
                    raise ValueError(f"unsupported config draft kind: {kind}")
        except Exception as exc:
            draft["status"] = "error"
            draft["error"] = str(exc)
            save_config_draft(repo, tenant_id=tenant, token=token, draft=draft)
            append_chat_audit(
                repo,
                tenant_id=tenant,
                action=CONFIG_CHANGE_APPLY_ACTION,
                status="error",
                detail={"approval_token": token, "error": str(exc)[:500]},
                user_email=user_email,
            )
            logger.warning("[yellow]Investment chat config apply failed[/yellow] tenant=%s err=%s", tenant, exc)
            return {"status": "error", "approval_token": token, "error": str(exc)}

        draft["status"] = "applied"
        draft["applied_at"] = utc_iso()
        draft["apply_result"] = apply_result
        save_config_draft(repo, tenant_id=tenant, token=token, draft=draft)
        append_chat_audit(
            repo,
            tenant_id=tenant,
            action=CONFIG_CHANGE_APPLY_ACTION,
            status="ok",
            detail={"approval_token": token, "summary": draft.get("summary") or "", "result": apply_result},
            user_email=user_email,
        )
        return {
            "status": "applied",
            "approval_token": token,
            "summary": draft.get("summary") or "",
            "apply_result": apply_result,
            "message": "설정 변경을 적용했습니다.",
        }

    entries = [
        ToolEntry(
            tool_id="list_chat_model_options",
            name="list_chat_model_options",
            description=(
                "Lists the provider-supported chat model IDs that were last discovered using the tenant's saved "
                "LLM API key. Use this before proposing chat advisor, router, or utility model changes."
            ),
            category="admin",
            callable=list_chat_model_options,
            tier="core",
            label_ko="채팅 모델 목록",
            sort_order=19,
        ),
        ToolEntry(
            tool_id="propose_agent_config_change",
            name="propose_agent_config_change",
            description=(
                "Creates an investment agent settings change draft. The tool never applies settings directly; "
                "ADK tool confirmation is required before it applies. Select the capital allocation mode that "
                "matches whether the user wants a final sleeve amount, an increase, an account-percentage "
                "assignment, or whole-account assignment."
            ),
            category="admin",
            callable=propose_agent_config_change,
            tier="core",
            label_ko="투자 에이전트 설정 초안",
            sort_order=20,
        ),
        ToolEntry(
            tool_id="propose_chat_agent_config_change",
            name="propose_chat_agent_config_change",
            description=(
                "Creates an investment chat agent settings change draft for advisor, router, utility, provider, "
                "tool, and memory settings. "
                "The tool never applies settings directly; ADK tool confirmation is required before it applies."
            ),
            category="admin",
            callable=propose_chat_agent_config_change,
            tier="core",
            label_ko="채팅 에이전트 설정 초안",
            sort_order=21,
        ),
        ToolEntry(
            tool_id="propose_tenant_config_change",
            name="propose_tenant_config_change",
            description=(
                "Creates a tenant-level settings change draft for allowed runtime config keys. The tool never "
                "applies settings directly; ADK tool confirmation is required before it applies."
            ),
            category="admin",
            callable=propose_tenant_config_change,
            tier="core",
            label_ko="테넌트 설정 초안",
            sort_order=22,
        ),
        ToolEntry(
            tool_id="get_config_change_status",
            name="get_config_change_status",
            description="Reads pending or applied investment-chat settings change approval drafts.",
            category="admin",
            callable=get_config_change_status,
            tier="core",
            label_ko="설정 승인 상태",
            sort_order=23,
        ),
    ]
    if include_internal_bridge:
        entries.append(
            ToolEntry(
                tool_id="apply_approved_config_change",
                name="apply_approved_config_change",
                description="Internal backend/UI approval bridge for applying a stored config draft. Not exposed to the LLM tool registry.",
                category="admin",
                callable=apply_approved_config_change,
                tier="internal",
                label_ko="승인 설정 적용",
                sort_order=24,
            )
        )
    return entries
