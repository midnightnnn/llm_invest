from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from arena.agents.investment_chat.audit import append_chat_audit
from arena.agents.investment_chat.context import REQUEST_PROVIDER, normalize_tenant
from arena.agents.investment_chat.drafts import (
    approval_token,
    load_credential_draft,
    save_credential_draft,
)
from arena.agents.investment_chat.scope import chat_actor_email
from arena.agents.investment_chat.utils import utc_iso
from arena.config import Settings
from arena.providers.registry import canonical_provider, list_adk_provider_specs
from arena.tools.registry import ToolEntry

CREDENTIAL_CHANGE_PROPOSE_ACTION = "chat_credential_change_propose"
CREDENTIAL_CHANGE_APPLY_ACTION = "chat_credential_change_apply"
CredentialAction = Literal["upsert", "delete"]
CredentialKind = Literal["model_key", "kis_account"]


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


def _parse_utc_datetime(value: Any) -> datetime | None:
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


def _allowed_provider_ids() -> set[str]:
    return {spec.provider_id for spec in list_adk_provider_specs() if spec.api_key_setting}


def _provider_label(provider: str) -> str:
    token = canonical_provider(provider) or str(provider or "").strip().lower()
    for spec in list_adk_provider_specs():
        if spec.provider_id == token:
            return spec.label
    return token


def _normalize_action(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"", "upsert", "add", "change", "update", "replace", "save", "set"}:
        return "upsert"
    if token in {"delete", "remove", "clear", "revoke"}:
        return "delete"
    return ""


def _normalize_kis_env(value: str | None) -> str:
    token = str(value or "").strip().lower()
    if token in {"", "demo", "paper", "mock", "vps", "virtual", "sandbox", "모의", "모의투자"}:
        return "demo"
    if token in {"real", "live", "prod", "production", "실전", "실전투자"}:
        return "real"
    return ""


def _normalize_provider(value: str | None) -> str:
    requested = canonical_provider(value) or str(value or "").strip().lower()
    if not requested:
        requested = canonical_provider(REQUEST_PROVIDER.get()) or str(REQUEST_PROVIDER.get() or "").strip().lower()
    return requested if requested in _allowed_provider_ids() else ""


def credential_draft_status_row(token: str, draft: dict[str, Any]) -> dict[str, Any]:
    status = str(draft.get("status") or "").strip().lower()
    provider = str(draft.get("provider") or "").strip().lower()
    kind = str(draft.get("credential_kind") or "").strip().lower()
    if kind not in {"model_key", "kis_account"}:
        kind = "model_key" if provider else "kis_account"
    return {
        "approval_token": token,
        "status": status,
        "submittable": status == "draft",
        "credential_kind": kind,
        "created_at": draft.get("created_at") or "",
        "applied_at": draft.get("applied_at") or "",
        "expires_at": draft.get("expires_at") or "",
        "action": draft.get("action") or "",
        "provider": provider,
        "provider_label": draft.get("provider_label") or _provider_label(provider),
        "model": draft.get("model") or "",
        "env": draft.get("env") or "",
        "summary": draft.get("summary") or "",
        "rationale": draft.get("rationale") or "",
        "message": draft.get("message") or draft.get("error") or "",
    }


def recent_credential_drafts(repo: Any, *, tenant_id: str, limit: int = 5) -> list[dict[str, Any]]:
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
        if str(row.get("action") or "").strip() != CREDENTIAL_CHANGE_PROPOSE_ACTION:
            continue
        detail = _audit_detail(row)
        token = str(detail.get("approval_token") or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        draft = load_credential_draft(repo, tenant_id=tenant, token=token)
        if isinstance(draft, dict):
            expires_at = _parse_utc_datetime(draft.get("expires_at"))
            if expires_at is not None and expires_at < datetime.now(timezone.utc):
                continue
            out.append(credential_draft_status_row(token, draft))
        if len(out) >= max(1, min(int(limit or 5), 20)):
            break
    return out


def build_credential_tool_entries(
    *,
    repo: Any,
    settings: Settings,
    tenant_id: str,
) -> list[ToolEntry]:
    tenant = normalize_tenant(tenant_id)

    def propose_model_key_change(
        provider: str = "",
        model: str = "",
        cheap_model: str = "",
        action: CredentialAction = "upsert",
        rationale: str = "",
    ) -> dict[str, Any]:
        """Create an LLM API key add/change/delete request for the UI credential panel."""
        action_token = _normalize_action(action)
        if not action_token:
            return {"status": "error", "tenant_id": tenant, "error": "action must be upsert or delete"}
        provider_token = _normalize_provider(provider)
        if not provider_token:
            return {"status": "error", "tenant_id": tenant, "error": "provider is required"}

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=15)
        provider_label = _provider_label(provider_token)
        if action_token == "delete":
            summary = f"{provider_label} LLM API key 삭제"
        else:
            summary = f"{provider_label} LLM API key 추가/변경"
        token = approval_token(
            {
                "tenant_id": tenant,
                "action": action_token,
                "provider": provider_token,
                "nonce": uuid4().hex,
            }
        )
        draft = {
            "status": "draft",
            "tenant_id": tenant,
            "credential_kind": "model_key",
            "action": action_token,
            "provider": provider_token,
            "provider_label": provider_label,
            "created_at": utc_iso(now),
            "expires_at": utc_iso(expires_at),
            "summary": summary,
            "rationale": str(rationale or "").strip(),
            "message": "UI에 표시되는 LLM API key 승인 패널에서 처리해야 합니다. API key 값을 채팅에 입력하지 마세요.",
        }
        model_token = str(model or "").strip()
        cheap_model_token = str(cheap_model or "").strip()
        if model_token:
            draft["model"] = model_token
        if cheap_model_token:
            draft["cheap_model"] = cheap_model_token
        save_credential_draft(repo, tenant_id=tenant, token=token, draft=draft)
        append_chat_audit(
            repo,
            tenant_id=tenant,
            action=CREDENTIAL_CHANGE_PROPOSE_ACTION,
            status="draft",
            detail={"approval_token": token, "action": action_token, "provider": provider_token},
            user_email=chat_actor_email(),
        )
        return {
            "status": "ok",
            "tenant_id": tenant,
            "approval_required": True,
            "approval_ui": "credential_input_panel",
            "approval_token": token,
            "credential_kind": "model_key",
            "action": action_token,
            "provider": provider_token,
            "provider_label": provider_label,
            "expires_at": draft["expires_at"],
            "summary": summary,
            "message": "LLM API key 입력/삭제 패널을 열었습니다. 실제 API key 값은 채팅에 쓰지 말고 화면의 별도 입력칸에 입력해야 합니다.",
        }

    def propose_kis_account_change(
        action: CredentialAction = "upsert",
        env: str = "demo",
        rationale: str = "",
    ) -> dict[str, Any]:
        """Create a KIS account API key add/change/delete request for the UI credential panel."""
        action_token = _normalize_action(action)
        if not action_token:
            return {"status": "error", "tenant_id": tenant, "error": "action must be upsert or delete"}
        env_token = _normalize_kis_env(env)
        if not env_token:
            return {"status": "error", "tenant_id": tenant, "error": "env must be demo or real"}

        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=15)
        env_label = "모의투자" if env_token == "demo" else "실전투자"
        summary = f"KIS {env_label} API key {'삭제' if action_token == 'delete' else '추가/변경'}"
        token = approval_token(
            {
                "tenant_id": tenant,
                "credential_kind": "kis_account",
                "action": action_token,
                "env": env_token,
                "nonce": uuid4().hex,
            }
        )
        draft = {
            "status": "draft",
            "tenant_id": tenant,
            "credential_kind": "kis_account",
            "action": action_token,
            "env": env_token,
            "created_at": utc_iso(now),
            "expires_at": utc_iso(expires_at),
            "summary": summary,
            "rationale": str(rationale or "").strip(),
            "message": "UI에 표시되는 KIS credential 승인 패널에서 처리해야 합니다. 계좌번호와 API key 값을 채팅에 입력하지 마세요.",
        }
        save_credential_draft(repo, tenant_id=tenant, token=token, draft=draft)
        append_chat_audit(
            repo,
            tenant_id=tenant,
            action=CREDENTIAL_CHANGE_PROPOSE_ACTION,
            status="draft",
            detail={"approval_token": token, "credential_kind": "kis_account", "action": action_token, "env": env_token},
            user_email=chat_actor_email(),
        )
        return {
            "status": "ok",
            "tenant_id": tenant,
            "approval_required": True,
            "approval_ui": "credential_input_panel",
            "approval_token": token,
            "credential_kind": "kis_account",
            "action": action_token,
            "env": env_token,
            "expires_at": draft["expires_at"],
            "summary": summary,
            "message": "KIS API key 입력/삭제 패널을 열었습니다. 실제 계좌번호와 API key 값은 채팅에 쓰지 말고 화면의 별도 입력칸에 입력해야 합니다.",
        }

    def get_model_key_change_status(approval_token: str = "", limit: int = 5) -> dict[str, Any]:
        """Read pending or applied credential change requests."""
        token = str(approval_token or "").strip()
        if token:
            draft = load_credential_draft(repo, tenant_id=tenant, token=token)
            if not isinstance(draft, dict):
                return {"status": "missing", "tenant_id": tenant, "approval_token": token}
            return {"status": "ok", "tenant_id": tenant, "draft": credential_draft_status_row(token, draft)}
        return {"status": "ok", "tenant_id": tenant, "drafts": recent_credential_drafts(repo, tenant_id=tenant, limit=limit)}

    return [
        ToolEntry(
            tool_id="propose_model_key_change",
            name="propose_model_key_change",
            description=(
                "Creates an LLM API key add/change/delete request for the investment chat UI, optionally preselecting "
                "advisor and router/utility models. "
                "Never ask the user to paste an API key into chat; this tool opens a separate credential input panel."
            ),
            category="admin",
            callable=propose_model_key_change,
            tier="core",
            label_ko="LLM API key 변경 요청",
            sort_order=25,
        ),
        ToolEntry(
            tool_id="propose_kis_account_change",
            name="propose_kis_account_change",
            description=(
                "Creates a KIS account API key add/change/delete request for the investment chat UI. "
                "Never ask the user to paste account numbers, app keys, or app secrets into chat; "
                "this tool opens a separate credential input panel."
            ),
            category="admin",
            callable=propose_kis_account_change,
            tier="core",
            label_ko="KIS API key 변경 요청",
            sort_order=26,
        ),
        ToolEntry(
            tool_id="get_model_key_change_status",
            name="get_model_key_change_status",
            description="Reads pending or applied LLM/KIS credential change requests.",
            category="admin",
            callable=get_model_key_change_status,
            tier="core",
            label_ko="LLM API key 변경 상태",
            sort_order=27,
        ),
    ]
