from __future__ import annotations

from datetime import datetime, timedelta, timezone

from arena.config import load_settings
from arena.agents.investment_chat.context import REQUEST_MODEL
from arena.agents.investment_chat.drafts import approval_token, save_credential_draft
from tests.ui.investment_chat_helpers import _ChatOrderRepo


def test_chat_credential_tool_creates_draft_without_secret_value() -> None:
    from arena.agents.investment_chat.credential_tools import (
        build_credential_tool_entries,
        load_credential_draft,
        recent_credential_drafts,
    )

    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    entries = build_credential_tool_entries(repo=repo, settings=load_settings(), tenant_id="local")
    tool = next(item.callable for item in entries if item.name == "propose_model_key_change")

    model_context = REQUEST_MODEL.set("gpt-5.2")
    try:
        result = tool(provider="openai", action="upsert", rationale="사용자가 OpenAI 키 변경을 요청함")
    finally:
        REQUEST_MODEL.reset(model_context)

    assert result["status"] == "ok"
    assert result["approval_required"] is True
    assert result["provider"] == "gpt"
    assert "model" not in result
    assert "gpt-5.2" not in result["summary"]
    assert "api_key" not in result
    draft = load_credential_draft(repo, tenant_id="local", token=result["approval_token"])
    assert draft is not None
    assert draft["action"] == "upsert"
    assert draft["provider"] == "gpt"
    assert "model" not in draft
    assert "gpt-5.2" not in draft["summary"]
    assert "api_key" not in draft
    recent = recent_credential_drafts(repo, tenant_id="local")
    assert recent[0]["approval_token"] == result["approval_token"]
    assert recent[0]["provider"] == "gpt"
    assert recent[0]["model"] == ""


def test_chat_credential_tool_creates_delete_draft_for_provider() -> None:
    from arena.agents.investment_chat.credential_tools import (
        build_credential_tool_entries,
        load_credential_draft,
    )

    repo = _ChatOrderRepo()
    entries = build_credential_tool_entries(repo=repo, settings=load_settings(), tenant_id="local")
    tool = next(item.callable for item in entries if item.name == "propose_model_key_change")

    result = tool(provider="claude", action="delete", rationale="사용자가 Claude 키 삭제를 요청함")

    assert result["status"] == "ok"
    assert result["approval_required"] is True
    assert result["provider"] == "claude"
    assert result["action"] == "delete"
    draft = load_credential_draft(repo, tenant_id="local", token=result["approval_token"])
    assert draft is not None
    assert draft["action"] == "delete"
    assert draft["provider"] == "claude"
    assert "api_key" not in draft


def test_chat_credential_tool_creates_kis_account_draft_without_secret_values() -> None:
    from arena.agents.investment_chat.credential_tools import (
        build_credential_tool_entries,
        load_credential_draft,
        recent_credential_drafts,
    )

    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    entries = build_credential_tool_entries(repo=repo, settings=load_settings(), tenant_id="local")
    tool = next(item.callable for item in entries if item.name == "propose_kis_account_change")

    result = tool(action="upsert", env="demo", rationale="사용자가 KIS API key 등록을 요청함")

    assert result["status"] == "ok"
    assert result["approval_required"] is True
    assert result["credential_kind"] == "kis_account"
    assert result["env"] == "demo"
    assert "app_key" not in result
    assert "app_secret" not in result
    assert "account_no" not in result
    draft = load_credential_draft(repo, tenant_id="local", token=result["approval_token"])
    assert draft is not None
    assert draft["credential_kind"] == "kis_account"
    assert draft["action"] == "upsert"
    assert draft["env"] == "demo"
    assert "app_key" not in draft
    assert "app_secret" not in draft
    assert "account_no" not in draft
    recent = recent_credential_drafts(repo, tenant_id="local")
    assert recent[0]["approval_token"] == result["approval_token"]
    assert recent[0]["credential_kind"] == "kis_account"


def test_recent_credential_drafts_excludes_expired_drafts() -> None:
    from arena.agents.investment_chat.credential_tools import (
        CREDENTIAL_CHANGE_PROPOSE_ACTION,
        recent_credential_drafts,
    )

    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    expired = datetime.now(timezone.utc) - timedelta(minutes=1)
    token = approval_token({"tenant_id": "local", "provider": "gpt", "expired": True})
    save_credential_draft(
        repo,
        tenant_id="local",
        token=token,
        draft={
            "status": "draft",
            "tenant_id": "local",
            "action": "upsert",
            "provider": "gpt",
            "provider_label": "OpenAI",
            "model": "gpt-5.5",
            "created_at": (expired - timedelta(minutes=15)).isoformat(),
            "expires_at": expired.isoformat(),
            "summary": "expired",
        },
    )
    repo.append_runtime_audit_log(
        tenant_id="local",
        action=CREDENTIAL_CHANGE_PROPOSE_ACTION,
        status="draft",
        detail={"approval_token": token},
    )

    assert recent_credential_drafts(repo, tenant_id="local") == []
