from __future__ import annotations

from fastapi import FastAPI

from arena.config import load_settings
from arena.ui.server import _build_app
from tests.direct_route_client import DirectRouteClient
from tests.ui.investment_chat_helpers import _ChatOrderRepo


def _build_chat_client(monkeypatch, repo: _ChatOrderRepo, tmp_path) -> DirectRouteClient:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setenv("ARENA_LOCAL_CREDENTIALS_FILE", str(tmp_path / "credentials.json"))
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(
        "arena.ui.routes.investment_chat.discover_model_options_with_api_key",
        lambda provider, api_key: {
            "provider": provider,
            "advisor_models": [f"{provider}-3-flash-preview" if provider == "gemini" else "gpt-5.5"],
            "router_models": [f"{provider}-3-flash-preview" if provider == "gemini" else "gpt-5.4-mini"],
            "utility_models": [f"{provider}-3-flash-preview" if provider == "gemini" else "gpt-5.4-mini"],
        },
    )
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    return DirectRouteClient(_build_app(repo=repo, settings=load_settings()))


def test_credential_draft_api_lists_and_applies_model_key(monkeypatch, tmp_path) -> None:
    from arena.agents.investment_chat.credential_tools import build_credential_tool_entries

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {
        "tenant_id": "local",
        "model_secret_name": "local-local-models",
        "has_openai": True,
        "has_gemini": False,
        "has_anthropic": False,
    }
    client = _build_chat_client(monkeypatch, repo, tmp_path)
    tool = next(
        item.callable
        for item in build_credential_tool_entries(repo=repo, settings=load_settings(), tenant_id="local")
        if item.name == "propose_model_key_change"
    )
    draft = tool(provider="gemini", action="upsert")
    token = draft["approval_token"]

    listing = client.get("/investment-chat/credential-drafts", params={"tenant_id": "local"})

    assert listing.status_code == 200
    payload = listing.json()
    assert payload["drafts"][0]["approval_token"] == token
    assert payload["drafts"][0]["action"] == "upsert"
    assert payload["drafts"][0]["provider"] == "gemini"

    response = client.post(
        f"/investment-chat/credential-drafts/{token}/apply",
        data={
            "tenant_id": "local",
            "model": "gemini-3-flash-preview",
            "cheap_model": "gemini-3-flash-preview",
            "api_key": "gemini-test-key",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "applied"
    assert body["provider"] == "gemini"
    assert body["model"]
    assert body["reload_url"] == f"/investment-chat?tenant_id=local&provider=gemini&model={body['model']}"
    assert repo.runtime_credentials["local"]["has_gemini"] is True


def test_credential_draft_api_deletes_provider_key(monkeypatch, tmp_path) -> None:
    from arena.agents.investment_chat.credential_tools import build_credential_tool_entries

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {
        "tenant_id": "local",
        "model_secret_name": "local-local-models",
        "has_openai": True,
        "has_gemini": True,
        "has_anthropic": False,
    }
    client = _build_chat_client(monkeypatch, repo, tmp_path)
    setup = client.post(
        "/investment-chat/model-key",
        data={
            "tenant_id": "local",
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "cheap_model": "gemini-3-flash-preview",
            "api_key": "gemini-test-key",
        },
    )
    assert setup.status_code == 303
    tool = next(
        item.callable
        for item in build_credential_tool_entries(repo=repo, settings=load_settings(), tenant_id="local")
        if item.name == "propose_model_key_change"
    )
    draft = tool(provider="gemini", action="delete")
    token = draft["approval_token"]

    response = client.post(
        f"/investment-chat/credential-drafts/{token}/apply",
        data={"tenant_id": "local", "confirmed": "true"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deleted"
    assert body["provider"] == "gemini"
    assert body["reload_url"] == "/investment-chat?tenant_id=local"
    assert repo.runtime_credentials["local"]["has_gemini"] is False


def test_credential_draft_api_lists_and_applies_kis_account(monkeypatch, tmp_path) -> None:
    from arena.agents.investment_chat.credential_tools import build_credential_tool_entries
    from arena.open_trading import sync as open_trading_sync

    repo = _ChatOrderRepo()
    client = _build_chat_client(monkeypatch, repo, tmp_path)
    tool = next(
        item.callable
        for item in build_credential_tool_entries(repo=repo, settings=load_settings(), tenant_id="local")
        if item.name == "propose_kis_account_change"
    )
    draft = tool(action="upsert", env="demo")
    token = draft["approval_token"]
    sync_calls: list[dict[str, object]] = []

    class _FakeAccountSyncService:
        def __init__(self, *, settings, repo):
            sync_calls.append({"market": settings.kis_target_market, "repo": repo})

        def sync_account_snapshot(self):
            return repo.account_snapshot

    monkeypatch.setattr(open_trading_sync, "AccountSyncService", _FakeAccountSyncService)

    listing = client.get("/investment-chat/credential-drafts", params={"tenant_id": "local"})

    assert listing.status_code == 200
    payload = listing.json()
    assert payload["drafts"][0]["approval_token"] == token
    assert payload["drafts"][0]["credential_kind"] == "kis_account"
    assert payload["drafts"][0]["action"] == "upsert"
    assert payload["drafts"][0]["env"] == "demo"

    response = client.post(
        f"/investment-chat/credential-drafts/{token}/apply",
        data={
            "tenant_id": "local",
            "kis_env": "demo",
            "kis_account_no": "64317603-01",
            "kis_paper_app_key": "paper-app-key",
            "kis_paper_app_secret": "paper-app-secret",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "applied"
    assert body["credential_kind"] == "kis_account"
    assert body["env"] == "demo"
    assert body["reload_url"] == "/investment-chat?tenant_id=local"
    assert repo.runtime_credentials["local"]["kis_env"] == "demo"
    assert repo.runtime_credentials["local"]["kis_account_no_masked"]
    assert sync_calls[-1]["market"] == "us,kospi,kosdaq"


def test_credential_draft_api_deletes_kis_account(monkeypatch, tmp_path) -> None:
    from arena.agents.investment_chat.credential_tools import build_credential_tool_entries

    repo = _ChatOrderRepo()
    client = _build_chat_client(monkeypatch, repo, tmp_path)
    setup_tool = next(
        item.callable
        for item in build_credential_tool_entries(repo=repo, settings=load_settings(), tenant_id="local")
        if item.name == "propose_kis_account_change"
    )
    setup = setup_tool(action="upsert", env="demo")
    setup_response = client.post(
        f"/investment-chat/credential-drafts/{setup['approval_token']}/apply",
        data={
            "tenant_id": "local",
            "kis_env": "demo",
            "kis_account_no": "64317603-01",
            "kis_paper_app_key": "paper-app-key",
            "kis_paper_app_secret": "paper-app-secret",
        },
    )
    assert setup_response.status_code == 200

    draft = setup_tool(action="delete", env="demo")
    token = draft["approval_token"]

    response = client.post(
        f"/investment-chat/credential-drafts/{token}/apply",
        data={
            "tenant_id": "local",
            "kis_env": "demo",
            "kis_account_no": "64317603-01",
            "confirmed": "true",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deleted"
    assert body["credential_kind"] == "kis_account"
    assert body["env"] == "demo"
    assert body["reload_url"] == "/investment-chat?tenant_id=local"
    assert repo.runtime_credentials["local"]["kis_account_no_masked"] == ""
