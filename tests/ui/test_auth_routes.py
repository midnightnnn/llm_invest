from __future__ import annotations

import json
from datetime import datetime, timezone

from arena.config import load_settings
from arena.models import AccountSnapshot
from arena.ui.server import _build_app
from arena.ui.layout import tailwind_layout as _tailwind_layout
from tests.direct_route_client import DirectRouteClient
from tests.ui.helpers import (
    _DummyRepo,
    _client,
    _client_with_repo,
    _client_with_repo_and_credential_store,
)

def test_layout_shows_auth_controls_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "true")
    html = _tailwind_layout("X", "<div>body</div>", active="board")
    assert '/auth/logout' in html
    assert 'sidebar-link' in html


def test_auth_google_callback_auto_provisions_new_user(monkeypatch) -> None:
    class _TokenResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"id_token": "fake-id-token"}

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr("arena.ui.app.requests.post", lambda *args, **kwargs: _TokenResponse())
    monkeypatch.setattr(
        "arena.ui.app.google_id_token.verify_oauth2_token",
        lambda raw, req, client_id: {
            "email": "pending@example.com",
            "name": "Pending User",
            "sub": "sub-123",
        },
    )

    repo = _DummyRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)
    client.session["oauth_state"] = "state-123"
    client.session["next_path"] = "/board?tenant_id=main"

    response = client.get("/auth/google/callback", params={"code": "oauth-code", "state": "state-123"})

    assert response.status_code == 302
    assert response.headers.get("location") == "/board?tenant_id=main"
    assert client.session["user"]["email"] == "pending@example.com"
    assert repo.has_runtime_user_tenant(user_email="pending@example.com", tenant_id="pending") is True
    assert repo.get_config("pending", "distribution_mode") == "simulated_only"
    assert repo.get_config("pending", "real_trading_approved") == "false"


def test_auth_google_callback_redirects_rejected_user_to_pending(monkeypatch) -> None:
    class _TokenResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"id_token": "fake-id-token"}

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "true")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr("arena.ui.app.requests.post", lambda *args, **kwargs: _TokenResponse())
    monkeypatch.setattr(
        "arena.ui.app.google_id_token.verify_oauth2_token",
        lambda raw, req, client_id: {
            "email": "blocked@example.com",
            "name": "Blocked User",
            "sub": "sub-999",
        },
    )

    repo = _DummyRepo()
    repo.access_requests.append(
        {
            "user_email": "blocked@example.com",
            "user_name": "Blocked User",
            "google_sub": "sub-999",
            "requested_at": "2026-03-21T00:00:00+00:00",
            "status": "rejected",
            "note": "manual block",
        }
    )
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)
    client.session["oauth_state"] = "state-123"

    response = client.get("/auth/google/callback", params={"code": "oauth-code", "state": "state-123"})

    assert response.status_code == 302
    assert response.headers.get("location") == "/auth/pending"
    assert repo.has_runtime_user_tenant(user_email="blocked@example.com", tenant_id="blocked") is False


def test_auth_google_callback_auto_grants_public_demo_viewer_access(monkeypatch) -> None:
    class _TokenResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"id_token": "fake-id-token"}

    class _DemoRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            tenant = str((params or {}).get("tenant_id") or "").strip().lower()
            if tenant == "midnightnnn":
                return [{"tenant_id": "midnightnnn"}]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "true")
    monkeypatch.setenv("ARENA_PUBLIC_DEMO_TENANT", "midnightnnn")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setattr("arena.ui.app.requests.post", lambda *args, **kwargs: _TokenResponse())
    monkeypatch.setattr(
        "arena.ui.app.google_id_token.verify_oauth2_token",
        lambda raw, req, client_id: {
            "email": "viewer@example.com",
            "name": "Viewer User",
            "sub": "viewer-sub",
        },
    )

    repo = _DemoRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)
    client.session["oauth_state"] = "state-123"
    client.session["next_path"] = "/board?tenant_id=midnightnnn"

    response = client.get("/auth/google/callback", params={"code": "oauth-code", "state": "state-123"})

    assert response.status_code == 302
    assert response.headers.get("location") == "/board?tenant_id=midnightnnn"
    assert repo.has_runtime_user_tenant(user_email="viewer@example.com", tenant_id="viewer") is True
    assert repo.has_runtime_user_tenant(user_email="viewer@example.com", tenant_id="midnightnnn") is True


def test_auth_pending_page_redirects_after_manual_approval(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "true")
    repo = _DummyRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)
    client.session["user"] = {
        "email": "viewer@example.com",
        "name": "Viewer User",
        "sub": "viewer-sub",
    }
    client.session["next_path"] = "/nav?tenant_id=main"

    first = client.get("/auth/pending")
    assert first.status_code == 200
    assert "승인 대기 중입니다" in first.text

    repo.ensure_runtime_user_tenant(
        user_email="viewer@example.com",
        tenant_id="main",
        role="viewer",
        created_by="admin@example.com",
    )

    second = client.get("/auth/pending")
    assert second.status_code == 302
    assert second.headers.get("location") == "/nav?tenant_id=main"


def test_settings_page_redirects_viewer_only_user_to_forbidden(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "true")
    repo = _DummyRepo()
    repo.ensure_runtime_user_tenant(
        user_email="viewer@example.com",
        tenant_id="main",
        role="viewer",
        created_by="admin@example.com",
    )
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)
    client.session["user"] = {
        "email": "viewer@example.com",
        "name": "Viewer User",
        "sub": "viewer-sub",
    }

    response = client.get("/settings", params={"tenant_id": "main"})

    assert response.status_code == 302
    assert response.headers.get("location") == "/auth/forbidden"
