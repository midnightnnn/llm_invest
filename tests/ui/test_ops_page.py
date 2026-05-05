from __future__ import annotations

import os

from arena.config import load_settings
from arena.ui.server import _build_app
from tests.direct_route_client import DirectRouteClient


class _OpsRepo:
    dataset_fqn = "proj.ds"
    project = "proj"
    location = "asia-northeast3"

    def __init__(self) -> None:
        self.user_tenants: dict[str, list[dict[str, str]]] = {}
        self.run_statuses: list[dict] = []
        self.cred_rows: list[dict] = []
        self.audit_rows: list[dict] = []
        self.tenant_ids: list[str] = []

    # ── stubs used by _build_app ──
    def list_runtime_user_tenants(self, *, user_email: str):
        return [dict(r) for r in self.user_tenants.get(str(user_email or "").strip().lower(), [])]

    def ensure_runtime_user_tenant(self, **kw):
        email = str(kw.get("user_email") or "").strip().lower()
        tid = str(kw.get("tenant_id") or "").strip().lower()
        role = str(kw.get("role") or "owner")
        rows = self.user_tenants.setdefault(email, [])
        if not any(r.get("tenant_id") == tid for r in rows):
            rows.append({"user_email": email, "tenant_id": tid, "role": role})

    def has_runtime_user_tenant(self, *, user_email: str, tenant_id: str):
        user = str(user_email or "").strip().lower()
        tenant = str(tenant_id or "").strip().lower()
        return any(r.get("tenant_id") == tenant for r in self.user_tenants.get(user, []))

    def latest_runtime_credentials(self, *, tenant_id: str):
        return {}

    def recent_runtime_credentials(self, *, limit: int = 20):
        return list(self.cred_rows)

    def latest_runtime_access_request(self, *, user_email: str):
        return None

    def ensure_runtime_access_request_pending(self, **kw):
        return {"status": "pending"}

    def append_runtime_audit_log(self, **kw):
        pass

    def latest_tenant_run_status(self, *, tenant_id: str, **kw):
        return None

    def latest_reconciliation_run(self, *, tenant_id: str | None = None):
        return None

    def get_config(self, tenant_id: str, config_key: str):
        return None

    def get_configs(self, tenant_id: str, config_keys: list[str]):
        return {}

    def set_config(self, tenant_id: str, config_key: str, value: str, updated_by=None, **kw):
        pass

    def fetch_rows(self, sql: str, params=None):
        return []

    def list_runtime_tenants(self, *, limit: int = 200):
        return list(self.tenant_ids)

    def all_tenant_run_statuses(self, *, limit: int = 100):
        return list(self.run_statuses)

    def recent_runtime_audit_logs(self, *, limit: int = 50):
        return list(self.audit_rows)


def _make_client(*, operator_emails: str = "") -> tuple[DirectRouteClient, _OpsRepo]:
    env = {
        "ARENA_UI_AUTH_ENABLED": "false",
        "ARENA_UI_SETTINGS_ENABLED": "false",
        "ARENA_OPERATOR_EMAILS": operator_emails,
    }
    for k, v in env.items():
        os.environ[k] = v
    try:
        repo = _OpsRepo()
        repo.tenant_ids = ["alice", "bob"]
        repo.run_statuses = [
            {"tenant_id": "alice", "status": "success", "run_type": "pipeline", "stage": "done",
             "started_at": "2026-03-21T10:00:00", "finished_at": "2026-03-21T10:05:00", "message": "ok"},
            {"tenant_id": "bob", "status": "failed", "run_type": "pipeline", "stage": "sync",
             "started_at": "2026-03-21T09:00:00", "finished_at": "2026-03-21T09:01:00", "message": "timeout"},
        ]
        repo.cred_rows = [
            {"tenant_id": "alice", "kis_env": "demo", "has_openai": True, "has_gemini": True, "has_anthropic": False,
             "updated_at": "2026-03-21T00:00:00"},
        ]
        repo.audit_rows = [
            {"created_at": "2026-03-21T10:00:00", "user_email": "op@test.com", "tenant_id": "alice",
             "action": "admin_sleeve_save", "status": "ok"},
        ]
        settings = load_settings()
        app = _build_app(repo=repo, settings=settings)
        client = DirectRouteClient(app)
        return client, repo
    finally:
        for k in env:
            os.environ.pop(k, None)


def test_ops_accessible_in_local_dev():
    """When auth is disabled, local@localhost is auto-added as operator."""
    client, _ = _make_client(operator_emails="")
    resp = client.get("/ops")
    assert resp.status_code == 200


def test_ops_accessible_with_explicit_operator():
    """Explicit operator email also works alongside local@localhost."""
    client, _ = _make_client(operator_emails="other@example.com")
    resp = client.get("/ops")
    # local@localhost is auto-added when auth_enabled=false
    assert resp.status_code == 200


def test_ops_renders_for_operator():
    client, _ = _make_client(operator_emails="local@localhost")
    resp = client.get("/ops")
    assert resp.status_code == 200
    body = resp.text
    assert "Operations" in body
    assert "Tenant Health" in body
    assert "alice" in body
    assert "bob" in body


def test_ops_api_health_forbidden_with_auth():
    """When auth is enabled but no session, API returns 403."""
    env = {
        "ARENA_UI_AUTH_ENABLED": "true",
        "ARENA_UI_SETTINGS_ENABLED": "false",
        "ARENA_OPERATOR_EMAILS": "admin@example.com",
        "ARENA_UI_SESSION_SECRET": "test-secret",
    }
    for k, v in env.items():
        os.environ[k] = v
    try:
        repo = _OpsRepo()
        repo.tenant_ids = []
        settings = load_settings()
        app = _build_app(repo=repo, settings=settings)
        client = DirectRouteClient(app)
        resp = client.get("/api/ops/health")
        # No session → _current_user returns None → not operator → 403
        assert resp.status_code == 403
    finally:
        for k in env:
            os.environ.pop(k, None)


def test_ops_api_health_ok():
    client, _ = _make_client(operator_emails="local@localhost")
    resp = client.get("/api/ops/health")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["tenant_id"] == "alice"


def test_ops_api_audit_ok():
    client, _ = _make_client(operator_emails="local@localhost")
    resp = client.get("/api/ops/audit")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["action"] == "admin_sleeve_save"


def test_ops_api_credentials_ok():
    client, _ = _make_client(operator_emails="local@localhost")
    resp = client.get("/api/ops/credentials")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["tenant_id"] == "alice"


def test_ops_nav_visible_to_operator():
    client, _ = _make_client(operator_emails="local@localhost")
    resp = client.get("/ops")
    assert resp.status_code == 200
    assert 'href="/ops"' in resp.text


def test_ops_nav_visible_in_local_dev():
    """In local dev (auth off), ops nav is always visible."""
    client, _ = _make_client(operator_emails="")
    resp = client.get("/ops")
    assert resp.status_code == 200
    assert 'href="/ops"' in resp.text
