from __future__ import annotations

import json
from datetime import datetime, timezone

from arena.config import load_settings
from arena.models import AccountSnapshot
from arena.ui.server import _build_app
from arena.ui.layout import tailwind_layout as _tailwind_layout
from tests.direct_route_client import DirectRouteClient


class _DummyRepo:
    dataset_fqn = "proj.ds"
    project = "proj"
    location = "asia-northeast3"

    def __init__(self) -> None:
        self.cfg: dict[tuple[str, str], str] = {}
        self.runtime_credentials: dict[str, dict[str, str]] = {}
        self.fetch_calls: list[tuple[str, dict | None]] = []
        self.sleeve_sync_calls: list[dict[str, object]] = []
        self.capital_sync_calls: list[dict[str, object]] = []
        self.nav_upsert_calls: list[dict[str, object]] = []
        self.snapshot_calls: list[dict[str, object]] = []
        self.latest_run_status_row: dict[str, object] | None = None
        self.latest_recon_row: dict[str, object] | None = None
        self.recon_issue_rows: list[dict[str, object]] = []
        self.user_tenants: dict[str, list[dict[str, str]]] = {}
        self.access_requests: list[dict[str, str]] = []

    def list_runtime_user_tenants(self, *, user_email: str) -> list[dict[str, str]]:
        return [dict(row) for row in self.user_tenants.get(str(user_email or "").strip().lower(), [])]

    def ensure_runtime_user_tenant(self, **kwargs) -> None:
        user_email = str(kwargs.get("user_email") or "").strip().lower()
        tenant_id = str(kwargs.get("tenant_id") or "").strip().lower()
        role = str(kwargs.get("role") or "owner").strip().lower() or "owner"
        if not user_email or not tenant_id:
            return None
        rows = self.user_tenants.setdefault(user_email, [])
        if not any(str(row.get("tenant_id") or "").strip().lower() == tenant_id for row in rows):
            rows.append({"user_email": user_email, "tenant_id": tenant_id, "role": role})
        return None

    def latest_runtime_credentials(self, *, tenant_id: str) -> dict[str, str]:
        return dict(self.runtime_credentials.get(str(tenant_id or "").strip().lower(), {}))

    def recent_runtime_credentials(self, *, limit: int = 20) -> list[dict[str, str]]:
        _ = limit
        return []

    def has_runtime_user_tenant(self, *, user_email: str, tenant_id: str) -> bool:
        user = str(user_email or "").strip().lower()
        tenant = str(tenant_id or "").strip().lower()
        return any(str(row.get("tenant_id") or "").strip().lower() == tenant for row in self.user_tenants.get(user, []))

    def latest_runtime_access_request(self, *, user_email: str) -> dict[str, str] | None:
        user = str(user_email or "").strip().lower()
        matches = [row for row in self.access_requests if str(row.get("user_email") or "").strip().lower() == user]
        return dict(matches[-1]) if matches else None

    def ensure_runtime_access_request_pending(self, *, user_email: str, user_name: str | None = None, google_sub: str | None = None):
        latest = self.latest_runtime_access_request(user_email=user_email)
        if latest and str(latest.get("status") or "").strip().lower() == "pending":
            return latest
        row = {
            "user_email": str(user_email or "").strip().lower(),
            "user_name": str(user_name or "").strip(),
            "google_sub": str(google_sub or "").strip(),
            "requested_at": "2026-03-21T00:00:00+00:00",
            "status": "pending",
            "note": "",
        }
        self.access_requests.append(row)
        return dict(row)

    def append_runtime_audit_log(self, **kwargs) -> None:
        _ = kwargs
        return None

    def set_config(self, tenant_id: str, config_key: str, value: str, updated_by: str | None = None, **kwargs) -> None:
        _ = updated_by, kwargs
        self.cfg[(tenant_id, config_key)] = value

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        return self.cfg.get((tenant_id, config_key))

    def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]:
        return {
            key: self.cfg[(tenant_id, key)]
            for key in config_keys
            if (tenant_id, key) in self.cfg
        }

    def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
        self.fetch_calls.append((sql, params))
        if "FROM `proj.ds.reconciliation_issues`" in sql:
            return [dict(row) for row in self.recon_issue_rows]
        return []

    def latest_tenant_run_status(self, *, tenant_id: str, run_type: str | None = None):
        _ = tenant_id, run_type
        return self.latest_run_status_row

    def latest_reconciliation_run(self, *, tenant_id: str | None = None):
        _ = tenant_id
        return self.latest_recon_row

    def retarget_agent_sleeves_preserve_positions(
        self,
        *,
        agent_ids: list[str],
        target_sleeve_capital_krw: float,
        target_capitals: dict[str, float] | None = None,
        initialized_at=None,
        include_simulated: bool = True,
        sources=None,
        tenant_id: str | None = None,
    ) -> dict[str, dict[str, object]]:
        _ = initialized_at
        self.sleeve_sync_calls.append(
            {
                "agent_ids": list(agent_ids),
                "target_sleeve_capital_krw": float(target_sleeve_capital_krw),
                "target_capitals": dict(target_capitals) if target_capitals else None,
                "include_simulated": include_simulated,
                "sources": sources,
                "tenant_id": tenant_id,
            }
        )
        return {
            str(a): {
                "over_target": False,
            }
            for a in agent_ids
        }

    def retarget_agent_capitals_preserve_positions(
        self,
        *,
        agent_ids: list[str],
        target_sleeve_capital_krw: float,
        target_capitals: dict[str, float] | None = None,
        occurred_at=None,
        include_simulated: bool = True,
        sources=None,
        tenant_id: str | None = None,
        created_by: str = "system",
    ) -> dict[str, dict[str, object]]:
        _ = occurred_at
        self.capital_sync_calls.append(
            {
                "agent_ids": list(agent_ids),
                "target_sleeve_capital_krw": float(target_sleeve_capital_krw),
                "target_capitals": dict(target_capitals) if target_capitals else None,
                "include_simulated": include_simulated,
                "sources": sources,
                "tenant_id": tenant_id,
                "created_by": created_by,
            }
        )
        return {
            str(a): {
                "over_target": False,
                "capital_flow_krw": 0.0,
            }
            for a in agent_ids
        }

    def build_agent_sleeve_snapshot(
        self,
        *,
        agent_id: str,
        sources=None,
        include_simulated: bool = True,
        tenant_id: str | None = None,
    ):
        self.snapshot_calls.append(
            {
                "agent_id": agent_id,
                "sources": list(sources) if isinstance(sources, list) else sources,
                "include_simulated": include_simulated,
                "tenant_id": tenant_id,
            }
        )
        return (
            AccountSnapshot(cash_krw=500000.0, total_equity_krw=500000.0, positions={}),
            500000.0,
            {"agent_id": agent_id},
        )

    def upsert_agent_nav_daily(
        self,
        *,
        nav_date,
        agent_id: str,
        nav_krw: float,
        baseline_equity_krw: float,
        cash_krw: float | None = None,
        market_value_krw: float | None = None,
        capital_flow_krw: float | None = None,
        fx_source: str | None = None,
        valuation_source: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        self.nav_upsert_calls.append(
            {
                "nav_date": nav_date,
                "agent_id": agent_id,
                "nav_krw": float(nav_krw),
                "baseline_equity_krw": float(baseline_equity_krw),
                "cash_krw": cash_krw,
                "market_value_krw": market_value_krw,
                "capital_flow_krw": capital_flow_krw,
                "fx_source": fx_source,
                "valuation_source": valuation_source,
                "tenant_id": tenant_id,
            }
        )


def _client(monkeypatch) -> DirectRouteClient:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _DummyRepo()
    app = _build_app(repo=repo, settings=load_settings())
    return DirectRouteClient(app)


def _client_with_repo(monkeypatch) -> tuple[DirectRouteClient, _DummyRepo]:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _DummyRepo()
    app = _build_app(repo=repo, settings=load_settings())
    return DirectRouteClient(app), repo


def _client_with_repo_and_credential_store(monkeypatch, store_cls) -> tuple[DirectRouteClient, _DummyRepo]:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr("arena.ui.app.CredentialStore", store_cls)
    repo = _DummyRepo()
    app = _build_app(repo=repo, settings=load_settings())
    return DirectRouteClient(app), repo


def test_settings_page_renders(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    client = _client(monkeypatch)
    # Default tab is agents (includes credentials)
    response = client.get("/settings")
    assert response.status_code == 200
    assert "Credentials" in response.text or "Agents" in response.text

    # Capital tab
    response_cap = client.get("/settings?tab=capital")
    assert response_cap.status_code == 200
    assert "/admin/recover" in response_cap.text
    assert "에이전트별 장부 계보" in response_cap.text
    assert "capitalLineageGraph" in response_cap.text
    assert "Target Capital" in response_cap.text
    assert "현재 sleeve 배분" not in response_cap.text

    # MCP tab
    response_mcp = client.get("/settings?tab=mcp")
    assert response_mcp.status_code == 200
    assert "data-mcp-add" in response_mcp.text


def test_settings_page_shows_research_status_when_gemini_missing(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    client = _client(monkeypatch)

    response = client.get("/settings")

    assert response.status_code == 200
    assert "Gemini 키가 없어 새로운 리서치 브리핑 생성은 비활성화됩니다." in response.text


def test_settings_page_shows_shared_live_research_status(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT", "midnightnnn")
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("local", "distribution_mode", "private")
    repo.set_config("local", "real_trading_approved", "true")
    repo.runtime_credentials["midnightnnn"] = {
        "tenant_id": "midnightnnn",
        "model_secret_name": "models-midnightnnn",
        "has_gemini": True,
    }

    monkeypatch.setattr(
        "arena.cli._load_secret_json",
        lambda **kwargs: {"providers": {"gemini": {"api_key": "shared-research-gemini"}}},
    )

    response = client.get("/settings")

    assert response.status_code == 200
    assert "승인된 live tenant라서 midnightnnn의 operator-managed Gemini로 리서치 브리핑을 생성합니다." in response.text


def test_settings_page_uses_tenant_model_secret_for_research_status(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    client, repo = _client_with_repo(monkeypatch)
    repo.runtime_credentials["local"] = {
        "tenant_id": "local",
        "model_secret_name": "models-local",
        "has_gemini": True,
    }

    monkeypatch.setattr(
        "arena.cli._load_secret_json",
        lambda **kwargs: {"providers": {"gemini": {"api_key": "tenant-gemini-key"}}},
    )

    response = client.get("/settings")

    assert response.status_code == 200
    assert "이 테넌트는 Gemini native grounding으로 새로운 리서치 브리핑을 생성할 수 있습니다." in response.text
    assert "Gemini 키가 없어 새로운 리서치 브리핑 생성은 비활성화됩니다." not in response.text


def test_settings_page_renders_saved_mcp_rows(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config(
        "local",
        "mcp_servers",
        json.dumps(
            [
                {
                    "name": "sig",
                    "url": "https://example.com/sse",
                    "transport": "sse",
                    "enabled": True,
                }
            ]
        ),
    )

    response = client.get("/settings?tab=mcp")

    assert response.status_code == 200
    assert 'data-mcp-row' in response.text
    assert "https://example.com/sse" in response.text
    assert "sig" in response.text


def test_settings_page_shows_active_kis_account_and_masked_keys(monkeypatch) -> None:
    class _FakeCredentialStore:
        def __init__(self, *, project, repo):
            _ = project, repo

        def list_kis_accounts_meta(self, *, tenant_id: str):
            _ = tenant_id
            return [
                {
                    "env": "real",
                    "cano": "64317603",
                    "prdt_cd": "01",
                    "app_key_masked": "appk****1234",
                    "app_secret_masked": "apps****5678",
                    "paper_app_key_masked": "",
                    "paper_app_secret_masked": "",
                }
            ]

    client, repo = _client_with_repo_and_credential_store(monkeypatch, _FakeCredentialStore)
    repo.set_config("local", "kis_account_no", "6431760301")
    repo.set_config("local", "kis_account_product_code", "01")
    repo.set_config("local", "real_trading_approved", "true")

    response = client.get("/settings?tab=capital")

    assert response.status_code == 200
    assert "현재 활성 계좌" in response.text
    assert "******0301" in response.text
    assert "현재 사용 중" in response.text
    assert "appk****1234" in response.text
    assert "apps****5678" in response.text


def test_settings_page_truncates_long_masked_kis_secrets(monkeypatch) -> None:
    class _FakeCredentialStore:
        def __init__(self, *, project, repo):
            _ = project, repo

        def list_kis_accounts_meta(self, *, tenant_id: str):
            _ = tenant_id
            return [
                {
                    "env": "real",
                    "cano": "64317603",
                    "prdt_cd": "01",
                    "app_key_masked": "appk****1234",
                    "app_secret_masked": "apps****5678MASKEDVALUE1234567890TAIL",
                    "paper_app_key_masked": "",
                    "paper_app_secret_masked": "",
                }
            ]

    client, repo = _client_with_repo_and_credential_store(monkeypatch, _FakeCredentialStore)
    repo.set_config("local", "kis_account_no", "6431760301")
    repo.set_config("local", "kis_account_product_code", "01")
    repo.set_config("local", "real_trading_approved", "true")

    response = client.get("/settings?tab=capital")

    assert response.status_code == 200
    assert "apps****...90TAIL" in response.text
    assert "title=\"apps****5678MASKEDVALUE1234567890TAIL\"" in response.text


def test_settings_page_hides_real_kis_fields_in_paper_only_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_DISTRIBUTION_MODE", "paper_only")
    client = _client(monkeypatch)

    response = client.get("/settings?tab=capital")

    assert response.status_code == 200
    assert "공개용 준비 모드" in response.text
    assert ">APP KEY<" not in response.text
    assert ">APP SECRET<" not in response.text
    assert "PAPER APP KEY" in response.text
    assert "PAPER APP SECRET" in response.text


def test_settings_save_rejects_real_kis_keys_when_tenant_unapproved(monkeypatch) -> None:
    class _CapturingCredentialStore:
        save_called = False

        def __init__(self, *, project, repo):
            _ = project, repo

        def list_kis_accounts_meta(self, *, tenant_id: str):
            _ = tenant_id
            return []

        def save_kis_accounts(self, *, tenant_id: str, updated_by: str, accounts: list[dict[str, str]], notes: str = ""):
            _ = tenant_id, updated_by, accounts, notes
            type(self).save_called = True
            raise AssertionError("save_kis_accounts should not be called")

    client, _ = _client_with_repo_and_credential_store(monkeypatch, _CapturingCredentialStore)
    payload = [
        {
            "env": "real",
            "account_no": "64317603-01",
            "app_key": "real-app-key",
            "app_secret": "real-app-secret",
            "paper_app_key": "",
            "paper_app_secret": "",
        }
    ]

    response = client.post(
        "/settings/save",
        data={"tenant_id": "local", "kis_accounts_json": json.dumps(payload)},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert "tenant is not approved for real KIS credentials" in response.text
    assert _CapturingCredentialStore.save_called is False


def test_settings_save_strips_real_kis_keys_in_paper_only_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_DISTRIBUTION_MODE", "paper_only")

    class _CapturingCredentialStore:
        last_accounts: list[dict[str, str]] | None = None

        def __init__(self, *, project, repo):
            _ = project, repo

        def list_kis_accounts_meta(self, *, tenant_id: str):
            _ = tenant_id
            return []

        def save_kis_accounts(self, *, tenant_id: str, updated_by: str, accounts: list[dict[str, str]], notes: str = ""):
            _ = tenant_id, updated_by, notes
            type(self).last_accounts = [dict(account) for account in accounts]

            class _Refs:
                tenant_id = "local"
                kis_secret_name = "tenant-kis"
                model_secret_name = "tenant-models"

            return _Refs()

    client, _ = _client_with_repo_and_credential_store(monkeypatch, _CapturingCredentialStore)
    payload = [
        {
            "env": "demo",
            "account_no": "64317603-01",
            "app_key": "",
            "app_secret": "",
            "paper_app_key": "paper-app-key",
            "paper_app_secret": "paper-app-secret",
        }
    ]

    response = client.post(
        "/settings/save",
        data={"tenant_id": "local", "kis_accounts_json": json.dumps(payload)},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert _CapturingCredentialStore.last_accounts == [
        {
            "env": "demo",
            "account_no": "64317603-01",
            "app_key": "",
            "app_secret": "",
            "paper_app_key": "paper-app-key",
            "paper_app_secret": "paper-app-secret",
        }
    ]


def test_settings_page_offers_paper_connection_in_simulated_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_DISTRIBUTION_MODE", "simulated_only")
    client = _client(monkeypatch)

    response = client.get("/settings?tab=capital")

    assert response.status_code == 200
    assert "초기 온보딩 모드" in response.text
    assert "PAPER APP KEY" in response.text
    assert "PAPER APP SECRET" in response.text
    assert ">APP KEY<" not in response.text
    assert ">APP SECRET<" not in response.text


def test_settings_save_promotes_simulated_tenant_to_paper_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_DISTRIBUTION_MODE", "simulated_only")

    class _CapturingCredentialStore:
        last_accounts: list[dict[str, str]] | None = None

        def __init__(self, *, project, repo):
            _ = project, repo

        def list_kis_accounts_meta(self, *, tenant_id: str):
            _ = tenant_id
            return []

        def save_kis_accounts(self, *, tenant_id: str, updated_by: str, accounts: list[dict[str, str]], notes: str = ""):
            _ = tenant_id, updated_by, notes
            type(self).last_accounts = [dict(account) for account in accounts]

            class _Refs:
                tenant_id = "local"
                kis_secret_name = "tenant-kis"
                model_secret_name = "tenant-models"

            return _Refs()

    client, repo = _client_with_repo_and_credential_store(monkeypatch, _CapturingCredentialStore)
    payload = [
        {
            "env": "demo",
            "account_no": "64317603-01",
            "app_key": "",
            "app_secret": "",
            "paper_app_key": "demo-app-key",
            "paper_app_secret": "demo-app-secret",
        }
    ]

    response = client.post(
        "/settings/save",
        data={"tenant_id": "local", "kis_accounts_json": json.dumps(payload)},
        follow_redirects=False,
    )

    assert response.status_code == 200
    assert "mode=paper_only" in response.text
    assert repo.get_config("local", "distribution_mode") == "paper_only"
    assert _CapturingCredentialStore.last_accounts == [
        {
            "env": "demo",
            "account_no": "64317603-01",
            "app_key": "",
            "app_secret": "",
            "paper_app_key": "demo-app-key",
            "paper_app_secret": "demo-app-secret",
        }
    ]


def test_admin_routes_save_config(monkeypatch) -> None:
    client = _client(monkeypatch)

    prompt_save = client.post(
        "/admin/prompt",
        data={"tenant_id": "local", "updated_by": "tester", "system_prompt": "hello {agent_id}"},
        follow_redirects=False,
    )
    assert prompt_save.status_code == 303

    agents_get = client.get("/admin/agents")
    assert agents_get.status_code == 200
    assert "agent_ids" in agents_get.json()

    mcp_save = client.post(
        "/admin/tools/mcp",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "mcp_servers_json": json.dumps(
                [{"name": "sig", "url": "https://example.com/sse", "transport": "sse", "enabled": True}]
            ),
        },
        follow_redirects=False,
    )
    assert mcp_save.status_code == 303


def test_admin_agents_exposes_invalid_runtime_config_keys(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("local", "risk_policy", "{bad json")
    repo.set_config("local", "disabled_tools", "{\"not\":\"a list\"}")
    repo.set_config("local", "mcp_servers", "{\"not\":\"a list\"}")

    response = client.get("/admin/agents")

    assert response.status_code == 200
    payload = response.json()
    assert payload["invalid_runtime_config_keys"] == ["risk_policy", "disabled_tools", "mcp_servers"]


def test_admin_agent_save_one_preserves_default_agents_when_agents_config_missing(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_AGENT_IDS", "gpt,gemini,claude")
    client, repo = _client_with_repo(monkeypatch)

    response = client.post(
        "/admin/agents/save-one",
        json={
            "tenant_id": "local",
            "agent": {
                "id": "gpt",
                "provider": "gpt",
                "model": "gpt-5.4",
                "capital_krw": 2000000,
                "target_market": "us",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True

    raw = repo.get_config("local", "agents_config")
    assert raw is not None
    saved = json.loads(raw)
    assert [str(entry["id"]) for entry in saved] == ["gpt", "gemini", "claude"]
    saved_by_id = {str(entry["id"]): entry for entry in saved}
    assert saved_by_id["gpt"]["model"] == "gpt-5.4"
    assert saved_by_id["gemini"]["model"] == "gemini-3-flash-preview"
    assert saved_by_id["claude"]["model"] == "claude-sonnet-4-6"


def test_admin_agent_save_one_syncs_tenant_market_from_agent_target_market(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_AGENT_IDS", "gpt,gemini,claude")
    client, repo = _client_with_repo(monkeypatch)

    response = client.post(
        "/admin/agents/save-one",
        json={
            "tenant_id": "local",
            "agent": {
                "id": "claude",
                "provider": "claude",
                "model": "claude-sonnet-4-6",
                "capital_krw": 2000000,
                "target_market": "kospi",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert repo.get_config("local", "kis_target_market") == "kospi"


def test_admin_agent_save_one_syncs_union_tenant_market(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_AGENT_IDS", "gpt,gemini,claude")
    client, repo = _client_with_repo(monkeypatch)

    first = client.post(
        "/admin/agents/save-one",
        json={
            "tenant_id": "local",
            "agent": {
                "id": "gpt",
                "provider": "gpt",
                "model": "gpt-5.4",
                "capital_krw": 2000000,
                "target_market": "us",
            },
        },
    )
    second = client.post(
        "/admin/agents/save-one",
        json={
            "tenant_id": "local",
            "agent": {
                "id": "claude",
                "provider": "claude",
                "model": "claude-sonnet-4-6",
                "capital_krw": 2000000,
                "target_market": "kospi",
            },
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert repo.get_config("local", "kis_target_market") == "us,kospi"


def test_admin_agent_save_one_does_not_rehydrate_defaults_from_explicit_empty_config(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_AGENT_IDS", "gpt,gemini,claude")
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("local", "agents_config", "[]")

    response = client.post(
        "/admin/agents/save-one",
        json={
            "tenant_id": "local",
            "agent": {
                "id": "gpt",
                "provider": "gpt",
                "model": "gpt-5.4",
                "capital_krw": 2000000,
                "target_market": "us",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True

    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert [str(entry["id"]) for entry in saved] == ["gpt"]


def test_admin_agent_save_one_saves_provider_scoped_api_key(monkeypatch) -> None:
    class _FakeCredentialStore:
        last_kwargs: dict[str, object] | None = None

        def __init__(self, *, project, repo):
            _ = project, repo

        def list_kis_accounts_meta(self, *, tenant_id: str):
            _ = tenant_id
            return []

        def save_model_keys(self, **kwargs):
            type(self).last_kwargs = dict(kwargs)

    client, _ = _client_with_repo_and_credential_store(monkeypatch, _FakeCredentialStore)

    response = client.post(
        "/admin/agents/save-one",
        json={
            "tenant_id": "local",
            "agent": {
                "id": "gpt",
                "provider": "openai",
                "model": "gpt-5.4",
                "capital_krw": 2000000,
                "api_key": "tenant-openai",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert _FakeCredentialStore.last_kwargs == {
        "tenant_id": "local",
        "updated_by": "local@localhost",
        "providers": {"gpt": {"api_key": "tenant-openai", "model": "gpt-5.4"}},
    }


def test_admin_agent_save_one_accepts_registry_backed_adk_provider(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)

    response = client.post(
        "/admin/agents/save-one",
        json={
            "tenant_id": "local",
            "agent": {
                "id": "deepseek",
                "provider": "deepseek",
                "model": "deepseek-chat",
                "capital_krw": 1000000,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert any(str(entry.get("provider")) == "deepseek" for entry in saved if isinstance(entry, dict))


def test_admin_agent_save_one_partial_update_preserves_existing_fields(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config(
        "local",
        "agents_config",
        json.dumps(
            [
                {
                    "id": "gpt",
                    "provider": "gpt",
                    "model": "gpt-5.4",
                    "capital_krw": 1500000,
                    "target_market": "kospi",
                    "system_prompt": "keep",
                    "risk_policy": {"max_order_krw": 123},
                }
            ]
        ),
    )

    response = client.post(
        "/admin/agents/save-one",
        json={
            "tenant_id": "local",
            "agent": {
                "agent_id": "gpt",
                "disabled_tools": ["screen_market"],
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert len(saved) == 1
    entry = saved[0]
    assert entry["id"] == "gpt"
    assert entry["provider"] == "gpt"
    assert entry["model"] == "gpt-5.4"
    assert entry["capital_krw"] == 1500000
    assert entry["target_market"] == "kospi"
    assert entry["system_prompt"] == "keep"
    assert entry["risk_policy"] == {"max_order_krw": 123}
    assert entry["disabled_tools"] == ["screen_market"]


def test_admin_agent_save_one_syncs_runtime_state(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_AGENT_IDS", "gpt,gemini,claude")
    client, repo = _client_with_repo(monkeypatch)

    response = client.post(
        "/admin/agents/save-one",
        json={
            "tenant_id": "local",
            "agent": {
                "id": "gpt",
                "provider": "gpt",
                "model": "gpt-5.4",
                "capital_krw": 2000000,
                "target_market": "us",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert repo.capital_sync_calls
    last = repo.capital_sync_calls[-1]
    assert last["tenant_id"] == "local"
    assert float(last["target_capitals"]["gpt"]) == 2000000.0
    assert repo.nav_upsert_calls
    assert {str(row["agent_id"]) for row in repo.nav_upsert_calls} == {"gpt", "gemini", "claude"}


def test_admin_agent_remove_one_persists_removed_default_agent(monkeypatch) -> None:
    class _ZeroSleeveRepo(_DummyRepo):
        def build_agent_sleeve_snapshot(self, *, agent_id: str, sources=None, include_simulated: bool = True, tenant_id: str | None = None):
            self.snapshot_calls.append(
                {
                    "agent_id": agent_id,
                    "sources": list(sources) if isinstance(sources, list) else sources,
                    "include_simulated": include_simulated,
                    "tenant_id": tenant_id,
                }
            )
            return (
                AccountSnapshot(cash_krw=0.0, total_equity_krw=0.0, positions={}),
                0.0,
                {"agent_id": agent_id},
            )

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setenv("ARENA_AGENT_IDS", "gpt,gemini,claude")
    repo = _ZeroSleeveRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.post(
        "/admin/agents/remove-one",
        json={"tenant_id": "local", "agent_id": "gpt"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["remaining_agent_ids"] == ["gemini", "claude"]

    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert [str(entry["id"]) for entry in saved] == ["gemini", "claude"]


def test_admin_agent_remove_one_requires_confirmation_for_key_or_active_capital(monkeypatch) -> None:
    class _KeyedRepo(_DummyRepo):
        def latest_runtime_credentials(self, *, tenant_id: str) -> dict[str, object]:
            _ = tenant_id
            return {"has_openai": True}

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setenv("ARENA_AGENT_IDS", "gpt,gemini,claude")
    repo = _KeyedRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.post(
        "/admin/agents/remove-one",
        json={"tenant_id": "local", "agent_id": "gpt"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is False
    assert payload["confirm_required"] is True
    assert "API key" in payload["message"]

    forced = client.post(
        "/admin/agents/remove-one",
        json={"tenant_id": "local", "agent_id": "gpt", "force": True},
    )

    assert forced.status_code == 200
    assert forced.json()["ok"] is True
    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert [str(entry["id"]) for entry in saved] == ["gemini", "claude"]


def test_admin_sleeve_save_prefers_capital_sync_over_legacy_sleeve_sync(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_AGENT_IDS", "gemini,gpt,claude")
    client, repo = _client_with_repo(monkeypatch)
    response = client.post(
        "/admin/sleeve",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "sleeve_capital_krw": "500000",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert repo.get_config("local", "sleeve_capital_krw") == "500000.0"
    assert repo.capital_sync_calls
    assert repo.sleeve_sync_calls == []
    last = repo.capital_sync_calls[-1]
    assert last["tenant_id"] == "local"
    assert float(last["target_sleeve_capital_krw"]) == 500000.0
    assert repo.nav_upsert_calls
    assert {str(r["agent_id"]) for r in repo.nav_upsert_calls} == {"gpt", "gemini", "claude"}


def test_admin_sleeve_save_uses_tenant_market_sources_in_live_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TRADING_MODE", "live")
    monkeypatch.setenv("ARENA_AGENT_IDS", "gemini,gpt,claude")
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("tenant-k", "kis_target_market", "kospi")

    response = client.post(
        "/admin/sleeve",
        data={
            "tenant_id": "tenant-k",
            "updated_by": "tester",
            "sleeve_capital_krw": "500000",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert repo.capital_sync_calls
    last = repo.capital_sync_calls[-1]
    assert last["tenant_id"] == "tenant-k"
    assert last["include_simulated"] is False
    assert last["sources"] == ["open_trading_kospi_quote", "open_trading_kospi"]
    assert repo.snapshot_calls
    assert all(call["tenant_id"] == "tenant-k" for call in repo.snapshot_calls)
    assert all(call["include_simulated"] is False for call in repo.snapshot_calls)
    assert all(call["sources"] == ["open_trading_kospi_quote", "open_trading_kospi"] for call in repo.snapshot_calls)


def test_admin_sleeve_save_uses_union_market_sources_in_live_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TRADING_MODE", "live")
    monkeypatch.setenv("ARENA_AGENT_IDS", "gemini,gpt,claude")
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("tenant-k", "kis_target_market", "us,kospi")

    response = client.post(
        "/admin/sleeve",
        data={
            "tenant_id": "tenant-k",
            "updated_by": "tester",
            "sleeve_capital_krw": "500000",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    last = repo.capital_sync_calls[-1]
    assert last["tenant_id"] == "tenant-k"
    assert last["include_simulated"] is False
    assert last["sources"] == [
        "open_trading_us_quote",
        "open_trading_us",
        "open_trading_nasdaq_quote",
        "open_trading_nasdaq",
        "open_trading_nyse_quote",
        "open_trading_nyse",
        "open_trading_amex_quote",
        "open_trading_amex",
        "open_trading_kospi_quote",
        "open_trading_kospi",
    ]


def test_memory_config_save_scopes_compaction_prompt_to_tenant(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)

    response = client.post(
        "/api/memory/config",
        params={"tenant_id": "local"},
        json={
            "policy": {"compaction": {"enabled": True}},
            "compaction_prompt": "TENANT PROMPT {agent_id}",
        },
    )

    assert response.status_code == 200
    assert repo.get_config("local", "memory_compactor_prompt") == "TENANT PROMPT {agent_id}"
    assert repo.get_config("global", "memory_compactor_prompt") is None
    payload = response.json()
    assert payload["meta"]["tenant_compaction_prompt"] == "TENANT PROMPT {agent_id}"
    assert payload["meta"]["prompt_source"] == "tenant"


def test_memory_settings_page_uses_compact_prompt_copy(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/settings?tenant_id=local&tab=memory")

    assert response.status_code == 200
    assert "Memory Map" in response.text
    assert "Map" in response.text
    assert "Activity" in response.text
    assert "Network" in response.text
    assert "회고 정리 안내문" in response.text
    assert "투자 논리 시작" in response.text
    assert "닫힌 논리 체인 우선" in response.text
    assert "현재 global 기본 프롬프트를 상속 중입니다" not in response.text
    assert "현재 tenant 전용 컴팩션 프롬프트가 적용됩니다" not in response.text


def test_api_memory_graph_exposes_runtime_stats_and_select_fields(monkeypatch) -> None:
    class _MemoryStatsRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "GROUP BY event_type, agent_id" in sql:
                return [
                    {
                        "event_type": "strategy_reflection",
                        "agent_id": "gpt",
                        "cnt": 3,
                        "last_created_at": "2026-03-15T09:10:11Z",
                    }
                ]
            if "COUNTIF(TRIM(COALESCE(graph_node_id" in sql:
                return [
                    {
                        "total_memory_events": 5,
                        "with_graph_node_id": 4,
                        "with_causal_chain_id": 3,
                        "with_last_accessed_at": 2,
                        "with_effective_score": 5,
                        "last_accessed_at": "2026-03-15T10:11:12Z",
                    }
                ]
            if "GROUP BY memory_tier" in sql:
                return [
                    {"memory_tier": "semantic", "cnt": 2},
                    {"memory_tier": "episodic", "cnt": 3},
                ]
            if "FROM `proj.ds.memory_access_events`" in sql:
                return [
                    {
                        "access_event_count": 12,
                        "prompt_use_count": 5,
                        "last_accessed_at": "2026-03-15T11:12:13Z",
                    }
                ]
            if "FROM `proj.ds.memory_graph_nodes`" in sql:
                return [
                    {"node_kind": "memory_event", "cnt": 6, "last_created_at": "2026-03-15T12:13:14Z"},
                    {"node_kind": "execution_report", "cnt": 2, "last_created_at": "2026-03-15T12:13:14Z"},
                ]
            if "FROM `proj.ds.memory_graph_edges`" in sql:
                return [
                    {"edge_type": "EXECUTED_AS", "cnt": 2, "last_created_at": "2026-03-15T12:14:15Z"},
                    {"edge_type": "ABSTRACTED_TO", "cnt": 1, "last_created_at": "2026-03-15T12:14:15Z"},
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _MemoryStatsRepo()
    repo.set_config(
        "local",
        "memory_forgetting_tuning_state",
        json.dumps(
            {
                "configured_mode": "shadow",
                "effective_mode": "bounded_ema",
                "transition": {"action": "auto_promote", "reason": "stable enough"},
                "drift": {"recommendation_drift": 0.12},
                "history": {"shadow_runs_since_transition": 0, "bounded_ema_runs_since_transition": 3},
            }
        ),
    )
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/memory/graph", params={"tenant_id": "local"})

    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-store"
    payload = response.json()
    runtime = payload["meta"]["runtime"]
    assert runtime["graph"]["total_nodes"] == 8
    assert runtime["graph"]["total_edges"] == 3
    assert runtime["memory"]["with_graph_node_id"] == 4
    assert runtime["forgetting_tuning_state"]["effective_mode"] == "bounded_ema"

    access_curve_node = next(node for node in payload["nodes"] if node["id"] == "forgetting.access_curve")
    tuning_mode_node = next(node for node in payload["nodes"] if node["id"] == "forgetting.tuning.mode")
    assert access_curve_node["type"] == "select"
    assert access_curve_node["options"] == ["sqrt", "log", "capped_linear"]
    assert tuning_mode_node["type"] == "select"
    assert tuning_mode_node["options"] == ["shadow", "bounded_ema"]


def test_memory_graph_runtime_payload_marks_invalid_tuning_state(monkeypatch) -> None:
    class _MemoryStatsRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _MemoryStatsRepo()
    repo.set_config("local", "memory_forgetting_tuning_state", "[1,2,3]")
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/memory/graph", params={"tenant_id": "local"})

    assert response.status_code == 200
    runtime = response.json()["meta"]["runtime"]
    assert runtime["forgetting_tuning_state"] == {}
    assert runtime["invalid_config_keys"] == ["memory_forgetting_tuning_state"]


def test_api_memory_activity_returns_examples(monkeypatch) -> None:
    class _MemoryActivityRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "GROUP BY event_type, agent_id" in sql:
                return [{"event_type": "strategy_reflection", "agent_id": "gpt", "cnt": 2, "last_created_at": "2026-03-15T09:10:11Z"}]
            if "COUNTIF(TRIM(COALESCE(graph_node_id" in sql:
                return [{"total_memory_events": 2, "with_graph_node_id": 1, "with_causal_chain_id": 1, "with_last_accessed_at": 1, "with_effective_score": 2, "last_accessed_at": "2026-03-15T10:11:12Z"}]
            if "GROUP BY memory_tier" in sql:
                return [{"memory_tier": "semantic", "cnt": 1}, {"memory_tier": "episodic", "cnt": 1}]
            if "FROM `proj.ds.memory_access_events`" in sql and "GROUP BY event_id" not in sql:
                return [{"access_event_count": 4, "prompt_use_count": 2, "last_accessed_at": "2026-03-15T11:12:13Z"}]
            if "FROM `proj.ds.memory_graph_nodes`" in sql:
                return [{"node_kind": "memory_event", "cnt": 2, "last_created_at": "2026-03-15T12:13:14Z"}]
            if "FROM `proj.ds.memory_graph_edges`" in sql:
                return [{"edge_type": "INFORMED_BY", "cnt": 1, "last_created_at": "2026-03-15T12:14:15Z"}]
            if "LEFT JOIN access_summary AS a" in sql:
                return [
                    {
                        "event_id": "evt_1",
                        "created_at": "2026-03-15T08:00:00Z",
                        "agent_id": "gpt",
                        "event_type": "strategy_reflection",
                        "summary": "AAPL breakout thesis improved after volume confirmation",
                        "memory_tier": "semantic",
                        "primary_regime": "bull",
                        "primary_strategy_tag": "breakout",
                        "primary_sector": "Technology",
                        "access_count": 3,
                        "last_accessed_at": "2026-03-15T11:12:13Z",
                        "effective_score": 0.81,
                        "context_tags_json": json.dumps({"regime_tags": ["bull"], "strategy_tags": ["breakout"], "sector_tags": ["tech"]}),
                        "payload_json": json.dumps({"ticker": "AAPL"}),
                        "access_events": 3,
                        "prompt_uses": 2,
                        "last_prompt_at": "2026-03-15T11:00:00Z",
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    app = _build_app(repo=_MemoryActivityRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/memory/activity", params={"tenant_id": "local"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["access_runtime"]["prompt_use_count"] == 2
    assert payload["examples"][0]["ticker"] == "AAPL"
    assert payload["examples"][0]["prompt_uses"] == 2
    assert "bull" in payload["examples"][0]["badges"]


def test_api_memory_network_returns_nodes_and_links(monkeypatch) -> None:
    class _MemoryNetworkRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "FROM `proj.ds.memory_graph_nodes` AS n" in sql:
                return [
                    {
                        "node_id": "mem_1",
                        "created_at": "2026-03-15T08:00:00Z",
                        "node_kind": "memory_event",
                        "source_table": "agent_memory_events",
                        "source_id": "evt_1",
                        "agent_id": "gpt",
                        "cycle_id": "cycle_1",
                        "summary": "AAPL thesis update",
                        "ticker": "AAPL",
                        "memory_tier": "semantic",
                        "primary_regime": "bull",
                        "context_tags_json": json.dumps({"strategy_tags": ["breakout"]}),
                        "payload_json": json.dumps({"ticker": "AAPL"}),
                        "event_type": "thesis_update",
                        "access_count": 4,
                        "last_accessed_at": "2026-03-15T11:12:13Z",
                        "effective_score": 0.88,
                        "access_events": 4,
                        "prompt_uses": 2,
                    },
                    {
                        "node_id": "mem_2",
                        "created_at": "2026-03-15T08:05:00Z",
                        "node_kind": "memory_event",
                        "source_table": "agent_memory_events",
                        "source_id": "evt_2",
                        "agent_id": "claude",
                        "cycle_id": "cycle_2",
                        "summary": "AAPL trade execution",
                        "ticker": "AAPL",
                        "memory_tier": "episodic",
                        "primary_regime": "bull",
                        "context_tags_json": json.dumps({"strategy_tags": ["breakout"]}),
                        "payload_json": json.dumps({"ticker": "AAPL"}),
                        "event_type": "trade_execution",
                        "access_count": 1,
                        "last_accessed_at": "2026-03-15T11:12:13Z",
                        "effective_score": 0.52,
                        "access_events": 1,
                        "prompt_uses": 0,
                    },
                ]
            if "FROM `proj.ds.memory_graph_edges`" in sql:
                return [
                    {
                        "edge_id": "edge_1",
                        "created_at": "2026-03-15T09:00:00Z",
                        "from_node_id": "mem_1",
                        "to_node_id": "mem_2",
                        "edge_type": "INFORMED_BY",
                        "edge_strength": 0.76,
                        "confidence": 0.82,
                        "causal_chain_id": "chain_1",
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    app = _build_app(repo=_MemoryNetworkRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/memory/network", params={"tenant_id": "local", "days": 30})

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["node_count"] == 2
    assert payload["meta"]["edge_count"] == 1
    assert payload["meta"]["available_agents"] == ["claude", "gpt"]
    assert payload["nodes"][0]["used_in_prompt"] is True
    assert payload["links"][0]["edge_type"] == "INFORMED_BY"


def test_admin_tools_lists_core_and_optional(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/admin/tools")
    assert response.status_code == 200
    payload = response.json()
    entries = payload["tool_entries"]
    tool_ids = {str(entry["tool_id"]) for entry in entries}
    assert len(tool_ids) >= 17
    assert "portfolio_diagnosis" in tool_ids
    assert "screen_market" in tool_ids
    assert "correlation_matrix" not in tool_ids
    assert "momentum_rank" not in tool_ids
    assert "fetch_reddit_sentiment" in tool_ids
    core_entry = next(entry for entry in entries if str(entry["tool_id"]) == "portfolio_diagnosis")
    optional_entry = next(entry for entry in entries if str(entry["tool_id"]) == "screen_market")
    forecast_entry = next(entry for entry in entries if str(entry["tool_id"]) == "forecast_returns")
    assert core_entry["configurable"] is True
    assert core_entry["tier"] == "core"
    assert core_entry["label_ko"] == "포트폴리오 진단"
    assert optional_entry["configurable"] is True
    assert optional_entry["tier"] == "optional"
    assert optional_entry["label_ko"] == "시장 스크리닝"
    assert "가치주 버킷" in str(optional_entry["description_ko"])
    assert forecast_entry["label_ko"] == "수익률 예측"
    assert "self-discovered 후보 바스켓" in str(forecast_entry["description_ko"])


def test_admin_tools_hides_reddit_when_runtime_disabled(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("local", "reddit_sentiment_enabled", "false")

    response = client.get("/admin/tools")

    assert response.status_code == 200
    payload = response.json()
    tool_ids = {str(entry["tool_id"]) for entry in payload["tool_entries"]}
    assert "fetch_reddit_sentiment" not in tool_ids


def test_admin_tools_save_allows_core_tool_ids(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    response = client.post(
        "/admin/tools",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "disabled_tools": ["screen_market", "portfolio_diagnosis"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    raw = repo.get_config("local", "disabled_tools")
    assert raw is not None
    saved = json.loads(raw)
    assert saved == ["portfolio_diagnosis", "screen_market"]


def test_admin_tools_apply_tools_config_overlay(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config(
        "local",
        "tools_config",
        json.dumps(
            [
                {
                    "tool_id": "portfolio_diagnosis",
                    "ui_label_ko": "포트 진단 오버라이드",
                    "ui_description_ko": "오버라이드 설명",
                },
                {
                    "tool_id": "screen_market",
                    "enabled": False,
                },
            ],
            ensure_ascii=False,
        ),
    )

    response = client.get("/admin/tools")

    assert response.status_code == 200
    payload = response.json()
    entries = {str(entry["tool_id"]): entry for entry in payload["tool_entries"]}
    assert "screen_market" not in entries
    assert entries["portfolio_diagnosis"]["label_ko"] == "포트 진단 오버라이드"
    assert entries["portfolio_diagnosis"]["description_ko"] == "오버라이드 설명"




def test_api_board_uses_tenant_filter(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.fetch_calls.clear()
    response = client.get("/api/board", params={"tenant_id": "tenant-x", "limit": 5})
    assert response.status_code == 200
    assert repo.fetch_calls
    _, params = repo.fetch_calls[-1]
    assert isinstance(params, dict)
    assert params.get("tenant_id") == "tenant-x"


def test_api_nav_uses_tenant_filter(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.fetch_calls.clear()
    response = client.get("/api/nav", params={"tenant_id": "tenant-y", "days": 10})
    assert response.status_code == 200
    assert repo.fetch_calls
    _, params = repo.fetch_calls[-1]
    assert isinstance(params, dict)
    assert params.get("tenant_id") == "tenant-y"


def test_nav_page_renders_blocked_status_in_header(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.latest_run_status_row = {
        "tenant_id": "local",
        "run_id": "cycle_1",
        "recorded_at": "2026-03-14T12:00:00+00:00",
        "run_type": "agent_cycle",
        "status": "blocked",
        "reason_code": "reconciliation_failed",
        "stage": "reconcile",
        "message": "실계좌와 AI 장부가 맞지 않아 거래를 중단했습니다.",
        "log_uri": "https://example.com/logs",
        "detail_json": {"exit_code": 3},
    }

    response = client.get("/nav", params={"tenant_id": "local"})

    assert response.status_code == 200
    # Status label shown in header indicator (not banner)
    assert "실행 중단" in response.text


def test_api_board_trades_requires_cycle_id(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.fetch_calls.clear()
    response = client.get("/api/board/trades", params={"tenant_id": "local"})
    assert response.status_code == 200
    assert response.json() == []
    assert repo.fetch_calls == []


def test_api_board_theses_returns_chain_and_compacted_lesson(monkeypatch) -> None:
    class _BoardThesisRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "event_type IN UNNEST(@event_types)" in sql:
                return [
                    {
                        "created_at": "2026-03-29T01:00:00Z",
                        "agent_id": "gpt",
                        "event_type": "thesis_open",
                        "summary": "AAPL thesis opened",
                        "semantic_key": "thesis:gpt:AAPL:1",
                        "payload_json": json.dumps(
                            {
                                "thesis_id": "thesis:gpt:AAPL:1",
                                "ticker": "AAPL",
                                "side": "BUY",
                                "state": "open",
                                "thesis_summary": "AI demand and margin recovery",
                                "strategy_refs": ["momentum", "quality"],
                            }
                        ),
                    },
                    {
                        "created_at": "2026-03-29T03:00:00Z",
                        "agent_id": "gpt",
                        "event_type": "thesis_invalidated",
                        "summary": "Guidance cut broke the thesis",
                        "semantic_key": "thesis:gpt:AAPL:1",
                        "payload_json": json.dumps(
                            {
                                "thesis_id": "thesis:gpt:AAPL:1",
                                "ticker": "AAPL",
                                "side": "BUY",
                                "state": "invalidated",
                                "thesis_summary": "AI demand and margin recovery",
                            }
                        ),
                    },
                ]
            if "JSON_VALUE(payload_json, '$.source') = 'thesis_chain_compaction'" in sql:
                return [
                    {
                        "created_at": "2026-03-29T05:00:00Z",
                        "summary": "Trim earlier when the thesis starts drifting.",
                        "payload_json": json.dumps(
                            {
                                "source": "thesis_chain_compaction",
                                "thesis_id": "thesis:gpt:AAPL:1",
                            }
                        ),
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _BoardThesisRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get("/api/board/theses", params={"tenant_id": "local", "cycle_id": "cycle_1", "agent_id": "gpt"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["chains"]) == 1
    chain = payload["chains"][0]
    assert chain["thesis_id"] == "thesis:gpt:AAPL:1"
    assert chain["ticker"] == "AAPL"
    assert chain["terminal_event_type"] == "thesis_invalidated"
    assert chain["reflection"]["summary"] == "Trim earlier when the thesis starts drifting."
    assert [event["event_type"] for event in chain["events"]] == ["thesis_open", "thesis_invalidated"]
    event_sql = repo.fetch_calls[0][0]
    assert "cycle_id = @cycle_id" not in event_sql
    assert "JSON_VALUE(payload_json, '$.cycle_id')" in event_sql
    assert "JSON_VALUE(payload_json, '$.intent.cycle_id')" in event_sql


def test_api_board_prompt_returns_prompt_bundle(monkeypatch) -> None:
    class _BoardPromptRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "event_type = 'react_tools_summary'" not in sql:
                return []
            return [
                {
                    "created_at": "2026-03-29T01:02:00Z",
                    "summary": "Board prompt bundle snapshot before post generation.",
                    "payload_json": json.dumps(
                        {
                            "phase": "board",
                            "analysis_funnel": {"pending_nonheld": 1},
                            "tool_events": [{"tool": "screen_market", "phase": "draft"}],
                            "tool_mix": {"quant": 1, "macro": 0, "sentiment": 0, "performance": 0, "context": 0, "other": 0},
                            "prompt_bundle": {
                                "system_prompt": "system body",
                                "phases": [
                                    {"phase": "draft", "session_id": "sid_1", "resume_session": False, "prompt": "draft body"},
                                    {"phase": "execution", "session_id": "sid_1", "resume_session": True, "prompt": "execution body"},
                                    {"phase": "board", "session_id": "sid_1", "resume_session": True, "prompt": "board body"},
                                ],
                            },
                        }
                    ),
                }
            ]

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _BoardPromptRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get(
        "/api/board/prompt",
        params={
            "tenant_id": "local",
            "agent_id": "gpt",
            "ts": "2026-03-29T01:00:00+00:00",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt_bundle"]["system_prompt"] == "system body"
    assert payload["prompt_bundle"]["phases"][0]["prompt"] == "draft body"
    assert payload["analysis_funnel"]["pending_nonheld"] == 1
    assert payload["tool_events"][0]["tool"] == "screen_market"


def test_board_page_includes_prompt_and_memory_panels(monkeypatch) -> None:
    class _BoardRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "FROM `proj.ds.board_posts`" in sql:
                return [
                    {
                        "post_id": "post_1",
                        "created_at": datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc),
                        "agent_id": "gpt",
                        "title": "AAPL review",
                        "body": "Revisited the thesis.",
                        "tickers": ["AAPL"],
                        "cycle_id": "cycle_1",
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _BoardRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get("/board", params={"tenant_id": "local"})

    assert response.status_code == 200
    assert 'data-prompt-panel' in response.text
    assert 'data-theses-panel' in response.text
    assert "/api/board/prompt" in response.text
    assert "/api/board/theses" in response.text
    assert "Prompt Details" in response.text
    assert "Related Memory" in response.text


def test_board_page_empty_state_mentions_missing_gemini_key(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    repo = _DummyRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get("/board", params={"tenant_id": "local"})

    assert response.status_code == 200
    assert "Gemini 키가 없어 새로운 리서치 브리핑 생성도 비활성화되어 있습니다." in response.text


def test_api_tool_frequency_returns_llm_tool_matrix(monkeypatch) -> None:
    class _ToolFreqRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "agent_memory_events" not in sql:
                return []
            return [
                {
                    "agent_id": "gpt",
                    "payload_json": json.dumps(
                        {
                            "tool_events": [
                                {"tool": "screen_market"},
                                {"tool": "screen_market"},
                                {"tool": "legacy_old_tool"},
                            ]
                        }
                    ),
                },
                {
                    "agent_id": "gemini",
                    "payload_json": json.dumps(
                        {
                            "tool_events": [
                                {"tool": "screen_market"},
                                {"tool": "optimize_portfolio"},
                            ]
                        }
                    ),
                },
            ]

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _ToolFreqRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/tool-frequency", params={"tenant_id": "local"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["tools"] == ["screen_market", "optimize_portfolio"]
    assert set(payload["agents"]) == {"gpt", "gemini"}
    assert payload["matrix"]["screen_market"] == {"gpt": 2, "gemini": 1}
    assert payload["matrix"]["optimize_portfolio"] == {"gpt": 0, "gemini": 1}
    assert "legacy_old_tool" not in payload["tools"]


def test_api_nav_chart_includes_token_usage_summary_and_trade_counts(monkeypatch) -> None:
    class _NavRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "agent_memory_events" in sql:
                return [
                    {
                        "agent_id": "gpt",
                        "llm_calls": 5,
                        "prompt_tokens": 1500,
                        "completion_tokens": 300,
                        "cached_tokens": 400,
                        "thinking_tokens": 40,
                    },
                    {
                        "agent_id": "gemini",
                        "llm_calls": 1,
                        "prompt_tokens": 700,
                        "completion_tokens": 60,
                        "cached_tokens": 200,
                        "thinking_tokens": 20,
                    },
                ]
            if "execution_reports" in sql:
                return [
                    {"agent_id": "gpt", "trade_count": 3},
                    {"agent_id": "gemini", "trade_count": 1},
                ]
            if "agent_nav_daily" in sql:
                return [
                    {"nav_date": "2026-03-10", "agent_id": "gpt", "nav_krw": 100.0, "pnl_krw": 0.0, "pnl_ratio": 0.0},
                    {"nav_date": "2026-03-11", "agent_id": "gpt", "nav_krw": 105.0, "pnl_krw": 5.0, "pnl_ratio": 0.05},
                    {"nav_date": "2026-03-10", "agent_id": "gemini", "nav_krw": 100.0, "pnl_krw": 0.0, "pnl_ratio": 0.0},
                    {"nav_date": "2026-03-11", "agent_id": "gemini", "nav_krw": 102.0, "pnl_krw": 2.0, "pnl_ratio": 0.02},
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _NavRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/nav/chart", params={"tenant_id": "local", "days": 30})
    assert response.status_code == 200
    payload = response.json()

    summary = {str(row["name"]): row for row in payload["summary"]}
    assert summary["gpt"]["trade_count"] == 3
    assert summary["gemini"]["trade_count"] == 1
    assert summary["gpt"]["llm_calls"] == 5
    assert summary["gpt"]["prompt_tokens"] == 1500
    assert summary["gpt"]["completion_tokens"] == 300
    assert summary["gpt"]["cached_tokens"] == 400
    assert summary["gpt"]["thinking_tokens"] == 40
    assert summary["gpt"]["total_tokens"] == 1840
    assert summary["gpt"]["cache_ratio"] == 26.7
    assert summary["gemini"]["llm_calls"] == 1
    assert summary["gemini"]["total_tokens"] == 780


def test_api_sleeve_snapshot_cards_returns_html_and_charts(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/api/sleeve-snapshot-cards", params={"tenant_id": "local"})
    assert response.status_code == 200
    payload = response.json()
    assert "html" in payload
    assert "charts" in payload
    assert "gpt" in payload["html"]
    assert isinstance(payload["charts"], list)


def test_api_sleeve_snapshot_cards_uses_tenant_market_sources_in_live_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TRADING_MODE", "live")
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("tenant-k", "kis_target_market", "kospi")

    response = client.get("/api/sleeve-snapshot-cards", params={"tenant_id": "tenant-k"})

    assert response.status_code == 200
    assert repo.snapshot_calls
    assert all(call["tenant_id"] == "tenant-k" for call in repo.snapshot_calls)
    assert all(call["include_simulated"] is False for call in repo.snapshot_calls)
    assert all(call["sources"] == ["open_trading_kospi_quote", "open_trading_kospi"] for call in repo.snapshot_calls)


def test_sleeves_page_uses_tenant_market_sources_in_live_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TRADING_MODE", "live")
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("tenant-k", "kis_target_market", "kospi")

    response = client.get("/sleeves", params={"tenant_id": "tenant-k"})

    assert response.status_code == 200
    assert repo.snapshot_calls
    assert all(call["tenant_id"] == "tenant-k" for call in repo.snapshot_calls)
    assert all(call["include_simulated"] is False for call in repo.snapshot_calls)
    assert all(call["sources"] == ["open_trading_kospi_quote", "open_trading_kospi"] for call in repo.snapshot_calls)


def test_api_capital_waterfall_uses_canonical_baseline_summary(monkeypatch) -> None:
    class _WaterfallRepo(_DummyRepo):
        def build_agent_sleeve_snapshot(
            self,
            *,
            agent_id: str,
            sources=None,
            include_simulated: bool = True,
            tenant_id: str | None = None,
        ):
            _ = (sources, include_simulated, tenant_id)
            return (
                AccountSnapshot(cash_krw=200_000.0, total_equity_krw=2_950_195.0, positions={}),
                3_000_000.0,
                {
                    "seed_cash_krw": 1_000_000.0,
                    "seed_positions_cost_krw": 938_568.0,
                    "capital_flow_krw": 1_061_432.0,
                    "capital_event_count": 1,
                    "transfer_equity_krw": 0.0,
                    "transfer_event_count": 0,
                    "manual_cash_adjustment_krw": 0.0,
                    "manual_cash_adjustment_count": 0,
                    "current_cash_krw": 200_000.0,
                    "current_positions_value_krw": 2_750_195.0,
                    "seed_source": "agent_state_checkpoint",
                    "initialized_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
                },
            )

        def capital_events_since(self, *, agent_id: str, since, tenant_id: str | None = None):
            _ = (agent_id, since, tenant_id)
            return [
                {
                    "occurred_at": datetime(2026, 3, 20, tzinfo=timezone.utc),
                    "event_type": "INJECTION",
                    "amount_krw": 1_061_432.0,
                }
            ]

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _WaterfallRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/capital/waterfall", params={"tenant_id": "local", "agent_id": "gpt"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_invested"] == 3_000_000
    assert payload["summary"]["seed_positions_cost_krw"] == 938_568
    assert payload["summary"]["capital_flow_krw"] == 1_061_432
    assert payload["summary"]["pnl_krw"] == -49_805


def test_api_capital_waterfall_prefers_traced_actual_basis_summary(monkeypatch) -> None:
    class _WaterfallRepo(_DummyRepo):
        def build_agent_sleeve_snapshot(
            self,
            *,
            agent_id: str,
            sources=None,
            include_simulated: bool = True,
            tenant_id: str | None = None,
        ):
            _ = (agent_id, sources, include_simulated, tenant_id)
            return (
                AccountSnapshot(cash_krw=500_000.0, total_equity_krw=3_950_000.0, positions={}),
                4_000_000.0,
                {
                    "seed_cash_krw": 1_000_000.0,
                    "seed_positions_cost_krw": 3_000_000.0,
                    "capital_flow_krw": 2_000_000.0,
                    "capital_event_count": 2,
                    "current_cash_krw": 500_000.0,
                    "current_positions_value_krw": 3_450_000.0,
                    "seed_source": "agent_state_checkpoint",
                    "initialized_at": datetime(2026, 3, 27, tzinfo=timezone.utc),
                },
            )

        def trace_agent_actual_capital_basis(self, *, agent_id: str, tenant_id: str | None = None):
            _ = (agent_id, tenant_id)
            return {
                "origin_at": datetime(2026, 3, 11, tzinfo=timezone.utc),
                "origin_source": "legacy_agent_sleeve",
                "seed_cash_krw": 2_000_000.0,
                "seed_positions_cost_krw": 0.0,
                "baseline_equity_krw": 4_110_000.0,
                "capital_flow_krw": 2_110_000.0,
                "capital_event_count": 2,
                "transfer_equity_krw": 0.0,
                "transfer_event_count": 0,
                "manual_cash_adjustment_krw": 0.0,
                "manual_cash_adjustment_count": 0,
            }

        def capital_events_since(self, *, agent_id: str, since, tenant_id: str | None = None):
            _ = (agent_id, since, tenant_id)
            return [
                {
                    "occurred_at": datetime(2026, 3, 17, tzinfo=timezone.utc),
                    "event_type": "INJECTION",
                    "amount_krw": 1_110_000.0,
                },
                {
                    "occurred_at": datetime(2026, 3, 27, tzinfo=timezone.utc),
                    "event_type": "INJECTION",
                    "amount_krw": 1_000_000.0,
                },
            ]

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _WaterfallRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/capital/waterfall", params={"tenant_id": "local", "agent_id": "gpt"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_invested"] == 4_110_000
    assert payload["summary"]["capital_flow_krw"] == 2_110_000
    assert payload["summary"]["pnl_krw"] == -160_000


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
