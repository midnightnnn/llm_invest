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
    assert "capitalSaveStatus" in response_cap.text
    assert "form.requestSubmit" in response_cap.text
    assert "form.submit();" not in response_cap.text
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
