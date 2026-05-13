from __future__ import annotations

import json

from tests.ui.helpers import _client_with_repo, _client_with_repo_and_credential_store


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


def test_admin_agent_save_one_rejects_deepseek_until_adk_implemented(monkeypatch) -> None:
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

    assert response.status_code == 400
    assert response.json()["ok"] is False
    assert "provider" in response.json()["message"].lower()


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
