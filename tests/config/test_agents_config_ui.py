from __future__ import annotations

import json

from tests.config.agents_config_helpers import _build_test_client


def test_admin_agents_save_with_agents_config_json(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    agents_config = [
        {"id": "gemini", "model": "gemini-3-flash", "capital_krw": 1_500_000, "api_key": ""},
        {"id": "gpt", "model": "gpt-4.1", "capital_krw": 1_000_000, "api_key": ""},
        {"id": "claude", "model": "claude-sonnet-4-6", "capital_krw": 2_000_000, "api_key": ""},
    ]

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "agents_config_json": json.dumps(agents_config),
        },
        follow_redirects=False,
    )

    assert response.status_code == 303

    # Verify agents_config saved to DB
    saved_raw = repo.get_config("local", "agents_config")
    assert saved_raw is not None
    saved = json.loads(saved_raw)
    assert len(saved) == 3
    assert saved[0]["id"] == "gemini"
    assert saved[0]["capital_krw"] == 1_500_000
    # api_key should NOT be in DB config
    assert "api_key" not in saved[0]

    assert repo.get_config("local", "agent_ids") is None
    assert repo.get_config("local", "agent_models") is None

    # Verify sleeve retarget called with per-agent capitals
    assert repo.sleeve_sync_calls
    last = repo.sleeve_sync_calls[-1]
    assert last["target_capitals"]["gemini"] == 1_500_000
    assert last["target_capitals"]["gpt"] == 1_000_000
    assert last["target_capitals"]["claude"] == 2_000_000
    assert last["tenant_id"] == "local"

    # Verify NAV upsert happened
    assert len(repo.nav_upsert_calls) == 3


def test_admin_agents_save_rejects_empty_config(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "agents_config_json": "[]",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    # Should redirect with error
    location = response.headers.get("location", "")
    assert "ok=0" in location


def test_admin_agents_save_rejects_unknown_provider(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "agents_config_json": json.dumps([
                {"id": "unknown_provider", "model": "model-x", "capital_krw": 500_000},
            ]),
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    location = response.headers.get("location", "")
    assert "ok=0" in location


def test_admin_agents_get_returns_agents_config(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    # Pre-save agents_config
    agents_config = [
        {"id": "gpt", "model": "gpt-4.1", "capital_krw": 800_000},
    ]
    repo.cfg[("local", "agents_config")] = json.dumps(agents_config)

    response = client.get("/admin/agents", params={"tenant_id": "local"})
    assert response.status_code == 200
    payload = response.json()
    assert "agents_config" in payload
    assert "api_key_status" in payload
    assert "research_status" in payload
    assert payload["agents_config"][0]["id"] == "gpt"
    assert payload["agents_config"][0]["capital_krw"] == 800_000


def test_settings_page_renders_unified_agents_panel(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    response = client.get("/settings")
    assert response.status_code == 200
    html = response.text

    assert "Agents" in html
    assert "agent-card" in html
    assert "agent-toggle-btn" in html
    assert "agent-save-btn" in html
    # Global checkboxes should be gone — per-agent only
    assert "agent-global-prompt" not in html
    assert "agent-global-risk" not in html
    assert "agent-global-tools" not in html
    # The old sleeve panel tab should be gone
    assert "Sleeve Capital" not in html


def test_admin_agents_save_does_not_sync_global_sleeve_capital(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    agents_config = [
        {"id": "gpt", "model": "gpt-4.1", "capital_krw": 600_000},
        {"id": "gemini", "model": "gemini-3", "capital_krw": 400_000},
    ]

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "agents_config_json": json.dumps(agents_config),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    assert repo.get_config("local", "sleeve_capital_krw") is None


def test_admin_agents_save_single_agent(monkeypatch) -> None:
    """User can configure just one agent."""
    client, repo = _build_test_client(monkeypatch)

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "agents_config_json": json.dumps([
                {"id": "claude", "model": "claude-opus-4-6", "capital_krw": 5_000_000},
            ]),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    saved = json.loads(repo.get_config("local", "agents_config"))
    assert len(saved) == 1
    assert saved[0]["id"] == "claude"
    assert saved[0]["capital_krw"] == 5_000_000

    assert repo.get_config("local", "agent_ids") is None
    assert repo.sleeve_sync_calls
    assert repo.sleeve_sync_calls[-1]["target_capitals"] == {"claude": 5_000_000}


def test_admin_agents_save_with_custom_id_and_provider(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    agents_config = [
        {"id": "aggressive-gpt", "provider": "gpt", "model": "gpt-5.2", "capital_krw": 2_000_000},
        {"id": "safe-claude", "provider": "claude", "model": "claude-sonnet-4-6", "capital_krw": 1_000_000},
    ]

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "agents_config_json": json.dumps(agents_config),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    saved = json.loads(repo.get_config("local", "agents_config"))
    assert len(saved) == 2
    assert saved[0]["id"] == "aggressive-gpt"
    assert saved[0]["provider"] == "gpt"
    assert saved[1]["id"] == "safe-claude"
    assert saved[1]["provider"] == "claude"

    assert repo.get_config("local", "agent_ids") is None


def test_admin_agents_save_with_per_agent_fields(monkeypatch) -> None:
    """Per-agent system_prompt, risk_policy, disabled_tools are saved to DB."""
    client, repo = _build_test_client(monkeypatch)

    agents_config = [
        {
            "id": "custom-agent",
            "provider": "gpt",
            "model": "gpt-5.2",
            "capital_krw": 1_000_000,
            "system_prompt": "Be aggressive trader.",
            "risk_policy": {"max_order_krw": 50_000_000},
            "disabled_tools": ["screen_market"],
        },
    ]

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "agents_config_json": json.dumps(agents_config),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    saved = json.loads(repo.get_config("local", "agents_config"))
    assert len(saved) == 1
    assert saved[0]["system_prompt"] == "Be aggressive trader."
    assert saved[0]["risk_policy"] == {"max_order_krw": 50_000_000}
    assert saved[0]["disabled_tools"] == ["screen_market"]
