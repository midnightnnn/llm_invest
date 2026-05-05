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
    location = response.headers.get("location", "")
    assert "tab=capital" in location
    assert "Target+Capital" in location
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


def test_admin_tools_lists_core_and_optional(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/admin/tools")
    assert response.status_code == 200
    payload = response.json()
    entries = payload["tool_entries"]
    tool_ids = {str(entry["tool_id"]) for entry in entries}
    assert len(tool_ids) >= 17
    assert "portfolio_diagnosis" in tool_ids
    assert "recommend_opportunities" in tool_ids
    assert "screen_market" not in tool_ids
    assert "correlation_matrix" not in tool_ids
    assert "momentum_rank" not in tool_ids
    assert "fetch_reddit_sentiment" in tool_ids
    core_entry = next(entry for entry in entries if str(entry["tool_id"]) == "portfolio_diagnosis")
    optional_entry = next(entry for entry in entries if str(entry["tool_id"]) == "recommend_opportunities")
    forecast_entry = next(entry for entry in entries if str(entry["tool_id"]) == "forecast_returns")
    assert core_entry["configurable"] is True
    assert core_entry["tier"] == "core"
    assert core_entry["label_ko"] == "포트폴리오 진단"
    assert optional_entry["configurable"] is True
    assert optional_entry["tier"] == "optional"
    assert optional_entry["label_ko"] == "통합 기회 추천"
    assert "신규 매수 후보" in str(optional_entry["description_ko"])
    assert "signal-IC" in str(optional_entry["description_ko"])
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
