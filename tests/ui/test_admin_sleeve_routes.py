from __future__ import annotations

from tests.ui.helpers import _client_with_repo


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
