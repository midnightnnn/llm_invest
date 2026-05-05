from __future__ import annotations

import json

from arena.config import load_settings
from arena.models import AccountSnapshot
from arena.ui.server import _build_app
from tests.direct_route_client import DirectRouteClient
from tests.ui.helpers import _DummyRepo


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
