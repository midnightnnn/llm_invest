from __future__ import annotations

from tests.ui.helpers import _client


def test_root_redirects_to_investment_chat(monkeypatch) -> None:
    client = _client(monkeypatch)
    resp = client.get("/")
    assert resp.status_code == 302
    assert resp.headers["location"].startswith("/investment-chat")


def test_root_redirect_preserves_tenant_query(monkeypatch) -> None:
    client = _client(monkeypatch)
    resp = client.get("/", params={"tenant_id": "local"})
    assert resp.status_code == 302
    location = resp.headers["location"]
    assert location.startswith("/investment-chat")
    assert "tenant_id=local" in location
