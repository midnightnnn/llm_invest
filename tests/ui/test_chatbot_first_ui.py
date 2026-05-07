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


def test_owner_nav_has_four_items_with_chat_first() -> None:
    from arena.ui.layout import tailwind_layout

    html = tailwind_layout("Test", "<p>body</p>", active="board", tenant="local")
    # Extract sidebar markup only.
    aside_segments = html.split('<aside')
    assert len(aside_segments) >= 2
    aside = aside_segments[1].split("</aside>")[0]

    # Required entries.
    for label in ("투자챗봇", "게시판", "운용성과", "환경설정"):
        assert label in aside, f"missing {label}"

    # Removed entries (no individual settings tabs in owner sidebar).
    for removed in ("에이전트", "자본관리", "도구관리", "기억관리"):
        assert removed not in aside, f"{removed} should not be a sidebar link in owner nav"

    # Chat is first.
    chat_idx = aside.index("투자챗봇")
    board_idx = aside.index("게시판")
    nav_idx = aside.index("운용성과")
    settings_idx = aside.index("환경설정")
    assert chat_idx < board_idx < nav_idx < settings_idx


def test_showcase_nav_has_chat_first() -> None:
    from arena.ui.layout import tailwind_layout

    html = tailwind_layout(
        "Test", "<p>body</p>", active="board", tenant="acme", showcase=True,
    )
    aside_segments = html.split('<aside')
    assert len(aside_segments) >= 2
    aside = aside_segments[1].split("</aside>")[0]

    # All 7 entries present in showcase.
    for label in ("투자챗봇", "게시판", "운용성과", "에이전트", "자본관리", "도구관리", "기억관리"):
        assert label in aside, f"missing {label} in showcase nav"

    # Chat is first.
    chat_idx = aside.index("투자챗봇")
    board_idx = aside.index("게시판")
    assert chat_idx < board_idx

    # Showcase chat link points to /showcase/{tenant}/investment-chat.
    assert "/showcase/acme/investment-chat" in aside


def test_adk_readonly_mount_exists(monkeypatch) -> None:
    from tests.ui.helpers import _client

    client = _client(monkeypatch)
    paths = {getattr(route, "path", "") for route in client.app.routes}
    # FastAPI's app.mount creates a Mount route whose path is the prefix.
    assert "/investment-chat/adk-readonly" in paths


def test_settings_page_shows_chatbot_banner(monkeypatch) -> None:
    client = _client(monkeypatch)
    resp = client.get("/settings", params={"tab": "agents"})
    body = resp.text
    assert "투자챗봇으로 변경 가능합니다" in body


def test_settings_banner_appears_on_all_tabs(monkeypatch) -> None:
    client = _client(monkeypatch)
    for tab in ("agents", "capital", "mcp", "memory"):
        resp = client.get("/settings", params={"tab": tab})
        assert "투자챗봇으로 변경 가능합니다" in resp.text, f"missing banner on tab={tab}"
