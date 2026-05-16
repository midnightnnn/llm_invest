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


def test_settings_page_has_internal_tab_links(monkeypatch) -> None:
    client = _client(monkeypatch)
    for active_tab in ("agents", "capital", "mcp", "memory"):
        body = client.get("/settings", params={"tab": active_tab}).text
        assert 'aria-label="환경설정 탭"' in body
        for tab, label in (
            ("agents", "에이전트"),
            ("capital", "자본관리"),
            ("mcp", "도구관리"),
            ("memory", "기억관리"),
        ):
            assert f"tab={tab}" in body
            assert label in body


def test_settings_agents_tab_has_chat_provider_card(monkeypatch) -> None:
    client = _client(monkeypatch)
    body = client.get("/settings", params={"tab": "agents"}).text
    assert "Chat Models" in body
    assert 'action="/settings/chat-model"' in body
    assert "/settings/chat-model/options" in body
    assert "settings-chat-model-map" not in body
    assert 'name="provider"' in body
    assert 'name="model"' in body


def test_showcase_agents_tab_has_no_chat_provider_card(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_SHOWCASE_TENANT", "midnightnnn")
    client = _client(monkeypatch)
    body = client.get("/showcase/midnightnnn/settings", params={"tab": "agents"}).text
    assert "Chat Provider/Model" not in body
    assert "/settings/chat-model" not in body


def test_chat_model_post_valid_saves_and_redirects(monkeypatch) -> None:
    from arena.agents.investment_chat.config_tools import load_chat_agent_config
    from tests.ui.helpers import _client_with_repo

    monkeypatch.setattr(
        "arena.ui.routes.settings_admin.discover_saved_model_options",
        lambda credential_store, tenant_id, provider: {
            "provider": provider,
            "advisor_models": ["gpt-5.5"],
            "router_models": ["gpt-5.4-mini"],
            "utility_models": ["gpt-5.4-mini"],
        },
    )
    client, repo = _client_with_repo(monkeypatch)
    resp = client.post(
        "/settings/chat-model",
        data={
            "tenant_id": "local",
            "provider": "gpt",
            "model": "gpt-5.5",
            "router_model": "gpt-5.4-mini",
            "utility_model": "gpt-5.4-mini",
        },
    )
    assert resp.status_code in (302, 303), resp.status_code
    location = resp.headers["location"]
    assert "/settings" in location
    assert "tab=agents" in location

    # Verify saved.
    saved = load_chat_agent_config(repo, tenant_id="local")
    assert saved.get("provider") == "gpt"
    assert saved.get("model") == "gpt-5.5"
    assert saved["model_routing"]["router_model"] == "gpt-5.4-mini"


def test_chat_model_options_endpoint_uses_saved_api_key(monkeypatch, tmp_path) -> None:
    import json

    from tests.ui.helpers import _client_with_repo

    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text(
        json.dumps({"local-local-models": {"providers": {"gpt": {"api_key": "sk-saved"}}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARENA_LOCAL_CREDENTIALS_FILE", str(credentials_path))
    seen: dict[str, str] = {}

    def _discover(provider: str, api_key: str):
        seen["provider"] = provider
        seen["api_key"] = api_key
        return {
            "provider": provider,
            "advisor_models": ["gpt-5.5"],
            "router_models": ["gpt-5.4-mini"],
            "utility_models": ["gpt-5.4-mini"],
        }

    monkeypatch.setattr("arena.ui.routes.settings_admin.discover_model_options_with_api_key", _discover)
    client, _repo = _client_with_repo(monkeypatch)

    resp = client.post(
        "/settings/chat-model/options",
        data={"tenant_id": "local", "provider": "gpt"},
    )

    assert resp.status_code == 200
    assert resp.json()["advisor_models"] == ["gpt-5.5"]
    assert seen == {"provider": "gpt", "api_key": "sk-saved"}


def test_chat_model_post_invalid_provider_rejects(monkeypatch) -> None:
    from tests.ui.helpers import _client_with_repo

    client, _repo = _client_with_repo(monkeypatch)
    resp = client.post(
        "/settings/chat-model",
        data={"tenant_id": "local", "provider": "no-such-provider", "model": "x"},
    )
    assert resp.status_code in (400, 422)


def test_investment_chat_page_has_no_selector_form(monkeypatch) -> None:
    client = _client(monkeypatch)
    body = client.get("/investment-chat").text
    # Selector form removed.
    assert "data-chat-selector-form" not in body
    assert "data-chat-provider" not in body
    assert "data-chat-model" not in body
    # Without investment_chat_config and saved model key, the setup panel is shown.
    assert "LLM API key 연결" in body
    assert "data-adk-chat-frame" not in body
    assert "data-order-draft-panel" not in body
    assert "data-config-draft-panel" not in body
    assert "initializeAdkSidePanelClosed" not in body
    assert "adk-side-panel-visible" not in body
    assert "installAdkNoShiftStyles" not in body
    assert "arena-adk-no-shift-styles" not in body
    assert "removeStaleAdkDrawerStabilizer" not in body
    assert "transition-duration: 0ms" not in body
    assert 'body[data-active="investment_chat"] .arena-sidebar' in body


def test_showcase_investment_chat_page_renders(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_SHOWCASE_TENANT", "midnightnnn")
    client = _client(monkeypatch)
    resp = client.get("/showcase/midnightnnn/investment-chat")
    assert resp.status_code == 200, resp.status_code
    body = resp.text
    # Without investment_chat_config, showcase renders the configured empty state.
    assert "챗봇 모델이 설정되지 않아" in body
    assert "data-adk-chat-frame" not in body
    # No approval toast panels.
    assert "data-order-draft-panel" not in body
    assert "data-config-draft-panel" not in body
    # Showcase sidebar with chat first.
    assert "투자챗봇" in body


def test_showcase_entry_redirects_to_investment_chat(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_SHOWCASE_TENANT", "midnightnnn")
    client = _client(monkeypatch)
    resp = client.get("/showcase/midnightnnn")
    assert resp.status_code == 302
    assert resp.headers["location"].endswith("/showcase/midnightnnn/investment-chat")


def test_showcase_root_redirects_to_investment_chat(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_SHOWCASE_TENANT", "midnightnnn")
    client = _client(monkeypatch)
    resp = client.get("/showcase")
    assert resp.status_code == 302
    assert resp.headers["location"].endswith("/showcase/midnightnnn/investment-chat")
