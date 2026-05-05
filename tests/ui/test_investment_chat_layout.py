from __future__ import annotations

from arena.ui.layout import tailwind_layout


def test_default_layout_places_investment_chat_under_memory_nav() -> None:
    html = tailwind_layout("Board", "<div>body</div>", active="investment_chat")

    assert "/investment-chat" in html
    assert "투자챗봇" in html
    assert 'href="/investment-chat" class="sidebar-link active"' in html
    assert html.index("기억관리") < html.index("투자챗봇")
    assert "bottom_nav_links" not in html


def test_layout_preserves_tenant_in_investment_chat_nav() -> None:
    html = tailwind_layout("Board", "<div>body</div>", active="board", tenant="MidNightNnN")

    assert 'href="/investment-chat?tenant_id=midnightnnn"' in html
