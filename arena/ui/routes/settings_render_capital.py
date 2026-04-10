"""Builds the Capital panel HTML for the Settings page.

The panel focuses on per-agent sleeve lineage and ledger history.
"""
from __future__ import annotations

import json

from arena.ui.templating import render_ui_template


_SPINNER = (
    '<div class="flex items-center justify-center h-full text-ink-400">'
    '<span class="inline-block h-4 w-4 animate-spin rounded-full border-2 border-ink-300 border-t-ink-700 mr-2"></span>'
    '<span class="text-sm">Loading...</span></div>'
)

_EMPTY_HTML = '<p class="py-12 text-center text-sm text-ink-400">데이터가 없습니다</p>'
_FAIL_HTML = '<p class="py-12 text-center text-sm text-ink-400">로드 실패</p>'


def build_capital_panel(
    *,
    tenant: str,
    agent_ids: list[str],
    sleeve_capital_krw: int = 0,
    agent_capitals: dict[str, int] | None = None,
    user_email: str = "",
    is_live: bool = False,
) -> str:
    """Return full HTML for the Capital settings panel (hidden by default)."""
    waterfall_url = f"/api/capital/waterfall?tenant_id={tenant}"
    caps = {str(agent_id): int(value) for agent_id, value in (agent_capitals or {}).items()}
    updated_by = user_email or "ui-admin"

    return render_ui_template(
        "settings_capital_panel.jinja2",
        tenant=tenant,
        agent_ids=list(agent_ids),
        sleeve_capital_krw=int(sleeve_capital_krw),
        updated_by=updated_by,
        is_live=bool(is_live),
        spinner_html=_SPINNER,
        waterfall_url_json=json.dumps(waterfall_url),
        agent_ids_json=json.dumps(list(agent_ids)),
        agent_capitals_json=json.dumps(caps),
        sleeve_capital_krw_json=json.dumps(int(sleeve_capital_krw)),
        spinner_json=json.dumps(_SPINNER),
        empty_html_json=json.dumps(_EMPTY_HTML),
        fail_html_json=json.dumps(_FAIL_HTML),
    )
