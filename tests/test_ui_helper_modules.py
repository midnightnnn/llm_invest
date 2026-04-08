from __future__ import annotations

import json
import pytest

from arena.ui.app_support import fmt_ts
from arena.ui.routes.settings_render import CredentialsPanelParts, build_credentials_panel
from arena.ui.routes.settings_render_capital import build_capital_panel
from arena.ui.run_status import build_run_status_helpers
from arena.ui.templating import render_ui_template
from arena.ui.runtime import _parse_json_object, _safe_float
from arena.ui.viewer_analytics import chained_index, drawdown, max_drawdown, metric_card, render_pnl_badge, total_return


class _RunStatusRepo:
    dataset_fqn = "proj.ds"

    def latest_tenant_run_status(self, *, tenant_id: str, exclude_statuses=None):
        _ = tenant_id, exclude_statuses
        return {
            "status": "blocked",
            "reason_code": "reconciliation_failed",
            "stage": "reconcile",
            "message": "",
            "detail_json": json.dumps({"attempt": 1}),
            "run_type": "pipeline",
        }

    def latest_reconciliation_run(self, *, tenant_id: str):
        _ = tenant_id
        return {"run_id": "run-1"}

    def fetch_rows(self, sql: str, params=None):
        _ = sql, params
        return [
            {
                "issue_type": "position_quantity_mismatch",
                "entity_key": "aapl",
                "expected_json": json.dumps({"ledger_quantity": 3}),
                "actual_json": json.dumps({"broker_quantity": 2}),
                "detail_json": "{}",
            }
        ]


def test_settings_render_facade_exports_credentials_builder() -> None:
    parts = build_credentials_panel(
        tenant="local",
        credentials_mode_note="",
        active_kis_account_no="",
        active_kis_account_no_masked="",
        kis_meta=[],
        allow_real_kis_credentials=True,
        allow_paper_kis_credentials=True,
        uses_broker_credentials=True,
        rows_html="",
    )

    assert isinstance(parts, CredentialsPanelParts)
    assert "kis-accounts-form" in parts.kis_section_html
    assert "Add Account" in parts.kis_section_html


def test_run_status_helpers_summarize_reconciliation_issues() -> None:
    helpers = build_run_status_helpers(
        repo=_RunStatusRepo(),
        cached_fetch=lambda _key, fn, tenant: fn(tenant),
        parse_json_object=_parse_json_object,
        safe_float=_safe_float,
        run_status_meta={
            "warning": {"label": "Warning", "wrap": "wrap-warning", "badge": "badge-warning", "text": "text-warning"},
            "blocked": {"label": "Blocked", "wrap": "wrap-blocked", "badge": "badge-blocked", "text": "text-blocked"},
        },
        run_reason_labels={"reconciliation_failed": "Reconciliation failed"},
        run_stage_labels={"reconcile": "Reconcile"},
    )

    payload = helpers.latest_tenant_status_payload("local")

    assert payload is not None
    assert payload["status"] == "blocked"
    assert payload["issues"] == ["AAPL: AI 3주, 실계좌 2주"]
    assert helpers.header_status_kwargs("local") == {
        "status_label": "Blocked",
        "status_color": "rose",
    }


def test_viewer_analytics_chain_and_drawdown_helpers() -> None:
    idx = chained_index(
        [100.0, 120.0, 220.0],
        [0.0, 20.0, 20.0],
        [0.0, 0.2, 0.1],
    )
    dd = drawdown([100.0, 120.0, 90.0])
    card = metric_card("PnL", "+10%", "since inception", value_id="pnl-value", note_id="pnl-note")

    assert idx == pytest.approx([100.0, 120.0, 132.0])
    assert round(total_return(idx), 4) == 0.32
    assert dd == [0.0, 0.0, -0.25]
    assert max_drawdown(dd) == -0.25
    assert max_drawdown([100.0, 120.0, 90.0]) == -0.25
    assert 'id="pnl-value"' in card
    assert 'id="pnl-note"' in card


def test_viewer_analytics_chained_index_neutralizes_large_capital_changes() -> None:
    idx = chained_index(
        [100.0, 90.0, 190.0, 199.0],
        [0.0, -10.0, -10.0, -1.0],
        [0.0, -0.1, -0.05, -0.005],
    )

    assert idx == pytest.approx([100.0, 90.0, 85.5, 89.55])
    assert round(total_return(idx), 4) == -0.1045


def test_render_pnl_badge_ignores_twr_suffix() -> None:
    html = render_pnl_badge(
        pnl_krw=-10_000.0,
        pnl_pct=-0.5,
        chained_stats={"return_ratio": -0.01},
    )

    assert "TWR" not in html
    assert "-0.50%" in html


def test_total_return_reads_latest_actual_return_index() -> None:
    assert round(total_return([98.0, 99.5, 101.25]), 4) == 0.0125
    assert round(total_return([98.0, 99.0]), 4) == -0.01


def test_fmt_ts_converts_iso_string_to_kst() -> None:
    assert fmt_ts("2026-03-27T19:16:36Z") == "2026-03-28 04:16 KST"


def test_base_layout_template_renders_shell_controls() -> None:
    html = render_ui_template(
        "base_layout.jinja2",
        title="Overview",
        body_html="<div>body</div>",
        active="overview",
        needs_charts=False,
        needs_datepicker=False,
        header_extra="",
        max_width_class="max-w-7xl",
        nav_links=[
            {"href": "/", "label": "Overview", "active": True},
            {"href": "/board", "label": "Board", "active": False},
        ],
        auth_enabled=True,
        status_display="Operational",
        status_ping_color="bg-emerald-400",
        status_dot_color="bg-emerald-500",
        status_text_color="text-emerald-600",
    )

    assert "LLM INVEST" in html
    assert "/auth/logout" in html
    assert "sidebar-link" in html
    assert "<div>body</div>" in html


def test_page_templates_render_expected_sections() -> None:
    auth_html = render_ui_template(
        "auth_notice.jinja2",
        eyebrow="Access Pending",
        eyebrow_text_class="text-amber-500",
        border_class="border-amber-200",
        title="승인 대기 중입니다",
        paragraphs=[{"classes": "mt-4 text-sm", "html": "로그인 완료"}],
        note="검토 중",
        actions=[{"href": "/auth/logout", "label": "Logout", "classes": "btn"}],
    )
    trades_html = render_ui_template(
        "trades_body.jinja2",
        auth_enabled=True,
        tenant="local",
        agent_options=[{"value": "gpt", "label": "gpt", "selected": True}],
        ticker="AAPL",
        days=7,
        limit=20,
        trade_rows=[
            {
                "created_at_label": "2026-03-22 10:00",
                "agent_id": "gpt",
                "ticker": "AAPL",
                "side": "BUY",
                "status": "FILLED",
                "requested_qty": "10",
                "filled_qty": "10",
                "avg_price_krw": "100,000",
                "message": "ok",
            }
        ],
        page=1,
        prev_url="/trades?page=1",
        next_url="/trades?page=2",
        prev_disabled=True,
        next_disabled=False,
    )
    board_html = render_ui_template(
        "board_body.jinja2",
        posts=[
            {
                "agent_id": "gpt",
                "ts_iso": "2026-03-22T10:00:00+09:00",
                "cycle_id": "cycle-1",
                "created_at_label": "2026-03-22 10:00",
                "title": "Board Title",
                "body_html": "<div data-md='1'>body</div>",
            }
        ],
        page=1,
        prev_url="/board?page=1",
        next_url="/board?page=2",
        prev_disabled=True,
        next_disabled=False,
        tool_accordion_js="",
        datepicker_js="",
    )
    sleeves_html = render_ui_template(
        "sleeves_body.jinja2",
        is_live=True,
        cards_html="<div>card</div>",
        charts_html="<script>chart</script>",
        sleeve_rows=[
            {
                "agent_id": "gpt",
                "initialized_at_label": "2026-03-22",
                "sleeve_capital": "1,000,000",
                "initial_positions": 2,
            }
        ],
    )

    assert "Access Pending" in auth_html
    assert "Trade History" in trades_html
    assert "Board Title" in board_html
    assert "Initial Sleeve Config" in sleeves_html


def test_build_capital_panel_focuses_on_agent_lineage() -> None:
    html = build_capital_panel(
        tenant="local",
        agent_ids=["gpt", "gemini"],
        sleeve_capital_krw=1_000_000,
        agent_capitals={"gpt": 1_200_000, "gemini": 800_000},
        user_email="tester@example.com",
        is_live=True,
    )

    assert "에이전트별 장부 계보" in html
    assert "capitalLineageGraph" in html
    assert "capitalLineageSummary" in html
    assert "capitalEventLog" in html
    assert "capital-agent-tab" in html
    assert "Target Capital" in html
    assert "Performance 탭은 TWR 기준" in html
    assert "현재 sleeve 배분" not in html
    assert "capitalSankey" not in html
    assert "capitalReconGauge" not in html
