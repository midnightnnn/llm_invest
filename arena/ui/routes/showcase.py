"""Showcase routes — read-only, unauthenticated access to a specific tenant.

Every ``/showcase/{tenant}/...`` route mirrors an existing viewer or settings
page but bypasses authentication and injects ``showcase=True`` into the layout
so that navigation links stay within the ``/showcase/`` prefix and all write
operations are disabled on the frontend.

**Design constraints**:
* No existing route handler is modified.
* All data-fetching helpers are reused via the same ``deps`` objects that the
  normal viewer / settings routes use.
* POST endpoints under ``/showcase/`` unconditionally return 403.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from arena.config import Settings
from arena.providers import default_model_for_provider, list_adk_provider_specs
from arena.ui.memory import build_memory_settings_panel
from arena.ui.routes.settings_render import (
    build_agents_panel,
    build_credentials_panel,
    build_mcp_panel,
    build_save_progress_script,
    build_tab_script,
)
from arena.ui.routes.settings_render_capital import build_capital_panel
from arena.ui.templating import render_ui_template

_SHOWCASE_READONLY_RESPONSE = JSONResponse(
    {"error": "showcase_readonly", "message": "쇼케이스 모드에서는 변경할 수 없습니다"},
    status_code=403,
)


@dataclass(frozen=True)
class ShowcaseRouteDeps:
    repo: Any
    executor: Any
    settings: Settings
    settings_enabled: bool
    kst: Any
    cached_fetch: Callable[..., Any]
    tailwind_layout: Callable[..., str]
    html_response: Callable[..., HTMLResponse]
    json_response: Callable[..., JSONResponse]
    get_default_registry: Callable[[str], Any]
    settings_for_tenant: Callable[[str], Settings]
    latest_tenant_status_payload: Callable[[str], dict[str, Any] | None]
    current_admin_view_model: Callable[..., dict[str, Any]]
    scoped_agent_ids_for_tenant: Callable[[str], list[str]]
    is_live_mode: Callable[[Any | None], bool]
    # data fetchers
    fetch_board: Callable[..., list[dict[str, Any]]]
    fetch_tool_events_for_post: Callable[..., dict[str, Any]]
    fetch_prompt_bundle_for_post: Callable[..., dict[str, Any]]
    fetch_theses_for_board_post: Callable[..., dict[str, Any]]
    fetch_nav: Callable[..., list[dict[str, Any]]]
    fetch_token_usage_summary: Callable[..., dict[str, dict[str, int | float]]]
    fetch_trade_count_summary: Callable[..., dict[str, int]]
    fetch_token_usage_daily: Callable[..., list[dict[str, Any]]]
    fetch_trade_count_daily: Callable[..., list[dict[str, Any]]]
    fetch_trades: Callable[..., list[dict[str, Any]]]
    fetch_trades_for_board_post: Callable[..., list[dict[str, Any]]]
    fetch_sleeve_snapshot_cards: Callable[..., dict[str, Any]]
    fetch_sleeves: Callable[..., list[dict[str, Any]]]
    default_benchmark: Callable[[Any | None], str]
    # formatting helpers
    metric_card: Callable[..., str]
    fmt_ts: Callable[[object], str]
    md_block: Callable[..., str]
    safe_float: Callable[[object, float], float]
    safe_int: Callable[[object, int], int]
    to_date: Callable[[object], str]
    chained_index: Callable[..., list[float | None]]
    drawdown: Callable[..., list[float | None]]
    total_return: Callable[..., float]
    max_drawdown: Callable[..., float]
    provider_api_key_help_html: Callable[[str], str]


def _allowed_showcase_tenant() -> str:
    return str(os.getenv("ARENA_SHOWCASE_TENANT", "")).strip().lower()


def _validate_tenant(tenant: str) -> str | None:
    """Return sanitised tenant id or None if not allowed."""
    allowed = _allowed_showcase_tenant()
    if not allowed:
        return None
    sanitised = str(tenant or "").strip().lower()
    if sanitised != allowed:
        return None
    return sanitised


def _showcase_layout(deps: ShowcaseRouteDeps, title: str, body: str, *, active: str, tenant: str, needs_charts: bool = False, needs_echarts: bool = False, needs_datepicker: bool = False) -> str:
    return deps.tailwind_layout(
        title,
        body,
        active=active,
        needs_charts=needs_charts,
        needs_echarts=needs_echarts,
        needs_datepicker=needs_datepicker,
        tenant=tenant,
        showcase=True,
    )


def _post_ts_iso(row: dict[str, object]) -> str:
    value = row.get("created_at")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value or "")


def register_showcase_routes(app: FastAPI, *, deps: ShowcaseRouteDeps) -> None:
    # ── Redirect /showcase → /showcase/{tenant}/board ──
    @app.get("/showcase", response_class=HTMLResponse)
    def showcase_index(request: Request) -> HTMLResponse:
        tenant = _allowed_showcase_tenant()
        if not tenant:
            return HTMLResponse("Showcase not configured", status_code=404)
        return HTMLResponse(status_code=302, headers={"Location": f"/showcase/{tenant}/board"})

    @app.get("/showcase/{tenant}", response_class=HTMLResponse)
    def showcase_tenant_index(tenant: str) -> HTMLResponse:
        t = _validate_tenant(tenant)
        if not t:
            return HTMLResponse("Not found", status_code=404)
        return HTMLResponse(status_code=302, headers={"Location": f"/showcase/{t}/board"})

    # ── Board ──
    @app.get("/showcase/{tenant}/board", response_class=HTMLResponse)
    def showcase_board(
        request: Request,
        tenant: str,
        agent_id: str = Query(default=""),
        date: str = Query(default=""),
        limit: int = Query(default=20, ge=1, le=400),
        page: int = Query(default=1, ge=1),
    ) -> HTMLResponse:
        t = _validate_tenant(tenant)
        if not t:
            return HTMLResponse("Not found", status_code=404)
        scoped_agent_ids = deps.scoped_agent_ids_for_tenant(t)
        token_agent = agent_id.strip().lower() or None
        if token_agent and token_agent not in scoped_agent_ids:
            token_agent = None
        token_date = date.strip()[:10] if date.strip() else None
        offset = (page - 1) * limit
        agent_key = ",".join(sorted(scoped_agent_ids))

        rows = deps.cached_fetch(
            f"board:{t}:{agent_key}:{limit}:{offset}:{token_agent or ''}:{token_date or ''}",
            deps.fetch_board,
            tenant_id=t,
            limit=limit,
            offset=offset,
            agent_ids=scoped_agent_ids,
            agent_id=token_agent,
            start_date=token_date,
            end_date=token_date,
        )

        posts = []
        for row in rows:
            posts.append({
                "created_at_label": deps.fmt_ts(row.get("created_at")),
                "ts_iso": _post_ts_iso(row),
                "agent_id": str(row.get("agent_id") or ""),
                "title": str(row.get("title") or ""),
                "body_html": deps.md_block(str(row.get("body") or ""), classes="mt-2 text-sm leading-relaxed text-ink-700"),
                "cycle_id": str(row.get("cycle_id") or ""),
            })

        qs_base = {k: v for k, v in {"agent_id": agent_id.strip(), "date": token_date or "", "limit": str(limit)}.items() if v}
        prev_qs = urlencode({**qs_base, "page": str(max(1, page - 1))})
        next_qs = urlencode({**qs_base, "page": str(page + 1)})
        has_next = len(rows) == limit

        body = render_ui_template(
            "board_body.jinja2",
            posts=posts,
            page=page,
            prev_url=f"/showcase/{t}/board?{prev_qs}" if page > 1 else "",
            next_url=f"/showcase/{t}/board?{next_qs}" if has_next else "",
            prev_disabled=page <= 1,
            next_disabled=not has_next,
            empty_state_message="게시글이 없습니다.",
            show_detail_buttons=False,
            tool_accordion_js="",
            datepicker_js="",
        )
        return deps.html_response(_showcase_layout(deps, "게시판", body, active="board", tenant=t, needs_datepicker=True), max_age=60)

    # ── NAV (운용성과) ──
    @app.get("/showcase/{tenant}/nav", response_class=HTMLResponse)
    def showcase_nav(
        request: Request,
        tenant: str,
        agent_id: str = Query(default=""),
        days: int = Query(default=180, ge=5, le=2500),
        bench: str = Query(default=""),
    ) -> HTMLResponse:
        t = _validate_tenant(tenant)
        if not t:
            return HTMLResponse("Not found", status_code=404)
        scoped_agent_ids = deps.scoped_agent_ids_for_tenant(t)
        tenant_settings = deps.settings_for_tenant(t)
        bench_ticker = (bench.strip() or deps.default_benchmark(tenant_settings)).upper()
        _ = bench_ticker

        chart_params = urlencode({k: v for k, v in {"tenant_id": t, "agent_id": agent_id.strip(), "days": str(days)}.items() if v})
        chart_api_url = f"/api/nav/chart?{chart_params}"
        tool_freq_url = f"/api/tool-frequency?tenant_id={t}"

        try:
            nav_registry = deps.get_default_registry(t)
            nav_tool_label_map = {e.tool_id: e.label_ko or e.tool_id for e in nav_registry.list_entries(include_disabled=True)}
        except Exception:
            nav_tool_label_map = {}

        is_live = deps.is_live_mode(tenant_settings)
        body = render_ui_template(
            "nav_body.jinja2",
            auth_enabled=True,
            tenant=t,
            agent_options=[{"value": a, "label": a, "selected": a == agent_id} for a in scoped_agent_ids],
            days=days,
            is_live=is_live,
            chart_api_url=chart_api_url,
            tool_freq_url=tool_freq_url,
            nav_tool_label_map=nav_tool_label_map,
        )
        return deps.html_response(_showcase_layout(deps, "운용성과", body, active="nav", tenant=t, needs_charts=True), max_age=60)

    # ── Trades ──
    @app.get("/showcase/{tenant}/trades", response_class=HTMLResponse)
    def showcase_trades(
        request: Request,
        tenant: str,
        agent_id: str = Query(default=""),
        ticker: str = Query(default=""),
        days: int = Query(default=7, ge=1, le=90),
        limit: int = Query(default=20, ge=1, le=1200),
        page: int = Query(default=1, ge=1),
    ) -> HTMLResponse:
        t = _validate_tenant(tenant)
        if not t:
            return HTMLResponse("Not found", status_code=404)
        scoped_agent_ids = deps.scoped_agent_ids_for_tenant(t)
        token_agent = agent_id.strip().lower() or None
        if token_agent and token_agent not in scoped_agent_ids:
            token_agent = None
        token_ticker = ticker.strip().upper() or None
        offset = (page - 1) * limit
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(
            f"trades:{t}:{agent_key}:{limit}:{offset}:{days}:{token_agent or ''}:{token_ticker or ''}",
            deps.fetch_trades,
            tenant_id=t,
            limit=limit,
            offset=offset,
            days=days,
            agent_ids=scoped_agent_ids,
            agent_id=token_agent,
            ticker=token_ticker,
        )

        tenant_settings = deps.settings_for_tenant(t)
        is_live = deps.is_live_mode(tenant_settings)

        trade_items = [
            {
                "created_at_label": deps.fmt_ts(row.get("created_at")),
                "agent_id": str(row.get("agent_id") or ""),
                "ticker": str(row.get("ticker") or ""),
                "side": str(row.get("side") or ""),
                "intent_qty": deps.safe_int(row.get("intent_qty"), 0),
                "filled_qty": deps.safe_int(row.get("filled_qty"), 0),
                "avg_price_krw": deps.safe_float(row.get("avg_price_krw"), 0.0),
                "status": str(row.get("status") or ""),
                "reject_reason": str(row.get("reject_reason") or ""),
            }
            for row in rows
        ]

        qs_base = {k: v for k, v in {"agent_id": agent_id.strip(), "ticker": ticker.strip(), "days": str(days), "limit": str(limit)}.items() if v}
        prev_qs = urlencode({**qs_base, "page": str(max(1, page - 1))})
        next_qs = urlencode({**qs_base, "page": str(page + 1)})
        has_next = len(rows) >= limit

        body = render_ui_template(
            "trades_body.jinja2",
            auth_enabled=True,
            tenant=t,
            trade_items=trade_items,
            agent_options=[{"value": a, "label": a, "selected": a == agent_id} for a in scoped_agent_ids],
            days=days,
            is_live=is_live,
            page=page,
            prev_url=f"/showcase/{t}/trades?{prev_qs}" if page > 1 else "",
            next_url=f"/showcase/{t}/trades?{next_qs}" if has_next else "",
        )
        return deps.html_response(_showcase_layout(deps, "매매내역", body, active="trades", tenant=t), max_age=60)

    # ── Sleeves ──
    @app.get("/showcase/{tenant}/sleeves", response_class=HTMLResponse)
    def showcase_sleeves(
        request: Request,
        tenant: str,
    ) -> HTMLResponse:
        t = _validate_tenant(tenant)
        if not t:
            return HTMLResponse("Not found", status_code=404)
        scoped_agent_ids = deps.scoped_agent_ids_for_tenant(t)
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(f"sleeves:{t}:{agent_key}", deps.fetch_sleeves, tenant_id=t, agent_ids=scoped_agent_ids)
        snapshot_payload = deps.cached_fetch(
            f"sleeve_cards:{t}:{agent_key}",
            deps.fetch_sleeve_snapshot_cards,
            tenant_id=t,
            agent_ids=scoped_agent_ids,
        )
        tenant_settings = deps.settings_for_tenant(t)
        is_live = deps.is_live_mode(tenant_settings)

        def _sleeve_capital(row: dict) -> float:
            return float(row.get("sleeve_capital_krw") or row.get("capital_krw") or 0)

        def _positions_len(val: Any) -> int:
            if not val:
                return 0
            if isinstance(val, list):
                return len(val)
            try:
                return len(json.loads(str(val)))
            except Exception:
                return 0

        sleeve_rows = [
            {
                "agent_id": str(row.get("agent_id") or ""),
                "initialized_at_label": deps.fmt_ts(row.get("initialized_at")),
                "sleeve_capital": f"{_sleeve_capital(row):,.0f}",
                "initial_positions": _positions_len(row.get("initial_positions_json")),
            }
            for row in rows
        ]

        cards_html = str((snapshot_payload or {}).get("html") or "")
        charts = (snapshot_payload or {}).get("charts")
        charts_html = ""
        if charts:
            specs = [dict(s) for s in charts if isinstance(s, dict) and s.get("id")]
            if specs:
                payload = json.dumps(specs, ensure_ascii=False).replace("<", "\\u003c")
                charts_html = (
                    "<script>"
                    'window.addEventListener("load",function(){'
                    f"const specs={payload};"
                    "const palette=['#cbd5e1','#f43f5e','#3b82f6','#8b5cf6','#10b981','#f59e0b','#ec4899','#14b8a6','#6366f1','#f97316'];"
                    "specs.forEach(function(spec){"
                    "const node=document.getElementById(String(spec.id||''));"
                    "if(!node){return;}"
                    "new Chart(node,{type:'doughnut',data:{labels:spec.labels||[],datasets:[{data:spec.data||[],backgroundColor:palette,borderWidth:0,hoverOffset:4}]},options:{responsive:true,maintainAspectRatio:true,cutout:'75%',plugins:{legend:{display:false},tooltip:{callbacks:{label:function(c){return ' ' + c.label + ': ' + (c.raw||0).toLocaleString() + ' 원';}}}}}});"
                    "});"
                    "});"
                    "</script>"
                )

        body = render_ui_template(
            "sleeves_body.jinja2",
            is_live=is_live,
            cards_html=cards_html,
            charts_html=charts_html,
            sleeve_rows=sleeve_rows,
        )
        return deps.html_response(_showcase_layout(deps, "Sleeves", body, active="sleeves", tenant=t, needs_charts=True), max_age=60)

    # ── Settings (read-only) ──
    @app.get("/showcase/{tenant}/settings", response_class=HTMLResponse)
    def showcase_settings(
        request: Request,
        tenant: str,
        tab: str = Query(default="agents"),
    ) -> HTMLResponse:
        t = _validate_tenant(tenant)
        if not t:
            return HTMLResponse("Not found", status_code=404)

        initial_tab = str(tab or "agents").strip().lower()
        if initial_tab == "credentials":
            initial_tab = "agents"
        allowed_tabs = {"agents", "mcp", "memory", "capital"}
        if initial_tab not in allowed_tabs:
            initial_tab = "agents"

        admin_vm = deps.current_admin_view_model(t)
        tool_entries = admin_vm["tool_entries"]
        risk = admin_vm["risk"]
        prompt_text = str(admin_vm["prompt_text"] or "")
        tenant_settings = admin_vm.get("tenant_settings", deps.settings)
        distribution_mode = str(getattr(tenant_settings, "distribution_mode", "private") or "private").strip().lower()

        credentials_parts = build_credentials_panel(
            tenant=t,
            credentials_mode_note="",
            active_kis_account_no="",
            active_kis_account_no_masked="",
            kis_meta=[],
            allow_real_kis_credentials=False,
            allow_paper_kis_credentials=False,
            uses_broker_credentials=False,
            rows_html="",
        )
        credentials_panel = credentials_parts.panel_html

        agents_cfg = admin_vm["agents_config"]
        api_status = admin_vm.get("api_key_status", {})
        provider_specs = list_adk_provider_specs()
        provider_options_html = "".join(
            f'<option value="{spec.provider_id}">{spec.provider_id}</option>'
            for spec in provider_specs
        )
        provider_key_help = {
            spec.provider_id: deps.provider_api_key_help_html(spec.provider_id)
            for spec in provider_specs
        }
        default_models = {
            spec.provider_id: default_model_for_provider(tenant_settings, spec.provider_id)
            for spec in provider_specs
        }
        configurable_tools = [
            {
                "tool_id": str(e.get("tool_id") or ""),
                "category": str(e.get("category") or ""),
                "tier": str(e.get("tier") or ""),
                "description": str(e.get("description") or ""),
                "description_ko": str(e.get("description_ko") or e.get("description") or ""),
                "label_ko": str(e.get("label_ko") or e.get("tool_id") or ""),
                "params": e.get("params") or [],
                "source": e.get("source") or "",
            }
            for e in tool_entries if e.get("configurable")
        ]
        risk_fields_meta = [
            ("max_order_krw", "최대 주문 금액", risk["max_order_krw"]),
            ("max_daily_turnover_ratio", "일일 최대 회전율", risk["max_daily_turnover_ratio"]),
            ("max_position_ratio", "최대 종목 비중", risk["max_position_ratio"]),
            ("min_cash_buffer_ratio", "최소 현금 비율", risk["min_cash_buffer_ratio"]),
            ("ticker_cooldown_seconds", "종목 쿨다운", risk["ticker_cooldown_seconds"]),
            ("max_daily_orders", "일일 최대 주문 수", risk["max_daily_orders"]),
            ("estimated_fee_bps", "예상 수수료 (bps)", risk["estimated_fee_bps"]),
        ]

        agents_panel = build_agents_panel(
            agents_cfg=agents_cfg,
            api_status=api_status,
            research_status=admin_vm.get("research_status", {}),
            tenant_settings=tenant_settings,
            prompt_text=prompt_text,
            provider_options_html=provider_options_html,
            provider_key_help=provider_key_help,
            default_models=default_models,
            configurable_tools=configurable_tools,
            risk_fields_meta=risk_fields_meta,
            tenant=t,
            user_email="showcase@readonly",
            provider_api_key_help_html=deps.provider_api_key_help_html,
            is_live_mode=deps.is_live_mode,
            default_capital_krw=int(admin_vm["sleeve_capital_krw"]),
        )

        mcp_panel = build_mcp_panel(
            tenant=t,
            mcp_servers=admin_vm["mcp_servers"],
            agents_cfg=agents_cfg,
            configurable_tools=configurable_tools,
        )
        memory_panel = build_memory_settings_panel(
            deps.repo,
            deps.settings,
            tenant_id=t,
            cached_fetch=deps.cached_fetch,
        )

        capital_agent_ids = [
            str(a.get("agent_id") or "").strip().lower()
            for a in agents_cfg
            if str(a.get("agent_id") or "").strip()
        ] or [str(a).strip().lower() for a in tenant_settings.agent_ids if str(a).strip()]
        capital_per_agent = {
            str(a.get("agent_id") or a.get("id") or "").strip().lower(): int(a.get("capital_krw") or admin_vm["sleeve_capital_krw"])
            for a in agents_cfg
            if str(a.get("agent_id") or a.get("id") or "").strip()
        }
        capital_panel = build_capital_panel(
            tenant=t,
            agent_ids=capital_agent_ids,
            sleeve_capital_krw=int(admin_vm["sleeve_capital_krw"]),
            agent_capitals=capital_per_agent,
            user_email="showcase@readonly",
            is_live=deps.is_live_mode(tenant_settings),
        )

        tab_script = build_tab_script(
            initial_tab=initial_tab,
            uses_broker_credentials=False,
            kis_env_options_html="",
            kis_template_real_fields="",
            kis_template_paper_fields="",
        )
        save_progress_script = build_save_progress_script()

        _TAB_TO_ACTIVE = {"agents": "agents", "capital": "capital", "mcp": "tools", "memory": "memory"}
        _TAB_TO_TITLE = {"agents": "에이전트", "capital": "자본관리", "mcp": "도구관리", "memory": "기억관리"}
        sidebar_active = _TAB_TO_ACTIVE.get(initial_tab, "agents")
        page_title = _TAB_TO_TITLE.get(initial_tab, "에이전트")

        body = render_ui_template(
            "settings_body.jinja2",
            popup_text="",
            popup_level="success",
            credential_store_error="",
            auth_enabled=True,
            tenant=t,
            initial_tab=initial_tab,
            credentials_panel=credentials_panel,
            agents_panel=agents_panel,
            capital_panel=capital_panel,
            mcp_panel=mcp_panel,
            memory_panel=memory_panel,
            tab_script=tab_script,
            save_progress_script=save_progress_script,
            showcase=True,
        )
        # Rewrite /settings tab links to showcase prefix
        _prefix = f"/showcase/{t}"
        body = body.replace('"/settings', f'"{_prefix}/settings')
        body = body.replace("'/settings", f"'{_prefix}/settings")

        return deps.html_response(
            _showcase_layout(deps, page_title, body, active=sidebar_active, tenant=t, needs_charts=True, needs_echarts=True),
            max_age=60,
        )

    # ── Catch-all POST → 403 ──
    @app.post("/showcase/{tenant}/{path:path}")
    def showcase_block_post(tenant: str, path: str) -> JSONResponse:
        return _SHOWCASE_READONLY_RESPONSE
