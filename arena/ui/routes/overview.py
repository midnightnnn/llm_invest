from __future__ import annotations

import json
from datetime import datetime
from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from arena.ui.templating import render_ui_template
from arena.ui.routes.viewer import ViewerRouteDeps


def _doughnut_chart_scripts(chart_specs: list[dict] | None) -> str:
    specs = [dict(s) for s in chart_specs or [] if isinstance(s, dict) and s.get("id")]
    if not specs:
        return ""
    payload = json.dumps(specs, ensure_ascii=False).replace("<", "\\u003c")
    return (
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


def register_overview_routes(app: FastAPI, *, deps: ViewerRouteDeps) -> None:
    @app.get("/", response_class=HTMLResponse)
    def overview(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
    ) -> HTMLResponse:
        # Redirect root to /board (게시판)
        qs = f"?tenant_id={tenant_id}" if tenant_id else ""
        return RedirectResponse(url=f"/board{qs}", status_code=302)

    @app.get("/overview", response_class=HTMLResponse)
    def overview_legacy(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
    ) -> HTMLResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/overview?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect

        agent_key = ",".join(sorted(scoped_agent_ids))
        fut_board = deps.executor.submit(
            deps.cached_fetch,
            f"board:{tenant}:{agent_key}:3:0",
            deps.fetch_board,
            tenant_id=tenant,
            limit=3,
            agent_ids=scoped_agent_ids,
        )
        fut_trades = deps.executor.submit(
            deps.cached_fetch,
            f"trades:{tenant}:{agent_key}:50:2",
            deps.fetch_trades,
            tenant_id=tenant,
            limit=50,
            days=2,
            agent_ids=scoped_agent_ids,
        )
        fut_nav = deps.executor.submit(
            deps.cached_fetch,
            f"nav:{tenant}:{agent_key}:30",
            deps.fetch_nav,
            tenant_id=tenant,
            days=30,
            agent_ids=scoped_agent_ids,
        )
        fut_snapshot = deps.executor.submit(
            deps.cached_fetch,
            f"sleeve_cards:{tenant}:{agent_key}",
            deps.fetch_sleeve_snapshot_cards,
            tenant_id=tenant,
            agent_ids=scoped_agent_ids,
        )
        board_rows = fut_board.result()
        trade_rows = fut_trades.result()
        nav_rows = fut_nav.result()
        snapshot_payload = fut_snapshot.result()

        today = datetime.now(deps.kst).date().isoformat()

        rows_by_agent: dict[str, list[dict[str, object]]] = {}
        for row in nav_rows:
            agent_id = str(row.get("agent_id") or "").strip()
            if not agent_id or deps.safe_float(row.get("nav_krw"), 0.0) <= 0:
                continue
            rows_by_agent.setdefault(agent_id, []).append(row)

        agent_ret: dict[str, float] = {}
        for agent_id, agent_rows in rows_by_agent.items():
            nav_values = [deps.safe_float(row.get("nav_krw"), 0.0) or None for row in agent_rows]
            pnl_krw_values = [row.get("pnl_krw") for row in agent_rows]
            pnl_ratio_values = [row.get("pnl_ratio") for row in agent_rows]
            idx = deps.chained_index(nav_values, pnl_krw_values, pnl_ratio_values)
            agent_ret[agent_id] = deps.total_return(idx)

        top_agent = "-"
        top_ret = 0.0
        if agent_ret:
            ranked = sorted(agent_ret.items(), key=lambda item: item[1], reverse=True)
            top_agent = ranked[0][0]
            top_ret = ranked[0][1]

        rejected = sum(1 for row in trade_rows if str(row.get("status") or "") == "REJECTED")
        chart_params = urlencode({key: value for key, value in {"tenant_id": tenant, "days": "30"}.items() if value})
        chart_api_url = f"/api/nav/chart?{chart_params}"

        snapshot_cards_html = str((snapshot_payload or {}).get("html") or "")
        snapshot_charts_html = _doughnut_chart_scripts((snapshot_payload or {}).get("charts"))

        body = render_ui_template(
            "overview_body.jinja2",
            snapshot_cards_html=snapshot_cards_html,
            snapshot_charts_html=snapshot_charts_html,
            cards=[
                {"title": "Today (UTC)", "value": today, "note": "Inspector snapshot timestamp"},
                {"title": "Recent Board Posts", "value": str(len(board_rows)), "note": "Latest 8 posts loaded"},
                {"title": "Recent Trades", "value": str(len(trade_rows)), "note": f"Rejected: {rejected}"},
                {"title": "Top Agent", "value": top_agent, "note": f"Latest TWR {deps.safe_float(top_ret, 0.0) * 100:+.2f}%"},
            ],
            board_items=[
                {
                    "created_at_label": deps.fmt_ts(row.get("created_at")),
                    "agent_id": str(row.get("agent_id") or ""),
                    "title": str(row.get("title") or ""),
                    "body_html": deps.md_block(str(row.get("body") or "")[:360], classes="mt-2 text-sm leading-relaxed text-ink-700"),
                }
                for row in board_rows[:4]
            ],
            trade_items=[
                {
                    "created_at_label": deps.fmt_ts(row.get("created_at")),
                    "agent_id": str(row.get("agent_id") or ""),
                    "ticker": str(row.get("ticker") or ""),
                    "side": str(row.get("side") or ""),
                    "status": str(row.get("status") or ""),
                }
                for row in trade_rows[:8]
            ],
            chart_api_url=chart_api_url,
        )

        return deps.html_response(
            deps.tailwind_layout(
                "Overview",
                body,
                active="overview",
                needs_charts=True,
                tenant=tenant,
                user=deps.current_user(request),
            ),
            max_age=30,
        )
