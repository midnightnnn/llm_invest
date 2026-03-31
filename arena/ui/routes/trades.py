from __future__ import annotations

from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from arena.ui.templating import render_ui_template
from arena.ui.routes.viewer import ViewerRouteDeps


def register_trades_routes(app: FastAPI, *, deps: ViewerRouteDeps) -> None:
    @app.get("/trades", response_class=HTMLResponse)
    def trades(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent_id filter"),
        ticker: str = Query(default="", description="ticker filter"),
        days: int = Query(default=7, ge=1, le=90),
        limit: int = Query(default=20, ge=1, le=1200),
        page: int = Query(default=1, ge=1),
    ) -> HTMLResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/trades?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        token_agent = agent_id.strip().lower() or None
        if token_agent and token_agent not in scoped_agent_ids:
            token_agent = None
        token_ticker = ticker.strip().upper() or None
        offset = (page - 1) * limit
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(
            f"trades:{tenant}:{agent_key}:{limit}:{offset}:{days}:{token_agent or ''}:{token_ticker or ''}",
            deps.fetch_trades,
            tenant_id=tenant,
            limit=limit,
            days=days,
            offset=offset,
            agent_id=token_agent,
            ticker=token_ticker,
            agent_ids=scoped_agent_ids,
        )
        has_next = len(rows) == limit

        agent_options = [
            {
                "value": agent,
                "label": agent,
                "selected": agent == agent_id,
            }
            for agent in scoped_agent_ids
        ]

        trade_rows = [
            {
                "created_at_label": deps.fmt_ts(row.get("created_at")),
                "agent_id": str(row.get("agent_id") or ""),
                "ticker": str(row.get("ticker") or ""),
                "side": str(row.get("side") or ""),
                "status": str(row.get("status") or ""),
                "requested_qty": f"{deps.safe_float(row.get('requested_qty'), 0.0):,.0f}",
                "filled_qty": f"{deps.safe_float(row.get('filled_qty'), 0.0):,.0f}",
                "avg_price_krw": f"{deps.safe_float(row.get('avg_price_krw'), 0.0):,.0f}",
                "message": str(row.get("message") or "")[:180],
            }
            for row in rows
        ]

        prev_page = page - 1 if page > 1 else 1
        next_page = page + 1 if has_next else page
        base_params = {
            "agent_id": agent_id,
            "ticker": ticker,
            "days": days,
            "limit": limit,
        }
        if deps.auth_enabled:
            base_params["tenant_id"] = tenant
        prev_url = "/trades?" + urlencode(base_params | {"page": prev_page})
        next_url = "/trades?" + urlencode(base_params | {"page": next_page})

        body = render_ui_template(
            "trades_body.jinja2",
            auth_enabled=deps.auth_enabled,
            tenant=tenant,
            agent_options=agent_options,
            ticker=ticker,
            days=days,
            limit=limit,
            trade_rows=trade_rows,
            page=page,
            prev_url=prev_url,
            next_url=next_url,
            prev_disabled=page <= 1,
            next_disabled=not has_next,
        )

        return deps.html_response(deps.tailwind_layout("Trades", body, active="trades", tenant=tenant), max_age=30)

    @app.get("/api/trades")
    def api_trades(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent id"),
        ticker: str = Query(default="", description="ticker filter"),
        days: int = Query(default=2, ge=1, le=30),
        limit: int = Query(default=200, ge=1, le=1200),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/trades?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        token_agent = agent_id.strip().lower() or None
        if token_agent and token_agent not in scoped_agent_ids:
            token_agent = None
        token_ticker = ticker.strip().upper() or None
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(
            f"trades:{tenant}:{agent_key}:{limit}:{days}:{token_agent or ''}:{token_ticker or ''}",
            deps.fetch_trades,
            tenant_id=tenant,
            limit=limit,
            days=days,
            agent_id=token_agent,
            ticker=token_ticker,
            agent_ids=scoped_agent_ids,
        )
        return deps.json_response(rows, max_age=30)
