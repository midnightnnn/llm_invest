from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from arena.config import research_generation_status
from arena.ui.templating import render_ui_template
from arena.ui.routes.viewer import ViewerRouteDeps


def register_board_routes(app: FastAPI, *, deps: ViewerRouteDeps) -> None:
    @app.get("/board", response_class=HTMLResponse)
    def board(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent_id filter"),
        date: str = Query(default="", description="date filter YYYY-MM-DD"),
        limit: int = Query(default=20, ge=1, le=400),
        page: int = Query(default=1, ge=1),
    ) -> HTMLResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/board?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        token = agent_id.strip().lower() or None
        if token and token not in scoped_agent_ids:
            token = None
        offset = (page - 1) * limit
        agent_key = ",".join(sorted(scoped_agent_ids))
        date_filter = date.strip()[:10] if date.strip() else datetime.now(deps.kst).strftime("%Y-%m-%d")
        fut_board_rows = deps.executor.submit(
            deps.cached_fetch,
            f"board:{tenant}:{agent_key}:{limit}:{offset}:{token or ''}:{date_filter or ''}",
            deps.fetch_board,
            tenant_id=tenant,
            limit=limit,
            offset=offset,
            agent_id=token,
            agent_ids=scoped_agent_ids,
            start_date=date_filter,
            end_date=date_filter,
        )
        fut_board_registry = deps.executor.submit(deps.get_default_registry, tenant)
        rows = fut_board_rows.result()
        has_next = len(rows) == limit
        selected_agent_id = token or ""
        research_status = research_generation_status(deps.settings_for_tenant(tenant))
        empty_state_message = "게시글이 없습니다."
        if not rows:
            code = str(research_status.get("code") or "").strip().lower()
            if code == "missing_gemini_key":
                empty_state_message = (
                    "아직 게시글이 없습니다. 이 테넌트는 Gemini 키가 없어 "
                    "새로운 리서치 브리핑 생성도 비활성화되어 있습니다."
                )
            elif code == "disabled_by_config":
                empty_state_message = (
                    "아직 게시글이 없습니다. 이 테넌트는 설정상 리서치 브리핑 생성을 꺼둔 상태입니다."
                )

        def _post_ts_iso(row: dict[str, object]) -> str:
            value = row.get("created_at")
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                return value.isoformat()
            return str(value or "")

        posts = [
            {
                "post_id": str(row.get("post_id") or ""),
                "agent_id": str(row.get("agent_id") or ""),
                "ts_iso": _post_ts_iso(row),
                "cycle_id": str(row.get("cycle_id") or ""),
                "created_at_label": deps.fmt_ts(row.get("created_at")),
                "title": str(row.get("title") or ""),
                "body_html": deps.md_block(row.get("body"), classes="mt-3 text-sm leading-relaxed text-ink-700"),
            }
            for row in rows
        ]

        prev_page = page - 1 if page > 1 else 1
        next_page = page + 1 if has_next else page

        page_params: dict[str, str | int] = {"page": prev_page if page <= 1 else prev_page}
        if date_filter:
            page_params["date"] = date_filter
        if deps.auth_enabled:
            page_params["tenant_id"] = tenant
        if selected_agent_id:
            page_params["agent_id"] = selected_agent_id
        prev_url = "/board?" + urlencode(page_params | {"page": prev_page})
        next_url = "/board?" + urlencode(page_params | {"page": next_page})
        try:
            board_tool_categories = {
                entry.tool_id: entry.category
                for entry in fut_board_registry.result().list_entries(include_disabled=True)
            }
        except Exception:
            board_tool_categories = {}

        tool_accordion_js = render_ui_template(
            "board_tool_accordion_js.jinja2",
            tenant_json=json.dumps(tenant),
            cat_map_json=json.dumps(board_tool_categories, ensure_ascii=False),
        )

        datepicker_js = render_ui_template("board_datepicker_js.jinja2")

        header_datepicker = render_ui_template(
            "board_header_datepicker.jinja2",
            auth_enabled=deps.auth_enabled,
            tenant=tenant,
            date_value=date_filter or "",
            agent_id=selected_agent_id,
        )

        body = render_ui_template(
            "board_body.jinja2",
            posts=posts,
            empty_state_message=empty_state_message,
            page=page,
            prev_url=prev_url,
            next_url=next_url,
            prev_disabled=page <= 1,
            next_disabled=not has_next,
            tool_accordion_js=tool_accordion_js,
            datepicker_js=datepicker_js,
        )
        return deps.html_response(
            deps.tailwind_layout(
                "\uac8c\uc2dc\ud310",
                body,
                active="board",
                needs_datepicker=True,
                header_extra=header_datepicker,
                tenant=tenant,
                user=deps.current_user(request),
            ),
            max_age=30,
        )

    @app.get("/api/board")
    def api_board(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent id"),
        limit: int = Query(default=80, ge=1, le=400),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        token = agent_id.strip().lower() or None
        if token and token not in scoped_agent_ids:
            token = None
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(
            f"board:{tenant}:{agent_key}:{limit}:0:{token or ''}",
            deps.fetch_board,
            tenant_id=tenant,
            limit=limit,
            agent_id=token,
            agent_ids=scoped_agent_ids,
        )
        return deps.json_response(rows, max_age=30)

    @app.get("/api/board/tools")
    def api_board_tools(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(..., description="agent id"),
        ts: str = Query(..., description="ISO timestamp of the board post"),
        cycle_id: str = Query(default="", description="cycle id from board post"),
        post_id: str = Query(default="", description="board post id"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board/tools?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        data = deps.fetch_tool_events_for_post(
            tenant_id=tenant,
            agent_id=agent_id.strip().lower(),
            ts_iso=ts,
            cycle_id=cycle_id.strip() or None,
            post_id=post_id.strip() or None,
        )
        return deps.json_response(data, max_age=60)

    @app.get("/api/board/prompt")
    def api_board_prompt(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(..., description="agent id"),
        ts: str = Query(..., description="ISO timestamp of the board post"),
        cycle_id: str = Query(default="", description="cycle id from board post"),
        post_id: str = Query(default="", description="board post id"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board/prompt?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        data = deps.fetch_prompt_bundle_for_post(
            tenant_id=tenant,
            agent_id=agent_id.strip().lower(),
            ts_iso=ts,
            cycle_id=cycle_id.strip() or None,
            post_id=post_id.strip() or None,
        )
        return deps.json_response(data, max_age=60)

    @app.get("/api/board/theses")
    def api_board_theses(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        cycle_id: str = Query(default="", description="cycle id from board post"),
        agent_id: str = Query(default="", description="agent id for filtering"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board/theses?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        data = deps.fetch_theses_for_board_post(
            tenant_id=tenant,
            cycle_id=cycle_id.strip() or None,
            agent_id=agent_id.strip() or None,
        )
        return deps.json_response(data, max_age=60)

    @app.get("/api/board/trades")
    def api_board_trades(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        cycle_id: str = Query(default="", description="cycle id from board post"),
        agent_id: str = Query(default="", description="agent id for filtering"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/board/trades?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        rows = deps.fetch_trades_for_board_post(
            tenant_id=tenant,
            cycle_id=cycle_id.strip() or None,
            agent_id=agent_id.strip() or None,
        )
        return deps.json_response(rows, max_age=60)
