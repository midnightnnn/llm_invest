from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from arena.ui.templating import render_ui_template
from arena.ui.routes.viewer import ViewerRouteDeps


def register_nav_routes(app: FastAPI, *, deps: ViewerRouteDeps) -> None:
    @app.get("/nav", response_class=HTMLResponse)
    def nav(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent_id filter"),
        days: int = Query(default=180, ge=5, le=2500),
        bench: str = Query(default="", description="benchmark ticker"),
    ) -> HTMLResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/nav?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        token = agent_id.strip().lower() or None
        if token and token not in scoped_agent_ids:
            token = None
        tenant_settings = deps.settings_for_tenant(tenant)
        bench_ticker = (bench.strip() or deps.default_benchmark(tenant_settings)).upper()
        _ = bench_ticker

        chart_params = urlencode({key: value for key, value in {"tenant_id": tenant, "agent_id": agent_id.strip(), "days": str(days)}.items() if value})
        chart_api_url = f"/api/nav/chart?{chart_params}"
        tool_freq_url = f"/api/tool-frequency?tenant_id={tenant}"

        try:
            nav_registry = deps.get_default_registry(tenant)
            nav_tool_label_map = {entry.tool_id: entry.label_ko or entry.tool_id for entry in nav_registry.list_entries(include_disabled=True)}
        except Exception:
            nav_tool_label_map = {}

        is_live = deps.is_live_mode(tenant_settings)
        body = render_ui_template(
            "nav_body.jinja2",
            auth_enabled=deps.auth_enabled,
            tenant=tenant,
            agent_options=[
                {"value": agent, "label": agent, "selected": agent == agent_id}
                for agent in scoped_agent_ids
            ],
            days=days,
            is_live=is_live,
            chart_api_url=chart_api_url,
            tool_freq_url=tool_freq_url,
            nav_tool_label_map=nav_tool_label_map,
        )

        return deps.html_response(
            deps.tailwind_layout("\uc6b4\uc6a9\uc131\uacfc", body, active="nav", needs_charts=True, tenant=tenant, user=deps.current_user(request)),
            max_age=60,
        )

    @app.get("/api/tool-frequency")
    def api_tool_frequency(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/tool-frequency?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        agent_key = ",".join(sorted(scoped_agent_ids))
        filters = ["tenant_id = @tenant_id", "event_type = 'react_tools_summary'"]
        params: dict[str, Any] = {"tenant_id": tenant}
        if scoped_agent_ids:
            filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
            params["agent_ids"] = scoped_agent_ids
        where = " AND ".join(filters)
        sql = f"""
        SELECT agent_id, payload_json
        FROM `{deps.repo.dataset_fqn}.agent_memory_events`
        WHERE {where}
          AND created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
        ORDER BY agent_id, created_at DESC
        LIMIT 500
        """
        cache_key = f"tool_freq:{tenant}:{agent_key}"
        rows = deps.cached_fetch(cache_key, deps.repo.fetch_rows, sql, params)

        try:
            reg = deps.get_default_registry(tenant)
            reg_entries = reg.list_entries(include_disabled=True)
            current_tools = {
                str(entry.tool_id).strip().lower()
                for entry in reg_entries
                if str(entry.tool_id).strip()
            }
        except Exception:
            current_tools = set()

        agent_tool_counts: dict[str, dict[str, int]] = {}
        tool_totals: dict[str, int] = {}
        for row in rows:
            raw = row.get("payload_json")
            if not raw:
                continue
            agent = str(row.get("agent_id") or "").strip().lower()
            if not agent:
                continue
            try:
                payload = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(payload, dict):
                continue
            events = payload.get("tool_events")
            if not isinstance(events, list):
                continue
            for event in events:
                name = str(event.get("tool") or "").strip().lower()
                if not name:
                    continue
                if current_tools and name not in current_tools:
                    continue
                bucket = agent_tool_counts.setdefault(agent, {})
                bucket[name] = bucket.get(name, 0) + 1
                tool_totals[name] = tool_totals.get(name, 0) + 1

        tools = [key for key, _ in sorted(tool_totals.items(), key=lambda item: (-item[1], item[0]))]
        if scoped_agent_ids:
            agents = [agent for agent in scoped_agent_ids if agent in agent_tool_counts]
        else:
            agents = []
        if not agents:
            agents = sorted(agent_tool_counts.keys())

        matrix = {
            tool: {
                agent: int(agent_tool_counts.get(agent, {}).get(tool, 0))
                for agent in agents
            }
            for tool in tools
        }
        return deps.json_response({"tools": tools, "agents": agents, "matrix": matrix}, max_age=60)

    @app.get("/api/nav")
    def api_nav(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent id"),
        days: int = Query(default=30, ge=1, le=365),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/nav?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        token = agent_id.strip().lower() or None
        if token and token not in scoped_agent_ids:
            token = None
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(
            f"nav:{tenant}:{agent_key}:{days}:{token or ''}",
            deps.fetch_nav,
            tenant_id=tenant,
            days=days,
            agent_id=token,
            agent_ids=scoped_agent_ids,
        )
        return deps.json_response(rows, max_age=30)

    @app.get("/api/tenant-status")
    def api_tenant_status(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/tenant-status?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        payload = deps.cached_fetch(f"tenant_run_status:{tenant}", deps.latest_tenant_status_payload, tenant)
        return deps.json_response(payload or {}, max_age=30)

    @app.get("/api/nav/chart")
    def api_nav_chart(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent_id filter"),
        days: int = Query(default=180, ge=5, le=2500),
        bench: str = Query(default="", description="benchmark ticker"),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/nav/chart?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        token = agent_id.strip().lower() or None
        if token and token not in scoped_agent_ids:
            token = None
        tenant_settings = deps.settings_for_tenant(tenant)
        bench_ticker = (bench.strip() or deps.default_benchmark(tenant_settings)).upper()
        _ = bench_ticker
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(
            f"nav:{tenant}:{agent_key}:{days}:{token or ''}",
            deps.fetch_nav,
            tenant_id=tenant,
            days=days,
            agent_id=token,
            agent_ids=scoped_agent_ids,
        )
        token_rows = deps.cached_fetch(
            f"token_usage:{tenant}:{agent_key}:{days}:{token or ''}",
            deps.fetch_token_usage_summary,
            tenant_id=tenant,
            days=days,
            agent_id=token,
            agent_ids=scoped_agent_ids,
        )
        trade_rows = deps.cached_fetch(
            f"trade_usage:{tenant}:{agent_key}:{days}:{token or ''}",
            deps.fetch_trade_count_summary,
            tenant_id=tenant,
            days=days,
            agent_id=token,
            agent_ids=scoped_agent_ids,
        )

        token_daily_raw = deps.cached_fetch(
            f"token_daily:{tenant}:{agent_key}:{days}:{token or ''}",
            deps.fetch_token_usage_daily,
            tenant_id=tenant,
            days=days,
            agent_id=token,
            agent_ids=scoped_agent_ids,
        )
        trade_daily_raw = deps.cached_fetch(
            f"trade_daily:{tenant}:{agent_key}:{days}:{token or ''}",
            deps.fetch_trade_count_daily,
            tenant_id=tenant,
            days=days,
            agent_id=token,
            agent_ids=scoped_agent_ids,
        )

        if not rows:
            return deps.json_response({"labels": [], "nav": [], "dd": [], "summary": [], "token_daily": {"labels": [], "datasets": []}, "trade_daily": {"labels": [], "datasets": []}}, max_age=60)

        labels = sorted({deps.to_date(row.get("nav_date")) for row in rows if row.get("nav_date")})
        series_by_agent: dict[str, dict[str, dict[str, float | None]]] = {}
        for row in rows:
            agent = str(row.get("agent_id") or "").strip()
            if not agent:
                continue
            nav_date = deps.to_date(row.get("nav_date"))
            nav_krw = deps.safe_float(row.get("nav_krw"), 0.0)
            if nav_krw <= 0:
                continue
            pnl_ratio: float | None = None
            if row.get("pnl_ratio") is not None:
                try:
                    pnl_ratio = float(row["pnl_ratio"])
                except (TypeError, ValueError):
                    pass
            pnl_krw: float | None = None
            if row.get("pnl_krw") is not None:
                try:
                    pnl_krw = float(row["pnl_krw"])
                except (TypeError, ValueError):
                    pass
            series_by_agent.setdefault(agent, {})[nav_date] = {
                "nav_krw": nav_krw,
                "pnl_ratio": pnl_ratio,
                "pnl_krw": pnl_krw,
            }

        palette = ["#0f766e", "#e76f51", "#0ea5e9", "#0891b2", "#475569", "#7c3aed", "#f59e0b"]
        nav_datasets: list[dict[str, Any]] = []
        dd_datasets: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []

        for idx, agent in enumerate(sorted(series_by_agent.keys())):
            raw_nav = [deps.safe_float((series_by_agent[agent].get(label) or {}).get("nav_krw"), 0.0) if series_by_agent[agent].get(label) else None for label in labels]
            raw_pnl_krw = [(series_by_agent[agent].get(label) or {}).get("pnl_krw") if series_by_agent[agent].get(label) else None for label in labels]
            raw_pnl_ratio = [(series_by_agent[agent].get(label) or {}).get("pnl_ratio") if series_by_agent[agent].get(label) else None for label in labels]
            idx_vals = deps.chained_index(raw_nav, raw_pnl_krw, raw_pnl_ratio)
            dd_vals = deps.drawdown(idx_vals)
            color = palette[idx % len(palette)]
            nav_datasets.append({"label": agent, "data": [None if value is None else round(float(value), 4) for value in idx_vals], "borderColor": color, "backgroundColor": color, "pointRadius": 0, "borderWidth": 2, "tension": 0.15})
            dd_datasets.append({"label": agent, "data": [None if value is None else round(float(value) * 100.0, 3) for value in dd_vals], "borderColor": color, "backgroundColor": color, "pointRadius": 0, "borderWidth": 2, "tension": 0.15})
            last_nav = None
            for value in reversed(raw_nav):
                if value is not None:
                    last_nav = float(value)
                    break
            token_usage = token_rows.get(agent, {})
            summary_rows.append(
                {
                    "name": agent,
                    "ret": deps.total_return(idx_vals),
                    "mdd": deps.max_drawdown(dd_vals),
                    "last": last_nav,
                    "trade_count": deps.safe_int(trade_rows.get(agent), 0),
                    "llm_calls": deps.safe_int(token_usage.get("llm_calls"), 0),
                    "prompt_tokens": deps.safe_int(token_usage.get("prompt_tokens"), 0),
                    "completion_tokens": deps.safe_int(token_usage.get("completion_tokens"), 0),
                    "cached_tokens": deps.safe_int(token_usage.get("cached_tokens"), 0),
                    "thinking_tokens": deps.safe_int(token_usage.get("thinking_tokens"), 0),
                    "total_tokens": deps.safe_int(token_usage.get("total_tokens"), 0),
                    "cache_ratio": float(token_usage.get("cache_ratio") or 0.0),
                }
            )

        summary_rows.sort(key=lambda row: float(row.get("ret") or 0.0), reverse=True)

        # --- Daily token usage & trade count series ---
        agent_colors = {"gpt": "#10a37f", "gemini": "#4285f4", "claude": "#d97757"}
        fallback_colors = ["#0ea5e9", "#8b5cf6", "#22c55e", "#f59e0b"]

        def _build_daily_series(raw_rows: list[dict[str, Any]], date_key: str, value_key: str) -> dict[str, Any]:
            by_date_agent: dict[str, dict[str, int]] = {}
            all_agents: set[str] = set()
            for r in raw_rows:
                d_val = r.get(date_key)
                d_str = str(d_val.isoformat() if hasattr(d_val, "isoformat") else d_val) if d_val else ""
                if not d_str:
                    continue
                agent = str(r.get("agent_id") or "").strip().lower()
                if not agent:
                    continue
                all_agents.add(agent)
                by_date_agent.setdefault(d_str, {})[agent] = int(r.get(value_key) or 0)
            d_labels = sorted(by_date_agent.keys())
            s_agents = sorted(all_agents)
            datasets = []
            for i, a in enumerate(s_agents):
                color = agent_colors.get(a, fallback_colors[i % len(fallback_colors)])
                datasets.append({
                    "label": a,
                    "data": [by_date_agent.get(d, {}).get(a, 0) for d in d_labels],
                    "backgroundColor": color + "cc",
                    "borderColor": color,
                    "borderWidth": 0,
                    "borderRadius": 3,
                    "barPercentage": 0.7,
                })
            return {"labels": d_labels, "datasets": datasets}

        token_daily = _build_daily_series(token_daily_raw, "usage_date", "total_tokens")
        trade_daily = _build_daily_series(trade_daily_raw, "trade_date", "trade_count")

        return deps.json_response({
            "labels": labels,
            "nav": nav_datasets,
            "dd": dd_datasets,
            "summary": summary_rows,
            "token_daily": token_daily,
            "trade_daily": trade_daily,
        }, max_age=60)
