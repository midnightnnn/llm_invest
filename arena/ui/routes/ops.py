from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from arena.data.bq import BigQueryRepository
from arena.providers.credentials import parse_model_secret_providers, runtime_credential_flags
from arena.ui.templating import render_ui_template


@dataclass(frozen=True)
class OpsRouteDeps:
    repo: BigQueryRepository
    executor: Any
    cached_fetch: Callable[..., Any]
    current_user: Callable[[Request], dict[str, Any] | None]
    is_operator: Callable[[dict[str, Any] | None], bool]
    tailwind_layout: Callable[..., str]
    html_response: Callable[..., HTMLResponse]
    json_response: Callable[..., JSONResponse]
    metric_card: Callable[..., str]
    run_status_meta: dict[str, dict[str, str]]
    credential_store: Any = None


def _ops_status_badge(status: str, run_status_meta: dict[str, dict[str, str]]) -> str:
    meta = run_status_meta.get(status, run_status_meta.get("warning", {}))
    label = html.escape(str(meta.get("label") or status))
    cls = html.escape(str(meta.get("badge") or "bg-ink-100 text-ink-700"), quote=True)
    return f'<span class="inline-block rounded-full px-2.5 py-0.5 text-[11px] font-semibold {cls}">{label}</span>'


def _ops_bool_dot(val: bool) -> str:
    if val:
        return '<span class="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" title="Active"></span>'
    return '<span class="inline-block h-2.5 w-2.5 rounded-full bg-ink-200" title="None"></span>'


def _ops_fmt_time(val: object) -> str:
    if not val:
        return '<span class="text-ink-400">-</span>'
    from datetime import datetime as _dt

    if isinstance(val, _dt):
        return html.escape(val.strftime("%m-%d %H:%M"))
    raw = str(val).strip()
    if len(raw) >= 16:
        return html.escape(raw[5:16].replace("T", " "))
    return html.escape(raw)

def _refresh_credential_flags(credentials: list[dict[str, Any]], credential_store: Any) -> None:
    """Override BQ-cached model flags with live values from Secret Manager."""
    if not credential_store:
        return
    seen: set[str] = set()
    for row in credentials:
        tenant_id = str(row.get("tenant_id") or "").strip()
        if not tenant_id or tenant_id in seen:
            continue
        seen.add(tenant_id)
        model_secret = str(row.get("model_secret_name") or "").strip()
        if not model_secret:
            continue
        try:
            payload = credential_store._latest_secret_json(secret_id=model_secret)
            flags = runtime_credential_flags(parse_model_secret_providers(payload))
            row["has_openai"] = flags["has_openai"]
            row["has_gemini"] = flags["has_gemini"]
            row["has_anthropic"] = flags["has_anthropic"]
        except Exception:
            pass


def _ops_page_context(
    *,
    deps: OpsRouteDeps,
    tenants: list[str],
    statuses: list[dict[str, Any]],
    credentials: list[dict[str, Any]],
    audit_logs: list[dict[str, Any]],
) -> dict[str, Any]:
    status_map: dict[str, dict[str, Any]] = {}
    for row in statuses:
        tenant_id = str(row.get("tenant_id") or "").strip()
        if tenant_id and tenant_id not in status_map:
            status_map[tenant_id] = row

    cred_map: dict[str, dict[str, Any]] = {}
    for row in credentials:
        tenant_id = str(row.get("tenant_id") or "").strip()
        if tenant_id and tenant_id not in cred_map:
            cred_map[tenant_id] = row

    n_total = len(tenants) or len(status_map)
    n_ok = sum(1 for row in status_map.values() if str(row.get("status") or "") in {"success", "running"})
    n_warn = sum(1 for row in status_map.values() if str(row.get("status") or "") in {"warning", "blocked"})
    n_fail = sum(1 for row in status_map.values() if str(row.get("status") or "") == "failed")

    all_tenants = sorted(set(list(tenants) + list(status_map.keys())))
    health_rows: list[dict[str, Any]] = []
    for tenant_id in all_tenants:
        row = status_map.get(tenant_id, {})
        status = str(row.get("status") or "unknown")
        run_type = str(row.get("run_type") or "-")
        stage = str(row.get("stage") or "-")
        started = _ops_fmt_time(row.get("started_at"))
        finished = _ops_fmt_time(row.get("finished_at"))
        message = str(row.get("message") or "-")[:80]
        log_uri = str(row.get("log_uri") or "").strip()
        health_rows.append(
            {
                "tenant_id": tenant_id,
                "status_badge_html": _ops_status_badge(status, deps.run_status_meta),
                "run_type": run_type,
                "stage": stage,
                "started_html": started,
                "finished_html": finished,
                "message": message,
                "log_uri": log_uri,
            }
        )

    cred_rows: list[dict[str, Any]] = []
    for tenant_id in all_tenants:
        row = cred_map.get(tenant_id, {})
        cred_rows.append(
            {
                "tenant_id": tenant_id,
                "kis_env": str(row.get("kis_env") or "-"),
                "openai_dot_html": _ops_bool_dot(bool(row.get("has_openai"))),
                "gemini_dot_html": _ops_bool_dot(bool(row.get("has_gemini"))),
                "anthropic_dot_html": _ops_bool_dot(bool(row.get("has_anthropic"))),
                "updated_at_html": _ops_fmt_time(row.get("updated_at")),
            }
        )

    audit_rows_out: list[dict[str, Any]] = []
    for log in audit_logs[:30]:
        timestamp = _ops_fmt_time(log.get("created_at"))
        user = str(log.get("user_email") or "-")[:30]
        tenant = str(log.get("tenant_id") or "-")
        action = str(log.get("action") or "-")
        status = str(log.get("status") or "").strip().lower()
        status_cls = "text-emerald-700 bg-emerald-50" if status == "ok" else "text-rose-700 bg-rose-50" if status == "error" else "text-amber-700 bg-amber-50"
        status_badge = f'<span class="rounded-full px-2 py-0.5 text-[10px] font-semibold {status_cls}">{html.escape(status or "-")}</span>'
        audit_rows_out.append(
            {
                "timestamp_html": timestamp,
                "user": user,
                "tenant": tenant,
                "action": action,
                "status_badge_html": status_badge,
            }
        )

    return {
        "cards": [
            {"title": "Total Tenants", "value": str(n_total), "note": f"{len(cred_map)} with credentials", "value_id": ""},
            {"title": "Healthy", "value": str(n_ok), "note": "success + running", "value_id": "ops-healthy"},
            {"title": "Warning", "value": str(n_warn), "note": "warning + blocked", "value_id": "ops-warning"},
            {"title": "Failed", "value": str(n_fail), "note": "execution failed", "value_id": "ops-failed"},
        ],
        "health_rows": health_rows,
        "credential_rows": cred_rows,
        "audit_rows": audit_rows_out,
    }


def register_ops_routes(app: FastAPI, *, deps: OpsRouteDeps) -> None:
    @app.get("/ops", response_class=HTMLResponse)
    def ops_page(request: Request) -> HTMLResponse:
        user = deps.current_user(request)
        if not deps.is_operator(user):
            return HTMLResponse(
                '<section class="reveal mt-12 text-center">'
                '<p class="text-lg text-ink-500">Access denied.</p></section>',
                status_code=403,
            )
        fut_tenants = deps.executor.submit(deps.cached_fetch, "ops:tenants", deps.repo.list_runtime_tenants)
        fut_statuses = deps.executor.submit(deps.cached_fetch, "ops:statuses", deps.repo.all_tenant_run_statuses, limit=100)
        fut_creds = deps.executor.submit(deps.cached_fetch, "ops:creds", deps.repo.recent_runtime_credentials, limit=100)
        fut_audit = deps.executor.submit(deps.cached_fetch, "ops:audit", deps.repo.recent_runtime_audit_logs, limit=50)
        credentials = fut_creds.result()
        _refresh_credential_flags(credentials, deps.credential_store)
        body_context = _ops_page_context(
            deps=deps,
            tenants=fut_tenants.result(),
            statuses=fut_statuses.result(),
            credentials=credentials,
            audit_logs=fut_audit.result(),
        )
        body = render_ui_template("ops_body.jinja2", **body_context)
        return deps.html_response(
            deps.tailwind_layout("Operations", body, active="ops", user=user, max_width_class="max-w-[1400px]"),
            max_age=15,
        )

    @app.get("/api/ops/health")
    def api_ops_health(request: Request) -> JSONResponse:
        user = deps.current_user(request)
        if not deps.is_operator(user):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        rows = deps.cached_fetch("ops:statuses", deps.repo.all_tenant_run_statuses, limit=100)
        return deps.json_response(rows, max_age=15)

    @app.get("/api/ops/credentials")
    def api_ops_credentials(request: Request) -> JSONResponse:
        user = deps.current_user(request)
        if not deps.is_operator(user):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        rows = deps.cached_fetch("ops:creds", deps.repo.recent_runtime_credentials, limit=100)
        return deps.json_response(rows, max_age=60)

    @app.get("/api/ops/audit")
    def api_ops_audit(request: Request) -> JSONResponse:
        user = deps.current_user(request)
        if not deps.is_operator(user):
            return JSONResponse({"error": "forbidden"}, status_code=403)
        rows = deps.cached_fetch("ops:audit", deps.repo.recent_runtime_audit_logs, limit=50)
        return deps.json_response(rows, max_age=15)
