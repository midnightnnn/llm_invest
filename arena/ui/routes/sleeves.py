from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from arena.reconciliation import StateRecoveryService
from arena.ui.routes.capital_data import (
    build_event_timeline_data,
    build_recon_dashboard_data,
    build_sankey_data,
    build_treemap_data,
)
from arena.ui.templating import render_ui_template

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SleeveRouteDeps:
    repo: Any
    settings_enabled: bool
    resolve_admin_context: Callable[..., Any]
    resolve_viewer_context: Callable[..., Any]
    tenant_access_denied: Callable[[str, str], bool]
    current_admin_view_model: Callable[[str], dict[str, Any]]
    settings_redirect: Callable[..., RedirectResponse]
    invalidate_tenant_cache: Callable[..., None]
    settings_for_tenant: Callable[[str], Any]
    scoped_agent_ids_for_tenant: Callable[[str], list[str]]
    is_live_mode: Callable[[Any | None], bool]
    live_market_sources: Callable[[Any | None], list[str] | None]
    cached_fetch: Callable[..., Any]
    fetch_sleeves: Callable[..., list[dict[str, Any]]]
    fetch_sleeve_snapshot_cards: Callable[..., dict[str, Any]]
    tailwind_layout: Callable[..., str]
    html_response: Callable[..., HTMLResponse]
    json_response: Callable[..., JSONResponse]
    fmt_ts: Callable[[object], str]
    float_env: Callable[[str, float], float]


def _fmt_qty(qty: object) -> str:
    try:
        value = float(qty or 0.0)
    except (TypeError, ValueError):
        return "0"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _positions_len(raw: object) -> int:
    text = str(raw or "")
    if not text:
        return 0
    try:
        parsed = json.loads(text)
    except Exception:
        return 0
    return len(parsed) if isinstance(parsed, (dict, list)) else 0


def _sleeve_capital(row: dict[str, Any]) -> float:
    cash = float(row.get("initial_cash_krw") or 0.0)
    raw = str(row.get("initial_positions_json") or "[]")
    try:
        parsed = json.loads(raw)
    except Exception:
        return cash
    if isinstance(parsed, dict):
        parsed = list(parsed.values())
    elif not isinstance(parsed, list):
        return cash
    for position in parsed:
        if not isinstance(position, dict):
            continue
        try:
            cash += float(position.get("quantity") or 0) * float(position.get("avg_price_krw") or 0)
        except (TypeError, ValueError):
            continue
    return cash


def _is_cash_transfer_event(row: dict[str, Any]) -> bool:
    token = str(row.get("transfer_type") or "").strip().upper()
    return token in {"CASH", "CASH_TRANSFER", "CASH_ONLY", "WITHDRAWAL", "DEPOSIT"} or not str(row.get("ticker") or "").strip()


def _signed_transfer_cash_amount(agent_id: str, row: dict[str, Any]) -> float:
    agent = str(agent_id or "").strip()
    from_agent = str(row.get("from_agent_id") or "").strip()
    to_agent = str(row.get("to_agent_id") or "").strip()
    if not agent or agent not in {from_agent, to_agent}:
        return 0.0

    try:
        qty = float(row.get("quantity") or 0.0)
    except (TypeError, ValueError):
        qty = 0.0
    try:
        price = float(row.get("price_krw") or 0.0)
    except (TypeError, ValueError):
        price = 0.0
    try:
        amount = float(row.get("amount_krw") or 0.0)
    except (TypeError, ValueError):
        amount = 0.0
    if abs(amount) <= 1e-9 and qty > 0 and price > 0:
        amount = qty * price
    if abs(amount) <= 1e-9:
        return 0.0

    if _is_cash_transfer_event(row):
        if agent == from_agent:
            return -abs(amount)
        if agent == to_agent:
            return abs(amount)
        return 0.0

    if agent == from_agent:
        return abs(amount)
    if agent == to_agent:
        return -abs(amount)
    return 0.0


def _event_date_str(value: object) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if value is None:
        return ""
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def _chart_scripts_html(chart_specs: list[dict[str, Any]] | None) -> str:
    specs = [dict(spec) for spec in chart_specs or [] if isinstance(spec, dict) and spec.get("id")]
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


def register_sleeve_routes(app: FastAPI, *, deps: SleeveRouteDeps) -> None:
    @app.get("/admin/sleeve")
    def admin_sleeve_get(
        request: Request,
        tenant_id: str = Query(default="local"),
    ) -> JSONResponse:
        if not deps.settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _, _, tenant, _, redirect = deps.resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/settings?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        view_model = deps.current_admin_view_model(tenant)
        return JSONResponse({"tenant_id": tenant, "sleeve_capital_krw": view_model["sleeve_capital_krw"]})

    @app.post("/admin/sleeve")
    def admin_sleeve_save(
        request: Request,
        tenant_id: str = Form(default="local"),
        updated_by: str = Form(default="ui-admin"),
        sleeve_capital_krw: str = Form(default=""),
        agent_capitals_json: str = Form(default=""),
    ) -> Response:
        if not deps.settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _, user_email, tenant, _, redirect = deps.resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/settings?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        if deps.tenant_access_denied(tenant_id, tenant):
            return deps.settings_redirect(tenant, ok=False, msg="tenant access denied")
        try:
            sleeve = float(sleeve_capital_krw)
        except ValueError:
            return deps.settings_redirect(tenant, ok=False, msg="sleeve_capital_krw must be numeric", tab="capital")
        if sleeve <= 0:
            return deps.settings_redirect(tenant, ok=False, msg="sleeve_capital_krw must be > 0", tab="capital")

        # Parse per-agent capital overrides
        target_capitals: dict[str, float] = {}
        if agent_capitals_json.strip():
            try:
                parsed = json.loads(agent_capitals_json)
                if isinstance(parsed, dict):
                    target_capitals = {str(k).strip().lower(): float(v) for k, v in parsed.items() if float(v) > 0}
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        try:
            deps.repo.set_config(
                tenant,
                "sleeve_capital_krw",
                str(sleeve),
                updated_by=user_email or updated_by,
            )

            # Sync per-agent capitals into agents_config
            if target_capitals:
                try:
                    agents_cfg_raw = deps.repo.get_config(tenant, "agents_config")
                    agents_cfg = json.loads(agents_cfg_raw) if agents_cfg_raw else []
                    if isinstance(agents_cfg, list):
                        for entry in agents_cfg:
                            if not isinstance(entry, dict):
                                continue
                            aid = str(entry.get("id") or "").strip().lower()
                            if aid in target_capitals:
                                entry["capital_krw"] = target_capitals[aid]
                        deps.repo.set_config(
                            tenant,
                            "agents_config",
                            json.dumps(agents_cfg, ensure_ascii=False),
                            updated_by=user_email or updated_by,
                        )
                except Exception as cfg_exc:
                    logger.warning("Per-agent capital config sync failed: %s", cfg_exc)

            deps.invalidate_tenant_cache(tenant, "runtime", "memory", "portfolio")
            tenant_settings = deps.settings_for_tenant(tenant)
            tenant_is_live = deps.is_live_mode(tenant_settings)
            tenant_live_sources = deps.live_market_sources(tenant_settings) if tenant_is_live else None
            scoped_agent_ids = deps.scoped_agent_ids_for_tenant(tenant)
            if not scoped_agent_ids:
                scoped_agent_ids = [str(agent_id).strip().lower() for agent_id in tenant_settings.agent_ids if str(agent_id).strip()]
            sync_summary: dict[str, Any] = {"agents": len(scoped_agent_ids)}
            sync_capitals = getattr(deps.repo, "retarget_agent_capitals_preserve_positions", None)
            sync_sleeves = getattr(deps.repo, "retarget_agent_sleeves_preserve_positions", None)
            sync_fn = sync_capitals if callable(sync_capitals) else sync_sleeves
            if callable(sync_fn) and scoped_agent_ids:
                sync_kwargs: dict[str, Any] = {
                    "agent_ids": scoped_agent_ids,
                    "target_sleeve_capital_krw": sleeve,
                    "include_simulated": not tenant_is_live,
                    "sources": tenant_live_sources,
                    "tenant_id": tenant,
                }
                if target_capitals:
                    sync_kwargs["target_capitals"] = target_capitals
                if sync_fn is sync_capitals:
                    sync_kwargs["created_by"] = user_email or updated_by
                try:
                    sync_out = sync_fn(**sync_kwargs)
                    over_target_agents = sorted([agent_id for agent_id, meta in dict(sync_out or {}).items() if bool(dict(meta).get("over_target"))])
                    sync_summary["over_target_agents"] = over_target_agents
                except Exception as retarget_exc:
                    logger.warning(
                        "[yellow]Sleeve retarget after sleeve save failed[/yellow] tenant=%s err=%s",
                        tenant,
                        str(retarget_exc),
                    )
                    sync_summary["retarget_error"] = str(retarget_exc)

            build_sleeve = getattr(deps.repo, "build_agent_sleeve_snapshot", None)
            upsert_nav = getattr(deps.repo, "upsert_agent_nav_daily", None)
            if callable(build_sleeve) and callable(upsert_nav) and scoped_agent_ids:
                nav_date = datetime.now(timezone.utc).date()
                nav_updated = 0
                nav_failed: list[str] = []
                for agent_id in scoped_agent_ids:
                    try:
                        snap, baseline, meta = build_sleeve(
                            agent_id=agent_id,
                            sources=tenant_live_sources,
                            include_simulated=not tenant_is_live,
                            tenant_id=tenant,
                        )
                        upsert_nav(
                            nav_date=nav_date,
                            agent_id=agent_id,
                            nav_krw=float(snap.total_equity_krw),
                            baseline_equity_krw=float(baseline),
                            cash_krw=float(snap.cash_krw),
                            market_value_krw=sum(pos.market_value_krw() for pos in snap.positions.values()),
                            capital_flow_krw=float((meta or {}).get("capital_flow_krw") or 0.0)
                            + float((meta or {}).get("manual_cash_adjustment_krw") or 0.0),
                            fx_source=str((meta or {}).get("fx_source") or ""),
                            valuation_source=str((meta or {}).get("valuation_source") or "agent_sleeve_snapshot"),
                            tenant_id=tenant,
                        )
                        nav_updated += 1
                    except Exception as nav_exc:
                        nav_failed.append(agent_id)
                        logger.warning(
                            "[yellow]Agent NAV sync after sleeve retarget failed[/yellow] tenant=%s agent=%s err=%s",
                            tenant,
                            agent_id,
                            str(nav_exc),
                        )
                sync_summary["nav_sync"] = {
                    "updated": nav_updated,
                    "failed_agents": nav_failed,
                    "nav_date": nav_date.isoformat(),
                }
            deps.repo.append_runtime_audit_log(
                action="admin_sleeve_save",
                status="ok",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={
                    "sleeve_capital_krw": sleeve,
                    "sync": sync_summary,
                },
            )
            return deps.settings_redirect(tenant, ok=True, msg="Target Capital 저장 완료", tab="capital")
        except Exception as exc:
            deps.repo.append_runtime_audit_log(
                action="admin_sleeve_save",
                status="error",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"error": str(exc)},
            )
            return deps.settings_redirect(tenant, ok=False, msg=str(exc), tab="capital")

    @app.post("/admin/recover")
    def admin_recover_sleeves(
        request: Request,
        tenant_id: str = Form(default="local"),
        updated_by: str = Form(default="ui-admin"),
        live: str = Form(default=""),
    ) -> Response:
        if not deps.settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _, user_email, tenant, _, redirect = deps.resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/settings?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        if deps.tenant_access_denied(tenant_id, tenant):
            return deps.settings_redirect(tenant, ok=False, msg="tenant access denied")

        live_flag = str(live or "").strip().lower() in {"1", "true", "yes", "y", "on"}
        tenant_settings = deps.settings_for_tenant(tenant)
        scoped_agent_ids = deps.scoped_agent_ids_for_tenant(tenant)
        if not scoped_agent_ids:
            scoped_agent_ids = [str(agent_id).strip().lower() for agent_id in tenant_settings.agent_ids if str(agent_id).strip()]
        if not scoped_agent_ids:
            return deps.settings_redirect(tenant, ok=False, msg="no configured agents")

        try:
            snapshot = None
            if live_flag:
                from arena.open_trading.sync import AccountSyncService, BrokerCashSyncService, BrokerTradeSyncService

                try:
                    BrokerTradeSyncService(settings=tenant_settings, repo=deps.repo).sync_broker_trade_events(
                        days=max(int(os.getenv("ARENA_BROKER_TRADE_SYNC_DAYS", "14") or "14"), 1)
                    )
                except Exception as sync_exc:
                    logger.warning(
                        "[yellow]Admin recover broker trade sync skipped[/yellow] tenant=%s err=%s",
                        tenant,
                        str(sync_exc),
                    )
                try:
                    BrokerCashSyncService(settings=tenant_settings, repo=deps.repo).sync_broker_cash_events(
                        days=max(int(os.getenv("ARENA_BROKER_CASH_SYNC_DAYS", "14") or "14"), 1)
                    )
                except Exception as sync_exc:
                    logger.warning(
                        "[yellow]Admin recover broker cash sync skipped[/yellow] tenant=%s err=%s",
                        tenant,
                        str(sync_exc),
                    )
                snapshot = AccountSyncService(settings=tenant_settings, repo=deps.repo).sync_account_snapshot()

            recovery = StateRecoveryService(
                settings=tenant_settings,
                repo=deps.repo,
                excluded_tickers=list(getattr(tenant_settings, "reconcile_excluded_tickers", [])),
                qty_tolerance=max(deps.float_env("ARENA_RECONCILE_QTY_TOLERANCE", 1e-9), 0.0),
                cash_tolerance_krw=max(deps.float_env("ARENA_RECONCILE_CASH_TOLERANCE_KRW", 50_000.0), 0.0),
                cash_reconciliation_enabled=str(os.getenv("ARENA_RECONCILE_CASH_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "y", "on"},
            ).recover_and_reconcile(
                agent_ids=scoped_agent_ids,
                tenant_id=tenant,
                include_simulated=tenant_settings.trading_mode != "live",
                auto_recover=True,
                account_snapshot=snapshot,
                created_by=user_email or updated_by,
            )

            build_sleeve = getattr(deps.repo, "build_agent_sleeve_snapshot", None)
            upsert_nav = getattr(deps.repo, "upsert_agent_nav_daily", None)
            nav_updated = 0
            if recovery.ok and callable(build_sleeve) and callable(upsert_nav):
                nav_date = datetime.now(timezone.utc).date()
                tenant_is_live = deps.is_live_mode(tenant_settings)
                tenant_live_sources = deps.live_market_sources(tenant_settings) if tenant_is_live else None
                for agent_id in scoped_agent_ids:
                    snap, baseline, meta = build_sleeve(
                        agent_id=agent_id,
                        sources=tenant_live_sources,
                        include_simulated=not tenant_is_live,
                        tenant_id=tenant,
                    )
                    upsert_nav(
                        nav_date=nav_date,
                        agent_id=agent_id,
                        nav_krw=float(snap.total_equity_krw),
                        baseline_equity_krw=float(baseline),
                        cash_krw=float(snap.cash_krw),
                        market_value_krw=sum(pos.market_value_krw() for pos in snap.positions.values()),
                        capital_flow_krw=float((meta or {}).get("capital_flow_krw") or 0.0)
                        + float((meta or {}).get("manual_cash_adjustment_krw") or 0.0),
                        fx_source=str((meta or {}).get("fx_source") or ""),
                        valuation_source=str((meta or {}).get("valuation_source") or "agent_sleeve_snapshot"),
                        tenant_id=tenant,
                    )
                    nav_updated += 1

            deps.repo.append_runtime_audit_log(
                action="admin_recover_sleeves",
                status="ok" if recovery.ok else "error",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={
                    "status": recovery.status,
                    "applied_checkpoints": recovery.applied_checkpoints,
                    "before_status": recovery.before.status,
                    "after_status": recovery.after.status,
                    "nav_updated": nav_updated,
                    "live": live_flag,
                },
            )
            deps.invalidate_tenant_cache(tenant, "portfolio", "status")
            if recovery.ok:
                return deps.settings_redirect(
                    tenant,
                    ok=True,
                    msg=f"recovery {recovery.status}: checkpoints={recovery.applied_checkpoints}",
                    tab="capital",
                )
            return deps.settings_redirect(
                tenant,
                ok=False,
                msg=f"recovery failed: checkpoints={recovery.applied_checkpoints}",
                tab="capital",
            )
        except Exception as exc:
            deps.repo.append_runtime_audit_log(
                action="admin_recover_sleeves",
                status="error",
                user_email=user_email or updated_by,
                tenant_id=tenant,
                detail={"error": str(exc), "live": live_flag},
            )
            return deps.settings_redirect(tenant, ok=False, msg=str(exc), tab="capital")

    @app.get("/sleeves", response_class=HTMLResponse)
    def sleeves(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
    ) -> HTMLResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/sleeves?tenant_id={tenant_id}",
        )
        if redirect:
            return redirect
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(f"sleeves:{tenant}:{agent_key}", deps.fetch_sleeves, tenant_id=tenant, agent_ids=scoped_agent_ids)
        snapshot_payload = deps.cached_fetch(
            f"sleeve_cards:{tenant}:{agent_key}",
            deps.fetch_sleeve_snapshot_cards,
            tenant_id=tenant,
            agent_ids=scoped_agent_ids,
        )
        tenant_settings = deps.settings_for_tenant(tenant)
        is_live = deps.is_live_mode(tenant_settings)

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
        charts_html = _chart_scripts_html((snapshot_payload or {}).get("charts"))

        body = render_ui_template(
            "sleeves_body.jinja2",
            is_live=is_live,
            cards_html=cards_html,
            charts_html=charts_html,
            sleeve_rows=sleeve_rows,
        )

        return deps.html_response(deps.tailwind_layout("Sleeves", body, active="sleeves", needs_charts=True, tenant=tenant), max_age=60)

    @app.get("/api/sleeves")
    def api_sleeves(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/sleeves?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        agent_key = ",".join(sorted(scoped_agent_ids))
        rows = deps.cached_fetch(f"sleeves:{tenant}:{agent_key}", deps.fetch_sleeves, tenant_id=tenant, agent_ids=scoped_agent_ids)
        return deps.json_response(rows, max_age=60)

    @app.get("/api/sleeve-snapshot-cards")
    def api_sleeve_snapshot_cards(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        compact: bool = Query(default=False, description="compact card mode"),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/sleeve-snapshot-cards?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        agent_key = ",".join(sorted(scoped_agent_ids))
        cache_suffix = ":compact" if compact else ""
        payload = deps.cached_fetch(
            f"sleeve_cards:{tenant}:{agent_key}{cache_suffix}",
            deps.fetch_sleeve_snapshot_cards,
            tenant_id=tenant,
            agent_ids=scoped_agent_ids,
            compact=compact,
        )
        return deps.json_response(payload, max_age=60)

    # ------------------------------------------------------------------
    # Capital Visualisation API endpoints
    # ------------------------------------------------------------------

    def _build_agent_snapshots_map(
        tenant: str,
        agent_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Fetch sleeve snapshots for all agents and return a simple dict map."""
        tenant_settings = deps.settings_for_tenant(tenant)
        is_live = deps.is_live_mode(tenant_settings)
        live_sources = deps.live_market_sources(tenant_settings) if is_live else None

        out: dict[str, dict[str, Any]] = {}
        build_fn = getattr(deps.repo, "build_agent_sleeve_snapshot", None)
        if not callable(build_fn):
            return out

        for aid in agent_ids:
            try:
                snap, baseline, meta = build_fn(
                    agent_id=aid,
                    sources=live_sources,
                    include_simulated=not is_live,
                    tenant_id=tenant,
                )
                positions = []
                for ticker, pos in sorted(snap.positions.items()):
                    if pos.quantity <= 0:
                        continue
                    mv = pos.market_value_krw()
                    ret_pct = ((pos.market_price_krw - pos.avg_price_krw) / pos.avg_price_krw * 100) if pos.avg_price_krw > 0 else None
                    positions.append({
                        "ticker": ticker,
                        "quantity": pos.quantity,
                        "avg_price_krw": pos.avg_price_krw,
                        "market_price_krw": pos.market_price_krw,
                        "market_value_krw": mv,
                        "return_pct": ret_pct,
                    })
                out[aid] = {
                    "cash_krw": snap.cash_krw,
                    "total_equity_krw": snap.total_equity_krw,
                    "positions": positions,
                    "baseline_equity_krw": float(baseline),
                    "meta": dict(meta) if isinstance(meta, dict) else {},
                }
            except Exception as exc:
                logger.warning("Capital snapshot failed for %s: %s", aid, exc)
        return out

    @app.get("/api/capital/sankey")
    def api_capital_sankey(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/capital/sankey?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        agent_key = ",".join(sorted(scoped_agent_ids))
        snapshots = deps.cached_fetch(
            f"capital_snapshots:{tenant}:{agent_key}",
            _build_agent_snapshots_map,
            tenant,
            scoped_agent_ids,
        )

        recon_status = "unknown"
        latest_recon_fn = getattr(deps.repo, "latest_reconciliation_run", None)
        if callable(latest_recon_fn):
            try:
                run = latest_recon_fn(tenant_id=tenant)
                if run:
                    recon_status = str(run.get("status") or "unknown")
            except Exception:
                pass

        return deps.json_response(build_sankey_data(snapshots, recon_status), max_age=60)

    @app.get("/api/capital/treemap")
    def api_capital_treemap(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/capital/treemap?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        agent_key = ",".join(sorted(scoped_agent_ids))
        snapshots = deps.cached_fetch(
            f"capital_snapshots:{tenant}:{agent_key}",
            _build_agent_snapshots_map,
            tenant,
            scoped_agent_ids,
        )
        return deps.json_response(build_treemap_data(snapshots), max_age=60)

    @app.get("/api/capital/recon")
    def api_capital_recon(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
    ) -> JSONResponse:
        tenant, _, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/capital/recon?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)

        latest_run = None
        issues: list[dict[str, Any]] = []
        latest_fn = getattr(deps.repo, "latest_reconciliation_run", None)
        if callable(latest_fn):
            try:
                latest_run = latest_fn(tenant_id=tenant)
            except Exception:
                pass
        if latest_run:
            run_id = str(latest_run.get("run_id") or "")
            if run_id:
                try:
                    sql = f"""
                    SELECT severity, issue_type, entity_key, expected_json, actual_json, detail
                    FROM `{deps.repo.dataset_fqn}.reconciliation_issues`
                    WHERE tenant_id = @tenant_id AND run_id = @run_id
                    ORDER BY severity DESC
                    LIMIT 100
                    """
                    issues = deps.repo.fetch_rows(sql, {"tenant_id": tenant, "run_id": run_id})
                except Exception:
                    pass

        return deps.json_response(build_recon_dashboard_data(latest_run, issues), max_age=60)

    @app.get("/api/capital/waterfall")
    def api_capital_waterfall(
        request: Request,
        tenant_id: str = Query(default="", description="tenant id"),
        agent_id: str = Query(default="", description="agent id"),
    ) -> JSONResponse:
        tenant, scoped_agent_ids, _, redirect = deps.resolve_viewer_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/capital/waterfall?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)

        aid = agent_id.strip().lower()
        if not aid or aid not in scoped_agent_ids:
            aid = scoped_agent_ids[0] if scoped_agent_ids else ""
        if not aid:
            return deps.json_response(build_event_timeline_data("", 0, [], 0), max_age=60)

        tenant_settings = deps.settings_for_tenant(tenant)
        is_live = deps.is_live_mode(tenant_settings)
        live_sources = deps.live_market_sources(tenant_settings) if is_live else None

        build_fn = getattr(deps.repo, "build_agent_sleeve_snapshot", None)
        if not callable(build_fn):
            return deps.json_response(build_event_timeline_data(aid, 0, [], 0), max_age=60)

        try:
            snap, baseline, meta = build_fn(
                agent_id=aid,
                sources=live_sources,
                include_simulated=not is_live,
                tenant_id=tenant,
            )
        except Exception:
            return deps.json_response(build_event_timeline_data(aid, 0, [], 0), max_age=60)

        meta = dict(meta) if isinstance(meta, dict) else {}
        actual_trace_fn = getattr(deps.repo, "trace_agent_actual_capital_basis", None)
        actual_trace = {}
        if callable(actual_trace_fn):
            try:
                actual_trace = actual_trace_fn(agent_id=aid, tenant_id=tenant)
            except Exception as exc:
                logger.warning("Actual capital trace failed for %s: %s", aid, exc)
                actual_trace = {}

        initial_cash = float((actual_trace or {}).get("seed_cash_krw") or meta.get("seed_cash_krw") or 0.0)
        seed_positions_cost = float((actual_trace or {}).get("seed_positions_cost_krw") or meta.get("seed_positions_cost_krw") or 0.0)
        actual_baseline = float((actual_trace or {}).get("baseline_equity_krw") or baseline or 0.0)
        current_positions_value = float(meta.get("current_positions_value_krw") or max(float(snap.total_equity_krw) - float(snap.cash_krw), 0.0))
        raw_seed_at = (actual_trace or {}).get("origin_at") or meta.get("initialized_at")
        seed_at = None
        if isinstance(raw_seed_at, datetime):
            seed_at = raw_seed_at
        elif raw_seed_at is not None:
            try:
                seed_at = datetime.fromisoformat(str(raw_seed_at).replace("Z", "+00:00"))
            except (ValueError, TypeError):
                seed_at = None
        if seed_at is None:
            seed_at = datetime(2020, 1, 1, tzinfo=timezone.utc)

        # Query capital-basis events from the canonical sleeve seed.
        events: list[dict[str, Any]] = []
        try:
            cap_fn = getattr(deps.repo, "capital_events_since", None)
            if callable(cap_fn):
                cap_rows = cap_fn(agent_id=aid, since=seed_at, tenant_id=tenant)
                for r in cap_rows:
                    etype = str(r.get("event_type") or "").lower()
                    amt = float(r.get("amount_krw") or 0)
                    d_str = _event_date_str(r.get("occurred_at"))
                    if "withdraw" in etype:
                        events.append({"date": d_str, "type": "withdrawal", "label": f"출금 {abs(amt):,.0f}원", "amount": round(-abs(amt))})
                    elif amt != 0:
                        events.append({"date": d_str, "type": "deposit", "label": f"입금 {abs(amt):,.0f}원", "amount": round(abs(amt))})

            manual_fn = getattr(deps.repo, "manual_cash_adjustments_since", None)
            if callable(manual_fn):
                manual_rows = manual_fn(agent_id=aid, since=seed_at, tenant_id=tenant)
                for row in manual_rows:
                    delta_cash = float(row.get("delta_cash_krw") or 0.0)
                    if abs(delta_cash) <= 1e-9:
                        continue
                    d_str = _event_date_str(row.get("occurred_at"))
                    sign = "+" if delta_cash > 0 else ""
                    events.append({
                        "date": d_str,
                        "type": "manual_adjustment",
                        "label": f"수동 조정 {sign}{delta_cash:,.0f}원",
                        "amount": round(delta_cash),
                    })

            transfer_fn = getattr(deps.repo, "agent_transfer_events_since", None)
            if callable(transfer_fn):
                transfer_rows = transfer_fn(agent_id=aid, since=seed_at, tenant_id=tenant)
                for row in transfer_rows:
                    if not _is_cash_transfer_event(row):
                        continue
                    delta_cash = _signed_transfer_cash_amount(aid, row)
                    if abs(delta_cash) <= 1e-9:
                        continue
                    other_raw = row.get("from_agent_id") if delta_cash > 0 else row.get("to_agent_id")
                    other_agent = str(other_raw or "").strip()
                    counterparty = f" ({other_agent})" if other_agent else ""
                    direction = "현금 이체 유입" if delta_cash > 0 else "현금 이체 유출"
                    d_str = _event_date_str(row.get("occurred_at"))
                    events.append({
                        "date": d_str,
                        "type": "transfer",
                        "label": f"{direction}{counterparty}",
                        "amount": round(delta_cash),
                    })

            events.sort(key=lambda e: ((e.get("date") or ""), str(e.get("label") or "")))

        except Exception as exc:
            logger.warning("Event timeline query failed for %s: %s", aid, exc)

        # Build holdings breakdown for doughnut chart
        holdings_labels = ["Cash"]
        holdings_data = [float(meta.get("current_cash_krw") or snap.cash_krw)]
        for ticker_s, pos in sorted(snap.positions.items()):
            if pos.quantity <= 0:
                continue
            holdings_labels.append(ticker_s)
            holdings_data.append(pos.quantity * pos.avg_price_krw)

        timeline = build_event_timeline_data(
            aid,
            initial_cash,
            events,
            snap.total_equity_krw,
            baseline_equity=actual_baseline,
            seed_positions_cost_krw=seed_positions_cost,
            capital_flow_krw=float((actual_trace or {}).get("capital_flow_krw") or meta.get("capital_flow_krw") or 0.0),
            capital_event_count=int((actual_trace or {}).get("capital_event_count") or meta.get("capital_event_count") or 0),
            transfer_equity_krw=float((actual_trace or {}).get("transfer_equity_krw") or meta.get("transfer_equity_krw") or 0.0),
            transfer_event_count=int((actual_trace or {}).get("transfer_event_count") or meta.get("transfer_event_count") or 0),
            manual_cash_adjustment_krw=float((actual_trace or {}).get("manual_cash_adjustment_krw") or meta.get("manual_cash_adjustment_krw") or 0.0),
            manual_cash_adjustment_count=int((actual_trace or {}).get("manual_cash_adjustment_count") or meta.get("manual_cash_adjustment_count") or 0),
            dividend_income_krw=float(meta.get("dividend_income_krw") or 0.0),
            current_cash_krw=float(meta.get("current_cash_krw") or snap.cash_krw),
            current_positions_value_krw=current_positions_value,
            seed_source=str((actual_trace or {}).get("origin_source") or meta.get("seed_source") or ""),
            initialized_at=seed_at.isoformat(),
        )
        timeline["holdings"] = {"labels": holdings_labels, "data": holdings_data}

        return deps.json_response(timeline, max_age=60)
