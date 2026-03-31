"""Pure transformation functions that build ECharts-ready payloads for the Capital tab.

Every function accepts already-fetched data and returns a JSON-serializable dict.
No repo or IO dependencies — easy to unit test.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Agent display colours (consistent with the rest of the UI)
# ---------------------------------------------------------------------------
AGENT_COLORS: dict[str, str] = {
    "gpt": "#10a37f",
    "gemini": "#4285f4",
    "claude": "#d97757",
}
_FALLBACK_COLORS = ["#0ea5e9", "#8b5cf6", "#22c55e", "#f59e0b", "#ec4899", "#14b8a6"]


def _agent_color(agent_id: str, idx: int = 0) -> str:
    return AGENT_COLORS.get(agent_id.lower(), _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


# ---------------------------------------------------------------------------
# Sankey — Total NAV → Agent → Cash / Positions
# ---------------------------------------------------------------------------
def build_sankey_data(
    agent_snapshots: dict[str, dict[str, Any]],
    recon_status: str = "unknown",
) -> dict[str, Any]:
    """Build ECharts Sankey nodes + links from per-agent snapshot data.

    agent_snapshots: {agent_id: {"cash_krw": float, "positions": [{ticker, market_value_krw}], "total_equity_krw": float}}
    """
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []

    nodes.append({"name": "총 NAV", "itemStyle": {"color": "#0f172a"}})

    for idx, (agent_id, snap) in enumerate(sorted(agent_snapshots.items())):
        color = _agent_color(agent_id, idx)
        total = float(snap.get("total_equity_krw") or 0)
        if total <= 0:
            continue

        nodes.append({"name": agent_id, "itemStyle": {"color": color}})
        links.append({"source": "총 NAV", "target": agent_id, "value": round(total)})

        cash = float(snap.get("cash_krw") or 0)
        if cash > 0:
            cash_name = f"{agent_id}:Cash"
            nodes.append({"name": cash_name, "itemStyle": {"color": "#94a3b8"}})
            links.append({"source": agent_id, "target": cash_name, "value": round(cash)})

        for pos in snap.get("positions") or []:
            ticker = str(pos.get("ticker") or "")
            mv = float(pos.get("market_value_krw") or 0)
            if not ticker or mv <= 0:
                continue
            pos_name = f"{agent_id}:{ticker}"
            nodes.append({"name": pos_name, "itemStyle": {"color": color + "99"}})
            links.append({"source": agent_id, "target": pos_name, "value": round(mv)})

    return {"nodes": nodes, "links": links, "recon_status": recon_status}


# ---------------------------------------------------------------------------
# Treemap — Agent → Position hierarchy
# ---------------------------------------------------------------------------
def build_treemap_data(
    agent_snapshots: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build ECharts treemap hierarchical data.

    Each position node: size=market_value, color mapped to return%.
    """
    children: list[dict[str, Any]] = []

    for idx, (agent_id, snap) in enumerate(sorted(agent_snapshots.items())):
        color = _agent_color(agent_id, idx)
        total = float(snap.get("total_equity_krw") or 0)
        if total <= 0:
            continue

        agent_children: list[dict[str, Any]] = []
        cash = float(snap.get("cash_krw") or 0)
        if cash > 0:
            agent_children.append({
                "name": "Cash",
                "value": round(cash),
                "return_pct": None,
                "itemStyle": {"color": "#94a3b8"},
            })

        for pos in snap.get("positions") or []:
            ticker = str(pos.get("ticker") or "")
            mv = float(pos.get("market_value_krw") or 0)
            if not ticker or mv <= 0:
                continue
            ret = pos.get("return_pct")
            # green (positive) to red (negative) color scale
            if ret is not None:
                ret_f = float(ret)
                if ret_f >= 0:
                    g = min(255, 160 + int(ret_f * 4))
                    pos_color = f"rgba(16,{g},129,0.85)"
                else:
                    r = min(255, 180 + int(abs(ret_f) * 4))
                    pos_color = f"rgba({r},68,68,0.85)"
            else:
                pos_color = color + "80"
            agent_children.append({
                "name": ticker,
                "value": round(mv),
                "return_pct": round(float(ret), 2) if ret is not None else None,
                "itemStyle": {"color": pos_color},
            })

        children.append({
            "name": agent_id,
            "value": round(total),
            "itemStyle": {"color": color, "borderColor": color, "borderWidth": 2},
            "children": agent_children,
        })

    return {"children": children}


# ---------------------------------------------------------------------------
# Event Timeline — ledger events + running balance
# ---------------------------------------------------------------------------
def build_event_timeline_data(
    agent_id: str,
    initial_cash: float,
    events: list[dict[str, Any]],
    current_equity: float,
    *,
    baseline_equity: float | None = None,
    seed_positions_cost_krw: float = 0.0,
    capital_flow_krw: float = 0.0,
    capital_event_count: int | None = None,
    transfer_equity_krw: float = 0.0,
    transfer_event_count: int = 0,
    manual_cash_adjustment_krw: float = 0.0,
    manual_cash_adjustment_count: int = 0,
    dividend_income_krw: float = 0.0,
    current_cash_krw: float | None = None,
    current_positions_value_krw: float | None = None,
    seed_source: str = "",
    initialized_at: str | None = None,
) -> dict[str, Any]:
    """Build event timeline: balance line chart + event log table.

    events: [{date, type, label, ticker, side, qty, price, amount, ...}]
    Returns: {agent_id, chart: {dates, balances}, events: [...], summary: {...}}
    """
    seed_cash = round(float(initial_cash or 0.0))
    seed_positions_cost = round(float(seed_positions_cost_krw or 0.0))
    running = seed_cash + seed_positions_cost
    chart_dates = ["시작"]
    chart_balances = [running]

    processed: list[dict[str, Any]] = []
    for ev in events:
        amount = round(float(ev.get("amount") or 0))
        running += amount
        date_str = str(ev.get("date") or "")
        ev_type = str(ev.get("type") or "")
        label = str(ev.get("label") or "")

        processed.append({
            "date": date_str,
            "type": ev_type,
            "label": label,
            "amount": amount,
            "balance": running,
        })
        chart_dates.append(date_str[5:] if len(date_str) >= 7 else date_str)
        chart_balances.append(running)

    total_invested = round(float(baseline_equity)) if baseline_equity is not None else running
    current_equity_rounded = round(float(current_equity or 0.0))
    current_cash = round(float(current_cash_krw)) if current_cash_krw is not None else None
    current_positions_value = (
        round(float(current_positions_value_krw))
        if current_positions_value_krw is not None
        else max(current_equity_rounded - round(float(current_cash_krw or 0.0)), 0)
        if current_cash_krw is not None
        else None
    )
    pnl_krw = current_equity_rounded - total_invested
    pnl_ratio_pct = (float(pnl_krw) / float(total_invested) * 100.0) if total_invested > 0 else None
    transfer_and_adjustment_krw = round(float(transfer_equity_krw or 0.0) + float(manual_cash_adjustment_krw or 0.0))
    transfer_and_adjustment_count = int(transfer_event_count or 0) + int(manual_cash_adjustment_count or 0)
    summary = {
        "seed_cash_krw": seed_cash,
        "seed_positions_cost_krw": seed_positions_cost,
        "capital_flow_krw": round(float(capital_flow_krw or 0.0)),
        "capital_event_count": int(capital_event_count if capital_event_count is not None else len(processed)),
        "transfer_equity_krw": round(float(transfer_equity_krw or 0.0)),
        "transfer_event_count": int(transfer_event_count or 0),
        "manual_cash_adjustment_krw": round(float(manual_cash_adjustment_krw or 0.0)),
        "manual_cash_adjustment_count": int(manual_cash_adjustment_count or 0),
        "transfer_and_adjustment_krw": transfer_and_adjustment_krw,
        "transfer_and_adjustment_count": transfer_and_adjustment_count,
        "dividend_income_krw": round(float(dividend_income_krw or 0.0)),
        "baseline_equity_krw": total_invested,
        "current_equity_krw": current_equity_rounded,
        "current_cash_krw": current_cash,
        "current_positions_value_krw": current_positions_value,
        "pnl_krw": pnl_krw,
        "pnl_ratio_pct": round(float(pnl_ratio_pct), 2) if pnl_ratio_pct is not None else None,
        "seed_source": str(seed_source or ""),
        "initialized_at": initialized_at,
    }

    return {
        "agent_id": agent_id,
        "initial_cash": seed_cash,
        "seed_positions_cost_krw": seed_positions_cost,
        "total_invested": total_invested,
        "current_equity": current_equity_rounded,
        "chart": {"dates": chart_dates, "balances": chart_balances},
        "events": processed,
        "event_count": len(processed),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Recon Dashboard — Gauge + Issues
# ---------------------------------------------------------------------------
def build_recon_dashboard_data(
    latest_run: dict[str, Any] | None,
    issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build recon dashboard payload: gauge value + issue list."""
    if not latest_run:
        return {
            "has_data": False,
            "gauge_value": 0,
            "status": "no_data",
            "run_at": None,
            "issues": [],
        }

    status = str(latest_run.get("status") or "unknown").strip().lower()
    summary = latest_run.get("summary_json")
    if isinstance(summary, str):
        import json
        try:
            summary = json.loads(summary)
        except (json.JSONDecodeError, TypeError):
            summary = {}
    if not isinstance(summary, dict):
        summary = {}

    issue_list = issues or []
    error_count = sum(1 for i in issue_list if str(i.get("severity") or "").lower() == "error")
    warning_count = sum(1 for i in issue_list if str(i.get("severity") or "").lower() == "warning")
    total_checks = max(len(issue_list), 1)

    if status == "ok" and error_count == 0:
        gauge_value = 100.0
    elif error_count == 0:
        gauge_value = max(80.0, 100.0 - warning_count * 5.0)
    else:
        gauge_value = max(0.0, 100.0 - error_count * 20.0 - warning_count * 5.0)

    formatted_issues = []
    for issue in issue_list:
        formatted_issues.append({
            "severity": str(issue.get("severity") or "info"),
            "issue_type": str(issue.get("issue_type") or ""),
            "entity_key": str(issue.get("entity_key") or ""),
            "expected": issue.get("expected_json"),
            "actual": issue.get("actual_json"),
            "detail": str(issue.get("detail") or ""),
        })

    run_at = latest_run.get("run_at")
    run_at_str = str(run_at) if run_at else None

    return {
        "has_data": True,
        "gauge_value": round(gauge_value, 1),
        "status": status,
        "run_at": run_at_str,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": formatted_issues,
    }
