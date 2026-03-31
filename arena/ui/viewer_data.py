from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from arena.config import Settings
from arena.ui.viewer_analytics import render_pnl_badge


@dataclass(frozen=True)
class ViewerDataHelpers:
    fetch_board: Callable[..., list[dict[str, Any]]]
    fetch_tool_events_for_post: Callable[..., dict[str, Any]]
    fetch_theses_for_board_post: Callable[..., dict[str, Any]]
    fetch_nav: Callable[..., list[dict[str, Any]]]
    fetch_token_usage_summary: Callable[..., dict[str, dict[str, int | float]]]
    fetch_trade_count_summary: Callable[..., dict[str, int]]
    fetch_token_usage_daily: Callable[..., list[dict[str, Any]]]
    fetch_trade_count_daily: Callable[..., list[dict[str, Any]]]
    fetch_trades: Callable[..., list[dict[str, Any]]]
    fetch_sleeves: Callable[..., list[dict[str, Any]]]
    fetch_sleeve_snapshot_cards: Callable[..., dict[str, Any]]
    fetch_trades_for_board_post: Callable[..., list[dict[str, Any]]]
    default_benchmark: Callable[[Settings | None], str]


def build_viewer_data_helpers(
    *,
    repo: Any,
    settings: Settings,
    executor: Any,
    settings_for_tenant: Callable[[str], Settings],
    is_live_mode: Callable[[Settings | None], bool],
    live_market_sources: Callable[[Settings | None], list[str] | None],
    agent_logo_svg: Callable[[str], str],
    safe_float: Callable[[object, float], float],
    safe_int: Callable[[object, int], int],
    to_date: Callable[[object], str],
) -> ViewerDataHelpers:
    def _json_dict(raw: object) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _iso_ts(value: object) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.isoformat()
        return str(value or "").strip()

    def fetch_board(
        *,
        tenant_id: str,
        limit: int,
        offset: int = 0,
        agent_id: str | None = None,
        agent_ids: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        filters: list[str] = ["tenant_id = @tenant_id"]
        params: dict[str, Any] = {"tenant_id": tenant, "limit": limit, "offset": offset}
        if agent_ids:
            filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
            params["agent_ids"] = agent_ids
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = agent_id
        if start_date:
            filters.append("DATE(created_at, 'Asia/Seoul') >= CAST(@start_date AS DATE)")
            params["start_date"] = start_date
        if end_date:
            filters.append("DATE(created_at, 'Asia/Seoul') <= CAST(@end_date AS DATE)")
            params["end_date"] = end_date
        where = "WHERE " + " AND ".join(filters)
        sql = f"""
        SELECT post_id, created_at, agent_id, title, body, tickers, cycle_id
        FROM `{repo.dataset_fqn}.board_posts`
        {where}
        ORDER BY created_at DESC
        LIMIT @limit OFFSET @offset
        """
        return repo.fetch_rows(sql, params)

    def fetch_tool_events_for_post(
        *,
        tenant_id: str,
        agent_id: str,
        ts_iso: str,
    ) -> dict[str, Any]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        try:
            ts_dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return {}
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        sql = f"""
        SELECT payload_json
        FROM `{repo.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND event_type = 'react_tools_summary'
          AND created_at BETWEEN TIMESTAMP_SUB(@ts, INTERVAL 5 MINUTE)
                              AND TIMESTAMP_ADD(@ts, INTERVAL 5 MINUTE)
        ORDER BY ABS(TIMESTAMP_DIFF(created_at, @ts, SECOND)) ASC
        LIMIT 1
        """
        rows = repo.fetch_rows(sql, {"tenant_id": tenant, "agent_id": agent_id, "ts": ts_dt})
        if not rows:
            return {}
        raw = rows[0].get("payload_json")
        if not raw:
            return {}
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return {}
        return {
            "tool_events": payload.get("tool_events", []),
            "tool_mix": payload.get("tool_mix", {}),
        }

    def fetch_theses_for_board_post(
        *,
        tenant_id: str,
        cycle_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        cycle_token = str(cycle_id or "").strip()
        if not cycle_token:
            return {"chains": []}
        filters = [
            "tenant_id = @tenant_id",
            "("
            "COALESCE(JSON_VALUE(payload_json, '$.cycle_id'), '') = @cycle_id "
            "OR COALESCE(JSON_VALUE(payload_json, '$.intent.cycle_id'), '') = @cycle_id"
            ")",
            "event_type IN UNNEST(@event_types)",
        ]
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "cycle_id": cycle_token,
            "event_types": [
                "thesis_open",
                "thesis_update",
                "thesis_invalidated",
                "thesis_realized",
            ],
        }
        agent_token = str(agent_id or "").strip().lower()
        if agent_token:
            filters.append("LOWER(agent_id) = @agent_id")
            params["agent_id"] = agent_token
        where = " AND ".join(filters)
        event_rows = repo.fetch_rows(
            f"""
            SELECT created_at, agent_id, event_type, summary, semantic_key, payload_json
            FROM `{repo.dataset_fqn}.agent_memory_events`
            WHERE {where}
            ORDER BY created_at ASC
            LIMIT 200
            """,
            params,
        )
        state_map = {
            "thesis_open": "open",
            "thesis_update": "updated",
            "thesis_invalidated": "invalidated",
            "thesis_realized": "realized",
        }
        chains_by_id: dict[str, dict[str, Any]] = {}
        for row in event_rows:
            payload = _json_dict(row.get("payload_json"))
            thesis_id = str(payload.get("thesis_id") or row.get("semantic_key") or "").strip()
            if not thesis_id:
                continue
            event_type = str(row.get("event_type") or "").strip().lower()
            created_at = _iso_ts(row.get("created_at"))
            chain = chains_by_id.setdefault(
                thesis_id,
                {
                    "thesis_id": thesis_id,
                    "agent_id": str(row.get("agent_id") or "").strip().lower(),
                    "ticker": str(payload.get("ticker") or "").strip().upper(),
                    "side": str(payload.get("side") or "").strip().upper(),
                    "state": "",
                    "thesis_summary": str(payload.get("thesis_summary") or row.get("summary") or "").strip(),
                    "strategy_refs": list(payload.get("strategy_refs") or []),
                    "terminal_event_type": "",
                    "last_event_at": "",
                    "events": [],
                    "reflection": None,
                },
            )
            if not chain["ticker"]:
                chain["ticker"] = str(payload.get("ticker") or "").strip().upper()
            if not chain["side"]:
                chain["side"] = str(payload.get("side") or "").strip().upper()
            if not chain["thesis_summary"]:
                chain["thesis_summary"] = str(payload.get("thesis_summary") or row.get("summary") or "").strip()
            if not chain["strategy_refs"] and isinstance(payload.get("strategy_refs"), list):
                chain["strategy_refs"] = list(payload.get("strategy_refs") or [])
            chain["state"] = str(payload.get("state") or state_map.get(event_type) or chain["state"]).strip().lower()
            chain["terminal_event_type"] = event_type
            chain["last_event_at"] = created_at or chain["last_event_at"]
            chain["events"].append(
                {
                    "event_type": event_type,
                    "created_at": created_at,
                    "summary": str(row.get("summary") or "").strip(),
                    "state": str(payload.get("state") or state_map.get(event_type) or "").strip().lower(),
                }
            )
        if not chains_by_id:
            return {"chains": []}

        thesis_ids = list(chains_by_id.keys())
        reflection_filters = [
            "tenant_id = @tenant_id",
            "event_type = 'strategy_reflection'",
            "JSON_VALUE(payload_json, '$.source') = 'thesis_chain_compaction'",
            "JSON_VALUE(payload_json, '$.thesis_id') IN UNNEST(@thesis_ids)",
        ]
        reflection_params: dict[str, Any] = {
            "tenant_id": tenant,
            "thesis_ids": thesis_ids,
        }
        if agent_token:
            reflection_filters.append("LOWER(agent_id) = @agent_id")
            reflection_params["agent_id"] = agent_token
        reflection_rows = repo.fetch_rows(
            f"""
            SELECT created_at, summary, payload_json
            FROM `{repo.dataset_fqn}.agent_memory_events`
            WHERE {' AND '.join(reflection_filters)}
            ORDER BY created_at DESC
            LIMIT 200
            """,
            reflection_params,
        )
        for row in reflection_rows:
            payload = _json_dict(row.get("payload_json"))
            thesis_id = str(payload.get("thesis_id") or "").strip()
            if not thesis_id or thesis_id not in chains_by_id:
                continue
            chain = chains_by_id[thesis_id]
            if chain.get("reflection"):
                continue
            chain["reflection"] = {
                "created_at": _iso_ts(row.get("created_at")),
                "summary": str(row.get("summary") or "").strip(),
                "source": str(payload.get("source") or "").strip(),
            }

        chains = sorted(
            chains_by_id.values(),
            key=lambda item: str(item.get("last_event_at") or ""),
            reverse=True,
        )
        return {"chains": chains}

    def fetch_nav(
        *,
        tenant_id: str,
        days: int,
        agent_id: str | None = None,
        agent_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        start = datetime.now(timezone.utc).date() - timedelta(days=int(days))
        scope_count = len(agent_ids) if agent_ids else len(settings.agent_ids)
        limit = max(500, min(int(days) * max(scope_count, 1) + 200, 30000))
        loader = getattr(repo, "fetch_agent_nav_history", None)
        if callable(loader):
            rows = loader(
                tenant_id=tenant,
                agent_id=agent_id,
                agent_ids=agent_ids,
                limit=limit,
            )
            start_str = start.isoformat() if isinstance(start, date) else str(start)[:10]
            return [row for row in rows if to_date(row.get("nav_date")) >= start_str]

        filters = ["tenant_id = @tenant_id", "nav_date >= @start"]
        params: dict[str, Any] = {"tenant_id": tenant, "start": start, "limit": limit}
        if agent_ids:
            filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
            params["agent_ids"] = agent_ids
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = agent_id
        where = " AND ".join(filters)
        sql = f"""
        SELECT nav_date, agent_id, nav_krw, pnl_krw, pnl_ratio
        FROM `{repo.dataset_fqn}.agent_nav_daily`
        WHERE {where}
        ORDER BY nav_date ASC, agent_id ASC
        LIMIT @limit
        """
        return repo.fetch_rows(sql, params)

    def fetch_token_usage_summary(
        *,
        tenant_id: str,
        days: int,
        agent_id: str | None = None,
        agent_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int | float]]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        start = datetime.now(timezone.utc) - timedelta(days=int(days))
        filters = [
            "tenant_id = @tenant_id",
            "event_type = 'react_tools_summary'",
            "created_at >= @start",
        ]
        params: dict[str, Any] = {"tenant_id": tenant, "start": start}
        if agent_ids:
            filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
            params["agent_ids"] = agent_ids
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = agent_id
        where = " AND ".join(filters)
        sql = f"""
        SELECT
          LOWER(agent_id) AS agent_id,
          SUM(COALESCE(SAFE_CAST(JSON_VALUE(payload_json, '$.token_usage.llm_calls') AS INT64), 0)) AS llm_calls,
          SUM(COALESCE(SAFE_CAST(JSON_VALUE(payload_json, '$.token_usage.prompt_tokens') AS INT64), 0)) AS prompt_tokens,
          SUM(COALESCE(SAFE_CAST(JSON_VALUE(payload_json, '$.token_usage.completion_tokens') AS INT64), 0)) AS completion_tokens,
          SUM(COALESCE(SAFE_CAST(JSON_VALUE(payload_json, '$.token_usage.cached_tokens') AS INT64), 0)) AS cached_tokens,
          SUM(COALESCE(SAFE_CAST(JSON_VALUE(payload_json, '$.token_usage.thinking_tokens') AS INT64), 0)) AS thinking_tokens
        FROM `{repo.dataset_fqn}.agent_memory_events`
        WHERE {where}
        GROUP BY agent_id
        """
        rows = repo.fetch_rows(sql, params)
        stats: dict[str, dict[str, int | float]] = {}
        for row in rows:
            agent = str(row.get("agent_id") or "").strip().lower()
            if not agent:
                continue
            stats[agent] = {
                "llm_calls": safe_int(row.get("llm_calls"), 0),
                "prompt_tokens": safe_int(row.get("prompt_tokens"), 0),
                "completion_tokens": safe_int(row.get("completion_tokens"), 0),
                "cached_tokens": safe_int(row.get("cached_tokens"), 0),
                "thinking_tokens": safe_int(row.get("thinking_tokens"), 0),
                "total_tokens": 0,
                "cache_ratio": 0.0,
            }

        for bucket in stats.values():
            prompt_tokens = safe_int(bucket.get("prompt_tokens"), 0)
            completion_tokens = safe_int(bucket.get("completion_tokens"), 0)
            thinking_tokens = safe_int(bucket.get("thinking_tokens"), 0)
            cached_tokens = safe_int(bucket.get("cached_tokens"), 0)
            bucket["total_tokens"] = prompt_tokens + completion_tokens + thinking_tokens
            bucket["cache_ratio"] = round(cached_tokens / prompt_tokens * 100.0, 1) if prompt_tokens > 0 else 0.0
        return stats

    def fetch_trade_count_summary(
        *,
        tenant_id: str,
        days: int,
        agent_id: str | None = None,
        agent_ids: list[str] | None = None,
    ) -> dict[str, int]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        start = datetime.now(timezone.utc) - timedelta(days=int(days))
        filters = [
            "tenant_id = @tenant_id",
            "created_at >= @start",
            "status IN UNNEST(@statuses)",
        ]
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "start": start,
            "statuses": ["FILLED", "SIMULATED"],
        }
        if agent_ids:
            filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
            params["agent_ids"] = agent_ids
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = agent_id
        where = " AND ".join(filters)
        sql = f"""
        SELECT
          LOWER(agent_id) AS agent_id,
          COUNT(DISTINCT intent_id) AS trade_count
        FROM `{repo.dataset_fqn}.execution_reports`
        WHERE {where}
        GROUP BY agent_id
        """
        rows = repo.fetch_rows(sql, params)
        counts: dict[str, int] = {}
        for row in rows:
            agent = str(row.get("agent_id") or "").strip().lower()
            if not agent:
                continue
            counts[agent] = safe_int(row.get("trade_count"), 0)
        return counts

    def fetch_token_usage_daily(
        *,
        tenant_id: str,
        days: int,
        agent_id: str | None = None,
        agent_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        start = datetime.now(timezone.utc) - timedelta(days=int(days))
        filters = [
            "tenant_id = @tenant_id",
            "event_type = 'react_tools_summary'",
            "created_at >= @start",
        ]
        params: dict[str, Any] = {"tenant_id": tenant, "start": start}
        if agent_ids:
            filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
            params["agent_ids"] = agent_ids
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = agent_id
        where = " AND ".join(filters)
        sql = f"""
        SELECT
          DATE(created_at, 'Asia/Seoul') AS usage_date,
          LOWER(agent_id) AS agent_id,
          SUM(
            COALESCE(SAFE_CAST(JSON_VALUE(payload_json, '$.token_usage.prompt_tokens') AS INT64), 0) +
            COALESCE(SAFE_CAST(JSON_VALUE(payload_json, '$.token_usage.completion_tokens') AS INT64), 0) +
            COALESCE(SAFE_CAST(JSON_VALUE(payload_json, '$.token_usage.thinking_tokens') AS INT64), 0)
          ) AS total_tokens
        FROM `{repo.dataset_fqn}.agent_memory_events`
        WHERE {where}
        GROUP BY usage_date, agent_id
        ORDER BY usage_date ASC
        """
        return repo.fetch_rows(sql, params)

    def fetch_trade_count_daily(
        *,
        tenant_id: str,
        days: int,
        agent_id: str | None = None,
        agent_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        start = datetime.now(timezone.utc) - timedelta(days=int(days))
        filters = [
            "tenant_id = @tenant_id",
            "created_at >= @start",
            "status IN UNNEST(@statuses)",
        ]
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "start": start,
            "statuses": ["FILLED", "SIMULATED"],
        }
        if agent_ids:
            filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
            params["agent_ids"] = agent_ids
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = agent_id
        where = " AND ".join(filters)
        sql = f"""
        SELECT
          DATE(created_at, 'Asia/Seoul') AS trade_date,
          LOWER(agent_id) AS agent_id,
          COUNT(DISTINCT intent_id) AS trade_count
        FROM `{repo.dataset_fqn}.execution_reports`
        WHERE {where}
        GROUP BY trade_date, agent_id
        ORDER BY trade_date ASC
        """
        return repo.fetch_rows(sql, params)

    def fetch_trades(
        *,
        tenant_id: str,
        limit: int,
        days: int,
        offset: int = 0,
        agent_id: str | None = None,
        ticker: str | None = None,
        agent_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        start = datetime.now(timezone.utc) - timedelta(days=int(days))
        filters = ["tenant_id = @tenant_id", "created_at >= @start"]
        params: dict[str, Any] = {"tenant_id": tenant, "start": start, "limit": limit, "offset": offset}
        if agent_ids:
            filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
            params["agent_ids"] = agent_ids
        if agent_id:
            filters.append("agent_id = @agent_id")
            params["agent_id"] = agent_id
        if ticker:
            filters.append("ticker = @ticker")
            params["ticker"] = ticker
        where = " AND ".join(filters)
        sql = f"""
        SELECT created_at, agent_id, ticker, side, requested_qty, filled_qty, avg_price_krw, status, message
        FROM `{repo.dataset_fqn}.execution_reports`
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT @limit OFFSET @offset
        """
        return repo.fetch_rows(sql, params)

    def fetch_sleeves(*, tenant_id: str, agent_ids: list[str] | None = None) -> list[dict[str, Any]]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        filters: list[str] = ["tenant_id = @tenant_id"]
        params: dict[str, Any] = {"tenant_id": tenant}
        if agent_ids:
            filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
            params["agent_ids"] = agent_ids
        where = "WHERE " + " AND ".join(filters)
        sql = f"""
        SELECT agent_id, initialized_at, initial_cash_krw, initial_positions_json
        FROM (
          SELECT agent_id, initialized_at, initial_cash_krw, initial_positions_json,
                 ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY initialized_at DESC) AS rn
          FROM `{repo.dataset_fqn}.agent_sleeves`
          {where}
        )
        WHERE rn = 1
        ORDER BY agent_id
        """
        return repo.fetch_rows(sql, params)

    def fetch_sleeve_snapshot_cards(
        *,
        tenant_id: str,
        agent_ids: list[str],
        compact: bool = False,
    ) -> dict[str, Any]:
        def fmt_qty(qty: object) -> str:
            try:
                q = float(qty or 0.0)
            except (TypeError, ValueError):
                return "0"
            if abs(q - round(q)) < 1e-9:
                return f"{int(round(q)):,}"
            return f"{q:,.4f}".rstrip("0").rstrip(".")

        tenant_settings = settings_for_tenant(tenant_id)
        is_live = is_live_mode(tenant_settings)
        live_sources = live_market_sources(tenant_settings) if is_live else None
        snapshot_cards_html = ""
        chart_specs: list[dict[str, Any]] = []
        def build_one(aid: str) -> tuple[str, dict[str, Any] | None]:
            try:
                snap, baseline, _meta = repo.build_agent_sleeve_snapshot(
                    agent_id=aid,
                    sources=live_sources,
                    include_simulated=not is_live,
                    tenant_id=tenant_id,
                )
                actual_basis = float(baseline or 0.0)
                actual_basis_loader = getattr(repo, "trace_agent_actual_capital_basis", None)
                if callable(actual_basis_loader):
                    traced = actual_basis_loader(agent_id=aid, tenant_id=tenant_id)
                    traced_basis = safe_float((traced or {}).get("baseline_equity_krw"), 0.0)
                    if traced_basis > 0:
                        actual_basis = traced_basis
                equity_str = f"{snap.total_equity_krw:,.0f}"
                pnl = snap.total_equity_krw - actual_basis
                pnl_pct = (pnl / actual_basis * 100) if actual_basis > 0 else 0.0
                pnl_badge = render_pnl_badge(
                    pnl_krw=pnl,
                    pnl_pct=pnl_pct,
                )
                position_count = sum(1 for p in snap.positions.values() if p.quantity > 0)

                # --- Compact card (no doughnut, just numbers) ---
                if compact:
                    card_html = (
                        f'<div class="reveal flex items-center gap-4 rounded-[20px] border border-ink-200/60 bg-white/80 px-5 py-4 shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-0.5 hover:shadow-md">'
                        f'<div class="flex items-center gap-2.5 min-w-0">'
                        f'{agent_logo_svg(aid)}'
                        f'<h4 class="font-display text-base font-bold text-ink-900 tracking-tight truncate">{html.escape(aid)}</h4>'
                        f'</div>'
                        f'<div class="flex items-center gap-5 ml-auto">'
                        f'<div class="text-right">'
                        f'<p class="text-[10px] font-medium uppercase tracking-widest text-ink-400">평가금</p>'
                        f'<p class="font-display text-lg font-bold text-ink-900 tracking-tight">{equity_str}</p>'
                        f'</div>'
                        f'<div class="text-right">{pnl_badge}</div>'
                        f'<div class="text-right">'
                        f'<p class="text-[10px] font-medium uppercase tracking-widest text-ink-400">종목</p>'
                        f'<p class="font-display text-lg font-bold text-ink-900 tracking-tight">{position_count}</p>'
                        f'</div>'
                        f'<a href="/settings?tab=capital" class="rounded-full border border-ink-200 px-3 py-1.5 text-[11px] font-semibold text-ink-600 hover:bg-ink-50 hover:border-ink-300 transition-all whitespace-nowrap">상세 &rarr;</a>'
                        f'</div></div>'
                    )
                    return card_html, None

                # --- Full card (with doughnut chart) ---
                chart_labels = ["Cash"]
                chart_data = [snap.cash_krw]
                pos_html = ""
                for ticker_s, pos in sorted(snap.positions.items()):
                    if pos.quantity <= 0:
                        continue
                    chart_labels.append(ticker_s)
                    chart_data.append(pos.quantity * pos.avg_price_krw)
                    pos_html += (
                        f'<div class="flex items-center justify-between py-1.5 border-b border-ink-50 last:border-0">'
                        f'<span class="font-medium text-sm">{html.escape(ticker_s)}</span>'
                        f'<span class="text-xs text-ink-500">{fmt_qty(pos.quantity)}주 @ {safe_float(pos.avg_price_krw, 0.0):,.0f}</span>'
                        "</div>"
                    )

                cid = f"nav_sleeve_chart_{re.sub(r'[^a-zA-Z0-9]', '_', aid)}"
                card_html = (
                    f'<div class="reveal flex flex-col rounded-[24px] border border-ink-200/60 bg-white/80 p-6 shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">'
                    f'<div class="mb-4 flex items-center justify-between"><div class="flex items-center gap-2">{agent_logo_svg(aid)}<h4 class="font-display text-lg font-bold text-ink-900 tracking-tight">{html.escape(aid)}</h4></div>{pnl_badge}</div>'
                    f'<div class="grid grid-cols-2 gap-4 flex-grow">'
                    f'<div class="flex flex-col items-center justify-center">'
                    f'<div class="relative w-full aspect-square max-w-[140px]"><canvas id="{cid}"></canvas></div>'
                    f'<div class="mt-4 text-center"><p class="text-[10px] font-medium uppercase tracking-widest text-ink-400">Total Equity</p><p class="font-display text-xl font-bold text-ink-900 tracking-tight">{equity_str}</p></div>'
                    f'</div>'
                    f'<div class="flex flex-col">'
                    f'<p class="mb-2 text-[10px] font-medium uppercase tracking-widest text-ink-400">Positions</p>'
                    f'<div class="flex-grow overflow-y-auto max-h-[160px] pr-1 styled-scrollbar">{pos_html or "<p class=\'text-xs italic text-ink-400 mt-2\'>No positions</p>"}</div>'
                    f'</div></div></div>'
                )
                return card_html, {"id": cid, "labels": chart_labels, "data": chart_data}
            except Exception as exc:
                return (
                    f'<div class="reveal flex flex-col items-center justify-center rounded-[24px] border border-red-200 bg-red-50/80 p-6 shadow-sm backdrop-blur-md">'
                    f"<h4 class='mb-2 font-display text-lg font-bold text-red-900 tracking-tight'>{html.escape(aid)}</h4>"
                    f"<p class='text-xs text-red-600 text-center'>Error: {html.escape(str(exc))}</p></div>",
                    None,
                )

        futures = {aid: executor.submit(build_one, aid) for aid in agent_ids}
        for aid in agent_ids:
            card_html, chart_spec = futures[aid].result()
            snapshot_cards_html += card_html
            if chart_spec:
                chart_specs.append(chart_spec)

        if not snapshot_cards_html:
            snapshot_cards_html = '<div class="col-span-full rounded-[24px] border border-ink-200 bg-white/80 p-8 text-center text-sm text-ink-500 shadow-sm backdrop-blur-md">포트폴리오 데이터가 없습니다.</div>'
        return {"html": snapshot_cards_html, "charts": chart_specs}

    def fetch_trades_for_board_post(
        *,
        tenant_id: str,
        cycle_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = str(tenant_id or "").strip().lower() or "local"
        token = str(cycle_id or "").strip()
        if not token:
            return []
        filters = ["tenant_id = @tenant_id", "cycle_id = @cycle_id"]
        params: dict[str, Any] = {"tenant_id": tenant, "cycle_id": token}
        agent_token = str(agent_id or "").strip()
        if agent_token:
            filters.append("LOWER(agent_id) = LOWER(@agent_id)")
            params["agent_id"] = agent_token
        where = " AND ".join(filters)
        sql = f"""
        SELECT created_at, agent_id, ticker, side, requested_qty,
               filled_qty, avg_price_krw, status, message
        FROM `{repo.dataset_fqn}.execution_reports`
        WHERE {where}
        ORDER BY created_at ASC
        LIMIT 50
        """
        return repo.fetch_rows(sql, params)

    def default_benchmark(active_settings: Settings | None = None) -> str:
        runtime_settings = active_settings or settings
        if runtime_settings.kis_target_market == "nasdaq":
            return "QQQ"
        if runtime_settings.kis_target_market == "kospi":
            return "069500"
        return ""

    return ViewerDataHelpers(
        fetch_board=fetch_board,
        fetch_tool_events_for_post=fetch_tool_events_for_post,
        fetch_theses_for_board_post=fetch_theses_for_board_post,
        fetch_nav=fetch_nav,
        fetch_token_usage_summary=fetch_token_usage_summary,
        fetch_trade_count_summary=fetch_trade_count_summary,
        fetch_token_usage_daily=fetch_token_usage_daily,
        fetch_trade_count_daily=fetch_trade_count_daily,
        fetch_trades=fetch_trades,
        fetch_sleeves=fetch_sleeves,
        fetch_sleeve_snapshot_cards=fetch_sleeve_snapshot_cards,
        fetch_trades_for_board_post=fetch_trades_for_board_post,
        default_benchmark=default_benchmark,
    )
