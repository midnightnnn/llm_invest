from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse


@dataclass(frozen=True)
class ViewerRouteDeps:
    repo: Any
    executor: Any
    auth_enabled: bool
    kst: Any
    resolve_viewer_context: Callable[..., Any]
    cached_fetch: Callable[..., Any]
    current_user: Callable[[Request], dict[str, Any] | None]
    tailwind_layout: Callable[..., str]
    html_response: Callable[..., HTMLResponse]
    json_response: Callable[..., JSONResponse]
    get_default_registry: Callable[[str], Any]
    settings_for_tenant: Callable[[str], Any]
    latest_tenant_status_payload: Callable[[str], dict[str, Any] | None]
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
    default_benchmark: Callable[[Any | None], str]
    is_live_mode: Callable[[Any | None], bool]
    metric_card: Callable[..., str]
    fmt_ts: Callable[[object], str]
    md_block: Callable[..., str]
    safe_float: Callable[[object, float], float]
    safe_int: Callable[[object, int], int]
    to_date: Callable[[object], str]
    chained_index: Callable[[list[float | None], list[float | None], list[float | None]], list[float | None]]
    drawdown: Callable[[list[float | None]], list[float | None]]
    total_return: Callable[[list[float | None]], float]
    max_drawdown: Callable[[list[float | None]], float]
