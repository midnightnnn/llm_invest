from __future__ import annotations

import html
import os

from arena.ui.templating import render_ui_template


def md_block(value: object, *, classes: str = "") -> str:
    safe_text = html.escape(str(value or ""))
    class_attr = f"md-render {classes}".strip()
    return f'<div class="{html.escape(class_attr, quote=True)}" data-md="1">{safe_text}</div>'


def tailwind_layout(
    title: str,
    body_html: str,
    *,
    active: str = "",
    needs_charts: bool = False,
    needs_datepicker: bool = False,
    needs_echarts: bool = False,
    header_extra: str = "",
    max_width_class: str = "max-w-7xl",
    status_label: str = "",
    status_color: str = "",
    extra_nav_items: list[tuple[str, str, str]] | None = None,
    tenant: str = "",
    showcase: bool = False,
) -> str:
    if showcase:
        _t = html.escape(tenant or "")
        nav_items: list[tuple[str, str, str]] = [
            (f"/showcase/{_t}/board", "\uac8c\uc2dc\ud310", "board"),
            (f"/showcase/{_t}/nav", "\uc6b4\uc6a9\uc131\uacfc", "nav"),
            (f"/showcase/{_t}/settings?tab=agents", "\uc5d0\uc774\uc804\ud2b8", "agents"),
            (f"/showcase/{_t}/settings?tab=capital", "\uc790\ubcf8\uad00\ub9ac", "capital"),
            (f"/showcase/{_t}/settings?tab=mcp", "\ub3c4\uad6c\uad00\ub9ac", "tools"),
            (f"/showcase/{_t}/settings?tab=memory", "\uae30\uc5b5\uad00\ub9ac", "memory"),
        ]
    else:
        nav_items: list[tuple[str, str, str]] = [
            ("/board", "\uac8c\uc2dc\ud310", "board"),
            ("/nav", "\uc6b4\uc6a9\uc131\uacfc", "nav"),
            ("/settings?tab=agents", "\uc5d0\uc774\uc804\ud2b8", "agents"),
            ("/settings?tab=capital", "\uc790\ubcf8\uad00\ub9ac", "capital"),
            ("/settings?tab=mcp", "\ub3c4\uad6c\uad00\ub9ac", "tools"),
            ("/settings?tab=memory", "\uae30\uc5b5\uad00\ub9ac", "memory"),
        ]
    if extra_nav_items:
        nav_items = nav_items + list(extra_nav_items)

    nav_links = [
        {
            "href": href,
            "label": label,
            "active": key == active,
        }
        for href, label, key in nav_items
    ]
    auth_enabled = str(os.getenv("ARENA_UI_AUTH_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    safe_max_width = str(max_width_class or "max-w-7xl")

    # Status indicator color mapping
    _COLOR_MAP = {
        "emerald": ("bg-emerald-400", "bg-emerald-500", "text-emerald-600"),
        "sky": ("bg-sky-400", "bg-sky-500", "text-sky-600"),
        "amber": ("bg-amber-400", "bg-amber-500", "text-amber-600"),
        "rose": ("bg-rose-400", "bg-rose-500", "text-rose-600"),
        "indigo": ("bg-indigo-400", "bg-indigo-500", "text-indigo-600"),
    }
    _sc = status_color or "emerald"
    _ping, _dot, _txt = _COLOR_MAP.get(_sc, _COLOR_MAP["emerald"])
    _status_display = html.escape(status_label) if status_label else "Operational"
    _status_ping_color = _ping
    _status_dot_color = _dot
    _status_text_color = _txt

    return render_ui_template(
        "base_layout.jinja2",
        title=title,
        body_html=body_html,
        active=active,
        needs_charts=needs_charts,
        needs_datepicker=needs_datepicker,
        needs_echarts=needs_echarts,
        header_extra=header_extra,
        max_width_class=safe_max_width,
        nav_links=nav_links,
        auth_enabled=auth_enabled,
        showcase=showcase,
        status_display=_status_display,
        status_ping_color=_status_ping_color,
        status_dot_color=_status_dot_color,
        status_text_color=_status_text_color,
    )
