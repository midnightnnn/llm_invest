from __future__ import annotations

import html
from typing import Any


def render_pnl_badge(
    *,
    pnl_krw: float,
    pnl_pct: float,
    chained_stats: dict[str, Any] | None = None,
) -> str:
    _ = chained_stats
    if pnl_krw > 0:
        return f'<span class="inline-flex items-center rounded-full bg-red-100/80 px-2.5 py-0.5 text-xs font-semibold text-red-700">+{pnl_krw:,.0f} (+{pnl_pct:.2f}%)</span>'
    if pnl_krw < 0:
        return f'<span class="inline-flex items-center rounded-full bg-blue-100/80 px-2.5 py-0.5 text-xs font-semibold text-blue-700">{pnl_krw:,.0f} ({pnl_pct:.2f}%)</span>'
    return '<span class="inline-flex items-center rounded-full bg-ink-100/80 px-2.5 py-0.5 text-xs font-semibold text-ink-700">0 (0.00%)</span>'


def chained_index(
    nav_vals: list[float | None],
    pnl_krw_vals: list[float | None],
    pnl_ratio_vals: list[float | None],
) -> list[float | None]:
    """Chains NAV returns across retargets into a baseline-100 TWR-style index."""
    out: list[float | None] = []
    cum = 1.0
    prev_baseline: float | None = None
    prev_nav: float | None = None
    for nav, pnl_k, pnl_r in zip(nav_vals, pnl_krw_vals, pnl_ratio_vals):
        if nav is None or float(nav) <= 0:
            out.append(None)
            continue
        nav_f = float(nav)
        pnl_krw_f = float(pnl_k) if pnl_k is not None else 0.0
        pnl_ratio_f = float(pnl_r) if pnl_r is not None else 0.0
        baseline = nav_f - pnl_krw_f
        if prev_nav is not None:
            if prev_baseline is not None and abs(prev_baseline) > 0 and abs(baseline - prev_baseline) / abs(prev_baseline) > 0.05:
                cum *= 1.0 + pnl_ratio_f
            else:
                cum *= nav_f / prev_nav
        out.append(100.0 * cum)
        prev_baseline = baseline
        prev_nav = nav_f
    return out


def drawdown(values: list[float | None]) -> list[float | None]:
    peak = None
    out: list[float | None] = []
    for v in values:
        if v is None:
            out.append(None)
        else:
            x = float(v)
            if peak is None or x > peak:
                peak = x
            out.append((x / peak) - 1.0 if peak and peak > 0 else 0.0)
    return out


def total_return(values: list[float | None]) -> float:
    """Returns the latest baseline-100 index value as a ratio over 100."""
    last = None
    for v in reversed(values):
        if v is not None:
            last = float(v)
            break
    if last is None or last <= 0:
        return 0.0
    return (last / 100.0) - 1.0


def max_drawdown(values: list[float | None]) -> float:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return 0.0
    if all(v <= 0 for v in xs) and any(v < 0 for v in xs):
        return min(xs)
    dd = drawdown(values)
    dd_xs = [float(v) for v in dd if v is not None]
    return min(dd_xs) if dd_xs else 0.0


def metric_card(title: str, value: str, note: str, *, value_id: str = "", note_id: str = "") -> str:
    vid = f' id="{html.escape(value_id, quote=True)}"' if value_id else ""
    nid = f' id="{html.escape(note_id, quote=True)}"' if note_id else ""
    return (
        '<div class="reveal flex flex-col rounded-[24px] border border-ink-200/60 bg-white/80 p-5 shadow-sm backdrop-blur-md transition-all duration-300 hover:-translate-y-1 hover:shadow-lg">'
        f'<p class="text-[10px] font-bold uppercase tracking-widest text-ink-500">{html.escape(title)}</p>'
        f'<p{vid} class="mt-2 font-display text-3xl font-bold tracking-tight text-ink-900">{html.escape(value)}</p>'
        f'<p{nid} class="mt-auto pt-4 text-xs font-medium text-ink-500 line-clamp-1 border-t border-ink-100/50">{html.escape(note)}</p>'
        "</div>"
    )
