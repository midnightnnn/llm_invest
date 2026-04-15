"""Per-tool evidence extraction for the cycle evidence index.

Each extractor takes a raw tool result and returns a list of evidence events.
An event is a dict with at least ``scope`` and ``summary``; ``ticker`` is set
for ``scope == "ticker"``. Tool authors own what fields appear in ``summary``
so the index never invents comparison signals.

Tools without an entry in :data:`TOOL_EVIDENCE` are considered non-evidence
(raw fetches, actions, utilities, state lookups) and skipped by the builder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

EvidenceEvent = dict[str, Any]


def _norm_ticker(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text or None


def _pick(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fields:
        value = row.get(field)
        if value is not None:
            out[field] = value
    return out


def _screen_market(result: Any) -> list[EvidenceEvent]:
    if not isinstance(result, list):
        return []
    out: list[EvidenceEvent] = []
    fields = (
        "bucket",
        "bucket_rank",
        "score",
        "ret_20d",
        "ret_5d",
        "volatility_20d",
        "sentiment_score",
        "close_price_krw",
    )
    for row in result:
        if not isinstance(row, dict):
            continue
        ticker = _norm_ticker(row.get("ticker"))
        if not ticker:
            continue
        out.append({"scope": "ticker", "ticker": ticker, "summary": _pick(row, fields)})
    return out


def _forecast_returns(result: Any) -> list[EvidenceEvent]:
    if not isinstance(result, list):
        return []
    out: list[EvidenceEvent] = []
    fields = (
        "consensus",
        "prob_up",
        "exp_return_period",
        "forecast_horizon",
        "model_votes_up",
        "model_votes_total",
    )
    for row in result:
        if not isinstance(row, dict):
            continue
        ticker = _norm_ticker(row.get("ticker"))
        if not ticker:
            continue
        out.append({"scope": "ticker", "ticker": ticker, "summary": _pick(row, fields)})
    return out


def _technical_one(row: dict[str, Any]) -> dict[str, Any]:
    macd = row.get("macd") if isinstance(row.get("macd"), dict) else {}
    ma = row.get("moving_averages") if isinstance(row.get("moving_averages"), dict) else {}
    summary: dict[str, Any] = _pick(row, ("rsi_14", "rsi_state", "trend_state"))
    if isinstance(macd, dict) and macd.get("state") is not None:
        summary["macd_state"] = macd.get("state")
    if isinstance(ma, dict) and ma.get("price_vs_sma20") is not None:
        summary["price_vs_sma20"] = ma.get("price_vs_sma20")
    return summary


def _technical_signals(result: Any) -> list[EvidenceEvent]:
    out: list[EvidenceEvent] = []
    if isinstance(result, dict) and isinstance(result.get("rows"), list):
        for row in result["rows"]:
            if not isinstance(row, dict):
                continue
            ticker = _norm_ticker(row.get("ticker"))
            if not ticker:
                continue
            out.append({"scope": "ticker", "ticker": ticker, "summary": _technical_one(row)})
        return out
    if isinstance(result, dict) and result.get("ticker") is not None:
        ticker = _norm_ticker(result.get("ticker"))
        if ticker:
            out.append({"scope": "ticker", "ticker": ticker, "summary": _technical_one(result)})
    return out


def _get_fundamentals(result: Any) -> list[EvidenceEvent]:
    if not isinstance(result, dict):
        return []
    out: list[EvidenceEvent] = []
    fields = ("per", "pbr", "roe", "debt_ratio", "eps", "bps", "market_cap", "currency")
    for row in result.get("rows") or []:
        if not isinstance(row, dict):
            continue
        ticker = _norm_ticker(row.get("ticker"))
        if not ticker:
            continue
        out.append({"scope": "ticker", "ticker": ticker, "summary": _pick(row, fields)})
    return out


def _portfolio_diagnosis(result: Any) -> list[EvidenceEvent]:
    if not isinstance(result, dict) or result.get("error"):
        return []
    summary = _pick(
        result,
        (
            "concentration_top3",
            "hhi",
            "risk_contribution",
            "momentum_score",
            "short_term_momentum",
            "volatility_weighted",
        ),
    )
    if not summary:
        return []
    return [{"scope": "portfolio", "summary": summary}]


def _get_research_briefing(result: Any) -> list[EvidenceEvent]:
    if not isinstance(result, list):
        return []
    out: list[EvidenceEvent] = []
    fields = ("category", "headline", "summary", "created_at")
    for row in result:
        if not isinstance(row, dict):
            continue
        picked = _pick(row, fields)
        ticker = _norm_ticker(row.get("ticker"))
        if ticker:
            out.append({"scope": "ticker", "ticker": ticker, "summary": picked})
        else:
            out.append({"scope": "market", "summary": picked})
    return out


TOOL_EVIDENCE: dict[str, Callable[[Any], list[EvidenceEvent]]] = {
    "screen_market": _screen_market,
    "forecast_returns": _forecast_returns,
    "technical_signals": _technical_signals,
    "get_fundamentals": _get_fundamentals,
    "portfolio_diagnosis": _portfolio_diagnosis,
    "get_research_briefing": _get_research_briefing,
}


def extract_evidence(tool_name: str, result: Any) -> list[EvidenceEvent]:
    """Returns evidence events for a tool call, or [] when not an evidence tool."""
    extractor = TOOL_EVIDENCE.get(str(tool_name or "").strip())
    if extractor is None:
        return []
    try:
        events = extractor(result) or []
    except Exception:
        return []
    out: list[EvidenceEvent] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        scope = str(ev.get("scope") or "").strip()
        if not scope:
            continue
        summary = ev.get("summary") if isinstance(ev.get("summary"), dict) else {}
        if scope == "ticker" and not _norm_ticker(ev.get("ticker")):
            continue
        normalized: EvidenceEvent = {"scope": scope, "summary": dict(summary)}
        if scope == "ticker":
            normalized["ticker"] = _norm_ticker(ev.get("ticker"))
        if ev.get("sector"):
            normalized["sector"] = str(ev["sector"])
        out.append(normalized)
    return out
