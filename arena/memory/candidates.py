from __future__ import annotations

import re
from datetime import date
from typing import Any

CANDIDATE_MEMORY_EVENT_TYPES: tuple[str, ...] = (
    "candidate_screen_hit",
    "candidate_watchlist",
    "candidate_rejected",
    "candidate_thesis",
)

CANDIDATE_NEXT_CHECKS: tuple[str, ...] = (
    "forecast_returns",
    "technical_signals",
    "get_fundamentals",
)

_BROAD_PROXY_TICKERS: frozenset[str] = frozenset(
    {
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "TLT",
        "GLD",
        "SLV",
        "BRENT",
        "SPX",
        "DJI",
        "VIX",
        "KOSPI",
        "KOSDAQ",
    }
)
_US_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")
_KR_TICKER_RE = re.compile(r"^\d{6}$")


def normalize_candidate_ticker(value: Any) -> str:
    """Returns a memory-eligible stock ticker, excluding broad proxies."""
    token = str(value or "").strip().upper()
    if not token or token in _BROAD_PROXY_TICKERS:
        return ""
    if _US_TICKER_RE.fullmatch(token) or _KR_TICKER_RE.fullmatch(token):
        return token
    return ""


def candidate_memory_event_type(entry: dict[str, Any]) -> str:
    """Chooses the least-inventive candidate memory state from ledger evidence."""
    if entry.get("skip_reasons"):
        return "candidate_rejected"
    if str(entry.get("thesis_summary") or "").strip():
        return "candidate_thesis"
    if entry.get("analyzed_by"):
        return "candidate_watchlist"
    try:
        discovery_count = int(entry.get("discovery_count") or 0)
    except (TypeError, ValueError):
        discovery_count = 0
    if discovery_count >= 2:
        return "candidate_watchlist"
    return "candidate_screen_hit"


def candidate_memory_score(event_type: str) -> float:
    """Keeps candidate memories below thesis/trade weight unless explicitly promoted."""
    token = str(event_type or "").strip().lower()
    if token == "candidate_thesis":
        return 0.55
    if token == "candidate_watchlist":
        return 0.38
    if token == "candidate_rejected":
        return 0.35
    return 0.25


def candidate_memory_ttl_days(event_type: str) -> int | None:
    """Uses shorter lifetimes for screen-only candidates and longer negative priors."""
    token = str(event_type or "").strip().lower()
    if token == "candidate_screen_hit":
        return 14
    if token == "candidate_watchlist":
        return 30
    if token == "candidate_rejected":
        return 45
    return None


def candidate_semantic_key(
    *,
    agent_id: str,
    ticker: str,
    trading_mode: str,
    as_of: date,
) -> str:
    """One candidate memory per ticker/day/agent prevents screen spam."""
    return (
        f"candidate:{str(agent_id or '').strip()}:"
        f"{str(trading_mode or 'paper').strip().lower()}:"
        f"{str(ticker or '').strip().upper()}:{as_of.isoformat()}"
    )


def _set_to_sorted_list(value: Any) -> list[str]:
    if isinstance(value, (set, tuple, list)):
        return sorted(str(token).strip() for token in value if str(token).strip())
    return []


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _safe_evidence(entry: dict[str, Any]) -> dict[str, Any]:
    evidence = entry.get("discovery_evidence")
    if not isinstance(evidence, dict):
        return {}
    out: dict[str, Any] = {}
    for key in (
        "bucket",
        "bucket_rank",
        "score",
        "reason",
        "reason_for",
        "reason_risk",
        "ret_20d",
        "ret_5d",
        "volatility_20d",
        "sentiment_score",
        "per",
        "pbr",
        "roe",
        "debt_ratio",
        "close_price_krw",
        "evidence_level",
    ):
        if evidence.get(key) is not None:
            out[key] = evidence.get(key)
    return out


def _workflow_status(entry: dict[str, Any]) -> str:
    if entry.get("executed_by"):
        return "executed"
    if entry.get("intended_by") or entry.get("ordered_by"):
        return "ordered"
    if entry.get("skip_reasons"):
        return "skipped"
    if entry.get("analyzed_by"):
        return "analyzed"
    return "pending"


def _evidence_level(entry: dict[str, Any], event_type: str) -> str:
    evidence = _safe_evidence(entry)
    raw = str(evidence.get("evidence_level") or "").strip().lower()
    if raw and raw != "screened_only":
        return raw
    if event_type == "candidate_thesis":
        return "candidate_thesis"
    if entry.get("analyzed_by"):
        return "screen_and_analysis"
    return "screened_only"


def _summary_line(
    *,
    ticker: str,
    event_type: str,
    source_tools: list[str],
    analyzed_by: list[str],
    discovery_count: int,
    rank: int | None,
    evidence: dict[str, Any],
    skip_reasons: dict[str, Any],
) -> str:
    source = ", ".join(source_tools[:3]) if source_tools else "screen_market"
    reason = str(evidence.get("reason_for") or evidence.get("reason") or "").strip()
    risk = str(evidence.get("reason_risk") or "").strip()
    metrics: list[str] = []
    for key in ("score", "ret_20d", "ret_5d", "volatility_20d"):
        if evidence.get(key) is not None:
            metrics.append(f"{key}={evidence.get(key)}")

    rank_part = f" rank={rank}" if rank is not None and rank > 0 else ""
    repeat_part = f" repeat={discovery_count}" if discovery_count > 1 else ""
    metric_part = f" ({', '.join(metrics[:3])})" if metrics else ""

    if event_type == "candidate_rejected":
        reasons = ", ".join(f"{k}:{v}" for k, v in sorted(skip_reasons.items())[:3]) or "skipped"
        base = f"{ticker} candidate_rejected: previously surfaced by {source}{rank_part}{repeat_part}; rejected/skipped because {reasons}."
    elif event_type == "candidate_watchlist":
        checks = ", ".join(analyzed_by[:3]) if analyzed_by else "repeated screens"
        base = f"{ticker} candidate_watchlist: surfaced by {source}{rank_part}{repeat_part}{metric_part}; follow-up seen via {checks}."
    elif event_type == "candidate_thesis":
        base = f"{ticker} candidate_thesis: promoted candidate from {source}{rank_part}{repeat_part}{metric_part}."
    else:
        base = f"{ticker} candidate_screen_hit: surfaced by {source}{rank_part}{repeat_part}{metric_part}; evidence is screen-only."

    if reason:
        base += f" Reason: {reason[:180]}."
    if risk:
        base += f" Risk: {risk[:140]}."
    if event_type == "candidate_screen_hit":
        base += " Next checks: forecast, technicals, fundamentals."
    return base[:600]


def candidate_memory_records(
    candidate_ledger: dict[str, dict[str, Any]],
    *,
    held_tickers: set[str] | None = None,
    agent_id: str,
    trading_mode: str,
    cycle_id: str = "",
    phase: str = "",
    as_of: date,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Builds bounded, source-grounded candidate memory records from a cycle ledger."""
    held = {str(token or "").strip().upper() for token in (held_tickers or set()) if str(token or "").strip()}
    rows: list[dict[str, Any]] = []
    for raw_ticker, raw_entry in candidate_ledger.items():
        ticker = normalize_candidate_ticker(raw_ticker)
        if not ticker or ticker in held or not isinstance(raw_entry, dict):
            continue
        event_type = candidate_memory_event_type(raw_entry)
        source_tools = _set_to_sorted_list(raw_entry.get("source_tools"))
        analyzed_by = _set_to_sorted_list(raw_entry.get("analyzed_by"))
        ordered_by = _set_to_sorted_list(raw_entry.get("ordered_by"))
        intended_by = _set_to_sorted_list(raw_entry.get("intended_by"))
        executed_by = _set_to_sorted_list(raw_entry.get("executed_by"))
        evidence = _safe_evidence(raw_entry)
        skip_reasons_raw = raw_entry.get("skip_reasons") if isinstance(raw_entry.get("skip_reasons"), dict) else {}
        skip_reasons = {
            str(key).strip().lower(): _int_or_zero(value)
            for key, value in skip_reasons_raw.items()
            if str(key).strip()
        }
        discovery_count = max(_int_or_none(raw_entry.get("discovery_count")) or 0, 0)
        rank = _int_or_none(raw_entry.get("last_seen_rank"))
        missing_checks = [tool for tool in CANDIDATE_NEXT_CHECKS if tool not in set(analyzed_by)]
        workflow_status = _workflow_status(raw_entry)
        evidence_level = _evidence_level(raw_entry, event_type)
        payload = {
            "source": "candidate_discovery",
            "ticker": ticker,
            "cycle_id": str(cycle_id or "").strip(),
            "phase": str(phase or "").strip().lower(),
            "candidate_status": event_type.replace("candidate_", ""),
            "workflow_status": workflow_status,
            "evidence_level": evidence_level,
            "source_tools": source_tools,
            "analyzed_by": analyzed_by,
            "ordered_by": ordered_by,
            "intended_by": intended_by,
            "executed_by": executed_by,
            "discovery_count": discovery_count,
            "last_seen_rank": rank,
            "discovery_evidence": evidence,
            "skip_reasons": skip_reasons,
            "suggested_next_checks": missing_checks[:3],
            "held_at_creation": False,
        }
        rows.append(
            {
                "ticker": ticker,
                "event_type": event_type,
                "summary": _summary_line(
                    ticker=ticker,
                    event_type=event_type,
                    source_tools=source_tools,
                    analyzed_by=analyzed_by,
                    discovery_count=discovery_count,
                    rank=rank,
                    evidence=evidence,
                    skip_reasons=skip_reasons,
                ),
                "payload": payload,
                "score": candidate_memory_score(event_type),
                "ttl_days": candidate_memory_ttl_days(event_type),
                "semantic_key": candidate_semantic_key(
                    agent_id=agent_id,
                    ticker=ticker,
                    trading_mode=trading_mode,
                    as_of=as_of,
                ),
                "sort_key": (
                    {"candidate_thesis": 0, "candidate_watchlist": 1, "candidate_rejected": 2}.get(event_type, 3),
                    rank if rank is not None and rank > 0 else 9999,
                    -discovery_count,
                    ticker,
                ),
            }
        )

    try:
        clean_limit = int(limit)
    except (TypeError, ValueError):
        clean_limit = 5
    if clean_limit <= 0:
        return []
    rows.sort(key=lambda row: row["sort_key"])
    bounded = rows[: min(clean_limit, 10)]
    for row in bounded:
        row.pop("sort_key", None)
    return bounded
