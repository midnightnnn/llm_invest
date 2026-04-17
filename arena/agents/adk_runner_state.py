from __future__ import annotations

from typing import Any

DISCOVERY_TOOLS: frozenset[str] = frozenset({"screen_market"})
ANALYSIS_TOOLS: frozenset[str] = frozenset(
    {
        "forecast_returns",
        "technical_signals",
        "get_fundamentals",
        "fetch_sec_filings",
    }
)
ORDERABLE_SIDES: frozenset[str] = frozenset({"BUY", "SELL"})
EXECUTED_REPORT_STATUSES: frozenset[str] = frozenset({"FILLED", "SIMULATED"})


def tickers_from_tool_args(args: dict[str, Any]) -> list[str]:
    """Extracts normalized tickers from common tool-argument shapes."""
    out: list[str] = []
    if isinstance(args.get("tickers"), list):
        out.extend(str(t).strip().upper() for t in args["tickers"] if str(t).strip())
    if isinstance(args.get("ticker"), str) and args["ticker"].strip():
        out.append(args["ticker"].strip().upper())
    deduped: list[str] = []
    for token in out:
        if token and token not in deduped:
            deduped.append(token)
    return deduped


def tickers_from_tool_result(result: Any) -> list[str]:
    """Extracts normalized tickers from common compact tool-result shapes."""
    out: list[str] = []

    def _append(token: Any) -> None:
        text = str(token or "").strip().upper()
        if text and text not in out:
            out.append(text)

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            ticker = value.get("ticker")
            if isinstance(ticker, str):
                _append(ticker)
            tickers = value.get("tickers")
            if isinstance(tickers, list):
                for token in tickers:
                    _append(token)
            rows = value.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        _walk(row)
            data = value.get("data")
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        _walk(row)
            return
        if isinstance(value, list):
            for row in value:
                if isinstance(row, dict):
                    _walk(row)

    _walk(result)
    return out


def discovery_rows_from_tool_result(tool_name: str, result: Any) -> list[dict[str, Any]]:
    """Extracts discovery rows with bucket-aware provenance from tool results."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    token = str(tool_name or "").strip().lower()

    def _source_tool(bucket: str | None) -> str:
        bucket_token = str(bucket or "").strip().lower()
        if bucket_token:
            return f"screen_market:{bucket_token}"
        return token or "screen_market"

    def _append(row: dict[str, Any], rank: int) -> None:
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker or ticker in seen:
            return
        seen.add(ticker)
        bucket = str(row.get("bucket") or "").strip().lower() or None
        evidence: dict[str, Any] = {}
        for field in (
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
            if row.get(field) is not None:
                evidence[field] = row.get(field)
        out.append(
            {
                "ticker": ticker,
                "rank": rank,
                "source_tool": _source_tool(bucket),
                "bucket": bucket,
                "evidence": evidence,
            }
        )

    rows: list[dict[str, Any]] = []
    if isinstance(result, list):
        rows = [row for row in result if isinstance(row, dict)]
    elif isinstance(result, dict):
        for key in ("rows", "data"):
            value = result.get(key)
            if isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
    for rank, row in enumerate(rows, start=1):
        _append(row, rank)
    return out


def update_candidate_ledger(
    candidate_ledger: dict[str, dict[str, Any]],
    held_tickers_cache: set[str],
    current_phase: str,
    *,
    tool_name: str,
    args: dict[str, Any],
    result: Any,
) -> None:
    """Updates discovery/analysis funnel state from one tool invocation."""
    if tool_name in DISCOVERY_TOOLS:
        discoveries = discovery_rows_from_tool_result(tool_name, result)
        if not discoveries:
            discoveries = [
                {
                    "ticker": ticker,
                    "rank": rank,
                    "source_tool": str(tool_name or "").strip().lower(),
                    "bucket": None,
                }
                for rank, ticker in enumerate(tickers_from_tool_result(result), start=1)
            ]
        for item in discoveries:
            ticker = str(item.get("ticker") or "").strip().upper()
            if not ticker or ticker in held_tickers_cache:
                continue
            entry = candidate_ledger.setdefault(
                ticker,
                {
                    "source_tools": set(),
                    "discovered_phase": current_phase,
                    "analyzed_by": set(),
                    "discovery_count": 0,
                    "last_seen_rank": None,
                },
            )
            source_tool = str(item.get("source_tool") or tool_name).strip().lower() or str(tool_name)
            entry.setdefault("source_tools", set()).add(source_tool)
            entry.setdefault("discovered_phase", current_phase)
            entry["discovery_count"] = int(entry.get("discovery_count") or 0) + 1
            prev_rank = entry.get("last_seen_rank")
            try:
                prev_rank_int = int(prev_rank) if prev_rank is not None else None
            except (TypeError, ValueError):
                prev_rank_int = None
            try:
                rank = int(item.get("rank") or 0)
            except (TypeError, ValueError):
                rank = 0
            entry["last_seen_rank"] = rank if prev_rank_int is None else min(prev_rank_int, rank)
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            if evidence:
                entry["discovery_evidence"] = dict(evidence)
        return

    if tool_name not in ANALYSIS_TOOLS:
        return

    tickers = tickers_from_tool_args(args)
    if not tickers:
        tickers = tickers_from_tool_result(result)
    for ticker in tickers:
        if ticker in candidate_ledger:
            candidate_ledger[ticker].setdefault("analyzed_by", set()).add(tool_name)


def unresolved_candidates(candidate_ledger: dict[str, dict[str, Any]]) -> list[str]:
    """Returns discovered non-held tickers that have not been deeply analyzed yet."""
    return [
        ticker
        for ticker, entry in candidate_ledger.items()
        if not entry.get("analyzed_by")
    ]


def discovered_candidate_tickers(
    candidate_ledger: dict[str, dict[str, Any]],
    *,
    limit: int = 50,
) -> list[str]:
    """Returns discovered candidate tickers sorted by the working-set priority."""
    rows = opportunity_working_set(candidate_ledger, limit=max(1, min(int(limit), 100)))
    return [str(row.get("ticker") or "").strip().upper() for row in rows if str(row.get("ticker") or "").strip()]


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


def _candidate_status(entry: dict[str, Any]) -> str:
    status = _workflow_status(entry)
    if status == "pending":
        return "screened_only"
    return status


def _candidate_evidence_level(entry: dict[str, Any]) -> str:
    evidence = entry.get("discovery_evidence") if isinstance(entry.get("discovery_evidence"), dict) else {}
    raw_level = str(evidence.get("evidence_level") or "").strip().lower()
    if raw_level and raw_level != "screened_only":
        return raw_level
    if entry.get("analyzed_by"):
        return "screen_and_analysis"
    return "screened_only"


def opportunity_working_set(
    candidate_ledger: dict[str, dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Builds a compact self-discovered opportunity list for prompt visibility."""

    priority = {
        "pending": 0,
        "analyzed": 1,
        "skipped": 2,
        "ordered": 3,
        "executed": 4,
    }

    rows: list[dict[str, Any]] = []
    for ticker, raw_entry in candidate_ledger.items():
        if not isinstance(raw_entry, dict):
            continue
        source_tools = sorted(str(tool).strip() for tool in raw_entry.get("source_tools", set()) if str(tool).strip())
        discovery_buckets = sorted(
            {
                tool.split(":", 1)[1]
                for tool in source_tools
                if tool.startswith("screen_market:") and ":" in tool
            }
        )
        analyzed_by = sorted(str(tool).strip() for tool in raw_entry.get("analyzed_by", set()) if str(tool).strip())
        workflow_status = _workflow_status(raw_entry)
        row: dict[str, Any] = {
            "ticker": ticker,
            "status": _candidate_status(raw_entry),
            "workflow_status": workflow_status,
            "evidence_level": _candidate_evidence_level(raw_entry),
            "source_tools": source_tools,
            "analyzed_by": analyzed_by,
        }
        if discovery_buckets:
            row["discovery_buckets"] = discovery_buckets
        try:
            discovery_count = int(raw_entry.get("discovery_count") or 0)
        except (TypeError, ValueError):
            discovery_count = 0
        if discovery_count > 0:
            row["discovery_count"] = discovery_count
        try:
            last_seen_rank = int(raw_entry.get("last_seen_rank")) if raw_entry.get("last_seen_rank") is not None else None
        except (TypeError, ValueError):
            last_seen_rank = None
        if last_seen_rank is not None:
            row["last_seen_rank"] = last_seen_rank
        if workflow_status == "skipped":
            skip_reasons = raw_entry.get("skip_reasons")
            if isinstance(skip_reasons, dict):
                row["skip_reasons"] = {
                    str(reason).strip().lower(): int(count)
                    for reason, count in skip_reasons.items()
                    if str(reason).strip()
                }
        rows.append(row)

    rows.sort(
        key=lambda item: (
            priority.get(str(item.get("workflow_status") or "pending"), 99),
            int(item.get("last_seen_rank") or 9999),
            -int(item.get("discovery_count") or 0),
            str(item.get("ticker") or ""),
        )
    )
    return rows[: max(1, min(int(limit), 10))]


def candidate_cases(
    candidate_ledger: dict[str, dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Builds model-facing non-held candidate cases without inventing thesis memory."""
    working_rows = opportunity_working_set(candidate_ledger, limit=limit)
    out: list[dict[str, Any]] = []
    for row in working_rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        entry = candidate_ledger.get(ticker)
        if not ticker or not isinstance(entry, dict):
            continue
        evidence = entry.get("discovery_evidence") if isinstance(entry.get("discovery_evidence"), dict) else {}
        source_tools = list(row.get("source_tools") or [])
        analyzed_by = list(row.get("analyzed_by") or [])
        case_for = str(evidence.get("reason_for") or evidence.get("reason") or "").strip()
        if not case_for:
            buckets = row.get("discovery_buckets") if isinstance(row.get("discovery_buckets"), list) else []
            if buckets:
                case_for = "Discovered by screen_market buckets: " + ", ".join(str(bucket) for bucket in buckets)
            elif source_tools:
                case_for = "Discovered by " + ", ".join(source_tools[:3])
            else:
                case_for = "Discovered by market screening."
        case_risk = str(evidence.get("reason_risk") or "").strip()
        if not case_risk:
            if analyzed_by:
                case_risk = "Review recent analysis outputs before initiating a new position."
            else:
                case_risk = "Screen-only evidence; confirm with forecast/technical/fundamental tools before initiating."
        item: dict[str, Any] = {
            "ticker": ticker,
            "current_status": "candidate",
            "candidate_status": row.get("status") or _candidate_status(entry),
            "workflow_status": row.get("workflow_status") or _workflow_status(entry),
            "evidence_level": row.get("evidence_level") or _candidate_evidence_level(entry),
            "case_for": case_for,
            "case_risk": case_risk,
            "source_tools": source_tools,
            "analyzed_by": analyzed_by,
        }
        if row.get("discovery_buckets"):
            item["discovery_buckets"] = row.get("discovery_buckets")
        out.append(item)
    return out


def record_candidate_orders(
    candidate_ledger: dict[str, dict[str, Any]],
    *,
    current_phase: str,
    orders: list[dict[str, Any]],
) -> None:
    """Marks candidate tickers that the model explicitly ordered in its JSON output."""
    for order in orders:
        if not isinstance(order, dict):
            continue
        ticker = str(order.get("ticker") or "").strip().upper()
        side = str(order.get("side") or "").strip().upper()
        sizing_key = "target_weight" if side == "BUY" else "sell_ratio"
        try:
            sizing_value = float(order.get(sizing_key) or 0.0)
        except (TypeError, ValueError):
            sizing_value = 0.0
        if not ticker or side not in ORDERABLE_SIDES or sizing_value <= 0:
            continue
        if ticker not in candidate_ledger:
            continue
        candidate_ledger[ticker].setdefault("ordered_by", set()).add(current_phase)


def record_candidate_order_feedback(
    candidate_ledger: dict[str, dict[str, Any]],
    *,
    current_phase: str,
    feedback_events: list[dict[str, Any]],
) -> None:
    """Records post-validation order feedback for candidate tickers."""
    for event in feedback_events:
        if not isinstance(event, dict):
            continue
        ticker = str(event.get("ticker") or "").strip().upper()
        if not ticker or ticker not in candidate_ledger:
            continue
        status = str(event.get("status") or "").strip().lower()
        if status == "intent_built":
            candidate_ledger[ticker].setdefault("intended_by", set()).add(current_phase)
            continue
        if status != "skipped":
            continue
        reason = str(event.get("reason") or "unknown").strip().lower() or "unknown"
        skip_reasons = candidate_ledger[ticker].setdefault("skip_reasons", {})
        skip_reasons[reason] = int(skip_reasons.get(reason) or 0) + 1


def record_candidate_executions(
    candidate_ledger: dict[str, dict[str, Any]],
    *,
    current_phase: str,
    intents: list[Any],
    reports: list[Any],
) -> None:
    """Marks candidate tickers that reached filled/simulated execution."""
    if not intents or not reports:
        return
    for intent, report in zip(intents, reports):
        ticker = str(getattr(intent, "ticker", "") or "").strip().upper()
        if not ticker or ticker not in candidate_ledger or report is None:
            continue
        status = getattr(report, "status", "")
        status_token = str(getattr(status, "value", status) or "").strip().upper()
        if status_token in EXECUTED_REPORT_STATUSES:
            candidate_ledger[ticker].setdefault("executed_by", set()).add(current_phase)


def funnel_metrics(
    candidate_ledger: dict[str, dict[str, Any]],
    tool_events: list[dict[str, Any]],
    held_tickers_cache: set[str],
) -> dict[str, int]:
    """Builds cycle-level discovery-to-analysis funnel counts."""
    discovered_nonheld = len(candidate_ledger)
    analyzed_nonheld = sum(1 for entry in candidate_ledger.values() if entry.get("analyzed_by"))
    pending_nonheld = max(discovered_nonheld - analyzed_nonheld, 0)
    ordered_nonheld = sum(1 for entry in candidate_ledger.values() if entry.get("ordered_by"))
    intended_nonheld = sum(1 for entry in candidate_ledger.values() if entry.get("intended_by"))
    executed_nonheld = sum(1 for entry in candidate_ledger.values() if entry.get("executed_by"))
    skipped_nonheld = sum(1 for entry in candidate_ledger.values() if entry.get("skip_reasons"))
    skip_reasons: dict[str, int] = {}
    for entry in candidate_ledger.values():
        raw = entry.get("skip_reasons")
        if not isinstance(raw, dict):
            continue
        for reason, count in raw.items():
            token = str(reason or "").strip().lower()
            if not token:
                continue
            try:
                count_val = int(count)
            except (TypeError, ValueError):
                count_val = 0
            skip_reasons[token] = skip_reasons.get(token, 0) + max(count_val, 0)

    analyzed_held_tickers: set[str] = set()
    for event in tool_events:
        tool_name = str(event.get("tool") or "").strip()
        if tool_name not in ANALYSIS_TOOLS:
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        result = event.get("result")
        tickers = tickers_from_tool_args(args)
        if not tickers:
            tickers = tickers_from_tool_result(result)
        for ticker in tickers:
            if ticker in held_tickers_cache:
                analyzed_held_tickers.add(ticker)

    return {
        "discovered_nonheld": discovered_nonheld,
        "analyzed_nonheld": analyzed_nonheld,
        "pending_nonheld": pending_nonheld,
        "analyzed_held": len(analyzed_held_tickers),
        "ordered_nonheld": ordered_nonheld,
        "intended_nonheld": intended_nonheld,
        "executed_nonheld": executed_nonheld,
        "skipped_nonheld": skipped_nonheld,
        "skip_reasons": skip_reasons,
    }


def model_facing_funnel_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Renames internal funnel counters for prompt-facing candidate evidence levels."""
    raw = metrics if isinstance(metrics, dict) else {}

    def _int(key: str, alias: str) -> int:
        try:
            value = raw.get(key)
            if value is None:
                value = raw.get(alias)
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    out: dict[str, Any] = {
        "discovered_candidates": _int("discovered_nonheld", "discovered_candidates"),
        "screened_only_candidates": _int("pending_nonheld", "screened_only_candidates"),
        "fully_analyzed_candidates": _int("analyzed_nonheld", "fully_analyzed_candidates"),
        "analyzed_held_positions": _int("analyzed_held", "analyzed_held_positions"),
        "ordered_candidates": _int("ordered_nonheld", "ordered_candidates"),
        "intended_candidates": _int("intended_nonheld", "intended_candidates"),
        "executed_candidates": _int("executed_nonheld", "executed_candidates"),
        "skipped_candidates": _int("skipped_nonheld", "skipped_candidates"),
    }
    skip_reasons = raw.get("skip_reasons")
    if isinstance(skip_reasons, dict) and skip_reasons:
        out["skip_reasons"] = dict(skip_reasons)
    return out
