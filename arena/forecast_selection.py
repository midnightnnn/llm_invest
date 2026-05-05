from __future__ import annotations

from typing import Any


FORECAST_RANKER_BUCKETS: tuple[str, ...] = ("momentum", "pullback", "recovery", "defensive")
FORECAST_RANKER_PROFILES: tuple[str, ...] = ("aggressive", "balanced", "defensive")


def clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper()


def ordered_unique_tickers(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        ticker = clean_ticker(value)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        out.append(ticker)
    return out


def ranker_rows_to_tickers(rows: list[dict[str, Any]], *, limit: int) -> list[str]:
    lim = max(1, int(limit or 1))
    ranked = sorted(
        [row for row in rows if isinstance(row, dict) and clean_ticker(row.get("ticker"))],
        key=lambda row: (
            _rank_value(row.get("recommendation_rank")),
            -_score_value(row.get("recommendation_score")),
            clean_ticker(row.get("ticker")),
        ),
    )
    return ordered_unique_tickers([row.get("ticker") for row in ranked])[:lim]


def merge_forecast_tickers(
    *,
    held_tickers: list[Any] | None = None,
    ranker_tickers: list[Any] | None = None,
    fallback_tickers: list[Any] | None = None,
    max_tickers: int = 80,
) -> list[str]:
    limit = max(1, int(max_tickers or 80))
    return ordered_unique_tickers(
        list(held_tickers or []) + list(ranker_tickers or []) + list(fallback_tickers or [])
    )[:limit]


def _rank_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1_000_000_000


def _score_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
