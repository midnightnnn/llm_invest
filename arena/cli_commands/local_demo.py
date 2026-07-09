"""Local quickstart data commands."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import logging
import math
import os
from typing import Any

from arena.config import load_settings
from arena.data.factory import get_repository
from arena.open_trading.sync import MarketDataSyncService
from arena.cli_commands.local_bootstrap import seed_local_memory_compaction_prompts

logger = logging.getLogger(__name__)


_DEMO_TICKERS = (
    ("AAPL", "Apple Inc.", "NASD", "USD", 185.0),
    ("MSFT", "Microsoft Corp.", "NASD", "USD", 420.0),
    ("NVDA", "NVIDIA Corp.", "NASD", "USD", 880.0),
    ("TSLA", "Tesla Inc.", "NASD", "USD", 175.0),
    ("GOOGL", "Alphabet Inc.", "NASD", "USD", 152.0),
    ("AMZN", "Amazon.com Inc.", "NASD", "USD", 182.0),
)


def _fx_rate() -> float:
    try:
        return float(os.getenv("ARENA_DEMO_USD_KRW_RATE", "1380"))
    except ValueError:
        return 1380.0


def _demo_rows(*, days: int = 60) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fx = _fx_rate()
    today = datetime.now(timezone.utc).date()
    rows: list[dict[str, Any]] = []
    instruments: list[dict[str, Any]] = []
    for ticker_idx, (ticker, name, exchange, currency, base_price) in enumerate(_DEMO_TICKERS):
        instrument_id = f"{exchange}:{ticker}"
        instruments.append(
            {
                "instrument_id": instrument_id,
                "ticker": ticker,
                "ticker_name": name,
                "exchange_code": exchange,
                "currency": currency,
                "lot_size": 1,
                "tick_size": 0.01,
                "tradable": True,
                "status": "ACTIVE",
                "updated_at": datetime.now(timezone.utc),
            }
        )
        closes: list[float] = []
        for offset in range(days, 0, -1):
            day = today - timedelta(days=offset)
            wave = math.sin((days - offset + ticker_idx) / 7.0) * 0.025
            drift = (days - offset) * (0.0007 + ticker_idx * 0.00005)
            close_native = base_price * (1.0 + wave + drift)
            closes.append(close_native)
            idx = len(closes) - 1
            ret_5d = (close_native / closes[idx - 5] - 1.0) if idx >= 5 else None
            ret_20d = (close_native / closes[idx - 20] - 1.0) if idx >= 20 else None
            if idx >= 20:
                window = closes[idx - 19 : idx + 1]
                mean = sum(window) / len(window)
                variance = sum((value - mean) ** 2 for value in window) / len(window)
                volatility = math.sqrt(variance) / mean
            else:
                volatility = None
            as_of_ts = datetime.combine(day, time(hour=21), tzinfo=timezone.utc)
            rows.append(
                {
                    "as_of_ts": as_of_ts,
                    "ingested_at": datetime.now(timezone.utc),
                    "ticker": ticker,
                    "exchange_code": exchange,
                    "instrument_id": instrument_id,
                    "close_price_krw": close_native * fx,
                    "close_price_native": close_native,
                    "quote_currency": currency,
                    "fx_rate_used": fx,
                    "ret_5d": ret_5d,
                    "ret_20d": ret_20d,
                    "volatility_20d": volatility,
                    "sentiment_score": math.sin((idx + ticker_idx) / 9.0) * 0.2,
                    "source": "local_demo",
                }
            )
    return rows, instruments


def cmd_seed_local_demo(*, days: int = 60) -> dict[str, int]:
    """Seeds deterministic local demo market data into DuckDB."""
    settings = load_settings()
    settings.arena_mode = "local"
    repo = get_repository(settings, tenant_id=os.getenv("ARENA_TENANT_ID") or "local")
    repo.ensure_tables()
    seeded = seed_local_memory_compaction_prompts(repo, tenant_id=repo.tenant_id, updated_by="seed-local-demo")
    rows, instruments = _demo_rows(days=max(10, int(days)))
    inserted = repo.insert_market_features(rows)
    latest = repo.refresh_market_features_latest(sources=["local_demo"])
    inst = repo.upsert_instrument_master(instruments)
    logger.info(
        "[green]Local demo data seeded[/green] market_rows=%d latest_rows=%d instruments=%d",
        inserted,
        latest,
        inst,
    )
    logger.info(
        "Memory compaction prompts seeded global=%s tenant=%s",
        "yes" if seeded["global"] else "no",
        "yes" if seeded["tenant"] else "no",
    )
    return {"market_rows": inserted, "latest_rows": latest, "instruments": inst}


def cmd_backfill_local_market() -> Any:
    """Runs the existing market backfill with the local DuckDB repository."""
    settings = load_settings()
    settings.arena_mode = "local"
    repo = get_repository(settings, tenant_id=os.getenv("ARENA_TENANT_ID") or "local")
    repo.ensure_tables()
    service = MarketDataSyncService(settings=settings, repo=repo)
    return service.sync_market_features()
