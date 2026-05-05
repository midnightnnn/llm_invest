from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("duckdb")

from arena.data.local.repository import LocalRepository


@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "arena.duckdb")
    r = LocalRepository(tenant_id="tenant-a", db_path=db_path)
    r.ensure_dataset()
    r.ensure_tables()
    yield r
    r.session.close()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _seed_market_features_latest(repo, rows):
    cols = (
        "as_of_ts, ticker, exchange_code, instrument_id, close_price_krw, "
        "close_price_native, quote_currency, fx_rate_used, ret_5d, ret_20d, "
        "volatility_20d, sentiment_score, source, updated_at"
    )
    placeholders = ", ".join(["?"] * 14)
    for r in rows:
        repo.execute(
            f"INSERT INTO market_features_latest ({cols}) VALUES ({placeholders})",
            [
                r["as_of_ts"], r["ticker"], r.get("exchange_code"), r.get("instrument_id"),
                r["close_price_krw"], r.get("close_price_native"), r.get("quote_currency", "USD"),
                r.get("fx_rate_used", 1.0), r.get("ret_5d"), r.get("ret_20d"),
                r.get("volatility_20d"), r.get("sentiment_score"), r["source"], r["updated_at"],
            ],
        )


def _seed_memory_event(repo, *, event_id: str, agent_id: str, summary: str, ts: datetime):
    repo.execute(
        """
        INSERT INTO agent_memory_events
          (tenant_id, event_id, created_at, agent_id, event_type, summary, trading_mode)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ["tenant-a", event_id, ts, agent_id, "lesson", summary, "paper"],
    )
