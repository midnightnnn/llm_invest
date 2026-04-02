from __future__ import annotations

import pytest

from arena.board.store import BoardStore
from arena.config import load_settings
from arena.context import ContextBuilder
from arena.data.bq import BigQueryRepository
from arena.memory.store import MemoryStore
from arena.models import AccountSnapshot, Position
from tests.integration.conftest import require_live_integration

pytestmark = pytest.mark.integration


def test_context_builder_live_bq_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    require_live_integration("GOOGLE_CLOUD_PROJECT")
    monkeypatch.setenv("ARENA_TRADING_MODE", "paper")
    monkeypatch.setenv("ARENA_AGENT_IDS", "gemini,gpt")
    monkeypatch.setenv("ARENA_AGENT_MODE", "adk")

    settings = load_settings()
    repo = BigQueryRepository(
        project=settings.google_cloud_project,
        dataset=settings.bq_dataset,
        location=settings.bq_location,
    )
    memory = MemoryStore(repo)
    board = BoardStore(repo)
    context_builder = ContextBuilder(
        repo=repo,
        memory=memory,
        board=board,
        settings=settings,
    )
    snapshot = AccountSnapshot(
        cash_krw=1_000_000.0,
        total_equity_krw=1_000_000.0,
        usd_krw_rate=1_350.0,
        positions={
            "TSLA": Position(
                ticker="TSLA",
                quantity=10,
                avg_price_krw=200.0,
                market_price_krw=210.0,
                quote_currency="USD",
            ),
            "AAPL": Position(
                ticker="AAPL",
                quantity=5,
                avg_price_krw=150.0,
                market_price_krw=155.0,
                quote_currency="USD",
            ),
        },
    )

    ctx = context_builder.build(
        agent_id="gpt",
        snapshot=snapshot,
        sleeve_baseline_equity_krw=1_000_000.0,
        sleeve_meta={},
    )

    assert ctx["agent_id"] == "gpt"
    assert isinstance(ctx.get("memory_events"), list)
    assert isinstance(ctx.get("board_posts"), list)
    assert isinstance(ctx.get("memory_context"), str)
    assert isinstance(ctx.get("board_context"), str)
