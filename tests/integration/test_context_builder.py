import os
import json
from dotenv import load_dotenv

load_dotenv()
os.environ["ARENA_TRADING_MODE"] = "paper"
os.environ["ARENA_AGENT_IDS"] = "gemini,gpt"
os.environ["ARENA_AGENT_MODE"] = "adk"

from arena.config import load_settings
from arena.data.bq import BigQueryRepository
from arena.memory.store import MemoryStore
from arena.board.store import BoardStore
from arena.context import ContextBuilder
from arena.models import AccountSnapshot, utc_now

def main():
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
    
    from arena.models import Position
    snapshot = AccountSnapshot(
        as_of_ts=utc_now(),
        cash_krw=1000000.0,
        total_equity_krw=1000000.0,
        positions={
            "TSLA": Position(ticker="TSLA", quantity=10, avg_price_krw=200, market_price_krw=210, currency="USD"),
            "AAPL": Position(ticker="AAPL", quantity=5, avg_price_krw=150, market_price_krw=155, currency="USD")
        }
    )
    
    # Let's seed a fake react_tools_summary if nothing exists? It will just read whatever is in BQ.
    ctx = context_builder.build(
        agent_id="gpt",
        snapshot=snapshot,
        sleeve_baseline_equity_krw=1000000.0,
        sleeve_meta={},
    )
    
    print("Memory Rows Count:", len(ctx.get("memory_events", [])))
    for r in ctx.get("memory_events", []):
        print("-", r)
        
    print("Board Rows Count:", len(ctx.get("board_posts", [])))
    for r in ctx.get("board_posts", []):
        print("-", r)

if __name__ == "__main__":
    main()
