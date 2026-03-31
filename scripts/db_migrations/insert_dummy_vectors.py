import os
import uuid
from dotenv import load_dotenv

load_dotenv()
os.environ["ARENA_TRADING_MODE"] = "paper"
os.environ["ARENA_AGENT_IDS"] = "gemini,gpt"
os.environ["ARENA_AGENT_MODE"] = "adk"

from arena.config import load_settings
from arena.data.bq import BigQueryRepository
from arena.memory.store import MemoryStore
from arena.board.store import BoardStore

def main():
    settings = load_settings()
    repo = BigQueryRepository(
        project=settings.google_cloud_project,
        dataset=settings.bq_dataset,
        location=settings.bq_location,
    )
    vector = MemoryStore(repo).vector_store
    board_vector = BoardStore(repo).vector_store
    
    print("Saving dummy memory vector for AAPL...")
    mem_event_id = str(uuid.uuid4())
    vector.save_memory_vector(event_id=mem_event_id, agent_id="gpt", summary="I bought AAPL because of good earnings.", score=1.0)
    
    print("Saving dummy board vector for TSLA...")
    post_id = str(uuid.uuid4())
    board_vector.save_board_vector(post_id=post_id, author_id="gemini", title="TSLA thesis", body="TSLA is going to moon because of FSD.")
    
    print("Inserted dummy vectors successfully.")

if __name__ == "__main__":
    main()
