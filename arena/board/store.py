from __future__ import annotations

from arena.data.bq import BigQueryRepository
from arena.models import BoardPost

class BoardStore:
    """Persists and retrieves inter-agent board discussions."""

    def __init__(self, repo: BigQueryRepository, trading_mode: str = "paper"):
        self.repo = repo
        self.trading_mode = trading_mode

    def publish(self, post: BoardPost) -> None:
        """Stores a board post for cross-agent information sharing."""
        post.trading_mode = self.trading_mode
        self.repo.write_board_post(post)

    def recent(self, limit: int) -> list[dict]:
        """Fetches recent board posts."""
        return self.repo.recent_board_posts(limit=limit, trading_mode=self.trading_mode)
