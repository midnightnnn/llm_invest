from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from arena.models import BoardPost, OrderIntent


class AgentOutput(BaseModel):
    """Packages one agent's trade intents and board post."""

    intents: list[OrderIntent] = Field(default_factory=list)
    board_post: BoardPost


class TradingAgent(Protocol):
    """Defines an agent that produces intents from context."""

    agent_id: str

    def generate(self, context: dict) -> AgentOutput:
        """Creates order intents plus one board post."""
        ...
