"""ADK web adapters for locally debugging arena trading agents.

Run from the repository root:

    ARENA_MODE=local adk web arena/agents/dev_ui
"""

from __future__ import annotations

__all__ = ["build_dev_agent"]

from arena.agents.dev_ui._factory import build_dev_agent
