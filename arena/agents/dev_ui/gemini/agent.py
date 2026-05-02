from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from arena.agents.dev_ui import _factory

root_agent = _factory.build_dev_agent("gemini")

__all__ = ["root_agent"]
