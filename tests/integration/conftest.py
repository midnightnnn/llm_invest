from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv


def require_live_integration(*env_names: str) -> None:
    """Skips opt-in integration tests unless the caller explicitly enables them."""
    load_dotenv()
    if str(os.getenv("ARENA_RUN_LIVE_INTEGRATION") or "").strip() != "1":
        pytest.skip("set ARENA_RUN_LIVE_INTEGRATION=1 to run live integration tests")
    missing = [name for name in env_names if not str(os.getenv(name) or "").strip()]
    if missing:
        pytest.skip(f"missing required integration env: {', '.join(missing)}")
