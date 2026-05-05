from __future__ import annotations

from typing import Any

from arena.config import Settings, load_settings, normalize_agent_settings


def make_test_settings(*, normalize_agents: bool = False, **overrides: Any) -> Settings:
    """Loads a fresh Settings object and applies explicit test overrides."""

    settings = load_settings()
    for key, value in overrides.items():
        if not hasattr(settings, key):
            raise AttributeError(f"unknown Settings field: {key}")
        setattr(settings, key, value)
    if normalize_agents:
        normalize_agent_settings(settings)
    return settings
