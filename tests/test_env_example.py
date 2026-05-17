from __future__ import annotations

import re
from pathlib import Path


def _env_example() -> str:
    return (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")


def _env_local_example() -> str:
    return (Path(__file__).resolve().parents[1] / ".env.local.example").read_text(encoding="utf-8")


def _active_env_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.add(stripped.split("=", 1)[0].strip())
    return keys


def test_env_example_tracks_current_local_ui_and_prep_defaults() -> None:
    text = _env_example()
    keys = _active_env_keys(text)

    assert "ARENA_MODE=local" in text
    assert "ARENA_LOCAL_DB_PATH=./data/arena.duckdb" in text
    assert "ARENA_UI_SETTINGS_ENABLED=true" in text
    assert "FRED_API_KEY=" in text
    assert "ECOS_API_KEY=" in text
    assert "ARENA_LOCAL_CREDENTIALS_FILE" not in keys
    assert "ARENA_SHARED_PREP_FORCE_MARKET_CLOSED" not in keys


def test_env_example_keeps_provider_models_as_runtime_config_fallbacks() -> None:
    text = _env_example()
    keys = _active_env_keys(text)

    assert "OPENAI_MODEL=" in text
    assert "GEMINI_MODEL=" in text
    assert "ANTHROPIC_MODEL=" in text
    assert "OPENAI_MODEL" not in keys
    assert "GEMINI_MODEL" not in keys
    assert "ANTHROPIC_MODEL" not in keys
    assert "OPENAI_API_KEY" not in keys
    assert "GEMINI_API_KEY" not in keys
    assert "ANTHROPIC_API_KEY" not in keys
    assert not re.search(r"^OPENAI_MODEL=gpt-", text, re.MULTILINE)
    assert not re.search(r"^GEMINI_MODEL=gemini-", text, re.MULTILINE)
    assert not re.search(r"^ANTHROPIC_MODEL=claude-", text, re.MULTILINE)


def test_env_example_keeps_tenant_credentials_out_of_active_env() -> None:
    keys = _active_env_keys(_env_example())

    assert "KIS_API_KEY" not in keys
    assert "KIS_API_SECRET" not in keys
    assert "KIS_PAPER_API_KEY" not in keys
    assert "KIS_PAPER_API_SECRET" not in keys
    assert "KIS_ACCOUNT_NO" not in keys


def test_env_local_example_uses_db_backed_credentials_and_process_macro_keys() -> None:
    text = _env_local_example()
    keys = _active_env_keys(text)

    assert "ARENA_MODE=local" in text
    assert "ARENA_UI_SETTINGS_ENABLED=true" in text
    assert "FRED_API_KEY=" in text
    assert "ECOS_API_KEY=" in text
    assert "This does not auto-create ~/.llm-arena/credentials.json" in text
    assert "OPENAI_API_KEY" not in keys
    assert "GEMINI_API_KEY" not in keys
    assert "ANTHROPIC_API_KEY" not in keys
    assert "KIS_PAPER_API_KEY" not in keys
    assert "KIS_PAPER_API_SECRET" not in keys
