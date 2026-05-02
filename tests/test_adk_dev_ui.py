from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def test_build_dev_agent_forces_local_read_only_defaults(monkeypatch, tmp_path) -> None:
    from arena.config import AgentConfig, load_settings
    from arena.tools.registry import ToolRegistry

    from arena.agents.dev_ui import _factory

    calls: dict[str, object] = {}
    settings = load_settings()
    settings.arena_mode = "gcp"
    settings.allow_live_trading = True
    settings.agent_configs["gpt"] = AgentConfig(
        agent_id="gpt",
        provider="gpt",
        model="gpt-test",
        capital_krw=1_000_000,
    )

    class _Repo:
        def ensure_tables(self) -> None:
            calls["ensured"] = True

        def get_config(self, tenant_id: str, config_key: str) -> None:
            calls.setdefault("config_reads", []).append((tenant_id, config_key))
            return None

    def fake_build_agent(**kwargs):
        calls["build_agent"] = kwargs
        return SimpleNamespace(name=f"{kwargs['agent_id']}_react")

    monkeypatch.setattr(_factory, "load_settings", lambda: settings)
    monkeypatch.delenv("ARENA_LOCAL_DB_PATH", raising=False)
    monkeypatch.setattr(_factory, "_repo_default_local_db_path", lambda: tmp_path / "missing.duckdb")

    def fake_get_repository(settings, tenant_id):
        calls["local_db_path"] = _factory.os.getenv("ARENA_LOCAL_DB_PATH")
        return _Repo()

    monkeypatch.setattr(_factory, "get_repository", fake_get_repository)
    monkeypatch.setattr(_factory, "apply_runtime_overrides", lambda settings, repo, tenant_id: settings)
    monkeypatch.setattr(_factory, "build_default_registry", lambda repo, settings, tenant_id: ToolRegistry([]))
    monkeypatch.setattr(_factory, "resolve_adk_tools", lambda **kwargs: SimpleNamespace(adk_tools=[]))
    monkeypatch.setattr(_factory, "resolve_max_tool_events", lambda settings: 17)
    monkeypatch.setattr(_factory, "build_agent", fake_build_agent)

    agent = _factory.build_dev_agent("gpt")

    assert agent.name == "gpt_react"
    assert calls["ensured"] is True
    build_kwargs = calls["build_agent"]
    assert build_kwargs["provider"] == "gpt"
    assert build_kwargs["tenant_id"] == "local"
    assert build_kwargs["max_tool_events"] == 17
    assert build_kwargs["agent_config"] is settings.agent_configs["gpt"]
    assert settings.arena_mode == "local"
    assert settings.allow_live_trading is False
    assert settings.trading_mode == "paper"
    assert calls["local_db_path"] == str(tmp_path / "missing.duckdb")


def test_default_local_db_path_uses_repo_clone_path(monkeypatch, tmp_path) -> None:
    from arena.agents.dev_ui import _factory

    cloned_db = tmp_path / "data" / "arena.duckdb"
    monkeypatch.delenv("ARENA_LOCAL_DB_PATH", raising=False)
    monkeypatch.setattr(_factory, "_repo_default_local_db_path", lambda: cloned_db)

    _factory._ensure_default_local_db_path()

    assert _factory.os.environ["ARENA_LOCAL_DB_PATH"] == str(cloned_db)


def test_provider_modules_export_root_agent(monkeypatch) -> None:
    from arena.agents.dev_ui import _factory

    created: list[str] = []

    def fake_build_dev_agent(provider: str):
        created.append(provider)
        return SimpleNamespace(name=f"{provider}_react")

    monkeypatch.setattr(_factory, "build_dev_agent", fake_build_dev_agent)

    for module_name, provider in [
        ("arena.agents.dev_ui.gemini", "gemini"),
        ("arena.agents.dev_ui.gpt", "gpt"),
        ("arena.agents.dev_ui.claude", "claude"),
        ("arena.agents.dev_ui.gemini.agent", "gemini"),
        ("arena.agents.dev_ui.gpt.agent", "gpt"),
        ("arena.agents.dev_ui.claude.agent", "claude"),
    ]:
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
        assert module.root_agent.name == f"{provider}_react"

    assert created == ["gemini", "gpt", "claude", "gemini", "gpt", "claude"]


def test_provider_package_imports_when_adk_loads_agent_dir_as_top_level(monkeypatch) -> None:
    dev_ui_dir = Path(__file__).resolve().parents[1] / "arena" / "agents" / "dev_ui"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(dev_ui_dir),
            "ARENA_DEV_UI_ENSURE_TABLES": "0",
            "ARENA_LOCAL_DB_PATH": "/tmp/arena-adk-dev-ui-import-test.duckdb",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": "/tmp",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", "import gemini; print(gemini.root_agent.name)"],
        cwd="/tmp",
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "gemini_react" in result.stdout
