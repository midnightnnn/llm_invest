from __future__ import annotations

import sys
from types import SimpleNamespace

import arena.cli as cli
from arena.config import load_settings


class _Repo:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ensure_dataset(self) -> None:
        self.calls.append("ensure_dataset")

    def ensure_tables(self) -> None:
        self.calls.append("ensure_tables")


def _patch_serve_ui_dependencies(monkeypatch, repo: _Repo):
    settings = load_settings()
    settings.arena_mode = "gcp"
    served: dict[str, object] = {}

    def _fake_serve_ui(**kwargs):
        served.update(kwargs)

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings: repo)
    monkeypatch.setitem(sys.modules, "arena.ui.server", SimpleNamespace(serve_ui=_fake_serve_ui))
    return served


def test_cmd_serve_ui_skips_schema_ensure_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_UI_ENSURE_SCHEMA_ON_STARTUP", raising=False)
    repo = _Repo()
    served = _patch_serve_ui_dependencies(monkeypatch, repo)

    cli.cmd_serve_ui()

    assert repo.calls == []
    assert served["repo"] is repo


def test_cmd_serve_ui_can_ensure_schema_on_startup(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_UI_ENSURE_SCHEMA_ON_STARTUP", "true")
    repo = _Repo()
    served = _patch_serve_ui_dependencies(monkeypatch, repo)

    cli.cmd_serve_ui()

    assert repo.calls == ["ensure_dataset", "ensure_tables"]
    assert served["repo"] is repo
