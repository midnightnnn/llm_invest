from __future__ import annotations

import pytest

import arena.cli as cli
from arena.config import load_settings


def test_build_parser_supports_smoke_thesis_compaction_command() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        ["smoke-thesis-compaction", "--cycle-id", "cycle_123", "--agent", "gpt", "--agent", "claude", "--save"]
    )

    assert args.command == "smoke-thesis-compaction"
    assert args.cycle_id == "cycle_123"
    assert args.agent == ["gpt", "claude"]
    assert args.timeout == 0
    assert args.save is True


def test_cmd_smoke_thesis_compaction_requires_cycle_id(monkeypatch) -> None:
    settings = load_settings()
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_smoke_thesis_compaction("")

    assert exc_info.value.code == 2


def test_cmd_smoke_thesis_compaction_runs_preview_and_prints_json(monkeypatch, capsys) -> None:
    settings = load_settings()
    settings.openai_api_key = "test-openai-key"
    settings.agent_ids = ["gpt", "claude"]
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings: type("Repo", (), {"ensure_dataset": lambda self: None, "ensure_tables": lambda self: None})())

    async def _fake_run_thesis_compaction_smoke(**kwargs):
        assert kwargs["cycle_id"] == "cycle_123"
        assert kwargs["agent_ids"] == ["gpt", "claude"]
        assert kwargs["save"] is False
        return [{"agent_id": "gpt", "reflections": [{"summary": "preview ok"}]}]

    monkeypatch.setattr(cli, "_run_thesis_compaction_smoke", _fake_run_thesis_compaction_smoke)

    cli.cmd_smoke_thesis_compaction("cycle_123", timeout_seconds=9)

    out = capsys.readouterr().out
    assert '"cycle_id": "cycle_123"' in out
    assert '"mode": "preview"' in out
    assert '"summary": "preview ok"' in out
    assert settings.llm_timeout_seconds == 9


def test_cmd_smoke_thesis_compaction_passes_save_flag(monkeypatch, capsys) -> None:
    settings = load_settings()
    settings.openai_api_key = "test-openai-key"
    settings.agent_ids = ["gpt"]
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings: type("Repo", (), {"ensure_dataset": lambda self: None, "ensure_tables": lambda self: None})())

    async def _fake_run_thesis_compaction_smoke(**kwargs):
        assert kwargs["save"] is True
        return [{"agent_id": "gpt", "saved": {"summary": "saved ok"}}]

    monkeypatch.setattr(cli, "_run_thesis_compaction_smoke", _fake_run_thesis_compaction_smoke)

    cli.cmd_smoke_thesis_compaction("cycle_123", agent_ids=["gpt"], save=True)

    out = capsys.readouterr().out
    assert '"mode": "save"' in out
    assert '"summary": "saved ok"' in out
