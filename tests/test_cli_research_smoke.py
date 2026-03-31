from __future__ import annotations

import pytest

import arena.cli as cli
from arena.config import load_settings


def test_build_parser_supports_smoke_research_command() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(["smoke-research", "--provider", "gpt"])

    assert args.command == "smoke-research"
    assert args.provider == "gpt"
    assert args.prompt == ""
    assert args.timeout == 0


def test_cmd_smoke_research_requires_provider_credentials(monkeypatch) -> None:
    settings = load_settings()
    settings.openai_api_key = ""
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_smoke_research("gpt")

    assert exc_info.value.code == 2


def test_cmd_smoke_research_runs_and_prints_response(monkeypatch, capsys) -> None:
    settings = load_settings()
    settings.openai_api_key = "test-openai-key"
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)

    async def _fake_run_research_smoke(**kwargs):
        _ = kwargs
        return "macro summary ok"

    monkeypatch.setattr(cli, "_run_research_smoke", _fake_run_research_smoke)

    cli.cmd_smoke_research("gpt", prompt="test prompt", timeout_seconds=7)

    out = capsys.readouterr().out
    assert "macro summary ok" in out
    assert settings.llm_timeout_seconds == 7
