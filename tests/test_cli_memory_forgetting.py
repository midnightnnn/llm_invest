from __future__ import annotations

import arena.cli as cli


def test_build_parser_supports_enable_memory_forgetting_command() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        ["enable-memory-forgetting", "--tenant", "alpha", "--tenant", "beta", "--updated-by", "ops@example.com"]
    )

    assert args.command == "enable-memory-forgetting"
    assert args.tenant == ["alpha", "beta"]
    assert args.updated_by == "ops@example.com"


def test_build_parser_supports_run_memory_forgetting_tuner_command() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        ["run-memory-forgetting-tuner", "--tenant", "alpha", "--updated-by", "cron@example.com"]
    )

    assert args.command == "run-memory-forgetting-tuner"
    assert args.tenant == ["alpha"]
    assert args.updated_by == "cron@example.com"


def test_cli_exports_post_cycle_memory_forgetting_helper() -> None:
    assert callable(cli._run_memory_forgetting_tuner_post_cycle)
