from __future__ import annotations

import arena.cli as cli


def test_build_parser_supports_extract_memory_relations_command() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "extract-memory-relations",
            "--tenant",
            "tenant-a",
            "--limit",
            "5",
            "--source-table",
            "agent_memory_events",
            "--event-type",
            "strategy_reflection",
            "--dry-run",
            "--provider",
            "gpt",
            "--model",
            "gpt-5.2",
            "--min-confidence",
            "0.7",
            "--max-triples",
            "4",
        ]
    )

    assert args.command == "extract-memory-relations"
    assert args.tenant == ["tenant-a"]
    assert args.limit == 5
    assert args.source_table == "agent_memory_events"
    assert args.event_type == ["strategy_reflection"]
    assert args.dry_run is True
    assert args.provider == "gpt"
    assert args.model == "gpt-5.2"
    assert args.min_confidence == 0.7
    assert args.max_triples == 4
