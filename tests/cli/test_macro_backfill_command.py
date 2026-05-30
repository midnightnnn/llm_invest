from __future__ import annotations

from arena.cli import build_parser


def test_backfill_macro_indicators_parser_options() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "backfill-macro-indicators",
            "--start-date",
            "2026-03-01",
            "--end-date",
            "2026-03-03",
            "--dry-run",
            "--append",
        ]
    )

    assert args.command == "backfill-macro-indicators"
    assert args.start_date == "2026-03-01"
    assert args.end_date == "2026-03-03"
    assert args.dry_run is True
    assert args.append is True
