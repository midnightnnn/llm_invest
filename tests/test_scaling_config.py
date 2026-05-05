from __future__ import annotations

from pathlib import Path

from arena.cli import build_parser
from arena.config import load_settings


def test_scaled_universe_and_ranker_defaults(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_UNIVERSE_RUN_TOP_N", raising=False)
    monkeypatch.delenv("ARENA_UNIVERSE_PER_EXCHANGE_CAP", raising=False)
    monkeypatch.delenv("ARENA_OPPORTUNITY_RANKER_MAX_SCORING_ROWS", raising=False)
    monkeypatch.delenv("ARENA_FORECAST_RANKER_TOP_PER_BUCKET", raising=False)
    monkeypatch.delenv("ARENA_FORECAST_MAX_TICKERS", raising=False)

    settings = load_settings()

    assert settings.universe_run_top_n == 1000
    assert settings.universe_per_exchange_cap == 500
    assert settings.opportunity_ranker_max_scoring_rows == 1000
    assert settings.forecast_ranker_top_per_bucket == 10
    assert settings.forecast_max_tickers == 80


def test_build_ranker_cli_default_scores_scaled_universe() -> None:
    parser = build_parser()

    args = parser.parse_args(["build-opportunity-ranker"])

    assert args.max_scoring_rows == 1000


def test_cloud_run_deploy_defaults_match_scaled_universe() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "deploy_cloud_run_job.sh"
    ).read_text(encoding="utf-8")

    assert "ARENA_UNIVERSE_RUN_TOP_N=1000" in script
    assert "ARENA_UNIVERSE_PER_EXCHANGE_CAP=500" in script
    assert "ARENA_OPPORTUNITY_RANKER_MAX_SCORING_ROWS=1000" in script
    assert "ARENA_FORECAST_RANKER_TOP_PER_BUCKET=10" in script
    assert "ARENA_FORECAST_MAX_TICKERS=80" in script
    assert 'PREP_TASK_TIMEOUT="${PREP_TASK_TIMEOUT:-7200s}"' in script
