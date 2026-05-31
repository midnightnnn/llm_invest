from __future__ import annotations

import json

from arena.memory.candidate_structured import backfill_candidate_memory_structures


class _Repo:
    def __init__(self) -> None:
        self.rows = [
            {
                "event_id": "mem_payload",
                "created_date": "2026-05-26",
                "event_type": "candidate_watchlist",
                "summary": "EOG candidate_watchlist: surfaced by recommend_opportunities:balanced rank=5.",
                "score": 0.38,
                "importance_score": 0.38,
                "payload_json": json.dumps(
                    {
                        "source": "candidate_discovery",
                        "ticker": "EOG",
                        "source_tools": ["recommend_opportunities:balanced"],
                        "last_seen_rank": 5,
                        "discovery_evidence": {
                            "score": 0.175339,
                            "reason_for": "energy tailwind",
                        },
                    }
                ),
            },
            {
                "event_id": "mem_summary",
                "created_date": "2026-05-26",
                "event_type": "candidate_watchlist",
                "summary": (
                    "APD candidate_watchlist: surfaced by recommend_opportunities:balanced "
                    "rank=6 (score=0.156272); follow-up seen via forecast_returns."
                ),
                "score": 0.38,
                "importance_score": 0.38,
                "payload_json": "{}",
            },
        ]
        self.updated: list[dict] = []

    def candidate_memory_events_for_structured_backfill(self, **kwargs):
        _ = kwargs
        return list(self.rows)

    def update_memory_event(self, **kwargs) -> None:
        self.updated.append(kwargs)


def test_backfill_candidate_memory_structures_updates_payload_rows_and_marks_quality() -> None:
    repo = _Repo()

    result = backfill_candidate_memory_structures(
        repo,
        agent_ids=["gpt"],
        trading_mode="paper",
        dry_run=False,
    )

    assert result.scanned == 2
    assert result.updated == 2
    assert result.quality_counts == {"payload_full": 1, "summary_parsed": 1}
    assert len(repo.updated) == 2
    payload_update = repo.updated[0]["payload"]
    assert payload_update["structured_memory"]["quality"] == "payload_full"
    assert payload_update["structured_memory"]["why"] == "energy tailwind"
    summary_update = repo.updated[1]["payload"]
    assert summary_update["structured_memory"]["quality"] == "summary_parsed"
    assert summary_update["structured_memory"]["checked"] == ["forecast_returns"]


def test_backfill_candidate_memory_structures_dry_run_does_not_update() -> None:
    repo = _Repo()

    result = backfill_candidate_memory_structures(
        repo,
        agent_ids=["gpt"],
        trading_mode="paper",
        dry_run=True,
    )

    assert result.scanned == 2
    assert result.updated == 0
    assert result.would_update == 2
    assert repo.updated == []
