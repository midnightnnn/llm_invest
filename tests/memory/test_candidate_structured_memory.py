from __future__ import annotations

import json

from arena.memory.candidate_structured import (
    build_structured_candidate_memory,
    enrich_candidate_memory_payload,
    format_candidate_memory_prompt_line,
)


def test_build_structured_candidate_memory_prefers_payload_without_truncating_fields() -> None:
    long_reason = "Ranker reason " + ("keeps full detail. " * 40)
    long_risk = "Risk detail " + ("also remains intact. " * 20)
    row = {
        "event_id": "mem_crwv",
        "created_date": "2026-05-12",
        "event_type": "candidate_watchlist",
        "summary": "CRWV candidate_watchlist: surfaced by recommend_opportunities:balanced rank=4; follow-up seen via fetch_sec_filings.",
        "payload_json": json.dumps(
            {
                "source": "candidate_discovery",
                "ticker": "CRWV",
                "candidate_status": "watchlist",
                "source_tools": ["recommend_opportunities:balanced"],
                "analyzed_by": ["fetch_sec_filings", "forecast_returns", "technical_signals"],
                "last_seen_rank": 4,
                "discovery_count": 2,
                "workflow_status": "analyzed",
                "evidence_level": "screen_and_analysis",
                "discovery_evidence": {
                    "score": 0.215288,
                    "reason_for": long_reason,
                    "reason_risk": long_risk,
                },
            },
            ensure_ascii=False,
        ),
    }

    structured = build_structured_candidate_memory(row)

    assert structured["quality"] == "payload_full"
    assert structured["event_id"] == "mem_crwv"
    assert structured["t"] == "CRWV"
    assert structured["type"] == "watchlist"
    assert structured["src"] == ["recommend_opportunities:balanced"]
    assert structured["checked"] == ["fetch_sec_filings", "forecast_returns", "technical_signals"]
    assert structured["rank"] == 4
    assert structured["score"] == 0.215288
    assert structured["why"] == long_reason
    assert structured["risk"] == long_risk
    assert "..." not in structured["why"]
    assert "..." not in structured["risk"]


def test_build_structured_candidate_memory_parses_summary_when_payload_is_missing() -> None:
    row = {
        "event_id": "mem_apd",
        "created_date": "2026-05-26",
        "event_type": "candidate_watchlist",
        "summary": (
            "APD candidate_watchlist: surfaced by recommend_opportunities:balanced "
            "rank=6 repeat=2 (score=0.156272); follow-up seen via fetch_sec_filings, "
            "forecast_returns. Reason: valuation and quality setup. Risk: weak momentum."
        ),
        "payload_json": "{}",
    }

    structured = build_structured_candidate_memory(row)

    assert structured["quality"] == "summary_parsed"
    assert structured["t"] == "APD"
    assert structured["type"] == "watchlist"
    assert structured["src"] == ["recommend_opportunities:balanced"]
    assert structured["rank"] == 6
    assert structured["score"] == 0.156272
    assert structured["checked"] == ["fetch_sec_filings", "forecast_returns"]
    assert structured["why"] == "valuation and quality setup."
    assert structured["risk"] == "weak momentum."


def test_enrich_candidate_memory_payload_preserves_original_payload_and_marks_quality() -> None:
    payload = {
        "source": "candidate_discovery",
        "ticker": "EOG",
        "source_tools": ["recommend_opportunities:balanced"],
        "discovery_evidence": {"score": 0.175339, "reason_for": "energy setup"},
    }
    row = {
        "event_id": "mem_eog",
        "created_date": "2026-05-26",
        "event_type": "candidate_watchlist",
        "summary": "EOG candidate_watchlist: surfaced by recommend_opportunities:balanced rank=5.",
        "payload_json": json.dumps(payload),
    }

    enriched, structured = enrich_candidate_memory_payload(row)

    assert enriched["ticker"] == "EOG"
    assert enriched["discovery_evidence"] == {"score": 0.175339, "reason_for": "energy setup"}
    assert enriched["structured_memory"] == structured
    assert structured["quality"] == "payload_full"


def test_format_candidate_memory_prompt_line_outputs_compact_json_without_summary_truncation() -> None:
    long_reason = "candidate reason " + ("full detail " * 30)
    row = {
        "event_id": "mem_cost",
        "created_date": "2026-05-26",
        "event_type": "candidate_watchlist",
        "summary": "COST candidate_watchlist: short text that should not be used as the only prompt payload.",
        "payload_json": json.dumps(
            {
                "source": "candidate_discovery",
                "ticker": "COST",
                "candidate_status": "watchlist",
                "source_tools": ["recommend_opportunities:balanced"],
                "analyzed_by": ["fetch_sec_filings", "forecast_returns"],
                "last_seen_rank": 3,
                "discovery_evidence": {"score": 0.234674, "reason_for": long_reason},
            },
            ensure_ascii=False,
        ),
    }

    line = format_candidate_memory_prompt_line(row)
    payload = json.loads(line.removeprefix("- "))

    assert payload["t"] == "COST"
    assert payload["checked"] == ["fetch_sec_filings", "forecast_returns"]
    assert payload["why"] == long_reason
    assert "forecast_retu..." not in line
    assert "..." not in payload["why"]
