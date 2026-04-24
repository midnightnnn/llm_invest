from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import arena.memory.semantic_extractor as semantic_module
from arena.config import Settings
from arena.memory.relation_validation import RelationSource, validate_extracted_relations
from arena.memory.semantic_extractor import SemanticRelationExtractor


class _FakeRepo:
    def __init__(self) -> None:
        self.pending_rows = [
            {
                "tenant_id": "tenant-a",
                "source_table": "agent_memory_events",
                "source_id": "evt_1",
                "source_node_id": "mem:evt_1",
                "source_created_at": datetime(2026, 3, 29, 1, 2, tzinfo=timezone.utc),
                "agent_id": "gpt",
                "trading_mode": "paper",
                "cycle_id": "cycle_1",
                "source_label": "NVDA note",
                "source_text": "AI demand supports NVDA margin recovery.",
                "source_hash": "hash_1",
            }
        ]
        self.pending_calls = []
        self.triples = []
        self.runs = []

    def relation_extraction_pending_sources(self, **kwargs):
        self.pending_calls.append(kwargs)
        return list(self.pending_rows)

    def upsert_memory_relation_triples_with_graph(self, rows, *, tenant_id=None):
        self.triples.append((tenant_id, list(rows)))

    def append_memory_relation_extraction_runs(self, rows, *, tenant_id=None):
        self.runs.append((tenant_id, list(rows)))


def _settings() -> Settings:
    settings = MagicMock(spec=Settings)
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.provider_secrets = {}
    settings.openai_api_key = "test-openai-key"
    settings.openai_model = "gpt-5.2"
    settings.gemini_api_key = ""
    settings.gemini_model = "models/gemini-2.5-flash"
    settings.anthropic_api_key = ""
    settings.anthropic_model = "claude-sonnet-4-6"
    settings.anthropic_use_vertexai = False
    settings.trading_mode = "paper"
    settings.llm_timeout_seconds = 10
    return settings


def _source(text: str = "AI demand supports NVDA margin recovery.") -> RelationSource:
    return RelationSource(
        tenant_id="tenant-a",
        source_table="agent_memory_events",
        source_id="evt_1",
        source_node_id="mem:evt_1",
        source_created_at=datetime(2026, 3, 29, 1, 2, tzinfo=timezone.utc),
        agent_id="gpt",
        trading_mode="paper",
        cycle_id="cycle_1",
        source_label="NVDA note",
        source_text=text,
        source_hash="hash_1",
    )


def test_validate_extracted_relations_accepts_grounded_triple() -> None:
    result = validate_extracted_relations(
        [
            {
                "subject": {"label": "AI demand", "type": "catalyst"},
                "predicate": "supports",
                "object": {"label": "NVDA", "type": "ticker"},
                "confidence": 0.86,
                "evidence_text": "AI demand supports NVDA margin recovery.",
            }
        ],
        source=_source(),
        extractor_version="semantic_relation_extractor_v1",
        prompt_version="semantic_relation_prompt_v1",
    )

    assert len(result.accepted) == 1
    row = result.accepted[0]
    assert row["subject_node_id"] == "entity:catalyst:ai_demand"
    assert row["predicate"] == "supports"
    assert row["object_node_id"] == "ticker:NVDA"
    assert row["extraction_method"] == "llm"
    assert row["detail_json"]["confidence_threshold"] == 0.7
    assert result.rejected == []


def test_validate_extracted_relations_accepts_minor_evidence_rephrase() -> None:
    result = validate_extracted_relations(
        [
            {
                "subject": {"label": "AI demand", "type": "catalyst"},
                "predicate": "supports",
                "object": {"label": "NVDA", "type": "ticker"},
                "confidence": 0.86,
                "evidence_text": "AI demand supported NVDA margin recovery",
            }
        ],
        source=_source(),
        extractor_version="semantic_relation_extractor_v1",
        prompt_version="semantic_relation_prompt_v1",
    )

    assert len(result.accepted) == 1
    assert result.rejected == []


def test_validate_extracted_relations_accepts_non_ascii_concept_labels() -> None:
    result = validate_extracted_relations(
        [
            {
                "subject": {"label": "수출 규제", "type": "risk"},
                "predicate": "risk_to",
                "object": {"label": "NVDA", "type": "ticker"},
                "confidence": 0.86,
                "evidence_text": "수출 규제는 NVDA 매출에 부담이다.",
            }
        ],
        source=_source("수출 규제는 NVDA 매출에 부담이다."),
        extractor_version="semantic_relation_extractor_v1",
        prompt_version="semantic_relation_prompt_v1",
    )

    assert len(result.accepted) == 1
    assert result.accepted[0]["subject_node_id"] == "entity:risk:수출_규제"
    assert result.rejected == []


def test_validate_extracted_relations_applies_predicate_confidence_threshold() -> None:
    result = validate_extracted_relations(
        [
            {
                "subject": {"label": "Guidance cut", "type": "risk"},
                "predicate": "invalidates",
                "object": {"label": "margin recovery thesis", "type": "thesis"},
                "confidence": 0.86,
                "evidence_text": "Guidance cut invalidates margin recovery thesis.",
            },
            {
                "subject": {"label": "Guidance cut", "type": "risk"},
                "predicate": "invalidates",
                "object": {"label": "margin recovery thesis", "type": "thesis"},
                "confidence": 0.90,
                "evidence_text": "Guidance cut invalidates margin recovery thesis.",
            },
        ],
        source=_source("Guidance cut invalidates margin recovery thesis."),
        extractor_version="semantic_relation_extractor_v1",
        prompt_version="semantic_relation_prompt_v1",
        min_confidence=0.65,
    )

    assert [item.reason for item in result.rejected] == ["low_confidence"]
    assert len(result.accepted) == 1
    assert result.accepted[0]["detail_json"]["confidence_threshold"] == 0.88


def test_validate_extracted_relations_rejects_unsupported_or_ungrounded() -> None:
    result = validate_extracted_relations(
        [
            {
                "subject": {"label": "AI demand", "type": "catalyst"},
                "predicate": "magically_predicts",
                "object": {"label": "NVDA", "type": "ticker"},
                "confidence": 0.9,
                "evidence_text": "AI demand supports NVDA margin recovery.",
            },
            {
                "subject": {"label": "AI demand", "type": "catalyst"},
                "predicate": "supports",
                "object": {"label": "NVDA", "type": "ticker"},
                "confidence": 0.9,
                "evidence_text": "This sentence is not in the source.",
            },
        ],
        source=_source(),
        extractor_version="semantic_relation_extractor_v1",
        prompt_version="semantic_relation_prompt_v1",
    )

    assert result.accepted == []
    assert [item.reason for item in result.rejected] == ["unsupported_predicate", "evidence_not_found"]


def test_semantic_relation_extractor_runs_pending_and_writes_shadow_rows(monkeypatch) -> None:
    repo = _FakeRepo()
    settings = _settings()
    captured = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"triples":[{"subject":{"label":"AI demand","type":"catalyst"},'
                            '"predicate":"supports","object":{"label":"NVDA","type":"ticker"},'
                            '"confidence":0.86,"evidence_text":"AI demand supports NVDA margin recovery."}]}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr(semantic_module.litellm, "acompletion", _fake_acompletion)
    extractor = SemanticRelationExtractor(settings=settings, repo=repo)

    rows = asyncio.run(extractor.run_pending(tenant_id="tenant-a", limit=3, dry_run=False))

    assert rows[0]["accepted_count"] == 1
    assert repo.pending_calls[0]["extractor_version"] == "semantic_relation_extractor_v1"
    assert repo.triples[0][0] == "tenant-a"
    assert repo.triples[0][1][0]["object_node_id"] == "ticker:NVDA"
    assert repo.runs[0][1][0]["status"] == "success"
    assert captured["model"] == "openai/gpt-5.2"
    assert "temperature" not in captured


def test_semantic_relation_extractor_omits_temperature_for_claude(monkeypatch) -> None:
    repo = _FakeRepo()
    settings = _settings()
    settings.agent_ids = ["claude"]
    settings.openai_api_key = ""
    settings.anthropic_api_key = "test-anthropic-key"
    captured = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": '{"triples":[]}'}}]}

    monkeypatch.setattr(semantic_module.litellm, "acompletion", _fake_acompletion)
    extractor = SemanticRelationExtractor(settings=settings, repo=repo)

    rows = asyncio.run(extractor.run_pending(tenant_id="tenant-a", limit=1, dry_run=False))

    assert rows[0]["status"] == "success"
    assert captured["model"] == "anthropic/claude-sonnet-4-6"
    assert captured["api_key"] == "test-anthropic-key"
    assert "temperature" not in captured


def test_semantic_relation_extractor_dry_run_skips_writes(monkeypatch) -> None:
    repo = _FakeRepo()

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        return {"choices": [{"message": {"content": '{"triples":[]}'}}]}

    monkeypatch.setattr(semantic_module.litellm, "acompletion", _fake_acompletion)
    extractor = SemanticRelationExtractor(settings=_settings(), repo=repo)

    rows = asyncio.run(extractor.run_pending(tenant_id="tenant-a", limit=1, dry_run=True))

    assert rows[0]["dry_run"] is True
    assert repo.triples == []
    assert repo.runs == []


def test_semantic_relation_extractor_marks_invalid_output_retryable(monkeypatch) -> None:
    repo = _FakeRepo()

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        return {"choices": [{"message": {"content": '{"not_triples":[]}'}}]}

    monkeypatch.setattr(semantic_module.litellm, "acompletion", _fake_acompletion)
    extractor = SemanticRelationExtractor(settings=_settings(), repo=repo)

    rows = asyncio.run(extractor.run_pending(tenant_id="tenant-a", limit=1, dry_run=False))

    assert rows[0]["status"] == "invalid_output"
    assert repo.triples == []
    assert repo.runs[0][1][0]["status"] == "invalid_output"
