from __future__ import annotations

import json
from datetime import datetime, timezone

from arena.memory.policy import MEMORY_POLICY_CONFIG_KEY, normalize_memory_policy
from arena.memory.semantic_tuning import run_memory_relation_tuner


class _FakeRepo:
    dataset_fqn = "proj.ds"

    def __init__(self, *, runs=None, triples=None, previous_state=None) -> None:
        self.runs = list(runs or [])
        self.triples = list(triples or [])
        self.previous_state = dict(previous_state or {})
        self.configs = []
        self.tuning_rows = []

    def fetch_rows(self, sql: str, params: dict):
        _ = params
        if "memory_relation_extraction_runs" in sql:
            return list(self.runs)
        if "memory_relation_triples" in sql:
            return list(self.triples)
        return []

    def get_config(self, tenant_id: str, config_key: str):
        _ = tenant_id
        if config_key == "memory_relation_tuning_state":
            return json.dumps(self.previous_state)
        return None

    def set_config(self, tenant_id: str, config_key: str, value: str, updated_by=None):
        self.configs.append((tenant_id, config_key, value, updated_by))

    def append_memory_relation_tuning_runs(self, rows, *, tenant_id=None):
        self.tuning_rows.append((tenant_id, list(rows)))


class _Settings:
    trading_mode = "paper"
    context_max_memory_events = 4
    memory_compaction_enabled = True
    memory_compaction_cycle_event_limit = 12
    memory_compaction_recent_lessons_limit = 4
    memory_compaction_max_reflections = 3
    memory_policy = {}


def _policy(mode: str = "shadow"):
    return normalize_memory_policy(
        {
            "graph": {
                "semantic_triples": {
                    "mode": mode,
                    "max_relation_context_items": 2,
                    "max_candidates": 2,
                    "tuning": {
                        "enabled": True,
                        "auto_transition_enabled": True,
                        "min_sources": 3,
                        "min_accepted_triples": 4,
                        "required_healthy_evaluations": 2,
                        "demote_unhealthy_evaluations": 2,
                    },
                }
            }
        }
    )


def _healthy_rows():
    now = datetime(2026, 3, 29, 12, tzinfo=timezone.utc)
    runs = [
        {
            "run_id": f"run_{idx}",
            "started_at": now,
            "source_table": "agent_memory_events",
            "source_id": f"evt_{idx}",
            "status": "success",
            "accepted_count": 2 if idx == 0 else 1,
            "rejected_count": 0,
            "detail_json": {"rejected": []},
            "model": "openai/gpt-5.2",
            "provider": "gpt",
        }
        for idx in range(3)
    ]
    triples = [
        {
            "triple_id": f"triple_{idx}",
            "created_at": now,
            "source_table": "agent_memory_events",
            "source_id": f"evt_{idx % 3}",
            "subject_node_id": f"entity:catalyst:ai_{idx}",
            "subject_type": "catalyst",
            "predicate": "supports",
            "object_node_id": f"ticker:T{idx}",
            "object_type": "ticker",
            "confidence": 0.88,
            "extraction_method": "llm",
            "extraction_version": "semantic_relation_extractor_v1",
        }
        for idx in range(4)
    ]
    return runs, triples


def test_relation_tuner_auto_promotes_shadow_to_inject_after_consecutive_health() -> None:
    runs, triples = _healthy_rows()
    repo = _FakeRepo(
        runs=runs,
        triples=triples,
        previous_state={"history": {"consecutive_healthy": 1}},
    )
    settings = _Settings()

    state = run_memory_relation_tuner(repo, settings, tenant_id="tenant-a", policy=_policy("shadow"))

    assert state["effective_mode"] == "inject"
    assert state["transition"]["action"] == "auto_promote_to_inject"
    policy_writes = [item for item in repo.configs if item[1] == MEMORY_POLICY_CONFIG_KEY]
    assert policy_writes
    assert '"mode":"inject"' in policy_writes[0][2]
    assert repo.tuning_rows[0][1][0]["health_ok"] is True


def test_relation_tuner_auto_demotes_inject_on_consecutive_unhealthy_runs() -> None:
    now = datetime(2026, 3, 29, 12, tzinfo=timezone.utc)
    runs = [
        {
            "run_id": f"bad_{idx}",
            "started_at": now,
            "source_table": "agent_memory_events",
            "source_id": f"evt_bad_{idx}",
            "status": "invalid_output",
            "accepted_count": 0,
            "rejected_count": 1,
            "detail_json": {"rejected": [{"reason": "missing_triples_array"}]},
            "model": "openai/gpt-5.2",
            "provider": "gpt",
        }
        for idx in range(3)
    ]
    repo = _FakeRepo(
        runs=runs,
        triples=[],
        previous_state={"history": {"consecutive_unhealthy": 1}},
    )

    state = run_memory_relation_tuner(repo, _Settings(), tenant_id="tenant-a", policy=_policy("inject"))

    assert state["effective_mode"] == "shadow"
    assert state["transition"]["action"] == "auto_demote_to_shadow"
    policy_writes = [item for item in repo.configs if item[1] == MEMORY_POLICY_CONFIG_KEY]
    assert policy_writes
    assert '"mode":"shadow"' in policy_writes[0][2]


def test_relation_tuner_demotes_inject_on_version_change_even_when_healthy() -> None:
    runs, triples = _healthy_rows()
    repo = _FakeRepo(
        runs=runs,
        triples=triples,
        previous_state={
            "versions": {
                "extractor_version": "old",
                "prompt_version": "old",
                "ontology_version": "old",
                "models": [],
                "providers": [],
            },
            "history": {"consecutive_healthy": 10},
        },
    )

    state = run_memory_relation_tuner(repo, _Settings(), tenant_id="tenant-a", policy=_policy("inject"))

    assert state["effective_mode"] == "shadow"
    assert state["gates"]["version_changed"] is True
    assert state["transition"]["action"] == "auto_demote_to_shadow"


def test_relation_tuner_canonicalizes_model_provider_order_for_version_compare() -> None:
    runs, triples = _healthy_rows()
    runs.append(
        {
            **runs[0],
            "run_id": "run_extra",
            "source_id": "evt_extra",
            "model": "anthropic/claude-sonnet-4-6",
            "provider": "claude",
            "accepted_count": 1,
        }
    )
    triples.append(
        {
            **triples[0],
            "triple_id": "triple_extra",
            "source_id": "evt_extra",
            "object_node_id": "ticker:EXTRA",
        }
    )
    repo = _FakeRepo(
        runs=runs,
        triples=triples,
        previous_state={
            "versions": {
                "extractor_version": "semantic_relation_extractor_v1",
                "prompt_version": "semantic_relation_prompt_v2",
                "ontology_version": "semantic_relation_ontology_v1",
                "models": ["openai/gpt-5.2", "anthropic/claude-sonnet-4-6"],
                "providers": ["gpt", "claude"],
            },
            "history": {"consecutive_healthy": 10},
        },
    )

    state = run_memory_relation_tuner(repo, _Settings(), tenant_id="tenant-a", policy=_policy("inject"))

    assert state["gates"]["version_changed"] is False
    assert state["effective_mode"] == "inject"
    assert state["transition"]["action"] == ""


def test_relation_tuner_blocks_repromotion_during_post_demote_cooldown() -> None:
    runs, triples = _healthy_rows()
    repo = _FakeRepo(
        runs=runs,
        triples=triples,
        previous_state={
            "history": {
                "last_transition_action": "auto_demote_to_shadow",
                "evaluations_since_demotion": 0,
                "consecutive_healthy": 1,
            }
        },
    )

    state = run_memory_relation_tuner(repo, _Settings(), tenant_id="tenant-a", policy=_policy("shadow"))

    assert state["gates"]["demotion_cooldown_ok"] is False
    assert state["effective_mode"] == "shadow"
    assert state["transition"]["action"] == ""
