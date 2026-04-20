from __future__ import annotations

from arena.data.schema import parse_ddl_columns, render_table_ddls
from arena.memory.policy import (
    default_memory_policy,
    memory_forgetting_enabled,
    memory_graph_enabled,
    memory_graph_semantic_triples_boost_enabled,
    memory_graph_semantic_triples_inject_enabled,
    memory_graph_semantic_triples_mode,
    memory_hierarchy_enabled,
    memory_tagging_enabled,
    normalize_memory_policy,
)


def test_schema_includes_phase0_memory_columns_and_tables() -> None:
    ddls = "\n".join(render_table_ddls("proj", "ds"))
    cols = parse_ddl_columns()

    assert "proj.ds.memory_access_events" in ddls
    assert "proj.ds.memory_graph_nodes" in ddls
    assert "proj.ds.memory_graph_edges" in ddls
    assert "proj.ds.memory_relation_extraction_runs" in ddls
    assert "proj.ds.memory_relation_tuning_runs" in ddls
    assert "proj.ds.memory_relation_triples" in ddls
    assert "proj.ds.agent_llm_interactions" in ddls
    assert "proj.ds.agent_llm_tool_events" in ddls
    assert "proj.ds.agent_llm_context_refs" in ddls
    assert "proj.ds.agent_llm_artifact_links" in ddls
    assert ("cycle_id", "STRING") in cols["agent_order_intents"]
    assert ("llm_call_id", "STRING") in cols["agent_order_intents"]
    assert ("cycle_id", "STRING") in cols["agent_memory_events"]
    assert ("llm_call_id", "STRING") in cols["agent_memory_events"]
    assert ("memory_tier", "STRING") in cols["agent_memory_events"]
    assert ("context_tags_json", "JSON") in cols["agent_memory_events"]
    assert ("access_count", "INT64") in cols["agent_memory_events"]
    assert ("effective_score", "FLOAT64") in cols["agent_memory_events"]
    assert ("causal_chain_id", "STRING") in cols["agent_memory_events"]
    assert ("predicate", "STRING") in cols["memory_relation_triples"]
    assert ("evidence_text", "STRING") in cols["memory_relation_triples"]
    assert ("detail_json", "JSON") in cols["memory_relation_triples"]
    assert ("source_hash", "STRING") in cols["memory_relation_extraction_runs"]
    assert ("raw_output_json", "JSON") in cols["memory_relation_extraction_runs"]
    assert ("recommended_mode", "STRING") in cols["memory_relation_tuning_runs"]
    assert ("health_ok", "BOOL") in cols["memory_relation_tuning_runs"]
    assert ("context_payload_json", "JSON") in cols["agent_llm_interactions"]
    assert ("available_tools_json", "JSON") in cols["agent_llm_interactions"]
    assert ("model_visible_result_json", "JSON") in cols["agent_llm_tool_events"]
    assert ("source_hash", "STRING") in cols["agent_llm_context_refs"]
    assert ("artifact_table", "STRING") in cols["agent_llm_artifact_links"]


def test_default_memory_policy_includes_phase0_defaults() -> None:
    policy = default_memory_policy()

    assert policy["hierarchy"]["enabled"] is True
    assert policy["tagging"]["enabled"] is True
    assert policy["forgetting"]["enabled"] is True
    assert policy["forgetting"]["access_log_enabled"] is True
    assert policy["forgetting"]["access_curve"] == "sqrt"
    assert policy["forgetting"]["tuning"]["enabled"] is True
    assert policy["forgetting"]["tuning"]["mode"] == "shadow"
    assert policy["forgetting"]["tuning"]["auto_promote_enabled"] is False
    assert policy["forgetting"]["tuning"]["auto_demote_enabled"] is False
    assert policy["retrieval"]["reranking"]["effective_score_bonus_scale"] == 0.08
    assert policy["retrieval"]["reranking"]["effective_score_bonus_cap"] == 0.08
    assert policy["graph"]["enabled"] is True
    assert policy["graph"]["semantic_triples"]["enabled"] is True
    assert policy["graph"]["semantic_triples"]["mode"] == "shadow"
    assert policy["graph"]["semantic_triples"]["tuning"]["enabled"] is True
    assert policy["graph"]["semantic_triples"]["tuning"]["auto_transition_enabled"] is True
    assert policy["graph"]["semantic_triples"]["tuning"]["post_demote_cooldown_evaluations"] == 0
    assert memory_graph_semantic_triples_boost_enabled(policy) is False
    assert memory_graph_semantic_triples_inject_enabled(policy) is False


def test_normalize_memory_policy_accepts_phase0_overrides() -> None:
    policy = normalize_memory_policy(
        {
            "hierarchy": {"enabled": True, "working_ttl_hours": 48},
            "tagging": {"enabled": True, "max_tags": 8, "regime_bonus": 0.3},
            "forgetting": {
                "enabled": True,
                "access_log_enabled": True,
                "default_decay_factor": 0.99,
                "access_curve": "log",
                "tier_weight_semantic": 0.25,
                "tuning": {
                    "enabled": True,
                    "mode": "bounded_ema",
                    "ema_alpha": 0.2,
                    "auto_promote_enabled": True,
                    "auto_promote_min_shadow_runs": 6,
                    "auto_demote_enabled": True,
                    "auto_demote_unhealthy_runs": 2,
                },
            },
            "graph": {
                "enabled": True,
                "max_expansion_hops": 2,
                "max_expanded_nodes": 24,
                "semantic_triples": {"mode": "inject", "max_candidates": 12, "max_relation_context_items": 3},
            },
        }
    )

    assert memory_hierarchy_enabled(policy) is True
    assert memory_tagging_enabled(policy) is True
    assert memory_forgetting_enabled(policy) is True
    assert memory_graph_enabled(policy) is True
    assert policy["hierarchy"]["working_ttl_hours"] == 48
    assert policy["tagging"]["max_tags"] == 8
    assert policy["forgetting"]["access_log_enabled"] is True
    assert policy["forgetting"]["access_curve"] == "log"
    assert policy["forgetting"]["tier_weight_semantic"] == 0.25
    assert policy["forgetting"]["tuning"]["enabled"] is True
    assert policy["forgetting"]["tuning"]["mode"] == "bounded_ema"
    assert policy["forgetting"]["tuning"]["auto_promote_enabled"] is True
    assert policy["forgetting"]["tuning"]["auto_promote_min_shadow_runs"] == 6
    assert policy["forgetting"]["tuning"]["auto_demote_enabled"] is True
    assert policy["forgetting"]["tuning"]["auto_demote_unhealthy_runs"] == 2
    assert policy["graph"]["max_expansion_hops"] == 2
    assert policy["graph"]["max_expanded_nodes"] == 24
    assert memory_graph_semantic_triples_mode(policy) == "inject"
    assert memory_graph_semantic_triples_boost_enabled(policy) is True
    assert memory_graph_semantic_triples_inject_enabled(policy) is True
    assert policy["graph"]["semantic_triples"]["max_candidates"] == 12
    assert policy["graph"]["semantic_triples"]["max_relation_context_items"] == 3


def test_normalize_memory_policy_backfills_semantic_triples_for_old_defaults() -> None:
    old_defaults = default_memory_policy()
    old_defaults["graph"].pop("semantic_triples", None)

    policy = normalize_memory_policy({}, defaults=old_defaults)

    assert policy["graph"]["semantic_triples"]["mode"] == "shadow"
    assert memory_graph_semantic_triples_boost_enabled(policy) is False
