from __future__ import annotations

from arena.data.schema import parse_ddl_columns, render_table_ddls
from arena.memory.policy import (
    default_memory_policy,
    memory_forgetting_enabled,
    memory_graph_enabled,
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
    assert ("memory_tier", "STRING") in cols["agent_memory_events"]
    assert ("context_tags_json", "JSON") in cols["agent_memory_events"]
    assert ("access_count", "INT64") in cols["agent_memory_events"]
    assert ("effective_score", "FLOAT64") in cols["agent_memory_events"]
    assert ("causal_chain_id", "STRING") in cols["agent_memory_events"]


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
            "graph": {"enabled": True, "max_expansion_hops": 2, "max_expanded_nodes": 24},
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
