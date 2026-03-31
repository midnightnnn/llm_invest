from __future__ import annotations

import json
from datetime import timedelta
from types import SimpleNamespace

from arena.memory.policy import MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY, normalize_memory_policy
from arena.memory.tuning import run_memory_forgetting_tuner
from arena.models import utc_now


class _TuningRepo:
    dataset_fqn = "proj.ds"

    def __init__(self, rows: list[dict], configs: dict[str, str] | None = None) -> None:
        self.rows = list(rows)
        self.configs = dict(configs or {})
        self.fetch_calls: list[tuple[str, dict]] = []
        self.set_calls: list[tuple[str, str, str, str | None]] = []

    def fetch_rows(self, sql: str, params: dict) -> list[dict]:
        self.fetch_calls.append((sql, dict(params)))
        return list(self.rows)

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        _ = tenant_id
        return self.configs.get(config_key)

    def set_config(self, tenant_id: str, config_key: str, value: str, updated_by: str | None = None, **kwargs) -> None:
        _ = kwargs
        self.configs[config_key] = value
        self.set_calls.append((tenant_id, config_key, value, updated_by))


def _settings(policy: dict) -> SimpleNamespace:
    return SimpleNamespace(
        trading_mode="paper",
        context_max_memory_events=32,
        memory_compaction_enabled=True,
        memory_compaction_cycle_event_limit=12,
        memory_compaction_recent_lessons_limit=4,
        memory_compaction_max_reflections=3,
        memory_policy=policy,
    )


def _sample_rows() -> list[dict]:
    now = utc_now()
    return [
        {
            "event_id": "semantic-1",
            "created_at": now - timedelta(days=140),
            "memory_tier": "semantic",
            "importance_score": 0.92,
            "score": 0.92,
            "outcome_score": None,
            "access_count": 18,
            "prompt_use_count": 9,
            "distinct_cycle_count": 6,
            "last_accessed_at": now - timedelta(days=1),
            "short_access_count": 5,
            "short_prompt_use_count": 3,
            "short_distinct_cycle_count": 3,
            "short_last_accessed_at": now - timedelta(hours=18),
        },
        {
            "event_id": "semantic-2",
            "created_at": now - timedelta(days=110),
            "memory_tier": "semantic",
            "importance_score": 0.88,
            "score": 0.88,
            "outcome_score": None,
            "access_count": 12,
            "prompt_use_count": 5,
            "distinct_cycle_count": 4,
            "last_accessed_at": now - timedelta(days=2),
            "short_access_count": 4,
            "short_prompt_use_count": 2,
            "short_distinct_cycle_count": 2,
            "short_last_accessed_at": now - timedelta(days=1),
        },
        {
            "event_id": "episodic-1",
            "created_at": now - timedelta(days=35),
            "memory_tier": "episodic",
            "importance_score": 0.65,
            "score": 0.65,
            "outcome_score": 0.72,
            "access_count": 7,
            "prompt_use_count": 2,
            "distinct_cycle_count": 3,
            "last_accessed_at": now - timedelta(days=4),
            "short_access_count": 2,
            "short_prompt_use_count": 1,
            "short_distinct_cycle_count": 1,
            "short_last_accessed_at": now - timedelta(days=3),
        },
        {
            "event_id": "working-1",
            "created_at": now - timedelta(days=20),
            "memory_tier": "working",
            "importance_score": 0.35,
            "score": 0.35,
            "outcome_score": None,
            "access_count": 3,
            "prompt_use_count": 0,
            "distinct_cycle_count": 1,
            "last_accessed_at": now - timedelta(days=12),
            "short_access_count": 0,
            "short_prompt_use_count": 0,
            "short_distinct_cycle_count": 0,
            "short_last_accessed_at": None,
        },
        {
            "event_id": "working-2",
            "created_at": now - timedelta(days=18),
            "memory_tier": "working",
            "importance_score": 0.22,
            "score": 0.22,
            "outcome_score": None,
            "access_count": 1,
            "prompt_use_count": 0,
            "distinct_cycle_count": 1,
            "last_accessed_at": now - timedelta(days=15),
            "short_access_count": 0,
            "short_prompt_use_count": 0,
            "short_distinct_cycle_count": 0,
            "short_last_accessed_at": None,
        },
    ]


def test_forgetting_tuner_shadow_mode_persists_state_only() -> None:
    policy = normalize_memory_policy(
        {
            "forgetting": {
                "enabled": True,
                "access_log_enabled": True,
                "tuning": {
                    "enabled": True,
                    "mode": "shadow",
                    "min_access_events": 10,
                    "min_prompt_uses": 2,
                    "min_unique_memories": 3,
                    "lookback_days": 30,
                    "stability_window_days": 7,
                    "objective_topk": 4,
                },
            }
        }
    )
    repo = _TuningRepo(_sample_rows())

    result = run_memory_forgetting_tuner(
        repo,
        _settings(policy),
        tenant_id="local",
        policy=policy,
        updated_by="tester",
        persist_state=True,
    )

    assert result["reason"] == "shadow only"
    assert result["recommended"] is not None
    assert result["gates"]["sample_ok"] is True
    assert result["gates"]["apply_allowed"] is False
    assert len(result["top_candidates"]) >= 1
    assert [call[1] for call in repo.set_calls] == [MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY]


def test_forgetting_tuner_bounded_ema_applies_numeric_policy_updates_only() -> None:
    policy = normalize_memory_policy(
        {
            "forgetting": {
                "enabled": True,
                "access_log_enabled": True,
                "default_decay_factor": 0.91,
                "min_effective_score": 0.02,
                "access_curve": "log",
                "tier_weight_working": 1.0,
                "tier_weight_episodic": 1.8,
                "tier_weight_semantic": 1.6,
                "tuning": {
                    "enabled": True,
                    "mode": "bounded_ema",
                    "min_access_events": 10,
                    "min_prompt_uses": 2,
                    "min_unique_memories": 3,
                    "lookback_days": 30,
                    "stability_window_days": 7,
                    "ema_alpha": 0.5,
                    "max_decay_factor_delta": 0.02,
                    "max_min_effective_score_delta": 0.05,
                    "max_tier_weight_delta": 0.5,
                    "objective_topk": 4,
                },
            }
        }
    )
    repo = _TuningRepo(_sample_rows())

    result = run_memory_forgetting_tuner(
        repo,
        _settings(policy),
        tenant_id="local",
        policy=policy,
        updated_by="tester",
        persist_state=True,
    )

    config_keys = [call[1] for call in repo.set_calls]
    assert "memory_policy" in config_keys
    assert MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY in config_keys
    assert result["gates"]["apply_allowed"] is True
    assert result["reason"] == "applied bounded ema"

    policy_call = next(call for call in repo.set_calls if call[1] == "memory_policy")
    updated_policy = json.loads(policy_call[2])
    assert updated_policy["forgetting"]["default_decay_factor"] != 0.91
    assert updated_policy["forgetting"]["tier_weight_semantic"] < 1.6
    assert updated_policy["forgetting"]["access_curve"] == "log"
    assert result["applied"]["access_curve_pending"] in {"", "sqrt", "capped_linear"}


def test_forgetting_tuner_auto_promotes_from_shadow_when_history_is_stable() -> None:
    policy = normalize_memory_policy(
        {
            "forgetting": {
                "enabled": True,
                "access_log_enabled": True,
                "tuning": {
                    "enabled": True,
                    "mode": "shadow",
                    "min_access_events": 10,
                    "min_prompt_uses": 2,
                    "min_unique_memories": 3,
                    "lookback_days": 30,
                    "stability_window_days": 7,
                    "objective_topk": 4,
                    "auto_promote_enabled": True,
                    "auto_promote_min_shadow_days": 3,
                    "auto_promote_min_shadow_runs": 3,
                    "auto_promote_required_stable_runs": 3,
                    "auto_promote_required_improving_runs": 2,
                    "auto_promote_max_recommendation_drift": 1.0,
                },
            }
        }
    )
    previous_state = {
        "mode": "shadow",
        "effective_mode": "shadow",
        "recommended": {
            "default_decay_factor": 0.99,
            "min_effective_score": 0.15,
            "access_curve": "sqrt",
            "tier_weight_working": 2.0,
            "tier_weight_episodic": 1.0,
            "tier_weight_semantic": 0.35,
        },
        "history": {
            "total_runs": 4,
            "consecutive_sample_ok": 4,
            "consecutive_stability_ok": 4,
            "consecutive_improvement_ok": 3,
            "consecutive_apply_eligible": 3,
            "consecutive_unhealthy_runs": 0,
            "shadow_runs_since_transition": 4,
            "shadow_started_at": (utc_now() - timedelta(days=5)).isoformat(),
            "shadow_days_since_transition": 5,
            "bounded_ema_runs_since_transition": 0,
            "bounded_ema_started_at": "",
            "last_transition_at": "",
            "last_transition_action": "",
            "auto_promotions": 0,
            "auto_demotions": 0,
        },
    }
    repo = _TuningRepo(_sample_rows(), {MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY: json.dumps(previous_state)})

    result = run_memory_forgetting_tuner(
        repo,
        _settings(policy),
        tenant_id="local",
        policy=policy,
        updated_by="tester",
        persist_state=True,
    )

    assert result["effective_mode"] == "bounded_ema"
    assert result["transition"]["action"] == "auto_promote"
    assert result["gates"]["apply_allowed"] is True
    policy_call = next(call for call in repo.set_calls if call[1] == "memory_policy")
    updated_policy = json.loads(policy_call[2])
    assert updated_policy["forgetting"]["tuning"]["mode"] == "bounded_ema"


def test_forgetting_tuner_auto_demotes_when_bounded_ema_health_keeps_failing() -> None:
    policy = normalize_memory_policy(
        {
            "forgetting": {
                "enabled": True,
                "access_log_enabled": True,
                "tuning": {
                    "enabled": True,
                    "mode": "bounded_ema",
                    "min_access_events": 50,
                    "min_prompt_uses": 10,
                    "min_unique_memories": 10,
                    "lookback_days": 30,
                    "stability_window_days": 7,
                    "objective_topk": 4,
                    "auto_demote_enabled": True,
                    "auto_demote_unhealthy_runs": 3,
                },
            }
        }
    )
    previous_state = {
        "mode": "bounded_ema",
        "effective_mode": "bounded_ema",
        "history": {
            "total_runs": 7,
            "consecutive_sample_ok": 0,
            "consecutive_stability_ok": 0,
            "consecutive_improvement_ok": 0,
            "consecutive_apply_eligible": 0,
            "consecutive_unhealthy_runs": 2,
            "shadow_runs_since_transition": 0,
            "shadow_started_at": "",
            "shadow_days_since_transition": 0,
            "bounded_ema_runs_since_transition": 5,
            "bounded_ema_started_at": (utc_now() - timedelta(days=4)).isoformat(),
            "last_transition_at": "",
            "last_transition_action": "",
            "auto_promotions": 1,
            "auto_demotions": 0,
        },
    }
    repo = _TuningRepo(_sample_rows(), {MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY: json.dumps(previous_state)})

    result = run_memory_forgetting_tuner(
        repo,
        _settings(policy),
        tenant_id="local",
        policy=policy,
        updated_by="tester",
        persist_state=True,
    )

    assert result["effective_mode"] == "shadow"
    assert result["transition"]["action"] == "auto_demote"
    assert result["gates"]["apply_allowed"] is False
    policy_call = next(call for call in repo.set_calls if call[1] == "memory_policy")
    updated_policy = json.loads(policy_call[2])
    assert updated_policy["forgetting"]["tuning"]["mode"] == "shadow"
