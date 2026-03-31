from __future__ import annotations

from datetime import timedelta

from arena.memory.cleanup import cleanup_candidates
from arena.memory.forgetting import access_boost, decay_multiplier, effective_memory_score
from arena.models import utc_now


def test_effective_memory_score_rewards_recent_access_and_semantic_tier() -> None:
    now = utc_now()
    row_cold = {
        "created_at": now - timedelta(days=120),
        "last_accessed_at": None,
        "access_count": 0,
        "memory_tier": "working",
        "importance_score": 0.8,
    }
    row_warm = {
        "created_at": now - timedelta(days=120),
        "last_accessed_at": now - timedelta(days=3),
        "access_count": 9,
        "memory_tier": "semantic",
        "importance_score": 0.8,
    }

    cold_decay, cold_effective = effective_memory_score(
        row_cold,
        default_decay_factor=0.985,
        min_decay_multiplier=0.15,
        now=now,
    )
    warm_decay, warm_effective = effective_memory_score(
        row_warm,
        default_decay_factor=0.985,
        min_decay_multiplier=0.15,
        now=now,
    )

    assert warm_decay > cold_decay
    assert warm_effective > cold_effective


def test_decay_multiplier_falls_faster_for_working_than_semantic() -> None:
    working = decay_multiplier(
        memory_tier="working",
        age_days=30,
        access_count=0,
        default_decay_factor=0.985,
        min_decay_multiplier=0.15,
    )
    semantic = decay_multiplier(
        memory_tier="semantic",
        age_days=30,
        access_count=0,
        default_decay_factor=0.985,
        min_decay_multiplier=0.15,
    )

    assert working < semantic


def test_access_curve_and_custom_tier_weights_change_decay_shape() -> None:
    sqrt_boost = access_boost(9, access_curve="sqrt")
    log_boost = access_boost(9, access_curve="log")
    capped_boost = access_boost(9, access_curve="capped_linear")

    assert log_boost > sqrt_boost
    assert capped_boost > 1.0

    aggressive = decay_multiplier(
        memory_tier="semantic",
        age_days=45,
        access_count=4,
        default_decay_factor=0.985,
        min_decay_multiplier=0.15,
        semantic_weight=0.9,
        access_curve="log",
    )
    protective = decay_multiplier(
        memory_tier="semantic",
        age_days=45,
        access_count=4,
        default_decay_factor=0.985,
        min_decay_multiplier=0.15,
        semantic_weight=0.2,
        access_curve="sqrt",
    )

    assert protective > aggressive


class _CleanupRepo:
    dataset_fqn = "proj.ds"

    def __init__(self) -> None:
        self.last_sql = ""
        self.last_params = {}

    def fetch_rows(self, sql, params):
        self.last_sql = str(sql)
        self.last_params = dict(params)
        return []


def test_cleanup_candidates_uses_effective_score_and_last_accessed_when_enabled() -> None:
    repo = _CleanupRepo()

    cleanup_candidates(
        repo,
        tenant_id="local",
        trading_mode="paper",
        max_age_days=90,
        min_score=0.3,
        limit=50,
        use_effective_score=True,
        use_last_accessed=True,
        preserve_semantic=True,
    )

    assert "COALESCE(last_accessed_at, created_at) < @cutoff_ts" in repo.last_sql
    assert "COALESCE(effective_score, outcome_score, importance_score, score, 0.0) < @min_score" in repo.last_sql
    assert "LOWER(COALESCE(memory_tier, '')) != 'semantic'" in repo.last_sql
    assert repo.last_params["limit"] == 50
