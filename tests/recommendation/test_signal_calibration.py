from __future__ import annotations

from arena.recommendation.signal_calibration import (
    PromotionMetrics,
    TransformCandidate,
    apply_signal_formula_specs,
    apply_signal_transform_specs,
    calibrate_signal_formula_specs,
    calibrate_signal_transforms,
    evaluate_promotion_gate,
    select_transform_candidate,
)
from arena.recommendation.signals import SignalDef


def test_promotion_gate_approves_candidate_that_beats_baseline_and_active() -> None:
    decision = evaluate_promotion_gate(
        candidate=PromotionMetrics(
            mean_rank_ic=0.024,
            rank_ic_std=0.006,
            top_bucket_excess_return_20d=0.018,
            hit_rate=0.57,
            coverage=0.93,
            folds_total=6,
            folds_won=5,
            ranking_churn=0.18,
        ),
        baseline=PromotionMetrics(
            mean_rank_ic=0.013,
            rank_ic_std=0.007,
            top_bucket_excess_return_20d=0.009,
            hit_rate=0.53,
            coverage=0.95,
            folds_total=6,
            folds_won=0,
            ranking_churn=0.0,
        ),
        active=PromotionMetrics(
            mean_rank_ic=0.018,
            rank_ic_std=0.006,
            top_bucket_excess_return_20d=0.014,
            hit_rate=0.55,
            coverage=0.94,
            folds_total=6,
            folds_won=3,
            ranking_churn=0.12,
        ),
    )

    assert decision.promote is True
    assert decision.score_source == "joint_policy_v2"
    assert decision.reasons == ["candidate_beats_baseline", "candidate_beats_active"]


def test_promotion_gate_rejects_candidate_with_coverage_collapse() -> None:
    decision = evaluate_promotion_gate(
        candidate=PromotionMetrics(
            mean_rank_ic=0.050,
            rank_ic_std=0.004,
            top_bucket_excess_return_20d=0.030,
            hit_rate=0.62,
            coverage=0.35,
            folds_total=6,
            folds_won=6,
            ranking_churn=0.10,
        ),
        baseline=PromotionMetrics(
            mean_rank_ic=0.012,
            rank_ic_std=0.006,
            top_bucket_excess_return_20d=0.008,
            hit_rate=0.52,
            coverage=0.94,
            folds_total=6,
            folds_won=0,
            ranking_churn=0.0,
        ),
    )

    assert decision.promote is False
    assert "coverage_below_floor" in decision.reasons


def test_select_transform_candidate_uses_oos_score_then_simplicity() -> None:
    selected = select_transform_candidate(
        [
            TransformCandidate(
                signal_name="pullback",
                transform_name="complex_quantile_band",
                params={"ret20_quantile": 0.60, "ret5_low_quantile": 0.10, "ret5_high_quantile": 0.40},
                metrics={"mean_rank_ic": 0.021, "rank_ic_std": 0.004, "coverage": 0.14, "hit_rate": 0.56},
                complexity=4,
            ),
            TransformCandidate(
                signal_name="pullback",
                transform_name="simple_quantile_band",
                params={"ret20_quantile": 0.60, "ret5_low_quantile": 0.20},
                metrics={"mean_rank_ic": 0.021, "rank_ic_std": 0.004, "coverage": 0.14, "hit_rate": 0.56},
                complexity=2,
            ),
        ]
    )

    assert selected.transform_name == "simple_quantile_band"
    assert selected.params == {"ret20_quantile": 0.60, "ret5_low_quantile": 0.20}


def test_calibrate_signal_transforms_can_flip_bad_historical_direction() -> None:
    signal = SignalDef(
        name="toy",
        column="signal_toy",
        direction="higher_better",
        group="test",
        description="toy",
    )
    rows = [
        {"as_of_date": "2026-01-01", "ticker": "A", "signal_toy": 3.0, "fwd_excess_return_20d": -0.03},
        {"as_of_date": "2026-01-01", "ticker": "B", "signal_toy": 2.0, "fwd_excess_return_20d": 0.01},
        {"as_of_date": "2026-01-01", "ticker": "C", "signal_toy": 1.0, "fwd_excess_return_20d": 0.04},
        {"as_of_date": "2026-01-02", "ticker": "A", "signal_toy": 4.0, "fwd_excess_return_20d": -0.02},
        {"as_of_date": "2026-01-02", "ticker": "B", "signal_toy": 2.0, "fwd_excess_return_20d": 0.00},
        {"as_of_date": "2026-01-02", "ticker": "C", "signal_toy": 1.0, "fwd_excess_return_20d": 0.03},
    ]

    specs = calibrate_signal_transforms(rows, signals=(signal,))
    transformed = apply_signal_transform_specs(
        [{"as_of_date": "2026-01-03", "ticker": "Z", "signal_toy": 2.5}],
        specs,
        signals=(signal,),
    )

    assert specs[0].transform_name == "negated"
    assert transformed[0]["signal_toy"] == -2.5


def test_formula_calibration_selects_raw_momentum_over_bad_existing_signal() -> None:
    signal = SignalDef(
        name="momentum_20d",
        column="signal_momentum_20d",
        direction="higher_better",
        group="price",
        description="toy momentum",
    )
    rows = [
        {
            "as_of_date": "2026-01-01",
            "ticker": "A",
            "signal_momentum_20d": 0.0,
            "ret_20d": 0.30,
            "volatility_20d": 0.20,
            "fwd_excess_return_20d": 0.05,
        },
        {
            "as_of_date": "2026-01-01",
            "ticker": "B",
            "signal_momentum_20d": 1.0,
            "ret_20d": 0.20,
            "volatility_20d": 0.20,
            "fwd_excess_return_20d": 0.02,
        },
        {
            "as_of_date": "2026-01-01",
            "ticker": "C",
            "signal_momentum_20d": -1.0,
            "ret_20d": 0.10,
            "volatility_20d": 0.20,
            "fwd_excess_return_20d": -0.01,
        },
        {
            "as_of_date": "2026-01-02",
            "ticker": "A",
            "signal_momentum_20d": 0.5,
            "ret_20d": 0.40,
            "volatility_20d": 0.30,
            "fwd_excess_return_20d": 0.04,
        },
        {
            "as_of_date": "2026-01-02",
            "ticker": "B",
            "signal_momentum_20d": -1.0,
            "ret_20d": 0.15,
            "volatility_20d": 0.30,
            "fwd_excess_return_20d": 0.00,
        },
        {
            "as_of_date": "2026-01-02",
            "ticker": "C",
            "signal_momentum_20d": 1.0,
            "ret_20d": 0.05,
            "volatility_20d": 0.30,
            "fwd_excess_return_20d": -0.02,
        },
    ]

    specs = calibrate_signal_formula_specs(rows, signals=(signal,))
    transformed = apply_signal_formula_specs(
        [
            {"as_of_date": "2026-01-03", "ticker": "HI", "signal_momentum_20d": -9.0, "ret_20d": 0.25},
            {"as_of_date": "2026-01-03", "ticker": "LO", "signal_momentum_20d": -1.0, "ret_20d": 0.02},
        ],
        specs,
        signals=(signal,),
    )

    assert specs[0].transform_name == "formula_ret20"
    assert specs[0].params["input_columns"] == ["ret_20d"]
    assert transformed[0]["signal_momentum_20d"] > transformed[1]["signal_momentum_20d"]
