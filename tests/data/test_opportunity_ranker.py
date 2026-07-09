from __future__ import annotations

import math
import random
from datetime import date, timedelta

import numpy as np

from arena.config import load_settings
from arena.data.bigquery.market_store import MarketStore
from arena.data.schema import parse_ddl_columns, render_table_ddls
from arena.recommendation import (
    ALL_SIGNALS,
    REGIME_FEATURES,
    SIGNAL_NAMES,
    build_and_store_opportunity_ranker,
    build_and_store_signal_ic_ranker,
)
from arena.recommendation.joint_policy_ranker import (
    JointPolicyParams,
    expand_characteristic_returns_with_macro_interactions,
    fit_turnover_regularized_policy,
)
from arena.recommendation.macro_interactions import macro_factor_frame_for_rows


def test_schema_includes_new_signal_and_fundamentals_tables() -> None:
    ddls = "\n".join(render_table_ddls("proj", "ds"))
    cols = parse_ddl_columns()

    # Signal / regime tables
    assert "proj.ds.signal_daily_values" in ddls
    assert "proj.ds.signal_daily_values_v2" in ddls
    assert "proj.ds.signal_daily_ic" in ddls
    assert "proj.ds.signal_calibration_runs" in ddls
    assert "proj.ds.signal_transform_specs" in ddls
    assert "proj.ds.regime_daily_features" in ddls
    # PIT fundamentals tables
    assert "proj.ds.fundamentals_history_raw" in ddls
    assert "proj.ds.fundamentals_derived_daily" in ddls
    assert "proj.ds.fundamentals_ingest_runs" in ddls

    # Critical columns
    assert ("label_ready", "BOOL") in cols["signal_daily_values"]
    assert ("calibration_run_id", "STRING") in cols["signal_daily_values_v2"]
    assert ("signal_quality_json", "JSON") in cols["signal_daily_values_v2"]
    assert ("promoted", "BOOL") in cols["signal_calibration_runs"]
    assert ("active", "BOOL") in cols["signal_transform_specs"]
    assert ("ic_20d", "FLOAT64") in cols["signal_daily_ic"]
    assert ("regime_trend", "FLOAT64") in cols["regime_daily_features"]
    assert ("announcement_date", "DATE") in cols["fundamentals_history_raw"]
    assert ("announcement_date_source", "STRING") in cols["fundamentals_history_raw"]
    assert ("ep", "FLOAT64") in cols["fundamentals_derived_daily"]
    # Legacy tables still present during transition
    assert "proj.ds.opportunity_ranker_scores_latest" in ddls
    assert "proj.ds.opportunity_ranker_runs" in ddls


def test_all_signals_module_exports_are_consistent() -> None:
    assert len(ALL_SIGNALS) == len(SIGNAL_NAMES)
    assert len(ALL_SIGNALS) == len({s.name for s in ALL_SIGNALS})
    assert {s.column for s in ALL_SIGNALS} == set(
        f"signal_{name}" for name in SIGNAL_NAMES
    )
    # All groups reference known families
    allowed_groups = {
        "price",
        "technical",
        "sentiment",
        "forecast",
        "fundamental_value",
        "fundamental_quality",
        "fundamental_growth",
        "fundamental_safety",
    }
    assert {s.group for s in ALL_SIGNALS}.issubset(allowed_groups)
    assert "regime_vol_level" in REGIME_FEATURES
    assert "regime_trend" in REGIME_FEATURES


class _FakeICRepo:
    """In-memory repo that mimics the ranker's BigQuery surface area."""

    def __init__(
        self,
        *,
        ic_rows: list[dict],
        regime_rows: list[dict],
        scoring_rows: list[dict],
    ) -> None:
        self._ic_rows = list(ic_rows)
        self._regime_rows = list(regime_rows)
        self._scoring_rows = list(scoring_rows)
        self.score_rows: list[dict] = []
        self.run_rows: list[dict] = []
        self.refreshes: dict[str, int] = {}
        self.refresh_value_calls: list[dict[str, object]] = []

    def refresh_signal_daily_values(self, **kwargs: object) -> int:
        self.refreshes["values"] = self.refreshes.get("values", 0) + 1
        self.refresh_value_calls.append(dict(kwargs))
        return 0

    def refresh_signal_daily_ic(self, **_: object) -> int:
        self.refreshes["ic"] = self.refreshes.get("ic", 0) + 1
        return 0

    def refresh_regime_daily_features(self, **_: object) -> int:
        self.refreshes["regime"] = self.refreshes.get("regime", 0) + 1
        return 0

    def load_signal_daily_ic(self, **_: object) -> list[dict]:
        return [dict(row) for row in self._ic_rows]

    def load_regime_daily_features(self, **_: object) -> list[dict]:
        return [dict(row) for row in self._regime_rows]

    def load_signal_scoring_rows(self, **_: object) -> list[dict]:
        return [dict(row) for row in self._scoring_rows]

    def insert_opportunity_ranker_scores_latest(self, rows: list[dict]) -> int:
        self.score_rows.extend(dict(row) for row in rows)
        return len(rows)

    def append_opportunity_ranker_run(self, row: dict) -> int:
        self.run_rows.append(dict(row))
        return 1


class _FakePolicyRepo(_FakeICRepo):
    def __init__(
        self,
        *,
        policy_rows: list[dict],
        regime_rows: list[dict],
        scoring_rows: list[dict],
        forecast_rows: list[dict] | None = None,
        macro_rows: list[dict] | None = None,
    ) -> None:
        super().__init__(ic_rows=[], regime_rows=regime_rows, scoring_rows=scoring_rows)
        self._policy_rows = list(policy_rows)
        self._forecast_rows = list(forecast_rows or [])
        self._macro_rows = list(macro_rows or [])

    def load_signal_policy_training_rows(self, **_: object) -> list[dict]:
        return [dict(row) for row in self._policy_rows]

    def get_predicted_returns(self, *, tickers=None, **_: object) -> list[dict]:
        tokens = {str(t or "").strip().upper() for t in (tickers or []) if str(t or "").strip()}
        return [
            dict(row)
            for row in self._forecast_rows
            if not tokens or str(row.get("ticker") or "").strip().upper() in tokens
        ]

    def macro_indicator_observation_history(
        self,
        *,
        indicator_keys=None,
        start_date=None,
        end_date=None,
        **_: object,
    ) -> list[dict]:
        keys = {str(key or "").strip() for key in (indicator_keys or []) if str(key or "").strip()}
        out: list[dict] = []
        for row in self._macro_rows:
            key = str(row.get("indicator_key") or "").strip()
            obs = row.get("observation_date")
            obs_date = obs if isinstance(obs, date) else date.fromisoformat(str(obs)[:10])
            if keys and key not in keys:
                continue
            if start_date is not None and obs_date < start_date:
                continue
            if end_date is not None and obs_date > end_date:
                continue
            out.append(dict(row))
        return out


class _FakeCalibratingPolicyRepo(_FakePolicyRepo):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calibration_rows: list[dict] = []
        self.transform_spec_rows: list[dict] = []

    def insert_signal_calibration_run(self, row: dict) -> int:
        self.calibration_rows.append(dict(row))
        return 1

    def insert_signal_transform_specs(self, rows: list[dict]) -> int:
        self.transform_spec_rows.extend(dict(row) for row in rows)
        return len(rows)


def _synthetic_ic_rows(days: int = 120) -> tuple[list[dict], list[dict], list[dict]]:
    """Builds deterministic IC history + regime features where higher vol_level
    depresses momentum IC and lifts lowvol IC. Scoring rows are four tickers
    where AAA has strong momentum signal, BBB has strong forecast signal, etc.
    """
    rng = random.Random(1337)
    start = date(2025, 1, 1)
    ic_rows: list[dict] = []
    regime_rows: list[dict] = []
    for d in range(days):
        as_of = start + timedelta(days=d)
        # Regime feature: low/high vol regime alternates in a slow cycle
        phase = math.sin(d / 20.0)
        vol_level = 0.02 + 0.02 * phase
        trend = 0.005 * math.cos(d / 15.0)
        regime_rows.append(
            {
                "as_of_date": as_of.isoformat(),
                "market": "us",
                "regime_vol_level": vol_level,
                "regime_vol_dispersion": 0.01,
                "regime_trend": trend,
                "regime_short_reversal": 0.0,
                "regime_dispersion": 0.02,
                "regime_sentiment": 0.0,
                "sample_size": 100,
            }
        )
        # IC: momentum ↑ in low vol, lowvol ↑ in high vol, noise elsewhere
        momentum_ic = 0.08 - 2.0 * phase + rng.gauss(0, 0.02)
        lowvol_ic = 0.04 + 1.5 * phase + rng.gauss(0, 0.02)
        for signal_name in SIGNAL_NAMES:
            if signal_name == "momentum_20d":
                ic = momentum_ic
            elif signal_name == "lowvol":
                ic = lowvol_ic
            elif signal_name == "forecast_er":
                ic = 0.05 + rng.gauss(0, 0.01)
            else:
                ic = rng.gauss(0, 0.02)
            ic_rows.append(
                {
                    "as_of_date": as_of.isoformat(),
                    "signal_name": signal_name,
                    "horizon_days": 20,
                    "ic_20d": ic,
                    "rank_ic_20d": ic * 0.9,
                    "sample_size": 50,
                    "market": "us",
                }
            )

    scoring_date = start + timedelta(days=days)
    scoring_rows = [
        {
            "as_of_date": scoring_date.isoformat(),
            "ticker": "AAA",
            "market": "us",
            "bucket": "momentum",
            "profile": "aggressive",
            "signal_momentum_20d": 1.5,
            "signal_pullback": 0.0,
            "signal_meanrev_5d": 0.0,
            "signal_lowvol": 0.2,
            "signal_sentiment": 0.3,
            "signal_forecast_er": 0.02,
            "signal_forecast_prob": 0.1,
            "signal_rsi_reversal": 0.0,
            "signal_ma_crossover": 1.0,
            "signal_bollinger_position": 0.4,
        },
        {
            "as_of_date": scoring_date.isoformat(),
            "ticker": "BBB",
            "market": "us",
            "bucket": "momentum",
            "profile": "balanced",
            "signal_momentum_20d": 0.5,
            "signal_lowvol": 1.2,
            "signal_sentiment": 0.0,
            "signal_forecast_er": 0.05,
            "signal_forecast_prob": 0.2,
            "signal_rsi_reversal": 0.0,
            "signal_ma_crossover": 1.0,
            "signal_bollinger_position": 0.0,
        },
        {
            "as_of_date": scoring_date.isoformat(),
            "ticker": "CCC",
            "market": "us",
            "bucket": "defensive",
            "profile": "defensive",
            "signal_momentum_20d": -1.2,
            "signal_lowvol": -0.8,
            "signal_sentiment": -0.5,
            "signal_forecast_er": -0.01,
            "signal_forecast_prob": -0.1,
            "signal_rsi_reversal": -1.0,
            "signal_ma_crossover": -1.0,
            "signal_bollinger_position": -0.5,
        },
        {
            "as_of_date": scoring_date.isoformat(),
            "ticker": "SQQQ",
            "market": "us",
            "bucket": "defensive",
            "profile": "balanced",
            "signal_momentum_20d": 0.2,
            "signal_lowvol": 0.1,
            "signal_forecast_er": -0.02,
            "signal_forecast_prob": -0.2,
            "signal_rsi_reversal": 1.0,
            "signal_ma_crossover": 0.0,
            "signal_bollinger_position": 0.0,
        },
    ]
    return ic_rows, regime_rows, scoring_rows


def _synthetic_policy_rows(days: int = 90) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(20260513)
    start = date(2025, 1, 1)
    tickers = [f"T{i:02d}" for i in range(20)]
    policy_rows: list[dict] = []
    regime_rows: list[dict] = []
    for d in range(days):
        as_of = start + timedelta(days=d)
        phase = math.sin(d / 18.0)
        regime_rows.append(
            {
                "as_of_date": as_of.isoformat(),
                "market": "us",
                "regime_vol_level": 0.02 + 0.01 * phase,
                "regime_vol_dispersion": 0.01,
                "regime_trend": 0.002 * math.cos(d / 12.0),
                "regime_short_reversal": 0.0,
                "regime_dispersion": 0.02,
                "regime_sentiment": 0.0,
                "sample_size": len(tickers),
            }
        )
        for idx, ticker in enumerate(tickers):
            momentum = (idx - 9.5) / 5.0
            lowvol = -momentum * 0.35 + rng.gauss(0.0, 0.15)
            y = 0.055 * momentum - 0.020 * lowvol + rng.gauss(0.0, 0.004)
            row = {
                "as_of_date": as_of.isoformat(),
                "ticker": ticker,
                "market": "us",
                "fwd_excess_return_20d": y,
                "label_ready": True,
                "signal_momentum_20d": momentum,
                "signal_lowvol": lowvol,
                "signal_pullback": 0.0,
                "signal_meanrev_5d": 0.0,
                "signal_sentiment": 0.0,
                "signal_forecast_er": 0.0,
                "signal_forecast_prob": 0.0,
                "signal_rsi_reversal": 0.0,
                "signal_ma_crossover": 0.0,
                "signal_bollinger_position": 0.0,
                "signal_ep": 0.0,
                "signal_bp": 0.0,
                "signal_sp": 0.0,
                "signal_roe": 0.0,
                "signal_revenue_growth": 0.0,
                "signal_eps_growth": 0.0,
                "signal_low_debt": 0.0,
            }
            policy_rows.append(row)
    scoring_date = start + timedelta(days=days)
    scoring_rows = [
        {
            "as_of_date": scoring_date.isoformat(),
            "ticker": "AAA",
            "market": "us",
            "bucket": "momentum",
            "profile": "aggressive",
            "signal_momentum_20d": 1.8,
            "signal_lowvol": -0.2,
            "signal_pullback": 0.0,
            "signal_meanrev_5d": 0.0,
            "signal_forecast_er": 0.03,
            "signal_forecast_prob": 0.1,
        },
        {
            "as_of_date": scoring_date.isoformat(),
            "ticker": "CCC",
            "market": "us",
            "bucket": "defensive",
            "profile": "defensive",
            "signal_momentum_20d": -1.4,
            "signal_lowvol": 0.1,
            "signal_pullback": 0.0,
            "signal_meanrev_5d": 0.0,
            "signal_forecast_er": -0.02,
            "signal_forecast_prob": -0.1,
        },
    ]
    return policy_rows, regime_rows, scoring_rows


def test_turnover_penalty_reduces_joint_policy_coefficient_change() -> None:
    char_returns = np.array([[0.04, 0.0]] * 20 + [[-0.04, 0.0]], dtype=float)

    loose = fit_turnover_regularized_policy(
        char_returns,
        signal_names=("momentum_20d", "lowvol"),
        params=JointPolicyParams(
            lambda_l2=0.05,
            lambda_turnover=0.0,
            gamma=0.0,
            trailing_window=1,
            min_training_dates=1,
        ),
    )
    sticky = fit_turnover_regularized_policy(
        char_returns,
        signal_names=("momentum_20d", "lowvol"),
        params=JointPolicyParams(
            lambda_l2=0.05,
            lambda_turnover=2.0,
            gamma=0.0,
            trailing_window=1,
            min_training_dates=1,
        ),
    )

    loose_delta = abs(loose.coefficients["momentum_20d"] - loose.previous_coefficients["momentum_20d"])
    sticky_delta = abs(sticky.coefficients["momentum_20d"] - sticky.previous_coefficients["momentum_20d"])
    assert sticky_delta < loose_delta


def test_l1_penalty_sparsifies_weak_joint_policy_coefficients() -> None:
    char_returns = np.array([[0.05, 0.0002]] * 10, dtype=float)

    no_l1 = fit_turnover_regularized_policy(
        char_returns,
        signal_names=("momentum_20d", "lowvol"),
        params=JointPolicyParams(
            lambda_l1=0.0,
            lambda_l2=0.05,
            lambda_turnover=0.0,
            gamma=0.0,
            trailing_window=10,
            min_training_dates=1,
            max_abs_weight=10.0,
        ),
    )
    with_l1 = fit_turnover_regularized_policy(
        char_returns,
        signal_names=("momentum_20d", "lowvol"),
        params=JointPolicyParams(
            lambda_l1=0.001,
            lambda_l2=0.05,
            lambda_turnover=0.0,
            gamma=0.0,
            trailing_window=10,
            min_training_dates=1,
            max_abs_weight=10.0,
        ),
    )

    assert abs(no_l1.coefficients["lowvol"]) > 0.0
    assert abs(with_l1.coefficients["lowvol"]) <= 1e-12
    assert abs(with_l1.lambda_l1 - 0.001) <= 1e-12


def test_build_and_store_opportunity_ranker_writes_joint_policy_scores() -> None:
    policy_rows, regime_rows, scoring_rows = _synthetic_policy_rows(days=90)
    repo = _FakePolicyRepo(policy_rows=policy_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_opportunity_ranker(
        repo,
        settings,
        lookback_days=120,
        horizon_days=20,
        min_ic_dates=30,
        max_scoring_rows=10,
    )

    assert result.status == "ok"
    assert result.ranker_version.startswith("opportunity_ranker_joint_policy_")
    assert result.scores_written == len(scoring_rows)
    assert repo.refreshes == {"values": 1, "regime": 1}
    top = repo.score_rows[0]
    assert top["score_source"] == "joint_policy_v1"
    assert top["ticker"] == "AAA"
    explanation = top["explanation_json"]
    assert explanation["model_family"] == "regularized_joint_policy_macro_interactions"
    assert explanation["policy_variant"] == "macro_interaction_joint_v1"
    assert "policy_coefficients" in explanation
    assert explanation["optimizer"]["lambda_l1"] > 0.0
    assert "top_contributions" in explanation
    assert repo.run_rows[-1]["score_source"] == "joint_policy_v1"
    assert "policy_coefficients" in repo.run_rows[-1]["detail_json"]


def test_build_and_store_opportunity_ranker_records_signal_v2_calibration_artifacts() -> None:
    policy_rows, regime_rows, scoring_rows = _synthetic_policy_rows(days=90)
    repo = _FakeCalibratingPolicyRepo(policy_rows=policy_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_opportunity_ranker(
        repo,
        settings,
        lookback_days=120,
        horizon_days=20,
        min_ic_dates=30,
        max_scoring_rows=10,
    )

    assert result.status == "ok"
    assert repo.calibration_rows
    assert repo.calibration_rows[-1]["baseline_score_source"] == "joint_policy_v1"
    assert repo.calibration_rows[-1]["score_source"] in {"joint_policy_v1", "joint_policy_v2"}
    assert repo.calibration_rows[-1]["validation_metrics_json"]["baseline"]["mean_rank_ic"] is not None
    assert repo.calibration_rows[-1]["detail_json"]["policy_variant"] == "signal_formula_calibrated_v2_1"
    assert repo.calibration_rows[-1]["detail_json"]["calibration"] == "walk_forward_formula_policy_validation"
    assert repo.transform_spec_rows
    assert {row["signal_name"] for row in repo.transform_spec_rows}.issubset(set(SIGNAL_NAMES))


def test_macro_factor_frame_uses_conservative_macro_lag() -> None:
    rows = [
        {
            "as_of_date": "2025-03-02",
            "ticker": "AAA",
            "market": "kospi",
            "signal_momentum_20d": 1.0,
            "signal_forecast_er": 0.02,
            "signal_lowvol": 0.5,
        },
        {
            "as_of_date": "2025-03-10",
            "ticker": "BBB",
            "market": "kospi",
            "signal_momentum_20d": 1.0,
            "signal_forecast_er": 0.02,
            "signal_lowvol": 0.5,
        },
    ]
    macro_rows = [
        {"indicator_key": "kr_m2", "observation_date": "2024-01-01", "value": 100.0, "frequency": "monthly"},
        {"indicator_key": "kr_m2", "observation_date": "2024-02-01", "value": 100.0, "frequency": "monthly"},
        {"indicator_key": "kr_m2", "observation_date": "2025-01-01", "value": 110.0, "frequency": "monthly"},
        {"indicator_key": "kr_m2", "observation_date": "2025-02-01", "value": 140.0, "frequency": "monthly"},
        {"indicator_key": "kr_cpi", "observation_date": "2024-01-01", "value": 100.0, "frequency": "monthly"},
        {"indicator_key": "kr_cpi", "observation_date": "2024-02-01", "value": 100.0, "frequency": "monthly"},
        {"indicator_key": "kr_cpi", "observation_date": "2025-01-01", "value": 103.0, "frequency": "monthly"},
        {"indicator_key": "kr_cpi", "observation_date": "2025-02-01", "value": 104.0, "frequency": "monthly"},
    ]
    repo = _FakePolicyRepo(policy_rows=[], regime_rows=[], scoring_rows=[], macro_rows=macro_rows)

    frame = macro_factor_frame_for_rows(repo, rows, lookback_days=500, market="kospi")

    early = frame.factors_for(date(2025, 3, 2), "kospi")["liquidity"]
    late = frame.factors_for(date(2025, 3, 10), "kospi")["liquidity"]
    diagnostics = frame.diagnostics
    assert diagnostics["enabled"] is True
    assert diagnostics["macro_rows_loaded"] == len(macro_rows)
    assert early is not None and late is not None
    assert early < late


def test_macro_interaction_weights_are_learned_by_joint_optimizer() -> None:
    expanded, expanded_names, interaction_specs = expand_characteristic_returns_with_macro_interactions(
        characteristic_returns=np.array([[-0.01], [0.02], [0.05], [0.08]], dtype=float),
        training_dates=[
            date(2025, 1, 1),
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 4),
        ],
        signal_names=("momentum_20d",),
        factor_by_date={
            date(2025, 1, 1): {"liquidity": -1.0},
            date(2025, 1, 2): {"liquidity": 0.0},
            date(2025, 1, 3): {"liquidity": 1.0},
            date(2025, 1, 4): {"liquidity": 2.0},
        },
        conditioning_pairs=(("momentum_20d", "liquidity"),),
    )
    fit = fit_turnover_regularized_policy(
        expanded,
        signal_names=expanded_names,
        params=JointPolicyParams(
            lambda_l1=0.0,
            lambda_l2=0.05,
            lambda_turnover=0.0,
            gamma=0.0,
            trailing_window=4,
            min_training_dates=1,
            max_abs_weight=0.50,
        ),
    )

    assert expanded.shape == (4, 2)
    assert expanded_names == ("momentum_20d", "macro_liquidity_x_momentum_20d")
    assert interaction_specs[0]["name"] == "macro_liquidity_x_momentum_20d"
    assert fit.coefficients["macro_liquidity_x_momentum_20d"] > 0.0


def test_build_and_store_opportunity_ranker_includes_macro_conditioned_coefficients() -> None:
    policy_rows, regime_rows, scoring_rows = _synthetic_policy_rows(days=90)
    start = date(2024, 1, 1)
    macro_rows: list[dict] = []
    for offset in range(0, 460):
        as_of = start + timedelta(days=offset)
        macro_rows.extend(
            [
                {
                    "indicator_key": "fed_funds_rate",
                    "observation_date": as_of.isoformat(),
                    "value": 3.0 + offset / 1000.0,
                    "frequency": "daily",
                },
                {
                    "indicator_key": "treasury_10y",
                    "observation_date": as_of.isoformat(),
                    "value": 4.0 + offset / 1200.0,
                    "frequency": "daily",
                },
                {
                    "indicator_key": "treasury_3m",
                    "observation_date": as_of.isoformat(),
                    "value": 3.5 + offset / 1300.0,
                    "frequency": "daily",
                },
                {
                    "indicator_key": "vix",
                    "observation_date": as_of.isoformat(),
                    "value": 15.0 + math.sin(offset / 30.0),
                    "frequency": "daily",
                },
            ]
        )
    monthly_points = [
        ("2024-01-01", 100.0, 100.0),
        ("2024-02-01", 100.0, 100.0),
        ("2025-01-01", 112.0, 103.0),
        ("2025-02-01", 118.0, 104.0),
    ]
    for obs_date, m2, cpi in monthly_points:
        macro_rows.append({"indicator_key": "m2_money_supply", "observation_date": obs_date, "value": m2, "frequency": "monthly"})
        macro_rows.append({"indicator_key": "cpi_index", "observation_date": obs_date, "value": cpi, "frequency": "monthly"})
    repo = _FakePolicyRepo(
        policy_rows=policy_rows,
        regime_rows=regime_rows,
        scoring_rows=scoring_rows,
        macro_rows=macro_rows,
    )
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_opportunity_ranker(
        repo,
        settings,
        lookback_days=500,
        horizon_days=20,
        min_ic_dates=30,
        max_scoring_rows=10,
    )

    top = repo.score_rows[0]
    detail = repo.run_rows[-1]["detail_json"]
    assert result.status == "ok"
    assert "macro_liquidity_x_momentum" not in top["feature_json"]
    assert "macro_conditioning" in top["explanation_json"]
    assert "base_policy_coefficients" in top["explanation_json"]
    assert "effective_policy_coefficients" in top["explanation_json"]["macro_conditioning"]
    assert detail["macro_conditioning"]["training"]["macro_rows_loaded"] > 0
    assert detail["macro_conditioning"]["joint_policy_coefficients"]
    assert detail["macro_conditioning"]["macro_interaction_coefficients"]
    assert detail["macro_conditioning"]["effective_policy_coefficients"]


def test_build_and_store_opportunity_ranker_filters_forecast_missing_scoring_rows() -> None:
    policy_rows, regime_rows, scoring_rows = _synthetic_policy_rows(days=90)
    missing_forecast = dict(scoring_rows[0])
    missing_forecast["ticker"] = "NOFC"
    missing_forecast["signal_momentum_20d"] = 50.0
    missing_forecast.pop("signal_forecast_er", None)
    missing_forecast.pop("signal_forecast_prob", None)
    scoring_rows = [missing_forecast, *scoring_rows]
    repo = _FakePolicyRepo(policy_rows=policy_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_opportunity_ranker(
        repo,
        settings,
        lookback_days=120,
        horizon_days=20,
        min_ic_dates=30,
        max_scoring_rows=10,
    )

    scored_tickers = {row["ticker"] for row in repo.score_rows}
    assert result.status == "ok"
    assert result.scores_written == len(scoring_rows) - 1
    assert "NOFC" not in scored_tickers
    forecast_filter = repo.run_rows[-1]["detail_json"]["forecast_scoring_filter"]
    assert forecast_filter["loaded_rows"] == len(scoring_rows)
    assert forecast_filter["scored_rows"] == len(scoring_rows) - 1
    assert forecast_filter["dropped_rows"] == 1
    assert forecast_filter["dropped_tickers_sample"] == ["NOFC"]


def test_build_and_store_opportunity_ranker_does_not_filter_scoring_rows_by_share_price() -> None:
    policy_rows, regime_rows, scoring_rows = _synthetic_policy_rows(days=90)
    low_price_row = dict(scoring_rows[0])
    low_price_row.update(
        {
            "ticker": "LOWP",
            "market": "us",
            "close_price_krw": 33.87,
            "signal_momentum_20d": 50.0,
            "signal_forecast_er": 0.20,
            "signal_forecast_prob": 0.35,
        }
    )
    scoring_rows = [low_price_row, *scoring_rows]
    repo = _FakePolicyRepo(policy_rows=policy_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"
    settings.usd_krw_rate = 1498.7

    result = build_and_store_opportunity_ranker(
        repo,
        settings,
        lookback_days=120,
        horizon_days=20,
        min_ic_dates=30,
        max_scoring_rows=10,
    )

    scored_tickers = {row["ticker"] for row in repo.score_rows}
    assert result.status == "ok"
    assert result.scores_written == len(scoring_rows)
    assert "LOWP" in scored_tickers
    assert "investability_filter" not in repo.run_rows[-1]["detail_json"]


def test_build_and_store_opportunity_ranker_overlays_latest_forecast_for_scoring_rows() -> None:
    policy_rows, regime_rows, scoring_rows = _synthetic_policy_rows(days=90)
    missing_forecast = dict(scoring_rows[0])
    missing_forecast["ticker"] = "NOFC"
    missing_forecast["signal_momentum_20d"] = 50.0
    missing_forecast.pop("signal_forecast_er", None)
    missing_forecast.pop("signal_forecast_prob", None)
    scoring_rows = [missing_forecast, *scoring_rows]
    repo = _FakePolicyRepo(
        policy_rows=policy_rows,
        regime_rows=regime_rows,
        scoring_rows=scoring_rows,
        forecast_rows=[
            {
                "ticker": "NOFC",
                "exp_return_period": 0.07,
                "prob_up": 0.8,
            }
        ],
    )
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_opportunity_ranker(
        repo,
        settings,
        lookback_days=120,
        horizon_days=20,
        min_ic_dates=30,
        max_scoring_rows=10,
    )

    nofc = next(row for row in repo.score_rows if row["ticker"] == "NOFC")
    forecast_filter = repo.run_rows[-1]["detail_json"]["forecast_scoring_filter"]
    assert result.status == "ok"
    assert result.scores_written == len(scoring_rows)
    assert nofc["feature_json"]["forecast_er"] == 0.07
    assert abs(nofc["feature_json"]["forecast_prob"] - 0.3) <= 1e-12
    assert forecast_filter["latest_forecast_overlay"]["filled_missing_rows"] == 1
    assert forecast_filter["dropped_rows"] == 0


def test_build_and_store_signal_ic_ranker_writes_ic_scores() -> None:
    ic_rows, regime_rows, scoring_rows = _synthetic_ic_rows(days=120)
    repo = _FakeICRepo(ic_rows=ic_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_signal_ic_ranker(
        repo,
        settings,
        lookback_days=200,
        horizon_days=20,
        min_ic_dates=60,
        max_scoring_rows=10,
    )

    assert result.status == "ok"
    assert result.ranker_version.startswith("opportunity_ranker_ic_")
    assert result.scores_written == len(scoring_rows)
    # Each refresh step was invoked exactly once
    assert repo.refreshes == {"values": 1, "ic": 1, "regime": 1}
    # Score source marker flipped to learned_ic
    assert repo.score_rows[0]["score_source"] == "learned_ic"
    # Top-ranked ticker should include signal contribution breakdown
    top = repo.score_rows[0]
    explanation = top["explanation_json"]
    assert "top_contributions" in explanation
    assert "predicted_ic" in explanation
    assert "model_family" in explanation and explanation["model_family"] == "signal_ic_meta_learner"
    # Tactical override — SQQQ must get tactical_* profile regardless of raw profile
    tactical = [row for row in repo.score_rows if row["ticker"] == "SQQQ"][0]
    assert tactical["profile"] == "tactical_inverse"
    # Run metadata captures per-signal accuracy
    assert repo.run_rows[-1]["status"] == "ok"
    assert repo.run_rows[-1]["score_source"] == "learned_ic"
    detail = repo.run_rows[-1]["detail_json"]
    assert "per_signal_oos_accuracy" in detail
    assert "predicted_ic" in detail


def test_build_and_store_opportunity_ranker_refreshes_daily_sources_only() -> None:
    ic_rows, regime_rows, scoring_rows = _synthetic_ic_rows(days=120)
    repo = _FakeICRepo(ic_rows=ic_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_signal_ic_ranker(
        repo,
        settings,
        lookback_days=200,
        horizon_days=20,
        min_ic_dates=60,
        max_scoring_rows=10,
    )

    assert result.status == "ok"
    assert repo.refresh_value_calls
    sources = repo.refresh_value_calls[-1]["sources"]
    assert sources == [
        "open_trading_us",
        "open_trading_nasdaq",
        "open_trading_nyse",
        "open_trading_amex",
    ]


def test_ranker_returns_unusable_when_ic_history_is_short() -> None:
    # Only 20 dates → below default min_ic_dates threshold
    ic_rows, regime_rows, scoring_rows = _synthetic_ic_rows(days=20)
    repo = _FakeICRepo(ic_rows=ic_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_signal_ic_ranker(
        repo,
        settings,
        lookback_days=60,
        min_ic_dates=60,
    )

    assert result.status == "unusable"
    assert result.scores_written == 0
    assert repo.score_rows == []
    assert repo.run_rows[-1]["status"] == "unusable"


def test_ranker_handles_empty_scoring_rows_gracefully() -> None:
    ic_rows, regime_rows, _ = _synthetic_ic_rows(days=120)
    repo = _FakeICRepo(ic_rows=ic_rows, regime_rows=regime_rows, scoring_rows=[])
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_signal_ic_ranker(
        repo,
        settings,
        lookback_days=200,
        min_ic_dates=60,
    )

    assert result.status == "unusable"
    assert result.scores_written == 0
    assert "no scoring rows" in (result.note or "").lower()


def test_aaa_outranks_ccc_when_momentum_ic_is_positive() -> None:
    ic_rows, regime_rows, scoring_rows = _synthetic_ic_rows(days=120)
    repo = _FakeICRepo(ic_rows=ic_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"

    build_and_store_signal_ic_ranker(
        repo,
        settings,
        lookback_days=200,
        min_ic_dates=60,
        max_scoring_rows=10,
    )
    rank_by_ticker = {row["ticker"]: row["recommendation_rank"] for row in repo.score_rows}
    # AAA has strong momentum + positive momentum IC → must outrank CCC (all negative signals)
    assert rank_by_ticker["AAA"] < rank_by_ticker["CCC"]


class _RecordingSession:
    dataset_fqn = "proj.ds"

    def __init__(self) -> None:
        self.executed: list[tuple[str, dict]] = []

    def execute(self, sql: str, params: dict) -> None:
        self.executed.append((sql, params))


def test_signal_refresh_replaces_market_window_before_insert() -> None:
    session = _RecordingSession()
    store = MarketStore(session)  # type: ignore[arg-type]

    store.refresh_signal_daily_values(lookback_days=540, horizon_days=20, market="us")

    assert len(session.executed) == 2
    delete_sql, delete_params = session.executed[0]
    insert_sql, insert_params = session.executed[1]
    assert "DELETE FROM `proj.ds.signal_daily_values`" in delete_sql
    assert "DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)" in delete_sql
    assert "market = @market" in delete_sql
    assert delete_params["market"] == "us"
    assert insert_sql.lstrip().startswith("INSERT INTO `proj.ds.signal_daily_values`")
    assert insert_params["market"] == "us"


def test_fundamentals_derived_refresh_replaces_market_window_before_insert() -> None:
    session = _RecordingSession()
    store = MarketStore(session)  # type: ignore[arg-type]

    store.refresh_fundamentals_derived_daily(lookback_days=600, market="kospi")

    assert len(session.executed) == 2
    delete_sql, delete_params = session.executed[0]
    insert_sql, insert_params = session.executed[1]
    assert "DELETE FROM `proj.ds.fundamentals_derived_daily`" in delete_sql
    assert "DATE_SUB(CURRENT_DATE(), INTERVAL @lookback_days DAY)" in delete_sql
    assert "market = @market" in delete_sql
    assert delete_params["market"] == "kospi"
    assert insert_sql.lstrip().startswith("INSERT INTO `proj.ds.fundamentals_derived_daily`")
    assert insert_params["market"] == "kospi"
