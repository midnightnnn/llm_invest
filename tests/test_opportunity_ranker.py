from __future__ import annotations

import math
import random
from datetime import date, timedelta

import numpy as np

from arena.config import load_settings
from arena.data.schema import parse_ddl_columns, render_table_ddls
from arena.recommendation import (
    ALL_SIGNALS,
    REGIME_FEATURES,
    SIGNAL_NAMES,
    build_and_store_opportunity_ranker,
)


def test_schema_includes_new_signal_and_fundamentals_tables() -> None:
    ddls = "\n".join(render_table_ddls("proj", "ds"))
    cols = parse_ddl_columns()

    # Signal / regime tables
    assert "proj.ds.signal_daily_values" in ddls
    assert "proj.ds.signal_daily_ic" in ddls
    assert "proj.ds.regime_daily_features" in ddls
    # PIT fundamentals tables
    assert "proj.ds.fundamentals_history_raw" in ddls
    assert "proj.ds.fundamentals_derived_daily" in ddls
    assert "proj.ds.fundamentals_ingest_runs" in ddls

    # Critical columns
    assert ("label_ready", "BOOL") in cols["signal_daily_values"]
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

    def refresh_signal_daily_values(self, **_: object) -> int:
        self.refreshes["values"] = self.refreshes.get("values", 0) + 1
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


def test_build_and_store_opportunity_ranker_writes_ic_scores() -> None:
    ic_rows, regime_rows, scoring_rows = _synthetic_ic_rows(days=120)
    repo = _FakeICRepo(ic_rows=ic_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_opportunity_ranker(
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


def test_ranker_returns_unusable_when_ic_history_is_short() -> None:
    # Only 20 dates → below default min_ic_dates threshold
    ic_rows, regime_rows, scoring_rows = _synthetic_ic_rows(days=20)
    repo = _FakeICRepo(ic_rows=ic_rows, regime_rows=regime_rows, scoring_rows=scoring_rows)
    settings = load_settings()
    settings.kis_target_market = "us"

    result = build_and_store_opportunity_ranker(
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

    result = build_and_store_opportunity_ranker(
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

    build_and_store_opportunity_ranker(
        repo,
        settings,
        lookback_days=200,
        min_ic_dates=60,
        max_scoring_rows=10,
    )
    rank_by_ticker = {row["ticker"]: row["recommendation_rank"] for row in repo.score_rows}
    # AAA has strong momentum + positive momentum IC → must outrank CCC (all negative signals)
    assert rank_by_ticker["AAA"] < rank_by_ticker["CCC"]
