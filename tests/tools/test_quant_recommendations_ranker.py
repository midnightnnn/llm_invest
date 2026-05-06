from __future__ import annotations

from arena.tools.quant_tools import QuantTools

from tests.tools.quant_helpers import (
    FakeRepo,
    _stable_quant_tool_now,
    _settings,
)

def test_recommend_opportunities_uses_precomputed_learned_scores() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.last_ranker_kwargs = None

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            self.last_ranker_kwargs = {
                "limit": limit,
                "max_age_hours": max_age_hours,
                "tickers": tickers,
                "profiles": profiles,
                "buckets": buckets,
                "markets": markets,
                "per_profile_limit": per_profile_limit,
            }
            return [
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "learned",
                    "ticker": "MSFT",
                    "market": "us",
                    "profile": "defensive",
                    "bucket": "defensive",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.041,
                    "predicted_excess_return_20d": 0.032,
                    "prob_outperform_20d": 0.61,
                    "predicted_drawdown_20d": -0.045,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {"ret_20d": 0.08, "forecast_exp_return": 0.03},
                    "explanation_json": {"top_features": ["forecast_exp_return", "screen_score"]},
                }
            ]

    repo = _Repo()
    qt = QuantTools(repo=repo, settings=_settings())

    out = qt.recommend_opportunities(top_n=3)

    assert out["status"] == "ok"
    assert out["ranker"]["score_source"] == "learned"
    assert out["recommendations"][0]["ticker"] == "MSFT"
    assert out["recommendations"][0]["score_source"] == "learned"
    assert out["recommendations"][0]["predicted_excess_return_20d"] == 0.032
    assert repo.last_ranker_kwargs["limit"] == 3
    assert repo.last_ranker_kwargs["per_profile_limit"] == 3
    assert repo.last_ranker_kwargs["markets"] == ["us"]
    assert repo.last_ranker_kwargs["tickers"] == ["AAPL", "MSFT", "TSLA"]
    assert out["diagnostics"]["selection_scope"]["mode"] == "ranked_union"
    assert out["diagnostics"]["selection_scope"]["global_limit"] == 3
    assert out["diagnostics"]["selection_scope"]["per_profile_limit"] == 3
    assert out["diagnostics"]["selection_scope"]["loaded_rows"] == 1
    assert out["diagnostics"]["selection_scope"]["markets"] == ["us"]


def test_recommend_opportunities_uses_ranker_bucket_filter() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.last_ranker_kwargs = None

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            self.last_ranker_kwargs = {
                "limit": limit,
                "max_age_hours": max_age_hours,
                "tickers": tickers,
                "profiles": profiles,
                "buckets": buckets,
                "markets": markets,
                "per_profile_limit": per_profile_limit,
            }
            return [
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "learned_ic",
                    "ticker": "MSFT",
                    "market": "us",
                    "profile": "aggressive",
                    "bucket": "momentum",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.041,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {},
                    "explanation_json": {},
                }
            ]

    repo = _Repo()
    qt = QuantTools(repo=repo, settings=_settings())

    out = qt.recommend_opportunities(top_n=8, buckets=["momentum"])

    assert out["status"] == "ok"
    assert repo.last_ranker_kwargs["limit"] == 8
    assert repo.last_ranker_kwargs["per_profile_limit"] == 8
    assert repo.last_ranker_kwargs["buckets"] == ["momentum"]
    assert repo.last_ranker_kwargs["profiles"] is None
    assert out["diagnostics"]["selection_scope"]["requested_buckets"] == ["momentum"]


def test_recommend_opportunities_accepts_profiles_filter() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.last_ranker_kwargs = None

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            self.last_ranker_kwargs = {
                "limit": limit,
                "max_age_hours": max_age_hours,
                "tickers": tickers,
                "profiles": profiles,
                "buckets": buckets,
                "markets": markets,
                "per_profile_limit": per_profile_limit,
            }
            rows = [
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "learned_ic",
                    "ticker": "MSFT",
                    "market": "us",
                    "profile": "balanced",
                    "bucket": "pullback",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.041,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {},
                    "explanation_json": {},
                },
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "learned_ic",
                    "ticker": "AAPL",
                    "market": "us",
                    "profile": "aggressive",
                    "bucket": "momentum",
                    "recommendation_rank": 2,
                    "recommendation_score": 0.039,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {},
                    "explanation_json": {},
                },
            ]
            if profiles:
                allowed_profiles = set(profiles)
                rows = [row for row in rows if row["profile"] in allowed_profiles]
            if buckets:
                allowed_buckets = set(buckets)
                rows = [row for row in rows if row["bucket"] in allowed_buckets]
            return rows

    repo = _Repo()
    qt = QuantTools(repo=repo, settings=_settings())

    out = qt.recommend_opportunities(top_n=3, profiles=["balanced"])

    assert out["status"] == "ok"
    assert [row["ticker"] for row in out["recommendations"]] == ["MSFT"]
    assert repo.last_ranker_kwargs["profiles"] == ["balanced"]
    assert repo.last_ranker_kwargs["buckets"] is None
    assert out["diagnostics"]["selection_scope"]["requested_profiles"] == ["balanced"]


def test_recommend_opportunities_normalizes_legacy_profile_bucket_filter() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.last_ranker_kwargs = None

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            self.last_ranker_kwargs = {
                "limit": limit,
                "max_age_hours": max_age_hours,
                "tickers": tickers,
                "profiles": profiles,
                "buckets": buckets,
                "markets": markets,
                "per_profile_limit": per_profile_limit,
            }
            rows = [
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "learned_ic",
                    "ticker": "MSFT",
                    "market": "us",
                    "profile": "balanced",
                    "bucket": "pullback",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.041,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {},
                    "explanation_json": {},
                }
            ]
            if profiles:
                rows = [row for row in rows if row["profile"] in set(profiles)]
            if buckets:
                rows = [row for row in rows if row["bucket"] in set(buckets)]
            return rows

    repo = _Repo()
    qt = QuantTools(repo=repo, settings=_settings())

    out = qt.recommend_opportunities(top_n=3, buckets=["balanced"])

    assert out["status"] == "ok"
    assert [row["ticker"] for row in out["recommendations"]] == ["MSFT"]
    assert repo.last_ranker_kwargs["profiles"] == ["balanced"]
    assert repo.last_ranker_kwargs["buckets"] is None
    assert out["diagnostics"]["selection_scope"]["legacy_profile_bucket_tokens"] == ["balanced"]


def test_recommend_opportunities_retries_unfiltered_for_empty_legacy_profile_bucket_filter() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.calls = []

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            self.calls.append(
                {
                    "limit": limit,
                    "max_age_hours": max_age_hours,
                    "tickers": tickers,
                    "profiles": profiles,
                    "buckets": buckets,
                    "markets": markets,
                    "per_profile_limit": per_profile_limit,
                }
            )
            if profiles or buckets:
                return []
            return [
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "learned_ic",
                    "ticker": "AAPL",
                    "market": "us",
                    "profile": "aggressive",
                    "bucket": "momentum",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.039,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {},
                    "explanation_json": {},
                }
            ]

    repo = _Repo()
    qt = QuantTools(repo=repo, settings=_settings())

    out = qt.recommend_opportunities(top_n=3, buckets=["defensive"])

    assert out["status"] == "ok"
    assert [row["ticker"] for row in out["recommendations"]] == ["AAPL"]
    assert repo.calls[0]["profiles"] == ["defensive"]
    assert repo.calls[1]["profiles"] is None
    assert out["diagnostics"]["selection_scope"]["loaded_rows_before_filter_fallback"] == 0
    assert "filter_fallback" in out["diagnostics"]


def test_recommend_opportunities_learned_missing_is_not_silent_heuristic_fallback() -> None:
    qt = QuantTools(repo=FakeRepo(), settings=_settings())

    out = qt.recommend_opportunities(top_n=3)

    assert out["status"] == "unusable"
    assert out["recommendations"] == []
    assert out["ranker"]["score_source"] == "missing"
