from __future__ import annotations

from datetime import datetime, timezone
import math
from types import SimpleNamespace
from typing import Literal, get_args, get_origin, get_type_hints

import pytest

from arena.config import Settings
from arena.tools.quant_tools import QuantTools

from tests.tools.quant_helpers import (
    _literal_args,
    _stable_quant_tool_now,
    FakeRepo,
    FakeOpenTradingClient,
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


def test_recommend_opportunities_uses_calendar_lookup_for_latest_weekend_rows(monkeypatch) -> None:
    import arena.tools.quant_tools as qt_module

    monkeypatch.setattr(
        qt_module,
        "_utc_now",
        lambda: datetime(2026, 5, 3, 8, 30, tzinfo=timezone.utc),
        raising=False,
    )

    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.last_ranker_kwargs = None

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            self.last_ranker_kwargs = {
                "limit": limit,
                "max_age_hours": max_age_hours,
                "markets": markets,
                "per_profile_limit": per_profile_limit,
            }
            if max_age_hours < 72:
                return []
            return [
                {
                    "as_of_date": "2026-05-01",
                    "computed_at": "2026-05-01T19:13:34+00:00",
                    "ranker_version": "opportunity_ranker_20260501_test",
                    "score_source": "learned_ic",
                    "ticker": "MSFT",
                    "market": "us",
                    "profile": "balanced",
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

    out = qt.recommend_opportunities(top_n=3, max_score_age_hours=30)

    assert out["status"] == "ok"
    assert out["recommendations"][0]["ticker"] == "MSFT"
    assert repo.last_ranker_kwargs["max_age_hours"] >= 72
    assert out["diagnostics"]["freshness"]["by_market"]["us"]["status"] == "ok"
    assert out["diagnostics"]["freshness"]["by_market"]["us"]["market_phase"] == "CLOSED"


def test_recommend_opportunities_marks_open_session_before_current_prep_degraded(monkeypatch) -> None:
    import arena.tools.quant_tools as qt_module

    monkeypatch.setattr(
        qt_module,
        "_utc_now",
        lambda: datetime(2026, 5, 4, 14, 0, tzinfo=timezone.utc),
        raising=False,
    )

    class _Repo(FakeRepo):
        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            return [
                {
                    "as_of_date": "2026-05-01",
                    "computed_at": "2026-05-01T19:13:34+00:00",
                    "ranker_version": "opportunity_ranker_20260501_test",
                    "score_source": "learned_ic",
                    "ticker": "MSFT",
                    "market": "us",
                    "profile": "balanced",
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

    qt = QuantTools(repo=_Repo(), settings=_settings())

    out = qt.recommend_opportunities(top_n=3)

    assert out["status"] == "degraded"
    freshness = out["diagnostics"]["freshness"]["by_market"]["us"]
    assert freshness["status"] == "degraded"
    assert freshness["reason_code"] == "current_session_prep_missing"
    assert freshness["market_phase"] == "OPEN"


def test_recommend_opportunities_rejects_ranker_before_latest_reference_session(monkeypatch) -> None:
    import arena.tools.quant_tools as qt_module

    monkeypatch.setattr(
        qt_module,
        "_utc_now",
        lambda: datetime(2026, 5, 5, 14, 0, tzinfo=timezone.utc),
        raising=False,
    )

    class _Repo(FakeRepo):
        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            return [
                {
                    "as_of_date": "2026-05-01",
                    "computed_at": "2026-05-01T19:13:34+00:00",
                    "ranker_version": "opportunity_ranker_20260501_test",
                    "score_source": "learned_ic",
                    "ticker": "MSFT",
                    "market": "us",
                    "profile": "balanced",
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

    qt = QuantTools(repo=_Repo(), settings=_settings())

    out = qt.recommend_opportunities(top_n=3)

    assert out["status"] == "unusable"
    assert out["recommendations"] == []
    assert out["diagnostics"]["freshness"]["by_market"]["us"]["reason_code"] == "older_than_latest_reference_session"
