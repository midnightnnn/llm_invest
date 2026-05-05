from __future__ import annotations

from datetime import datetime, timezone

from arena.tools.quant_tools import QuantTools

from tests.tools.quant_helpers import FakeRepo, _settings


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

        def latest_opportunity_ranker_scores(
            self,
            *,
            limit=50,
            max_age_hours=30,
            tickers=None,
            profiles=None,
            buckets=None,
            markets=None,
            per_profile_limit=None,
        ):
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
        def latest_opportunity_ranker_scores(
            self,
            *,
            limit=50,
            max_age_hours=30,
            tickers=None,
            profiles=None,
            buckets=None,
            markets=None,
            per_profile_limit=None,
        ):
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
        def latest_opportunity_ranker_scores(
            self,
            *,
            limit=50,
            max_age_hours=30,
            tickers=None,
            profiles=None,
            buckets=None,
            markets=None,
            per_profile_limit=None,
        ):
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
