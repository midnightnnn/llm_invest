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


def test_recommend_opportunities_market_scope_us_narrows_multi_market_agent() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.last_ranker_kwargs = None

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            self.last_ranker_kwargs = {
                "limit": limit,
                "max_age_hours": max_age_hours,
                "tickers": list(tickers) if tickers is not None else None,
                "profiles": profiles,
                "buckets": buckets,
                "markets": list(markets) if markets is not None else None,
                "per_profile_limit": per_profile_limit,
            }
            rows = [
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "joint_policy_v1",
                    "ticker": "AAPL",
                    "market": "us",
                    "profile": "balanced",
                    "bucket": "momentum",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.91,
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
                    "score_source": "joint_policy_v1",
                    "ticker": "005930",
                    "market": "kospi",
                    "profile": "balanced",
                    "bucket": "momentum",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.93,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {},
                    "explanation_json": {},
                },
            ]
            if markets:
                allowed_markets = {str(m).strip().lower() for m in markets}
                rows = [row for row in rows if str(row["market"]).lower() in allowed_markets]
            if tickers:
                allowed_tickers = {str(t).strip().upper() for t in tickers}
                rows = [row for row in rows if str(row["ticker"]).upper() in allowed_tickers]
            return rows

    settings = _settings()
    settings.kis_target_market = "us,kospi,kosdaq"
    settings.default_universe = ["AAPL", "MSFT", "005930", "000660"]
    repo = _Repo()
    qt = QuantTools(repo=repo, settings=settings)

    out = qt.recommend_opportunities(top_n=3, market_scope="us")

    assert out["status"] == "ok"
    assert [row["ticker"] for row in out["recommendations"]] == ["AAPL"]
    assert repo.last_ranker_kwargs["markets"] == ["us"]
    assert repo.last_ranker_kwargs["tickers"] == ["AAPL", "MSFT"]
    assert out["diagnostics"]["selection_scope"]["requested_market_scope"] == "us"
    assert out["diagnostics"]["selection_scope"]["effective_market_scope"] == "us"


def test_recommend_opportunities_market_scope_kr_narrows_multi_market_agent() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.last_ranker_kwargs = None

        def latest_opportunity_ranker_scores(self, *, limit=50, max_age_hours=30, tickers=None, profiles=None, buckets=None, markets=None, per_profile_limit=None):
            self.last_ranker_kwargs = {
                "limit": limit,
                "max_age_hours": max_age_hours,
                "tickers": list(tickers) if tickers is not None else None,
                "profiles": profiles,
                "buckets": buckets,
                "markets": list(markets) if markets is not None else None,
                "per_profile_limit": per_profile_limit,
            }
            rows = [
                {
                    "as_of_date": "2026-04-17",
                    "computed_at": "2026-04-18T00:00:00+00:00",
                    "ranker_version": "opportunity_ranker_20260417_test",
                    "score_source": "joint_policy_v1",
                    "ticker": "AAPL",
                    "market": "us",
                    "profile": "balanced",
                    "bucket": "momentum",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.91,
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
                    "score_source": "joint_policy_v1",
                    "ticker": "005930",
                    "market": "kospi",
                    "profile": "balanced",
                    "bucket": "momentum",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.93,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {},
                    "explanation_json": {},
                },
            ]
            if markets:
                allowed_markets = {str(m).strip().lower() for m in markets}
                rows = [row for row in rows if str(row["market"]).lower() in allowed_markets]
            if tickers:
                allowed_tickers = {str(t).strip().upper() for t in tickers}
                rows = [row for row in rows if str(row["ticker"]).upper() in allowed_tickers]
            return rows

    settings = _settings()
    settings.kis_target_market = "us,kospi,kosdaq"
    settings.default_universe = ["AAPL", "MSFT", "005930", "000660"]
    repo = _Repo()
    qt = QuantTools(repo=repo, settings=settings)

    out = qt.recommend_opportunities(top_n=3, market_scope="kr")

    assert out["status"] == "ok"
    assert [row["ticker"] for row in out["recommendations"]] == ["005930"]
    assert repo.last_ranker_kwargs["markets"] == ["kospi"]
    assert repo.last_ranker_kwargs["tickers"] == ["005930", "000660"]
    assert out["diagnostics"]["selection_scope"]["requested_market_scope"] == "kr"
    assert out["diagnostics"]["selection_scope"]["effective_market_scope"] == "kr"


def test_recommend_opportunities_blocks_market_scope_outside_batch_agent_scope() -> None:
    class _Repo(FakeRepo):
        def latest_opportunity_ranker_scores(self, **kwargs):
            raise AssertionError("ranker should not be queried when market_scope is outside agent scope")

    settings = _settings()
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL", "MSFT"]
    qt = QuantTools(repo=_Repo(), settings=settings)

    out = qt.recommend_opportunities(top_n=3, market_scope="kr")

    assert out["status"] == "unusable"
    assert out["recommendations"] == []
    assert out["diagnostics"]["selection_scope"]["requested_market_scope"] == "kr"
    assert out["diagnostics"]["selection_scope"]["scope_error"] == "out_of_market_scope"


def test_recommend_opportunities_does_not_filter_saved_rows_by_share_price() -> None:
    class _Repo(FakeRepo):
        def __init__(self):
            super().__init__()
            self.universe_rows = ["LOWP", "MSFT"]

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
                    "score_source": "joint_policy_v1",
                    "ticker": "LOWP",
                    "market": "us",
                    "profile": "aggressive",
                    "bucket": "momentum",
                    "recommendation_rank": 1,
                    "recommendation_score": 0.90,
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
                    "score_source": "joint_policy_v1",
                    "ticker": "MSFT",
                    "market": "us",
                    "profile": "balanced",
                    "bucket": "pullback",
                    "recommendation_rank": 2,
                    "recommendation_score": 0.80,
                    "model_confidence": "medium",
                    "action": "candidate",
                    "evidence_level": "validated",
                    "feature_json": {},
                    "explanation_json": {},
                },
            ]

        def latest_market_features(self, tickers, limit, sources=None):
            self.last_market_kwargs = {
                "tickers": list(tickers),
                "limit": limit,
                "sources": list(sources) if sources is not None else None,
            }
            return [
                {
                    "ticker": "LOWP",
                    "market": "us",
                    "exchange_code": "NASD",
                    "close_price_native": 0.0226,
                    "close_price_krw": 33.87,
                    "quote_currency": "USD",
                },
                {
                    "ticker": "MSFT",
                    "market": "us",
                    "exchange_code": "NASD",
                    "close_price_native": 425.0,
                    "close_price_krw": 552_500.0,
                    "quote_currency": "USD",
                },
            ]

    settings = _settings()
    settings.kis_target_market = "us"
    settings.default_universe = ["LOWP", "MSFT"]
    repo = _Repo()
    qt = QuantTools(repo=repo, settings=settings)

    out = qt.recommend_opportunities(top_n=3)

    assert out["status"] == "ok"
    assert [row["ticker"] for row in out["recommendations"]] == ["LOWP", "MSFT"]
    assert "investability_filter" not in out["diagnostics"]


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
