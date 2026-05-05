from __future__ import annotations

from arena.tools.quant_tools import QuantTools

from tests.tools.quant_helpers import (
    _stable_quant_tool_now,
    FakeRepo,
    _settings,
)

def test_forecast_returns_reads_predictions() -> None:
    repo = FakeRepo()
    qt = QuantTools(repo=repo, settings=_settings())
    rows = qt.forecast_returns()
    assert len(rows) >= 1
    assert rows[0]["run_date"]
    assert "ticker" in rows[0]
    assert "exp_return_period" in rows[0]
    assert all(r["ticker"] in {"AAPL", "MSFT", "TSLA", "PLTD"} for r in rows)
    assert repo.last_forecast_mode == "all"


def test_forecast_returns_prefers_dynamic_candidate_tickers_from_context() -> None:
    class _Repo(FakeRepo):
        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

    repo = _Repo()
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=repo, settings=settings)
    context = {
        "target_market": "nasdaq",
        "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
        "_candidate_tickers": ["TSLA"],
    }
    qt.set_context(context)
    context["_candidate_tickers"] = ["PLTD", "TSLA"]

    rows = qt.forecast_returns()

    assert set(repo.last_forecast_tickers) == {"AAPL", "PLTD", "TSLA"}
    assert {row["ticker"] for row in rows} == {"AAPL", "PLTD", "TSLA"}


def test_forecast_returns_prefers_opportunity_working_set_over_raw_candidate_list() -> None:
    class _Repo(FakeRepo):
        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

    repo = _Repo()
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=repo, settings=settings)
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
            "_candidate_tickers": ["TSLA"],
            "opportunity_working_set": [{"ticker": "PLTD", "status": "pending"}],
        }
    )

    rows = qt.forecast_returns()

    assert set(repo.last_forecast_tickers) == {"AAPL", "PLTD"}
    assert {row["ticker"] for row in rows} == {"AAPL", "PLTD"}


def test_forecast_returns_prefers_full_discovered_basket_over_working_set() -> None:
    class _Repo(FakeRepo):
        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

    repo = _Repo()
    settings = _settings()
    settings.default_universe = ["AAPL", "MSFT", "TSLA", "PLTD"]
    qt = QuantTools(repo=repo, settings=settings)
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
            "_candidate_tickers": ["TSLA"],
            "_discovered_candidate_tickers": ["PLTD", "TSLA", "MSFT"],
            "opportunity_working_set": [{"ticker": "PLTD", "status": "pending"}],
        }
    )

    rows = qt.forecast_returns()

    assert set(repo.last_forecast_tickers) == {"AAPL", "PLTD", "TSLA", "MSFT"}
    assert {row["ticker"] for row in rows} == {"AAPL", "PLTD", "TSLA", "MSFT"}


def test_forecast_returns_defaults_to_ranker_buckets_plus_holdings() -> None:
    class _Repo(FakeRepo):
        def __init__(self) -> None:
            super().__init__()
            self._preds.extend(
                {"run_date": "2026-01-02", "ticker": f"M{i:02d}", "exp_return_period": 0.01, "forecast_horizon": 20}
                for i in range(12)
            )
            self._preds.extend(
                {"run_date": "2026-01-02", "ticker": f"P{i:02d}", "exp_return_period": 0.01, "forecast_horizon": 20}
                for i in range(12)
            )
            self._preds.extend(
                {
                    "run_date": "2026-01-02",
                    "ticker": f"A{i:02d}",
                    "exp_return_period": 0.01,
                    "forecast_horizon": 20,
                }
                for i in range(12)
            )
            self._preds.extend(
                {
                    "run_date": "2026-01-02",
                    "ticker": f"B{i:02d}",
                    "exp_return_period": 0.01,
                    "forecast_horizon": 20,
                }
                for i in range(12)
            )
            self._preds.extend(
                {
                    "run_date": "2026-01-02",
                    "ticker": f"D{i:02d}",
                    "exp_return_period": 0.01,
                    "forecast_horizon": 20,
                }
                for i in range(12)
            )
            self._preds.append({"run_date": "2026-01-02", "ticker": "HOLD", "exp_return_period": 0.04, "forecast_horizon": 20})

        def get_predicted_returns(self, tickers=None, limit=50, mode="stacked", table_id=None, staleness_days=None):
            self.last_forecast_tickers = list(tickers) if tickers is not None else None
            return super().get_predicted_returns(
                tickers=tickers,
                limit=limit,
                mode=mode,
                table_id=table_id,
                staleness_days=staleness_days,
            )

        def latest_opportunity_ranker_scores(self, **kwargs):
            self.ranker_calls = getattr(self, "ranker_calls", [])
            self.ranker_calls.append(dict(kwargs))
            rows = []
            for idx in range(12):
                rows.append(
                    {
                        "ticker": f"M{idx:02d}",
                        "market": "us",
                        "bucket": "momentum",
                        "profile": "aggressive",
                        "recommendation_rank": idx + 20,
                        "recommendation_score": 1.0 - idx / 100.0,
                    }
                )
                rows.append(
                    {
                        "ticker": f"P{idx:02d}",
                        "market": "us",
                        "bucket": "pullback",
                        "profile": "balanced",
                        "recommendation_rank": idx + 20,
                        "recommendation_score": 0.8 - idx / 100.0,
                    }
                )
                rows.append(
                    {
                        "ticker": f"X{idx:02d}",
                        "market": "us",
                        "bucket": "recovery",
                        "profile": "value",
                        "recommendation_rank": idx + 1,
                        "recommendation_score": 0.9 - idx / 100.0,
                    }
                )
                rows.append(
                    {
                        "ticker": f"A{idx:02d}",
                        "market": "us",
                        "bucket": "profile_aggressive",
                        "profile": "aggressive",
                        "recommendation_rank": idx + 1,
                        "recommendation_score": 0.7 - idx / 100.0,
                    }
                )
                rows.append(
                    {
                        "ticker": f"B{idx:02d}",
                        "market": "us",
                        "bucket": "profile_balanced",
                        "profile": "balanced",
                        "recommendation_rank": idx + 1,
                        "recommendation_score": 0.6 - idx / 100.0,
                    }
                )
                rows.append(
                    {
                        "ticker": f"D{idx:02d}",
                        "market": "us",
                        "bucket": "profile_defensive",
                        "profile": "defensive",
                        "recommendation_rank": idx + 1,
                        "recommendation_score": 0.5 - idx / 100.0,
                    }
                )
            buckets = kwargs.get("buckets") or []
            if buckets:
                allow = {str(bucket).strip().lower() for bucket in buckets}
                rows = [row for row in rows if str(row.get("bucket") or "").lower() in allow]
            profiles = kwargs.get("profiles") or []
            if profiles:
                allow = {str(profile).strip().lower() for profile in profiles}
                rows = [row for row in rows if str(row.get("profile") or "").lower() in allow]
            return rows

    repo = _Repo()
    settings = _settings()
    settings.default_universe = (
        ["HOLD"]
        + [f"M{i:02d}" for i in range(12)]
        + [f"P{i:02d}" for i in range(12)]
        + [f"A{i:02d}" for i in range(12)]
        + [f"B{i:02d}" for i in range(12)]
        + [f"D{i:02d}" for i in range(12)]
    )
    qt = QuantTools(repo=repo, settings=settings)
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"HOLD": {"quantity": 1.0}}},
        }
    )

    rows = qt.forecast_returns()

    assert [call["buckets"] for call in repo.ranker_calls if call.get("buckets")] == [
        ["momentum"],
        ["pullback"],
        ["recovery"],
        ["defensive"],
    ]
    assert [call["profiles"] for call in repo.ranker_calls if call.get("profiles")] == [
        ["aggressive"],
        ["balanced"],
        ["defensive"],
    ]
    assert set(repo.last_forecast_tickers) == (
        {"HOLD"}
        | {f"M{i:02d}" for i in range(10)}
        | {f"P{i:02d}" for i in range(10)}
        | {f"A{i:02d}" for i in range(10)}
        | {f"B{i:02d}" for i in range(10)}
        | {f"D{i:02d}" for i in range(10)}
    )
    assert not ({f"X{i:02d}" for i in range(10)} & set(repo.last_forecast_tickers))
    assert {row["ticker"] for row in rows} == set(repo.last_forecast_tickers)

