from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from types import SimpleNamespace

import arena.cli as cli
import pandas as pd
import pytest
from arena.config import load_settings

from tests.cli.helpers import _FakeRepo, _stub_shared_prep_environment

def test_build_runtime_execution_market_overrides_tenant_market(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.kis_target_market = "kospi"
    settings.openai_api_key = "shared-openai"
    settings.gemini_api_key = ""
    settings.anthropic_api_key = ""
    settings.kis_secret_name = "shared-kis"
    settings.kis_account_no = "12345678"

    class _RuntimeRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    repo = _RuntimeRepo(row={"tenant_id": "tenant-a"})

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "apply_runtime_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "MemoryStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "BoardStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ContextBuilder", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "RiskEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "PaperBroker", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ExecutionGateway", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_build_agents", lambda *args, **kwargs: ["gpt-agent"])
    monkeypatch.setattr(cli, "ArenaOrchestrator", lambda **kwargs: object())

    out_settings, _, _ = cli._build_runtime(
        live=False,
        require_kis=True,
        tenant_id="tenant-a",
        execution_market="us",
    )

    assert out_settings.kis_target_market == "us"


def test_build_runtime_syncs_trading_mode_to_live_flag(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    settings.openai_api_key = "shared-openai"
    settings.gemini_api_key = ""
    settings.anthropic_api_key = ""

    class _RuntimeRepo(_FakeRepo):
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

    repo = _RuntimeRepo(row={"tenant_id": "tenant-a"})

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "apply_runtime_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "MemoryStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "BoardStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ContextBuilder", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "RiskEngine", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "KISOpenTradingBroker", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "ExecutionGateway", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "_build_agents", lambda *args, **kwargs: ["gpt-agent"])
    monkeypatch.setattr(cli, "ArenaOrchestrator", lambda **kwargs: object())

    out_settings, _, _ = cli._build_runtime(
        live=True,
        require_kis=False,
        tenant_id="tenant-a",
    )

    assert out_settings.trading_mode == "live"


def test_build_forecast_tickers_uses_quote_aware_sources() -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    settings.default_universe = []
    settings.universe_run_top_n = 3

    class _Repo:
        def __init__(self) -> None:
            self.last_sources = None

        def latest_universe_candidate_tickers(self, *, limit=200):
            _ = limit
            return ["AAPL", "MSFT", "TSLA"]

        def latest_market_features(self, *, tickers, limit, sources=None):
            self.last_sources = list(sources or [])
            _ = (tickers, limit)
            return [
                {"ticker": "AAPL", "ret_20d": 0.12, "ret_5d": 0.03, "volatility_20d": 0.18, "sentiment_score": 0.2},
                {"ticker": "MSFT", "ret_20d": 0.05, "ret_5d": -0.01, "volatility_20d": 0.12, "sentiment_score": 0.1},
                {"ticker": "TSLA", "ret_20d": -0.08, "ret_5d": 0.04, "volatility_20d": 0.35, "sentiment_score": 0.0},
            ]

        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            _ = (lookback_days, sources)
            base = {"AAPL": 100.0, "MSFT": 120.0, "TSLA": 80.0}
            slopes = {"AAPL": 0.6, "MSFT": 0.2, "TSLA": -0.15}
            out = {}
            for ticker in tickers:
                start = base.get(ticker, 100.0)
                slope = slopes.get(ticker, 0.1)
                out[ticker] = [start + slope * idx for idx in range(140)]
            return out

        def latest_fundamentals_snapshot(self, *, tickers=None, limit=500):
            _ = limit
            allow = set(tickers or [])
            rows = [
                {"ticker": "AAPL", "per": 25.0, "pbr": 8.0, "eps": 5.0, "bps": 18.0, "roe": 16.0, "debt_ratio": 110.0},
                {"ticker": "MSFT", "per": 12.0, "pbr": 2.0, "eps": 9.0, "bps": 35.0, "roe": 19.0, "debt_ratio": 70.0},
            ]
            return [row for row in rows if not allow or row["ticker"] in allow]

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            assert all_tenants is True
            if market == "us,kospi,kosdaq":
                return ["GILD", "AAPL"]
            assert market == "us"
            return ["GILD", "AAPL"]

    repo = _Repo()

    out = cli._build_forecast_tickers(repo, settings, top_n=5)

    assert repo.last_sources == [
        "open_trading_us_quote",
        "open_trading_us",
        "open_trading_nasdaq_quote",
        "open_trading_nasdaq",
        "open_trading_nyse_quote",
        "open_trading_nyse",
        "open_trading_amex_quote",
        "open_trading_amex",
    ]
    assert "GILD" in out
    assert {"AAPL", "MSFT"}.issubset(set(out))


def test_build_forecast_tickers_passes_market_scope_to_runtime_universe() -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    settings.default_universe = []
    settings.universe_run_top_n = 3

    class _Repo:
        def __init__(self) -> None:
            self.last_markets = None

        def latest_universe_candidate_tickers(self, *, limit=200, markets=None):
            _ = limit
            self.last_markets = list(markets or [])
            return ["AAPL", "MSFT", "TSLA"]

        def latest_market_features(self, *, tickers, limit, sources=None):
            _ = (tickers, limit, sources)
            return [
                {"ticker": "AAPL", "ret_20d": 0.12, "ret_5d": 0.03, "volatility_20d": 0.18, "sentiment_score": 0.2},
                {"ticker": "MSFT", "ret_20d": 0.05, "ret_5d": -0.01, "volatility_20d": 0.12, "sentiment_score": 0.1},
                {"ticker": "TSLA", "ret_20d": -0.08, "ret_5d": 0.04, "volatility_20d": 0.35, "sentiment_score": 0.0},
            ]

        def get_daily_closes(self, *, tickers, lookback_days, sources=None):
            _ = (lookback_days, sources)
            return {str(ticker): [100.0 + idx for idx in range(140)] for ticker in tickers}

        def latest_fundamentals_snapshot(self, *, tickers=None, limit=500):
            _ = (tickers, limit)
            return []

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            _ = (market, all_tenants)
            return []

    repo = _Repo()

    cli._build_forecast_tickers(repo, settings, top_n=5)

    assert repo.last_markets == ["us"]


def test_build_forecast_tickers_includes_account_wide_holdings() -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL", "MSFT"]
    settings.forecast_max_tickers = 20

    class _Repo:
        def latest_universe_candidate_tickers(self, *, limit=200, markets=None):
            _ = (limit, markets)
            return ["AAPL", "MSFT"]

        def latest_market_features(self, *, tickers, limit, sources=None):
            _ = (tickers, limit, sources)
            return []

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            assert all_tenants is True
            if market == "us":
                return ["AAPL"]
            if market == "us,kospi,kosdaq":
                return ["AAPL", "053580"]
            return []

    out = cli._build_forecast_tickers(_Repo(), settings, top_n=10)

    assert "AAPL" in out
    assert "053580" in out


def test_cmd_build_forecasts_broadens_sources_for_account_wide_holdings(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL"]

    class _Repo:
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

        def latest_universe_candidate_tickers(self, *, limit=200, markets=None):
            _ = (limit, markets)
            return ["AAPL"]

        def latest_market_features(self, *, tickers, limit, sources=None):
            _ = (tickers, limit, sources)
            return []

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            assert all_tenants is True
            if market == "us,kospi,kosdaq":
                return ["AAPL", "053580"]
            return ["AAPL"]

    captured: dict[str, object] = {}

    def _fake_build(repo, build_settings, **kwargs):
        _ = repo
        captured["market"] = build_settings.kis_target_market
        captured["tickers"] = kwargs.get("tickers")
        return SimpleNamespace(
            run_date="2026-03-13",
            rows_written=2,
            tickers_used=2,
            used_neuralforecast=True,
            model_names=["nbeatsx"],
            note="",
        )

    import arena.forecasting as forecasting_mod

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings: _Repo())
    monkeypatch.setattr(forecasting_mod, "build_and_store_stacked_forecasts", _fake_build)

    args = SimpleNamespace(top_n=10, lookback_days=360, horizon=20, min_series_length=160, max_steps=200)
    cli.cmd_build_forecasts(args)

    assert captured["market"] == "us,kospi,kosdaq"
    assert captured["tickers"] == ["AAPL", "053580"]


def test_cmd_build_forecasts_backfills_only_short_forecast_history(monkeypatch) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj-x"
    settings.bq_dataset = "ds"
    settings.bq_location = "asia-northeast3"
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL", "MSFT", "NEW"]

    class _Repo:
        def ensure_dataset(self):
            return None

        def ensure_tables(self):
            return None

        def latest_universe_candidate_tickers(self, *, limit=200, markets=None):
            _ = (limit, markets)
            return list(settings.default_universe)

        def latest_market_features(self, *, tickers, limit, sources=None):
            _ = (tickers, limit, sources)
            return []

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            _ = (market, all_tenants)
            return ["AAPL", "MSFT", "NEW"]

        def feature_date_spans(self, tickers, source):
            assert source == "open_trading_us"
            assert tickers == ["AAPL", "MSFT", "NEW"]
            return {
                "AAPL": {"min_d": date(2025, 1, 1), "max_d": date(2026, 5, 1), "row_count": 220},
                "MSFT": {"min_d": date(2025, 1, 1), "max_d": date(2026, 5, 1), "row_count": 220},
            }

        def get_daily_close_frame(self, *, tickers, start, end, sources=None):
            _ = (tickers, start, end, sources)
            idx = pd.date_range(end=datetime(2026, 5, 17, tzinfo=timezone.utc), periods=221, freq="D")
            return pd.DataFrame(
                {
                    "AAPL": [100.0 + idx for idx in range(221)],
                    "MSFT": ([None] * 120) + [200.0 + idx for idx in range(101)],
                },
                index=idx,
            )

    repo = _Repo()
    history_calls: list[tuple[str, tuple[str, ...], int]] = []
    captured: dict[str, object] = {}

    def _fake_market_service_factory(**kwargs):
        service_settings = kwargs["settings"]

        class _S:
            def sync_market_features_for_tickers(self_inner, tickers, *, min_daily_rows=0):
                history_calls.append((
                    service_settings.kis_target_market,
                    tuple(tickers),
                    int(min_daily_rows),
                ))
                return SimpleNamespace(
                    inserted_rows=5,
                    attempted_tickers=len(tickers),
                    failed_tickers=[],
                )

        return _S()

    def _fake_build(repo_arg, build_settings, **kwargs):
        _ = (repo_arg, build_settings)
        captured["tickers"] = kwargs.get("tickers")
        return SimpleNamespace(
            run_date="2026-05-17",
            rows_written=2,
            tickers_used=2,
            used_neuralforecast=True,
            model_names=["nbeatsx"],
            note="",
        )

    import arena.forecasting as forecasting_mod

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings: repo)
    monkeypatch.setattr(cli, "MarketDataSyncService", _fake_market_service_factory)
    monkeypatch.setattr(forecasting_mod, "build_and_store_stacked_forecasts", _fake_build)

    args = SimpleNamespace(top_n=10, lookback_days=360, horizon=20, min_series_length=160, max_steps=200)
    cli.cmd_build_forecasts(args)

    assert history_calls == [("us", ("MSFT", "NEW"), 160)]
    assert captured["tickers"] == ["AAPL", "MSFT", "NEW"]


def test_build_forecast_tickers_prefers_ranker_bucket_basket_plus_holdings() -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    settings.default_universe = (
        ["HOLD"]
        + [f"M{i:02d}" for i in range(12)]
        + [f"P{i:02d}" for i in range(12)]
        + [f"A{i:02d}" for i in range(12)]
        + [f"B{i:02d}" for i in range(12)]
        + [f"D{i:02d}" for i in range(12)]
    )
    settings.forecast_ranker_top_per_bucket = 10
    settings.forecast_max_tickers = 80

    class _Repo:
        def __init__(self) -> None:
            self.ranker_calls: list[dict] = []

        def latest_universe_candidate_tickers(self, *, limit=200, markets=None):
            _ = (limit, markets)
            return list(settings.default_universe)

        def latest_opportunity_ranker_scores(self, **kwargs):
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

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            assert all_tenants is True
            if market == "us,kospi,kosdaq":
                return ["HOLD", "M00"]
            assert market == "us"
            return ["HOLD", "M00"]

    repo = _Repo()

    out = cli._build_forecast_tickers(repo, settings, top_n=50)

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
    assert out == (
        ["HOLD"]
        + [f"M{i:02d}" for i in range(10)]
        + [f"P{i:02d}" for i in range(10)]
        + [f"A{i:02d}" for i in range(10)]
        + [f"B{i:02d}" for i in range(10)]
        + [f"D{i:02d}" for i in range(10)]
    )
    assert not ({f"X{i:02d}" for i in range(10)} & set(out))


def test_build_forecast_tickers_merges_discovery_fallback_when_ranker_exists(monkeypatch) -> None:
    import arena.cli_commands.sync as sync_mod

    settings = load_settings()
    settings.kis_target_market = "us"
    settings.default_universe = []
    settings.universe_run_top_n = 10
    settings.forecast_ranker_top_per_bucket = 2
    settings.forecast_max_tickers = 10

    discovery_calls: list[str] = []

    def _fake_build_discovery_rows(latest_rows, *, bucket=None, **kwargs):
        _ = (latest_rows, kwargs)
        bucket_name = str(bucket or "").strip().lower()
        discovery_calls.append(bucket_name)
        if bucket_name == "value":
            return [{"ticker": "DISC"}]
        return []

    monkeypatch.setattr(sync_mod, "build_discovery_rows", _fake_build_discovery_rows)

    class _Repo:
        def latest_universe_candidate_tickers(self, *, limit=200, markets=None):
            _ = (limit, markets)
            return ["HOLD", "RANK", "DISC"]

        def latest_opportunity_ranker_scores(self, **kwargs):
            _ = kwargs
            return [
                {
                    "ticker": "RANK",
                    "market": "us",
                    "bucket": "momentum",
                    "profile": "aggressive",
                    "recommendation_rank": 1,
                    "recommendation_score": 1.0,
                }
            ]

        def latest_market_features(self, *, tickers, limit, sources=None):
            _ = (limit, sources)
            assert tickers == ["HOLD", "RANK", "DISC"]
            return [
                {
                    "ticker": "DISC",
                    "ret_20d": 0.12,
                    "ret_5d": 0.03,
                    "volatility_20d": 0.18,
                }
            ]

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            assert all_tenants is True
            assert market in {"us", "us,kospi,kosdaq"}
            return ["HOLD"]

    out = cli._build_forecast_tickers(_Repo(), settings, top_n=5)

    assert discovery_calls == ["momentum", "pullback", "recovery", "defensive", "value"]
    assert out == ["HOLD", "RANK", "DISC"]


def test_build_forecast_tickers_queries_ranker_with_runtime_universe() -> None:
    settings = load_settings()
    settings.kis_target_market = "us"
    settings.default_universe = ["HOLD"] + [f"IN{i:02d}" for i in range(12)]
    settings.forecast_ranker_top_per_bucket = 10
    settings.forecast_max_tickers = 20

    class _Repo:
        def __init__(self) -> None:
            self.ranker_calls: list[dict] = []

        def latest_universe_candidate_tickers(self, *, limit=200, markets=None):
            _ = (limit, markets)
            return list(settings.default_universe)

        def latest_opportunity_ranker_scores(self, **kwargs):
            self.ranker_calls.append(dict(kwargs))
            assert kwargs.get("tickers") == list(settings.default_universe)
            requested = list(kwargs.get("tickers") or [])
            return [
                {
                    "ticker": ticker,
                    "market": "us",
                    "bucket": "momentum",
                    "profile": "aggressive",
                    "recommendation_rank": idx + 1,
                    "recommendation_score": 1.0 - idx / 100.0,
                }
                for idx, ticker in enumerate(requested)
                if ticker != "HOLD"
            ]

        def get_latest_position_tickers(self, *, market="", all_tenants=False):
            assert all_tenants is True
            assert market in {"us", "us,kospi,kosdaq"}
            return ["HOLD"]

    repo = _Repo()

    out = cli._build_forecast_tickers(repo, settings, top_n=20)

    assert out[:2] == ["HOLD", "IN00"]
    assert all(call.get("tickers") == list(settings.default_universe) for call in repo.ranker_calls)


def test_batch_phase_uses_daily_sources_for_live_us_seed_probe(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    class _Repo:
        def __init__(self) -> None:
            self.coverage_calls: list[tuple[str, date]] = []
            self.distinct_calls: list[str] = []

        def market_daily_ticker_coverage(self, *, source: str, day: date) -> int:
            self.coverage_calls.append((source, day))
            return 0

        def market_source_distinct_tickers(self, *, source: str) -> int:
            self.distinct_calls.append(source)
            return 120

    repo = _Repo()
    now = datetime(2026, 3, 24, 19, 0, tzinfo=timezone.utc)
    window = SimpleNamespace(
        phase="OPEN",
        now_local=now,
        open_utc=now,
        close_utc=now,
        trading_date=date(2026, 3, 24),
    )

    monkeypatch.setenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", "true")
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(cli, "nasdaq_window", lambda _: window)

    phase, out_window = cli._batch_phase(True, settings, repo)

    assert phase == "seed"
    assert out_window is window
    assert repo.coverage_calls[0][0] == "open_trading_us"
    assert all(not source.endswith("_quote") for source, _ in repo.coverage_calls)
    assert repo.distinct_calls == []


def test_batch_phase_uses_recent_daily_coverage_for_live_us_open_cycle(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    class _Repo:
        def __init__(self) -> None:
            self.coverage_calls: list[tuple[str, date]] = []

        def market_daily_ticker_coverage(self, *, source: str, day: date) -> int:
            self.coverage_calls.append((source, day))
            if source == "open_trading_us" and day == date(2026, 3, 23):
                return 120
            return 0

    repo = _Repo()
    now = datetime(2026, 3, 24, 19, 0, tzinfo=timezone.utc)
    window = SimpleNamespace(
        phase="OPEN",
        now_local=now,
        open_utc=now,
        close_utc=now,
        trading_date=date(2026, 3, 24),
    )

    monkeypatch.setenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", "true")
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(cli, "nasdaq_window", lambda _: window)

    phase, out_window = cli._batch_phase(True, settings, repo)

    assert phase == "open_cycle"
    assert out_window is window
    assert ("open_trading_us", date(2026, 3, 24)) in repo.coverage_calls
    assert ("open_trading_us", date(2026, 3, 23)) in repo.coverage_calls


def test_batch_phase_uses_freshest_recent_daily_coverage_for_seed_decision(monkeypatch) -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    class _Repo:
        def market_daily_ticker_coverage(self, *, source: str, day: date) -> int:
            if source != "open_trading_us":
                return 0
            if day == date(2026, 3, 24):
                return 10
            if day == date(2026, 3, 23):
                return 120
            return 0

    repo = _Repo()
    now = datetime(2026, 3, 24, 19, 0, tzinfo=timezone.utc)
    window = SimpleNamespace(
        phase="OPEN",
        now_local=now,
        open_utc=now,
        close_utc=now,
        trading_date=date(2026, 3, 24),
    )

    monkeypatch.setenv("ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD", "true")
    monkeypatch.setattr(cli, "utc_now", lambda: now)
    monkeypatch.setattr(cli, "nasdaq_window", lambda _: window)

    phase, out_window = cli._batch_phase(True, settings, repo)

    assert phase == "seed"
    assert out_window is window
