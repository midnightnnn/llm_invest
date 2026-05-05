from __future__ import annotations

import numpy as np
import pandas as pd

from arena.config import Settings
import arena.forecasting.stacked as stacked_mod
from arena.models import utc_now


class _FakeRepo:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.universe_rows: list[str] = ["AAPL", "MSFT", "TSLA", "NVDA"]
        self.last_universe_limit: int | None = None

    def get_daily_close_frame(self, *, tickers, start, end, sources=None):  # noqa: ANN001
        _ = (start, end, sources)
        days = pd.bdate_range(end=utc_now().date(), periods=260)
        out = {}
        for i, t in enumerate(tickers):
            base = 100.0 + float(i * 7)
            noise = np.sin(np.linspace(0, 9, len(days))) * 0.005
            drift = np.linspace(0, 0.2 + i * 0.03, len(days))
            rets = noise + drift / len(days)
            px = base * np.exp(np.cumsum(rets))
            out[str(t).upper()] = px
        return pd.DataFrame(out, index=days)

    def replace_predicted_returns(self, rows, *, run_date):  # noqa: ANN001
        _ = run_date
        self.rows = list(rows)
        return len(self.rows)

    def latest_universe_candidate_tickers(self, *, limit=200):
        self.last_universe_limit = limit
        return list(self.universe_rows[:limit])


def _settings() -> Settings:
    return Settings(
        google_cloud_project="p",
        bq_dataset="llm_arena",
        bq_location="loc",
        agent_ids=["gpt"],
        agent_mode="adk",
        base_currency="KRW",
        sleeve_capital_krw=2_000_000,
        log_level="INFO",
        log_format="rich",
        trading_mode="paper",
        kis_order_endpoint="",
        kis_api_key="",
        kis_api_secret="",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no="",
        kis_account_product_code="01",
        kis_account_key_suffix="",
        kis_env="demo",
        kis_target_market="nasdaq",
        kis_overseas_quote_excd="NAS",
        kis_overseas_order_excd="NASD",
        kis_us_natn_cd="840",
        kis_us_tr_mket_cd="01",
        kis_secret_name="KISAPI",
        kis_secret_version="latest",
        kis_http_timeout_seconds=20,
        kis_http_max_retries=0,
        kis_http_backoff_base_seconds=0.1,
        kis_http_backoff_max_seconds=0.1,
        kis_confirm_fills=False,
        kis_confirm_timeout_seconds=25,
        kis_confirm_poll_seconds=2.0,
        usd_krw_rate=1300.0,
        market_sync_history_days=60,
        max_order_krw=350_000,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=0.35,
        min_cash_buffer_ratio=0.10,
        ticker_cooldown_seconds=120,
        max_daily_orders=0,
        estimated_fee_bps=10.0,
        context_max_board_posts=24,
        context_max_memory_events=32,
        context_max_market_rows=64,
        openai_api_key="",
        openai_model="gpt-5.2",
        gemini_api_key="",
        gemini_model="gemini-3-pro-preview",
        research_gemini_model="gemini-2.5-flash",
        llm_timeout_seconds=25,
        default_universe=["AAPL", "MSFT", "TSLA", "NVDA"],
        allow_live_trading=False,
    )


def test_normalize_universe_for_us_excludes_numeric_codes() -> None:
    settings = _settings()
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL", "EXC", "005930", "123456"]
    # Without repo, only seed tickers are returned (no DB expansion)
    assert stacked_mod._normalize_universe(settings) == ["AAPL", "EXC"]


def test_normalize_universe_uses_default_universe_only() -> None:
    settings = _settings()
    settings.kis_target_market = "us"
    settings.default_universe = ["AAPL", "NVDA"]

    result = stacked_mod._normalize_universe(settings)
    assert result == ["AAPL", "NVDA"]


def test_normalize_universe_loads_latest_universe_candidates_when_default_empty() -> None:
    settings = _settings()
    settings.kis_target_market = "us"
    settings.default_universe = []
    settings.universe_run_top_n = 3
    repo = _FakeRepo()
    repo.universe_rows = ["AAPL", "005930", "NVDA", "123456"]

    result = stacked_mod._normalize_universe(settings, repo)

    assert result == ["AAPL", "NVDA"]
    assert repo.last_universe_limit == 3


def test_forecast_sources_for_us_include_quote_and_legacy() -> None:
    settings = _settings()
    settings.kis_target_market = "us"
    assert stacked_mod._forecast_sources(settings) == [
        "open_trading_us_quote",
        "open_trading_us",
        "open_trading_nasdaq_quote",
        "open_trading_nasdaq",
        "open_trading_nyse_quote",
        "open_trading_nyse",
        "open_trading_amex_quote",
        "open_trading_amex",
    ]


def test_build_and_store_stacked_forecasts_writes_rows(monkeypatch) -> None:  # noqa: ANN001
    def _fake_collect_base(log_returns, *, horizon, min_len, max_steps):  # noqa: ANN001
        _ = (horizon, min_len, max_steps)
        out = {
            "NBEATSx": {},
            "NHITS": {},
            "PatchTST": {},
            "iTransformer": {},
            "Chronos": {},
            "TimesFM": {},
            "LagLlama": {},
        }
        for t in log_returns.columns:
            s = log_returns[t].dropna().to_numpy(dtype=float)
            if s.size < 20:
                continue
            out["NBEATSx"][t] = float(np.mean(s[-40:]))
            out["NHITS"][t] = float(np.median(s[-40:]))
            out["PatchTST"][t] = float(np.mean(s[-20:]))
            out["iTransformer"][t] = float(np.mean(s[-60:]))
            out["Chronos"][t] = float(np.mean(s[-15:]))
            out["TimesFM"][t] = float(np.mean(s[-25:]))
            out["LagLlama"][t] = float(np.mean(s[-35:]))
        return out, True

    monkeypatch.setattr(stacked_mod, "_require_forecasting_dependencies", lambda: None)
    monkeypatch.setattr(stacked_mod, "_collect_base_predictions", _fake_collect_base)

    repo = _FakeRepo()
    settings = _settings()
    result = stacked_mod.build_and_store_stacked_forecasts(
        repo,
        settings,
        lookback_days=220,
        horizon=10,
        min_series_length=120,
        max_steps=50,
    )
    assert result.rows_written > 0
    assert result.tickers_used >= 2
    assert "ensemble_wmae" in result.model_names
    assert "ensemble_avg" in result.model_names
    assert any(bool(r.get("is_stacked")) for r in repo.rows)
    assert all("forecast_model" in r for r in repo.rows)
    assert all("exp_return_period" in r for r in repo.rows)
    assert all("forecast_horizon" in r for r in repo.rows)

    # ensemble_wmae should produce varied per-ticker predictions (not constant)
    wmae_returns = [r["exp_return_period"] for r in repo.rows if r["forecast_model"] == "ensemble_wmae"]
    assert len(wmae_returns) >= 2
    assert len(set(round(v, 10) for v in wmae_returns)) > 1


def test_build_and_store_stacked_forecasts_keeps_tickers_with_partial_model_coverage(monkeypatch) -> None:  # noqa: ANN001
    calls = {"count": 0}

    def _fake_collect_base(log_returns, *, horizon, min_len, max_steps):  # noqa: ANN001
        _ = (horizon, min_len, max_steps)
        calls["count"] += 1
        tickers = [str(t) for t in log_returns.columns]
        val_stage = calls["count"] == 1
        out = {"ModelA": {}, "ModelB": {}}
        for ticker in tickers:
            series = log_returns[ticker].dropna().to_numpy(dtype=float)
            if series.size < 20:
                continue
            out["ModelA"][ticker] = float(np.mean(series[-20:]))
            out["ModelB"][ticker] = float(np.mean(series[-30:]))
        if not val_stage:
            out["ModelA"].pop("TSLA", None)
        return out, False

    monkeypatch.setattr(stacked_mod, "_require_forecasting_dependencies", lambda: None)
    monkeypatch.setattr(stacked_mod, "_collect_base_predictions", _fake_collect_base)

    repo = _FakeRepo()
    settings = _settings()
    result = stacked_mod.build_and_store_stacked_forecasts(
        repo,
        settings,
        lookback_days=220,
        horizon=10,
        min_series_length=120,
        max_steps=50,
    )

    assert result.rows_written > 0
    assert result.tickers_used >= 3
    tsla_rows = [row for row in repo.rows if row["ticker"] == "TSLA"]
    assert tsla_rows
    assert any(row["forecast_model"] == "ensemble_wmae" for row in tsla_rows)
    assert any(row["forecast_model"] == "ModelB" for row in tsla_rows)
    assert not any(row["forecast_model"] == "ModelA" for row in tsla_rows)
    tsla_stack = next(row for row in tsla_rows if row["forecast_model"] == "ensemble_wmae")
    assert int(tsla_stack["model_votes_total"]) == 1
