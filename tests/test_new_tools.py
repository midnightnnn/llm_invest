from __future__ import annotations

import json

from arena.config import Settings
from arena.tools.default_registry import build_default_registry
from arena.tools.quant_tools import QuantTools
from arena.tools.sentiment_tools import SentimentTools


class _FakeRepo:
    def __init__(self, series: list[float] | None = None):
        self._series = series or []
        self._cfg: dict[tuple[str, str], str] = {}

    def get_daily_closes(self, *, tickers, lookback_days, sources=None):
        _ = (lookback_days, sources)
        out = {}
        for t in tickers:
            out[t] = list(self._series)
        return out

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        return self._cfg.get((tenant_id, config_key))


class _FakeResp:
    def __init__(self, text: str = "", payload: dict | None = None):
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


def _settings() -> Settings:
    return Settings(
        google_cloud_project="p",
        bq_dataset="d",
        bq_location="loc",
        agent_ids=["gpt"],
        agent_mode="adk",
        base_currency="KRW",
        sleeve_capital_krw=1_000_000,
        log_level="INFO",
        log_format="rich",
        trading_mode="live",
        kis_order_endpoint="",
        kis_api_key="",
        kis_api_secret="",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no="12345678-01",
        kis_account_product_code="01",
        kis_account_key_suffix="",
        kis_env="real",
        kis_target_market="nasdaq",
        kis_overseas_quote_excd="NAS",
        kis_overseas_order_excd="NASD",
        kis_us_natn_cd="840",
        kis_us_tr_mket_cd="01",
        kis_secret_name="KISAPI",
        kis_secret_version="latest",
        kis_http_timeout_seconds=20,
        kis_http_max_retries=1,
        kis_http_backoff_base_seconds=0.1,
        kis_http_backoff_max_seconds=0.2,
        kis_confirm_fills=False,
        kis_confirm_timeout_seconds=20,
        kis_confirm_poll_seconds=1.0,
        usd_krw_rate=1300.0,
        market_sync_history_days=60,
        max_order_krw=500_000,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=0.35,
        min_cash_buffer_ratio=0.1,
        ticker_cooldown_seconds=60,
        max_daily_orders=0,
        estimated_fee_bps=10.0,
        context_max_board_posts=24,
        context_max_memory_events=32,
        context_max_market_rows=64,
        openai_api_key="",
        openai_model="gpt-5.2",
        gemini_api_key="",
        gemini_model="gemini-3-flash-preview",
        research_gemini_model="gemini-2.5-flash",
        llm_timeout_seconds=20,
        default_universe=["AAPL", "MSFT"],
        allow_live_trading=False,
        autonomy_working_set_enabled=True,
        autonomy_tool_default_candidates_enabled=True,
        autonomy_opportunity_context_enabled=True,
    )


def test_default_registry_contains_new_tools() -> None:
    reg = build_default_registry(repo=_FakeRepo(), settings=_settings())
    ids = {e.tool_id for e in reg.list_entries()}
    assert "search_past_experiences" in ids
    assert "portfolio_diagnosis" in ids
    assert "fear_greed_index" in ids
    assert "earnings_calendar" in ids
    assert "technical_signals" in ids
    assert "fetch_reddit_sentiment" not in ids
    assert "get_overseas_fundamentals" not in ids


def test_default_registry_can_enable_reddit_sentiment() -> None:
    settings = _settings()
    settings.reddit_sentiment_enabled = True

    reg = build_default_registry(repo=_FakeRepo(), settings=settings)
    ids = {e.tool_id for e in reg.list_entries()}

    assert "fetch_reddit_sentiment" in ids


def test_default_registry_applies_tools_config_overlay() -> None:
    repo = _FakeRepo()
    repo._cfg[("local", "tools_config")] = json.dumps(
        [
            {
                "tool_id": "portfolio_diagnosis",
                "ui_label_ko": "포트 진단 오버라이드",
                "ui_description_ko": "오버라이드 설명",
                "model_description_override": "Override model description.",
                "sort_order": 5,
            },
            {
                "tool_id": "screen_market",
                "enabled": False,
            },
        ],
        ensure_ascii=False,
    )

    reg = build_default_registry(repo=repo, settings=_settings(), tenant_id="local")
    entries = {entry.tool_id: entry for entry in reg.list_entries(include_disabled=True)}
    visible_ids = {entry.tool_id for entry in reg.list_entries()}

    assert "screen_market" not in visible_ids
    assert entries["screen_market"].enabled is False
    assert entries["portfolio_diagnosis"].label_ko == "포트 진단 오버라이드"
    assert entries["portfolio_diagnosis"].description_ko == "오버라이드 설명"
    assert entries["portfolio_diagnosis"].description == "Override model description."


def test_technical_signals_computes_indicators() -> None:
    series = [100.0 + (i * 0.8) for i in range(90)]
    qt = QuantTools(repo=_FakeRepo(series), settings=_settings())

    out = qt.technical_signals("AAPL", lookback_days=80)

    assert out["ticker"] == "AAPL"
    assert out["points"] >= 80
    assert "rsi_14" in out
    assert "macd" in out
    assert "bollinger_20_2" in out


def test_technical_signals_allows_ticker_outside_default_universe() -> None:
    series = [120.0 + (i * 0.5) for i in range(90)]
    qt = QuantTools(repo=_FakeRepo(series), settings=_settings())

    out = qt.technical_signals("EXC", lookback_days=80)

    assert out["ticker"] == "EXC"
    assert out["points"] >= 80
    assert "error" not in out


def test_technical_signals_supports_multiple_tickers() -> None:
    series = [100.0 + (i * 0.6) for i in range(90)]
    qt = QuantTools(repo=_FakeRepo(series), settings=_settings())

    out = qt.technical_signals(tickers=["AAPL", "MSFT"], lookback_days=80)

    assert out["tickers"] == ["AAPL", "MSFT"]
    assert out["count"] == 2
    assert len(out["rows"]) == 2
    assert {row["ticker"] for row in out["rows"]} == {"AAPL", "MSFT"}


def test_technical_signals_defaults_to_opportunity_working_set() -> None:
    series = [100.0 + (i * 0.6) for i in range(90)]
    qt = QuantTools(repo=_FakeRepo(series), settings=_settings())
    qt.set_context(
        {
            "target_market": "nasdaq",
            "portfolio": {"positions": {"AAPL": {"quantity": 1.0}}},
            "opportunity_working_set": [{"ticker": "MSFT", "status": "pending"}],
        }
    )

    out = qt.technical_signals(lookback_days=80)

    assert out["ticker"] == "MSFT"
    assert out["points"] >= 80


def test_fear_greed_index_vix_proxy(monkeypatch) -> None:
    csv = "DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2026,20,21,19,20\n01/03/2026,25,26,24,25\n01/04/2026,30,31,29,30\n"

    def _fake_get(url, *, headers=None, timeout=10):
        _ = (headers, timeout)
        if "VIX_History" in url:
            return _FakeResp(text=csv)
        return None

    import arena.tools.sentiment_tools as st_mod

    monkeypatch.setattr(st_mod, "_safe_get", _fake_get)
    st = SentimentTools(settings=_settings())

    out = st.fear_greed_index(lookback_days=60)

    assert out["source"] == "cboe_vix"
    assert 0.0 <= float(out["fear_greed_score"]) <= 100.0
    assert out["regime"] in {"Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"}


def test_earnings_calendar_filters_by_ticker(monkeypatch) -> None:
    payload = {
        "data": {
            "rows": [
                {
                    "symbol": "AAPL",
                    "name": "Apple Inc",
                    "time": "After Market Close",
                    "epsForecast": "2.35",
                    "noOfEsts": "24",
                    "lastYearRptDt": "11/02/2025",
                    "lastYearEPS": "2.11",
                },
                {
                    "symbol": "MSFT",
                    "name": "Microsoft Corp",
                    "time": "After Market Close",
                    "epsForecast": "3.10",
                    "noOfEsts": "22",
                    "lastYearRptDt": "10/28/2025",
                    "lastYearEPS": "2.95",
                },
            ]
        }
    }

    def _fake_get(url, *, headers=None, timeout=10):
        _ = (url, headers, timeout)
        return _FakeResp(payload=payload)

    import arena.tools.sentiment_tools as st_mod

    monkeypatch.setattr(st_mod, "_safe_get", _fake_get)
    st = SentimentTools(settings=_settings())

    out = st.earnings_calendar(ticker="AAPL", days_ahead=3, limit=5)

    assert out["ticker"] == "AAPL"
    assert out["count"] == 1
    assert out["rows"][0]["symbol"] == "AAPL"
    assert out["rows"][0]["eps_forecast"] == "2.35"
