from dataclasses import replace
from datetime import timedelta

from arena.config import Settings
from arena.models import AccountSnapshot, OrderIntent, Position, Side, utc_now
from arena.risk import RiskEngine


def _settings() -> Settings:
    """Builds a deterministic Settings object for risk tests."""
    return Settings(
        google_cloud_project="p",
        bq_dataset="d",
        bq_location="loc",
        agent_ids=["a"],
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
        kis_target_market="kospi",
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
        max_order_krw=200_000,
        max_daily_turnover_ratio=0.5,
        max_position_ratio=0.5,
        min_cash_buffer_ratio=0.1,
        ticker_cooldown_seconds=60,
        max_daily_orders=0,
        estimated_fee_bps=10.0,
        context_max_board_posts=10,
        context_max_memory_events=10,
        context_max_market_rows=10,
        openai_api_key="",
        openai_model="gpt-5.3",
        gemini_api_key="",
        gemini_model="gemini-3",
        research_gemini_model="gemini-2.5-flash",
        llm_timeout_seconds=20,
        default_universe=["005930", "000660"],
        allow_live_trading=False,
    )


def _snapshot() -> AccountSnapshot:
    """Builds a test snapshot with one current position."""
    return AccountSnapshot(
        cash_krw=400_000,
        total_equity_krw=1_000_000,
        positions={
            "005930": Position(
                ticker="005930",
                quantity=10,
                avg_price_krw=60_000,
                market_price_krw=70_000,
            )
        },
    )


def test_risk_allows_simple_buy() -> None:
    """Approves a buy intent that satisfies all policy constraints."""
    engine = RiskEngine(_settings())
    intent = OrderIntent(
        agent_id="a",
        ticker="000660",
        side=Side.BUY,
        quantity=1,
        price_krw=100_000,
        rationale="test",
    )
    decision = engine.evaluate(intent, _snapshot(), daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert decision.allowed is True


def test_risk_rejects_large_order() -> None:
    """Rejects order intent above max_order_krw."""
    engine = RiskEngine(_settings())
    intent = OrderIntent(
        agent_id="a",
        ticker="000660",
        side=Side.BUY,
        quantity=5,
        price_krw=100_000,
        rationale="test",
    )
    decision = engine.evaluate(intent, _snapshot(), daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert decision.allowed is False
    assert "max_order_krw" in decision.policy_hits


def test_risk_allows_single_share_above_max_order_within_limits() -> None:
    """Allows a 1-share buy above max_order_krw if cash and position ratio are valid."""
    engine = RiskEngine(_settings())
    snapshot = AccountSnapshot(
        cash_krw=700_000,
        total_equity_krw=1_000_000,
        positions={
            "005930": Position(
                ticker="005930",
                quantity=3,
                avg_price_krw=100_000,
                market_price_krw=100_000,
            )
        },
    )
    intent = OrderIntent(
        agent_id="a",
        ticker="000660",
        side=Side.BUY,
        quantity=1,
        price_krw=450_000,
        rationale="single share exception",
    )
    decision = engine.evaluate(intent, snapshot, daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert decision.allowed is True


def test_risk_rejects_cooldown() -> None:
    """Rejects order when same ticker cooldown is still active."""
    engine = RiskEngine(_settings())
    intent = OrderIntent(
        agent_id="a",
        ticker="005930",
        side=Side.SELL,
        quantity=1,
        price_krw=70_000,
        rationale="test",
    )
    decision = engine.evaluate(
        intent,
        _snapshot(),
        daily_turnover_krw=0,
        daily_order_count=0,
        last_trade_at=utc_now() - timedelta(seconds=5),
        now=utc_now(),
    )
    assert decision.allowed is False
    assert "ticker_cooldown_seconds" in decision.policy_hits


def test_risk_rejects_max_daily_orders() -> None:
    """Rejects intent when daily order cap is reached."""
    base = _settings()
    engine = RiskEngine(replace(base, max_daily_orders=2))
    intent = OrderIntent(
        agent_id="a",
        ticker="000660",
        side=Side.BUY,
        quantity=1,
        price_krw=100_000,
        rationale="test",
    )
    decision = engine.evaluate(intent, _snapshot(), daily_turnover_krw=0, daily_order_count=2, last_trade_at=None, now=utc_now())
    assert decision.allowed is False
    assert "max_daily_orders" in decision.policy_hits



def test_risk_rejects_ticker_market_mismatch() -> None:
    """Rejects intent when ticker format mismatches the configured market."""
    base = _settings()

    engine = RiskEngine(replace(base, kis_target_market="nasdaq"))
    intent = OrderIntent(agent_id="a", ticker="005930", side=Side.BUY, quantity=1, price_krw=70_000, rationale="test")
    decision = engine.evaluate(intent, _snapshot(), daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert decision.allowed is False
    assert "ticker_market_mismatch" in decision.policy_hits

    engine = RiskEngine(replace(base, kis_target_market="kospi"))
    intent = OrderIntent(agent_id="a", ticker="AAPL", side=Side.BUY, quantity=1, price_krw=100_000, rationale="test")
    decision = engine.evaluate(intent, _snapshot(), daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert decision.allowed is False
    assert "ticker_market_mismatch" in decision.policy_hits


def test_risk_handles_multi_market_ticker_validation() -> None:
    base = _settings()
    engine = RiskEngine(replace(base, kis_target_market="us,kospi"))

    us_intent = OrderIntent(agent_id="a", ticker="AAPL", side=Side.BUY, quantity=1, price_krw=100_000, rationale="test")
    us_decision = engine.evaluate(us_intent, _snapshot(), daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert us_decision.allowed is True

    kr_intent = OrderIntent(agent_id="a", ticker="000660", side=Side.BUY, quantity=1, price_krw=70_000, rationale="test")
    kr_decision = engine.evaluate(kr_intent, _snapshot(), daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert kr_decision.allowed is True

    bad_intent = OrderIntent(agent_id="a", ticker="???", side=Side.BUY, quantity=1, price_krw=70_000, rationale="test")
    bad_decision = engine.evaluate(bad_intent, _snapshot(), daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert bad_decision.allowed is False
    assert "ticker_market_mismatch" in bad_decision.policy_hits


def test_risk_allows_buy_when_sleeve_equity_above_target_if_cash_buffer_ok() -> None:
    """Allows buys above target sleeve equity when other risk checks pass."""
    base = _settings()
    engine = RiskEngine(replace(base, sleeve_capital_krw=500_000))
    intent = OrderIntent(
        agent_id="a",
        ticker="000660",
        side=Side.BUY,
        quantity=1,
        price_krw=50_000,
        rationale="no add when above target",
    )
    decision = engine.evaluate(intent, _snapshot(), daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert decision.allowed is True


def test_risk_allows_sell_when_sleeve_equity_above_target() -> None:
    """Allows reducing exposure via SELL when sleeve equity is above target."""
    base = _settings()
    engine = RiskEngine(replace(base, sleeve_capital_krw=500_000, max_order_krw=300_000))
    intent = OrderIntent(
        agent_id="a",
        ticker="005930",
        side=Side.SELL,
        quantity=3,
        price_krw=70_000,
        rationale="reduce exposure",
    )
    decision = engine.evaluate(intent, _snapshot(), daily_turnover_krw=0, daily_order_count=0, last_trade_at=None, now=utc_now())
    assert decision.allowed is True
