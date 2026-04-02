from __future__ import annotations

import logging

import pytest

from arena.broker.open_trading import KISOpenTradingBroker, _normalize_us_order_exchange
from arena.config import Settings
from arena.models import OrderIntent, Side


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
        kis_http_max_retries=0,
        kis_http_backoff_base_seconds=0.1,
        kis_http_backoff_max_seconds=0.1,
        kis_confirm_fills=False,
        kis_confirm_timeout_seconds=10,
        kis_confirm_poll_seconds=0.5,
        usd_krw_rate=1300.0,
        market_sync_history_days=60,
        max_order_krw=500_000,
        max_daily_turnover_ratio=0.65,
        max_position_ratio=0.35,
        min_cash_buffer_ratio=0.1,
        ticker_cooldown_seconds=120,
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
        llm_timeout_seconds=25,
        default_universe=["AAPL"],
        allow_live_trading=True,
        live_slippage_bps_base=10.0,
        live_slippage_bps_impact=0.0,
        live_slippage_bps_max=50.0,
    )


def test_to_order_payload_applies_buy_slippage() -> None:
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.2,
        price_krw=130_000,
        rationale="test",
        fx_rate=1300.0,
    )

    market, qty, local_limit, limit_krw, bps, order_exchange, fx_rate = broker._to_order_payload(intent)

    assert market == "us"
    assert qty == 1
    assert bps == 10.0
    assert order_exchange == "NASD"
    assert fx_rate == 1300.0
    assert local_limit == 100.1
    assert limit_krw == pytest.approx(130130.0)


def test_to_order_payload_prefers_explicit_fx_rate() -> None:
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=145_000,
        rationale="test",
        fx_rate=1300.0,
    )

    market, qty, local_limit, limit_krw, bps, order_exchange, fx_rate = broker._to_order_payload(
        intent,
        fx_rate=1450.0,
    )

    assert market == "us"
    assert qty == 1
    assert bps == 10.0
    assert order_exchange == "NASD"
    assert fx_rate == 1450.0
    assert local_limit == pytest.approx(100.1)
    assert limit_krw == pytest.approx(145145.0)


def test_to_order_payload_requires_exchange_for_mixed_us_market() -> None:
    settings = _settings()
    settings.kis_target_market = "us"
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=145_000,
        rationale="test",
        fx_rate=1300.0,
    )

    with pytest.raises(ValueError, match="unable to resolve US order exchange"):
        broker._to_order_payload(intent)


def test_to_order_payload_accepts_combo_market_with_us_exchange() -> None:
    settings = _settings()
    settings.kis_target_market = "us,kospi"
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=145_000,
        rationale="test",
        fx_rate=1300.0,
        exchange_code="NASD",
    )

    market, qty, _local_limit, _limit_krw, _bps, order_exchange, fx_rate = broker._to_order_payload(intent)

    assert market == "us"
    assert qty == 1
    assert order_exchange == "NASD"
    assert fx_rate == 1300.0


def test_to_order_payload_accepts_combo_market_with_krx_exchange() -> None:
    settings = _settings()
    settings.kis_target_market = "us,kospi"
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="005930",
        side=Side.BUY,
        quantity=1.0,
        price_krw=70_000,
        rationale="test",
        exchange_code="KRX",
        quote_currency="KRW",
    )

    market, qty, local_limit, limit_krw, _bps, order_exchange, fx_rate = broker._to_order_payload(intent)

    assert market == "kospi"
    assert qty == 1
    assert local_limit == 70100.0
    assert limit_krw == 70100.0
    assert order_exchange == "KRX"
    assert fx_rate == 1.0


def test_live_slippage_bps_is_capped() -> None:
    s = _settings()
    s.live_slippage_bps_base = 5.0
    s.live_slippage_bps_impact = 20.0
    s.live_slippage_bps_max = 30.0
    broker = KISOpenTradingBroker(settings=s)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=100.0,
        price_krw=130_000,
        rationale="test",
    )

    assert broker._live_slippage_bps(intent) == 30.0


def test_query_fill_once_converts_nasdaq_price_to_krw(monkeypatch) -> None:
    broker = KISOpenTradingBroker(settings=_settings())

    def _fake_inquire(**kwargs):
        _ = kwargs
        return [{"ODNO": "123", "CCLD_QTY": "2", "CCLD_UNPR": "50.5"}]

    monkeypatch.setattr(broker.client, "inquire_overseas_ccnl", _fake_inquire)

    report = broker._query_fill_once(
        market="us",
        order_id="123",
        ticker="AAPL",
        qty=2,
        fallback_price_krw=100_000,
        message="confirmed",
        fx_rate=1300.0,
    )

    assert report is not None
    assert report.filled_qty == 2
    assert report.avg_price_krw == 50.5 * 1300.0


def test_query_fill_once_uses_explicit_fx_rate(monkeypatch) -> None:
    broker = KISOpenTradingBroker(settings=_settings())

    def _fake_inquire(**kwargs):
        _ = kwargs
        return [{"ODNO": "123", "CCLD_QTY": "2", "CCLD_UNPR": "50.5"}]

    monkeypatch.setattr(broker.client, "inquire_overseas_ccnl", _fake_inquire)

    report = broker._query_fill_once(
        market="us",
        order_id="123",
        ticker="AAPL",
        qty=2,
        fallback_price_krw=100_000,
        message="confirmed",
        fx_rate=1450.0,
    )

    assert report is not None
    assert report.avg_price_krw == pytest.approx(50.5 * 1450.0)
    assert report.fx_rate == pytest.approx(1450.0)


def test_query_fill_once_scans_multiple_us_exchanges(monkeypatch) -> None:
    broker = KISOpenTradingBroker(settings=_settings())
    calls: list[str] = []

    def _fake_inquire(**kwargs):
        exchange = str(kwargs.get("exchange_code") or "")
        calls.append(exchange)
        if exchange == "NYSE":
            return [{"ODNO": "777", "CCLD_QTY": "1", "CCLD_UNPR": "20.0"}]
        return []

    monkeypatch.setattr(broker.client, "inquire_overseas_ccnl", _fake_inquire)

    report = broker._query_fill_once(
        market="us",
        order_id="777",
        ticker="EXC",
        qty=1,
        fallback_price_krw=100_000,
        message="confirmed",
        exchange_code="NASD",
        fx_rate=1300.0,
    )

    assert report is not None
    assert report.status.value == "FILLED"
    assert report.filled_qty == 1
    assert "NASD" in calls
    assert "NYSE" in calls


def test_resolved_fx_rate_rejects_missing_rate() -> None:
    """Orders must not proceed with a stale config default — ValueError expected."""
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=130_000,
        rationale="test",
    )
    with pytest.raises(ValueError, match="No live USD/KRW rate"):
        broker._resolved_fx_rate(intent)


def test_resolved_fx_rate_uses_api_value_even_below_config() -> None:
    """API FX rate 1455 should be used even when config default is 1460."""
    settings = _settings()
    settings.usd_krw_rate = 1460.0
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=130_000,
        rationale="test",
        fx_rate=1455.0,
    )
    assert broker._resolved_fx_rate(intent) == 1455.0


def test_reconcile_submitted_uses_fill_lookup(monkeypatch) -> None:
    broker = KISOpenTradingBroker(settings=_settings())

    def _fake_query_fill_once(**kwargs):
        _ = kwargs
        from arena.models import ExecutionReport, ExecutionStatus, utc_now

        return ExecutionReport(
            status=ExecutionStatus.FILLED,
            order_id="abc",
            filled_qty=3,
            avg_price_krw=99_000,
            message="reconciled",
            created_at=utc_now(),
        )

    monkeypatch.setattr(broker, "_query_fill_once", _fake_query_fill_once)

    report = broker.reconcile_submitted(
        order_id="abc",
        ticker="AAPL",
        side="BUY",
        requested_qty=3.9,
        fallback_price_krw=98_000,
    )

    assert report is not None
    assert report.status.value == "FILLED"


def test_normalize_us_order_exchange_requires_resolvable_code() -> None:
    with pytest.raises(ValueError, match="unable to resolve US order exchange"):
        _normalize_us_order_exchange("UNKNOWN", "")


def test_place_order_adjusts_limit_to_live_price(monkeypatch) -> None:
    """When live ask > stale limit, place_order re-anchors limit to live price + slippage."""
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="GILD",
        side=Side.BUY,
        quantity=2.0,
        price_krw=130_000,
        rationale="test",
        fx_rate=1300.0,
        exchange_code="NASD",
    )

    # Stale limit from _to_order_payload: 130_000 * (1 + 10/10000) / 1300 = 100.10 USD
    # Live price is 102.50 USD — higher than stale limit
    monkeypatch.setattr(
        broker.client, "get_overseas_price",
        lambda **kw: {"last": "102.50"},
    )

    captured: dict = {}

    def _fake_place(**kwargs):
        captured.update(kwargs)
        return {"output": {"ODNO": "test123"}, "msg1": "ok"}

    monkeypatch.setattr(broker.client, "place_overseas_order", _fake_place)

    report = broker.place_order(intent)

    submitted_limit = captured["limit_price"]
    # Should be ≥ live price (102.50), not stale limit (100.10)
    assert submitted_limit >= 102.50
    # Should include slippage buffer on top of live price
    expected = 102.50 * (1.0 + 10.0 / 10_000.0)  # ≈ 102.6025 → rounded to 102.61
    assert submitted_limit == pytest.approx(expected, abs=0.01)
    assert report.order_id == "test123"


def test_place_order_keeps_limit_when_live_lower(monkeypatch) -> None:
    """When live price < stale limit for BUY, no adjustment needed."""
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=130_000,
        rationale="test",
        fx_rate=1300.0,
        exchange_code="NASD",
    )

    # Live price 99.00 is below stale limit 100.10 — no adjustment needed
    monkeypatch.setattr(
        broker.client, "get_overseas_price",
        lambda **kw: {"last": "99.00"},
    )

    captured: dict = {}

    def _fake_place(**kwargs):
        captured.update(kwargs)
        return {"output": {"ODNO": "test456"}, "msg1": "ok"}

    monkeypatch.setattr(broker.client, "place_overseas_order", _fake_place)

    broker.place_order(intent)

    # Should remain at stale limit (100.10), not adjusted down
    assert captured["limit_price"] == pytest.approx(100.10, abs=0.01)


def test_place_order_tolerates_live_price_failure(monkeypatch) -> None:
    """When live quote API fails, order proceeds with stale limit."""
    broker = KISOpenTradingBroker(settings=_settings())
    intent = OrderIntent(
        agent_id="gpt",
        ticker="AAPL",
        side=Side.BUY,
        quantity=1.0,
        price_krw=130_000,
        rationale="test",
        fx_rate=1300.0,
        exchange_code="NASD",
    )

    def _fail(**kw):
        raise ConnectionError("API down")

    monkeypatch.setattr(broker.client, "get_overseas_price", _fail)

    captured: dict = {}

    def _fake_place(**kwargs):
        captured.update(kwargs)
        return {"output": {"ODNO": "test789"}, "msg1": "ok"}

    monkeypatch.setattr(broker.client, "place_overseas_order", _fake_place)

    report = broker.place_order(intent)

    # Should fall back to stale limit
    assert captured["limit_price"] == pytest.approx(100.10, abs=0.01)
    assert report.order_id == "test789"


def test_to_order_payload_rounds_kospi_buy_to_tick() -> None:
    settings = _settings()
    settings.kis_target_market = "kospi"
    settings.live_slippage_bps_base = 8.0
    settings.live_slippage_bps_impact = 12.0
    settings.live_slippage_bps_max = 80.0
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="claude",
        ticker="001510",
        side=Side.BUY,
        quantity=94.0,
        price_krw=2375.0,
        rationale="test",
        fx_rate=1.0,
    )

    market, qty, local_limit, limit_krw, *_rest = broker._to_order_payload(intent)

    assert market == "kospi"
    assert qty == 94
    assert local_limit == 2380.0
    assert limit_krw == 2380.0


def test_place_order_adjusts_kospi_limit_to_live_tick_and_logs_tenant(monkeypatch, caplog) -> None:
    settings = _settings()
    settings.kis_target_market = "kospi"
    settings.live_slippage_bps_base = 8.0
    settings.live_slippage_bps_impact = 12.0
    settings.live_slippage_bps_max = 80.0
    broker = KISOpenTradingBroker(settings=settings)
    intent = OrderIntent(
        agent_id="claude",
        ticker="001510",
        side=Side.BUY,
        quantity=94.0,
        price_krw=2375.0,
        rationale="test",
        fx_rate=1.0,
    )

    monkeypatch.setenv("ARENA_TENANT_ID", "cxznms")
    monkeypatch.setattr(
        broker.client,
        "get_domestic_price",
        lambda **kw: {"stck_prpr": "2390"},
    )

    captured: dict = {}

    def _fake_place(**kwargs):
        captured.update(kwargs)
        return {"output": {"ODNO": "krx123"}, "msg1": "ok"}

    monkeypatch.setattr(broker.client, "place_domestic_cash_order", _fake_place)

    with caplog.at_level(logging.INFO):
        report = broker.place_order(intent)

    assert captured["limit_price"] == 2395.0
    assert "tenant=cxznms" in caplog.text
    assert report.order_id == "krx123"
