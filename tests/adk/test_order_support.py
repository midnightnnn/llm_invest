from __future__ import annotations

import pytest

from arena.agents.adk_order_support import (
    build_order_intents,
    fetch_market_row_from_bq,
    format_execution_summary,
    format_orders_summary,
    resolve_order_price,
)
from arena.config import load_settings
from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, Side


class _RepoForMarketLookup:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[dict[str, object]] = []

    def latest_market_features(self, *, tickers, limit, sources=None):
        self.calls.append(
            {
                "tickers": list(tickers),
                "limit": limit,
                "sources": list(sources) if isinstance(sources, list) else sources,
            }
        )
        return list(self.rows)


class _RepoForAdkGenerate:
    def latest_market_features(self, tickers, limit, sources=None):
        _ = (tickers, limit, sources)
        return []


def test_fetch_market_row_from_bq_uses_live_kospi_sources() -> None:
    settings = load_settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "kospi"
    repo = _RepoForMarketLookup(
        [
            {"ticker": "005930", "close_price_krw": 70500.0, "close_price_native": 70500.0},
        ]
    )

    row = fetch_market_row_from_bq(repo, settings, "005930")

    assert row is not None
    assert row["ticker"] == "005930"
    assert repo.calls[0]["sources"] == ["open_trading_kospi_quote", "open_trading_kospi"]


def test_resolve_order_price_prefers_live_fx_for_us_quotes() -> None:
    settings = load_settings()
    settings.kis_target_market = "nasdaq"
    settings.usd_krw_rate = 1300.0

    price_krw, native_price, quote_currency, fx_rate = resolve_order_price(
        settings,
        market_row={
            "close_price_native": 10.0,
            "fx_rate_used": 1250.0,
        },
        portfolio={"usd_krw_rate": 1400.0},
    )

    assert price_krw == pytest.approx(14000.0)
    assert native_price == pytest.approx(10.0)
    assert quote_currency == "USD"
    assert fx_rate == pytest.approx(1400.0)


def test_resolve_order_price_returns_zero_when_us_fx_is_missing() -> None:
    settings = load_settings()
    settings.kis_target_market = "nasdaq"
    settings.usd_krw_rate = 1300.0

    price_krw, native_price, quote_currency, fx_rate = resolve_order_price(
        settings,
        market_row={
            "close_price_native": 10.0,
            "fx_rate_used": 0.0,
        },
        portfolio={"usd_krw_rate": 0.0},
    )

    assert price_krw == pytest.approx(0.0)
    assert native_price == pytest.approx(10.0)
    assert quote_currency == "USD"
    assert fx_rate == pytest.approx(0.0)


def test_format_orders_summary_includes_hold_rows() -> None:
    summary = format_orders_summary(
        [
            OrderIntent(
                agent_id="gpt",
                ticker="AAPL",
                side=Side.BUY,
                quantity=3.0,
                price_krw=10000.0,
                rationale="Breakout continuation with supportive breadth.",
            )
        ],
        [{"ticker": "MSFT", "side": "HOLD", "rationale": "No edge after recent gap."}],
    )

    assert "AAPL BUY 3.0주" in summary
    assert "MSFT HOLD" in summary


def test_format_orders_summary_uses_known_kospi_ticker_name() -> None:
    summary = format_orders_summary(
        [
            OrderIntent(
                agent_id="claude",
                ticker="025860",
                side=Side.BUY,
                quantity=3.0,
                price_krw=8270.0,
                rationale="실적 회복 기대.",
            )
        ],
        [],
        ticker_names={"025860": "남해화학"},
    )

    assert "남해화학(025860) BUY 3.0주" in summary


def test_format_execution_summary_includes_error_reference_price() -> None:
    intent = OrderIntent(
        agent_id="claude",
        ticker="001510",
        side=Side.SELL,
        quantity=10.0,
        price_krw=1880.0,
        rationale="거래정지 리스크 축소.",
    )
    report = ExecutionReport(
        status=ExecutionStatus.ERROR,
        order_id="err_1",
        filled_qty=0.0,
        avg_price_krw=1861.0,
        avg_price_native=1861.0,
        quote_currency="KRW",
        fx_rate=1.0,
        message="거래정지종목(주식)은 취소주문만 가능(정정불가)합니다.",
    )

    summary = format_execution_summary([intent], [report], ticker_names={"001510": "SK증권"})

    assert "SK증권(001510) SELL 10주 ERROR 시도호가 @₩1,861" in summary
    assert "거래정지종목" in summary


def test_build_order_intents_defaults_single_market_us_exchange() -> None:
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "nasdaq"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0

    intents, tickers_mentioned = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_1",
        context={
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "positions": {},
            },
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        },
        orders=[
            {
                "ticker": "AAPL",
                "side": "BUY",
                "target_weight": 0.5,
                "rationale": "single-market default exchange",
            }
        ],
        row_map={
            "AAPL": {
                "ticker": "AAPL",
                "exchange_code": "",
                "instrument_id": "",
                "close_price_krw": 130000.0,
                "close_price_native": 100.0,
                "quote_currency": "USD",
                "fx_rate_used": 1300.0,
            }
        },
    )

    assert tickers_mentioned == {"AAPL"}
    assert len(intents) == 1
    assert intents[0].exchange_code == "NASD"
    assert intents[0].instrument_id == "NASD:AAPL"


def test_build_order_intents_preserves_full_rationale_text() -> None:
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "nasdaq"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    long_rationale = (
        "AAPL의 서비스 매출 성장과 잉여현금흐름 안정성이 AAPL 매수 thesis를 지지한다. "
        + "sleeve context를 감안해 목표 비중을 유지 가능한 범위에서 올린다. " * 35
        + "TAIL_MARKER"
    )

    intents, _ = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_long_rationale",
        context={
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "positions": {},
            },
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        },
        orders=[
            {
                "ticker": "AAPL",
                "side": "BUY",
                "target_weight": 0.5,
                "rationale": long_rationale,
            }
        ],
        row_map={
            "AAPL": {
                "ticker": "AAPL",
                "exchange_code": "",
                "instrument_id": "",
                "close_price_krw": 130000.0,
                "close_price_native": 100.0,
                "quote_currency": "USD",
                "fx_rate_used": 1300.0,
            }
        },
    )

    assert len(intents) == 1
    assert intents[0].rationale == long_rationale
    assert intents[0].rationale.endswith("TAIL_MARKER")


def test_build_order_intents_buy_target_weight_only_adds_shortfall() -> None:
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "nasdaq"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0

    intents, _ = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_target_weight",
        context={
            "portfolio": {
                "cash_krw": 1_480_000.0,
                "total_equity_krw": 2_000_000.0,
                "positions": {"AAPL": {"quantity": 4.0}},
            },
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        },
        orders=[
            {
                "ticker": "AAPL",
                "side": "BUY",
                "target_weight": 0.5,
                "rationale": "raise to target weight",
            }
        ],
        row_map={
            "AAPL": {
                "ticker": "AAPL",
                "exchange_code": "NASD",
                "instrument_id": "NASD:AAPL",
                "close_price_krw": 130000.0,
                "close_price_native": 100.0,
                "quote_currency": "USD",
                "fx_rate_used": 1300.0,
            }
        },
    )

    assert len(intents) == 1
    assert intents[0].quantity == 3.6923


def test_resolve_order_price_multi_market_infers_usd_from_exchange_identity() -> None:
    settings = load_settings()
    settings.kis_target_market = "us,kospi"

    price_krw, native_price, quote_currency, fx_rate = resolve_order_price(
        settings,
        market_row={
            "ticker": "AAPL",
            "exchange_code": "NAS",
            "instrument_id": "NASD:AAPL",
            "close_price_native": 100.0,
            "fx_rate_used": 1300.0,
        },
        portfolio={},
    )

    assert price_krw == 130000.0
    assert native_price == 100.0
    assert quote_currency == "USD"
    assert fx_rate == 1300.0


def test_build_order_intents_multi_market_defaults_korean_exchange() -> None:
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us,kospi"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0

    intents, tickers_mentioned = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_combo_kr",
        context={
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "positions": {},
            },
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        },
        orders=[
            {
                "ticker": "005930",
                "side": "BUY",
                "target_weight": 0.5,
                "rationale": "combo-market KRX inference",
            }
        ],
        row_map={
            "005930": {
                "ticker": "005930",
                "exchange_code": "",
                "instrument_id": "",
                "close_price_krw": 70000.0,
                "quote_currency": "KRW",
            }
        },
    )

    assert tickers_mentioned == {"005930"}
    assert len(intents) == 1
    assert intents[0].exchange_code == "KRX"
    assert intents[0].instrument_id == "KRX:005930"


def test_build_order_intents_collects_feedback_events() -> None:
    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "nasdaq"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    feedback_events: list[dict[str, object]] = []

    intents, tickers_mentioned = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_feedback",
        context={
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_000_000.0,
                "positions": {},
            },
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        },
        orders=[
            {
                "ticker": "AAPL",
                "side": "BUY",
                "target_weight": 0.5,
                "rationale": "build intent",
            },
            {
                "ticker": "TSLA",
                "side": "BUY",
                "target_weight": 0.3,
                "rationale": "missing price",
            },
        ],
        row_map={
            "AAPL": {
                "ticker": "AAPL",
                "exchange_code": "",
                "instrument_id": "",
                "close_price_krw": 130000.0,
                "close_price_native": 100.0,
                "quote_currency": "USD",
                "fx_rate_used": 1300.0,
            }
        },
        feedback_events=feedback_events,
    )

    assert tickers_mentioned == {"AAPL", "TSLA"}
    assert len(intents) == 1
    assert feedback_events == [
        {"ticker": "AAPL", "side": "BUY", "status": "intent_built"},
        {"ticker": "TSLA", "side": "BUY", "status": "skipped", "reason": "no_price"},
    ]


def test_build_order_intents_live_sell_rounds_up_small_position() -> None:
    settings = load_settings()
    settings.trading_mode = "live"
    settings.kis_target_market = "kospi"
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0

    intents, tickers_mentioned = build_order_intents(
        repo=_RepoForAdkGenerate(),
        settings=settings,
        agent_id="gpt",
        sleeve_capital_krw=2_000_000.0,
        cycle_id="cycle_order_2",
        context={
            "portfolio": {
                "cash_krw": 100000.0,
                "total_equity_krw": 100000.0,
                "positions": {
                    "005930": {
                        "quantity": 1.0,
                        "avg_price_krw": 70000.0,
                    }
                },
            },
            "order_budget": {},
        },
        orders=[
            {
                "ticker": "005930",
                "side": "SELL",
                "sell_ratio": 0.1,
                "rationale": "small live trim",
            }
        ],
        row_map={
            "005930": {
                "ticker": "005930",
                "exchange_code": "",
                "instrument_id": "",
                "close_price_krw": 70000.0,
                "quote_currency": "KRW",
            }
        },
    )

    assert tickers_mentioned == {"005930"}
    assert len(intents) == 1
    assert intents[0].quantity == 1.0
    assert intents[0].exchange_code == "KRX"
    assert intents[0].instrument_id == "KRX:005930"
