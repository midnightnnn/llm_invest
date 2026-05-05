from __future__ import annotations

from arena.agents.adk_order_support import build_order_intents
from arena.config import load_settings
from tests.adk.order_support_helpers import _RepoForAdkGenerate


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
