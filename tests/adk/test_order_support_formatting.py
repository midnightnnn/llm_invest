from __future__ import annotations

from arena.agents.adk_order_support import format_execution_summary, format_orders_summary
from arena.models import ExecutionReport, ExecutionStatus, OrderIntent, Side


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
