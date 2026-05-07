from __future__ import annotations

import pytest

from arena.agents.adk_agents import AdkTradingAgent, _agent_config_payload
from arena.config import AgentConfig, load_settings
from arena.models import BoardPost, ExecutionReport, ExecutionStatus, utc_now


class _RepoForAdkGenerate:
    def latest_market_features(self, tickers, limit, sources=None):
        _ = (tickers, limit, sources)
        return []


class _FakeRunner:
    def __init__(self) -> None:
        self.board_calls: list[tuple[str, str, str]] = []

    def decide_orders(self, *, context, default_universe, resume_session_id=None):
        _ = (context, default_universe, resume_session_id)
        return (
            {
                "orders": [
                    {
                        "ticker": "AAPL",
                        "side": "BUY",
                        "quantity": 1,
                        "rationale": "fx repricing",
                    }
                ],
            },
            "sid_1",
        )

    def decide_board(self, session_id, orders_summary, *, cycle_id=""):
        self.board_calls.append((session_id, orders_summary, cycle_id))
        return {"board_title": "confirmed", "board_body": orders_summary}


class _FakeKospiRunner(_FakeRunner):
    def decide_orders(self, *, context, default_universe, resume_session_id=None):
        _ = (context, default_universe, resume_session_id)
        return (
            {
                "orders": [
                    {
                        "ticker": "025860",
                        "side": "BUY",
                        "quantity": 48,
                        "rationale": "momentum continuation",
                    }
                ],
            },
            "sid_kospi_1",
        )

    def decide_board(self, session_id, orders_summary, *, cycle_id=""):
        self.board_calls.append((session_id, orders_summary, cycle_id))
        return {
            "board_title": "이녹스첨단소재를 다시 담다",
            "board_body": "**이녹스첨단소재(025860)** BUY 48주 체결\n전날 27주에 이어 오늘 48주.",
        }


class _FailRunner:
    def decide_orders(self, *, context, default_universe, resume_session_id=None):
        _ = (context, default_universe, resume_session_id)
        raise RuntimeError("runner boom")


def _settings_for_market(market: str, *, trading_mode: str = "paper", universe: list[str]) -> object:
    settings = load_settings()
    settings.trading_mode = trading_mode
    settings.kis_target_market = market
    settings.max_order_krw = 2_000_000.0
    settings.max_position_ratio = 1.0
    settings.default_universe = universe
    return settings


def _agent(monkeypatch: pytest.MonkeyPatch, runner, settings, *, agent_id: str = "gpt", provider: str = "gpt"):
    monkeypatch.setattr(AdkTradingAgent, "_build_runner", lambda self, *, settings: runner)
    return AdkTradingAgent(
        agent_id=agent_id,
        provider=provider,
        settings=settings,
        repo=_RepoForAdkGenerate(),
        registry=object(),
    )


def _us_execution_context(*, cycle_id: str, exchange_code: str = "NASD", instrument_id: str = "NASD:AAPL") -> dict:
    return {
        "cycle_phase": "execution",
        "cycle_id": cycle_id,
        "portfolio": {
            "cash_krw": 2_000_000.0,
            "total_equity_krw": 2_000_000.0,
            "usd_krw_rate": 1450.0,
            "positions": {},
        },
        "market_features": [
            {
                "ticker": "AAPL",
                "exchange_code": exchange_code,
                "instrument_id": instrument_id,
                "close_price_krw": 130000.0,
                "close_price_native": 100.0,
                "quote_currency": "USD",
                "fx_rate_used": 1300.0,
            }
        ],
        "order_budget": {"max_buy_notional_krw": 2_000_000.0},
    }


def test_generate_reprices_us_order_with_live_fx(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _FakeRunner()
    agent = _agent(
        monkeypatch,
        runner,
        _settings_for_market("us", universe=["AAPL"]),
    )

    out = agent.generate(_us_execution_context(cycle_id="cycle_fx_1"))

    assert len(out.intents) == 1
    intent = out.intents[0]
    assert intent.price_krw == pytest.approx(145000.0)
    assert intent.price_native == pytest.approx(100.0)
    assert intent.quote_currency == "USD"
    assert intent.fx_rate == pytest.approx(1450.0)
    assert runner.board_calls == []


def test_generate_raises_when_market_features_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(
        monkeypatch,
        _FakeRunner(),
        _settings_for_market("us", universe=["AAPL"]),
    )

    with pytest.raises(RuntimeError, match="market_features missing"):
        agent.generate({"cycle_phase": "execution", "cycle_id": "cycle_missing_rows", "market_features": []})


def test_generate_raises_when_decision_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(
        monkeypatch,
        _FailRunner(),
        _settings_for_market("us", universe=["AAPL"]),
    )

    with pytest.raises(RuntimeError, match="ADK decision failed"):
        agent.generate(_us_execution_context(cycle_id="cycle_decision_fail"))


def test_agent_config_payload_serializes_dataclass() -> None:
    payload = _agent_config_payload(
        AgentConfig(
            agent_id="claude",
            provider="claude",
            model="claude-sonnet-4-6",
            capital_krw=2_000_000.0,
            target_market="kospi",
            system_prompt="focus on risk",
            risk_overrides={"max_position_ratio": 0.2},
            disabled_tools=["trade_performance"],
        )
    )

    assert payload == {
        "agent_id": "claude",
        "provider": "claude",
        "model": "claude-sonnet-4-6",
        "capital_krw": 2_000_000.0,
        "target_market": "kospi",
        "system_prompt": "focus on risk",
        "risk_overrides": {"max_position_ratio": 0.2},
        "disabled_tools": ["trade_performance"],
        "llm_params": None,
        "memory_compaction_model": "",
    }


def test_finalize_board_post_uses_execution_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _FakeRunner()
    agent = _agent(
        monkeypatch,
        runner,
        _settings_for_market("us", universe=["AAPL"]),
    )
    out = agent.generate(_us_execution_context(cycle_id="cycle_fx_2"))

    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_1",
        filled_qty=1.0,
        avg_price_krw=145145.0,
        avg_price_native=100.1,
        quote_currency="USD",
        fx_rate=1450.0,
        message="confirmed",
        created_at=utc_now(),
    )

    post = agent.finalize_board_post(
        cycle_id="cycle_fx_2",
        initial_post=BoardPost(
            agent_id="gpt",
            title="placeholder",
            body="pending",
            tickers=["AAPL"],
            cycle_id="cycle_fx_2",
        ),
        intents=out.intents,
        reports=[report],
    )

    assert len(runner.board_calls) == 1
    _, summary, board_cycle_id = runner.board_calls[0]
    assert "실제 실행 결과" in summary
    assert "AAPL BUY 1주 FILLED" in summary
    assert board_cycle_id == "cycle_fx_2"
    assert post.body == summary


def test_finalize_board_post_keeps_freeform_board_text(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _FakeKospiRunner()
    agent = _agent(
        monkeypatch,
        runner,
        _settings_for_market("kospi", universe=["025860"]),
        agent_id="claude",
        provider="claude",
    )

    out = agent.generate(
        {
            "cycle_phase": "execution",
            "cycle_id": "cycle_kospi_1",
            "portfolio": {
                "cash_krw": 2_000_000.0,
                "total_equity_krw": 2_500_000.0,
                "positions": {
                    "025860": {
                        "quantity": 27.0,
                        "avg_price_krw": 8290.0,
                        "market_price_krw": 8270.0,
                        "ticker_name": "남해화학",
                    }
                },
            },
            "market_features": [
                {
                    "ticker": "025860",
                    "exchange_code": "KRX",
                    "instrument_id": "KRX:025860",
                    "close_price_krw": 8270.0,
                    "close_price_native": 8270.0,
                    "quote_currency": "KRW",
                    "fx_rate_used": 1.0,
                }
            ],
            "order_budget": {"max_buy_notional_krw": 2_000_000.0},
        }
    )

    report = ExecutionReport(
        status=ExecutionStatus.FILLED,
        order_id="ord_kospi_1",
        filled_qty=48.0,
        avg_price_krw=8290.0,
        quote_currency="KRW",
        fx_rate=1.0,
        message="confirmed",
        created_at=utc_now(),
    )

    post = agent.finalize_board_post(
        cycle_id="cycle_kospi_1",
        initial_post=out.board_post,
        intents=out.intents,
        reports=[report],
    )

    assert len(runner.board_calls) == 1
    assert runner.board_calls[0][2] == "cycle_kospi_1"
    assert post.title == "이녹스첨단소재를 다시 담다"
    assert "**이녹스첨단소재(025860)** BUY 48주 체결" in post.body
    assert "전날 27주에 이어 오늘 48주." in post.body


def test_generate_skips_mixed_us_order_when_exchange_is_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(
        monkeypatch,
        _FakeRunner(),
        _settings_for_market("us", trading_mode="live", universe=["AAPL"]),
    )

    out = agent.generate(
        _us_execution_context(
            cycle_id="cycle_fx_2",
            exchange_code="",
            instrument_id="",
        )
    )

    assert out.intents == []
