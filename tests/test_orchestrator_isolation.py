from __future__ import annotations

from arena.agents.base import AgentOutput
from arena.config import Settings
from arena.models import AccountSnapshot, BoardPost, ExecutionReport, ExecutionStatus, OrderIntent, Side, utc_now
from arena.orchestrator import ArenaOrchestrator


class FakeRepo:
    def __init__(self):
        self.ensure_calls: list[dict] = []
        self.reinit_calls: list[dict] = []
        self.build_calls: list[dict] = []

    def ensure_agent_sleeves(self, *, agent_ids, total_cash_krw, capital_per_agent=None, initialized_at=None):
        _ = initialized_at
        self.ensure_calls.append(
            {
                "agent_ids": list(agent_ids),
                "total_cash_krw": float(total_cash_krw),
                "capital_per_agent": dict(capital_per_agent) if capital_per_agent else None,
            }
        )
        return {str(a): {} for a in agent_ids}

    def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True):
        self.build_calls.append(
            {
                "agent_id": agent_id,
                "sources": sources,
                "include_simulated": bool(include_simulated),
            }
        )
        return AccountSnapshot(cash_krw=1_000_000.0, total_equity_krw=1_000_000.0, positions={}), 1_000_000.0, {}

    def reinitialize_agent_sleeves(self, *, agent_ids, total_cash_krw, initialized_at=None):
        _ = initialized_at
        self.reinit_calls.append(
            {
                "agent_ids": list(agent_ids),
                "total_cash_krw": float(total_cash_krw),
            }
        )
        return {str(a): {} for a in agent_ids}

    def upsert_agent_nav_daily(self, *, nav_date, agent_id, nav_krw, baseline_equity_krw, **kwargs):
        _ = (nav_date, agent_id, nav_krw, baseline_equity_krw, kwargs)


class FakeGateway:
    def __init__(self, repo):
        self.repo = repo

    def process(self, intent, snapshot):
        raise AssertionError("No intents expected in this test")


class FakeContextBuilder:
    def build(self, agent_id, snapshot, sleeve_baseline_equity_krw=None, sleeve_meta=None, **kwargs):
        _ = (agent_id, snapshot, sleeve_baseline_equity_krw, sleeve_meta)
        return {
            "portfolio": {},
            "market_features": [],
            "memory_events": [],
            "board_posts": [],
        }


class FakeBoardStore:
    def __init__(self):
        self.published: list[BoardPost] = []
        self.events: list[str] = []

    def recent(self, limit):
        _ = limit
        return [
            {
                "agent_id": "system",
                "title": "real-account-daily-report",
                "body": "secret",
                "tickers": ["PLTD"],
            }
        ]

    def publish(self, post):
        self.events.append("publish")
        self.published.append(post)


class DummyAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.seen_posts: list[list[dict]] = []

    def generate(self, context: dict) -> AgentOutput:
        posts = list(context.get("board_posts") or [])
        self.seen_posts.append(posts)
        return AgentOutput(
            intents=[],
            board_post=BoardPost(agent_id=self.agent_id, title=f"{self.agent_id}-note", body="ok", tickers=[]),
        )


class _ExecGateway:
    def __init__(self, repo, events: list[str]):
        self.repo = repo
        self.events = events

    def process(self, intent, snapshot):
        _ = snapshot
        self.events.append(f"process:{intent.ticker}")
        return ExecutionReport(
            status=ExecutionStatus.FILLED,
            order_id="ord_1",
            filled_qty=float(intent.quantity),
            avg_price_krw=float(intent.price_krw),
            avg_price_native=None,
            quote_currency=intent.quote_currency,
            fx_rate=float(intent.fx_rate),
            message="confirmed",
            created_at=utc_now(),
        )


class _ExecAgent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.finalize_calls = 0

    def generate(self, context: dict) -> AgentOutput:
        _ = context
        intent = OrderIntent(
            agent_id=self.agent_id,
            ticker="AAPL",
            side=Side.BUY,
            quantity=1.0,
            price_krw=100_000.0,
            rationale="test",
            quote_currency="KRW",
            fx_rate=1.0,
        )
        return AgentOutput(
            intents=[intent],
            board_post=BoardPost(agent_id=self.agent_id, title="placeholder", body="pending", tickers=["AAPL"]),
        )

    def finalize_board_post(self, *, cycle_id: str, initial_post: BoardPost, intents, reports):
        self.finalize_calls += 1
        assert cycle_id
        assert initial_post.body == "pending"
        assert len(intents) == 1
        assert len(reports) == 1
        return BoardPost(
            agent_id=self.agent_id,
            title="final",
            body=f"{reports[0].status.value}:{intents[0].ticker}",
            tickers=["AAPL"],
            cycle_id=cycle_id,
        )


def _settings(*, trading_mode: str = "live") -> Settings:
    return Settings(
        google_cloud_project="p",
        bq_dataset="d",
        bq_location="loc",
        agent_ids=["gpt", "gemini"],
        agent_mode="adk",
        base_currency="KRW",
        sleeve_capital_krw=1_000_000,
        log_level="INFO",
        log_format="rich",
        trading_mode=trading_mode,
        kis_order_endpoint="",
        kis_api_key="",
        kis_api_secret="",
        kis_paper_api_key="",
        kis_paper_api_secret="",
        kis_account_no="",
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
        default_universe=["AAPL", "MSFT"],
        allow_live_trading=True,
    )


def test_orchestrator_uses_virtual_sleeve_cash_not_live_account_cash() -> None:
    repo = FakeRepo()
    agents = [DummyAgent("gpt"), DummyAgent("gemini")]
    orchestrator = ArenaOrchestrator(
        settings=_settings(),
        context_builder=FakeContextBuilder(),
        board_store=FakeBoardStore(),
        gateway=FakeGateway(repo),
        agents=agents,
    )

    live_snapshot = AccountSnapshot(cash_krw=999_999_999.0, total_equity_krw=1_234_567_890.0, positions={})
    reports = orchestrator.run_cycle(snapshot=live_snapshot)

    assert reports == []
    assert repo.ensure_calls
    assert repo.ensure_calls[0]["total_cash_krw"] == 2_000_000.0
    assert repo.build_calls
    assert all(call["include_simulated"] is False for call in repo.build_calls)


def test_orchestrator_filters_system_posts_from_agent_context() -> None:
    repo = FakeRepo()
    agents = [DummyAgent("gpt"), DummyAgent("gemini")]
    orchestrator = ArenaOrchestrator(
        settings=_settings(),
        context_builder=FakeContextBuilder(),
        board_store=FakeBoardStore(),
        gateway=FakeGateway(repo),
        agents=agents,
    )

    orchestrator.run_cycle(snapshot=None)

    for agent in agents:
        assert agent.seen_posts
        for call_posts in agent.seen_posts:
            assert all(str(p.get("agent_id", "")).strip() != "system" for p in call_posts)


def test_orchestrator_can_force_reinitialize_sleeves(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_FORCE_SLEEVE_REINIT", "true")
    repo = FakeRepo()
    agents = [DummyAgent("gpt"), DummyAgent("gemini")]
    orchestrator = ArenaOrchestrator(
        settings=_settings(),
        context_builder=FakeContextBuilder(),
        board_store=FakeBoardStore(),
        gateway=FakeGateway(repo),
        agents=agents,
    )

    orchestrator.run_cycle(snapshot=None)

    assert repo.reinit_calls
    assert repo.reinit_calls[0]["total_cash_krw"] == 2_000_000.0
    assert repo.ensure_calls == []


def test_orchestrator_in_paper_mode_includes_simulated_history() -> None:
    repo = FakeRepo()
    agents = [DummyAgent("gpt")]
    orchestrator = ArenaOrchestrator(
        settings=_settings(trading_mode="paper"),
        context_builder=FakeContextBuilder(),
        board_store=FakeBoardStore(),
        gateway=FakeGateway(repo),
        agents=agents,
    )

    orchestrator.run_cycle(snapshot=None)

    assert repo.build_calls
    assert all(call["include_simulated"] is True for call in repo.build_calls)


def test_orchestrator_publishes_board_after_execution() -> None:
    repo = FakeRepo()
    board = FakeBoardStore()
    agent = _ExecAgent("gpt")
    gateway = _ExecGateway(repo, board.events)
    orchestrator = ArenaOrchestrator(
        settings=_settings(trading_mode="paper"),
        context_builder=FakeContextBuilder(),
        board_store=board,
        gateway=gateway,
        agents=[agent],
    )

    orchestrator.run_cycle(snapshot=None)

    assert board.events == ["process:AAPL", "publish"]
    assert agent.finalize_calls == 1
    assert board.published
    assert board.published[0].body == "FILLED:AAPL"
