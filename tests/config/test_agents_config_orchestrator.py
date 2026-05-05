from __future__ import annotations

from arena.config import load_settings
from arena.models import AccountSnapshot


def test_orchestrator_per_agent_capital_total_cash() -> None:
    from arena.agents.base import AgentOutput
    from arena.config import Settings
    from arena.models import BoardPost
    from arena.orchestrator import ArenaOrchestrator

    class _FakeRepo:
        def __init__(self):
            self.ensure_calls = []

        def ensure_agent_sleeves(self, *, agent_ids, total_cash_krw, capital_per_agent=None, initialized_at=None):
            _ = initialized_at
            self.ensure_calls.append({
                "agent_ids": list(agent_ids),
                "total_cash_krw": float(total_cash_krw),
                "capital_per_agent": dict(capital_per_agent) if capital_per_agent else None,
            })
            return {a: {} for a in agent_ids}

        def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True):
            return AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_000_000, positions={}), 1_000_000, {}

        def upsert_agent_nav_daily(self, **kwargs):
            pass

    class _FakeGateway:
        def __init__(self, repo):
            self.repo = repo

    class _FakeCtx:
        def build(self, agent_id, snapshot, sleeve_baseline_equity_krw=None, sleeve_meta=None):
            return {"portfolio": {}, "market_features": [], "memory_events": [], "board_posts": []}

    class _FakeBoard:
        def publish(self, post):
            pass
        def recent(self, limit):
            return []

    class _DummyAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id
        def generate(self, context):
            return AgentOutput(
                intents=[],
                board_post=BoardPost(agent_id=self.agent_id, title="t", body="b", tickers=[]),
            )

    s = load_settings()
    s.agent_ids = ["gpt", "gemini"]
    s.sleeve_capital_krw = 1_000_000
    s.agent_capitals = {"gpt": 1_500_000, "gemini": 500_000}
    s.trading_mode = "paper"

    repo = _FakeRepo()
    orch = ArenaOrchestrator(
        settings=s,
        context_builder=_FakeCtx(),
        board_store=_FakeBoard(),
        gateway=_FakeGateway(repo),
        agents=[_DummyAgent("gpt"), _DummyAgent("gemini")],
    )

    orch.run_cycle(snapshot=None)

    assert repo.ensure_calls
    call = repo.ensure_calls[0]
    # Total should be sum of per-agent capitals: 1_500_000 + 500_000 = 2_000_000
    assert call["total_cash_krw"] == 2_000_000
    assert call["capital_per_agent"] == {"gpt": 1_500_000, "gemini": 500_000}
