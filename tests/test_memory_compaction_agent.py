from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import arena.agents.memory_compaction_agent as memory_compaction_module
from arena.agents.memory_compaction_agent import MemoryCompactionAgent
from arena.config import Settings


class _FakeRepo:
    def __init__(self) -> None:
        self.cycle_rows = []
        self.board_rows = []
        self.research_rows = []
        self.configs = {}
        self.closed_thesis_keys = []
        self.semantic_rows_by_key = {}
        self.existing_reflection_keys = set()

    def resolve_tenant_id(self):
        return "local"

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        return self.configs.get((tenant_id, config_key))

    def memory_events_for_cycle(self, *, agent_id, cycle_id, event_types, limit, trading_mode="paper"):
        _ = (agent_id, cycle_id, event_types, limit, trading_mode)
        return list(self.cycle_rows)

    def board_posts_for_cycle(self, *, cycle_id, agent_id=None, limit=10, trading_mode="paper"):
        _ = (cycle_id, agent_id, limit, trading_mode)
        return list(self.board_rows)

    def get_research_briefings(self, *, tickers=None, categories=None, limit=10, trading_mode="paper", tenant_id=None):
        _ = (tickers, trading_mode, tenant_id)
        rows = list(self.research_rows)
        if categories:
            allowed = {str(cat).strip().lower() for cat in categories}
            rows = [row for row in rows if str(row.get("category") or "").strip().lower() in allowed]
        return rows[:limit]

    def closed_thesis_keys_for_cycle(self, *, agent_id, cycle_id, limit=4, trading_mode="paper"):
        _ = (agent_id, cycle_id, trading_mode)
        return list(self.closed_thesis_keys[:limit])

    def memory_events_by_semantic_keys(self, *, agent_id, semantic_keys, event_types=None, trading_mode="paper"):
        _ = (agent_id, trading_mode)
        allowed_types = {str(token).strip() for token in (event_types or []) if str(token).strip()}
        rows = []
        for semantic_key in semantic_keys:
            for row in self.semantic_rows_by_key.get(semantic_key, []):
                event_type = str(row.get("event_type") or "").strip()
                if allowed_types and event_type not in allowed_types:
                    continue
                rows.append(dict(row))
        return rows

    def memory_event_exists_by_semantic_key(self, *, agent_id, event_type, semantic_key, trading_mode="paper"):
        _ = (agent_id, trading_mode)
        return (str(event_type), str(semantic_key)) in self.existing_reflection_keys


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.recent_rows = []
        self.saved = []

    def recent(self, agent_id: str, limit: int) -> list[dict]:
        _ = agent_id
        return list(self.recent_rows[:limit])

    def record_reflection(
        self,
        *,
        agent_id: str,
        summary: str,
        score: float,
        payload: dict | None = None,
        semantic_key: str | None = None,
    ) -> None:
        self.saved.append(
            {
                "agent_id": agent_id,
                "summary": summary,
                "score": score,
                "payload": payload or {},
                "semantic_key": semantic_key,
            }
        )


def _settings() -> Settings:
    settings = MagicMock(spec=Settings)
    settings.agent_ids = ["gemini"]
    settings.agent_configs = {}
    settings.provider_secrets = {}
    settings.gemini_api_key = "test-gemini-key"
    settings.openai_api_key = ""
    settings.anthropic_api_key = ""
    settings.anthropic_use_vertexai = False
    settings.openai_model = "gpt-5.2"
    settings.gemini_model = "models/gemini-2.5-flash"
    settings.research_gemini_model = "models/gemini-2.5-flash"
    settings.anthropic_model = "claude-sonnet-4-6"
    settings.trading_mode = "paper"
    settings.llm_timeout_seconds = 10
    settings.memory_compaction_enabled = True
    settings.memory_compaction_cycle_event_limit = 12
    settings.memory_compaction_recent_lessons_limit = 4
    settings.memory_compaction_max_reflections = 3
    settings.memory_policy = {}
    return settings


def _thesis_rows(thesis_id: str) -> list[dict[str, object]]:
    return [
        {
            "event_id": "evt_thesis_open",
            "event_type": "thesis_open",
            "summary": "AAPL thesis open status=FILLED thesis=AI demand and margin recovery",
            "payload_json": (
                '{"thesis_id":"%s","ticker":"AAPL","state":"open","thesis_summary":"AI demand and margin recovery","position_action":"entry","strategy_refs":["momentum"]}'
                % thesis_id
            ),
        },
        {
            "event_id": "evt_thesis_update",
            "event_type": "thesis_update",
            "summary": "AAPL thesis update action=add status=FILLED thesis=Services mix now carries the thesis",
            "payload_json": (
                '{"thesis_id":"%s","ticker":"AAPL","state":"active","thesis_summary":"Services mix now carries the thesis","position_action":"add","strategy_refs":["momentum","services"]}'
                % thesis_id
            ),
        },
        {
            "event_id": "evt_thesis_close",
            "event_type": "thesis_invalidated",
            "summary": "AAPL thesis invalidated status=FILLED thesis=Guidance cut broke the margin recovery thesis",
            "payload_json": (
                '{"thesis_id":"%s","ticker":"AAPL","state":"invalidated","thesis_summary":"Guidance cut broke the margin recovery thesis","position_action":"exit","strategy_refs":["thesis_broken"]}'
                % thesis_id
            ),
        },
    ]


def test_memory_compaction_agent_follows_configured_single_agent_provider() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["claude"]
    settings.gemini_api_key = ""
    settings.anthropic_api_key = "test-anthropic-key"
    settings.anthropic_model = "claude-sonnet-4-6"

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    assert agent.provider == "claude"
    assert agent.model == "anthropic/claude-sonnet-4-6"


def test_memory_compaction_agent_prefers_direct_key_provider_over_non_direct_fallback() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["gemini", "gpt"]
    settings.gemini_api_key = ""
    settings.openai_api_key = "tenant-openai-key"

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    assert agent.provider == "gpt"
    assert agent.model == "openai/gpt-5.2"


def test_memory_compaction_agent_calls_litellm_direct(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["claude"]
    settings.gemini_api_key = ""
    settings.openai_api_key = ""
    settings.anthropic_api_key = "tenant-anthropic-key"
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert captured["model"] == "anthropic/claude-sonnet-4-6"
    assert captured["api_key"] == "tenant-anthropic-key"
    assert captured["temperature"] == 0.1


def test_memory_compaction_agent_omits_temperature_for_gpt5_models(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["gpt"]
    settings.gemini_api_key = ""
    settings.openai_api_key = "tenant-openai-key"
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert captured["model"] == "openai/gpt-5.2"
    assert captured["api_key"] == "tenant-openai-key"
    assert "temperature" not in captured


def test_memory_compaction_agent_supports_deepseek_helper_from_provider_payload(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = []
    settings.gemini_api_key = ""
    settings.openai_api_key = ""
    settings.anthropic_api_key = ""
    settings.provider_secrets = {
        "deepseek": {
            "api_key": "tenant-deepseek-key",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
        }
    }

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert agent.provider == "deepseek"
    assert captured["model"] == "deepseek/deepseek-chat"
    assert captured["api_key"] == "tenant-deepseek-key"
    assert captured["base_url"] == "https://api.deepseek.com/v1"


def test_memory_compaction_agent_retries_transient_helper_errors(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    calls = {"count": 0}
    sleeps: list[float] = []

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("503 high demand")
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(memory_compaction_module, "retry_policy_from_env", lambda: (2, 0.5))
    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(memory_compaction_module.asyncio, "sleep", _fake_sleep)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert calls["count"] == 2
    assert sleeps == [0.5]


def test_memory_compaction_agent_retries_empty_response(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    calls = {"count": 0}
    sleeps: list[float] = []

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            return {"choices": [{"message": {"content": ""}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(memory_compaction_module, "retry_policy_from_env", lambda: (1, 0.25))
    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(memory_compaction_module.asyncio, "sleep", _fake_sleep)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert calls["count"] == 2
    assert sleeps == [0.25]


def test_memory_compaction_agent_raises_after_retry_budget_exhausted(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    calls = {"count": 0}
    sleeps: list[float] = []

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        calls["count"] += 1
        raise RuntimeError("503 high demand")

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(memory_compaction_module, "retry_policy_from_env", lambda: (2, 0.5))
    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(memory_compaction_module.asyncio, "sleep", _fake_sleep)

    try:
        asyncio.run(agent._collect_response_text(prompt="PROMPT"))
    except RuntimeError as exc:
        assert "503 high demand" in str(exc)
    else:
        raise AssertionError("expected retry exhaustion to re-raise the last helper error")

    assert calls["count"] == 3
    assert sleeps == [0.5, 1.0]


def test_memory_compaction_agent_saves_reflections_from_cycle_outputs(monkeypatch) -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = (
        "최대 {max_reflections}개의 reflection만 생성하라.\n{payload_json}"
    )
    repo.cycle_rows = [
        {
            "event_id": "evt_trade",
            "event_type": "trade_execution",
            "summary": "AAPL BUY qty=2 status=FILLED policy=ok broker=filled",
            "importance_score": 0.75,
            "outcome_score": 0.8,
            "payload_json": '{"intent":{"ticker":"AAPL","side":"BUY","rationale":"breakout"},"report":{"status":"FILLED"},"decision":{"policy_hits":[]}}',
        },
        {
            "event_id": "evt_tools",
            "event_type": "react_tools_summary",
            "summary": "ReAct tools used (execution): 3",
            "importance_score": 0.6,
            "payload_json": '{"phase":"execution","tool_mix":{"quant":3},"tool_events":[{"tool":"technical_signals"},{"tool":"screen_market"},{"tool":"forecast_returns"}]}',
        },
    ]
    repo.board_rows = [
        {
            "post_id": "post_cycle",
            "title": "거래 아이디어",
            "body": "AAPL breakout 재진입",
            "draft_summary": "AAPL momentum continuation",
            "tickers": ["AAPL"],
        }
    ]
    repo.research_rows = [
        {
            "briefing_id": "brf_global",
            "category": "global_market",
            "headline": "Higher-for-longer rates keep growth multiples under pressure",
            "summary": "Macro backdrop remains selective for crowded momentum trades.",
        },
        {
            "briefing_id": "brf_geo",
            "category": "geopolitical",
            "headline": "Shipping disruptions add energy and logistics volatility",
            "summary": "Geopolitical stress is reinforcing a defensive bias.",
        },
    ]
    memory_store = _FakeMemoryStore()
    memory_store.recent_rows = [
        {
            "event_id": "evt_old",
            "event_type": "strategy_reflection",
            "summary": "Avoid chasing late-stage semis after euphoric gaps.",
        }
    ]
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    async def _fake_compact_one(*, agent_id, cycle_id, inputs):
        assert agent_id == "gpt"
        assert cycle_id == "cycle_123"
        assert inputs["cycle_memories"][0]["event_id"] == "evt_trade"
        assert inputs["board_posts"][0]["post_id"] == "post_cycle"
        assert inputs["environment_research"][0]["briefing_id"] == "brf_global"
        return [
            {
                "summary": "AAPL breakout trades worked best when momentum confirmation, size discipline, and a favorable macro regime aligned.",
                "importance_score": 0.82,
                "tags": ["momentum", "sizing", "macro"],
                "source_event_ids": ["evt_trade", "evt_tools"],
                "source_post_ids": ["post_cycle"],
                "source_briefing_ids": ["brf_global"],
            }
        ]

    monkeypatch.setattr(agent, "_compact_one", _fake_compact_one)

    saved = asyncio.run(agent.run(cycle_id="cycle_123", agent_ids=["gpt", "gpt"]))

    assert len(saved) == 1
    assert len(memory_store.saved) == 1
    assert memory_store.saved[0]["agent_id"] == "gpt"
    assert memory_store.saved[0]["summary"].startswith("AAPL breakout trades worked best")
    assert memory_store.saved[0]["score"] == 0.82
    assert memory_store.saved[0]["payload"]["source"] == "memory_compaction"
    assert memory_store.saved[0]["payload"]["cycle_id"] == "cycle_123"
    assert memory_store.saved[0]["payload"]["source_event_ids"] == ["evt_trade", "evt_tools"]
    assert memory_store.saved[0]["payload"]["source_post_ids"] == ["post_cycle"]
    assert memory_store.saved[0]["payload"]["source_briefing_ids"] == ["brf_global"]
    assert memory_store.saved[0]["semantic_key"] is None


def test_memory_compaction_agent_loads_closed_thesis_chains() -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = "{payload_json}"
    thesis_id = "thesis:gpt:AAPL:paper:2026-03-29:intent_1"
    repo.closed_thesis_keys = [thesis_id]
    repo.semantic_rows_by_key[thesis_id] = _thesis_rows(thesis_id)
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    inputs = agent._load_agent_inputs("gpt", "cycle_123")

    assert len(inputs["closed_thesis_chains"]) == 1
    chain = inputs["closed_thesis_chains"][0]
    assert chain["thesis_id"] == thesis_id
    assert chain["terminal_event_type"] == "thesis_invalidated"
    assert chain["event_ids"] == ["evt_thesis_open", "evt_thesis_update", "evt_thesis_close"]
    assert chain["events"][-1]["event_type"] == "thesis_invalidated"


def test_memory_compaction_agent_saves_thesis_chain_reflection(monkeypatch) -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = "{payload_json}"
    thesis_id = "thesis:gpt:AAPL:paper:2026-03-29:intent_1"
    repo.closed_thesis_keys = [thesis_id]
    repo.semantic_rows_by_key[thesis_id] = _thesis_rows(thesis_id)
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    async def _fake_compact_one(*, agent_id, cycle_id, inputs):
        assert agent_id == "gpt"
        assert cycle_id == "cycle_123"
        assert inputs["closed_thesis_chains"][0]["thesis_id"] == thesis_id
        return [
            {
                "summary": "AAPL thesis drifted from AI demand to services mix and should have been de-risked before the guidance cut invalidated it.",
                "importance_score": 0.84,
                "tags": ["thesis", "risk"],
                "source_event_ids": ["evt_thesis_open", "evt_thesis_close"],
                "thesis_id": thesis_id,
                "terminal_event_type": "thesis_invalidated",
            }
        ]

    monkeypatch.setattr(agent, "_compact_one", _fake_compact_one)

    saved = asyncio.run(agent.run(cycle_id="cycle_123", agent_ids=["gpt"]))

    assert len(saved) == 1
    assert saved[0]["thesis_id"] == thesis_id
    assert len(memory_store.saved) == 1
    assert memory_store.saved[0]["payload"]["source"] == "thesis_chain_compaction"
    assert memory_store.saved[0]["payload"]["thesis_id"] == thesis_id
    assert memory_store.saved[0]["payload"]["terminal_event_type"] == "thesis_invalidated"
    assert memory_store.saved[0]["semantic_key"] == f"reflection:{thesis_id}"


def test_memory_compaction_agent_skips_already_compacted_thesis_chain(monkeypatch) -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = "{payload_json}"
    thesis_id = "thesis:gpt:AAPL:paper:2026-03-29:intent_1"
    repo.closed_thesis_keys = [thesis_id]
    repo.semantic_rows_by_key[thesis_id] = _thesis_rows(thesis_id)
    repo.existing_reflection_keys.add(("strategy_reflection", f"reflection:{thesis_id}"))
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    async def _should_not_run(*, agent_id, cycle_id, inputs):
        raise AssertionError("duplicate thesis chain should not trigger compaction")

    monkeypatch.setattr(agent, "_compact_one", _should_not_run)

    saved = asyncio.run(agent.run(cycle_id="cycle_123", agent_ids=["gpt"]))

    assert saved == []
    assert memory_store.saved == []


def test_memory_compaction_prompt_must_exist_in_db() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    try:
        agent._build_prompt(agent_id="gpt", cycle_id="cycle_123", inputs={})
    except RuntimeError as exc:
        assert "memory_compactor_prompt" in str(exc)
    else:
        raise AssertionError("expected missing DB prompt to raise")


def test_memory_compaction_prompt_uses_db_template() -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = (
        "AGENT={agent_id}\nCYCLE={cycle_id}\nMAX={max_reflections}\n{payload_json}"
    )
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    prompt = agent._build_prompt(
        agent_id="gpt",
        cycle_id="cycle_123",
        inputs={
            "closed_thesis_chains": [{"thesis_id": "thesis:gpt:AAPL:paper:2026-03-29:intent_1"}],
            "cycle_memories": [{"event_id": "evt_1"}],
            "board_posts": [],
            "environment_research": [{"briefing_id": "brf_1", "headline": "Rates steady"}],
            "prior_lessons": [],
        },
    )

    assert "AGENT=gpt" in prompt
    assert "CYCLE=cycle_123" in prompt
    assert '"closed_thesis_chains"' in prompt
    assert '"event_id": "evt_1"' in prompt
    assert '"briefing_id": "brf_1"' in prompt


def test_memory_compaction_prompt_prefers_tenant_override_over_global() -> None:
    repo = _FakeRepo()
    repo.configs[("global", "memory_compactor_prompt")] = "GLOBAL {agent_id}"
    repo.configs[("local", "memory_compactor_prompt")] = "LOCAL {agent_id}"
    memory_store = _FakeMemoryStore()
    agent = MemoryCompactionAgent(settings=_settings(), repo=repo, memory_store=memory_store)

    prompt = agent._build_prompt(
        agent_id="gpt",
        cycle_id="cycle_123",
        inputs={
            "closed_thesis_chains": [],
            "cycle_memories": [],
            "board_posts": [],
            "environment_research": [],
            "prior_lessons": [],
        },
    )

    assert "LOCAL gpt" in prompt
    assert "GLOBAL gpt" not in prompt
