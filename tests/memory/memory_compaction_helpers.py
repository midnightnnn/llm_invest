from __future__ import annotations

from unittest.mock import MagicMock

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
        self.existing_compaction_rows = []
        self.last_board_limit = None

    def resolve_tenant_id(self):
        return "local"

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        return self.configs.get((tenant_id, config_key))

    def memory_events_for_cycle(self, *, agent_id, cycle_id, event_types, limit, trading_mode="paper"):
        _ = (agent_id, cycle_id, event_types, limit, trading_mode)
        return list(self.cycle_rows)

    def board_posts_for_cycle(self, *, cycle_id, agent_id=None, limit=10, trading_mode="paper"):
        _ = (cycle_id, agent_id, trading_mode)
        self.last_board_limit = limit
        return list(self.board_rows[:limit])

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

    def compaction_reflections_for_cycle(self, *, agent_id, cycle_id, trading_mode="paper", limit=10):
        _ = (agent_id, cycle_id, trading_mode, limit)
        return list(self.existing_compaction_rows)


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
    settings.memory_compaction_board_post_limit = 3
    settings.memory_compaction_board_body_chars = 1200
    settings.memory_compaction_cycle_summary_chars = 900
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
