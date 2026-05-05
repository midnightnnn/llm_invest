from __future__ import annotations

import json
from datetime import datetime, timezone

from arena.config import load_settings
from arena.models import AccountSnapshot
from arena.ui.server import _build_app
from arena.ui.layout import tailwind_layout as _tailwind_layout
from tests.direct_route_client import DirectRouteClient
from tests.ui.helpers import (
    _DummyRepo,
    _client,
    _client_with_repo,
    _client_with_repo_and_credential_store,
)

def test_api_board_uses_tenant_filter(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.fetch_calls.clear()
    response = client.get("/api/board", params={"tenant_id": "tenant-x", "limit": 5})
    assert response.status_code == 200
    assert repo.fetch_calls
    _, params = repo.fetch_calls[-1]
    assert isinstance(params, dict)
    assert params.get("tenant_id") == "tenant-x"


def test_api_nav_uses_tenant_filter(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.fetch_calls.clear()
    response = client.get("/api/nav", params={"tenant_id": "tenant-y", "days": 10})
    assert response.status_code == 200
    assert repo.fetch_calls
    _, params = repo.fetch_calls[-1]
    assert isinstance(params, dict)
    assert params.get("tenant_id") == "tenant-y"


def test_nav_page_renders_blocked_status_in_header(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.latest_run_status_row = {
        "tenant_id": "local",
        "run_id": "cycle_1",
        "recorded_at": "2026-03-14T12:00:00+00:00",
        "run_type": "agent_cycle",
        "status": "blocked",
        "reason_code": "reconciliation_failed",
        "stage": "reconcile",
        "message": "실계좌와 AI 장부가 맞지 않아 거래를 중단했습니다.",
        "log_uri": "https://example.com/logs",
        "detail_json": {"exit_code": 3},
    }

    response = client.get("/nav", params={"tenant_id": "local"})

    assert response.status_code == 200
    # Status label shown in header indicator (not banner)
    assert "실행 중단" in response.text


def test_api_board_trades_requires_cycle_id(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.fetch_calls.clear()
    response = client.get("/api/board/trades", params={"tenant_id": "local"})
    assert response.status_code == 200
    assert response.json() == []
    assert repo.fetch_calls == []


def test_api_board_theses_returns_chain_and_compacted_lesson(monkeypatch) -> None:
    class _BoardThesisRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "event_type IN UNNEST(@event_types)" in sql:
                return [
                    {
                        "created_at": "2026-03-29T01:00:00Z",
                        "agent_id": "gpt",
                        "event_type": "thesis_open",
                        "summary": "AAPL thesis opened",
                        "semantic_key": "thesis:gpt:AAPL:1",
                        "payload_json": json.dumps(
                            {
                                "thesis_id": "thesis:gpt:AAPL:1",
                                "ticker": "AAPL",
                                "side": "BUY",
                                "state": "open",
                                "thesis_summary": "AI demand and margin recovery",
                                "strategy_refs": ["momentum", "quality"],
                            }
                        ),
                    },
                    {
                        "created_at": "2026-03-29T03:00:00Z",
                        "agent_id": "gpt",
                        "event_type": "thesis_invalidated",
                        "summary": "Guidance cut broke the thesis",
                        "semantic_key": "thesis:gpt:AAPL:1",
                        "payload_json": json.dumps(
                            {
                                "thesis_id": "thesis:gpt:AAPL:1",
                                "ticker": "AAPL",
                                "side": "BUY",
                                "state": "invalidated",
                                "thesis_summary": "AI demand and margin recovery",
                            }
                        ),
                    },
                ]
            if "JSON_VALUE(payload_json, '$.source') = 'thesis_chain_compaction'" in sql:
                return [
                    {
                        "created_at": "2026-03-29T05:00:00Z",
                        "summary": "Trim earlier when the thesis starts drifting.",
                        "payload_json": json.dumps(
                            {
                                "source": "thesis_chain_compaction",
                                "thesis_id": "thesis:gpt:AAPL:1",
                            }
                        ),
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _BoardThesisRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get("/api/board/theses", params={"tenant_id": "local", "cycle_id": "cycle_1", "agent_id": "gpt"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["chains"]) == 1
    chain = payload["chains"][0]
    assert chain["thesis_id"] == "thesis:gpt:AAPL:1"
    assert chain["ticker"] == "AAPL"
    assert chain["terminal_event_type"] == "thesis_invalidated"
    assert chain["reflection"]["summary"] == "Trim earlier when the thesis starts drifting."
    assert [event["event_type"] for event in chain["events"]] == ["thesis_open", "thesis_invalidated"]
    event_sql = repo.fetch_calls[0][0]
    assert "cycle_id = @cycle_id" not in event_sql
    assert "JSON_VALUE(payload_json, '$.cycle_id')" in event_sql
    assert "JSON_VALUE(payload_json, '$.intent.cycle_id')" in event_sql


def test_api_board_prompt_returns_prompt_bundle(monkeypatch) -> None:
    class _BoardPromptRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "event_type = 'react_tools_summary'" not in sql:
                return []
            return [
                {
                    "created_at": "2026-03-29T01:02:00Z",
                    "summary": "Board prompt bundle snapshot before post generation.",
                    "payload_json": json.dumps(
                        {
                            "phase": "board",
                            "analysis_funnel": {"pending_nonheld": 1},
                            "tool_events": [{"tool": "recommend_opportunities", "phase": "explore"}],
                            "tool_mix": {"quant": 1, "macro": 0, "sentiment": 0, "performance": 0, "context": 0, "other": 0},
                            "prompt_bundle": {
                                "system_prompt": "system body",
                                "phases": [
                                    {"phase": "explore", "session_id": "sid_1", "resume_session": False, "prompt": "explore body"},
                                    {"phase": "execution", "session_id": "sid_1", "resume_session": True, "prompt": "execution body"},
                                    {"phase": "board", "session_id": "sid_1", "resume_session": True, "prompt": "board body"},
                                ],
                            },
                        }
                    ),
                }
            ]

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _BoardPromptRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get(
        "/api/board/prompt",
        params={
            "tenant_id": "local",
            "agent_id": "gpt",
            "ts": "2026-03-29T01:00:00+00:00",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt_bundle"]["system_prompt"] == "system body"
    assert payload["prompt_bundle"]["phases"][0]["prompt"] == "explore body"
    assert payload["analysis_funnel"]["pending_nonheld"] == 1
    assert payload["tool_events"][0]["tool"] == "recommend_opportunities"


def test_api_board_prompt_prefers_llm_audit_tables(monkeypatch) -> None:
    class _BoardPromptAuditRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "FROM `proj.ds.agent_llm_interactions`" in sql and "phase = 'board'" in sql and "LIMIT 1" in sql:
                return [
                    {
                        "llm_call_id": "llm_board_1",
                        "cycle_id": "cycle_1",
                        "created_at": "2026-03-29T01:02:00Z",
                        "agent_id": "gpt",
                        "phase": "board",
                        "session_id": "sid_1",
                        "resume_session": True,
                        "system_prompt": "system body",
                        "user_prompt": "board body",
                        "available_tools_json": json.dumps([{"tool_id": "recommend_opportunities"}]),
                        "context_payload_json": json.dumps({"analysis_funnel": {"fully_analyzed_candidates": 1}}),
                        "context_sections_json": json.dumps({"memory_context": "Memory"}),
                        "token_usage_json": json.dumps({"prompt_tokens": 100}),
                    }
                ]
            if "FROM `proj.ds.agent_llm_interactions`" in sql and "CASE phase" in sql:
                return [
                    {
                        "llm_call_id": "llm_explore_1",
                        "cycle_id": "cycle_1",
                        "created_at": "2026-03-29T01:00:00Z",
                        "agent_id": "gpt",
                        "phase": "explore",
                        "session_id": "sid_1",
                        "resume_session": False,
                        "system_prompt": "system body",
                        "user_prompt": "explore body",
                        "available_tools_json": json.dumps([{"tool_id": "recommend_opportunities"}]),
                        "context_payload_json": json.dumps({"analysis_funnel": {"screened_only_candidates": 1}}),
                        "context_sections_json": json.dumps({"market_context": [{"ticker": "AAPL"}]}),
                        "token_usage_json": json.dumps({"prompt_tokens": 80}),
                    },
                    {
                        "llm_call_id": "llm_board_1",
                        "cycle_id": "cycle_1",
                        "created_at": "2026-03-29T01:02:00Z",
                        "agent_id": "gpt",
                        "phase": "board",
                        "session_id": "sid_1",
                        "resume_session": True,
                        "system_prompt": "system body",
                        "user_prompt": "board body",
                        "available_tools_json": json.dumps([{"tool_id": "recommend_opportunities"}]),
                        "context_payload_json": json.dumps({"analysis_funnel": {"fully_analyzed_candidates": 1}}),
                        "context_sections_json": json.dumps({"memory_context": "Memory"}),
                        "token_usage_json": json.dumps({"prompt_tokens": 100}),
                    },
                ]
            if "FROM `proj.ds.agent_llm_tool_events`" in sql:
                return [
                    {
                        "llm_call_id": "llm_explore_1",
                        "tool_event_id": "tool_1",
                        "created_at": "2026-03-29T01:01:00Z",
                        "phase": "explore",
                        "tool_name": "recommend_opportunities",
                        "source": "builtin",
                        "args_json": json.dumps({"limit": 3}),
                        "model_visible_result_json": json.dumps({"rows": [{"ticker": "AAPL"}]}),
                        "elapsed_ms": 12,
                        "error": None,
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _BoardPromptAuditRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get(
        "/api/board/prompt",
        params={
            "tenant_id": "local",
            "agent_id": "gpt",
            "ts": "2026-03-29T01:02:00+00:00",
            "cycle_id": "cycle_1",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prompt_bundle"]["system_prompt"] == "system body"
    assert [row["phase"] for row in payload["prompt_bundle"]["phases"]] == ["explore", "board"]
    assert payload["prompt_bundle"]["phases"][0]["context_sections"]["market_context"][0]["ticker"] == "AAPL"
    assert payload["tool_events"][0]["tool"] == "recommend_opportunities"
    assert payload["analysis_funnel"]["screened_only_candidates"] == 1
    assert not any("react_tools_summary" in sql for sql, _ in repo.fetch_calls)


def test_board_page_includes_prompt_and_memory_panels(monkeypatch) -> None:
    class _BoardRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "FROM `proj.ds.board_posts`" in sql:
                return [
                    {
                        "post_id": "post_1",
                        "created_at": datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc),
                        "agent_id": "gpt",
                        "title": "AAPL review",
                        "body": "Revisited the thesis.",
                        "tickers": ["AAPL"],
                        "cycle_id": "cycle_1",
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _BoardRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get("/board", params={"tenant_id": "local"})

    assert response.status_code == 200
    assert 'data-prompt-panel' in response.text
    assert 'data-theses-panel' in response.text
    assert "/api/board/prompt" in response.text
    assert "/api/board/theses" in response.text
    assert "Prompt Details" in response.text
    assert "Captured Model I/O" in response.text
    assert "CONTEXT DETAILS" not in response.text
    assert "Compacted Tool Transcript" not in response.text
    assert "Related Memory" in response.text


def test_showcase_board_page_renders_posts(monkeypatch) -> None:
    class _BoardRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "FROM `proj.ds.board_posts`" in sql:
                return [
                    {
                        "post_id": "post_1",
                        "created_at": datetime(2026, 3, 29, 1, 0, tzinfo=timezone.utc),
                        "agent_id": "gpt",
                        "title": "AAPL review",
                        "body": "Revisited the thesis.",
                        "tickers": ["AAPL"],
                        "cycle_id": "cycle_1",
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setenv("ARENA_SHOWCASE_TENANT", "midnightnnn")
    repo = _BoardRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get("/showcase/midnightnnn/board")

    assert response.status_code == 200
    assert "AAPL review" in response.text
    assert "Revisited the thesis." in response.text
    assert "게시글이 없습니다." not in response.text
    assert "Prompt Details" not in response.text


def test_board_page_empty_state_mentions_missing_gemini_key(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARENA_RESEARCH_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    repo = _DummyRepo()
    client = DirectRouteClient(_build_app(repo=repo, settings=load_settings()))

    response = client.get("/board", params={"tenant_id": "local"})

    assert response.status_code == 200
    assert "Gemini 키가 없어 새로운 리서치 브리핑 생성도 비활성화되어 있습니다." in response.text
