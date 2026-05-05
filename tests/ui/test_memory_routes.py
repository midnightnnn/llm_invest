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

def test_memory_config_save_scopes_compaction_prompt_to_tenant(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)

    response = client.post(
        "/api/memory/config",
        params={"tenant_id": "local"},
        json={
            "policy": {"compaction": {"enabled": True}},
            "compaction_prompt": "TENANT PROMPT {agent_id}",
        },
    )

    assert response.status_code == 200
    assert repo.get_config("local", "memory_compactor_prompt") == "TENANT PROMPT {agent_id}"
    assert repo.get_config("global", "memory_compactor_prompt") is None
    payload = response.json()
    assert payload["meta"]["tenant_compaction_prompt"] == "TENANT PROMPT {agent_id}"
    assert payload["meta"]["prompt_source"] == "tenant"


def test_memory_settings_page_uses_compact_prompt_copy(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.get("/settings?tenant_id=local&tab=memory")

    assert response.status_code == 200
    assert "Memory Map" in response.text
    assert "Map" in response.text
    assert "Activity" in response.text
    assert "Network" in response.text
    assert "회고 정리 안내문" in response.text
    assert "투자 논리 시작" in response.text
    assert "닫힌 논리 체인 우선" in response.text
    assert "현재 global 기본 프롬프트를 상속 중입니다" not in response.text
    assert "현재 tenant 전용 컴팩션 프롬프트가 적용됩니다" not in response.text


def test_api_memory_graph_exposes_runtime_stats_and_select_fields(monkeypatch) -> None:
    class _MemoryStatsRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "GROUP BY event_type, agent_id" in sql:
                return [
                    {
                        "event_type": "strategy_reflection",
                        "agent_id": "gpt",
                        "cnt": 3,
                        "last_created_at": "2026-03-15T09:10:11Z",
                    }
                ]
            if "COUNTIF(TRIM(COALESCE(graph_node_id" in sql:
                return [
                    {
                        "total_memory_events": 5,
                        "with_graph_node_id": 4,
                        "with_causal_chain_id": 3,
                        "with_last_accessed_at": 2,
                        "with_effective_score": 5,
                        "last_accessed_at": "2026-03-15T10:11:12Z",
                    }
                ]
            if "GROUP BY memory_tier" in sql:
                return [
                    {"memory_tier": "semantic", "cnt": 2},
                    {"memory_tier": "episodic", "cnt": 3},
                ]
            if "FROM `proj.ds.memory_access_events`" in sql:
                return [
                    {
                        "access_event_count": 12,
                        "prompt_use_count": 5,
                        "last_accessed_at": "2026-03-15T11:12:13Z",
                    }
                ]
            if "FROM `proj.ds.memory_graph_nodes`" in sql:
                return [
                    {"node_kind": "memory_event", "cnt": 6, "last_created_at": "2026-03-15T12:13:14Z"},
                    {"node_kind": "execution_report", "cnt": 2, "last_created_at": "2026-03-15T12:13:14Z"},
                ]
            if "FROM `proj.ds.memory_graph_edges`" in sql:
                return [
                    {"edge_type": "EXECUTED_AS", "cnt": 2, "last_created_at": "2026-03-15T12:14:15Z"},
                    {"edge_type": "ABSTRACTED_TO", "cnt": 1, "last_created_at": "2026-03-15T12:14:15Z"},
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _MemoryStatsRepo()
    repo.set_config(
        "local",
        "memory_forgetting_tuning_state",
        json.dumps(
            {
                "configured_mode": "shadow",
                "effective_mode": "bounded_ema",
                "transition": {"action": "auto_promote", "reason": "stable enough"},
                "drift": {"recommendation_drift": 0.12},
                "history": {"shadow_runs_since_transition": 0, "bounded_ema_runs_since_transition": 3},
            }
        ),
    )
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/memory/graph", params={"tenant_id": "local"})

    assert response.status_code == 200
    assert response.headers.get("cache-control") == "no-store"
    payload = response.json()
    runtime = payload["meta"]["runtime"]
    assert runtime["graph"]["total_nodes"] == 8
    assert runtime["graph"]["total_edges"] == 3
    assert runtime["memory"]["with_graph_node_id"] == 4
    assert runtime["forgetting_tuning_state"]["effective_mode"] == "bounded_ema"

    access_curve_node = next(node for node in payload["nodes"] if node["id"] == "forgetting.access_curve")
    tuning_mode_node = next(node for node in payload["nodes"] if node["id"] == "forgetting.tuning.mode")
    assert access_curve_node["type"] == "select"
    assert access_curve_node["options"] == ["sqrt", "log", "capped_linear"]
    assert tuning_mode_node["type"] == "select"
    assert tuning_mode_node["options"] == ["shadow", "bounded_ema"]


def test_memory_graph_runtime_payload_marks_invalid_tuning_state(monkeypatch) -> None:
    class _MemoryStatsRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _MemoryStatsRepo()
    repo.set_config("local", "memory_forgetting_tuning_state", "[1,2,3]")
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/memory/graph", params={"tenant_id": "local"})

    assert response.status_code == 200
    runtime = response.json()["meta"]["runtime"]
    assert runtime["forgetting_tuning_state"] == {}
    assert runtime["invalid_config_keys"] == ["memory_forgetting_tuning_state"]


def test_api_memory_activity_returns_examples(monkeypatch) -> None:
    class _MemoryActivityRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "GROUP BY event_type, agent_id" in sql:
                return [{"event_type": "strategy_reflection", "agent_id": "gpt", "cnt": 2, "last_created_at": "2026-03-15T09:10:11Z"}]
            if "COUNTIF(TRIM(COALESCE(graph_node_id" in sql:
                return [{"total_memory_events": 2, "with_graph_node_id": 1, "with_causal_chain_id": 1, "with_last_accessed_at": 1, "with_effective_score": 2, "last_accessed_at": "2026-03-15T10:11:12Z"}]
            if "GROUP BY memory_tier" in sql:
                return [{"memory_tier": "semantic", "cnt": 1}, {"memory_tier": "episodic", "cnt": 1}]
            if "FROM `proj.ds.memory_access_events`" in sql and "GROUP BY event_id" not in sql:
                return [{"access_event_count": 4, "prompt_use_count": 2, "last_accessed_at": "2026-03-15T11:12:13Z"}]
            if "FROM `proj.ds.memory_graph_nodes`" in sql:
                return [{"node_kind": "memory_event", "cnt": 2, "last_created_at": "2026-03-15T12:13:14Z"}]
            if "FROM `proj.ds.memory_graph_edges`" in sql:
                return [{"edge_type": "INFORMED_BY", "cnt": 1, "last_created_at": "2026-03-15T12:14:15Z"}]
            if "LEFT JOIN access_summary AS a" in sql:
                return [
                    {
                        "event_id": "evt_1",
                        "created_at": "2026-03-15T08:00:00Z",
                        "agent_id": "gpt",
                        "event_type": "strategy_reflection",
                        "summary": "AAPL breakout thesis improved after volume confirmation",
                        "memory_tier": "semantic",
                        "primary_regime": "bull",
                        "primary_strategy_tag": "breakout",
                        "primary_sector": "Technology",
                        "access_count": 3,
                        "last_accessed_at": "2026-03-15T11:12:13Z",
                        "effective_score": 0.81,
                        "context_tags_json": json.dumps({"regime_tags": ["bull"], "strategy_tags": ["breakout"], "sector_tags": ["tech"]}),
                        "payload_json": json.dumps({"ticker": "AAPL"}),
                        "access_events": 3,
                        "prompt_uses": 2,
                        "last_prompt_at": "2026-03-15T11:00:00Z",
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    app = _build_app(repo=_MemoryActivityRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/memory/activity", params={"tenant_id": "local"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["stats"]["access_runtime"]["prompt_use_count"] == 2
    assert payload["examples"][0]["ticker"] == "AAPL"
    assert payload["examples"][0]["prompt_uses"] == 2
    assert "bull" in payload["examples"][0]["badges"]


def test_api_memory_network_returns_nodes_and_links(monkeypatch) -> None:
    class _MemoryNetworkRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "FROM `proj.ds.memory_graph_nodes` AS n" in sql:
                return [
                    {
                        "node_id": "mem_1",
                        "created_at": "2026-03-15T08:00:00Z",
                        "node_kind": "memory_event",
                        "source_table": "agent_memory_events",
                        "source_id": "evt_1",
                        "agent_id": "gpt",
                        "cycle_id": "cycle_1",
                        "summary": "AAPL thesis update",
                        "ticker": "AAPL",
                        "memory_tier": "semantic",
                        "primary_regime": "bull",
                        "context_tags_json": json.dumps({"strategy_tags": ["breakout"]}),
                        "payload_json": json.dumps({"ticker": "AAPL"}),
                        "event_type": "thesis_update",
                        "access_count": 4,
                        "last_accessed_at": "2026-03-15T11:12:13Z",
                        "effective_score": 0.88,
                        "access_events": 4,
                        "prompt_uses": 2,
                    },
                    {
                        "node_id": "mem_2",
                        "created_at": "2026-03-15T08:05:00Z",
                        "node_kind": "memory_event",
                        "source_table": "agent_memory_events",
                        "source_id": "evt_2",
                        "agent_id": "claude",
                        "cycle_id": "cycle_2",
                        "summary": "AAPL trade execution",
                        "ticker": "AAPL",
                        "memory_tier": "episodic",
                        "primary_regime": "bull",
                        "context_tags_json": json.dumps({"strategy_tags": ["breakout"]}),
                        "payload_json": json.dumps({"ticker": "AAPL"}),
                        "event_type": "trade_execution",
                        "access_count": 1,
                        "last_accessed_at": "2026-03-15T11:12:13Z",
                        "effective_score": 0.52,
                        "access_events": 1,
                        "prompt_uses": 0,
                    },
                ]
            if "FROM `proj.ds.memory_graph_edges`" in sql:
                return [
                    {
                        "edge_id": "edge_1",
                        "created_at": "2026-03-15T09:00:00Z",
                        "from_node_id": "mem_1",
                        "to_node_id": "mem_2",
                        "edge_type": "INFORMED_BY",
                        "edge_strength": 0.76,
                        "confidence": 0.82,
                        "causal_chain_id": "chain_1",
                    }
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    app = _build_app(repo=_MemoryNetworkRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/memory/network", params={"tenant_id": "local", "days": 30})

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["node_count"] == 2
    assert payload["meta"]["edge_count"] == 1
    assert payload["meta"]["available_agents"] == ["claude", "gpt"]
    assert payload["nodes"][0]["used_in_prompt"] is True
    assert payload["links"][0]["edge_type"] == "INFORMED_BY"
