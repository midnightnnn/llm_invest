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

def test_api_tool_frequency_returns_llm_tool_matrix(monkeypatch) -> None:
    class _ToolFreqRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "agent_memory_events" not in sql:
                return []
            return [
                {
                    "agent_id": "gpt",
                    "payload_json": json.dumps(
                        {
                            "tool_events": [
                                {"tool": "recommend_opportunities"},
                                {"tool": "recommend_opportunities"},
                                {"tool": "legacy_old_tool"},
                            ]
                        }
                    ),
                },
                {
                    "agent_id": "gemini",
                    "payload_json": json.dumps(
                        {
                            "tool_events": [
                                {"tool": "recommend_opportunities"},
                                {"tool": "optimize_portfolio"},
                            ]
                        }
                    ),
                },
            ]

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _ToolFreqRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/tool-frequency", params={"tenant_id": "local"})
    assert response.status_code == 200
    payload = response.json()

    assert payload["tools"] == ["recommend_opportunities", "optimize_portfolio"]
    assert set(payload["agents"]) == {"gpt", "gemini"}
    assert payload["matrix"]["recommend_opportunities"] == {"gpt": 2, "gemini": 1}
    assert payload["matrix"]["optimize_portfolio"] == {"gpt": 0, "gemini": 1}
    assert "legacy_old_tool" not in payload["tools"]


def test_api_sleeve_snapshot_cards_returns_html_and_charts(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/api/sleeve-snapshot-cards", params={"tenant_id": "local"})
    assert response.status_code == 200
    payload = response.json()
    assert "html" in payload
    assert "charts" in payload
    assert "gpt" in payload["html"]
    assert isinstance(payload["charts"], list)


def test_api_sleeve_snapshot_cards_uses_tenant_market_sources_in_live_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TRADING_MODE", "live")
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("tenant-k", "kis_target_market", "kospi")

    response = client.get("/api/sleeve-snapshot-cards", params={"tenant_id": "tenant-k"})

    assert response.status_code == 200
    assert repo.snapshot_calls
    assert all(call["tenant_id"] == "tenant-k" for call in repo.snapshot_calls)
    assert all(call["include_simulated"] is False for call in repo.snapshot_calls)
    assert all(call["sources"] == ["open_trading_kospi_quote", "open_trading_kospi"] for call in repo.snapshot_calls)


def test_sleeves_page_uses_tenant_market_sources_in_live_mode(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_TRADING_MODE", "live")
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("tenant-k", "kis_target_market", "kospi")

    response = client.get("/sleeves", params={"tenant_id": "tenant-k"})

    assert response.status_code == 200
    assert repo.snapshot_calls
    assert all(call["tenant_id"] == "tenant-k" for call in repo.snapshot_calls)
    assert all(call["include_simulated"] is False for call in repo.snapshot_calls)
    assert all(call["sources"] == ["open_trading_kospi_quote", "open_trading_kospi"] for call in repo.snapshot_calls)


def test_api_capital_waterfall_uses_canonical_baseline_summary(monkeypatch) -> None:
    class _WaterfallRepo(_DummyRepo):
        def build_agent_sleeve_snapshot(
            self,
            *,
            agent_id: str,
            sources=None,
            include_simulated: bool = True,
            tenant_id: str | None = None,
        ):
            _ = (sources, include_simulated, tenant_id)
            return (
                AccountSnapshot(cash_krw=200_000.0, total_equity_krw=2_950_195.0, positions={}),
                3_000_000.0,
                {
                    "seed_cash_krw": 1_000_000.0,
                    "seed_positions_cost_krw": 938_568.0,
                    "capital_flow_krw": 1_061_432.0,
                    "capital_event_count": 1,
                    "transfer_equity_krw": 0.0,
                    "transfer_event_count": 0,
                    "manual_cash_adjustment_krw": 0.0,
                    "manual_cash_adjustment_count": 0,
                    "current_cash_krw": 200_000.0,
                    "current_positions_value_krw": 2_750_195.0,
                    "seed_source": "agent_state_checkpoint",
                    "initialized_at": datetime(2026, 3, 1, tzinfo=timezone.utc),
                },
            )

        def capital_events_since(self, *, agent_id: str, since, tenant_id: str | None = None):
            _ = (agent_id, since, tenant_id)
            return [
                {
                    "occurred_at": datetime(2026, 3, 20, tzinfo=timezone.utc),
                    "event_type": "INJECTION",
                    "amount_krw": 1_061_432.0,
                }
            ]

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _WaterfallRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/capital/waterfall", params={"tenant_id": "local", "agent_id": "gpt"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_invested"] == 3_000_000
    assert payload["summary"]["seed_positions_cost_krw"] == 938_568
    assert payload["summary"]["capital_flow_krw"] == 1_061_432
    assert payload["summary"]["pnl_krw"] == -49_805


def test_api_capital_waterfall_prefers_traced_actual_basis_summary(monkeypatch) -> None:
    class _WaterfallRepo(_DummyRepo):
        def build_agent_sleeve_snapshot(
            self,
            *,
            agent_id: str,
            sources=None,
            include_simulated: bool = True,
            tenant_id: str | None = None,
        ):
            _ = (agent_id, sources, include_simulated, tenant_id)
            return (
                AccountSnapshot(cash_krw=500_000.0, total_equity_krw=3_950_000.0, positions={}),
                4_000_000.0,
                {
                    "seed_cash_krw": 1_000_000.0,
                    "seed_positions_cost_krw": 3_000_000.0,
                    "capital_flow_krw": 2_000_000.0,
                    "capital_event_count": 2,
                    "current_cash_krw": 500_000.0,
                    "current_positions_value_krw": 3_450_000.0,
                    "seed_source": "agent_state_checkpoint",
                    "initialized_at": datetime(2026, 3, 27, tzinfo=timezone.utc),
                },
            )

        def trace_agent_actual_capital_basis(self, *, agent_id: str, tenant_id: str | None = None):
            _ = (agent_id, tenant_id)
            return {
                "origin_at": datetime(2026, 3, 11, tzinfo=timezone.utc),
                "origin_source": "legacy_agent_sleeve",
                "seed_cash_krw": 2_000_000.0,
                "seed_positions_cost_krw": 0.0,
                "baseline_equity_krw": 4_110_000.0,
                "capital_flow_krw": 2_110_000.0,
                "capital_event_count": 2,
                "transfer_equity_krw": 0.0,
                "transfer_event_count": 0,
                "manual_cash_adjustment_krw": 0.0,
                "manual_cash_adjustment_count": 0,
            }

        def capital_events_since(self, *, agent_id: str, since, tenant_id: str | None = None):
            _ = (agent_id, since, tenant_id)
            return [
                {
                    "occurred_at": datetime(2026, 3, 17, tzinfo=timezone.utc),
                    "event_type": "INJECTION",
                    "amount_krw": 1_110_000.0,
                },
                {
                    "occurred_at": datetime(2026, 3, 27, tzinfo=timezone.utc),
                    "event_type": "INJECTION",
                    "amount_krw": 1_000_000.0,
                },
            ]

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _WaterfallRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/capital/waterfall", params={"tenant_id": "local", "agent_id": "gpt"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_invested"] == 4_110_000
    assert payload["summary"]["capital_flow_krw"] == 2_110_000
    assert payload["summary"]["pnl_krw"] == -160_000


def test_api_nav_chart_includes_token_usage_summary_and_trade_counts(monkeypatch) -> None:
    class _NavRepo(_DummyRepo):
        def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
            self.fetch_calls.append((sql, params))
            if "agent_memory_events" in sql:
                return [
                    {
                        "agent_id": "gpt",
                        "llm_calls": 5,
                        "prompt_tokens": 1500,
                        "completion_tokens": 300,
                        "cached_tokens": 400,
                        "thinking_tokens": 40,
                    },
                    {
                        "agent_id": "gemini",
                        "llm_calls": 1,
                        "prompt_tokens": 700,
                        "completion_tokens": 60,
                        "cached_tokens": 200,
                        "thinking_tokens": 20,
                    },
                ]
            if "execution_reports" in sql:
                return [
                    {"agent_id": "gpt", "trade_count": 3},
                    {"agent_id": "gemini", "trade_count": 1},
                ]
            if "agent_nav_daily" in sql:
                return [
                    {"nav_date": "2026-03-10", "agent_id": "gpt", "nav_krw": 100.0, "pnl_krw": 0.0, "pnl_ratio": 0.0},
                    {"nav_date": "2026-03-11", "agent_id": "gpt", "nav_krw": 105.0, "pnl_krw": 5.0, "pnl_ratio": 0.05},
                    {"nav_date": "2026-03-10", "agent_id": "gemini", "nav_krw": 100.0, "pnl_krw": 0.0, "pnl_ratio": 0.0},
                    {"nav_date": "2026-03-11", "agent_id": "gemini", "nav_krw": 102.0, "pnl_krw": 2.0, "pnl_ratio": 0.02},
                ]
            return []

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _NavRepo()
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/api/nav/chart", params={"tenant_id": "local", "days": 30})
    assert response.status_code == 200
    payload = response.json()

    summary = {str(row["name"]): row for row in payload["summary"]}
    assert summary["gpt"]["trade_count"] == 3
    assert summary["gemini"]["trade_count"] == 1
    assert summary["gpt"]["llm_calls"] == 5
    assert summary["gpt"]["prompt_tokens"] == 1500
    assert summary["gpt"]["completion_tokens"] == 300
    assert summary["gpt"]["cached_tokens"] == 400
    assert summary["gpt"]["thinking_tokens"] == 40
    assert summary["gpt"]["total_tokens"] == 1840
    assert summary["gpt"]["cache_ratio"] == 26.7
    assert summary["gemini"]["llm_calls"] == 1
    assert summary["gemini"]["total_tokens"] == 780
