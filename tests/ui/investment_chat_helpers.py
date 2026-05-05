from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from arena.config import AgentConfig, load_settings
from arena.models import AccountSnapshot, ExecutionReport, ExecutionStatus, Position
from arena.tools.registry import ToolEntry, ToolRegistry
from arena.ui.layout import tailwind_layout
from arena.ui.server import _build_app
from tests.direct_route_client import DirectRouteClient
from tests.ui.helpers import _DummyRepo

class _ChatOrderRepo(_DummyRepo):
    tenant_id = "local"

    def __init__(
        self,
        *,
        account_snapshot: AccountSnapshot | None = None,
        sleeve_snapshot: AccountSnapshot | None = None,
    ) -> None:
        super().__init__()
        self.audit_logs: list[dict[str, object]] = []
        self.order_intents: list[dict[str, object]] = []
        self.execution_reports: list[dict[str, object]] = []
        self.trade_history_rows: list[dict[str, object]] = []
        self.trade_history_calls: list[dict[str, object]] = []
        self.sleeve_calls: list[dict[str, object]] = []
        self.account_snapshot = account_snapshot or AccountSnapshot(
            cash_krw=9_000_000.0,
            total_equity_krw=10_000_000.0,
            usd_krw_rate=1400.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL",
                    quantity=2,
                    avg_price_krw=120_000,
                    market_price_krw=130_000,
                )
            },
        )
        self.sleeve_snapshot = sleeve_snapshot or AccountSnapshot(
            cash_krw=4_000_000.0,
            total_equity_krw=5_000_000.0,
            usd_krw_rate=1400.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL",
                    quantity=2,
                    avg_price_krw=120_000,
                    market_price_krw=130_000,
                )
            },
        )

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return str(tenant_id or self.tenant_id or "local").strip().lower() or "local"

    def latest_account_snapshot(self, *, tenant_id: str | None = None):
        _ = tenant_id
        return self.account_snapshot

    def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True, tenant_id=None):
        self.sleeve_calls.append(
            {
                "agent_id": agent_id,
                "sources": sources,
                "include_simulated": include_simulated,
                "tenant_id": tenant_id,
            }
        )
        return self.sleeve_snapshot, float(self.sleeve_snapshot.total_equity_krw), {"source": "test_sleeve"}

    def recent_turnover_krw(self, *args, **kwargs) -> float:
        _ = args, kwargs
        return 0.0

    def recent_intent_count(self, *args, **kwargs) -> int:
        _ = args, kwargs
        return 0

    def last_trade_time(self, *args, **kwargs):
        _ = args, kwargs
        return None

    def recent_trade_history(self, **kwargs):
        self.trade_history_calls.append(dict(kwargs))
        return list(self.trade_history_rows)

    def write_order_intent(self, intent, decision) -> None:
        self.order_intents.append({"intent": intent, "decision": decision})

    def write_execution_report(self, intent, report) -> None:
        self.execution_reports.append({"intent": intent, "report": report})

    def append_runtime_audit_log(self, **kwargs) -> None:
        self.audit_logs.append(dict(kwargs))


class _FakeExecutionMemory:
    instances: list["_FakeExecutionMemory"] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.executions: list[dict[str, object]] = []
        self.theses: list[dict[str, object]] = []
        self.reflections: list[dict[str, object]] = []
        self.__class__.instances.append(self)

    def record_execution(self, *, intent, decision, report) -> None:
        self.executions.append({"intent": intent, "decision": decision, "report": report})

    def record_thesis_lifecycle(self, *, intent, decision, report, snapshot_before=None) -> None:
        self.theses.append(
            {
                "intent": intent,
                "decision": decision,
                "report": report,
                "snapshot_before": snapshot_before,
            }
        )

    def record_reflection(self, agent_id, summary, *, score=0.5, payload=None, semantic_key=None) -> None:
        self.reflections.append(
            {
                "agent_id": agent_id,
                "summary": summary,
                "score": score,
                "payload": payload or {},
                "semantic_key": semantic_key,
            }
        )


class _FakeToolContext:
    def __init__(self, *, function_call_id: str = "fc-order-1", state: dict | None = None, tool_confirmation=None):
        self.function_call_id = function_call_id
        self.state = state if state is not None else {}
        self.tool_confirmation = tool_confirmation
        self.actions = SimpleNamespace(skip_summarization=False)
        self.confirmation_request: dict[str, object] | None = None

    def request_confirmation(self, *, hint=None, payload=None) -> None:
        self.confirmation_request = {"hint": hint, "payload": payload}


def _build_fake_chat_agent(monkeypatch, repo: _ChatOrderRepo):
    from arena.agents.investment_chat import factory
    from arena.agents.investment_chat import memory as chat_memory

    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"

    _FakeExecutionMemory.instances.clear()
    monkeypatch.setattr(chat_memory, "MemoryStore", _FakeExecutionMemory, raising=False)
    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    return factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=ToolRegistry([]),
    )


def _build_raw_chat_tools(monkeypatch, repo: _ChatOrderRepo, *, settings=None, include_internal_bridge: bool = False):
    from arena.agents.investment_chat import memory as chat_memory
    from arena.agents.investment_chat.config_tools import build_config_bridge_tool_entries
    from arena.agents.investment_chat.order_tools import build_order_bridge_tool_entries
    from arena.agents.investment_chat.registry import build_chat_registry

    tool_settings = settings or load_settings()
    if settings is None:
        tool_settings.trading_mode = "paper"
    tool_settings.kis_target_market = str(getattr(tool_settings, "kis_target_market", "") or "us")

    _FakeExecutionMemory.instances.clear()
    monkeypatch.setattr(chat_memory, "MemoryStore", _FakeExecutionMemory, raising=False)
    registry = build_chat_registry(repo=repo, settings=tool_settings, tenant_id="local", registry=ToolRegistry([]))
    entries = list(registry.list_entries(require_callable=True))
    if include_internal_bridge:
        entries.extend(
            build_order_bridge_tool_entries(repo=repo, settings=tool_settings, tenant_id="local")
        )
        entries.extend(
            build_config_bridge_tool_entries(repo=repo, settings=tool_settings, tenant_id="local")
        )
    return {entry.name: entry.callable for entry in entries}
