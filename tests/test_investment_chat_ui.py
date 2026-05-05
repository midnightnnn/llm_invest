from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from arena.config import load_settings
from arena.models import AccountSnapshot, ExecutionReport, ExecutionStatus, Position
from arena.tools.registry import ToolEntry, ToolRegistry
from arena.ui.layout import tailwind_layout
from arena.ui.server import _build_app
from tests.direct_route_client import DirectRouteClient
from tests.test_ui_admin_routes import _DummyRepo


def test_investment_chat_factory_delegates_tool_implementations() -> None:
    import inspect

    from arena.agents.investment_chat import factory

    source = inspect.getsource(factory)

    assert "def get_account_snapshot(" not in source
    assert "def validate_order_draft(" not in source
    assert "def submit_approved_order(" not in source


def test_default_layout_places_investment_chat_under_memory_nav() -> None:
    html = tailwind_layout("Board", "<div>body</div>", active="investment_chat")

    assert "/investment-chat" in html
    assert "투자챗봇" in html
    assert 'href="/investment-chat" class="sidebar-link active"' in html
    assert html.index("기억관리") < html.index("투자챗봇")
    assert "bottom_nav_links" not in html


def test_layout_preserves_tenant_in_investment_chat_nav() -> None:
    html = tailwind_layout("Board", "<div>body</div>", active="board", tenant="MidNightNnN")

    assert 'href="/investment-chat?tenant_id=midnightnnn"' in html


def test_build_investment_chat_agent_filters_write_tools(monkeypatch) -> None:
    from arena.agents.investment_chat import factory

    settings = load_settings()
    repo = _DummyRepo()
    registry = ToolRegistry(
        [
            ToolEntry(
                tool_id="recommend_opportunities",
                name="recommend_opportunities",
                description="read tool",
                category="quant",
                callable=lambda top_n=8: {"top_n": top_n},
            ),
            ToolEntry(
                tool_id="execute_order",
                name="execute_order",
                description="write tool",
                category="execution",
                callable=lambda: {"submitted": True},
            ),
            ToolEntry(
                tool_id="screen_market",
                name="screen_market",
                description="safe diagnostic read tool",
                category="quant",
                callable=lambda bucket="momentum": {"bucket": bucket},
            ),
            ToolEntry(
                tool_id="scratch_run_python",
                name="scratch_run_python",
                description="scratch tool",
                category="analysis",
                callable=lambda code="": {"code": code},
            ),
        ]
    )

    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=registry,
    )

    tool_names = {getattr(tool, "__name__", "") for tool in agent.tools}
    assert "recommend_opportunities" in tool_names
    assert "screen_market" in tool_names
    assert "get_account_snapshot" in tool_names
    assert "get_agent_sleeve_snapshot" in tool_names
    assert "get_trade_history" in tool_names
    assert "get_order_approval_status" in tool_names
    assert "submit_order_with_confirmation" in tool_names
    assert "validate_order_draft" in tool_names
    assert "refresh_account_snapshot" in tool_names
    assert "submit_approved_order" not in tool_names
    assert "execute_order" not in tool_names
    assert "submit_order" not in tool_names
    assert "scratch_run_python" not in tool_names
    assert "live" not in agent.instruction.lower()


def test_investment_chat_builds_analysis_tools_with_total_account_market_scope(monkeypatch) -> None:
    from arena.agents.investment_chat import factory
    from arena.agents.investment_chat import registry as chat_registry

    settings = load_settings()
    settings.kis_target_market = "us"
    repo = _ChatOrderRepo()
    repo.set_config("local", "investment_chat_account_markets", "us,kospi", "tester")
    captured: dict[str, object] = {}

    def fake_default_registry(repo, settings, *, tenant_id="local"):
        _ = repo, tenant_id
        captured["kis_target_market"] = settings.kis_target_market
        return ToolRegistry([])

    monkeypatch.setattr(chat_registry, "build_default_registry", fake_default_registry)
    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=None,
    )

    assert captured["kis_target_market"] == "us,kospi"


def test_build_investment_chat_agent_injects_tool_memory_for_request_tenant(monkeypatch) -> None:
    from arena.agents.investment_chat import factory
    from arena.memory.query_builders import MemoryQuerySpec

    settings = load_settings()
    repo = _ChatOrderRepo()
    registry = ToolRegistry(
        [
            ToolEntry(
                tool_id="recommend_opportunities",
                name="recommend_opportunities",
                description="read tool",
                category="quant",
                callable=lambda top_n=8: {"top_n": top_n},
            )
        ]
    )

    class _VectorStore:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def search_similar_memories(self, **kwargs):
            self.calls.append(dict(kwargs))
            return [
                {
                    "event_id": "mem-chat-1",
                    "summary": "AAPL 급등 후 추격매수보다 분할 접근이 나았다.",
                    "importance_score": 0.82,
                }
            ]

    class _MemoryStore:
        instances: list["_MemoryStore"] = []

        def __init__(self, *, repo, trading_mode, memory_policy):
            self.repo = repo
            self.trading_mode = trading_mode
            self.memory_policy = memory_policy
            self.vector_store = _VectorStore()
            self.__class__.instances.append(self)

        def _tenant(self) -> str:
            return self.repo.resolve_tenant_id()

    captured: dict[str, object] = {}

    def fake_build_tool_wrapper(
        entry,
        *,
        settings,
        agent_id,
        tool_events,
        update_candidate_ledger,
        search_tool_memories,
        apply_tool_schema_metadata,
    ):
        _ = settings, tool_events, update_candidate_ledger, apply_tool_schema_metadata
        captured["agent_id"] = agent_id

        def wrapped():
            return search_tool_memories(
                MemoryQuerySpec(
                    tool_name="recommend_opportunities",
                    key_type="ticker",
                    keys=("AAPL",),
                    query="AAPL opportunity",
                )
            )

        wrapped.__name__ = str(entry.name)
        return wrapped

    monkeypatch.setattr(factory, "MemoryStore", _MemoryStore, raising=False)
    monkeypatch.setattr(factory, "build_tool_wrapper", fake_build_tool_wrapper)
    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="czxnms",
        registry=registry,
    )

    memories = agent.tools[0]()

    assert captured["agent_id"] == "investment_chat"
    assert memories == [
        {
            "summary": "AAPL 급등 후 추격매수보다 분할 접근이 나았다.",
            "importance_score": 0.82,
        }
    ]
    store = _MemoryStore.instances[0]
    assert store.vector_store.calls[0]["agent_id"] == "investment_chat"
    assert store.vector_store.calls[0]["tenant_id"] == "czxnms"
    assert repo.resolve_tenant_id() == "local"


def test_build_investment_chat_agent_uses_stored_chat_agent_config(monkeypatch) -> None:
    from arena.agents.investment_chat import factory

    settings = load_settings()
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "investment_chat_config",
        json.dumps(
            {
                "provider": "gpt",
                "model": "gpt-5.5",
                "llm_params": {"reasoning_effort": "high", "verbosity": "low"},
            }
        ),
        "seed",
    )
    captured: dict[str, object] = {}

    def fake_resolve_model(provider, settings, *, model_override="", llm_params=None):
        captured["provider"] = provider
        captured["model_override"] = model_override
        captured["llm_params"] = dict(llm_params or {})
        return "fake-model"

    monkeypatch.setattr(factory, "_resolve_model", fake_resolve_model)
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=ToolRegistry([]),
    )

    assert captured["provider"] == "gpt"
    assert captured["model_override"] == "gpt-5.5"
    assert captured["llm_params"] == {"reasoning_effort": "high", "verbosity": "low"}


def test_build_investment_chat_agent_applies_stored_chat_tool_filter(monkeypatch) -> None:
    from arena.agents.investment_chat import factory

    settings = load_settings()
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "investment_chat_config",
        json.dumps({"disabled_tools": ["recommend_opportunities"]}),
        "seed",
    )
    registry = ToolRegistry(
        [
            ToolEntry(
                tool_id="recommend_opportunities",
                name="recommend_opportunities",
                description="read tool",
                category="quant",
                callable=lambda top_n=8: {"top_n": top_n},
            )
        ]
    )

    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=registry,
    )

    tool_names = {getattr(tool, "__name__", "") for tool in agent.tools}
    assert "recommend_opportunities" not in tool_names
    assert "get_account_snapshot" in tool_names


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


def test_investment_chat_wrapped_adk_confirmation_tool_builds_declaration(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "submit_order_with_confirmation")

    declaration = FunctionTool(tool)._get_declaration()

    assert declaration is not None
    assert declaration.name == "submit_order_with_confirmation"
    assert "tool_context" not in json.dumps(declaration.model_dump(), default=str)


def test_chat_order_tool_schema_describes_ontology_friendly_rationale(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "submit_order_with_confirmation")

    declaration = FunctionTool(tool)._get_declaration()
    dumped = json.dumps(declaration.model_dump(), ensure_ascii=False, default=str)

    assert "ontology-friendly investment memo" in dumped
    assert "explicit ticker names" in dumped


def test_chat_order_tool_schema_preserves_required_fields_and_enums(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "submit_order_with_confirmation")

    declaration = FunctionTool(tool)._get_declaration()
    params = declaration.parameters.model_dump(mode="json", exclude_none=True)
    props = params["properties"]

    assert set(params["required"]) >= {"ticker", "side", "quantity", "price_krw", "rationale"}
    assert props["side"]["enum"] == ["BUY", "SELL"]
    assert props["scope"]["enum"] == ["account", "agent_sleeve"]
    assert props["price_native"]["type"] == "NUMBER"
    assert props["price_native"]["nullable"] is True


def test_chat_config_tools_expose_structured_schema(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    repo = _ChatOrderRepo()
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {candidate.__name__: candidate for candidate in agent.tools}

    assert "propose_config_change" not in tools
    assert {
        "propose_agent_config_change",
        "propose_chat_agent_config_change",
        "propose_tenant_config_change",
    }.issubset(tools)

    declaration = FunctionTool(tools["propose_agent_config_change"])._get_declaration()
    params = declaration.parameters.model_dump(mode="json", exclude_none=True)
    props = params["properties"]

    assert "change_json" not in props
    assert "agent_id" in params["required"]
    assert props["action"]["enum"] == ["update", "upsert", "add", "remove"]
    assert props["capital_allocation_mode"]["enum"] == ["", "fixed_krw", "account_percent", "whole_account"]


def test_chat_analysis_tool_schema_keeps_required_fields_with_optional_params(monkeypatch) -> None:
    from google.adk.tools.function_tool import FunctionTool

    from arena.agents.investment_chat import factory
    from arena.agents.investment_chat import memory as chat_memory

    settings = load_settings()
    settings.trading_mode = "paper"
    settings.kis_target_market = "us"
    repo = _ChatOrderRepo()

    _FakeExecutionMemory.instances.clear()
    monkeypatch.setattr(chat_memory, "MemoryStore", _FakeExecutionMemory, raising=False)
    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))
    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="local",
        registry=None,
    )
    tool = next(candidate for candidate in agent.tools if candidate.__name__ == "optimize_portfolio")

    declaration = FunctionTool(tool)._get_declaration()
    params = declaration.parameters.model_dump(mode="json", exclude_none=True)

    assert "tickers" in params["required"]
    assert params["properties"]["tickers"]["items"]["type"] == "STRING"
    assert params["properties"]["strategy"]["enum"] == ["sharpe", "risk_parity", "forecast"]
    assert params["properties"]["forecast_mode"]["enum"] == ["", "all", "stacked", "base", "balanced", "lgbm", "ridge", "avg"]


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


def test_chat_order_draft_does_not_block_rationale_by_phrase(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)

    result = tools["validate_order_draft"](
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="매수 근거",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
    )

    assert result["status"] == "ok"
    assert result["intent"]["rationale"] == "매수 근거"


def test_chat_order_tools_require_confirmation_and_are_idempotent(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_USER_EMAIL
    from arena.agents.investment_chat.drafts import draft_key

    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo, include_internal_bridge=True)

    user_token = REQUEST_USER_EMAIL.set("trader@example.com")
    try:
        draft = tools["validate_order_draft"](
            ticker="AAPL",
            side="BUY",
            quantity=1,
            price_krw=100_000,
            rationale="test buy",
            exchange_code="NASD",
            instrument_id="NASD:AAPL",
        )

        token = str(draft.get("approval_token") or "")
        assert draft["submission_status"] == "not_submitted"
        assert token
        assert repo.get_config("local", draft_key(token))

        blocked = tools["submit_approved_order"](approval_token=token, confirmation_text="승인")

        assert blocked["status"] == "blocked"
        assert "CONFIRM" in blocked["required_confirmation"]
        assert repo.execution_reports == []

        submitted = tools["submit_approved_order"](
            approval_token=token,
            confirmation_text=f"CONFIRM {token}",
        )
        repeated = tools["submit_approved_order"](
            approval_token=token,
            confirmation_text=f"CONFIRM {token}",
        )
    finally:
        REQUEST_USER_EMAIL.reset(user_token)

    assert submitted["status"] == "submitted"
    assert submitted["execution_report"]["status"] == ExecutionStatus.SIMULATED.value
    assert repeated["status"] == "already_submitted"
    assert len(repo.execution_reports) == 1
    assert any(row.get("action") == "chat_order_submit" for row in repo.audit_logs)
    assert {row.get("user_email") for row in repo.audit_logs} == {"trader@example.com"}

    second_draft = tools["validate_order_draft"](
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test buy",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
    )

    assert second_draft["approval_token"] != token
    assert tools["submit_approved_order"](token, f"CONFIRM {token}")["status"] == "already_submitted"


def test_chat_config_change_tool_requires_button_approval(monkeypatch) -> None:
    from arena.agents.investment_chat.drafts import config_draft_key

    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "agents_config",
        json.dumps(
            [
                {
                    "id": "gpt",
                    "provider": "gpt",
                    "model": "gpt-5.2",
                    "capital_krw": 1_000_000,
                    "target_market": "us",
                }
            ]
        ),
        "seed",
    )
    tools = _build_raw_chat_tools(monkeypatch, repo, include_internal_bridge=True)

    proposed = tools["propose_agent_config_change"](
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        capital_allocation_mode="account_percent",
        capital_allocation_percent=50,
        target_market="us",
        disabled_tools=["screen_market"],
        memory_compaction_model="gpt-5.4",
        rationale="gpt sleeve should manage half of the account",
    )

    token = str(proposed.get("approval_token") or "")
    assert proposed["status"] == "ok"
    assert proposed["approval_required"] is True
    assert token
    assert repo.get_config("local", config_draft_key(token))
    saved_before_approval = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert saved_before_approval[0]["model"] == "gpt-5.2"

    status = tools["get_config_change_status"](approval_token=token)

    assert status["status"] == "ok"
    assert status["drafts"][0]["approval_token"] == token
    assert status["drafts"][0]["submittable"] is True
    assert "gpt" in status["drafts"][0]["summary"]

    blocked = tools["apply_approved_config_change"](approval_token=token, confirmation_text="승인")

    assert blocked["status"] == "blocked"
    assert "CONFIRM" in blocked["required_confirmation"]
    assert repo.capital_sync_calls == []

    applied = tools["apply_approved_config_change"](
        approval_token=token,
        confirmation_text=f"CONFIRM {token}",
    )
    repeated = tools["apply_approved_config_change"](
        approval_token=token,
        confirmation_text=f"CONFIRM {token}",
    )

    assert applied["status"] == "applied"
    assert repeated["status"] == "already_applied"
    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert saved[0]["id"] == "gpt"
    assert saved[0]["model"] == "gpt-5.5"
    assert saved[0]["capital_krw"] == 5_000_000
    assert saved[0]["disabled_tools"] == ["screen_market"]
    assert saved[0]["memory_compaction_model"] == "gpt-5.4"
    assert repo.capital_sync_calls
    assert repo.capital_sync_calls[-1]["target_capitals"]["gpt"] == 5_000_000


def test_investment_chat_config_draft_api_lists_and_applies_pending_draft(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "agents_config",
        json.dumps([{"id": "gpt", "provider": "gpt", "model": "gpt-5.2", "capital_krw": 1_000_000}]),
        "seed",
    )
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    settings = load_settings()
    tools = _build_raw_chat_tools(monkeypatch, repo, settings=settings, include_internal_bridge=True)
    draft = tools["propose_agent_config_change"](
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        capital_krw=2_000_000,
        rationale="upgrade model and capital",
    )
    token = str(draft["approval_token"])
    client = DirectRouteClient(_build_app(repo=repo, settings=settings))

    listed = client.get("/investment-chat/config-drafts", params={"tenant_id": "local"})

    assert listed.status_code == 200
    payload = listed.json()
    assert payload["status"] == "ok"
    assert payload["drafts"][0]["approval_token"] == token
    assert payload["drafts"][0]["submittable"] is True

    applied = client.post(f"/investment-chat/config-drafts/{token}/apply", params={"tenant_id": "local"})
    repeated = client.post(f"/investment-chat/config-drafts/{token}/apply", params={"tenant_id": "local"})

    assert applied.status_code == 200
    assert applied.json()["status"] == "applied"
    assert "chat_delivery_text" in applied.json()
    assert repeated.json()["status"] == "already_applied"
    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert saved[0]["model"] == "gpt-5.5"
    assert saved[0]["capital_krw"] == 2_000_000


def test_chat_order_tool_uses_adk_confirmation_before_execution(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)
    submit_with_confirmation = tools["submit_order_with_confirmation"]

    first_context = _FakeToolContext()
    waiting = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test buy through ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=first_context,
    )

    assert waiting["status"] == "waiting_for_confirmation"
    assert waiting["submission_status"] == "not_submitted"
    assert first_context.actions.skip_summarization is True
    assert first_context.confirmation_request is not None
    payload = first_context.confirmation_request["payload"]
    assert isinstance(payload, dict)
    assert payload["ticker"] == "AAPL"
    assert "approval_token" not in payload
    assert "ADK Web 확인창" in str(first_context.confirmation_request["hint"])
    assert "Confirmed 체크박스" in str(first_context.confirmation_request["hint"])
    assert repo.execution_reports == []

    confirmed_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(confirmed=True, payload={}),
    )
    submitted = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test buy through ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=confirmed_context,
    )

    assert submitted["status"] == "submitted"
    assert submitted["execution_report"]["status"] == ExecutionStatus.SIMULATED.value
    assert len(repo.execution_reports) == 1


def test_chat_order_tool_rejects_adk_confirmation_without_execution(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)
    submit_with_confirmation = tools["submit_order_with_confirmation"]

    first_context = _FakeToolContext()
    waiting = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test rejected ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=first_context,
    )
    assert waiting["status"] == "waiting_for_confirmation"

    rejected_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(confirmed=False, payload={}),
    )
    rejected = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test rejected ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=rejected_context,
    )

    assert rejected["status"] == "rejected"
    assert repo.execution_reports == []


def test_chat_order_tool_explains_unchecked_adk_confirmation(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    tools = _build_raw_chat_tools(monkeypatch, repo)
    submit_with_confirmation = tools["submit_order_with_confirmation"]

    first_context = _FakeToolContext()
    waiting = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test unchecked ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=first_context,
    )
    assert waiting["status"] == "waiting_for_confirmation"

    unchecked_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(
            confirmed=False,
            payload={
                "ticker": "AAPL",
                "side": "BUY",
                "quantity": 1,
                "price_krw": 100_000,
            },
        ),
    )
    rejected = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test unchecked ADK confirmation",
        exchange_code="NASD",
        instrument_id="NASD:AAPL",
        tool_context=unchecked_context,
    )

    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "confirmed_checkbox_unchecked"
    assert "Confirmed 체크박스" in rejected["message"]
    assert repo.execution_reports == []


def test_get_trade_history_tool_reads_tenant_scoped_execution_history(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    repo.trade_history_rows = [
        {
            "order_id": "order-1",
            "intent_id": "intent-1",
            "created_at": datetime(2026, 4, 28, 1, 1, tzinfo=timezone.utc),
            "trading_mode": "paper",
            "agent_id": "gpt",
            "ticker": "AAPL",
            "exchange_code": "NASD",
            "instrument_id": "NASD:AAPL",
            "side": "SELL",
            "requested_qty": 2.0,
            "filled_qty": 1.0,
            "avg_price_krw": 150_000.0,
            "avg_price_native": 100.0,
            "quote_currency": "USD",
            "fx_rate": 1500.0,
            "status": "SIMULATED",
            "message": "paper fill",
            "rationale": "사용자와 투자챗봇이 과열 구간 일부 익절을 판단함",
            "risk_reason": "ok",
            "policy_hits": ["chat_confirmation"],
            "strategy_refs": ["scope:agent_sleeve", "judgment:user+investment_chat"],
        }
    ]
    tools = _build_raw_chat_tools(monkeypatch, repo)

    result = tools["get_trade_history"](
        ticker="aapl",
        agent_id="gpt",
        scope="agent_sleeve",
        days=30,
        limit=5,
    )

    assert result["status"] == "ok"
    assert result["tenant_id"] == "local"
    assert result["count"] == 1
    assert repo.trade_history_calls == [
        {
            "tenant_id": "local",
            "ticker": "AAPL",
            "agent_id": "gpt",
            "scope": "agent_sleeve",
            "days": 30,
            "limit": 5,
            "statuses": ["FILLED", "SIMULATED", "SUBMITTED"],
        }
    ]
    trade = result["trades"][0]
    assert trade["ticker"] == "AAPL"
    assert trade["scope"] == "agent_sleeve"
    assert trade["judgment_source"] == "user+investment_chat"
    assert trade["notional_krw"] == 150_000.0
    assert trade["rationale"] == "사용자와 투자챗봇이 과열 구간 일부 익절을 판단함"


def test_chat_sleeve_order_uses_sleeve_snapshot_and_syncs_target_agent_memory(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_USER_EMAIL

    repo = _ChatOrderRepo(
        account_snapshot=AccountSnapshot(cash_krw=9_000_000.0, total_equity_krw=10_000_000.0, positions={}),
        sleeve_snapshot=AccountSnapshot(
            cash_krw=1_000_000.0,
            total_equity_krw=1_260_000.0,
            usd_krw_rate=1400.0,
            positions={
                "AAPL": Position(
                    ticker="AAPL",
                    quantity=2,
                    avg_price_krw=120_000,
                    market_price_krw=130_000,
                )
            },
        ),
    )
    tools = _build_raw_chat_tools(monkeypatch, repo, include_internal_bridge=True)

    user_token = REQUEST_USER_EMAIL.set("trader@example.com")
    try:
        draft = tools["validate_order_draft"](
            ticker="AAPL",
            side="SELL",
            quantity=1,
            price_krw=130_000,
            rationale="사용자가 AAPL 비중을 낮추고 현금 여력을 확보하기로 판단함",
            scope="agent_sleeve",
            agent_id="gpt",
            exchange_code="NASD",
            instrument_id="NASD:AAPL",
        )
        token = str(draft["approval_token"])
        submitted = tools["submit_approved_order"](
            approval_token=token,
            confirmation_text=f"CONFIRM {token}",
        )
    finally:
        REQUEST_USER_EMAIL.reset(user_token)

    assert draft["status"] == "ok"
    assert draft["risk"]["allowed"] is True
    assert draft["scope"] == "agent_sleeve"
    assert draft["intent"]["agent_id"] == "gpt"
    assert submitted["status"] == "submitted"
    assert len(repo.sleeve_calls) >= 2
    assert {str(call["agent_id"]) for call in repo.sleeve_calls} == {"gpt"}
    assert len(repo.order_intents) == 1
    intent = repo.order_intents[0]["intent"]
    assert intent.agent_id == "gpt"
    assert "scope:agent_sleeve" in intent.strategy_refs
    assert "judgment:user+investment_chat" in intent.strategy_refs
    assert "approved_by:trader@example.com" in intent.strategy_refs
    assert len(_FakeExecutionMemory.instances) == 1
    memory = _FakeExecutionMemory.instances[0]
    assert [row["intent"].agent_id for row in memory.executions] == ["gpt"]
    assert [row["intent"].agent_id for row in memory.theses] == ["gpt"]
    assert [row["agent_id"] for row in memory.reflections] == ["gpt"]
    reflection = memory.reflections[0]
    assert "사용자+투자챗봇 판단" in reflection["summary"]
    assert "AAPL 비중을 낮추고" in reflection["summary"]
    assert reflection["payload"]["source"] == "investment_chat_order_decision"
    assert reflection["payload"]["judgment_source"] == "user+investment_chat"
    assert reflection["payload"]["scope"] == "agent_sleeve"
    assert reflection["payload"]["approved_by"] == "trader@example.com"


def test_chat_order_submit_blocks_live_mode_without_explicit_permission(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    settings = load_settings()
    settings.trading_mode = "live"
    settings.allow_live_trading = False
    settings.kis_target_market = "us"

    tools = _build_raw_chat_tools(monkeypatch, repo, settings=settings, include_internal_bridge=True)

    draft = tools["validate_order_draft"](
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="test buy",
    )
    token = str(draft["approval_token"])
    submitted = tools["submit_approved_order"](
        approval_token=token,
        confirmation_text=f"CONFIRM {token}",
    )

    assert submitted["status"] == "blocked"
    assert "live trading" in submitted["error"]
    assert repo.execution_reports == []


def test_refresh_account_snapshot_tool_calls_sync_service(monkeypatch) -> None:
    from arena.agents.investment_chat import account_tools

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {"kis_secret_name": "local-local-kis"}
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}
    calls: dict[str, object] = {}

    class _FakeAccountSyncService:
        def __init__(self, *, settings, repo):
            calls["settings"] = settings
            calls["repo"] = repo

        def sync_account_snapshot(self):
            calls["synced_at"] = datetime.now(timezone.utc)
            return AccountSnapshot(cash_krw=1.0, total_equity_krw=2.0, positions={})

    monkeypatch.setattr(account_tools, "AccountSyncService", _FakeAccountSyncService)

    result = tools["refresh_account_snapshot"]()

    assert result["status"] == "ok"
    assert result["total_equity_krw"] == 2.0
    assert calls["repo"] is repo


def test_refresh_account_snapshot_logs_unexpected_sync_failure(monkeypatch, caplog) -> None:
    from arena.agents.investment_chat import account_tools

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {"kis_secret_name": "local-local-kis"}
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}

    class _FailingAccountSyncService:
        def __init__(self, *, settings, repo):
            _ = settings, repo

        def sync_account_snapshot(self):
            raise RuntimeError("sync boom")

    monkeypatch.setattr(account_tools, "AccountSyncService", _FailingAccountSyncService)

    with caplog.at_level(logging.WARNING):
        result = tools["refresh_account_snapshot"]()

    assert result["status"] == "error"
    failure_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", "") == "chat_account_refresh_failed"
    )
    assert failure_record.exc_info is not None


def test_refresh_account_snapshot_defaults_to_total_account_markets(monkeypatch) -> None:
    from arena.agents.investment_chat import account_tools

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {"kis_secret_name": "local-local-kis"}
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}
    calls: dict[str, object] = {}

    class _FakeAccountSyncService:
        def __init__(self, *, settings, repo):
            calls["market"] = settings.kis_target_market
            calls["repo"] = repo

        def sync_account_snapshot(self):
            return AccountSnapshot(cash_krw=1.0, total_equity_krw=2.0, positions={})

    monkeypatch.setattr(account_tools, "AccountSyncService", _FakeAccountSyncService)

    result = tools["refresh_account_snapshot"]()

    assert result["status"] == "ok"
    assert calls["market"] == "us,kospi"
    assert repo.audit_logs[-1]["detail"]["target_market"] == "us,kospi"


def test_refresh_account_snapshot_uses_chat_account_market_override(monkeypatch) -> None:
    from arena.agents.investment_chat import account_tools

    repo = _ChatOrderRepo()
    repo.runtime_credentials["local"] = {"kis_secret_name": "local-local-kis"}
    repo.set_config("local", "investment_chat_account_markets", "us,kospi", "tester")
    agent = _build_fake_chat_agent(monkeypatch, repo)
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}
    calls: dict[str, object] = {}

    class _FakeAccountSyncService:
        def __init__(self, *, settings, repo):
            calls["market"] = settings.kis_target_market
            calls["repo"] = repo

        def sync_account_snapshot(self):
            return AccountSnapshot(cash_krw=1.0, total_equity_krw=2.0, positions={})

    monkeypatch.setattr(account_tools, "AccountSyncService", _FakeAccountSyncService)

    result = tools["refresh_account_snapshot"]()

    assert result["status"] == "ok"
    assert calls["market"] == "us,kospi"
    assert repo.audit_logs[-1]["detail"]["target_market"] == "us,kospi"


def test_refresh_account_snapshot_blocks_server_fallback_credentials(monkeypatch) -> None:
    from arena.agents.investment_chat import account_tools, factory

    repo = _ChatOrderRepo()
    settings = load_settings()
    settings.kis_secret_name = "KISAPI"
    settings.kis_api_key = "server-default-key"
    settings.kis_api_secret = "server-default-secret"
    settings.kis_account_no = "1234567890"
    settings.kis_target_market = "us"

    monkeypatch.setattr(factory, "_resolve_model", lambda *args, **kwargs: "fake-model")
    monkeypatch.setattr(factory, "Agent", lambda **kwargs: SimpleNamespace(**kwargs))

    calls: dict[str, object] = {}

    class _FakeAccountSyncService:
        def __init__(self, *, settings, repo):
            calls["settings"] = settings
            calls["repo"] = repo

        def sync_account_snapshot(self):
            calls["synced"] = True
            return AccountSnapshot(cash_krw=1.0, total_equity_krw=2.0, positions={})

    monkeypatch.setattr(account_tools, "AccountSyncService", _FakeAccountSyncService)

    agent = factory.build_investment_chat_agent(
        repo=repo,
        settings=settings,
        tenant_id="czxnms",
        registry=ToolRegistry([]),
    )
    tools = {getattr(tool, "__name__", ""): tool for tool in agent.tools}

    result = tools["refresh_account_snapshot"]()

    assert result["status"] == "blocked"
    assert result["tenant_id"] == "czxnms"
    assert "tenant KIS credentials" in result["error"]
    assert "synced" not in calls


def test_investment_chat_loader_binds_default_tenant(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}
    settings = load_settings()

    def fake_build_agent(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(name="investment_chat", description="chat")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)

    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="MidNightNnN",
    )

    agent = loader.load_agent("investment_chat")

    assert agent.name == "investment_chat"
    assert loader.list_agents() == ["investment_chat__midnightnnn__gemini__m_Z2VtaW5pLTMtZmxhc2gtcHJldmlldw"]
    assert calls["tenant_id"] == "midnightnnn"
    assert calls["settings"] is settings
    assert calls["registry"] is None

    loader.load_agent("investment_chat__research")
    assert calls["tenant_id"] == "research"


def test_investment_chat_loader_separates_model_selection(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_MODEL, REQUEST_PROVIDER
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}
    settings = load_settings()

    def fake_build_agent(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(name="investment_chat", description="chat")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )

    provider_token = REQUEST_PROVIDER.set("gpt")
    model_token = REQUEST_MODEL.set("gpt-5.5")
    try:
        listed = loader.list_agents()
        loader.load_agent(listed[0])
    finally:
        REQUEST_MODEL.reset(model_token)
        REQUEST_PROVIDER.reset(provider_token)

    assert listed == ["investment_chat__local__gpt__m_Z3B0LTUuNQ"]
    assert calls["tenant_id"] == "local"
    assert calls["provider"] == "gpt"
    assert calls["model_override"] == "gpt-5.5"

    calls.clear()
    second_loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )
    second_loader.load_agent(listed[0])

    assert calls["provider"] == "gpt"
    assert calls["model_override"] == "gpt-5.5"

    calls.clear()
    gemini_provider = REQUEST_PROVIDER.set("gemini")
    gemini_model = REQUEST_MODEL.set("gemini-3.1-pro-preview")
    try:
        gemini_listed = second_loader.list_agents()
    finally:
        REQUEST_MODEL.reset(gemini_model)
        REQUEST_PROVIDER.reset(gemini_provider)
    third_loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )
    third_loader.load_agent(gemini_listed[0])

    assert gemini_listed == ["investment_chat__local__gemini__m_Z2VtaW5pLTMuMS1wcm8tcHJldmlldw"]
    assert calls["provider"] == "gemini"
    assert calls["model_override"] == "gemini-3.1-pro-preview"


def test_investment_chat_loader_request_selection_overrides_stale_adk_app_name(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_MODEL, REQUEST_PROVIDER, REQUEST_TENANT
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}
    settings = load_settings()

    def fake_build_agent(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(name="investment_chat", description="chat")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )

    tenant_token = REQUEST_TENANT.set("czxnms")
    provider_token = REQUEST_PROVIDER.set("gemini")
    model_token = REQUEST_MODEL.set("gemini-3-flash-preview")
    try:
        stale_app_name = "investment_chat__midnightnnn__gpt__m_Z3B0LTUuMg"
        loader.load_agent(stale_app_name)
    finally:
        REQUEST_MODEL.reset(model_token)
        REQUEST_PROVIDER.reset(provider_token)
        REQUEST_TENANT.reset(tenant_token)

    assert calls["tenant_id"] == "czxnms"
    assert calls["provider"] == "gemini"
    assert calls["model_override"] == "gemini-3-flash-preview"


def test_investment_chat_loader_rebuilds_after_settings_fingerprint_changes(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    builds: list[dict[str, object]] = []
    settings = load_settings()
    settings.openai_api_key = "old-key"

    def fake_build_agent(**kwargs):
        builds.append(dict(kwargs))
        return SimpleNamespace(name=f"agent-{len(builds)}")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )
    app_name = "investment_chat__local__gpt__m_Z3B0LTUuNQ"

    first = loader.load_agent(app_name)
    second = loader.load_agent(app_name)
    settings.openai_api_key = "new-key"
    third = loader.load_agent(app_name)

    assert first is second
    assert third is not first
    assert [item["model_override"] for item in builds] == ["gpt-5.5", "gpt-5.5"]


def test_investment_chat_loader_uses_encoded_claude_selection_without_request_context(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}
    settings = load_settings()

    def fake_build_agent(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(name="investment_chat", description="chat")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )

    claude_app_name = "investment_chat__local__claude__m_Y2xhdWRlLXNvbm5ldC00LTY"
    loader.load_agent(claude_app_name)

    assert calls["provider"] == "claude"
    assert calls["model_override"] == "claude-sonnet-4-6"


def test_ui_registers_investment_chat_page_and_adk_mount(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )

    app = _build_app(repo=_DummyRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/investment-chat", params={"tenant_id": "local", "provider": "gpt", "model": "gpt-5.2"})

    assert response.status_code == 200
    assert "투자챗봇" in response.text
    assert ">투자챗봇</h2>" not in response.text
    assert "/investment-chat/adk/dev-ui/?tenant_id=local&amp;provider=gpt&amp;model=gpt-5.2" in response.text
    assert 'id="sidebar-collapse-toggle"' in response.text
    assert response.text.index('id="arena-sidebar"') < response.text.index('id="sidebar-collapse-toggle"') < response.text.index("<nav")
    collapsed_rule = response.text.split("body.sidebar-collapsed .arena-sidebar {", 1)[1].split("}", 1)[0]
    assert "margin-left" not in collapsed_rule
    assert "width: var(--sidebar-collapsed-w)" in collapsed_rule
    assert "flex-basis: var(--sidebar-collapsed-w)" in collapsed_rule
    assert "transform:" not in collapsed_rule
    assert "body.sidebar-collapsed #arena-sidebar nav" in response.text
    assert "body.sidebar-collapsed .sidebar-footer" in response.text
    assert client.session["investment_chat_tenant_id"] == "local"
    assert client.session["investment_chat_provider"] == "gpt"
    assert client.session["investment_chat_model"] == "gpt-5.2"
    assert 'name="provider"' in response.text
    assert "data-chat-selector-form" in response.text
    assert '<select\n      name="model"' in response.text
    assert "requestSubmit" in response.text
    assert "model.addEventListener('change', submitSelection)" in response.text
    assert "submitResultMessage" in response.text
    assert "execution_report.message" in response.text
    assert "deliverOrderResultToChat" in response.text
    assert "Config Approval" in response.text
    assert "data-config-draft-panel" in response.text
    assert "/investment-chat/config-drafts" in response.text
    assert "data-config-draft-apply" in response.text
    assert "isAdkChatBusy" in response.text
    assert "mat-progress-bar" in response.text
    assert "textarea.chat-input-box" in response.text
    assert "button.send-message-btn" in response.text
    assert 'list="investment-chat-model-presets"' not in response.text
    assert 'value="gpt-5.2"' in response.text
    assert "gpt-5.5" in response.text
    assert "gemini-3.1-flash-preview" in response.text
    assert "gemini-3.1-pro-preview" in response.text
    assert 'data-active="investment_chat"' in response.text
    assert "investment-chat-shell" in response.text
    assert "calc(100dvh - var(--mobile-topbar-h))" in response.text
    assert "investment-chat-frame" in response.text
    assert 'body[data-active="investment_chat"] .sidebar-backdrop.open' in response.text
    assert any(getattr(route, "path", "") == "/investment-chat/adk" for route in app.routes)


def test_investment_chat_model_select_renders_all_provider_presets(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )

    app = _build_app(repo=_DummyRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get(
        "/investment-chat",
        params={"tenant_id": "local", "provider": "claude", "model": "claude-sonnet-4-6"},
    )

    assert response.status_code == 200
    assert '<select\n      name="model"' in response.text
    assert '<option value="claude-sonnet-4-6" selected>claude-sonnet-4-6</option>' in response.text
    assert 'value="claude-opus-4-7"' in response.text
    assert 'value="claude-opus-4-5"' in response.text
    assert 'value="claude-sonnet-4-5"' in response.text


def test_investment_chat_provider_options_come_from_adk_provider_registry(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )

    app = _build_app(repo=_DummyRepo(), settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get(
        "/investment-chat",
        params={"tenant_id": "local", "provider": "deepseek", "model": "deepseek-reasoner"},
    )

    assert response.status_code == 200
    assert '<option value="deepseek" selected>DeepSeek</option>' in response.text
    assert '<option value="gpt"' in response.text
    assert '<option value="gemini"' in response.text
    assert '<option value="claude"' in response.text
    assert client.session["investment_chat_provider"] == "deepseek"
    assert client.session["investment_chat_model"] == "deepseek-reasoner"


def test_investment_chat_page_defaults_to_stored_chat_agent_config(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _DummyRepo()
    repo.set_config(
        "local",
        "investment_chat_config",
        json.dumps({"provider": "claude", "model": "claude-opus-4-7"}),
        "seed",
    )
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/investment-chat", params={"tenant_id": "local"})

    assert response.status_code == 200
    assert '<option value="claude" selected>Anthropic Claude</option>' in response.text
    assert '<option value="claude-opus-4-7" selected>claude-opus-4-7</option>' in response.text
    assert "/investment-chat/adk/dev-ui/?tenant_id=local&amp;provider=claude&amp;model=claude-opus-4-7" in response.text
    assert client.session["investment_chat_provider"] == "claude"
    assert client.session["investment_chat_model"] == "claude-opus-4-7"


def test_investment_chat_provider_options_are_limited_to_tenant_model_keys(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _DummyRepo()
    repo.runtime_credentials["czxnms"] = {
        "tenant_id": "czxnms",
        "model_secret_name": "local-czxnms-models",
        "has_openai": False,
        "has_gemini": False,
        "has_anthropic": True,
    }
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get(
        "/investment-chat",
        params={"tenant_id": "czxnms", "provider": "gemini", "model": "gemini-3-flash-preview"},
    )

    assert response.status_code == 200
    assert '<option value="claude" selected>Anthropic Claude</option>' in response.text
    assert '<option value="gemini"' not in response.text
    assert '<option value="gpt"' not in response.text
    assert "/investment-chat/adk/dev-ui/?tenant_id=czxnms&amp;provider=claude" in response.text
    assert "gemini-3-flash-preview" not in response.text
    assert client.session["investment_chat_provider"] == "claude"
    assert client.session["investment_chat_model"].startswith("claude-")


def test_investment_chat_provider_options_show_no_iframe_when_no_tenant_model_keys(monkeypatch) -> None:
    import arena.ui.app as ui_app

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    repo = _DummyRepo()
    repo.runtime_credentials["czxnms"] = {
        "tenant_id": "czxnms",
        "model_secret_name": "local-czxnms-models",
        "has_openai": False,
        "has_gemini": False,
        "has_anthropic": False,
    }
    app = _build_app(repo=repo, settings=load_settings())
    client = DirectRouteClient(app)

    response = client.get("/investment-chat", params={"tenant_id": "czxnms"})

    assert response.status_code == 200
    assert "등록된 LLM API key가 없습니다" in response.text
    assert "<iframe" not in response.text


def test_investment_chat_loader_restricts_selection_to_tenant_model_keys(monkeypatch) -> None:
    from arena.agents.investment_chat.context import REQUEST_MODEL, REQUEST_PROVIDER
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}
    settings = load_settings()
    repo = _DummyRepo()
    repo.runtime_credentials["czxnms"] = {
        "tenant_id": "czxnms",
        "model_secret_name": "local-czxnms-models",
        "has_openai": False,
        "has_gemini": False,
        "has_anthropic": True,
    }

    def fake_build_agent(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(name="investment_chat", description="chat")

    monkeypatch.setattr(investment_chat_adk, "build_investment_chat_agent", fake_build_agent)
    loader = investment_chat_adk.InvestmentChatAgentLoader(
        repo=repo,
        settings_for_tenant=lambda tenant: settings,
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="czxnms",
    )

    provider_token = REQUEST_PROVIDER.set("gemini")
    model_token = REQUEST_MODEL.set("gemini-3-flash-preview")
    try:
        listed = loader.list_agents()
        loader.load_agent(listed[0])
    finally:
        REQUEST_MODEL.reset(model_token)
        REQUEST_PROVIDER.reset(provider_token)

    assert listed[0].startswith("investment_chat__czxnms__claude__m_")
    assert calls["provider"] == "claude"
    assert str(calls["model_override"]).startswith("claude-")


def test_investment_chat_order_draft_api_lists_and_submits_pending_draft(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    validate = registry.get("validate_order_draft").callable

    draft = validate(ticker="AAPL", side="BUY", quantity=1, price_krw=100_000, rationale="button approval test")
    token = str(draft["approval_token"])
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)

    listed = client.get("/investment-chat/order-drafts", params={"tenant_id": "local"})

    assert listed.status_code == 200
    payload = listed.json()
    assert payload["drafts"][0]["approval_token"] == token
    assert payload["drafts"][0]["submittable"] is True
    assert payload["drafts"][0]["intent"]["ticker"] == "AAPL"
    assert "required_confirmation" not in payload["drafts"][0]

    submitted = client.post(f"/investment-chat/order-drafts/{token}/submit", params={"tenant_id": "local"})

    assert submitted.status_code == 200
    result = submitted.json()
    assert result["status"] == "submitted"
    assert len(repo.execution_reports) == 1


def test_investment_chat_order_draft_api_hides_adk_confirmation_drafts(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    submit_with_confirmation = registry.get("submit_order_with_confirmation").callable

    waiting = submit_with_confirmation(
        ticker="AAPL",
        side="BUY",
        quantity=1,
        price_krw=100_000,
        rationale="ADK confirmation should not create host approval card",
        tool_context=_FakeToolContext(),
    )
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)

    listed = client.get("/investment-chat/order-drafts", params={"tenant_id": "local"})

    assert waiting["status"] == "waiting_for_confirmation"
    assert listed.status_code == 200
    assert listed.json()["drafts"] == []


def test_investment_chat_order_draft_api_surfaces_broker_error_message(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())

    class _RejectingBroker:
        def place_order(self, intent, *, fx_rate=None):
            _ = intent, fx_rate
            return ExecutionReport(
                status=ExecutionStatus.ERROR,
                order_id="err_holiday",
                filled_qty=0,
                avg_price_krw=0,
                message="금일은 해외 휴장일로 주문이 불가합니다.",
            )

    monkeypatch.setattr(order_tools, "PaperBroker", _RejectingBroker)
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    validate = registry.get("validate_order_draft").callable

    draft = validate(ticker="AAPL", side="BUY", quantity=1, price_krw=100_000, rationale="button approval test")
    token = str(draft["approval_token"])
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)

    submitted = client.post(f"/investment-chat/order-drafts/{token}/submit", params={"tenant_id": "local"})

    assert submitted.status_code == 200
    result = submitted.json()
    assert result["status"] == "error"
    assert result["message"] == "금일은 해외 휴장일로 주문이 불가합니다."
    assert result["error"] == "금일은 해외 휴장일로 주문이 불가합니다."
    assert result["execution_report"]["message"] == "금일은 해외 휴장일로 주문이 불가합니다."
    assert result["chat_delivery_text"] == "방금 AAPL BUY 1주 주문 승인 결과를 확인해서 알려줘."
    assert "[주문 승인 패널 결과]" not in result["chat_delivery_text"]
    assert "금일은 해외 휴장일로 주문이 불가합니다." not in result["chat_delivery_text"]
    assert "/uapi/" not in result["chat_delivery_text"]
    assert token not in result["chat_delivery_text"]
    assert "CONFIRM" not in result["chat_delivery_text"]


def test_get_order_approval_status_reads_latest_button_result(monkeypatch) -> None:
    import arena.ui.app as ui_app
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr(
        ui_app,
        "build_investment_chat_adk_app",
        lambda **kwargs: FastAPI(title="stub-adk"),
    )
    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())

    class _RejectingBroker:
        def place_order(self, intent, *, fx_rate=None):
            _ = intent, fx_rate
            return ExecutionReport(
                status=ExecutionStatus.ERROR,
                order_id="err_holiday",
                filled_qty=0,
                avg_price_krw=0,
                message="금일은 해외 휴장일로 주문이 불가합니다.",
            )

    monkeypatch.setattr(order_tools, "PaperBroker", _RejectingBroker)
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    validate = registry.get("validate_order_draft").callable
    status_tool = registry.get("get_order_approval_status").callable

    draft = validate(ticker="AAPL", side="BUY", quantity=1, price_krw=100_000, rationale="button approval test")
    token = str(draft["approval_token"])
    app = _build_app(repo=repo, settings=settings)
    client = DirectRouteClient(app)
    client.post(f"/investment-chat/order-drafts/{token}/submit", params={"tenant_id": "local"})

    status = status_tool(approval_token=token)

    assert status["status"] == "ok"
    assert status["orders"][0]["status"] == "error"
    assert status["orders"][0]["ticker"] == "AAPL"
    assert status["orders"][0]["message"] == "금일은 해외 휴장일로 주문이 불가합니다."


def test_get_order_approval_status_reads_latest_button_result_from_detail_json(monkeypatch) -> None:
    from arena.agents.investment_chat import order_tools
    from arena.agents.investment_chat.order_tools import build_order_bridge_tool_entries
    from arena.agents.investment_chat.registry import build_chat_registry

    monkeypatch.setattr(order_tools, "build_execution_memory", lambda repo, settings: _FakeExecutionMemory())

    class _RejectingBroker:
        def place_order(self, intent, *, fx_rate=None):
            _ = intent, fx_rate
            return ExecutionReport(
                status=ExecutionStatus.ERROR,
                order_id="err_holiday",
                filled_qty=0,
                avg_price_krw=0,
                message="금일은 해외 휴장일로 주문이 불가합니다.",
            )

    monkeypatch.setattr(order_tools, "PaperBroker", _RejectingBroker)
    settings = load_settings()
    settings.trading_mode = "paper"
    repo = _ChatOrderRepo()
    repo.recent_runtime_audit_logs = lambda limit=50: list(reversed(repo.audit_logs[-limit:]))  # type: ignore[method-assign]
    registry = build_chat_registry(repo=repo, settings=settings, tenant_id="local", registry=None)
    validate = registry.get("validate_order_draft").callable
    assert registry.get("submit_approved_order") is None
    bridge_entries = build_order_bridge_tool_entries(repo=repo, settings=settings, tenant_id="local")
    submit = {entry.name: entry.callable for entry in bridge_entries}["submit_approved_order"]
    status_tool = registry.get("get_order_approval_status").callable

    draft = validate(ticker="AAPL", side="BUY", quantity=1, price_krw=100_000, rationale="button approval test")
    token = str(draft["approval_token"])
    submit(approval_token=token, confirmation_text=f"CONFIRM {token}")
    for row in repo.audit_logs:
        detail = row.pop("detail", None)
        if detail is not None:
            row["detail_json"] = json.dumps(detail)

    status = status_tool()

    assert status["status"] == "ok"
    assert status["count"] == 1
    assert status["orders"][0]["approval_token"] == token
    assert status["orders"][0]["status"] == "error"
    assert status["orders"][0]["message"] == "금일은 해외 휴장일로 주문이 불가합니다."


def test_investment_chat_adk_api_requires_ui_auth(monkeypatch) -> None:
    import asyncio

    from arena.ui import investment_chat_adk

    def fake_get_fast_api_app(**kwargs):
        _ = kwargs
        app = FastAPI(title="fake-adk")

        @app.get("/list-apps")
        def list_apps():
            return ["investment_chat"]

        return app

    monkeypatch.setattr(investment_chat_adk, "get_fast_api_app", fake_get_fast_api_app)
    monkeypatch.setattr(investment_chat_adk, "_mount_adk_static", lambda app, url_prefix: None)

    blocked_app = investment_chat_adk.build_investment_chat_adk_app(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: load_settings(),
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
        auth_enabled=True,
        current_user=lambda request: None,
    )
    allowed_app = investment_chat_adk.build_investment_chat_adk_app(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: load_settings(),
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
        auth_enabled=True,
        current_user=lambda request: {"email": "user@example.com"},
    )

    def make_request(headers: dict[str, str] | None = None) -> Request:
        raw_headers = [
            (str(key).lower().encode("latin-1"), str(value).encode("latin-1"))
            for key, value in (headers or {}).items()
        ]
        return Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/list-apps",
                "raw_path": b"/list-apps",
                "query_string": b"",
                "headers": raw_headers,
                "client": ("testclient", 50000),
                "server": ("testserver", 80),
                "session": {},
            },
            receive=lambda: {"type": "http.request", "body": b"", "more_body": False},
        )

    async def passthrough(_request: Request):
        return JSONResponse(["investment_chat"])

    blocked_dispatch = blocked_app.user_middleware[0].kwargs["dispatch"]
    allowed_dispatch = allowed_app.user_middleware[0].kwargs["dispatch"]
    blocked_response = asyncio.run(blocked_dispatch(make_request({"accept": "application/json"}), passthrough))
    allowed_response = asyncio.run(allowed_dispatch(make_request(), passthrough))

    assert blocked_response.status_code == 401
    assert allowed_response.status_code == 200
    assert json.loads(allowed_response.body) == ["investment_chat"]


def test_investment_chat_adk_rejects_stale_path_app_name_tenant() -> None:
    import asyncio

    from arena.ui import investment_chat_adk

    stale_app_name = investment_chat_adk._chat_app_name("midnightnnn", "gpt", "gpt-5.5")
    request = SimpleNamespace(url=SimpleNamespace(path=f"/apps/{stale_app_name}/app-info"))

    response = asyncio.run(
        investment_chat_adk._stale_app_name_response(
            request,
            tenant="czxnms",
            provider="gemini",
            model="gemini-3-flash-preview",
        )
    )

    assert response is not None
    assert response.status_code == 409
    payload = json.loads(response.body)
    assert payload["error"] == "stale adk app_name tenant"
    assert payload["tenant_id"] == "czxnms"
    assert payload["app_name_tenant"] == "midnightnnn"


def test_investment_chat_adk_defaults_to_data_sqlite_sessions(monkeypatch) -> None:
    from arena.ui import investment_chat_adk

    calls: dict[str, object] = {}

    def fake_get_fast_api_app(**kwargs):
        calls.update(kwargs)
        return FastAPI(title="fake-adk")

    monkeypatch.delenv("ARENA_CHAT_SESSION_SERVICE_URI", raising=False)
    monkeypatch.setattr(investment_chat_adk, "get_fast_api_app", fake_get_fast_api_app)
    monkeypatch.setattr(investment_chat_adk, "_mount_adk_static", lambda app, url_prefix: None)

    investment_chat_adk.build_investment_chat_adk_app(
        repo=_DummyRepo(),
        settings_for_tenant=lambda tenant: load_settings(),
        get_default_registry=lambda tenant: ToolRegistry([]),
        default_tenant="local",
    )

    expected = investment_chat_adk.Path(investment_chat_adk.__file__).resolve().parents[2]
    expected = expected / "data" / "arena-investment-chat-adk-sessions.sqlite"
    assert calls["session_service_uri"] == f"sqlite:///{expected}"
