from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

from arena.config import load_settings
from arena.models import AccountSnapshot
from arena.tools.registry import ToolRegistry
from tests.ui.investment_chat_helpers import _ChatOrderRepo, _build_fake_chat_agent, _chat_advisor_agent


def test_investment_chat_account_tools_expose_available_agent_ids() -> None:
    from arena.agents.investment_chat.account_tools import build_account_tool_entries

    settings = load_settings()
    repo = _ChatOrderRepo()
    tools = {
        entry.name: entry.callable
        for entry in build_account_tool_entries(repo=repo, settings=settings, tenant_id="local")
    }

    payload = tools["get_account_snapshot"]()

    assert payload["available_agent_ids"] == ["gemini", "gpt", "claude"]


def test_investment_chat_sleeve_tool_normalizes_model_aliases_to_agent_ids() -> None:
    from arena.agents.investment_chat.account_tools import build_account_tool_entries

    settings = load_settings()
    repo = _ChatOrderRepo()
    tools = {
        entry.name: entry.callable
        for entry in build_account_tool_entries(repo=repo, settings=settings, tenant_id="local")
    }

    for alias, expected_agent_id in [
        ("gpt4o", "gpt"),
        ("gemini_2_0_flash_exp", "gemini"),
        ("claude_3_7_sonnet", "claude"),
    ]:
        payload = tools["get_agent_sleeve_snapshot"](agent_id=alias)

        assert payload["agent_id"] == expected_agent_id
        assert payload["requested_agent_id"] == alias
        assert payload["available_agent_ids"] == ["gemini", "gpt", "claude"]
        assert repo.sleeve_calls[-1]["agent_id"] == expected_agent_id


def test_investment_chat_sleeve_tool_rejects_unknown_agent_id() -> None:
    from arena.agents.investment_chat.account_tools import build_account_tool_entries

    settings = load_settings()
    repo = _ChatOrderRepo()
    tools = {
        entry.name: entry.callable
        for entry in build_account_tool_entries(repo=repo, settings=settings, tenant_id="local")
    }

    payload = tools["get_agent_sleeve_snapshot"](agent_id="quant_bot")

    assert payload["status"] == "blocked"
    assert payload["requested_agent_id"] == "quant_bot"
    assert payload["available_agent_ids"] == ["gemini", "gpt", "claude"]
    assert repo.sleeve_calls == []


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
    tools = {getattr(tool, "__name__", ""): tool for tool in _chat_advisor_agent(agent).tools}

    result = tools["refresh_account_snapshot"]()

    assert result["status"] == "blocked"
    assert result["tenant_id"] == "czxnms"
    assert "tenant KIS credentials" in result["error"]
    assert "synced" not in calls
