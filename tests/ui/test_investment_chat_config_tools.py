from __future__ import annotations

import json
from types import SimpleNamespace

from tests.ui.investment_chat_helpers import _ChatOrderRepo, _FakeToolContext, _build_raw_chat_tools


def test_chat_config_tool_lists_cached_provider_models(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "investment_chat_model_options",
        json.dumps(
            {
                "providers": {
                    "gpt": {
                        "provider": "gpt",
                        "advisor_models": ["gpt-5.5"],
                        "router_models": ["gpt-5.4-mini"],
                        "utility_models": ["gpt-5.4-mini"],
                    }
                }
            }
        ),
        "seed",
    )
    repo.set_config(
        "local",
        "investment_chat_config",
        json.dumps(
            {
                "provider": "gpt",
                "model": "gpt-5.5",
                "model_routing": {
                    "router_model": "gpt-5.4-mini",
                    "utility_model": "gpt-5.4-mini",
                },
            }
        ),
        "seed",
    )
    tools = _build_raw_chat_tools(monkeypatch, repo, include_internal_bridge=True)

    result = tools["list_chat_model_options"](provider="gpt")

    assert result["status"] == "ok"
    assert result["provider"] == "gpt"
    assert result["advisor_models"] == ["gpt-5.5"]
    assert result["router_models"] == ["gpt-5.4-mini"]
    assert result["current"] == {
        "provider": "gpt",
        "advisor_model": "gpt-5.5",
        "router_model": "gpt-5.4-mini",
        "utility_model": "gpt-5.4-mini",
    }


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


def test_chat_config_tool_uses_adk_confirmation_before_apply(monkeypatch) -> None:
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
    tools = _build_raw_chat_tools(monkeypatch, repo)
    propose = tools["propose_agent_config_change"]

    first_context = _FakeToolContext(function_call_id="fc-config-1")
    waiting = propose(
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        rationale="upgrade model through ADK confirmation",
        tool_context=first_context,
    )

    assert waiting["status"] == "waiting_for_confirmation"
    assert waiting["approval_required"] is True
    assert waiting["approval_ui"] == "adk_tool_confirmation"
    assert waiting["apply_status"] == "not_applied"
    assert first_context.actions.skip_summarization is False
    assert first_context.confirmation_request is not None
    payload = first_context.confirmation_request["payload"]
    assert isinstance(payload, dict)
    assert payload["action"] == "apply_config_change"
    assert payload["scope"] == "agent"
    assert "approval_token" not in payload
    assert "ADK Web 확인창" in str(first_context.confirmation_request["hint"])
    assert "Confirmed 체크박스" in str(first_context.confirmation_request["hint"])
    saved_before_approval = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert saved_before_approval[0]["model"] == "gpt-5.2"

    confirmed_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(confirmed=True, payload={}),
    )
    applied = propose(
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        rationale="upgrade model through ADK confirmation",
        tool_context=confirmed_context,
    )

    assert applied["status"] == "applied"
    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert saved[0]["model"] == "gpt-5.5"


def test_chat_config_adk_confirmation_merges_latest_agent_config(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "agents_config",
        json.dumps(
            [
                {"id": "gpt", "provider": "gpt", "model": "gpt-5.2", "capital_krw": 1_000_000},
                {"id": "gemini", "provider": "gemini", "model": "gemini-3-flash-preview", "capital_krw": 1_000_000},
                {"id": "claude", "provider": "claude", "model": "claude-sonnet-4-6", "capital_krw": 1_000_000},
            ]
        ),
        "seed",
    )
    tools = _build_raw_chat_tools(monkeypatch, repo)
    propose = tools["propose_agent_config_change"]

    gpt_context = _FakeToolContext(function_call_id="fc-config-gpt")
    gemini_context = _FakeToolContext(function_call_id="fc-config-gemini")
    assert (
        propose(agent_id="gpt", action="update", capital_krw=2_000_000, tool_context=gpt_context)["status"]
        == "waiting_for_confirmation"
    )
    assert (
        propose(agent_id="gemini", action="update", capital_krw=3_000_000, tool_context=gemini_context)["status"]
        == "waiting_for_confirmation"
    )

    applied_gpt = propose(
        agent_id="gpt",
        action="update",
        capital_krw=2_000_000,
        tool_context=_FakeToolContext(
            function_call_id=gpt_context.function_call_id,
            state=gpt_context.state,
            tool_confirmation=SimpleNamespace(confirmed=True, payload={}),
        ),
    )
    applied_gemini = propose(
        agent_id="gemini",
        action="update",
        capital_krw=3_000_000,
        tool_context=_FakeToolContext(
            function_call_id=gemini_context.function_call_id,
            state=gemini_context.state,
            tool_confirmation=SimpleNamespace(confirmed=True, payload={}),
        ),
    )

    assert applied_gpt["status"] == "applied"
    assert applied_gemini["status"] == "applied"
    saved = {
        str(entry["id"]): entry
        for entry in json.loads(repo.get_config("local", "agents_config") or "[]")
    }
    assert saved["gpt"]["capital_krw"] == 2_000_000
    assert saved["gemini"]["capital_krw"] == 3_000_000
    assert saved["claude"]["capital_krw"] == 1_000_000


def test_chat_config_tool_adds_krw_to_existing_agent_capital(monkeypatch) -> None:
    repo = _ChatOrderRepo()
    repo.set_config(
        "local",
        "agents_config",
        json.dumps(
            [
                {"id": "gpt", "provider": "gpt", "model": "gpt-5.5", "capital_krw": 5_000_000},
                {"id": "gemini", "provider": "gemini", "model": "gemini-3-flash-preview", "capital_krw": 3_000_000},
            ]
        ),
        "seed",
    )
    tools = _build_raw_chat_tools(monkeypatch, repo, include_internal_bridge=True)

    proposed = tools["propose_agent_config_change"](
        agent_id="gpt",
        action="update",
        capital_allocation_mode="add_krw",
        capital_allocation_amount_krw=1_000_000,
        rationale="사용자가 gpt sleeve에 100만원 추가 배분을 요청함",
    )
    token = str(proposed.get("approval_token") or "")

    assert proposed["status"] == "ok"
    assert {"field": "capital_krw", "before": 5_000_000, "after": 6_000_000} in proposed["diffs"]

    applied = tools["apply_approved_config_change"](
        approval_token=token,
        confirmation_text=f"CONFIRM {token}",
    )

    assert applied["status"] == "applied"
    saved = {
        str(entry["id"]): entry
        for entry in json.loads(repo.get_config("local", "agents_config") or "[]")
    }
    assert saved["gpt"]["capital_krw"] == 6_000_000
    assert saved["gemini"]["capital_krw"] == 3_000_000
    assert repo.capital_sync_calls[-1]["target_capitals"]["gpt"] == 6_000_000


def test_chat_config_adk_confirmation_invalidates_runtime_cache(monkeypatch) -> None:
    from arena.agents.investment_chat.config_tools import build_config_tool_entries
    from arena.config import load_settings

    _ = monkeypatch
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
                }
            ]
        ),
        "seed",
    )
    invalidation_calls: list[tuple[object, ...]] = []
    tools = {
        entry.name: entry.callable
        for entry in build_config_tool_entries(
            repo=repo,
            settings=load_settings(),
            tenant_id="local",
            invalidate_tenant_cache=lambda *args: invalidation_calls.append(args),
        )
    }
    propose = tools["propose_agent_config_change"]

    first_context = _FakeToolContext(function_call_id="fc-config-cache")
    waiting = propose(
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        rationale="cache invalidation after ADK confirmation",
        tool_context=first_context,
    )
    confirmed_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(confirmed=True, payload={}),
    )
    applied = propose(
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        rationale="cache invalidation after ADK confirmation",
        tool_context=confirmed_context,
    )

    assert waiting["status"] == "waiting_for_confirmation"
    assert applied["status"] == "applied"
    assert invalidation_calls == [("local", "runtime", "memory", "portfolio")]


def test_chat_config_tool_rejects_adk_confirmation_without_apply(monkeypatch) -> None:
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
                }
            ]
        ),
        "seed",
    )
    tools = _build_raw_chat_tools(monkeypatch, repo)
    propose = tools["propose_agent_config_change"]

    first_context = _FakeToolContext(function_call_id="fc-config-2")
    waiting = propose(
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        rationale="rejected config change",
        tool_context=first_context,
    )
    assert waiting["status"] == "waiting_for_confirmation"

    rejected_context = _FakeToolContext(
        function_call_id=first_context.function_call_id,
        state=first_context.state,
        tool_confirmation=SimpleNamespace(confirmed=False, payload={}),
    )
    rejected = propose(
        agent_id="gpt",
        action="update",
        provider="gpt",
        model="gpt-5.5",
        rationale="rejected config change",
        tool_context=rejected_context,
    )

    assert rejected["status"] == "rejected"
    assert rejected["apply_status"] == "not_applied"
    saved = json.loads(repo.get_config("local", "agents_config") or "[]")
    assert saved[0]["model"] == "gpt-5.2"
