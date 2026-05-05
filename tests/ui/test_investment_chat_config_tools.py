from __future__ import annotations

import json

from tests.ui.investment_chat_helpers import _ChatOrderRepo, _build_raw_chat_tools


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
