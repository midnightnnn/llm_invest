from __future__ import annotations

import json

from tests.ui.helpers import _client, _client_with_repo


def test_admin_tools_lists_core_and_optional(monkeypatch) -> None:
    client = _client(monkeypatch)
    response = client.get("/admin/tools")
    assert response.status_code == 200
    payload = response.json()
    entries = payload["tool_entries"]
    tool_ids = {str(entry["tool_id"]) for entry in entries}
    assert len(tool_ids) >= 17
    assert "portfolio_diagnosis" in tool_ids
    assert "recommend_opportunities" in tool_ids
    assert "screen_market" not in tool_ids
    assert "correlation_matrix" not in tool_ids
    assert "momentum_rank" not in tool_ids
    assert "fetch_reddit_sentiment" in tool_ids
    core_entry = next(entry for entry in entries if str(entry["tool_id"]) == "portfolio_diagnosis")
    optional_entry = next(entry for entry in entries if str(entry["tool_id"]) == "recommend_opportunities")
    forecast_entry = next(entry for entry in entries if str(entry["tool_id"]) == "forecast_returns")
    assert core_entry["configurable"] is True
    assert core_entry["tier"] == "core"
    assert core_entry["label_ko"] == "포트폴리오 진단"
    assert optional_entry["configurable"] is True
    assert optional_entry["tier"] == "optional"
    assert optional_entry["label_ko"] == "통합 기회 추천"
    assert "신규 매수 후보" in str(optional_entry["description_ko"])
    assert "signal-IC" in str(optional_entry["description_ko"])
    assert forecast_entry["label_ko"] == "수익률 예측"
    assert "self-discovered 후보 바스켓" in str(forecast_entry["description_ko"])


def test_admin_tools_hides_reddit_when_runtime_disabled(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config("local", "reddit_sentiment_enabled", "false")

    response = client.get("/admin/tools")

    assert response.status_code == 200
    payload = response.json()
    tool_ids = {str(entry["tool_id"]) for entry in payload["tool_entries"]}
    assert "fetch_reddit_sentiment" not in tool_ids


def test_admin_tools_save_allows_core_tool_ids(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    response = client.post(
        "/admin/tools",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "disabled_tools": ["screen_market", "portfolio_diagnosis"],
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    raw = repo.get_config("local", "disabled_tools")
    assert raw is not None
    saved = json.loads(raw)
    assert saved == ["portfolio_diagnosis", "screen_market"]


def test_admin_tools_apply_tools_config_overlay(monkeypatch) -> None:
    client, repo = _client_with_repo(monkeypatch)
    repo.set_config(
        "local",
        "tools_config",
        json.dumps(
            [
                {
                    "tool_id": "portfolio_diagnosis",
                    "ui_label_ko": "포트 진단 오버라이드",
                    "ui_description_ko": "오버라이드 설명",
                },
                {
                    "tool_id": "screen_market",
                    "enabled": False,
                },
            ],
            ensure_ascii=False,
        ),
    )

    response = client.get("/admin/tools")

    assert response.status_code == 200
    payload = response.json()
    entries = {str(entry["tool_id"]): entry for entry in payload["tool_entries"]}
    assert "screen_market" not in entries
    assert entries["portfolio_diagnosis"]["label_ko"] == "포트 진단 오버라이드"
    assert entries["portfolio_diagnosis"]["description_ko"] == "오버라이드 설명"
