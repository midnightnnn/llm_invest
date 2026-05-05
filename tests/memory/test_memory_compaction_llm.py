from __future__ import annotations

import asyncio

import arena.agents.memory_compaction_agent as memory_compaction_module
from arena.agents.memory_compaction_agent import MemoryCompactionAgent
from tests.memory.memory_compaction_helpers import _FakeMemoryStore, _FakeRepo, _settings


def test_memory_compaction_agent_calls_litellm_direct_without_claude_temperature(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["claude"]
    settings.gemini_api_key = ""
    settings.openai_api_key = ""
    settings.anthropic_api_key = "tenant-anthropic-key"
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert captured["model"] == "anthropic/claude-sonnet-4-6"
    assert captured["api_key"] == "tenant-anthropic-key"
    assert "temperature" not in captured


def test_memory_compaction_agent_omits_temperature_for_gpt5_models(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["gpt"]
    settings.gemini_api_key = ""
    settings.openai_api_key = "tenant-openai-key"
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert captured["model"] == "openai/gpt-5.2"
    assert captured["api_key"] == "tenant-openai-key"
    assert "temperature" not in captured


def test_memory_compaction_agent_supports_deepseek_helper_from_provider_payload(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = []
    settings.gemini_api_key = ""
    settings.openai_api_key = ""
    settings.anthropic_api_key = ""
    settings.provider_secrets = {
        "deepseek": {
            "api_key": "tenant-deepseek-key",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
        }
    }

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    captured: dict[str, object] = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert agent.provider == "deepseek"
    assert captured["model"] == "deepseek/deepseek-chat"
    assert captured["api_key"] == "tenant-deepseek-key"
    assert captured["base_url"] == "https://api.deepseek.com/v1"


def test_memory_compaction_agent_retries_transient_helper_errors(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    calls = {"count": 0}
    sleeps: list[float] = []

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("503 high demand")
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(memory_compaction_module, "retry_policy_from_env", lambda: (2, 0.5))
    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(memory_compaction_module.asyncio, "sleep", _fake_sleep)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert calls["count"] == 2
    assert sleeps == [0.5]


def test_memory_compaction_agent_treats_bad_gateway_as_retryable() -> None:
    assert memory_compaction_module._is_retryable_compaction_error(
        RuntimeError("litellm.BadGatewayError: 502 Bad gateway")
    ) is True


def test_memory_compaction_error_formatter_keeps_blank_exception_type() -> None:
    assert memory_compaction_module._format_compaction_error(TimeoutError()) == "TimeoutError"


def test_memory_compaction_agent_retries_empty_response(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    calls = {"count": 0}
    sleeps: list[float] = []

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        calls["count"] += 1
        if calls["count"] == 1:
            return {"choices": [{"message": {"content": ""}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"reflections":[]}',
                    }
                }
            ]
        }

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(memory_compaction_module, "retry_policy_from_env", lambda: (1, 0.25))
    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(memory_compaction_module.asyncio, "sleep", _fake_sleep)

    out = asyncio.run(agent._collect_response_text(prompt="PROMPT"))

    assert out == '{"reflections":[]}'
    assert calls["count"] == 2
    assert sleeps == [0.25]


def test_memory_compaction_agent_raises_after_retry_budget_exhausted(monkeypatch) -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    calls = {"count": 0}
    sleeps: list[float] = []

    async def _fake_acompletion(**kwargs):
        _ = kwargs
        calls["count"] += 1
        raise RuntimeError("503 high demand")

    async def _fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(memory_compaction_module, "retry_policy_from_env", lambda: (2, 0.5))
    monkeypatch.setattr(memory_compaction_module.litellm, "acompletion", _fake_acompletion)
    monkeypatch.setattr(memory_compaction_module.asyncio, "sleep", _fake_sleep)

    try:
        asyncio.run(agent._collect_response_text(prompt="PROMPT"))
    except RuntimeError as exc:
        assert "503 high demand" in str(exc)
    else:
        raise AssertionError("expected retry exhaustion to re-raise the last helper error")

    assert calls["count"] == 3
    assert sleeps == [0.5, 1.0]
