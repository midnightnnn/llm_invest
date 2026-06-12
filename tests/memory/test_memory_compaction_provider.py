from __future__ import annotations

from arena.agents.memory_compaction_agent import MemoryCompactionAgent
from tests.memory.memory_compaction_helpers import _FakeMemoryStore, _FakeRepo, _settings


def test_memory_compaction_agent_follows_configured_single_agent_provider() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["claude"]
    settings.gemini_api_key = ""
    settings.anthropic_api_key = "test-anthropic-key"
    settings.anthropic_model = "claude-sonnet-4-6"

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    assert agent.provider == "claude"
    assert agent.model == "anthropic/claude-sonnet-4-6"


def test_memory_compaction_agent_maps_claude_opus_to_stable_sonnet_default() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["claude"]
    settings.gemini_api_key = ""
    settings.anthropic_api_key = "test-anthropic-key"
    settings.anthropic_model = "claude-opus-4-8"

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    assert agent.provider == "claude"
    assert agent.model == "anthropic/claude-sonnet-4-6"


def test_memory_compaction_agent_prefers_direct_key_provider_over_non_direct_fallback() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["gemini", "gpt"]
    settings.gemini_api_key = ""
    settings.openai_api_key = "tenant-openai-key"

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    assert agent.provider == "gpt"
    assert agent.model == "openai/gpt-5.2"


def test_memory_compaction_agent_defaults_to_economical_provider_model() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["gpt"]
    settings.gemini_api_key = ""
    settings.openai_api_key = "tenant-openai-key"
    settings.openai_model = "gpt-5.4-pro"

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    assert agent.provider == "gpt"
    assert agent.model == "openai/gpt-5.4"


def test_memory_compaction_agent_respects_explicit_memory_model_override() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["gpt"]
    settings.gemini_api_key = ""
    settings.openai_api_key = "tenant-openai-key"
    settings.openai_model = "gpt-5.4-pro"
    settings.memory_compaction_models = {"gpt": "gpt-5.2"}

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    assert agent.provider == "gpt"
    assert agent.model == "openai/gpt-5.2"


def test_memory_compaction_agent_respects_per_agent_memory_model_override() -> None:
    from arena.config import AgentConfig

    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["gpt"]
    settings.gemini_api_key = ""
    settings.openai_api_key = "tenant-openai-key"
    settings.openai_model = "gpt-5.4-pro"
    settings.agent_configs = {
        "gpt": AgentConfig(
            agent_id="gpt",
            provider="gpt",
            model="gpt-5.4-pro",
            capital_krw=1_000_000,
            memory_compaction_model="gpt-5.2",
        )
    }

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    assert agent._helper_client_for_agent("gpt")["model"] == "openai/gpt-5.2"


def test_memory_compaction_agent_derives_economical_models_for_other_providers() -> None:
    repo = _FakeRepo()
    memory_store = _FakeMemoryStore()
    settings = _settings()
    settings.agent_ids = ["gemini", "claude"]
    settings.gemini_api_key = "tenant-gemini-key"
    settings.anthropic_api_key = "tenant-anthropic-key"
    settings.gemini_model = "gemini-3.1-pro-preview"
    settings.anthropic_model = "claude-opus-4-7"

    agent = MemoryCompactionAgent(settings=settings, repo=repo, memory_store=memory_store)

    assert agent._helper_client_for_agent("gemini")["model"] == "gemini/gemini-3-flash-preview"
    assert agent._helper_client_for_agent("claude")["model"] == "anthropic/claude-sonnet-4-6"
