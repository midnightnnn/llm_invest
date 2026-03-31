from __future__ import annotations

import asyncio

import pytest
from unittest.mock import MagicMock

from arena.agents.research_agent import ResearchAgent
from arena.config import Settings
from arena.data.bq import BigQueryRepository


@pytest.fixture
def mock_settings():
    settings = MagicMock(spec=Settings)
    settings.agent_ids = ["gemini"]
    settings.agent_configs = {}
    settings.gemini_api_key = "test-gemini-key"
    settings.research_gemini_api_key = ""
    settings.research_gemini_source = ""
    settings.research_gemini_source_tenant = ""
    settings.openai_api_key = ""
    settings.anthropic_api_key = ""
    settings.anthropic_use_vertexai = False
    settings.openai_model = "gpt-5.2"
    settings.research_enabled = True
    settings.research_max_tickers = 5
    settings.gemini_model = "models/gemini-2.5-flash"
    settings.research_gemini_model = "models/gemini-2.5-flash"
    settings.anthropic_model = "claude-sonnet-4-6"
    settings.trading_mode = "paper"
    settings.llm_timeout_seconds = 10
    return settings


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    return repo


@pytest.fixture
def research_agent(mock_settings, mock_repo):
    return ResearchAgent(settings=mock_settings, repo=mock_repo)


def test_research_agent_stays_gemini_for_single_gpt_trader(mock_repo):
    settings = MagicMock(spec=Settings)
    settings.agent_ids = ["gpt"]
    settings.agent_configs = {}
    settings.gemini_api_key = "test-gemini-key"
    settings.research_gemini_api_key = ""
    settings.research_gemini_source = ""
    settings.research_gemini_source_tenant = ""
    settings.openai_api_key = "test-openai-key"
    settings.anthropic_api_key = ""
    settings.anthropic_use_vertexai = False
    settings.openai_model = "gpt-5.2"
    settings.gemini_model = "models/gemini-2.5-flash"
    settings.research_gemini_model = "models/gemini-2.5-flash"
    settings.anthropic_model = "claude-sonnet-4-6"
    settings.research_enabled = True
    settings.research_max_tickers = 5
    settings.trading_mode = "paper"
    settings.llm_timeout_seconds = 10

    agent = ResearchAgent(settings=settings, repo=mock_repo)

    assert agent.provider == "gemini"


# ---------------------------------------------------------------------------
# _research_phase 범용 메서드 테스트
# ---------------------------------------------------------------------------

class TestResearchPhase:
    def test_success(self, research_agent):
        """_research_phase가 정상 결과 dict를 반환한다."""
        async def _mock_phase():
            # Mock the inner method to avoid actual API call
            return {
                "briefing_id": "brf_GLOBAL_123",
                "ticker": "GLOBAL",
                "category": "global_market",
                "summary": "시장 요약",
            }

        research_agent._research_phase_inner = MagicMock(
            side_effect=lambda *a, **kw: _mock_phase()
        )

        result = asyncio.run(
            research_agent._research_phase(
                category="global_market",
                label="GLOBAL",
                prompt="test prompt",
            )
        )
        assert result is not None
        assert result["ticker"] == "GLOBAL"
        assert result["category"] == "global_market"

    def test_failure_filtered_by_gather(self, research_agent):
        """_research_phase 예외는 gather(return_exceptions=True)로 필터링된다."""
        async def _failing_phase(category, label, prompt, *, ticker=None, use_semaphore=False):
            raise RuntimeError("API error")

        async def _ok_phase(category, label, prompt, *, ticker=None, use_semaphore=False):
            return {"ticker": "GLOBAL", "category": category, "summary": "ok"}

        async def _test():
            tasks = [
                _ok_phase("global_market", "GLOBAL", "p"),
                _failing_phase("geopolitical", "GEOPOLITICAL", "p"),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            briefings = [r for r in results if isinstance(r, dict)]
            return briefings

        briefings = asyncio.run(_test())
        assert len(briefings) == 1
        assert briefings[0]["category"] == "global_market"

    def test_empty_summary_returns_none(self, research_agent):
        """빈 summary를 반환하는 LLM 응답 → None."""
        async def _empty_inner(*a, **kw):
            return None

        research_agent._research_phase_inner = MagicMock(side_effect=_empty_inner)

        result = asyncio.run(
            research_agent._research_phase(
                category="global_market",
                label="GLOBAL",
                prompt="test prompt",
            )
        )
        assert result is None


# ---------------------------------------------------------------------------
# run() 통합 테스트
# ---------------------------------------------------------------------------

class TestRun:
    def test_produces_global_briefings(self, research_agent, mock_repo):
        """run()이 3개 글로벌 + N개 종목 브리핑을 생성한다."""
        call_count = 0

        async def _mock_phase(category, label, prompt, *, ticker=None, use_semaphore=False):
            nonlocal call_count
            call_count += 1
            effective_ticker = ticker or label.upper()
            return {
                "briefing_id": f"brf_{effective_ticker}_{call_count}",
                "ticker": effective_ticker,
                "category": category,
                "summary": f"{label} 요약",
                "headline": f"{label} 리서치 브리핑",
                "created_at": "2026-03-04T00:00:00Z",
                "sources": "[]",
                "trading_mode": "paper",
            }

        research_agent._research_phase = _mock_phase
        research_agent._research_held_ticker = lambda t: _mock_phase(
            "held", t, f"{t} prompt", ticker=t
        )

        results = asyncio.run(research_agent.run(["AAPL", "NVDA"]))

        # 3 global (GLOBAL, GEOPOLITICAL, SECTOR) + 2 held (AAPL, NVDA)
        assert len(results) == 5
        categories = [r["category"] for r in results]
        assert "global_market" in categories
        assert "geopolitical" in categories
        assert "sector_trends" in categories
        assert categories.count("held") == 2

        mock_repo.insert_research_briefings.assert_called_once()

    def test_disabled_returns_empty(self, research_agent, mock_settings, mock_repo):
        """research_enabled=False면 빈 리스트를 반환한다."""
        mock_settings.research_enabled = False
        results = asyncio.run(research_agent.run(["AAPL"]))
        assert results == []
        mock_repo.insert_research_briefings.assert_not_called()

    def test_missing_gemini_key_returns_empty(self, mock_repo):
        settings = MagicMock(spec=Settings)
        settings.agent_ids = ["gpt"]
        settings.agent_configs = {}
        settings.gemini_api_key = ""
        settings.research_gemini_api_key = ""
        settings.research_gemini_source = ""
        settings.research_gemini_source_tenant = ""
        settings.openai_api_key = "test-openai-key"
        settings.anthropic_api_key = ""
        settings.anthropic_use_vertexai = False
        settings.openai_model = "gpt-5.2"
        settings.gemini_model = "models/gemini-2.5-flash"
        settings.research_gemini_model = "models/gemini-2.5-flash"
        settings.anthropic_model = "claude-sonnet-4-6"
        settings.research_enabled = True
        settings.research_max_tickers = 5
        settings.trading_mode = "paper"
        settings.llm_timeout_seconds = 10

        agent = ResearchAgent(settings=settings, repo=mock_repo)
        results = asyncio.run(agent.run(["AAPL"]))

        assert results == []
        mock_repo.insert_research_briefings.assert_not_called()

    def test_handles_phase_failures(self, research_agent, mock_repo):
        """일부 페이즈 실패 시 나머지 결과만 저장한다."""
        call_idx = 0

        async def _mock_phase(category, label, prompt, *, ticker=None, use_semaphore=False):
            nonlocal call_idx
            call_idx += 1
            # Phase 2 (geopolitical) fails
            if category == "geopolitical":
                raise RuntimeError("Geopolitical search failed")
            effective_ticker = ticker or label.upper()
            return {
                "briefing_id": f"brf_{effective_ticker}_{call_idx}",
                "ticker": effective_ticker,
                "category": category,
                "summary": f"{label} 요약",
                "headline": f"{label} 리서치 브리핑",
                "created_at": "2026-03-04T00:00:00Z",
                "sources": "[]",
                "trading_mode": "paper",
            }

        research_agent._research_phase = _mock_phase
        research_agent._research_held_ticker = lambda t: _mock_phase(
            "held", t, f"{t} prompt", ticker=t
        )

        results = asyncio.run(research_agent.run(["AAPL"]))

        # 2 global succeed + 1 held = 3 (geopolitical failed)
        assert len(results) == 3
        categories = [r["category"] for r in results]
        assert "geopolitical" not in categories
        assert "global_market" in categories
        assert "sector_trends" in categories

        mock_repo.insert_research_briefings.assert_called_once()

    def test_deduplicates_held_tickers(self, research_agent, mock_repo):
        """중복 보유종목은 deduplicate된다."""
        call_count = 0

        async def _mock_phase(category, label, prompt, *, ticker=None, use_semaphore=False):
            nonlocal call_count
            call_count += 1
            effective_ticker = ticker or label.upper()
            return {
                "briefing_id": f"brf_{effective_ticker}_{call_count}",
                "ticker": effective_ticker,
                "category": category,
                "summary": f"{label} 요약",
                "headline": f"{label} 리서치 브리핑",
                "created_at": "2026-03-04T00:00:00Z",
                "sources": "[]",
                "trading_mode": "paper",
            }

        research_agent._research_phase = _mock_phase
        research_agent._research_held_ticker = lambda t: _mock_phase(
            "held", t, f"{t} prompt", ticker=t
        )

        results = asyncio.run(research_agent.run(["AAPL", "AAPL", "NVDA"]))

        held_tickers = [r["ticker"] for r in results if r["category"] == "held"]
        assert held_tickers == ["AAPL", "NVDA"]  # no duplicate AAPL

    def test_caps_held_tickers_at_max(self, research_agent, mock_settings, mock_repo):
        """보유종목이 research_max_tickers를 초과하면 잘린다."""
        mock_settings.research_max_tickers = 2

        async def _mock_phase(category, label, prompt, *, ticker=None, use_semaphore=False):
            effective_ticker = ticker or label.upper()
            return {
                "briefing_id": f"brf_{effective_ticker}",
                "ticker": effective_ticker,
                "category": category,
                "summary": "요약",
                "headline": "브리핑",
                "created_at": "2026-03-04T00:00:00Z",
                "sources": "[]",
                "trading_mode": "paper",
            }

        research_agent._research_phase = _mock_phase
        research_agent._research_held_ticker = lambda t: _mock_phase(
            "held", t, f"{t} prompt", ticker=t
        )

        results = asyncio.run(research_agent.run(["AAPL", "NVDA", "TSLA", "GOOGL"]))

        held = [r for r in results if r["category"] == "held"]
        assert len(held) == 2  # capped at max
