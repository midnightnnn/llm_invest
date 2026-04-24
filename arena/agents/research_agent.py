from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from google.adk import Agent, Runner
from google.adk.models import Gemini
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search

from arena.config import Settings, effective_research_gemini_api_key, research_generation_status
from arena.data.bq import BigQueryRepository
from arena.models import utc_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

def _today_str() -> str:
    """Returns today's date as YYYY-MM-DD string (UTC)."""
    return utc_now().date().isoformat()


def _global_prompt() -> str:
    today = _today_str()
    return (
        f"오늘은 {today}이다. 오늘({today}) 기준 글로벌 금융시장 주요 이슈를 Google 검색으로 조사하라. "
        "미국·유럽·아시아 주요 지수의 **오늘 장중 또는 가장 최근 마감 기준** 등락률, "
        "중앙은행 정책(금리·양적완화), 원자재·유가·금 가격 변동, 달러 인덱스, 채권 수익률 등을 포괄하라. "
        "반드시 오늘 또는 직전 거래일의 수치를 사용하고, 며칠 전 데이터를 최신인 것처럼 쓰지 마라. "
        "핵심 팩트·수치·방향만 구조화해서 한국어 300자 내외로 요약하라."
    )


def _geopolitical_prompt() -> str:
    today = _today_str()
    return (
        f"오늘은 {today}이다. 최근 1주일({today} 기준)간 글로벌 지정학 리스크를 Google 검색으로 조사하라. "
        "전쟁·군사 충돌, 무역 분쟁·관세, 제재, 선거·정권 교체, "
        "에너지 공급 위기, 사이버 공격 등 금융시장에 영향을 줄 수 있는 "
        "지정학 이벤트를 빠짐없이 파악하라. "
        "핵심 이벤트·관련국·시장 영향만 한국어 300자 내외로 요약하라."
    )


def _sector_prompt() -> str:
    today = _today_str()
    return (
        f"오늘은 {today}이다. 최근 1주일({today} 기준)간 미국 주식시장 섹터별 동향을 Google 검색으로 조사하라. "
        "기술(AI/반도체), 헬스케어, 에너지, 금융, 소비재, 산업재, 유틸리티 등 "
        "주요 섹터의 실적 발표, 규제 변화, 수급 변동, 핵심 테마를 파악하라. "
        "섹터 로테이션 시그널과 강세/약세 섹터를 한국어 300자 내외로 요약하라."
    )


def _ticker_prompt(ticker: str) -> str:
    today = _today_str()
    return (
        f"오늘은 {today}이다. 종목 {ticker}의 최근 1주일({today} 기준) 주요 뉴스와 이슈를 정리하라. "
        "다른 투자 에이전트가 매매 판단에 바로 활용할 수 있도록, "
        "핵심 요인·리스크·촉매만 구조화해서 한국어 200~300자 내외로 핵심만 요약하라."
    )


class ResearchAgent:
    """Gemini + Google Search Grounding 기반 멀티 페이즈 리서치 에이전트.

    Phase 1: 글로벌 시장 동향        (ticker=GLOBAL,       category=global_market)
    Phase 2: 지정학 리스크            (ticker=GEOPOLITICAL, category=geopolitical)
    Phase 3: 섹터 트렌드              (ticker=SECTOR,       category=sector_trends)
    Phase 4: 보유종목 보충 리서치      (ticker=실제심볼,      category=held)
    """

    def __init__(self, settings: Settings, repo: BigQueryRepository):
        self.settings = settings
        self.repo = repo
        self.agent_id = "research_agent"
        self.provider = "gemini"
        self.research_status = research_generation_status(settings)
        self._research_api_key = effective_research_gemini_api_key(settings)
        self._previous_gemini_api_key_env = os.getenv("GEMINI_API_KEY")
        self._restore_gemini_api_key_env = False
        self.model = None
        self.agent = None
        self.session_service = None
        self.runner = None
        self._semaphore = asyncio.Semaphore(3)

        if not self.research_status.get("can_generate"):
            logger.info(
                "[cyan]Research agent disabled[/cyan] reason=%s",
                str(self.research_status.get("code") or "unknown"),
            )
            return

        if self._research_api_key and not self.research_status.get("uses_vertex"):
            if self._previous_gemini_api_key_env != self._research_api_key:
                os.environ["GEMINI_API_KEY"] = self._research_api_key
                self._restore_gemini_api_key_env = True

        try:
            model_id = self.settings.research_gemini_model.strip()
            if model_id.startswith("models/"):
                model_id = model_id.split("/", 1)[1]

            self.model = Gemini(model=model_id)

            # 에이전트 선언 (Google Search Grounding 도구 장착)
            today = _today_str()
            self.agent = Agent(
                name=self.agent_id,
                model=self.model,
                instruction=(
                    f"당신은 금융 리서치 전문 에이전트입니다. 오늘 날짜는 {today}입니다. "
                    "글로벌 시장 동향, 지정학 리스크, 섹터 트렌드, 개별 종목 뉴스를 "
                    "Google 검색을 통해 파악하고 구조화하여 요약합니다. "
                    "반드시 오늘 또는 가장 최근 거래일의 데이터만 사용하십시오. "
                    "며칠 전 데이터를 최신인 것처럼 제시하지 마십시오. "
                    "다른 투자 에이전트들이 매매 판단에 즉시 활용할 수 있도록 "
                    "핵심 팩트와 수치 중심으로 간결하게 전달합니다."
                ),
                tools=[google_search],
            )
            self.session_service = InMemorySessionService()
            self.runner = Runner(
                app_name=f"llm_arena_{self.agent_id}",
                agent=self.agent,
                session_service=self.session_service,
            )
        except Exception:
            self._restore_gemini_env()
            raise

    def _restore_gemini_env(self) -> None:
        if not self._restore_gemini_api_key_env:
            return
        previous = self._previous_gemini_api_key_env
        if previous:
            os.environ["GEMINI_API_KEY"] = previous
        else:
            os.environ.pop("GEMINI_API_KEY", None)
        self._restore_gemini_api_key_env = False

    # ------------------------------------------------------------------
    # 범용 리서치 메서드
    # ------------------------------------------------------------------

    async def _research_phase(
        self,
        category: str,
        label: str,
        prompt: str,
        *,
        ticker: str | None = None,
        use_semaphore: bool = False,
    ) -> dict[str, Any] | None:
        """범용 리서치 실행. category/label/prompt로 어떤 페이즈든 처리."""
        if use_semaphore:
            return await self._research_phase_inner(category, label, prompt, ticker=ticker)
        else:
            return await self._research_phase_inner(category, label, prompt, ticker=ticker)

    async def _research_phase_inner(
        self,
        category: str,
        label: str,
        prompt: str,
        *,
        ticker: str | None = None,
    ) -> dict[str, Any] | None:
        if self.session_service is None or self.runner is None:
            return None
        effective_ticker = ticker or label.upper()
        session_id = f"{self.agent_id}_{effective_ticker}_{int(utc_now().timestamp() * 1000)}"
        await self.session_service.create_session(
            app_name=f"llm_arena_{self.agent_id}",
            user_id="arena",
            session_id=session_id,
        )

        try:
            async def _run():
                text = ""
                from google.genai import types
                message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
                async for event in self.runner.run_async(
                    user_id="arena",
                    session_id=session_id,
                    new_message=message,
                ):
                    if event.content:
                        for part in event.content.parts:
                            if getattr(part, "text", None):
                                text += part.text
                return text.strip()

            summary = await asyncio.wait_for(_run(), timeout=self.settings.timeout_for("research"))

            if not summary:
                return None

            briefing_id = f"brf_{effective_ticker}_{int(utc_now().timestamp() * 1000)}"
            return {
                "briefing_id": briefing_id,
                "created_at": utc_now(),
                "ticker": effective_ticker,
                "category": category,
                "headline": f"{label} 리서치 브리핑",
                "summary": summary,
                "sources": "[]",
                "trading_mode": self.settings.trading_mode,
            }
        except Exception as exc:
            logger.warning(
                "[yellow]Research phase failed[/yellow] category=%s label=%s err=%s",
                category, label, str(exc),
            )
            return None

    # ------------------------------------------------------------------
    # Phase 1~3: 글로벌 브리핑 (병렬)
    # ------------------------------------------------------------------

    async def _research_global(self) -> dict[str, Any] | None:
        return await self._research_phase(
            category="global_market",
            label="GLOBAL",
            prompt=_global_prompt(),
        )

    async def _research_geopolitical(self) -> dict[str, Any] | None:
        return await self._research_phase(
            category="geopolitical",
            label="GEOPOLITICAL",
            prompt=_geopolitical_prompt(),
        )

    async def _research_sectors(self) -> dict[str, Any] | None:
        return await self._research_phase(
            category="sector_trends",
            label="SECTOR",
            prompt=_sector_prompt(),
        )

    # ------------------------------------------------------------------
    # Phase 4: 보유종목 보충 리서치 (Semaphore 제어)
    # ------------------------------------------------------------------

    async def _research_held_ticker(self, ticker: str) -> dict[str, Any] | None:
        async with self._semaphore:
            return await self._research_phase(
                category="held",
                label=ticker,
                prompt=_ticker_prompt(ticker),
                ticker=ticker,
                use_semaphore=True,
            )

    # ------------------------------------------------------------------
    # 메인 실행
    # ------------------------------------------------------------------

    async def run(self, held_tickers: list[str]) -> list[dict[str, Any]]:
        """전체 리서치 파이프라인 실행. 결과를 BQ에 저장."""
        try:
            if not self.settings.research_enabled:
                return []
            if not self.research_status.get("can_generate"):
                logger.info(
                    "[cyan]Research phase skipped[/cyan] reason=%s held_tickers=%d",
                    str(self.research_status.get("code") or "unknown"),
                    len(held_tickers),
                )
                return []

            logger.info(
                "[cyan]Research phase started[/cyan] held_tickers=%d",
                len(held_tickers),
            )

            # Phase 1~3: 글로벌 브리핑 병렬 실행
            global_tasks = [
                self._research_global(),
                self._research_geopolitical(),
                self._research_sectors(),
            ]
            global_results = await asyncio.gather(*global_tasks, return_exceptions=True)

            briefings: list[dict[str, Any]] = []
            for res in global_results:
                if isinstance(res, dict):
                    briefings.append(res)

            # Phase 4: 보유종목 보충 리서치 (max research_max_tickers개)
            ticker_targets = list(dict.fromkeys(held_tickers))[:self.settings.research_max_tickers]
            if ticker_targets:
                ticker_tasks = [self._research_held_ticker(t) for t in ticker_targets]
                ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
                for res in ticker_results:
                    if isinstance(res, dict):
                        briefings.append(res)

            if briefings:
                try:
                    self.repo.insert_research_briefings(briefings)
                except Exception as exc:
                    logger.error("[red]Research briefings insert failed[/red] err=%s", str(exc))

            logger.info(
                "[cyan]Research phase completed[/cyan] total_briefings=%d (global=%d, held=%d)",
                len(briefings),
                sum(1 for b in briefings if b.get("category") != "held"),
                sum(1 for b in briefings if b.get("category") == "held"),
            )

            return briefings
        finally:
            self._restore_gemini_env()
