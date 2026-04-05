from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from arena.config import Settings
from arena.data.bq import BigQueryRepository

from .macro_tools import MacroTools
from .quant_tools import QuantTools
from .registry import ToolEntry, ToolRegistry
from .sentiment_tools import SentimentTools

logger = logging.getLogger(__name__)


def _tool(
    *,
    tool_id: str,
    description: str,
    category: str,
    tier: str,
    label_ko: str,
    description_ko: str,
    callable=None,
    enabled: bool = True,
    sort_order: int = 100,
) -> ToolEntry:
    return ToolEntry(
        tool_id=tool_id,
        name=tool_id,
        description=description,
        category=category,
        callable=callable,
        tier=tier,
        label_ko=label_ko,
        description_ko=description_ko,
        enabled=enabled,
        sort_order=sort_order,
    )


def _base_entries(
    *,
    qt: QuantTools,
    st: SentimentTools,
    mt: MacroTools,
    settings: Settings,
) -> list[ToolEntry]:
    return [
        _tool(
            tool_id="search_past_experiences",
            description="Search your own past trades, lessons, and manual notes.",
            category="context",
            tier="core",
            label_ko="과거 경험 검색",
            description_ko="나(에이전트) 자신의 과거 거래 이력, 실패·성공에서 얻은 교훈, 직접 남긴 수동 메모를 벡터 검색으로 찾아 현재 의사결정에 참고합니다. 같은 종목이나 비슷한 시장 상황에서 어떤 판단을 했는지 되돌아볼 때 사용합니다.",
            sort_order=10,
        ),
        _tool(
            tool_id="search_peer_lessons",
            description="Search compacted lessons from other models for peer takeaways.",
            category="context",
            tier="core",
            label_ko="피어 교훈 검색",
            description_ko="같은 테넌트·모드에서 활동 중인 다른 에이전트(GPT, Gemini, Claude)가 컴팩션을 통해 축적한 교훈 메모를 벡터 검색합니다. 다른 모델의 시각과 경험을 빌려 자신의 판단을 보완할 때 유용합니다.",
            sort_order=20,
        ),
        _tool(
            tool_id="get_research_briefing",
            description="Fetches a global market, geopolitics, sector, and single-name research briefing.",
            category="context",
            tier="core",
            label_ko="리서치 브리핑",
            description_ko="글로벌 시장 동향, 지정학 이슈, 섹터 로테이션, 보유·관심 종목에 대한 리서치 브리핑을 요약 조회합니다. 사이클 시작 시 '오늘 시장에 무슨 일이 있는지' 빠르게 파악하는 첫 번째 도구입니다.",
            sort_order=30,
        ),
        _tool(
            tool_id="portfolio_diagnosis",
            description="Diagnoses current holdings, concentration, risk contribution, MDD, benchmark alpha, and returns an HRP rebalance plan for existing positions.",
            category="quant",
            tier="core",
            label_ko="포트폴리오 진단",
            description_ko="현재 보유 종목의 집중도, 개별 리스크 기여도, 최대 낙폭(MDD), 벤치마크 대비 초과수익(alpha)을 종합 진단합니다. 결과로 HRP(Hierarchical Risk Parity) 기반 리밸런스 계획을 제안하여 기존 포지션을 어떻게 조정하면 좋을지 안내합니다.",
            sort_order=40,
        ),
        _tool(
            tool_id="save_memory",
            description="Save a short manual note only for non-obvious exceptions the automatic compactor may miss.",
            category="context",
            tier="core",
            label_ko="메모리 저장",
            description_ko="자동 컴팩션이 놓칠 수 있는 예외적인 교훈이나 중요한 관찰을 짧은 수동 메모로 저장합니다. 다음 사이클에서 '과거 경험 검색'으로 다시 불러올 수 있으며, 반복하지 말아야 할 실수나 기억해야 할 전략 인사이트를 남길 때 사용합니다.",
            sort_order=50,
        ),
        _tool(
            tool_id="screen_market",
            description=(
                "Single discovery entry point for the runtime universe. "
                "Surfaces opportunities across multiple styles including momentum, pullback, recovery, defensive, and value. "
                "Use bucket='...' to focus on one style or leave it empty for a balanced mix."
            ),
            category="quant",
            tier="optional",
            callable=qt.screen_market,
            label_ko="시장 스크리닝",
            description_ko="런타임 유니버스에서 여러 스타일의 기회를 탐색하는 핵심 discovery 도구입니다. 모멘텀, 눌림목, 회복, 방어주, 가치주 버킷을 지원하며, bucket을 비워두면 균형 잡힌 mixed 결과를 반환합니다. 기존 보유 종목 외 대안을 넓게 탐색한 뒤 기술 지표, 예측, 펀더멘털 분석으로 이어갈 때 사용합니다.",
            sort_order=110,
        ),
        _tool(
            tool_id="optimize_portfolio",
            description=(
                "Given a basket of tickers, computes mathematically optimal allocation weights and generates ready-to-execute rebalance orders. "
                "Strategies: 'sharpe', 'risk_parity', and 'forecast'. Includes backtest MDD. "
                "Optional regime_scale (0.3-1.0) scales all weights down for risk-off environments."
            ),
            category="quant",
            tier="optional",
            callable=qt.optimize_portfolio,
            label_ko="포트폴리오 최적화",
            description_ko="후보 종목 바스켓을 입력하면 수학적 최적 배분 비중을 계산하고, 현재 포지션 기준 리밸런스 주문을 생성합니다. 샤프 비율 극대화(sharpe), 리스크 패리티(risk_parity), 수익률 예측 기반(forecast) 전략을 지원하며, 백테스트 MDD도 함께 제공합니다. regime_scale(0.3~1.0)로 리스크오프 환경에서 비중을 일괄 축소할 수 있습니다.",
            sort_order=120,
        ),
        _tool(
            tool_id="forecast_returns",
            description=(
                "Runs seven time-series models and summarizes each ticker with direction probability, vote counts, consensus, "
                "and compact model details. If tickers are omitted, it defaults to the self-discovered candidate basket plus current holdings."
            ),
            category="quant",
            tier="optional",
            callable=qt.forecast_returns,
            label_ko="수익률 예측",
            description_ko="7가지 시계열 모델(ARIMA, ETS, Prophet 등)을 동시에 돌려 각 종목의 방향 확률, 투표 수, 컨센서스를 요약합니다. ticker를 명시하지 않으면 방금 탐색한 self-discovered 후보 바스켓과 현재 보유 종목을 기본 분석 대상으로 사용합니다. 여러 후보의 기대수익률을 한눈에 비교하여 매수·매도 판단의 정량적 근거로 활용합니다.",
            sort_order=130,
        ),
        _tool(
            tool_id="technical_signals",
            description=(
                "Returns RSI, MACD, Bollinger Bands, moving-average trend, volume analysis (volume ratio, OBV trend, "
                "price-volume confirmation), and KOSPI investor flow signals (foreign/institutional net buy)."
            ),
            category="quant",
            tier="optional",
            callable=qt.technical_signals,
            label_ko="기술 지표 분석",
            description_ko="RSI, MACD, 볼린저 밴드, 이동평균 추세를 계산하고, 거래량 분석(거래량 비율·OBV 추세·가격-거래량 확인)도 수행합니다. KOSPI 종목은 외국인·기관 순매수 수급 신호가 추가됩니다. 매매 타이밍 판단의 기술적 근거를 제공합니다.",
            sort_order=150,
        ),
        _tool(
            tool_id="sector_summary",
            description="Summarizes sector rotation so you can see which groups are leading, lagging, and attracting capital.",
            category="quant",
            tier="optional",
            callable=qt.sector_summary,
            label_ko="섹터 요약",
            description_ko="현재 유니버스를 섹터별로 그룹화하여 어떤 업종이 주도하고, 어떤 업종이 뒤처지며, 어디로 자본이 유입되고 있는지 섹터 로테이션 현황을 한눈에 요약합니다.",
            sort_order=170,
        ),
        _tool(
            tool_id="get_fundamentals",
            description=(
                "Fetches valuation metrics for a basket. US stocks get PER/PBR/EPS/BPS, while KOSPI stocks get "
                "EPS/BPS/ROE/debt ratio/growth metrics."
            ),
            category="quant",
            tier="optional",
            callable=qt.get_fundamentals,
            label_ko="펀더멘탈 조회",
            description_ko="후보 종목의 밸류에이션 지표를 일괄 조회합니다. US 종목은 PER·PBR·EPS·BPS, KOSPI 종목은 EPS·BPS·ROE·부채비율·성장성 지표를 제공합니다. 기술적 분석과 함께 펀더멘탈 관점의 균형 잡힌 판단을 돕습니다.",
            sort_order=180,
        ),
        _tool(
            tool_id="index_snapshot",
            description=(
                "Fetches latest quotes and returns for market indices, commodities, and bond yields. "
                "Automatically adapts to the agent's target market when indices=None."
            ),
            category="macro",
            tier="optional",
            callable=qt.index_snapshot,
            label_ko="시장지수 조회",
            description_ko="주요 시장지수, 원자재(금·유가 등), 채권 수익률의 최신 시세와 수익률을 조회합니다. indices 파라미터를 비워두면 에이전트의 타겟 마켓(US/KOSPI)에 맞는 지표가 자동 선택됩니다.",
            sort_order=190,
        ),
        _tool(
            tool_id="fear_greed_index",
            description=(
                "Composite market regime indicator (0=extreme fear/risk-off, 100=extreme greed/risk-on). "
                "Combines volatility index (VKOSPI/VIX), market breadth, momentum trend, and institutional flow. "
                "Returns regime_label (risk_on/neutral/risk_off) with sub-component scores."
            ),
            category="macro",
            tier="optional",
            callable=st.fear_greed_index,
            label_ko="시장 레짐 지표",
            description_ko="변동성 지수(VIX/VKOSPI), 시장 breadth, 모멘텀 추세, 기관 수급을 종합한 복합 시장 레짐 지표입니다. 0=극단적 공포(risk-off), 100=극단적 탐욕(risk-on)으로 표시하며, risk_on/neutral/risk_off 라벨과 서브컴포넌트 점수를 반환합니다. 포트폴리오 최적화의 regime_scale과 연계할 수 있습니다.",
            sort_order=210,
        ),
        _tool(
            tool_id="earnings_calendar",
            description=(
                "Fetches upcoming earnings/dividend events. US: Nasdaq earnings calendar. "
                "KOSPI: KIS dividend schedule + consensus earnings estimates."
            ),
            category="macro",
            tier="optional",
            callable=st.earnings_calendar,
            label_ko="이벤트 캘린더",
            description_ko="다가오는 실적 발표와 배당 이벤트를 조회합니다. US 종목은 Nasdaq 어닝 캘린더, KOSPI 종목은 KIS 배당 일정과 컨센서스 추정실적을 제공합니다. 이벤트 전후 리스크 관리에 활용합니다.",
            sort_order=220,
        ),
        _tool(
            tool_id="fetch_reddit_sentiment",
            description="Fetches recent Reddit posts mentioning a ticker from finance subreddits for retail sentiment.",
            category="sentiment",
            tier="optional",
            callable=st.fetch_reddit_sentiment,
            label_ko="레딧 여론 수집",
            description_ko="레딧의 금융 서브레딧(r/wallstreetbets, r/stocks 등)에서 특정 종목에 대한 최근 게시글과 댓글을 수집하여 개인 투자자 심리를 파악합니다. 밈 주식 열풍이나 소셜 모멘텀을 포착할 때 유용합니다.",
            enabled=bool(settings.reddit_sentiment_enabled),
            sort_order=230,
        ),
        _tool(
            tool_id="fetch_sec_filings",
            description="Fetches recent SEC filings (10-K, 10-Q, 8-K, etc.) for a ticker from EDGAR.",
            category="sentiment",
            tier="optional",
            callable=st.fetch_sec_filings,
            label_ko="SEC 공시 조회",
            description_ko="EDGAR에서 특정 종목의 최근 SEC 공시(10-K 연간보고서, 10-Q 분기보고서, 8-K 수시공시 등)를 조회합니다. 공시 종류·제출일·제목을 반환하여 중요한 기업 이벤트를 빠르게 확인할 수 있습니다.",
            sort_order=240,
        ),
        _tool(
            tool_id="macro_snapshot",
            description=(
                "Fetches macro indicators adapted to the agent market. US: Fed rate, CPI, unemployment, Treasury yields. "
                "KOSPI: BOK base rate, KR CPI, unemployment, government bond yields, USD/KRW."
            ),
            category="macro",
            tier="optional",
            callable=mt.macro_snapshot,
            label_ko="거시 지표 조회",
            description_ko="에이전트의 타겟 마켓에 맞는 거시경제 지표를 일괄 조회합니다. US: 연방기금금리·CPI·실업률·국채 수익률(FRED 데이터), KR: BOK 기준금리·소비자물가·실업률·국고채 수익률·USD/KRW 환율(ECOS 데이터). 거시 환경을 파악한 뒤 레짐 지표나 포트폴리오 최적화와 연계하면 효과적입니다.",
            sort_order=250,
        ),
    ]


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    if not token:
        return None
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _load_tools_config(repo: BigQueryRepository, tenant_id: str) -> dict[str, dict[str, Any]]:
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return {}
    try:
        raw = getter(tenant_id, "tools_config")
    except Exception as exc:
        logger.warning(
            "[yellow]tools_config load failed[/yellow] tenant=%s err=%s",
            tenant_id,
            str(exc),
        )
        return {}
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception as exc:
        logger.warning(
            "[yellow]tools_config parse failed[/yellow] tenant=%s err=%s raw=%s",
            tenant_id,
            str(exc),
            text[:200],
        )
        return {}

    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        if "tool_id" in payload:
            rows = [payload]
        else:
            rows = [
                {"tool_id": tool_id, **row}
                for tool_id, row in payload.items()
                if isinstance(row, dict)
            ]
    else:
        logger.warning(
            "[yellow]tools_config ignored[/yellow] tenant=%s reason=unsupported_root_type type=%s",
            tenant_id,
            type(payload).__name__,
        )

    overlay: dict[str, dict[str, Any]] = {}
    for row in rows:
        tool_id = str(row.get("tool_id") or "").strip()
        if not tool_id:
            continue
        data: dict[str, Any] = {}
        enabled = _coerce_bool(row.get("enabled"))
        if enabled is not None:
            data["enabled"] = enabled
        label_ko = str(row.get("label_ko") or row.get("ui_label_ko") or "").strip()
        if label_ko:
            data["label_ko"] = label_ko
        description_ko = str(row.get("description_ko") or row.get("ui_description_ko") or "").strip()
        if description_ko:
            data["description_ko"] = description_ko
        description = str(row.get("model_description_override") or row.get("description") or "").strip()
        if description:
            data["description"] = description
        sort_order: int | None
        try:
            sort_order = int(row.get("sort_order"))
        except (TypeError, ValueError):
            sort_order = None
        if sort_order is not None:
            data["sort_order"] = sort_order
        if data:
            overlay[tool_id] = data
    return overlay


def _apply_overlay(entry: ToolEntry, overlay: dict[str, Any]) -> ToolEntry:
    enabled = bool(entry.enabled)
    if "enabled" in overlay:
        enabled = enabled and bool(overlay["enabled"])
    return replace(
        entry,
        description=str(overlay.get("description") or entry.description),
        label_ko=str(overlay.get("label_ko") or entry.label_ko),
        description_ko=str(overlay.get("description_ko") or entry.description_ko),
        enabled=enabled,
        sort_order=int(overlay.get("sort_order", entry.sort_order)),
    )


def build_default_registry(
    repo: BigQueryRepository,
    settings: Settings,
    *,
    tenant_id: str = "local",
) -> ToolRegistry:
    """Builds the canonical tool registry used by runtime, UI, and analytics."""
    qt = QuantTools(repo=repo, settings=settings)
    st = SentimentTools(settings=settings)
    mt = MacroTools(settings=settings)
    overlay = _load_tools_config(repo, str(tenant_id or "").strip().lower() or "local")

    entries: list[ToolEntry] = []
    base_entries = _base_entries(qt=qt, st=st, mt=mt, settings=settings)
    base_tool_ids = {entry.tool_id for entry in base_entries}
    for tool_id in sorted(overlay.keys()):
        if tool_id not in base_tool_ids:
            logger.warning(
                "[yellow]tools_config entry ignored[/yellow] tenant=%s tool_id=%s reason=unknown_tool",
                tenant_id,
                tool_id,
            )
    for entry in base_entries:
        if entry.tool_id in overlay:
            entry = _apply_overlay(entry, overlay[entry.tool_id])
        entries.append(entry)
    return ToolRegistry(entries)
