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
    category: str,
    tier: str,
    description: str,
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
            category="context",
            tier="core",
            description="Search your own past trades, lessons, and manual notes.",
            label_ko="과거 경험 검색",
            description_ko="나(에이전트) 자신의 과거 거래 이력, 실패·성공에서 얻은 교훈, 직접 남긴 수동 메모를 벡터 검색으로 찾아 현재 의사결정에 참고합니다. 같은 종목이나 비슷한 시장 상황에서 어떤 판단을 했는지 되돌아볼 때 사용합니다.",
            sort_order=10,
        ),
        _tool(
            tool_id="search_peer_lessons",
            category="context",
            tier="core",
            description="Search compacted lessons from other models for peer takeaways.",
            label_ko="피어 교훈 검색",
            description_ko="같은 테넌트·모드에서 활동 중인 다른 에이전트(GPT, Gemini, Claude)가 컴팩션을 통해 축적한 교훈 메모를 벡터 검색합니다. 다른 모델의 시각과 경험을 빌려 자신의 판단을 보완할 때 유용합니다.",
            sort_order=20,
        ),
        _tool(
            tool_id="get_research_briefing",
            category="context",
            tier="core",
            description="Fetch cached or on-demand research context for broad market themes and single-name analysis.",
            label_ko="리서치 브리핑",
            description_ko="글로벌 시장 동향, 지정학 이슈, 섹터 로테이션, 보유·관심 종목에 대한 리서치 브리핑을 조회합니다. 필요하면 권한이 있는 Gemini+Google Search 리서치 에이전트로 누락된 요청 항목을 새로 조사할 수 있습니다.",
            sort_order=30,
        ),
        _tool(
            tool_id="read_official_macro_research",
            category="macro",
            tier="optional",
            description="Browse official BOK and St. Louis Fed macro research documents, then read the original source text by source_doc_id for forward-looking policy, credit, labor, productivity, housing, liquidity, and external-sector trend signals before they are obvious in prices or indicators.",
            label_ko="공식 거시 연구 읽기",
            description_ko="BOK와 St. Louis Fed의 공식 이슈노트·연구자료·논문을 룰 기반 메타데이터로 리스트업합니다. 목록에서 source_doc_id를 고른 뒤 같은 도구로 원문 링크를 드릴다운해 읽습니다. 요약자료가 아니라 전문을 통해 아직 가격·지표·섹터 로테이션에 충분히 선반영되지 않은 정책·신용·소비·노동·생산성·주택·유동성 변화의 초기 단서를 볼 때 사용합니다. 어떤 문서를 읽을지는 에이전트가 자유롭게 고릅니다.",
            sort_order=32,
        ),
        _tool(
            tool_id="scratch_run_python",
            category="analysis",
            tier="core",
            description="Temporary Python scratch workspace.",
            label_ko="파이썬 낙서장",
            description_ko="현재 사이클 안에서만 유지되는 임시 Python 작업 공간입니다.",
            sort_order=35,
        ),
        _tool(
            tool_id="portfolio_diagnosis",
            category="quant",
            tier="core",
            description="Diagnose current holdings, concentration, risk contribution, drawdown, and benchmark context.",
            label_ko="포트폴리오 진단",
            description_ko="현재 보유 종목의 집중도(HHI), 개별 리스크 기여도, 최대 낙폭(MDD), 가중 모멘텀/변동성, 벤치마크 대비 초과수익, 보유종목별 joint-policy ranker 점수를 종합 진단합니다. 진단 전용 — 리밸런싱이 필요하면 optimize_portfolio를 사용하세요.",
            sort_order=40,
        ),
        _tool(
            tool_id="trade_performance",
            category="context",
            tier="optional",
            description="Analyze closed round-trip trades and current unrealized P&L.",
            label_ko="매매 성과 분석",
            description_ko="과거 라운드트립 매매의 승률·평균수익률·보유기간·행동 패턴(처분 효과, 포지션 크기별 승률)을 분석하고, 현재 미실현 손익도 함께 제공합니다. 자기 매매 패턴을 정량적으로 점검하여 의사결정을 보정할 때 사용합니다.",
            sort_order=55,
        ),
        _tool(
            tool_id="recommend_opportunities",
            category="quant",
            tier="optional",
            callable=qt.recommend_opportunities,
            description="Find fresh buy and replacement ideas across the runtime universe.",
            label_ko="통합 기회 추천",
            description_ko="런타임 유니버스에서 신규 매수 후보, 포트폴리오 교체 후보, 약한 보유종목의 대체 아이디어를 찾는 고수준 discovery 도구입니다. shared prep에서 계산한 regularized joint-policy ranker 점수를 사용해 모멘텀·눌림목·평균회귀·저변동성·센티먼트·forecast·RSI/MA/볼린저·EP/BP/SP/ROE/성장/부채 signal을 조합합니다. 전체 top_n 추천과 공격형/균형형/방어형/가치형/전술형 profile 문맥을 함께 반환하며, action·model confidence·risk note·signal별 joint-policy 기여도로 왜 해당 종목이 올라왔는지 설명합니다. freshness는 시장 캘린더 기준으로 판단합니다. 주말/휴일의 직전 거래일 데이터는 freshness metadata와 함께 허용하고, 장중인데 해당 세션 prep이 아직 없으면 status='degraded', 정말 오래된 데이터는 status='unusable'로 명시합니다.",
            sort_order=105,
        ),
        _tool(
            tool_id="screen_market",
            category="quant",
            tier="optional",
            callable=qt.screen_market,
            description="Low-level diagnostic candidate generator used by recommend_opportunities.",
            label_ko="시장 스크리닝",
            description_ko="원시 버킷 스크린을 점검하는 저수준 진단 도구입니다. discovery bucket의 screen-only 결과를 반환하며 joint-policy confidence나 signal별 ranker 기여도는 제공하지 않습니다.",
            enabled=False,
            sort_order=110,
        ),
        _tool(
            tool_id="optimize_portfolio",
            category="quant",
            tier="optional",
            callable=qt.optimize_portfolio,
            description="Answer how much of each ticker to hold using portfolio optimization.",
            label_ko="포트폴리오 최적화",
            description_ko="한 가지 질문에 답합니다: 각 종목을 얼마나 담을지. 보유+후보 바스켓의 목표 비중과 리밸런스 주문(BUY/SELL+비중)을 생성합니다. forecast-enhanced, 샤프 극대화, HRP 최적화 모드를 지원합니다. 데이터 품질이 나쁘면 graceful degrade — 히스토리 부족 종목은 data_quality.excluded에 리포트, forecast coverage<50%면 HRP로 fallback, 단일 종목만 usable하면 weight=1.0. 제약 옵션: max_weight(종목당 상한, 예 0.35), min_weight(하한 drop, 예 0.02), cash_buffer(현금 유보 0.0~0.5). regime_scale(0.3~1.0) 리스크오프 축소. 출력: weights, rebalance_orders, backtest_mdd, data_quality, status(ok/degraded/unusable), decision_summary(headline_code+turnover+confidence), evidence_gaps.",
            sort_order=120,
        ),
        _tool(
            tool_id="forecast_returns",
            category="quant",
            tier="optional",
            callable=qt.forecast_returns,
            description="Run time-series model forecasts and summarize model-implied return direction.",
            label_ko="수익률 예측",
            description_ko="7가지 시계열 모델(ARIMA, ETS, Prophet 등)을 동시에 돌려 각 종목의 모델상 방향 확률, 투표 수, model_direction 라벨을 요약합니다. ticker를 명시하지 않으면 방금 탐색한 self-discovered 후보 바스켓과 현재 보유 종목을 기본 분석 대상으로 사용합니다. 여러 후보의 기대수익률을 한눈에 비교하여 의사결정의 정량적 참고값으로 활용합니다.",
            sort_order=130,
        ),
        _tool(
            tool_id="technical_signals",
            category="quant",
            tier="optional",
            callable=qt.technical_signals,
            description="Return technical signals such as RSI, MACD, Bollinger Bands, trend, and volume confirmation.",
            label_ko="기술 지표 분석",
            description_ko="RSI, MACD, 볼린저 밴드, 이동평균 추세를 계산하고, 거래량 분석(거래량 비율·OBV 추세·가격-거래량 확인)도 수행합니다. KOSPI 종목은 외국인·기관 순매수 수급 신호가 추가됩니다. 매매 타이밍 판단의 기술적 근거를 제공합니다.",
            sort_order=150,
        ),
        _tool(
            tool_id="sector_summary",
            category="quant",
            tier="optional",
            callable=qt.sector_summary,
            description="Summarize sector rotation, leaders, laggards, and capital flow context.",
            label_ko="섹터 요약",
            description_ko="현재 유니버스를 섹터별로 그룹화하여 어떤 업종이 주도하고, 어떤 업종이 뒤처지며, 어디로 자본이 유입되고 있는지 섹터 로테이션 현황을 한눈에 요약합니다.",
            sort_order=170,
        ),
        _tool(
            tool_id="get_fundamentals",
            category="quant",
            tier="optional",
            callable=qt.get_fundamentals,
            description="Fetch valuation and fundamental metrics for a basket of tickers.",
            label_ko="펀더멘탈 조회",
            description_ko="후보 종목의 밸류에이션 지표를 일괄 조회합니다. US 종목은 PER·PBR·EPS·BPS, KOSPI 종목은 EPS·BPS·ROE·부채비율·성장성 지표를 제공합니다. 기술적 분석과 함께 펀더멘탈 관점의 균형 잡힌 판단을 돕습니다.",
            sort_order=180,
        ),
        _tool(
            tool_id="index_snapshot",
            category="macro",
            tier="optional",
            callable=qt.index_snapshot,
            description="Fetch latest quotes and returns for market indices, commodities, and bond yields.",
            label_ko="시장지수 조회",
            description_ko="주요 시장지수, 원자재(금·유가 등), 채권 수익률의 최신 시세와 수익률을 조회합니다. indices 파라미터를 비워두면 에이전트의 타겟 마켓(US/KOSPI)에 맞는 지표가 자동 선택됩니다.",
            sort_order=190,
        ),
        _tool(
            tool_id="fear_greed_index",
            category="macro",
            tier="optional",
            callable=st.fear_greed_index,
            description="Build a composite market regime indicator from volatility, breadth, momentum, and flows.",
            label_ko="시장 레짐 지표",
            description_ko="변동성 지수(VIX/VKOSPI), 시장 breadth, 모멘텀 추세, 기관 수급을 종합한 복합 시장 레짐 지표입니다. 0=극단적 공포(risk-off), 100=극단적 탐욕(risk-on)으로 표시하며, risk_on/neutral/risk_off 라벨과 서브컴포넌트 점수를 반환합니다. 포트폴리오 최적화의 regime_scale과 연계할 수 있습니다.",
            sort_order=210,
        ),
        _tool(
            tool_id="earnings_calendar",
            category="macro",
            tier="optional",
            callable=st.earnings_calendar,
            description="Fetch upcoming earnings and dividend events.",
            label_ko="이벤트 캘린더",
            description_ko="다가오는 실적 발표와 배당 이벤트를 조회합니다. US 종목은 Nasdaq 어닝 캘린더, KOSPI 종목은 KIS 배당 일정과 컨센서스 추정실적을 제공합니다. 이벤트 전후 리스크 관리에 활용합니다.",
            sort_order=220,
        ),
        _tool(
            tool_id="fetch_reddit_sentiment",
            category="sentiment",
            tier="optional",
            callable=st.fetch_reddit_sentiment,
            description="Fetch recent Reddit posts from finance subreddits for retail sentiment.",
            label_ko="레딧 여론 수집",
            description_ko="레딧의 금융 서브레딧(r/wallstreetbets, r/stocks 등)에서 최근 게시글과 댓글을 수집하여 개인 투자자 심리를 파악합니다. 밈 주식 열풍이나 소셜 모멘텀을 포착할 때 유용합니다.",
            enabled=bool(settings.reddit_sentiment_enabled),
            sort_order=230,
        ),
        _tool(
            tool_id="fetch_sec_filings",
            category="sentiment",
            tier="optional",
            callable=st.fetch_sec_filings,
            description="Fetch recent SEC filings from EDGAR.",
            label_ko="SEC 공시 조회",
            description_ko="EDGAR에서 최근 SEC 공시(10-K 연간보고서, 10-Q 분기보고서, 8-K 수시공시 등)를 조회합니다. 공시 종류·제출일·제목을 반환하여 중요한 기업 이벤트를 빠르게 확인할 수 있습니다.",
            sort_order=240,
        ),
        _tool(
            tool_id="macro_snapshot",
            category="macro",
            tier="optional",
            callable=mt.macro_snapshot,
            description="Fetch a macro snapshot adapted to the agent market.",
            label_ko="거시 지표 조회",
            description_ko="에이전트의 타겟 마켓에 맞는 확장 거시경제 지표를 일괄 조회합니다. US: 정책금리·SOFR·국채곡선·물가/PCE·고용·GDP/생산·유동성/신용·시장·원자재·환율·주택(FRED 데이터). KR: 기준금리·시장/여수신금리·통화/신용·환율·주식/채권·성장·생산·소비·투자·심리·고용·대외/무역·물가·부동산·원자재(ECOS 데이터). 주요 YoY와 스프레드도 함께 반환합니다.",
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
        description = str(row.get("ui_description") or row.get("description") or "").strip()
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
        enabled = bool(overlay["enabled"])
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
    mt = MacroTools(settings=settings, repo=repo)
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
