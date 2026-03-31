from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.memory.cleanup import run_memory_cleanup
from arena.memory.policy import (
    GLOBAL_MEMORY_PROMPT_CONFIG_KEY,
    MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY,
    MEMORY_POLICY_CONFIG_KEY,
    build_memory_graph,
    default_memory_policy,
    load_global_compaction_prompt,
    load_tenant_compaction_prompt,
    load_memory_policy,
    normalize_memory_policy,
    serialize_memory_policy,
)
from arena.ui.http import html_response, json_response

_VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
_THREE_JS_PATH = _VENDOR_DIR / "three.min.js"
_FORCE_GRAPH_JS_PATH = _VENDOR_DIR / "3d-force-graph.min.js"


def _load_json_config(repo: BigQueryRepository, *, tenant_id: str, config_key: str) -> dict[str, Any]:
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return {}
    try:
        raw = getter(tenant_id, config_key)
    except Exception:
        return {}
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _stats_payload(repo: BigQueryRepository, *, tenant_id: str, trading_mode: str) -> dict[str, Any]:
    sql = f"""
    SELECT event_type, agent_id, COUNT(1) AS cnt, MAX(created_at) AS last_created_at
    FROM `{repo.dataset_fqn}.agent_memory_events`
    WHERE tenant_id = @tenant_id
      AND trading_mode = @trading_mode
    GROUP BY event_type, agent_id
    """
    rows = repo.fetch_rows(sql, {"tenant_id": tenant_id, "trading_mode": trading_mode})
    counts_by_event_type: dict[str, int] = {}
    counts_by_agent: dict[str, int] = {}
    counts_by_agent_event_type: dict[str, dict[str, int]] = {}
    last_created_at = ""
    total_events = 0
    for row in rows:
        event_type = str(row.get("event_type") or "").strip().lower()
        agent_id = str(row.get("agent_id") or "").strip().lower()
        count = int(row.get("cnt") or 0)
        if not event_type or count <= 0:
            continue
        total_events += count
        counts_by_event_type[event_type] = counts_by_event_type.get(event_type, 0) + count
        if agent_id:
            counts_by_agent[agent_id] = counts_by_agent.get(agent_id, 0) + count
            bucket = counts_by_agent_event_type.setdefault(agent_id, {})
            bucket[event_type] = bucket.get(event_type, 0) + count
        created_at = str(row.get("last_created_at") or "").strip()
        if created_at and created_at > last_created_at:
            last_created_at = created_at
    coverage_rows = repo.fetch_rows(
        f"""
        SELECT
          COUNT(1) AS total_memory_events,
          COUNTIF(TRIM(COALESCE(graph_node_id, '')) != '') AS with_graph_node_id,
          COUNTIF(TRIM(COALESCE(causal_chain_id, '')) != '') AS with_causal_chain_id,
          COUNTIF(last_accessed_at IS NOT NULL) AS with_last_accessed_at,
          COUNTIF(effective_score IS NOT NULL) AS with_effective_score,
          MAX(last_accessed_at) AS last_accessed_at
        FROM `{repo.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    coverage_row = coverage_rows[0] if coverage_rows else {}
    total_memory_events = int(coverage_row.get("total_memory_events") or total_events or 0)
    with_graph_node_id = int(coverage_row.get("with_graph_node_id") or 0)
    with_causal_chain_id = int(coverage_row.get("with_causal_chain_id") or 0)
    with_last_accessed_at = int(coverage_row.get("with_last_accessed_at") or 0)
    with_effective_score = int(coverage_row.get("with_effective_score") or 0)

    tier_rows = repo.fetch_rows(
        f"""
        SELECT memory_tier, COUNT(1) AS cnt
        FROM `{repo.dataset_fqn}.agent_memory_events`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        GROUP BY memory_tier
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    counts_by_memory_tier: dict[str, int] = {}
    for row in tier_rows:
        memory_tier = str(row.get("memory_tier") or "").strip().lower()
        count = int(row.get("cnt") or 0)
        if memory_tier and count > 0:
            counts_by_memory_tier[memory_tier] = count

    access_rows = repo.fetch_rows(
        f"""
        SELECT
          COUNT(1) AS access_event_count,
          COUNTIF(COALESCE(used_in_prompt, FALSE)) AS prompt_use_count,
          MAX(accessed_at) AS last_accessed_at
        FROM `{repo.dataset_fqn}.memory_access_events`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    access_row = access_rows[0] if access_rows else {}

    graph_node_rows = repo.fetch_rows(
        f"""
        SELECT node_kind, COUNT(1) AS cnt, MAX(created_at) AS last_created_at
        FROM `{repo.dataset_fqn}.memory_graph_nodes`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        GROUP BY node_kind
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    counts_by_node_kind: dict[str, int] = {}
    last_graph_node_at = ""
    total_graph_nodes = 0
    for row in graph_node_rows:
        node_kind = str(row.get("node_kind") or "").strip().lower()
        count = int(row.get("cnt") or 0)
        if not node_kind or count <= 0:
            continue
        total_graph_nodes += count
        counts_by_node_kind[node_kind] = count
        created_at = str(row.get("last_created_at") or "").strip()
        if created_at and created_at > last_graph_node_at:
            last_graph_node_at = created_at

    graph_edge_rows = repo.fetch_rows(
        f"""
        SELECT edge_type, COUNT(1) AS cnt, MAX(created_at) AS last_created_at
        FROM `{repo.dataset_fqn}.memory_graph_edges`
        WHERE tenant_id = @tenant_id
          AND trading_mode = @trading_mode
        GROUP BY edge_type
        """,
        {"tenant_id": tenant_id, "trading_mode": trading_mode},
    )
    counts_by_edge_type: dict[str, int] = {}
    last_graph_edge_at = ""
    total_graph_edges = 0
    for row in graph_edge_rows:
        edge_type = str(row.get("edge_type") or "").strip().upper()
        count = int(row.get("cnt") or 0)
        if not edge_type or count <= 0:
            continue
        total_graph_edges += count
        counts_by_edge_type[edge_type] = count
        created_at = str(row.get("last_created_at") or "").strip()
        if created_at and created_at > last_graph_edge_at:
            last_graph_edge_at = created_at

    return {
        "tenant_id": tenant_id,
        "total_events": total_events,
        "counts_by_event_type": counts_by_event_type,
        "counts_by_memory_tier": counts_by_memory_tier,
        "counts_by_agent": counts_by_agent,
        "counts_by_agent_event_type": counts_by_agent_event_type,
        "last_created_at": last_created_at,
        "memory_runtime": {
            "total_memory_events": total_memory_events,
            "with_graph_node_id": with_graph_node_id,
            "with_causal_chain_id": with_causal_chain_id,
            "with_last_accessed_at": with_last_accessed_at,
            "with_effective_score": with_effective_score,
            "graph_node_coverage": _ratio(with_graph_node_id, total_memory_events),
            "causal_chain_coverage": _ratio(with_causal_chain_id, total_memory_events),
            "last_accessed_at": str(coverage_row.get("last_accessed_at") or "").strip(),
        },
        "access_runtime": {
            "access_event_count": int(access_row.get("access_event_count") or 0),
            "prompt_use_count": int(access_row.get("prompt_use_count") or 0),
            "last_accessed_at": str(access_row.get("last_accessed_at") or "").strip(),
        },
        "graph_runtime": {
            "total_nodes": total_graph_nodes,
            "total_edges": total_graph_edges,
            "counts_by_node_kind": counts_by_node_kind,
            "counts_by_edge_type": counts_by_edge_type,
            "last_node_created_at": last_graph_node_at,
            "last_edge_created_at": last_graph_edge_at,
        },
    }


def _memory_defaults(settings: Settings) -> dict[str, Any]:
    defaults = default_memory_policy(
        context_limit=settings.context_max_memory_events,
        compaction_enabled=settings.memory_compaction_enabled,
        cycle_event_limit=settings.memory_compaction_cycle_event_limit,
        recent_lessons_limit=settings.memory_compaction_recent_lessons_limit,
        max_reflections=settings.memory_compaction_max_reflections,
    )
    current = getattr(settings, "memory_policy", None)
    if isinstance(current, dict) and current:
        return normalize_memory_policy(current, defaults=defaults)
    return defaults


def _graph_payload(
    repo: BigQueryRepository,
    settings: Settings,
    *,
    tenant_id: str,
    cached_fetch: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    tenant = str(tenant_id or "").strip().lower() or "local"
    defaults = _memory_defaults(settings)
    policy = load_memory_policy(repo, tenant, defaults=defaults)
    tenant_prompt = load_tenant_compaction_prompt(repo, tenant)
    global_prompt = load_global_compaction_prompt(repo)
    effective_prompt = tenant_prompt or global_prompt
    prompt_source = "tenant" if tenant_prompt else "global"
    stats_key = f"memory_stats:{tenant}:{settings.trading_mode}"
    if callable(cached_fetch):
        stats = cached_fetch(stats_key, _stats_payload, repo, tenant_id=tenant, trading_mode=settings.trading_mode)
    else:
        stats = _stats_payload(repo, tenant_id=tenant, trading_mode=settings.trading_mode)
    tuning_state = _load_json_config(
        repo,
        tenant_id=tenant,
        config_key=MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY,
    )
    graph = build_memory_graph(
        policy,
        tenant_id=tenant,
        stats=stats,
        tenant_compaction_prompt=tenant_prompt,
        global_compaction_prompt=global_prompt,
        effective_compaction_prompt=effective_prompt,
        prompt_source=prompt_source,
        compaction_prompt_editable=True,
    )
    meta = graph.setdefault("meta", {})
    meta["stats"] = stats
    meta["runtime"] = {
        "trading_mode": str(settings.trading_mode or "").strip().lower() or "paper",
        "memory": stats.get("memory_runtime") or {},
        "access": stats.get("access_runtime") or {},
        "graph": stats.get("graph_runtime") or {},
        "event_types": stats.get("counts_by_event_type") or {},
        "memory_tiers": stats.get("counts_by_memory_tier") or {},
        "forgetting_tuning_state": tuning_state,
    }
    return graph


def build_memory_settings_panel(
    repo: BigQueryRepository,
    settings: Settings,
    *,
    tenant_id: str,
    cached_fetch: Callable[..., Any] | None = None,
) -> str:
    graph = _graph_payload(repo, settings, tenant_id=tenant_id, cached_fetch=cached_fetch)
    return '<div data-settings-panel="memory" class="settings-panel hidden">' + _memory_page_html(graph) + "</div>"


def _memory_page_html(graph_payload: dict[str, Any]) -> str:
    initial_json = json.dumps(graph_payload, ensure_ascii=False).replace("<", "\\u003c")
    initial_node = next(
        (node for node in (graph_payload.get("nodes") or []) if str(node.get("id") or "") == "root"),
        ((graph_payload.get("nodes") or [None])[0] if (graph_payload.get("nodes") or []) else None),
    )
    initial_title = "메모리 정책" if str((initial_node or {}).get("id") or "") == "root" else html.escape(str((initial_node or {}).get("label") or "Memory System"))
    initial_desc = ""
    initial_form = '<div class="py-2"></div>'
    return (
        '<section class="overflow-hidden rounded-[30px] border border-ink-200/70 bg-white/95 shadow-sm backdrop-blur-md">'
        '<div class="grid gap-4 p-4 xl:grid-cols-[minmax(0,1fr)_296px]">'
        '<div class="min-w-0 rounded-[28px] border border-ink-200/80 bg-white shadow-sm">'
        '<div class="flex flex-wrap items-center justify-between gap-3 border-b border-ink-200/80 px-4 py-3">'
        '<div>'
        '<h2 class="font-display text-xl font-bold tracking-tight text-ink-900">Memory Policy Graph</h2>'
        '</div>'
        '<span id="memory-save-status" class="hidden rounded-full border px-3 py-1 text-xs font-semibold transition-all duration-300"></span>'
        '</div>'
        '<div class="relative min-w-0 overflow-hidden rounded-b-[28px] bg-[linear-gradient(135deg,#ffffff_0%,#f7fbff_52%,#eef5ff_100%)]">'
        '<div id="memory-graph" class="h-[440px] w-full min-w-0 xl:h-[460px]"></div>'
        '</div>'
        '</div>'
        '<aside id="memory-panel" class="relative xl:sticky xl:top-24 xl:max-h-[calc(100vh-7rem)] xl:overflow-y-auto styled-scrollbar">'
        '<div id="memory-panel-loading" class="pointer-events-none absolute inset-0 z-20 hidden items-center justify-center rounded-[28px] bg-white/70 backdrop-blur-[2px]">'
        '<div class="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 shadow-sm">'
        '<span class="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-slate-300 border-t-slate-700"></span>'
        '<span id="memory-panel-loading-text">저장 중...</span>'
        '</div>'
        '</div>'
        '<div class="overflow-hidden rounded-[28px] border border-slate-200/60 bg-white shadow-[0_1px_3px_rgba(15,23,42,0.04),0_12px_40px_rgba(15,23,42,0.06)]">'
        '<div id="memory-panel-header" class="relative overflow-hidden px-6 pb-5 pt-6">'
        '<div class="absolute inset-0 bg-gradient-to-br from-slate-50/80 via-white to-slate-50/40"></div>'
        '<div class="relative">'
        '<div id="memory-panel-breadcrumb"></div>'
        '<div class="flex items-center gap-2.5">'
        '<span id="memory-panel-dot" class="h-2.5 w-2.5 rounded-full bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.35)] transition-colors duration-300"></span>'
        f'<h3 id="memory-panel-title" class="font-display text-[1.3rem] font-bold leading-tight tracking-tight text-slate-900">{initial_title}</h3>'
        '</div>'
        f'<p id="memory-panel-desc" class="hidden mt-2.5 text-[13px] leading-relaxed text-slate-500">{initial_desc}</p>'
        '</div>'
        '</div>'
        '<div class="h-px bg-gradient-to-r from-transparent via-slate-200/80 to-transparent"></div>'
        f'<div id="memory-panel-form" class="px-5 py-5">{initial_form}</div>'
        '</div>'
        '</aside>'
        '</div>'
        '</section>'
        '<script src="/assets/vendor/three.min.js"></script>'
        '<script src="/assets/vendor/3d-force-graph.min.js"></script>'
        '<script>'
        '(function(){'
        f'const INITIAL={initial_json};'
        'const graphHost=document.getElementById("memory-graph");'
        'const statusEl=document.getElementById("memory-save-status");'
        'const titleEl=document.getElementById("memory-panel-title");'
        'const descEl=document.getElementById("memory-panel-desc");'
        'const formEl=document.getElementById("memory-panel-form");'
        'const colors={root:"#0891b2",storage:"#2563eb",event_types:"#0f766e",hierarchy:"#22c55e",tagging:"#14b8a6",forgetting:"#f59e0b",graph:"#ef4444",compaction:"#7c3aed",retrieval:"#ea580c",react_injection:"#dc2626",cleanup:"#64748b"};'
        'const labelMap={root:"메모리 정책",storage:"저장소","storage.bigquery":"기록 저장소","storage.firestore":"검색 저장소",event_types:"기억 유형",hierarchy:"기억 계층",tagging:"상황 태그",forgetting:"망각 관리","forgetting.tuning":"자동 보정",graph:"인과 그래프",compaction:"회고 압축",retrieval:"기억 검색","retrieval.vector_search":"유사 검색","retrieval.reranking":"재정렬",react_injection:"도구 메모리 주입","react_injection.tools":"대상 도구",cleanup:"정리 정책","storage.embed_cache_max":"검색 캐시 개수","event_types.trade_execution":"거래 기록","event_types.strategy_reflection":"전략 교훈","event_types.manual_note":"수동 메모","event_types.react_tools_summary":"도구 요약","event_types.thesis_open":"투자 논리 시작","event_types.thesis_update":"투자 논리 갱신","event_types.thesis_invalidated":"투자 논리 무효화","event_types.thesis_realized":"투자 논리 실현 종료","hierarchy.enabled":"계층 사용","hierarchy.working_ttl_hours":"작업 메모 보관 시간","hierarchy.episodic_ttl_days":"사례 메모 보관 일수","hierarchy.semantic_promotion_min_support":"교훈 승격 최소 반복 수","tagging.enabled":"상황 태그 사용","tagging.max_tags":"태그 최대 개수","tagging.regime_bonus":"장세 일치 가산점","tagging.strategy_bonus":"전략 일치 가산점","tagging.sector_bonus":"섹터 일치 가산점","forgetting.enabled":"망각 관리 사용","forgetting.access_log_enabled":"조회 기록 저장","forgetting.default_decay_factor":"기본 감쇠 속도","forgetting.access_curve":"접근 보호 곡선","forgetting.tier_weight_working":"작업 메모 감쇠 강도","forgetting.tier_weight_episodic":"사례 메모 감쇠 강도","forgetting.tier_weight_semantic":"교훈 메모 감쇠 강도","forgetting.min_effective_score":"최소 유지 점수","forgetting.tuning.enabled":"자동 보정 사용","forgetting.tuning.mode":"적용 모드","forgetting.tuning.lookback_days":"최근 데이터 범위","forgetting.tuning.stability_window_days":"안정성 확인 기간","forgetting.tuning.min_access_events":"최소 조회 수","forgetting.tuning.min_prompt_uses":"최소 실제 사용 수","forgetting.tuning.min_unique_memories":"최소 메모리 수","forgetting.tuning.ema_alpha":"반영 비율","forgetting.tuning.max_decay_factor_delta":"감쇠 속도 최대 변화폭","forgetting.tuning.max_min_effective_score_delta":"최소 점수 최대 변화폭","forgetting.tuning.max_tier_weight_delta":"강도 최대 변화폭","forgetting.tuning.objective_topk":"평가 상위 범위","forgetting.tuning.auto_promote_enabled":"자동 반영 시작","forgetting.tuning.auto_promote_min_shadow_days":"추천 모드 최소 일수","forgetting.tuning.auto_promote_min_shadow_runs":"추천 모드 최소 실행 수","forgetting.tuning.auto_promote_required_stable_runs":"필요 안정 실행 수","forgetting.tuning.auto_promote_required_improving_runs":"필요 개선 실행 수","forgetting.tuning.auto_promote_max_recommendation_drift":"허용 최대 흔들림","forgetting.tuning.auto_demote_enabled":"자동 되돌리기","forgetting.tuning.auto_demote_unhealthy_runs":"되돌리기 대기 횟수","graph.enabled":"그래프 사용","graph.max_expansion_hops":"연결 확장 단계","graph.max_expanded_nodes":"최대 연결 메모 수","graph.inferred_edge_min_confidence":"추정 연결 최소 신뢰도","compaction.enabled":"회고 압축 사용","compaction.cycle_event_limit":"입력 기록 수","compaction.recent_lessons_limit":"최근 교훈 수","compaction.max_reflections":"최대 회고 수","compaction.thesis_chain_enabled":"닫힌 논리 체인 우선","compaction.thesis_chain_max_chains_per_cycle":"사이클당 체인 수","compaction.thesis_chain_max_events_per_chain":"체인당 이벤트 수","compaction.global_prompt":"회고 정리 안내문","retrieval.context_limit":"컨텍스트 메모 수","retrieval.vector_search_enabled":"유사 검색 사용","retrieval.vector_search_limit":"후보 수","retrieval.peer_lessons_enabled":"피어 교훈 포함","retrieval.reranking.type_bonus_reflection":"교훈 가산점","retrieval.reranking.type_bonus_trade":"거래 가산점","retrieval.reranking.type_bonus_manual":"수동 메모 가산점","retrieval.reranking.type_bonus_react_tools":"도구 요약 가산점","retrieval.reranking.recency_bonus_3d":"최근 3일 가산점","retrieval.reranking.recency_bonus_14d":"최근 14일 가산점","retrieval.reranking.recency_bonus_45d":"최근 45일 가산점","retrieval.reranking.ticker_bonus_base":"종목 기본 가산점","retrieval.reranking.ticker_bonus_step":"종목 추가 가산점","retrieval.reranking.ticker_bonus_max":"종목 가산점 상한","retrieval.reranking.outcome_bonus_max":"성과 가산점 상한","retrieval.reranking.effective_score_bonus_scale":"유지 점수 가산 강도","retrieval.reranking.effective_score_bonus_cap":"유지 점수 가산 상한","react_injection.enabled":"도구 메모리 주입 사용","react_injection.tools.technical_signals":"기술 지표","react_injection.tools.screen_market":"시장 스크리닝","react_injection.tools.forecast_returns":"수익률 예측","react_injection.tools.get_fundamentals":"기본 정보","react_injection.tools.optimize_portfolio":"포트폴리오 조정","cleanup.enabled":"자동 정리 사용","cleanup.max_age_days":"최대 보관 일수","cleanup.min_score":"정리 기준 점수"};'
        'const state={graph:INITIAL,selectedId:"root",hasFramed:false,lastCleanup:null,fitTimer:0,hoveredId:null};'
        'const visuals={};'
        'let fg=null;'
        'let animationFrame=0;'
        'function badge(level,text){if(!statusEl)return;statusEl.textContent=text;statusEl.className="rounded-full border px-3 py-1 text-xs font-semibold transition-all duration-300";statusEl.classList.remove("hidden");if(level==="ok"){statusEl.classList.add("border-emerald-200/60","bg-emerald-50/80","text-emerald-600");}else if(level==="busy"){statusEl.classList.add("border-slate-200/80","bg-white","text-slate-700");}else{statusEl.classList.add("border-red-200/60","bg-red-50/80","text-red-600");}if(level!=="busy"){setTimeout(function(){statusEl.classList.add("hidden");},3000);}}'
        'function setPanelBusy(on,text){const mask=document.getElementById("memory-panel-loading");const textEl=document.getElementById("memory-panel-loading-text");if(textEl&&text){textEl.textContent=text;}if(mask){mask.classList.toggle("hidden",!on);mask.classList.toggle("flex",!!on);}document.querySelectorAll("#memory-panel button, #memory-panel input, #memory-panel textarea").forEach(function(el){if(on){el.setAttribute("disabled","disabled");el.classList.add("opacity-60","cursor-not-allowed");}else{el.removeAttribute("disabled");el.classList.remove("opacity-60","cursor-not-allowed");}});if(on){badge("busy",text||"저장 중...");}}'
        'function escapeHtml(value){return String(value||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");}'
        'function runtimeMeta(){return (((state.graph||{}).meta||{}).runtime)||{};}'
        'function tuningState(){return (runtimeMeta().forgetting_tuning_state)||{};}'
        'function formatNumber(value){const num=Number(value||0);return Number.isFinite(num)?num.toLocaleString("ko-KR"):"0";}'
        'function formatRatio(numerator,denominator){const den=Number(denominator||0);if(!(den>0)){return "0%";}return `${Math.round((Number(numerator||0)/den)*100)}%`; }'
        'function formatTimestamp(value){const text=String(value||"").trim();return text?text.replace("T"," ").replace("Z"," UTC").slice(0,19):"없음";}'
        'function prettyOption(value){const token=String(value||"").trim();const map={paper:"모의투자",live:"실거래",shadow:"추천만 보기",bounded_ema:"자동 반영",sqrt:"자주 쓸수록 천천히 약해짐",log:"많이 써도 완만하게 보호",capped_linear:"선형 보호(상한 있음)"};return map[token]||token||"-";}'
        'function prettyNodeKind(value){const token=String(value||"").trim().toLowerCase();const map={memory_event:"메모리",order_intent:"주문 의도",execution_report:"체결 결과",board_post:"게시판 글",research_briefing:"리서치 요약"};return map[token]||token.replace(/_/g," ");}'
        'function prettyEdgeType(value){const token=String(value||"").trim().toUpperCase();const map={INFORMED_BY:"근거로 연결",PRECEDES:"앞뒤 흐름",EXECUTED_AS:"주문에서 체결로",RESULTED_IN:"결과로 이어짐",ABSTRACTED_TO:"교훈으로 정리",REFERENCES:"참조",FEEDBACK_TO:"피드백 연결"};return map[token]||token.replace(/_/g," ");}'
        'function prettyTransitionAction(value){const token=String(value||"").trim();const map={auto_promote:"자동 반영 시작",auto_demote:"추천만 보기로 전환"};return map[token]||token||"-";}'
        'function statusPill(label,value,tone){return `<div class="rounded-[14px] border px-3 py-2 ${tone||"border-slate-100 bg-slate-50"}"><p class="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">${escapeHtml(label)}</p><p class="mt-1 text-sm font-semibold text-slate-800">${escapeHtml(value)}</p></div>`;}'
        'function graphStatusCard(){const runtime=runtimeMeta();const memory=runtime.memory||{};return `<div class="mt-5 grid gap-3 sm:grid-cols-2"><div class="rounded-[18px] border border-rose-100 bg-white px-5 py-4"><p class="text-[11px] font-semibold tracking-[0.08em] text-slate-400">메모리 연결률</p><p class="mt-2 text-3xl font-bold tracking-tight text-slate-900">${formatRatio(memory.with_graph_node_id||0,memory.total_memory_events||0)}</p></div><div class="rounded-[18px] border border-rose-100 bg-white px-5 py-4"><p class="text-[11px] font-semibold tracking-[0.08em] text-slate-400">흐름 연결률</p><p class="mt-2 text-3xl font-bold tracking-tight text-slate-900">${formatRatio(memory.with_causal_chain_id||0,memory.total_memory_events||0)}</p></div></div>`;}'
        'function forgettingStatusCard(){const runtime=runtimeMeta();const access=runtime.access||{};const memory=runtime.memory||{};const tuning=tuningState();const drift=((tuning.drift||{}).recommendation_drift);const transition=tuning.transition||{};const history=tuning.history||{};const mode=String(tuning.effective_mode||tuning.configured_mode||"shadow");const action=String(transition.action||"");const reason=String(transition.reason||"");return `<div class="mt-5 rounded-[20px] border border-amber-100 bg-amber-50/40 p-5"><div class="flex items-center justify-between gap-3"><div><h4 class="text-sm font-semibold text-slate-800">오래된 메모리 관리 상태</h4><p class="mt-1 text-[11px] text-slate-500">자주 쓰는 메모리는 더 오래 남기고, 덜 쓰는 메모리는 뒤로 보내는 현재 상태입니다.</p></div><span class="rounded-full bg-white px-3 py-1 text-[10px] font-bold uppercase tracking-[0.18em] text-amber-600 ring-1 ring-amber-100">${escapeHtml(prettyOption(mode))}</span></div><div class="mt-4 grid gap-2 sm:grid-cols-2">${statusPill("조회 기록",formatNumber(access.access_event_count||0),"border-white bg-white")}${statusPill("실제 사용 기록",formatNumber(access.prompt_use_count||0),"border-white bg-white")}${statusPill("유지 점수 계산됨",formatRatio(memory.with_effective_score||0,memory.total_memory_events||0),"border-white bg-white")}${statusPill("최근 사용 시각",formatTimestamp(access.last_accessed_at),"border-white bg-white")}</div><div class="mt-4 rounded-[16px] border border-white/80 bg-white px-4 py-3 text-xs text-slate-600"><div class="flex flex-wrap items-center gap-2"><span class="font-semibold text-slate-800">저장된 설정:</span><span>${escapeHtml(prettyOption(tuning.configured_mode||"shadow"))}</span><span class="font-semibold text-slate-800">실제 적용:</span><span>${escapeHtml(prettyOption(mode))}</span><span class="font-semibold text-slate-800">최근 흔들림:</span><span>${escapeHtml(drift!=null?String(drift):"-")}</span></div><div class="mt-2 flex flex-wrap items-center gap-2"><span class="font-semibold text-slate-800">추천만 보기 실행:</span><span>${escapeHtml(String(history.shadow_runs_since_transition||0))}</span><span class="font-semibold text-slate-800">자동 반영 실행:</span><span>${escapeHtml(String(history.bounded_ema_runs_since_transition||0))}</span></div>${action?`<p class="mt-2 text-[11px] text-amber-700">최근 전환: ${escapeHtml(prettyTransitionAction(action))} · ${escapeHtml(reason||"-")}</p>`:`<p class="mt-2 text-[11px] text-slate-500">아직 자동 전환 이력이 없습니다.</p>`}</div></div>`;}'
        'function nodeColor(node){if(node&&node.type==="toggle"&&!node.value){return "#94a3b8";}return (node&&node.color)||colors[(node&&node.group)||"cleanup"]||"#64748b";}'
        'function getSelectedNode(){return state.graph.nodes.find(function(item){return item.id===state.selectedId;})||state.graph.nodes.find(function(item){return item.id==="root";})||state.graph.nodes[0]||null;}'
        'function sourceId(link){if(!link){return null;}return link.source&&typeof link.source==="object"?link.source.id:link.source;}'
        'function targetId(link){if(!link){return null;}return link.target&&typeof link.target==="object"?link.target.id:link.target;}'
        'function getChildNodes(node){if(!node||!node.id){return [];}const childIds={};(state.graph.links||[]).forEach(function(link){const source=sourceId(link);const target=targetId(link);if(source===node.id&&target){childIds[String(target)]=true;}});return (state.graph.nodes||[]).filter(function(item){return !!childIds[String(item.id)];});}'
        'function jumpToNode(nodeId){const next=state.graph.nodes.find(function(item){return item.id===nodeId;});if(!next){return false;}state.selectedId=next.id;renderPanel(next);renderGraph();return true;}'
        'document.addEventListener("click",function(ev){const btn=ev.target.closest("[data-node-select]");if(!btn){return;}ev.preventDefault();jumpToNode(btn.getAttribute("data-node-select"));});'
        'document.addEventListener("click",function(ev){const btn=ev.target.closest("[data-toggle-node]");if(!btn){return;}ev.preventDefault();ev.stopPropagation();quickToggleNode(btn.getAttribute("data-toggle-node"));});'
        'function displayLabel(node){if(!node){return "";}const key=(node.path&&labelMap[node.path])?node.path:((node.id&&labelMap[node.id])?node.id:"");return key?labelMap[key]:(node.label||node.id||"");}'
        'function panelTitle(node){if(!node){return "";}return displayLabel(node);}'
        'const descMap={root:"메모리 저장, 검색, 정리 방식을 한 화면에서 조정합니다.",storage:"메모리 저장소 구성을 정합니다.","storage.bigquery":"원본 기록 보관용 저장소입니다.","storage.firestore":"비슷한 메모리 검색용 저장소입니다.","storage.embed_cache_max":"자주 검색하는 메모리의 벡터 임베딩을 캐시에 보관하는 최대 개수입니다. 많을수록 검색이 빨라집니다.",event_types:"메모리로 남길 기록 종류를 정합니다.","event_types.trade_execution":"에이전트의 매수·매도 거래 결과를 메모리로 저장합니다. 켜면 이전 거래 경험을 토대로 더 나은 투자 판단을 합니다.","event_types.strategy_reflection":"투자 전략에 대한 교훈과 반성을 메모리로 저장합니다. 같은 실수를 반복하지 않도록 도와줍니다.","event_types.manual_note":"사용자가 직접 작성한 메모를 에이전트 메모리로 저장합니다. 에이전트에게 특별한 지시사항을 전달할 때 유용합니다.","event_types.react_tools_summary":"도구(기술 분석, 시장 스크리닝 등) 실행 결과의 요약을 메모리로 저장합니다. 이전 분석 결과를 다음 판단에 재활용할 수 있게 합니다.","event_types.thesis_open":"실제로 진입한 투자 논리를 새 메모리 체인으로 엽니다. 같은 종목을 왜 샀는지 나중에 그대로 복기할 수 있게 합니다.","event_types.thesis_update":"보유 논리의 중심이 바뀌었을 때 변화를 기록합니다. 처음과 지금의 투자 이유가 달라졌는지 추적하는 데 쓰입니다.","event_types.thesis_invalidated":"매도나 감축의 이유가 투자 논리 붕괴일 때 기록합니다. 손실 자체보다 어떤 전제가 깨졌는지를 남깁니다.","event_types.thesis_realized":"투자 논리가 의도대로 전개되어 정상 종료됐을 때 기록합니다. 목표 달성형 종료와 실패형 종료를 분리합니다.",hierarchy:"짧은 기록, 사례 기록, 장기 교훈을 나눕니다.","hierarchy.enabled":"메모리를 작업(단기)·사례(중기)·교훈(장기) 3단계로 나누어 관리합니다. 꺼두면 모든 메모리가 동일하게 취급됩니다.","hierarchy.working_ttl_hours":"작업 메모(오늘의 분석, 당일 판단 근거 등 단기 기록)가 자동 만료되는 시간입니다. 예: 24시간이면 하루가 지나면 사라집니다.","hierarchy.episodic_ttl_days":"사례 메모(특정 종목 거래 경험 등 중기 기록)가 자동 만료되는 일수입니다. 이 기간 안에 반복되면 장기 교훈으로 승격됩니다.","hierarchy.semantic_promotion_min_support":"사례 메모가 장기 교훈으로 승격되려면 최소 몇 번 비슷한 패턴이 반복되어야 하는지 설정합니다. 높을수록 충분히 검증된 패턴만 교훈이 됩니다.",tagging:"상황에 맞는 메모리를 더 잘 찾게 합니다.","tagging.enabled":"메모리에 장세·전략·섹터 태그를 자동으로 붙여, 현재 시장 상황과 비슷한 과거 메모리를 우선 검색합니다.","tagging.max_tags":"하나의 메모리에 붙일 수 있는 태그 최대 개수입니다. 너무 많으면 검색 정확도가 떨어질 수 있습니다.","tagging.regime_bonus":"현재 장세(상승·하락·횡보)와 같은 태그를 가진 메모리의 검색 가산점입니다. 높을수록 장세가 비슷한 과거 경험을 더 우선합니다.","tagging.strategy_bonus":"같은 투자 전략 태그를 가진 메모리의 검색 가산점입니다. 전략이 일치하는 과거 경험을 더 잘 찾게 합니다.","tagging.sector_bonus":"같은 업종·섹터 태그를 가진 메모리의 검색 가산점입니다. 동일 섹터 거래 경험을 더 잘 활용하게 합니다.",forgetting:"자주 쓰는 메모리는 남기고 덜 쓰는 메모리는 천천히 뒤로 보냅니다.","forgetting.enabled":"오래되고 에이전트가 잘 안 쓰는 메모리의 중요도를 시간이 지나면서 자동으로 낮춥니다. 꺼두면 모든 메모리가 영구히 같은 중요도를 유지합니다.","forgetting.access_log_enabled":"에이전트가 메모리를 조회할 때마다 기록을 남깁니다. 이 조회 기록을 바탕으로 자주 쓰는 메모리는 보호하고, 안 쓰는 메모리는 서서히 중요도를 낮춥니다.","forgetting.default_decay_factor":"메모리 중요도가 시간에 따라 줄어드는 속도입니다. 값이 클수록 빠르게 잊고, 작을수록 오래 기억합니다. (0~1 사이, 기본 0.02)","forgetting.access_curve":"자주 쓰는 메모리를 얼마나 더 오래 살릴지 정합니다.","forgetting.tier_weight_working":"작업 메모(단기 기록)의 감쇠 강도입니다. 높을수록 단기 메모가 더 빠르게 약해집니다. 단기 기록은 빨리 잊어도 되므로 보통 높게 설정합니다.","forgetting.tier_weight_episodic":"사례 메모(중기 경험)의 감쇠 강도입니다. 높을수록 중기 경험이 더 빠르게 약해집니다.","forgetting.tier_weight_semantic":"교훈 메모(장기 지식)의 감쇠 강도입니다. 보통 가장 낮게 설정하여 검증된 교훈은 오래 유지합니다.","forgetting.min_effective_score":"이 점수 이하인 메모리는 검색 결과에서 제외됩니다. 높이면 품질 좋은 메모리만 남고, 낮추면 더 많은 메모리를 활용합니다.","forgetting.tuning":"자동 보정 반영 조건을 정합니다.","forgetting.tuning.enabled":"메모리 사용 패턴을 분석하여 감쇠 속도와 유지 점수를 자동으로 최적화합니다. 꺼두면 수동으로 설정한 값만 사용합니다.","forgetting.tuning.mode":"추천만 볼지, 조금씩 자동 반영할지 정합니다.","forgetting.tuning.lookback_days":"자동 보정 시 최근 며칠간의 메모리 사용 데이터를 분석할지 정합니다. 길게 설정하면 더 안정적이지만 변화 반영이 느립니다.","forgetting.tuning.stability_window_days":"보정 결과가 안정적인지 판단하기 위해 며칠간 관찰할지 정합니다.","forgetting.tuning.min_access_events":"자동 보정 전에 필요한 조회 기록 수입니다.","forgetting.tuning.min_prompt_uses":"자동 보정 전에 필요한 실제 사용 수입니다.","forgetting.tuning.min_unique_memories":"자동 보정 전에 필요한 서로 다른 메모리 수입니다.","forgetting.tuning.ema_alpha":"새 보정 값을 기존 값에 반영하는 비율입니다. 높으면(1에 가까우면) 빠르게 변하고, 낮으면(0에 가까우면) 서서히 변합니다.","forgetting.tuning.max_decay_factor_delta":"한 번에 크게 바뀌지 않게 막는 범위입니다.","forgetting.tuning.max_min_effective_score_delta":"최소 유지 점수 변화 범위입니다.","forgetting.tuning.max_tier_weight_delta":"메모리 종류별 강도 변화 범위입니다.","forgetting.tuning.objective_topk":"좋은 메모리로 볼 상위 범위를 정합니다.","forgetting.tuning.auto_promote_enabled":"추천 모드에서 충분히 안정적이면 자동으로 실제 반영 모드로 전환합니다. 꺼두면 수동으로만 모드를 전환할 수 있습니다.","forgetting.tuning.auto_promote_min_shadow_days":"자동 반영으로 전환하기 전에 추천 모드로 최소 며칠간 관찰해야 하는지 설정합니다.","forgetting.tuning.auto_promote_min_shadow_runs":"자동 반영으로 전환하기 전에 추천 모드 실행을 최소 몇 회 관찰해야 하는지 설정합니다.","forgetting.tuning.auto_promote_required_stable_runs":"전환 전 연속으로 안정적 결과가 나와야 하는 횟수입니다. 높을수록 더 신중하게 전환합니다.","forgetting.tuning.auto_promote_required_improving_runs":"전환 전 연속으로 결과가 개선되어야 하는 횟수입니다.","forgetting.tuning.auto_promote_max_recommendation_drift":"추천 값의 흔들림이 이 범위 안에 있어야 자동 전환합니다. 작을수록 엄격한 기준입니다.","forgetting.tuning.auto_demote_enabled":"반영 모드에서 결과가 나빠지면 자동으로 추천 모드로 되돌립니다. 안전장치 역할을 합니다.","forgetting.tuning.auto_demote_unhealthy_runs":"연속으로 이 횟수만큼 나쁜 결과가 나오면 자동으로 추천 모드로 되돌립니다.",graph:"메모리 사이의 이어짐을 함께 봅니다.","graph.enabled":"메모리 간 인과 관계(분석→판단→거래→결과)를 연결 그래프로 추적합니다. 켜면 관련 메모리를 연쇄적으로 검색할 수 있습니다.","graph.max_expansion_hops":"한 메모리에서 몇 단계까지 함께 볼지 정합니다.","graph.max_expanded_nodes":"한 번에 보여줄 연결 수를 제한합니다.","graph.inferred_edge_min_confidence":"추정 연결을 믿는 기준값입니다.",compaction:"사이클이 끝난 뒤 배운 점을 짧게 정리합니다.","compaction.enabled":"투자 사이클이 끝난 후 거래·분석 기록을 AI가 요약하여 짧은 교훈으로 압축합니다. 꺼두면 원본 기록만 남습니다.","compaction.cycle_event_limit":"회고 압축 시 입력으로 사용할 최근 기록의 최대 개수입니다. 많을수록 더 풍부한 교훈을 생성합니다.","compaction.recent_lessons_limit":"회고 압축 시 참고할 기존 교훈의 최대 개수입니다. 이미 있는 교훈과 중복되지 않게 합니다.","compaction.max_reflections":"한 번의 압축에서 생성할 교훈의 최대 개수입니다.","compaction.thesis_chain_enabled":"닫힌 thesis 체인을 일반 cycle 로그보다 먼저 요약 대상으로 올립니다. thesis lifecycle이 선명한 경우 더 좋은 post-mortem lesson을 만듭니다.","compaction.thesis_chain_max_chains_per_cycle":"한 cycle에서 compactor가 동시에 검토할 닫힌 thesis 체인의 최대 개수입니다. 너무 많으면 교훈이 흐려질 수 있습니다.","compaction.thesis_chain_max_events_per_chain":"thesis 하나당 prompt에 남길 lifecycle 이벤트 수입니다. 너무 크면 장황해지고 너무 작으면 맥락이 잘립니다.","compaction.global_prompt":"회고 압축 시 AI에게 전달하는 안내 문구입니다. 어떤 관점에서 교훈을 정리할지 지시할 수 있습니다.",retrieval:"저장된 메모리를 언제 어떻게 다시 보여줄지 정합니다.","retrieval.context_limit":"에이전트 프롬프트에 한 번에 포함할 과거 메모리의 최대 개수입니다. 너무 많으면 핵심이 묻히고, 적으면 참고 정보가 부족합니다.","retrieval.vector_search":"비슷한 과거 메모리를 먼저 찾습니다.","retrieval.vector_search_enabled":"의미적으로 비슷한 메모리를 벡터 유사도로 검색합니다. 꺼두면 키워드 기반으로만 검색합니다.","retrieval.vector_search_limit":"벡터 검색 시 가져올 후보 메모리 수입니다. 많을수록 더 정밀하게 찾습니다.","retrieval.peer_lessons_enabled":"다른 에이전트(예: GPT가 학습한 교훈을 Claude도 참고)의 교훈도 함께 검색합니다. 에이전트 간 지식을 공유할 수 있습니다.","retrieval.reranking":"지금 더 도움이 되는 메모리를 위로 올립니다.","retrieval.reranking.type_bonus_reflection":"교훈(전략 반성) 유형 메모리의 검색 가산점입니다. 높을수록 과거 교훈이 상위에 노출됩니다.","retrieval.reranking.type_bonus_trade":"거래 기록 유형 메모리의 검색 가산점입니다. 높을수록 실제 거래 경험이 상위에 노출됩니다.","retrieval.reranking.type_bonus_manual":"사용자가 직접 작성한 메모의 검색 가산점입니다. 높이면 수동 지시사항이 우선됩니다.","retrieval.reranking.type_bonus_react_tools":"도구 실행 요약 메모리의 검색 가산점입니다.","retrieval.reranking.recency_bonus_3d":"최근 3일 이내 생성된 메모리의 가산점입니다. 높을수록 아주 최근 경험을 강하게 우선합니다.","retrieval.reranking.recency_bonus_14d":"최근 14일 이내 생성된 메모리의 가산점입니다.","retrieval.reranking.recency_bonus_45d":"최근 45일 이내 생성된 메모리의 가산점입니다.","retrieval.reranking.ticker_bonus_base":"현재 분석 중인 종목과 같은 종목의 메모리에 주는 기본 가산점입니다. 같은 종목 경험을 우선 참고합니다.","retrieval.reranking.ticker_bonus_step":"같은 종목 메모리가 여러 개일 때 추가로 주는 가산점입니다.","retrieval.reranking.ticker_bonus_max":"종목 일치 가산점의 최대 상한입니다. 하나의 종목이 검색을 독점하지 않도록 제한합니다.","retrieval.reranking.outcome_bonus_max":"좋은 성과를 낸 거래 메모리에 주는 가산점의 최대 상한입니다. 성공 경험을 더 잘 활용합니다.","retrieval.reranking.effective_score_bonus_scale":"최근에도 도움 된 메모리를 얼마나 더 올릴지 정합니다.","retrieval.reranking.effective_score_bonus_cap":"그 가산점 상한을 정합니다.",react_injection:"도구 실행 전에 관련 메모리를 붙입니다.","react_injection.enabled":"에이전트가 분석 도구를 실행하기 전에 관련 과거 메모리를 함께 전달합니다. 이전에 비슷한 분석을 했던 경험을 참고하여 더 나은 판단을 합니다.","react_injection.tools":"도구별 적용 대상을 정합니다.","react_injection.tools.technical_signals":"기술적 분석(이동평균, RSI 등) 도구 실행 시 과거 기술 분석 메모리를 함께 제공합니다.","react_injection.tools.screen_market":"시장 종목 스크리닝 도구 실행 시 이전 스크리닝 경험 메모리를 함께 제공합니다.","react_injection.tools.forecast_returns":"수익률 예측 도구 실행 시 이전 예측 결과와 실제 성과 메모리를 함께 제공합니다.","react_injection.tools.get_fundamentals":"기업 기본 정보 조회 도구 실행 시 과거 펀더멘털 분석 메모리를 함께 제공합니다.","react_injection.tools.optimize_portfolio":"포트폴리오 최적화 도구 실행 시 이전 리밸런싱 경험 메모리를 함께 제공합니다.",cleanup:"오래되고 덜 중요한 메모리를 정리합니다.","cleanup.enabled":"유지 점수가 낮고 오래된 메모리를 자동으로 삭제합니다. 꺼두면 메모리가 계속 누적되어 검색 정확도가 떨어질 수 있습니다.","cleanup.max_age_days":"이 일수보다 오래된 메모리를 정리 대상으로 봅니다. 길게 설정하면 오래된 경험도 참고할 수 있습니다.","cleanup.min_score":"유지 점수가 이 값 이하인 메모리만 삭제합니다. 높이면 더 적극적으로 정리하고, 낮추면 정말 안 쓰는 것만 삭제합니다."};'
        'function genericNodeDescription(node){if(!node){return "";}const label=displayLabel(node)||"이 항목";if(node.kind==="group"||node.kind==="branch"){return label+" 관련 설정입니다.";}if(node.type==="toggle"){return label+" 사용 여부입니다.";}if(node.type==="select"){return label+" 방식 선택입니다.";}if(node.type==="prompt"){return label+" 문구 편집입니다.";}return label+" 값 조정입니다."; }'
        'function describeNode(node){if(!node){return "";}const key=(node.path&&descMap[node.path])?node.path:((node.id&&descMap[node.id])?node.id:"");return key?descMap[key]:genericNodeDescription(node);}'
        'var hoverLabel=null;function showHoverLabel(node){if(!hoverLabel){hoverLabel=document.createElement("div");hoverLabel.style.cssText="position:absolute;pointer-events:none;z-index:20;font-size:11px;font-weight:700;white-space:nowrap;transition:opacity 0.15s ease;text-shadow:0 0 4px #fff,0 0 4px #fff,0 0 8px #fff;";graphHost.style.position="relative";graphHost.appendChild(hoverLabel);}if(!node||!fg){hoverLabel.style.opacity="0";return;}var coords=fg.graph2ScreenCoords(node.x||0,node.y||0,node.z||0);if(!coords){hoverLabel.style.opacity="0";return;}hoverLabel.textContent=displayLabel(node);hoverLabel.style.color=nodeColor(node);hoverLabel.style.left=(coords.x+12)+"px";hoverLabel.style.top=(coords.y-8)+"px";hoverLabel.style.opacity="1";}'
        'function showGraphFallback(message){if(!graphHost){return;}graphHost.innerHTML="<div class=\\"flex h-full items-center justify-center px-6 text-center text-sm text-ink-600\\"><div><p class=\\"font-semibold uppercase tracking-[0.2em] text-ink-500\\">그래프 사용 불가</p><p class=\\"mt-3 text-ink-700\\">"+escapeHtml(message)+"</p><p class=\\"mt-2 text-ink-500\\">우측 패널에서는 계속 편집할 수 있습니다.</p></div></div>";}'
        'function cleanupSummaryNode(){return document.getElementById("memory-cleanup-summary");}'
        'function cleanupButtonNode(){return document.getElementById("memory-cleanup-run");}'
        'function graphVisible(){return !!(graphHost&&graphHost.clientWidth>80&&graphHost.clientHeight>120&&graphHost.offsetParent!==null);}'
        'function resizeGraph(){if(!fg||!graphHost||!graphVisible()){return false;}try{fg.width(graphHost.clientWidth);fg.height(graphHost.clientHeight);return true;}catch(_e){return false;}}'
        'function frameGraph(animated){if(!fg||!graphVisible()){return false;}resizeGraph();try{fg.zoomToFit(animated?850:0,0);state.hasFramed=true;return true;}catch(_e){return false;}}'
        'function scheduleFrameGraph(){if(state.fitTimer){clearTimeout(state.fitTimer);}const delays=[80,260,700];delays.forEach(function(delay,index){state.fitTimer=setTimeout(function(){const ok=frameGraph(index>0);if(ok&&index===delays.length-1){state.fitTimer=0;}},delay);});}'
        'function fibonacciPoint(index,total,radius){const offset=2/total;const increment=Math.PI*(3-Math.sqrt(5));const y=((index*offset)-1)+(offset/2);const r=Math.sqrt(1-y*y);const phi=index*increment;return new THREE.Vector3(Math.cos(phi)*r*radius,y*radius,Math.sin(phi)*r*radius);}'
        'function buildNodeObject(node){if(typeof THREE==="undefined"){return null;}const base=new THREE.Color(nodeColor(node));const group=new THREE.Group();const radius=Math.max(node.id==="root"?7:3.2,Number(node.size||10)*0.22);const spokeCount=node.id==="root"?28:(node.kind==="group"?18:(node.kind==="branch"?14:10));const coreMat=new THREE.MeshBasicMaterial({color:base.clone().lerp(new THREE.Color("#ffffff"),0.08),transparent:true,opacity:0.92});const haloMat=new THREE.MeshBasicMaterial({color:base.clone().lerp(new THREE.Color("#ffffff"),0.55),transparent:true,opacity:0.18});const shellMat=new THREE.MeshBasicMaterial({color:base.clone().lerp(new THREE.Color("#ffffff"),0.28),transparent:true,opacity:0.34,wireframe:true});const core=new THREE.Mesh(new THREE.IcosahedronGeometry(radius,1),coreMat);const halo=new THREE.Mesh(new THREE.SphereGeometry(radius*1.35,14,14),haloMat);const shell=new THREE.Mesh(new THREE.IcosahedronGeometry(radius*1.8,1),shellMat);group.add(halo);group.add(shell);group.add(core);const spokeMats=[];const puffMats=[];for(let i=0;i<spokeCount;i+=1){const end=fibonacciPoint(i,spokeCount,radius*2.3);const lineMat=new THREE.LineBasicMaterial({color:base.clone().lerp(new THREE.Color("#ffffff"),0.36),transparent:true,opacity:0.42});const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0),end]),lineMat);const puffMat=new THREE.MeshBasicMaterial({color:"#ffffff",transparent:true,opacity:0.72});const puff=new THREE.Mesh(new THREE.SphereGeometry(Math.max(0.55,radius*0.18),8,8),puffMat);puff.position.copy(end);group.add(line);group.add(puff);spokeMats.push(lineMat);puffMats.push(puffMat);}group.userData={nodeId:node.id,phase:Math.random()*Math.PI*2,baseScale:1,coreMat:coreMat,haloMat:haloMat,shellMat:shellMat,spokeMats:spokeMats,puffMats:puffMats};visuals[node.id]=group;return group;}'
        'function startNodeAnimation(){if(animationFrame){cancelAnimationFrame(animationFrame);}function tick(){const now=performance.now()*0.001;Object.keys(visuals).forEach(function(id){const visual=visuals[id];if(!visual||!visual.userData){return;}const node=state.graph.nodes.find(function(item){return item.id===id;});if(!node){return;}const selected=id===state.selectedId;const hovered=id===state.hoveredId;const ud=visual.userData;if(typeof ud.hoverT==="undefined"){ud.hoverT=0;}const hoverTarget=hovered?1:0;ud.hoverT+=(hoverTarget-ud.hoverT)*0.12;const pulse=0.96+0.05*Math.sin(now*1.6+ud.phase);const hoverScale=1+ud.hoverT*0.35;const boost=selected?1.15:1;visual.scale.setScalar((ud.baseScale||1)*pulse*boost*hoverScale);const hoverSpin=hovered?0.008:0;visual.rotation.y+=(selected?0.004:0.0012)+hoverSpin;visual.rotation.x+=0.0008;const fade=0.22+0.1*Math.sin(now*1.8+ud.phase);if(ud.coreMat){ud.coreMat.opacity=selected?1:(0.82+0.08*Math.sin(now*2.1+ud.phase)+ud.hoverT*0.18);}if(ud.haloMat){ud.haloMat.opacity=(selected?0.28:fade)+ud.hoverT*0.32;}if(ud.shellMat){ud.shellMat.opacity=(selected?0.48:0.26)+ud.hoverT*0.2;}if(Array.isArray(ud.spokeMats)){ud.spokeMats.forEach(function(mat,idx){mat.opacity=(selected?0.68:(0.28+0.12*Math.sin(now*2+ud.phase+(idx*0.12))))+ud.hoverT*0.3;});}if(Array.isArray(ud.puffMats)){ud.puffMats.forEach(function(mat,idx){mat.opacity=(selected?0.95:(0.55+0.16*Math.sin(now*2.2+ud.phase+(idx*0.15))))+ud.hoverT*0.2;});}});animationFrame=requestAnimationFrame(tick);}tick();}'
        'function initGraph(){if(typeof ForceGraph3D!=="function"||typeof THREE==="undefined"){showGraphFallback("그래프 렌더러를 불러오지 못했습니다.");badge("err","그래프 로드 실패");return false;}try{graphHost.innerHTML="";Object.keys(visuals).forEach(function(key){delete visuals[key];});fg=ForceGraph3D()(graphHost).backgroundColor("#ffffff").showNavInfo(false).enableNodeDrag(false).nodeRelSize(4).nodeOpacity(1).linkOpacity(1).linkWidth(0).linkColor(function(){return "transparent";}).linkThreeObjectExtend(false).linkThreeObject(function(link){const src=(link&&link.source&&typeof link.source==="object")?link.source:{};const c=new THREE.Color(nodeColor(src));const mat=new THREE.MeshBasicMaterial({color:c,transparent:true,opacity:0.45});const geo=new THREE.CylinderGeometry(0.35,0.35,1,6,1,false);geo.translate(0,0.5,0);geo.rotateX(Math.PI/2);const mesh=new THREE.Mesh(geo,mat);mesh.userData.linkMat=mat;mesh.userData.srcColor=c;return mesh;}).linkPositionUpdate(function(obj,coords){if(!obj||!coords||!coords.start||!coords.end){return false;}var sx=coords.start.x||0,sy=coords.start.y||0,sz=coords.start.z||0;var ex=coords.end.x||0,ey=coords.end.y||0,ez=coords.end.z||0;var dx=ex-sx,dy=ey-sy,dz=ez-sz;var dist=Math.sqrt(dx*dx+dy*dy+dz*dz)||1;obj.position.set(sx,sy,sz);obj.scale.set(1,1,dist);obj.lookAt(ex,ey,ez);return true;}).linkDirectionalParticles(function(link){const target=(link&&link.target&&typeof link.target==="object")?link.target:{};return target.kind==="leaf"?2:(target.kind==="branch"?3:4);}).linkDirectionalParticleWidth(function(link){const target=(link&&link.target&&typeof link.target==="object")?link.target:{};return target.kind==="leaf"?1.4:(target.kind==="branch"?1.8:2.2);}).linkDirectionalParticleSpeed(function(link){const target=(link&&link.target&&typeof link.target==="object")?link.target:{};return target.kind==="leaf"?0.004:(target.kind==="branch"?0.005:0.006);}).linkDirectionalParticleColor(function(link){const source=(link&&link.source&&typeof link.source==="object")?link.source:{};const c=nodeColor(source);return c;}).nodeThreeObject(function(node){return buildNodeObject(node);}).nodeLabel(function(){return null;}).onNodeHover(function(node){state.hoveredId=node?node.id:null;graphHost.style.cursor=node?"pointer":"";showHoverLabel(node);}).onNodeClick(function(node){if(!node){return;}state.selectedId=node.id;renderPanel(node);renderGraph();});try{fg.d3Force("charge").strength(-142);fg.d3Force("link").distance(function(link){const target=(link&&link.target&&typeof link.target==="object")?link.target:{};return target.kind==="leaf"?38:(target.kind==="branch"?52:74);});}catch(_e){}try{const controls=fg.controls&&fg.controls();if(controls){controls.enablePan=true;controls.panSpeed=0.9;controls.rotateSpeed=1.0;controls.zoomSpeed=1.05;}}catch(_e){}try{const scene=fg.scene();if(scene&&!scene.userData.memoryLights){scene.add(new THREE.AmbientLight(0xffffff,1.7));const keyLight=new THREE.DirectionalLight(0xffffff,0.9);keyLight.position.set(120,90,140);scene.add(keyLight);const rimLight=new THREE.DirectionalLight(0xdbeafe,0.55);rimLight.position.set(-120,-40,80);scene.add(rimLight);scene.userData.memoryLights=true;}}catch(_e){}startNodeAnimation();return true;}catch(err){console.error(err);showGraphFallback("그래프 초기화에 실패했습니다.");badge("err","그래프 초기화 실패");return false;}}'
        'function renderGraph(){'
        'if(fg){Object.keys(visuals).forEach(function(key){delete visuals[key];});const nodes=state.graph.nodes.map(function(node){return Object.assign({},node);});const links=state.graph.links.map(function(link){return Object.assign({},link);});fg.graphData({nodes:nodes,links:links});setTimeout(function(){try{resizeGraph();fg.d3ReheatSimulation();if(!state.hasFramed){scheduleFrameGraph();}}catch(_e){}},0);}'
        'const node=getSelectedNode();'
        'if(node){renderPanel(node);}renderCleanupSummary();'
        '}'
        'function setByPath(obj,path,value){const parts=String(path||"").split(".").filter(Boolean);if(!parts.length){return;}let cursor=obj;for(let i=0;i<parts.length-1;i+=1){const key=parts[i];if(typeof cursor[key]!=="object"||cursor[key]===null){cursor[key]={};}cursor=cursor[key];}cursor[parts[parts.length-1]]=value;}'
        'function readInput(node){if(node.type==="toggle"){const input=formEl.querySelector("input[type=\\"checkbox\\"]");return !!(input&&input.checked);}if(node.type==="prompt"){const input=formEl.querySelector("textarea");return input?input.value:"";}if(node.type==="select"){const input=formEl.querySelector("select");return input?input.value:node.value;}const input=formEl.querySelector("input[type=\\"number\\"]");if(!input){return node.value;}if(node.type==="int"){return parseInt(input.value||"0",10)||0;}return parseFloat(input.value||"0")||0;}'
        'function cleanupCard(){return `<div class="mt-5"><div class="rounded-[20px] border border-slate-100 bg-gradient-to-b from-white to-slate-50/50 p-5"><div class="flex items-start justify-between gap-3"><div class="min-w-0"><div class="flex items-center gap-2"><svg class="h-4 w-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg><h4 class="text-sm font-semibold text-slate-800">메모리 정리</h4></div></div><button id="memory-cleanup-run" type="button" class="shrink-0 rounded-full bg-slate-800 px-4 py-2 text-[11px] font-semibold text-white shadow-sm transition-all duration-200 hover:bg-slate-700 hover:shadow-md active:scale-[0.97]">실행</button></div><div id="memory-cleanup-summary" class="mt-4 rounded-[14px] bg-slate-50/80 px-4 py-3 text-xs leading-relaxed text-slate-500">정리 기준을 불러오는 중...</div></div></div>`;}'
        'function fieldTip(_desc){return "";}'
        'function isAdvancedNode(node){const path=String((node&&node.path)||"");return path==="graph.max_expansion_hops"||path==="graph.max_expanded_nodes"||path==="graph.inferred_edge_min_confidence"||path==="retrieval.reranking.effective_score_bonus_scale"||path==="retrieval.reranking.effective_score_bonus_cap"||path.indexOf("forgetting.tuning.")===0;}'
        'function childCard(child,rootMode){const id=escapeHtml(String(child.id||""));const label=escapeHtml(displayLabel(child));const grandCount=getChildNodes(child).length;const color=escapeHtml(nodeColor(child));let meta="";if(child.type==="toggle"){meta=`<button type="button" data-toggle-node="${id}" class="shrink-0 rounded-full px-2.5 py-1 text-[10px] font-bold tracking-wide transition-all duration-200 ${child.value?"bg-emerald-50 text-emerald-600 ring-1 ring-emerald-200/60":"bg-slate-50 text-slate-400 ring-1 ring-slate-200/60"}">${child.value?"사용 중":"꺼짐"}</button>`;}else if(child.type==="prompt"){meta=`<span class="shrink-0 rounded-full bg-slate-50 px-2.5 py-1 text-[10px] font-bold tracking-wide text-slate-500 ring-1 ring-slate-100">편집</span>`;}else if(rootMode){meta=`<span class="shrink-0 rounded-full bg-slate-50 px-2.5 py-1 text-[10px] font-bold tracking-wide text-slate-400 ring-1 ring-slate-100">${grandCount}</span>`;}else if(child.editable&&child.value!=null){const valueText=child.type==="select"?prettyOption(child.value):String(child.value);const preview=valueText.length>18?valueText.slice(0,18)+"...":valueText;meta=`<span class="shrink-0 rounded-full bg-slate-50 px-2.5 py-1 text-[10px] font-bold tracking-wide text-slate-500 ring-1 ring-slate-100">${escapeHtml(preview)}</span>`;}const advanced=isAdvancedNode(child)?`<span class="ml-2 shrink-0 rounded-full bg-amber-50 px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.18em] text-amber-600 ring-1 ring-amber-200/70">주의</span>`:"";if(rootMode){const rightSlot=child.type==="toggle"?meta:`<span class="shrink-0 text-[10px] tabular-nums text-slate-400">${grandCount}</span>`;return `<button type="button" data-node-select="${id}" class="group rounded-[14px] border border-slate-100 bg-white px-3 py-2.5 text-left transition-all duration-200 hover:border-slate-200 hover:shadow-[0_4px_20px_rgba(15,23,42,0.06)]"><div class="flex items-start justify-between gap-2"><div class="min-w-0 flex-1"><div class="flex items-center gap-2 min-w-0"><span class="mt-1 h-1.5 w-1.5 shrink-0 rounded-full transition-transform duration-200 group-hover:scale-125" style="background:${color}"></span><span class="block min-w-0 text-[12px] font-semibold leading-snug text-slate-800 break-words">${label}</span>${advanced}</div></div>${rightSlot}</div></button>`;}return `<button type="button" data-node-select="${id}" class="group flex w-full items-start justify-between gap-3 rounded-[14px] border border-slate-100 bg-white px-4 py-3 text-left transition-all duration-200 hover:border-slate-200 hover:shadow-[0_2px_12px_rgba(15,23,42,0.05)]"><div class="min-w-0 flex-1"><div class="flex items-center gap-2.5 min-w-0"><span class="mt-1 h-1.5 w-1.5 shrink-0 rounded-full" style="background:${color}"></span><span class="block min-w-0 text-[13px] font-medium leading-snug text-slate-700 break-words group-hover:text-slate-900">${label}</span>${advanced}</div></div>${meta}</button>`;}'
        'function readOnlyPanel(node){const lines=[];if(typeof node.count==="number"&&node.count>0){lines.push({icon:`<svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4"/></svg>`,text:`메모리 ${String(node.count)}개`});}if(node.scope==="global"){lines.push({icon:`<svg class="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>`,text:"전역 설정"});}let body="";if(lines.length){body+=`<div class="flex flex-wrap gap-2">`+lines.map(function(item){return `<div class="inline-flex items-center gap-1.5 rounded-full bg-slate-50 px-3 py-1.5 text-[11px] font-medium text-slate-500 ring-1 ring-slate-100">${item.icon}<span>${item.text}</span></div>`;}).join("")+`</div>`;}if(node.id==="graph"){body+=(body?`<div class="mt-5">`:`<div>`)+graphStatusCard()+`</div>`;}if(node.id==="forgetting"){body+=(body?`<div class="mt-5">`:`<div>`)+forgettingStatusCard()+`</div>`;}const children=getChildNodes(node);if(children.length){const rootMode=node.id==="root";const layoutClass=rootMode?"grid gap-2 sm:grid-cols-2":"space-y-1.5";body+=`<div class="${body?"mt-5":""}"><div class="${layoutClass}">`+children.map(function(child){return childCard(child,rootMode);}).join("")+`</div></div>`;}if(node.id==="cleanup"){body+=cleanupCard();}if(!body){body=`<div class="py-2"></div>`;}return body;}'
        'function getParentNode(node){if(!node||!node.id){return null;}const parentLink=(state.graph.links||[]).find(function(link){return targetId(link)===node.id;});if(!parentLink){return null;}const pid=sourceId(parentLink);return state.graph.nodes.find(function(item){return item.id===pid;})||null;}'
        'function renderBreadcrumb(node){const bc=document.getElementById("memory-panel-breadcrumb");if(!bc){return;}const parent=getParentNode(node);if(!parent||node.id==="root"){bc.innerHTML="";return;}bc.innerHTML=`<button type="button" data-node-select="${escapeHtml(parent.id)}" class="mb-3 inline-flex items-center gap-1 rounded-full bg-slate-50 px-2.5 py-1 text-[11px] font-medium text-slate-400 ring-1 ring-slate-100 transition-colors hover:bg-slate-100 hover:text-slate-600"><svg class="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5"><path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/></svg>${escapeHtml(displayLabel(parent))}</button>`;}'
        'function updatePanelDot(node){const dot=document.getElementById("memory-panel-dot");if(!dot){return;}dot.style.background=nodeColor(node);dot.style.boxShadow="0 0 8px "+nodeColor(node)+"50";}'
        'function animatePanel(){formEl.style.animation="none";formEl.offsetHeight;formEl.style.animation="";}'
        'function renderPanel(node){const nodeDesc=escapeHtml(describeNode(node)||"");titleEl.textContent=panelTitle(node);descEl.textContent="";descEl.classList.add("hidden");renderBreadcrumb(node);updatePanelDot(node);animatePanel();'
        'if(!node.editable){formEl.innerHTML=readOnlyPanel(node);renderCleanupSummary();const rootCleanupBtn=cleanupButtonNode();if(rootCleanupBtn&&!rootCleanupBtn.dataset.bound){rootCleanupBtn.dataset.bound="1";rootCleanupBtn.addEventListener("click",runCleanup);}return;}'
        'let body="<div class=\\"space-y-4\\">";'
        'if(nodeDesc){body+=`<div class="rounded-[16px] border border-slate-100 bg-slate-50/80 px-4 py-3 text-[12px] leading-relaxed text-slate-600">${nodeDesc}</div>`;}'
        'if(isAdvancedNode(node)){body+=`<div class="rounded-[16px] border border-amber-200/70 bg-amber-50/80 px-4 py-3 text-[12px] leading-relaxed text-amber-800">고급 설정입니다. 메모리 연결률, 자동 보정 상태, 검색 결과를 보면서 천천히 조정하는 편이 안전합니다.</div>`;}'
        'if(node.type==="toggle"){body+=`<label class="flex cursor-pointer items-center justify-between gap-4 rounded-[16px] border border-slate-100 bg-white p-4 transition-all duration-200 hover:border-slate-200"><div><p class="text-[13px] font-semibold text-slate-800">사용 여부</p></div><span class="relative inline-flex h-7 w-12 shrink-0 items-center"><input type="checkbox" ${(node.value?"checked":"")} class="peer sr-only" /><span class="absolute inset-0 rounded-full bg-slate-200 transition-colors duration-200 peer-checked:bg-emerald-400"></span><span class="absolute left-[3px] top-[3px] h-[22px] w-[22px] rounded-full bg-white shadow-sm transition-transform duration-200 peer-checked:translate-x-5"></span></span></label>`;}'
        'else if(node.type==="prompt"){body+=`<div class="rounded-[16px] border border-slate-100 bg-white p-4"><p class="text-[12px] font-semibold text-slate-700">안내 문구</p><textarea rows="12" class="mt-2 h-64 w-full resize-none rounded-[14px] border-0 bg-slate-50 px-4 py-3 font-mono text-xs leading-relaxed text-slate-700 outline-none ring-1 ring-slate-100 transition-shadow duration-200 focus:ring-2 focus:ring-slate-300 placeholder:text-slate-300" placeholder="안내 문구 입력">${escapeHtml(String(node.value||""))}</textarea></div>`;}'
        'else if(node.type==="select"){const options=(Array.isArray(node.options)?node.options:[]).map(function(option){const selected=String(option)===String(node.value)?"selected":"";return `<option value="${escapeHtml(String(option))}" ${selected}>${escapeHtml(prettyOption(option))}</option>`;}).join("");body+=`<div class="rounded-[16px] border border-slate-100 bg-white p-4"><p class="text-[11px] font-medium text-slate-400">옵션</p><select class="mt-2 w-full rounded-[12px] border-0 bg-slate-50 px-4 py-3 text-base font-semibold text-slate-800 outline-none ring-1 ring-slate-100 transition-shadow duration-200 focus:ring-2 focus:ring-slate-300">${options}</select></div>`;}'
        'else {body+=`<div class="rounded-[16px] border border-slate-100 bg-white p-4"><p class="text-[11px] font-medium text-slate-400">값</p><input type="number" class="mt-2 w-full rounded-[12px] border-0 bg-slate-50 px-4 py-3 text-lg font-bold tabular-nums text-slate-800 outline-none ring-1 ring-slate-100 transition-shadow duration-200 focus:ring-2 focus:ring-slate-300" value="${String(node.value??"")}" ${(node.min!=null?`min="${node.min}"`:"")} ${(node.max!=null?`max="${node.max}"`:"")} ${(node.step!=null?`step="${node.step}"`:"")}/></div>`;}'
        'body+="<button id=\\"memory-save-btn\\" type=\\"button\\" class=\\"w-full rounded-[14px] bg-slate-800 py-3 text-[13px] font-semibold text-white shadow-sm transition-all duration-200 hover:bg-slate-700 hover:shadow-md active:scale-[0.98]\\">변경사항 저장</button>";'
        'body+="</div>";'
        'formEl.innerHTML=body;'
        'const btn=document.getElementById("memory-save-btn");'
        'if(btn){btn.addEventListener("click",function(){saveNode(node);});}'
        '}'
        'function renderCleanupSummary(){const cleanupSummaryEl=cleanupSummaryNode();const cleanupBtn=cleanupButtonNode();if(!cleanupSummaryEl){return;}const cleanup=((((state.graph||{}).meta||{}).policy||{}).cleanup)||{};const forgetting=((((state.graph||{}).meta||{}).policy||{}).forgetting)||{};const enabled=!!cleanup.enabled;const age=cleanup.max_age_days!=null?cleanup.max_age_days:180;const score=cleanup.min_score!=null?cleanup.min_score:0.3;const runtime=runtimeMeta();const memory=runtime.memory||{};const access=runtime.access||{};const tuning=tuningState();const mode=String(tuning.effective_mode||tuning.configured_mode||"shadow");const last=state.lastCleanup;if(last&&typeof last==="object"){cleanupSummaryEl.innerHTML=`<div class="grid grid-cols-3 gap-2 text-center"><div><p class="text-lg font-bold text-slate-800">${String(last.candidate_count||0)}</p><p class="text-[10px] text-slate-400">대상</p></div><div><p class="text-lg font-bold text-slate-800">${String(last.deleted_bigquery||0)}</p><p class="text-[10px] text-slate-400">기록 저장소 삭제</p></div><div><p class="text-lg font-bold text-slate-800">${String(last.deleted_firestore||0)}</p><p class="text-[10px] text-slate-400">검색 저장소 삭제</p></div></div>`+(last.firestore_error?`<p class="mt-2 text-[11px] text-amber-600">${escapeHtml(String(last.firestore_error))}</p>`:"")+`<p class="mt-2 text-[10px] text-slate-400">${String(age)}일 / 유지 점수 ${String(score)} 이하</p><p class="mt-1 text-[10px] text-slate-400">오래된 메모리 관리 ${forgetting.enabled?"사용 중":"꺼짐"} · 자동 보정 ${escapeHtml(prettyOption(mode))}</p>`;}else if(enabled){cleanupSummaryEl.innerHTML=`<div class="flex items-center gap-2"><span class="h-1.5 w-1.5 rounded-full bg-emerald-400"></span><span>준비됨</span></div><p class="mt-1">${String(age)}일 초과 · 유지 점수 ${String(score)} 이하 대상</p><p class="mt-2 text-[11px] text-slate-500">메모리 연결률 ${formatRatio(memory.with_graph_node_id||0,memory.total_memory_events||0)} · 조회 기록 ${formatNumber(access.access_event_count||0)} · 자동 보정 ${escapeHtml(prettyOption(mode))}</p>`;}else{cleanupSummaryEl.innerHTML=`<div class="flex items-center gap-2"><span class="h-1.5 w-1.5 rounded-full bg-slate-300"></span><span>비활성</span></div><p class="mt-1 text-[11px] text-slate-500">자동 정리가 꺼져 있습니다. 오래된 메모리 관리 ${forgetting.enabled?"사용 중":"꺼짐"} · 유지 점수 계산됨 ${formatRatio(memory.with_effective_score||0,memory.total_memory_events||0)}</p><button type="button" data-node-select="cleanup.enabled" class="mt-2 inline-flex items-center gap-1 rounded-full bg-white px-3 py-1.5 text-[11px] font-semibold text-slate-600 ring-1 ring-slate-200 transition-colors hover:bg-slate-50">자동 정리 켜기</button>`;}if(cleanupBtn){cleanupBtn.disabled=!enabled;cleanupBtn.classList.toggle("opacity-40",!enabled);cleanupBtn.classList.toggle("cursor-not-allowed",!enabled);cleanupBtn.textContent=enabled?"실행":"꺼짐";}}'
        'async function saveNodeValue(node,nextValue){'
        'const payload={policy:JSON.parse(JSON.stringify(state.graph.meta.policy||{}))};'
        'if(node.id==="compaction.global_prompt"){payload.compaction_prompt=String(nextValue||"");}'
        'else{setByPath(payload.policy,node.path,nextValue);}'
        'try{'
        'setPanelBusy(true,node.type==="toggle"?"적용 중...":"저장 중...");'
        'const res=await fetch("/api/memory/config?tenant_id="+encodeURIComponent(state.graph.meta.tenant_id),{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});'
        'const data=await res.json();'
        'if(!res.ok){throw new Error(String(data&&data.error||"save failed"));}'
        'state.graph=data;'
        'state.selectedId=node.id;'
        'state.lastCleanup=null;'
        'badge("ok","저장됨");'
        'renderGraph();'
        'return true;}catch(err){badge("err",err&&err.message?err.message:"저장 실패");return false;}finally{setPanelBusy(false);}'
        '}'
        'async function quickToggleNode(nodeId){const node=state.graph.nodes.find(function(item){return item.id===nodeId;});if(!node||!node.editable||node.type!=="toggle"){return false;}return saveNodeValue(node,!node.value);}'
        'async function saveNode(node){'
        'const nextValue=readInput(node);'
        'return saveNodeValue(node,nextValue);'
        '}'
        'async function runCleanup(){const cleanupBtn=cleanupButtonNode();if(!cleanupBtn||cleanupBtn.disabled){badge("err","정리 기능이 꺼져 있습니다");return;}const original=cleanupBtn.textContent;cleanupBtn.disabled=true;cleanupBtn.textContent="정리 중...";try{setPanelBusy(true,"정리 중...");const res=await fetch("/api/memory/cleanup?tenant_id="+encodeURIComponent(state.graph.meta.tenant_id),{method:"POST"});const data=await res.json();if(!res.ok){throw new Error(String(data&&data.error||"cleanup failed"));}state.lastCleanup=data;badge("ok","정리 완료");const graphRes=await fetch("/api/memory/graph?tenant_id="+encodeURIComponent(state.graph.meta.tenant_id));const nextGraph=await graphRes.json();if(graphRes.ok){state.graph=nextGraph;renderGraph();}else{renderCleanupSummary();}}catch(err){badge("err",err&&err.message?err.message:"정리 실패");}finally{setPanelBusy(false);const currentBtn=cleanupButtonNode();if(currentBtn){currentBtn.disabled=false;currentBtn.textContent=original;}renderCleanupSummary();}}'
        'function ensureGraphReady(){if(!graphVisible()){return false;}if(!fg){if(!initGraph()){return false;}debugApi.graph=fg;}resizeGraph();renderGraph();scheduleFrameGraph();return true;}'
        'const debugApi={graph:null,getState:function(){return state;},selectNodeById:function(id){const node=state.graph.nodes.find(function(item){return item.id===id;});if(!node){return false;}state.selectedId=node.id;renderPanel(node);renderGraph();return true;},saveSelected:function(){const node=getSelectedNode();if(!node||!node.editable){return Promise.resolve(false);}return saveNode(node).then(function(){return true;});},runCleanup:runCleanup,fitView:function(){return frameGraph(true);}};'
        'window.__memoryGraphDebug=debugApi;'
        'renderPanel(getSelectedNode());renderCleanupSummary();'
        'setTimeout(ensureGraphReady,0);'
        'document.addEventListener("settings-tab-activated",function(ev){const tab=((ev&&ev.detail&&ev.detail.tab)||"");if(tab==="memory"){setTimeout(ensureGraphReady,30);}});'
        'window.addEventListener("resize",function(){if(!fg){return;}if(resizeGraph()){scheduleFrameGraph();}});'
        '})();'
        '</script>'
    )


def register_memory_routes(
    app: FastAPI,
    *,
    repo: BigQueryRepository,
    settings: Settings,
    settings_enabled: bool,
    resolve_admin_context: Callable[..., Any],
    cached_fetch: Callable[..., Any] | None = None,
    invalidate_tenant_cache: Callable[..., Any] | None = None,
) -> None:
    @app.get("/assets/vendor/three.min.js")
    def memory_vendor_three() -> FileResponse:
        return FileResponse(_THREE_JS_PATH, media_type="application/javascript")

    @app.get("/assets/vendor/3d-force-graph.min.js")
    def memory_vendor_force_graph() -> FileResponse:
        return FileResponse(_FORCE_GRAPH_JS_PATH, media_type="application/javascript")

    @app.get("/admin/memory", response_class=HTMLResponse)
    def admin_memory_page(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> RedirectResponse:
        if not settings_enabled:
            return HTMLResponse("settings disabled", status_code=403)
        return RedirectResponse(url=f"/settings?tenant_id={tenant_id}&tab=memory", status_code=302)

    @app.get("/api/memory/graph")
    def api_memory_graph(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, _user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/graph?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        return json_response(_graph_payload(repo, settings, tenant_id=tenant, cached_fetch=cached_fetch), max_age=0)

    @app.get("/api/memory/config")
    def api_memory_config(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, _user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/config?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        graph = _graph_payload(repo, settings, tenant_id=tenant, cached_fetch=cached_fetch)
        meta = graph.get("meta") or {}
        return json_response(
            {
                "tenant_id": meta.get("tenant_id"),
                "policy": meta.get("policy") or {},
                "compaction_prompt": meta.get("effective_compaction_prompt") or "",
                "tenant_compaction_prompt": meta.get("tenant_compaction_prompt") or "",
                "global_compaction_prompt": meta.get("global_compaction_prompt") or "",
                "effective_compaction_prompt": meta.get("effective_compaction_prompt") or "",
                "prompt_source": meta.get("prompt_source") or "global",
                "compaction_prompt_editable": bool(meta.get("compaction_prompt_editable", True)),
                "runtime": meta.get("runtime") or {},
            },
            max_age=0,
        )

    @app.get("/api/memory/stats")
    def api_memory_stats(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, _user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/stats?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        stats_key = f"memory_stats:{tenant}:{settings.trading_mode}"
        if callable(cached_fetch):
            stats = cached_fetch(stats_key, _stats_payload, repo, tenant_id=tenant, trading_mode=settings.trading_mode)
        else:
            stats = _stats_payload(repo, tenant_id=tenant, trading_mode=settings.trading_mode)
        return json_response(stats, max_age=0)

    @app.post("/api/memory/cleanup")
    def api_memory_cleanup(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/cleanup?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)
        try:
            result = run_memory_cleanup(
                repo,
                settings,
                tenant_id=tenant,
                require_enabled=True,
            )
            if not bool(result.get("enabled")):
                return JSONResponse({"error": "cleanup disabled", **result}, status_code=400)
            try:
                repo.append_runtime_audit_log(
                    action="memory_cleanup_run",
                    status="ok",
                    user_email=user_email or "ui-admin",
                    tenant_id=tenant,
                    detail={
                        "candidate_count": result.get("candidate_count"),
                        "deleted_bigquery": result.get("deleted_bigquery"),
                        "deleted_firestore": result.get("deleted_firestore"),
                        "firestore_error": result.get("firestore_error") or "",
                    },
                )
            except Exception:
                pass
            if callable(invalidate_tenant_cache):
                invalidate_tenant_cache(tenant, "memory")
            return json_response(result, max_age=0)
        except Exception as exc:
            try:
                repo.append_runtime_audit_log(
                    action="memory_cleanup_run",
                    status="error",
                    user_email=user_email or "ui-admin",
                    tenant_id=tenant,
                    detail={"error": str(exc)},
                )
            except Exception:
                pass
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/api/memory/config")
    async def api_memory_config_save(
        request: Request,
        tenant_id: str = Query(default="local", description="tenant id"),
    ) -> JSONResponse:
        if not settings_enabled:
            return JSONResponse({"error": "settings disabled"}, status_code=403)
        _user, user_email, tenant, _allowed_tenants, redirect = resolve_admin_context(
            request,
            requested_tenant=tenant_id,
            next_path=f"/api/memory/config?tenant_id={tenant_id}",
        )
        if redirect:
            return JSONResponse({"error": "auth required"}, status_code=401)

        try:
            payload = await request.json()
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            return JSONResponse({"error": "invalid json body"}, status_code=400)

        defaults = _memory_defaults(settings)
        policy = normalize_memory_policy(payload.get("policy"), defaults=defaults)
        try:
            repo.set_config(
                tenant,
                MEMORY_POLICY_CONFIG_KEY,
                serialize_memory_policy(policy),
                updated_by=user_email or "ui-admin",
            )
            if "compaction_prompt" in payload or "global_compaction_prompt" in payload:
                prompt_value = payload.get("compaction_prompt", payload.get("global_compaction_prompt"))
                compaction_prompt = str(prompt_value or "").strip()
                if not compaction_prompt:
                    return JSONResponse({"error": f"{GLOBAL_MEMORY_PROMPT_CONFIG_KEY} cannot be empty"}, status_code=400)
                repo.set_config(
                    tenant,
                    GLOBAL_MEMORY_PROMPT_CONFIG_KEY,
                    compaction_prompt,
                    updated_by=user_email or "ui-admin",
                )
            try:
                repo.append_runtime_audit_log(
                    action="memory_policy_save",
                    status="ok",
                    user_email=user_email or "ui-admin",
                    tenant_id=tenant,
                    detail={"config_key": MEMORY_POLICY_CONFIG_KEY},
                )
            except Exception:
                pass
        except Exception as exc:
            try:
                repo.append_runtime_audit_log(
                    action="memory_policy_save",
                    status="error",
                    user_email=user_email or "ui-admin",
                    tenant_id=tenant,
                    detail={"error": str(exc)},
                )
            except Exception:
                pass
            return JSONResponse({"error": str(exc)}, status_code=500)

        if callable(invalidate_tenant_cache):
            invalidate_tenant_cache(tenant, "runtime", "memory")
        return json_response(_graph_payload(repo, settings, tenant_id=tenant, cached_fetch=cached_fetch), max_age=0)
