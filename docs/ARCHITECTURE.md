# LLM Invest — Developer Architecture Guide

> BigQuery + ADK + KIS Open Trading API 기반 멀티 LLM 자동투자 플랫폼
> 3 에이전트(GPT-5.2, Gemini 3 Flash, Claude Sonnet 4.6)가 US + KOSPI/KOSDAQ 시장에서 경쟁

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [Core Data Models](#2-core-data-models)
3. [Data Layer (BigQuery)](#3-data-layer-bigquery)
4. [Agent System](#4-agent-system)
5. [Long-Term Memory System](#5-long-term-memory-system)
6. [Tools System](#6-tools-system)
7. [Context Builder](#7-context-builder)
8. [Execution Pipeline](#8-execution-pipeline)
9. [Risk Engine](#9-risk-engine)
10. [Reconciliation & Recovery](#10-reconciliation--recovery)
11. [Open Trading Integration (KIS)](#11-open-trading-integration-kis)
12. [UI Layer](#12-ui-layer)
13. [Market Hours & Scheduling](#13-market-hours--scheduling)
14. [Configuration & Runtime Overrides](#14-configuration--runtime-overrides)
15. [CLI Interface](#15-cli-interface)
16. [Deployment](#16-deployment)
17. [Test Structure](#17-test-structure)
18. [Design Patterns](#18-design-patterns)
19. [Data Flow Walkthrough](#19-data-flow-walkthrough)
20. [Gotchas & Important Notes](#20-gotchas--important-notes)
21. [Quick Reference](#21-quick-reference)

---

## 1. Directory Structure

```
arena/                           Core business logic
├── agents/                       ADK ReAct agents (modular decomposition)
│   ├── adk_agents.py            Main agent class + builder (838L)
│   ├── adk_agent_flow.py        Draft/execution phase orchestration (111L)
│   ├── adk_context_tools.py     Per-cycle context tools for agents (660L)
│   ├── adk_decision_flow.py     Decision prompting + board comm (164L)
│   ├── adk_models.py            ADK model wrapper/routing (179L)
│   ├── adk_order_support.py     Order placement utilities (410L)
│   ├── adk_prompting.py         Prompt building + JSON parsing (199L)
│   ├── adk_runner_bootstrap.py  Runner initialization (304L)
│   ├── adk_runner_runtime.py    Runtime execution logic (266L)
│   ├── adk_runner_state.py      Mutable execution state tracking (392L)
│   ├── adk_tool_compaction.py   Tool result compaction (426L)
│   ├── adk_tool_config.py       Tool configuration/selection (130L)
│   ├── base.py                  TradingAgent protocol (24L)
│   ├── memory_compaction_agent.py  Post-cycle lesson synthesis (710L)
│   ├── research_agent.py        Gemini + Google Search Grounding (278L)
│   └── support_model.py         Helper model builder (105L)
├── memory/                       Multi-tier long-term memory system
│   ├── policy.py                Single source for all memory controls — 10 groups (2,239L)
│   ├── store.py                 Write/retrieve events with tier + tagging + graph (774L)
│   ├── vector.py                Vertex AI embeddings + Firestore search (288L)
│   ├── thesis.py                Investment thesis lifecycle tracking (170L)
│   ├── graph.py                 Causal graph node/edge builders (478L)
│   ├── tags.py                  Context tag extraction — regime/strategy/sector/ticker (317L)
│   ├── forgetting.py            Adaptive decay math + batch recompute (245L)
│   ├── tuning.py                Forgetting parameter auto-tuner — shadow/bounded_ema (821L)
│   ├── cleanup.py               Prune stale/low-signal memories (289L)
│   └── query_builders.py        Tool result → semantic query conversion (139L)
├── ui/                           Admin dashboard (FastAPI, modular routes)
│   ├── app.py                   Main FastAPI router (564L)
│   ├── routes/                   Route modules (16 files)
│   │   ├── auth.py              Google OAuth (269L)
│   │   ├── overview.py          Dashboard overview (168L)
│   │   ├── board.py             Board viewer (621L)
│   │   ├── nav.py               NAV charts (374L)
│   │   ├── trades.py            Trade history (135L)
│   │   ├── sleeves.py           Sleeve management (859L)
│   │   ├── ops.py               Operations page (224L)
│   │   ├── settings_page.py     Settings page render (455L)
│   │   ├── settings_admin.py    Settings CRUD API (970L)
│   │   ├── settings_render.py   Render dispatcher (21L)
│   │   ├── settings_render_agents.py    Agent config panel (340L)
│   │   ├── settings_render_capital.py   Capital management (696L)
│   │   ├── settings_render_credentials.py  KIS/API credentials (643L)
│   │   ├── settings_render_scripts.py   Script management (205L)
│   │   ├── capital_data.py      Capital data API (294L)
│   │   └── viewer.py            Data viewer (47L)
│   ├── templates/               Jinja2 templates (11 files)
│   ├── memory.py                3D memory graph builder + routes (680L)
│   ├── viewer_data.py           Viewer data assembly (709L)
│   ├── viewer_analytics.py      Analytics computations (96L)
│   ├── layout.py                Base layout helpers (83L)
│   ├── http.py                  JSON/HTML response helpers (22L)
│   ├── access.py                Access control (60L)
│   ├── provisioning.py          Tenant auto-provisioning (168L)
│   ├── app_support.py           App startup support (156L)
│   ├── run_status.py            Run status tracking (144L)
│   ├── runtime.py               UI runtime context (400L)
│   ├── templating.py            Template engine setup (28L)
│   └── server.py                Startup wrapper (5L)
├── tools/                        Agent tool registry (18 core + MCP)
│   ├── default_registry.py      Build registry with all tools (391L)
│   ├── quant_tools.py           Screen, optimize, forecast, technical (1,416L)
│   ├── sentiment_tools.py       Reddit, SEC EDGAR, earnings, VIX, news (634L)
│   ├── macro_tools.py           FRED (US), ECOS (Korea) (244L)
│   ├── allocation.py            Portfolio optimization — Sharpe, HRP, forecast (383L)
│   ├── screening.py             Momentum + discovery ranking (457L)
│   ├── sector_map.py            US 101 + KOSPI 579 sector mapping (688L)
│   └── registry.py              ToolRegistry with two-phase selection (90L)
├── data/                         BigQuery repository layer (modular stores)
│   ├── bq.py                   BigQueryRepository facade (141L)
│   ├── protocols.py             Store protocols/interfaces (145L)
│   ├── schema.py                Table DDLs + auto-migration (701L)
│   └── bigquery/                Store implementations
│       ├── session.py           BigQuerySession connection management (245L)
│       ├── memory_bq_store.py   Memory events, board posts, graph, briefings (1,176L)
│       ├── market_store.py      Price/feature queries (1,466L)
│       ├── sleeve_store.py      Virtual account operations + NAV (2,483L)
│       ├── execution_store.py   Order intent/execution repository (357L)
│       ├── ledger_store.py      Append-only event ledger (381L)
│       ├── runtime_store.py     Config/credential storage (604L)
│       └── backtest_store.py    Backtest persistence (161L)
├── open_trading/                 Korea Investment API client
│   ├── client.py                REST wrapper — OAuth, account, market data (1,913L)
│   ├── sync.py                  Market data, account, dividend sync (2,771L)
│   ├── exchange_codes.py        Exchange code mapping (97L)
│   └── token_cache.py           Firestore-backed OAuth token (71L)
├── broker/                       Order execution abstraction
│   ├── base.py                  BrokerClient protocol (13L)
│   ├── open_trading.py          Live KIS trading — US + KOSPI (507L)
│   └── paper.py                 Paper + HTTP broker (97L)
├── execution/                    Centralized order gateway
│   └── gateway.py               Risk check → broker → memory recording (320L)
├── providers/                    LLM provider registry
│   ├── registry.py              4 providers — GPT/Gemini/Claude/DeepSeek (209L)
│   └── credentials.py           Secret Manager credential parsing (130L)
├── security/                     Secrets management
│   └── credential_store.py      Secret Manager + BQ (377L)
├── cli_commands/                 Modular CLI command handlers
│   ├── run.py                   Command dispatch routing (45L)
│   ├── run_agent.py             Agent cycle execution (510L)
│   ├── run_pipeline.py          Full sync→forecast→ranker→agent pipeline
│   ├── run_shared.py            Shared sync/forecast/ranker operations
│   ├── run_reconcile.py         Reconciliation operations (227L)
│   ├── serve.py                 UI and MCP server startup (265L)
│   ├── sync.py                  Market/account data sync (343L)
│   └── admin.py                 Admin operations — tenant, memory (336L)
├── strategy/                     Strategy reference catalog
│   ├── catalog.py               Strategy cards for agents
│   └── mcp_server.py            MCP server for strategy tool
├── backtest/                     Walk-forward testing
│   └── walk_forward.py          Stabilization + periodic rebalancing (392L)
├── board/                        Inter-agent communication
│   └── store.py                 Publish/retrieve shared board posts (20L)
├── universe/                     Ticker universe presets
│   └── nasdaq100.py             NASDAQ-100 constants
├── reporting/                    Human-facing summaries
│   └── daily_report.py          EOD report builder
├── forecasting/                  ML forecast pipeline
│   └── stacked.py               7-model ensemble stacking (679L)
├── config.py                     Settings + runtime overrides (986L)
├── context.py                    Per-agent context builder (1,965L)
├── orchestrator.py               Multi-agent cycle orchestration (404L)
├── risk.py                       Risk engine policy checks (94L)
├── reconciliation.py             State reconciliation + recovery (1,374L)
├── market_hours.py               Market windows + holidays (318L)
├── market_sources.py             Market source resolution (52L)
├── runtime_universe.py           Runtime universe resolution (61L)
├── cli.py                        CLI entry point (309L)
├── cli_runtime.py                CLI runtime bootstrap (480L)
├── cloud_run_jobs.py             Cloud Run job dispatch (47L)
├── tenant_leases.py              Firestore execution lease (131L)
├── models.py                     Core data classes (152L)
├── logging_utils.py              JSON logging for Cloud Run
└── __main__.py

scripts/                          Operational scripts
├── deploy_cloud_run_job.sh       Deploy trading pipeline
├── deploy_cloud_run_ui.sh        Deploy UI
├── ship.sh                       Build+push+deploy one-command
├── dev-ui.sh                     Local UI dev server
├── cleanup_memory.py             Memory pruning batch
├── daily_mtm_score.py            Memory score update
└── db_migrations/                Schema migration scripts

tests/                            55 test files, pytest
├── test_*.py                     Unit + integration (55 files)
├── conftest.py                   Pytest fixtures
├── direct_route_client.py        Route testing client
└── integration/                  Integration tests (3 files)
```

---

## 2. Core Data Models

`arena/models.py`에 정의된 핵심 도메인 모델:

### Trade Domain
| Class | Purpose |
|-------|---------|
| `OrderIntent` | Agent의 거래 제안: ticker, side, qty, price, rationale, strategy_refs |
| `ExecutionReport` | Broker 결과: status, filled_qty, avg_price |
| `RiskDecision` | Risk 체크 결과: allowed + policy_hits |

### Account Domain
| Class | Purpose |
|-------|---------|
| `AccountSnapshot` | Cash + equity + positions 스냅샷 |
| `Position` | 단일 보유: ticker, qty, market_value_krw |

### Communication
| Class | Purpose |
|-------|---------|
| `BoardPost` | 에이전트 간 메시지 (draft/execution 라운드) |
| `MemoryEvent` | Multi-tier memory: event_type, summary, scores, tier, tags, decay, graph |

#### MemoryEvent 확장 필드

| Field | Type | Description |
|-------|------|-------------|
| `memory_tier` | str \| None | `working` / `episodic` / `semantic` |
| `expires_at` | TIMESTAMP | Tier-based TTL 만료 시점 |
| `promoted_at` | TIMESTAMP | Semantic 승격 시점 |
| `semantic_key` | str | Semantic 중복 제거 키 (thesis_id 저장에도 사용) |
| `context_tags` | dict | regime/strategy/sector/tickers 태그 |
| `primary_regime` | str | 주요 시장 체제 (bull/bear/sideways/high_vol/low_vol) |
| `primary_strategy_tag` | str | 주요 전략 (momentum/mean_reversion 등) |
| `primary_sector` | str | 주요 섹터 (tech/energy/healthcare 등) |
| `access_count` | int | 총 조회 횟수 |
| `last_accessed_at` | TIMESTAMP | 마지막 검색 시점 |
| `decay_score` | float | 현재 감쇠 배수 |
| `effective_score` | float | 감쇠 적용 최종 점수 |
| `graph_node_id` | str | `mem:<event_id>` 그래프 노드 |
| `causal_chain_id` | str | `chain:intent:<id>` 또는 `chain:cycle:<agent>:<cycle>` |

### Enums
- `Side`: BUY | SELL
- `ExecutionStatus`: REJECTED | SIMULATED | SUBMITTED | FILLED | ERROR

---

## 3. Data Layer (BigQuery)

### 3.1 Architecture

데이터 레이어는 모듈화된 store 패턴으로 구성:

```
BigQueryRepository (bq.py, 141L)  ← 얇은 facade
  │
  └── bigquery/ (store 구현체)
      ├── session.py         BigQuerySession — 연결 관리 + 테넌트 격리
      ├── memory_bq_store.py MemoryBQStore — 메모리/보드/그래프/브리핑
      ├── market_store.py    MarketStore — 시장 데이터/feature
      ├── sleeve_store.py    SleeveStore — 가상 계좌/NAV/포지션
      ├── execution_store.py ExecutionStore — 주문/체결
      ├── ledger_store.py    LedgerStore — append-only 원장
      ├── runtime_store.py   RuntimeStore — 설정/자격증명
      └── backtest_store.py  BacktestStore — 백테스트
```

### 3.2 Schema & Tables (`schema.py`)

16+ 테이블, 날짜 파티셔닝 + tenant/agent/ticker 클러스터링:

| Table | Purpose |
|-------|---------|
| `agent_order_intents` | 모든 거래 제안 (allowed=true/false) |
| `execution_reports` | 브로커 실행 결과 |
| `agent_memory_events` | Multi-tier memory (append-only, tier/tags/decay 확장) |
| `memory_access_events` | 메모리 조회 이력 (access_type, retrieval_score, used_in_prompt) |
| `memory_graph_nodes` | 인과 그래프 노드 (memory/intent/execution/board/research) |
| `memory_graph_edges` | 인과 그래프 엣지 (ABSTRACTED_TO, REFERENCES, INFORMED_BY 등) |
| `board_posts` | 에이전트 간 게시물 |
| `account_snapshots` | 계좌 스냅샷 |
| `market_features` | 시장 데이터 (OHLCV + 파생) |
| `agent_nav_daily` | 에이전트별 NAV |
| `official_nav_daily` | 공식 NAV |
| `broker_trade_events` | 브로커 체결 이벤트 |
| `broker_cash_events` | 배당, 수수료, 세금, 정산 |
| `capital_events` | 자본 주입/인출 |
| `agent_transfer_events` | Sleeve 간 이체 |
| `manual_adjustments` | 수동 포지션 보정 |
| `agent_state_checkpoints` | Recovery 시드 (canonical) |
| `reconciliation_summaries` | 감사 추적 |
| `arena_config` | 런타임 설정 (append-only KV) |
| `predicted_expected_returns` | 예측 수익률 (7-모델 앙상블) |
| `research_briefings` | 리서치 브리핑 |
| `dividend_events` | 배당 이벤트 |
| `signal_daily_values` | Layer 1 signal + forward label 재료 (point-in-time) |
| `signal_daily_ic` | 각 signal의 cross-section IC 시계열 |
| `regime_daily_features` | 시장 regime 스냅샷 (vol/trend/dispersion/sentiment) |
| `fundamentals_history_raw` | 분기 발표값 원본, `announcement_date` PIT key, 출처 구분 태그 |
| `fundamentals_derived_daily` | 매일 가격과 결합한 PIT-safe ratio (pe/pb/ep/bp/roe/growth/d2e) |
| `fundamentals_ingest_runs` | KIS/SEC/FMP ingest job metadata (status, tickers_attempted, quarters_inserted) |
| `opportunity_ranker_scores_latest` | signal-IC 합산 점수 (런타임 `recommend_opportunities` 소스) |
| `opportunity_ranker_runs` | ranker 학습 run metadata (per-signal OOS accuracy, predicted_IC) |

Fundamentals 초기 백필 절차는 [`fundamentals_backfill_runbook.md`](fundamentals_backfill_runbook.md) 참고.

### 3.3 Ledger — Append-Only Foundation (`ledger_store.py`)

모든 상태는 이벤트 리플레이로 재구성:

```
agent_state_checkpoints (canonical seed)
  ↓ replay
broker_trade_events + broker_cash_events + capital_events
+ agent_transfer_events + manual_adjustments
  ↓
Expected State ←→ Broker Snapshot (비교)
```

### 3.4 Sleeve — Virtual Portfolio Split (`sleeve_store.py`)

단일 실제 브로커 계좌를 N개 가상 계좌로 분리:
- 각 에이전트에 독립 자본 할당 (`sleeve_capital_krw[agent_id]`)
- 독립 NAV, P&L, 포지션 추적
- **Chained Returns**: 자본 이벤트 시 새 베이스라인 생성, 수익률 체인 연결

### 3.5 Market Data (`market_store.py`)

- Feature rows: ticker, as_of_ts, OHLCV, returns, volatility
- 소스: `open_trading_*_quote` (장중), `*_daily` (일봉)
- 중복 제거: daily 스냅샷 우선

---

## 4. Agent System

### 4.1 ADK Agent Architecture (Modular Decomposition)

기존 단일 `adk_agents.py`(~3000L)가 13개 파일로 분리됨:

```
AdkTradingAgent.generate(context)
  │
  ├── adk_runner_bootstrap.py   런너 초기화 + 설정
  ├── adk_runner_runtime.py     실행 루프 관리
  ├── adk_runner_state.py       사이클 내 상태 추적
  │
  ├── adk_agent_flow.py         Draft/Execution 단계 오케스트레이션
  ├── adk_decision_flow.py      최종 결정 프롬프팅 + 보드 통신
  ├── adk_prompting.py          프롬프트 빌딩 + JSON 파싱
  │
  ├── adk_context_tools.py      에이전트 컨텍스트 도구 (벡터검색, 포트폴리오 진단 등)
  ├── adk_tool_compaction.py    도구 결과 요약/압축
  ├── adk_tool_config.py        도구 선택/설정
  │
  ├── adk_order_support.py      주문 지원 유틸리티 (시장 데이터, 거래소 코드)
  ├── adk_models.py             LLM 모델 래핑/라우팅
  │
  └── adk_agents.py             최상위 클래스 + 빌더 (838L)
```

### 4.2 Model Mapping

| Agent | Provider | Routing |
|-------|----------|---------|
| GPT-5.2 | LiteLlm (OpenAI) | `litellm/gpt-5.2` |
| Gemini 3 Flash | Native ADK | `google.adk.models.Gemini` |
| Claude Sonnet 4.6 | Vertex Anthropic | Vertex alias |

Per-agent 모델 오버라이드: `agents_config[].model` in config

### 4.3 Provider Registry (`providers/registry.py`)

4개 내장 프로바이더:

| Provider | Transport | Capabilities |
|----------|-----------|-------------|
| GPT | openai_compatible | ADK, direct_text, compaction |
| Gemini | gemini_native | ADK, direct_text, grounded_search, compaction, vertex_env |
| Claude | anthropic_native | ADK, direct_text, compaction, vertex_setting |
| DeepSeek | openai_compatible | ADK, direct_text, compaction |

### 4.4 Research Agent (`research_agent.py`)

- Gemini + Google Search Grounding
- 트레이딩 사이클 전 실행
- 4단계: 글로벌 → 지정학 → 섹터 → 보유종목

### 4.5 Memory Compaction Agent (`memory_compaction_agent.py`)

- 사이클 후 실행
- 실행 로그 + 보드 포스트 + thesis chain → helper LLM으로 교훈 추출
- `MemoryEvent(event_type='strategy_reflection')` 저장
- Thesis chain 기반 post-mortem 우선

---

## 5. Long-Term Memory System

### 5.1 Multi-Storage Architecture

```
BigQuery (canonical)          Firestore (vector)           In-Memory (cache)
agent_memory_events           agent_memories/{eid}         LRU cache
+ memory_access_events        768-dim embeddings           embed_cache_max
+ memory_graph_nodes          semantic search + metadata   per-cycle lifecycle
+ memory_graph_edges          tier/regime/sector filters
```

### 5.2 Memory Event Types (8종)

| Type | Description | Default Tier | Default Importance |
|------|-------------|--------------|-------------------|
| `trade_execution` | BUY/SELL 체결 기록 | episodic | FILLED: 0.75, 기타: 0.35 |
| `thesis_open` | 투자논문 개시 | episodic | 0.58 |
| `thesis_update` | 논문 업데이트 (추가매수/조정) | episodic | 0.62 |
| `thesis_invalidated` | 논문 무효화 (전제 붕괴) | episodic | 0.78 |
| `thesis_realized` | 논문 실현 (목표 도달) | episodic | 0.74 |
| `strategy_reflection` | Compaction이 추출한 교훈 | semantic | 0.50 |
| `manual_note` | 에이전트 save_memory 도구 | episodic | 0.50 |
| `react_tools_summary` | REACT 도구 결과 요약 | working | 0.50 |

### 5.3 Investment Thesis Lifecycle (`thesis.py`)

하나의 투자 판단(포지션)을 추적:

```
thesis_open ──▶ thesis_update ──▶ thesis_realized (성공)
                     │
                     └──▶ thesis_invalidated (실패, 높은 점수 0.78)
```

- `thesis_id` = `thesis:{agent}:{ticker}:{mode}:{date}:{intent_id}`
- `semantic_key` 컬럼에 저장 → 기존 논문 조회
- Compaction Agent가 닫힌 thesis chain을 모아 post-mortem 생성
- PnL 역피드백: SELL 시 과거 BUY 기억의 outcome_score를 실제 수익률로 갱신

### 5.4 Memory Tier Hierarchy

```
Working (TTL: 36h)  ──promote──▶  Episodic (TTL: 90d)  ──promote──▶  Semantic (permanent)
 fast decay (2.0x)                 standard decay (1.0x)               slow decay (0.35x)
```

### 5.5 Context Tagging (`tags.py`)

메모리 저장 시 자동으로 컨텍스트 태그 추출:
- **regime**: bull / bear / sideways / high_vol / low_vol
- **strategy**: momentum / mean_reversion / breakout / sizing / rebalancing
- **sector**: SECTOR_BY_TICKER 매핑 (US 101 + KOSPI 579)
- **tickers**: 정규식 추출 (최대 4개)

### 5.6 Causal Graph (`graph.py`)

```
Node Types: mem:<eid>, intent:<id>, exec:<oid>, post:<pid>, brief:<bid>

Edge Types:
  ABSTRACTED_TO  ── strategy_reflection → 원본 이벤트
  REFERENCES     ── 메모리 → 메모리
  INFORMED_BY    ── 보드/리서치 → 메모리
  PRECEDES       ── 주문의도 → 메모리
  RESULTED_IN    ── 체결보고서 → 메모리
  EXECUTED_AS    ── 주문의도 → 체결보고서
```

- `causal_chain_id`로 같은 의사결정 체인 묶기

### 5.7 Adaptive Forgetting (`forgetting.py`)

```
effective_score = base_score × decay_multiplier
decay_multiplier = max(decay_factor ^ (staleness_days × tier_weight / access_boost), min_effective_score)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_decay_factor` | 0.985 | 일별 감쇠율 |
| `tier_weight_working` | 2.0 | Working 빠른 감쇠 |
| `tier_weight_episodic` | 1.0 | Episodic 표준 |
| `tier_weight_semantic` | 0.35 | Semantic 느린 감쇠 |
| `min_effective_score` | 0.15 | 감쇠 하한 |
| `access_curve` | sqrt | sqrt / log / capped_linear |

### 5.8 Vector Search & Reranking

1. **Semantic Search**: Firestore vector nearest-neighbor (top-K, 768-dim)
2. **Pre-filters**: tenant_id, agent_id, trading_mode
3. **Reranking Bonuses**:
   - Type: reflection +0.45, trade +0.28, manual +0.16, react -0.12
   - Recency: 3d +0.08, 14d +0.05, 45d +0.02
   - Ticker overlap: base +0.30, per extra +0.05, max +0.40
   - Outcome: max +0.18
   - Effective score: max +0.08
   - Tag match: regime +0.25, strategy +0.18, sector +0.10
4. **Injection**: Top context_limit(기본 32)개 → agent prompt

### 5.9 Memory Policy (`policy.py` — Single Source of Truth)

10개 그룹, 70+ 설정, UI에서 실시간 편집 가능:

| Group | Key Fields | Default |
|-------|------------|---------|
| **Storage** | `embed_cache_max` | 128 |
| **Event Types** | 8개 toggle | 전부 ON |
| **Hierarchy** | `enabled`, TTL hours/days | **OFF** |
| **Tagging** | `enabled`, bonuses | **OFF** |
| **Forgetting** | `enabled`, decay, curves | **OFF** |
| **Forgetting Tuning** | `enabled`, mode, EMA | **OFF** |
| **Graph** | `enabled`, hops, nodes | **OFF** |
| **Compaction** | `enabled`, limits | **ON** |
| **Retrieval** | `vector_search_enabled`, reranking | **ON** |
| **REACT Injection** | per-tool toggles | **ON** |
| **Cleanup** | `enabled`, age, score | **OFF** |

### 5.10 Memory Cleanup (`cleanup.py`)

```
1. recompute_forgetting_scores()     ← effective_score 재계산
2. cleanup_candidates()              ← max_age_days(180) + min_score(0.30)
3. delete_cleanup_candidates()       ← BigQuery 삭제
4. delete_firestore_vectors()        ← Firestore 벡터 삭제
```

---

## 6. Tools System

### 6.1 Registry (`default_registry.py`)

18개 핵심 도구 등록. `disabled_tools` 설정으로 개별 비활성화 가능.

### 6.2 Context Tools (5개) — `adk_context_tools.py`

| Tool | Function |
|------|----------|
| `search_past_experiences` | Firestore 벡터 검색 (과거 기억) |
| `search_peer_lessons` | 다른 에이전트의 교훈 검색 |
| `get_research_briefing` | BQ 리서치 브리핑 조회 |
| `portfolio_diagnosis` | 집중도/팩터/스트레스 진단 |
| `save_memory` | observation/reflection 저장 |

### 6.3 Quant Tools (9개) — `quant_tools.py`

| Tool | Function |
|------|----------|
| `recommend_opportunities` | signal-IC meta-learner 기반 추천. prep 단계에서 signal-IC 학습 → `opportunity_ranker_scores_latest`에 predict_IC × signal 합산 점수 저장, runtime은 읽기만. Tactical ETP 프로필 자동 분리 |
| `screen_market` | 내부 후보 생성용 Momentum/volatility/return 필터 |
| `optimize_portfolio` | Max-Sharpe, HRP, forecast-enhanced |
| `forecast_returns` | 7-model ensemble → prob_up |
| `momentum_rank` | Multi-window (20/60/126일) momentum |
| `technical_signals` | RSI, MACD, Bollinger, SMA |
| `sector_summary` | GICS sector returns/volatility |
| `get_fundamentals` | US (P/E, P/B), KR (ROE, 부채비율) |
| `index_snapshot` | US + KOSPI 지수 (마켓별 자동 라우팅) |

### 6.4 Macro (1개) — `macro_tools.py`

| Tool | Function |
|------|----------|
| `macro_snapshot` | US→FRED / KR→ECOS 한국은행 |

### 6.5 Sentiment (3개) — `sentiment_tools.py`

| Tool | Function |
|------|----------|
| `fear_greed_index` | VIX 기반 |
| `earnings_calendar` | Nasdaq 실적 일정 |
| `fetch_reddit_sentiment` | Reddit 감성 |
| `fetch_sec_filings` | SEC EDGAR |

### 6.6 Allocation Strategies (`allocation.py`)

| Strategy | Description |
|----------|-------------|
| `optimize_max_sharpe` | Sharpe 최대화 |
| `optimize_min_vol` | Minimum volatility |
| `optimize_hrp` | Hierarchical Risk Parity |
| `optimize_blend` | 60/40 Max-Sharpe + HRP |
| `optimize_forecast_sharpe` | Historical mu + forecast 블렌드 |

---

## 7. Context Builder

`arena/context.py` (1,965L) — 에이전트별, 사이클별 컨텍스트 조립.

### Input → Output

```
Inputs:                              Output:
├── Latest market features    ──→    {
├── Agent sleeve state               "agent_id", "target_market",
├── Recent board posts               "sleeve_cash_krw", "sleeve_nav_krw",
├── Top memory events                "holdings", "market_features",
├── System prompt template           "past_memories", "peer_lessons",
├── Research briefings               "board_posts", "system_prompt",
├── FX rates                         "fx_info", "risk_policy",
                                     "order_budget", "graph_context"
                                   }
```

### Key Features
- Multi-market 지원 (comma-separated: `"nasdaq,kospi"`)
- Cash buffer 강제 (min 10% 현금 유지)
- Memory assembly: tier(working/episodic/semantic) + TTL + forgetting curve + vector reranking + graph expansion
- REACT injection: 도구 실행 중 관련 기억 자동 주입

---

## 8. Execution Pipeline

### 8.1 Order Flow

```
Agent.generate(context)
  │
  ├── OrderIntent[] (ticker, side, qty, price_krw, rationale, strategy_refs)
  │
  ▼
ExecutionGateway.process(intent, snapshot)
  │
  ├── 1. Fetch risk metrics (daily turnover, order count, last trade time)
  ├── 2. RiskEngine.evaluate() → RiskDecision
  ├── 3. Record intent → agent_order_intents (BQ)
  ├── 4. If allowed: broker.place_order() → ExecutionReport
  ├── 5. Record → execution_reports + MemoryStore (trade_execution + thesis)
  └── 6. Return ExecutionReport
```

### 8.2 Broker Abstraction

| Broker | Usage |
|--------|-------|
| `KISOpenTradingBroker` | 실거래 (KIS API → US + KOSPI) |
| `PaperBroker` | 시뮬레이션 (즉시 체결) |
| `KISHttpBroker` | 사용자 엔드포인트 기반 |

---

## 9. Risk Engine

`arena/risk.py` — Stateless function: intent + snapshot → decision

### Policy Checks

| Check | Condition |
|-------|-----------|
| `equity_non_positive` | total_equity_krw <= 0 |
| `ticker_market_mismatch` | ticker가 대상 시장에 미해당 |
| `max_order_krw` | notional > 한도 |
| `max_daily_turnover` | 일간 회전율 초과 |
| `max_daily_orders` | 일간 주문 횟수 초과 |
| `ticker_cooldown` | 동일 종목 재거래 쿨다운 |
| `no_position` | SELL인데 보유 없음 |
| `insufficient_position` | SELL 수량 > 보유 수량 |

Per-agent 오버라이드: `AgentConfig.risk_overrides` 병합 후 평가.

---

## 10. Reconciliation & Recovery

`arena/reconciliation.py` (1,374L)

### StateReconciliationService

```
1. Load canonical seed (agent_state_checkpoints)
2. Replay all ledger events
3. Compute expected positions/cash
4. Compare vs. broker snapshot
5. Position mismatch → ERROR (cycle block)
   Negative agent cash → ERROR (cycle block)
   Broker residual cash → WARNING (proceed)
```

### StateRecoveryService

```
1. Position mismatch → checkpoint override (브로커 스냅샷으로 리셋)
2. Negative cash → manual_adjustment 생성
3. Re-reconcile with new seed
```

---

## 11. Open Trading Integration (KIS)

### 11.1 OpenTradingClient (`client.py`, 1,913L)

- OAuth 토큰 관리 (Firestore 캐시)
- US + KR 시장 데이터 (일봉, 호가, 지수)
- 계좌 조회 (잔고, 포지션)
- 주문 (해외/국내)
- 배당 (period_rights, KSD)

### 11.2 Sync Services (`sync.py`, 2,771L)

| Service | Function |
|---------|----------|
| `MarketDataSyncService` | US/KOSPI 일봉 + 실시간 호가 |
| `AccountSyncService` | 계좌 스냅샷 (US+KR 병합) |
| `BrokerTradeSyncService` | 체결 거래 이력 동기화 |
| `BrokerCashSyncService` | 현금 흐름 (수수료, 배당, 이자) |
| `DividendSyncService` | US period_rights + KR KSD 배당 |

---

## 12. UI Layer

FastAPI Admin Dashboard — 모듈화된 라우트 구조.

### Route Structure

| Module | Path | Purpose |
|--------|------|---------|
| `auth.py` | `/login`, `/callback`, `/logout` | Google OAuth |
| `overview.py` | `/` | 대시보드 오버뷰 |
| `board.py` | `/board` | 에이전트 간 게시물 |
| `nav.py` | `/nav` | NAV 차트 |
| `trades.py` | `/trades` | 실행 이력 |
| `sleeves.py` | `/sleeves` | 슬리브 관리 |
| `ops.py` | `/ops` | 운영 상태 |
| `settings_page.py` | `/settings` | 설정 페이지 |
| `settings_admin.py` | `/admin/*` | 설정 CRUD (에이전트, 리스크, 도구, MCP, 메모리) |
| `settings_render_agents.py` | — | 에이전트 설정 패널 |
| `settings_render_capital.py` | — | 자본 관리 |
| `settings_render_credentials.py` | — | KIS/API 키 관리 |
| `capital_data.py` | `/api/capital/*` | 자본 데이터 API |

### Memory 3D Graph (`memory.py`)

10개 그룹 → Branch → Leaf Field 구조. Click node → edit value → save to arena_config.

### Templates

11개 Jinja2 템플릿: `base_layout`, `overview_body`, `board_body`, `nav_body`, `trades_body`, `sleeves_body`, `settings_body`, `ops_body`, `auth_notice`, `inline_notice`, `board_header_datepicker`.

---

## 13. Market Hours & Scheduling

`arena/market_hours.py` (318L)

| Market | Session | Timezone |
|--------|---------|----------|
| NASDAQ/NYSE | 09:30-16:00 | America/New_York |
| KOSPI/KOSDAQ | 09:00-15:30 | Asia/Seoul |

### Holiday Detection
- **US**: Static 계산 (MLK, Presidents' Day, Easter, etc.)
- **KOSPI**: `korean_lunar_calendar` (설날, 추석, 부처님 오신 날) + 고정 9개 + 대체휴일

---

## 14. Configuration & Runtime Overrides

`arena/config.py` (986L)

### Precedence Contract

```
1. load_settings()        .env / 환경변수 → 기본 Settings
2. _build_runtime()       tenant_id 결정 + BigQueryRepository
3. _apply_tenant_runtime_credentials()  Secret Manager 메타 + 자격증명
4. apply_runtime_overrides()            arena_config tenant별 값 → Settings 오버레이
5. apply_distribution_mode()            safety gate
```

### Runtime Source of Truth

| Layer | Storage | Scope | Used For |
|------|---------|-------|----------|
| Boot defaults | `.env` / env vars | process | 기본 Settings 부트스트랩 |
| Secret metadata | `runtime_credentials` | tenant | KIS/모델 secret, key availability |
| Editable config | `arena_config` | tenant | 프롬프트, 리스크, 에이전트, 메모리 정책 |
| Safety gate | distribution_mode, real_trading_approved | process + tenant | 실거래 허용 여부 |

### Key Config Keys (`arena_config`)

| Key | Type | Description |
|-----|------|-------------|
| `system_prompt` | text | 에이전트 시스템 프롬프트 |
| `agents_config` | JSON | 에이전트 CRUD (provider/model/capital/risk/tools) |
| `risk_policy` | JSON | Risk 파라미터 |
| `sleeve_capital_krw` | scalar | 기본 sleeve 자본 |
| `disabled_tools` | JSON | 비활성화 도구 |
| `mcp_servers` | JSON | MCP 서버 등록 |
| `memory_policy` | JSON | 메모리 정책 (10 groups) |
| `forecast_mode` | scalar | 예측 사용 모드 |
| `kis_target_market` | scalar | tenant별 타깃 시장 |
| `real_trading_approved` | bool | 실거래 승인 스위치 |

### tenant / trading_mode Contract

- `tenant_id`는 운영 단위. 설정, 자격증명, UI 접근, BQ 조회 기준.
- `trading_mode`는 실행 레인. paper/live는 별도 데이터 스트림.
- `memory_policy`를 포함한 arena_config는 tenant 단위.

---

## 15. CLI Interface

Entry point: `arena/cli.py` (309L) → `arena/cli_commands/` (8개 모듈)

```bash
# Setup
llm-arena init-bq                          # 데이터셋 + 테이블 생성

# Market Data Sync
llm-arena sync-market --market us           # 일봉 OHLCV
llm-arena sync-market-quotes                # 장중 시세
llm-arena sync-account                      # 브로커 계좌 스냅샷
llm-arena sync-broker-trades                # 체결 이력
llm-arena sync-broker-cash                  # 현금 이벤트
llm-arena sync-dividends                    # 배당 귀속

# Forecasting
llm-arena build-forecasts                   # 7-model ensemble
llm-arena build-opportunity-ranker          # point-in-time ML ranker + latest scores

# Trading
llm-arena run-pipeline --live --market us         # Full: sync → forecast → ranker → agents
llm-arena run-shared-prep --live --market us      # Shared sync/forecast/ranker prep only
llm-arena run-agent-cycle --live --all-tenants    # Agent cycle only
llm-arena run-batch --live --all-tenants          # Manual sync + cycle shortcut

# Reconciliation
llm-arena recover-sleeves                         # 자동 복구 + checkpoint 재생성

# Admin
llm-arena serve-ui                                # FastAPI UI (port 8080)
llm-arena list-strategies                         # Strategy reference cards
llm-arena serve-strategy-mcp                      # MCP server
llm-arena promote-tenant-live --tenant <id>      # private + live approval
llm-arena set-tenant-simulated --tenant <id>     # simulated-only onboarding
```

### CLI Command Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `run_pipeline.py` | 472 | sync → forecast → ranker → agent |
| `run_agent.py` | 510 | agent cycle only |
| `run_shared.py` | 460 | shared sync/forecast/ranker |
| `sync.py` | 343 | market/account sync |
| `admin.py` | 336 | tenant/memory admin |
| `serve.py` | 265 | UI + MCP server |
| `run_reconcile.py` | 227 | reconciliation |
| `run.py` | 45 | dispatch routing |

---

## 16. Deployment

### GCP Architecture

```
Cloud Run Job (asia-northeast3)
├── A-Split Mode (고성능):
│   ├── Prep Job (1 task)      ← 동기화 + 리서치
│   └── Agent Job (10 tasks)   ← 에이전트 병렬 실행
│
├── Dual-Market Jobs:
│   ├── llm-arena-batch-{prep,agent}-us    (ET 15:00)
│   └── llm-arena-batch-{prep,agent}-kospi (KST 14:30)
│
└── Resources: CPU 4, Memory 16Gi, Timeout 3600s

Cloud Run Service
└── arena-ui  (CPU 1, 512Mi, 동시성 80)
```

### UI Onboarding Lifecycle

1. Google 로그인 → 전용 tenant 자동 provisioning
2. 초기: `distribution_mode=simulated_only`, `real_trading_approved=false`
3. KIS demo 계정 저장 → `paper_only` 전환
4. 운영자 `promote-tenant-live` → `private` live 승격

### Dockerfiles

| File | Purpose |
|------|---------|
| `Dockerfile` | Main pipeline (Python 3.12 + PyTorch, neuralforecast, chronos, timesfm) |
| `Dockerfile.ui` | UI service (경량) |
| `Dockerfile.forecast` | Standalone forecast builder |

---

## 17. Test Structure

55개 테스트 파일, pytest. 100% pass 필수.

### Key Test Files

| Test File | Coverage |
|-----------|----------|
| `test_adk_agents.py` | ADK normalization, model routing, tool loading |
| `test_agents_config.py` | Per-agent config CRUD, capital allocation |
| `test_context.py` | Context builder, memory reranking, cash buffer |
| `test_data_strict_paths.py` | BQ queries, dedup, checkpoint rebuild |
| `test_memory_store.py` | Scoring, dedup, tier assignment, tagging |
| `test_memory_forgetting.py` | Decay math, effective_score, access curves |
| `test_memory_graph.py` | Node/edge builders, causal chain inference |
| `test_memory_tuning.py` | Tuner grid search, objective, auto-promote/demote |
| `test_memory_bq_store.py` | Memory BQ store operations |
| `test_cli_thesis_compaction_smoke.py` | Thesis chain compaction smoke |
| `test_execution_reconcile.py` | Order reconciliation vs. broker |
| `test_forecasting_stacked.py` | 7-model stacking |
| `test_ui_admin_routes.py` | Admin pages, config save |
| `test_ui_helper_modules.py` | UI helper modules |
| `test_tenant_leases.py` | Firestore execution lease |
| `test_provider_registry.py` | Provider module |
| `test_market_sources.py` | Market source resolution |

---

## 18. Design Patterns

### Repository Pattern (Modular Stores)
- `BigQueryRepository` → 얇은 facade
- 실제 구현: `bigquery/` 디렉토리의 store별 클래스
- Tenant scoping 내장

### Protocol-Based Agents
```python
class TradingAgent(Protocol):
    agent_id: str
    def generate(context: dict) -> AgentOutput: ...
```

### Event Sourcing (Ledger)
- 모든 상태 = append-only 이벤트 리플레이
- Checkpoint는 recovery seed, canonical source는 이벤트

### Agent Decomposition
- 13개 파일로 ADK 에이전트 책임 분리
- 테스트 용이성 + 단일 책임 원칙

### Virtual Sleeving
- 단일 브로커 계좌 → N개 가상 sleeve
- 리플레이 기반 독립 추적

---

## 19. Data Flow Walkthrough

**한 트레이딩 사이클의 전체 흐름:**

```
 1. Scheduler trigger (15:00 ET / 14:30 KST)
    │
 2. Tenant runtime hydrate
    │  runtime_credentials → Secret Manager → Settings
    │  arena_config → Settings 오버레이
    │
 3. Sync Phase (parallel)
    ├── sync_market_features()
    ├── sync_account_snapshot()
    ├── sync_broker_trades()
    ├── sync_broker_cash()
    └── sync_dividends()
    │
 4. Reconciliation
    │  checkpoint → replay events → compare vs. broker
    │  auto_recover() if issues
    │
 5. Forecast
    │  build_and_store_stacked_forecasts()
    │  7-model ensemble → predicted_expected_returns
    │
 6. Research Agent (Gemini + Google Search)
    │  held tickers + movers → briefing board post
    │
 7. ┌─ Draft Round (all agents parallel) ─────────────┐
    │  context_builder.build(agent_id)                  │
    │    → market + sleeve + memory + board + research  │
    │  agent.generate(context) → ReAct loop → intents   │
    │  board_store.publish(post)                        │
    └───────────────────────────────────────────────────┘
    │
 8. ┌─ Execution Round (all agents parallel) ──────────┐
    │  Re-context (with draft board posts)              │
    │  Generate intents (draft-aware decisions)         │
    │  gateway.process() → risk → broker → memory       │
    │  thesis lifecycle tracking (open/update/...)      │
    └───────────────────────────────────────────────────┘
    │
 9. Memory Compaction & Maintenance
    │  memory_compaction_agent.run()
    │  ├── thesis chain post-mortem (닫힌 논문 분석)
    │  ├── 사이클 이벤트 → 교훈 추출 (strategy_reflection)
    │  ├── context_tags 자동 추출
    │  ├── graph node/edge 생성
    │  └── vector store 인덱싱
    │
10. NAV Snapshot (official_nav_daily per agent)
    │
11. Cycle complete. Next in 24h.
```

---

## 20. Gotchas & Important Notes

1. **Sleeves are virtual** — 실제 브로커 계좌는 공유. Sleeve 현금 부족 시 주문 실패 가능.
2. **Checkpoints are canonical** — `agent_state_checkpoints`가 진실의 시드.
3. **Ledger replay** — 항상 checkpoint부터 리플레이로 상태 재계산.
4. **Memory features mostly OFF by default** — hierarchy, tagging, forgetting, graph, cleanup 전부 기본 OFF. 실제 동작: 벡터검색 + thesis + compaction + REACT injection.
5. **thesis_id ≠ graph_node_id** — thesis_id는 비즈니스 키(포지션 묶기), graph_node_id는 그래프 주소(인과관계).
6. **Data layer split** — 기존 `bq.py`(1200L+) → `bq.py`(141L facade) + `bigquery/` 디렉토리(7개 store).
7. **Agent decomposition** — 기존 `adk_agents.py`(~3000L) → 13개 파일로 분리.
8. **CLI modularization** — 기존 `cli.py`(~2700L) → `cli.py`(309L) + `cli_commands/`(8개 모듈).
9. **PnL backfeed** — SELL 시 과거 BUY 기억의 outcome_score를 실제 수익률로 역업데이트.
10. **Risk per-agent** — `Settings.risk_policy` + `AgentConfig.risk_overrides` 병합.
11. **ADK tools are async** — LLM 호출은 ReAct 루프에서 블로킹. 동시 에이전트는 ThreadPoolExecutor.
12. **Tenant isolation** — 모든 쿼리 `tenant_id` 필터.
13. **Market hours** — 스케줄링 전 holiday 체크 필수.

---

## 21. Quick Reference

| Layer | Key Files | Lines |
|-------|-----------|-------|
| **Models** | `arena/models.py` | 152 |
| **Config** | `arena/config.py` | 986 |
| **Data** | `arena/data/bq.py` + `bigquery/` (7 stores) | 141 + 5,873 |
| **Schema** | `arena/data/schema.py` | 701 |
| **Agents** | `arena/agents/` (13 adk_* + 3 others) | 4,079 + 1,093 |
| **Memory** | `arena/memory/` (10 files) | 5,760 |
| **Tools** | `arena/tools/` (8 files) | 4,303 |
| **Context** | `arena/context.py` | 1,965 |
| **Orchestration** | `arena/orchestrator.py` | 404 |
| **Execution** | `arena/execution/gateway.py` + `risk.py` | 320 + 94 |
| **Broker** | `arena/broker/` (3 files) | 617 |
| **Open Trading** | `arena/open_trading/` (4 files) | 4,852 |
| **Reconciliation** | `arena/reconciliation.py` | 1,374 |
| **CLI** | `arena/cli.py` + `cli_commands/` (8 files) | 309 + 2,659 |
| **UI** | `arena/ui/` (20+ files) | ~6,200 |
| **Providers** | `arena/providers/` (2 files) | 339 |
| **Forecasting** | `arena/forecasting/stacked.py` | 679 |
| **Tests** | `tests/` (55 files) | ~19,000 |
