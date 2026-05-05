# LLM Invest — Developer Architecture Guide

> BigQuery + ADK + KIS Open Trading API 기반 멀티 LLM 자동투자 플랫폼
> 3 에이전트(GPT-5.2, Gemini 3 Flash, Claude Sonnet 4.6)가 US + KOSPI/KOSDAQ 시장에서 경쟁

---

## Table of Contents

1. [Directory Structure](#1-directory-structure)
2. [Core Data Models](#2-core-data-models)
3. [Data Layer (BigQuery + Local DuckDB)](#3-data-layer-bigquery--local-duckdb)
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
│   ├── adk_agents.py            Main agent class + builder (1,680L)
│   ├── adk_agent_flow.py        Draft/execution phase orchestration (89L)
│   ├── adk_context_tools.py     Per-cycle context tools for agents (1,063L)
│   ├── adk_decision_flow.py     Decision prompting + board comm (147L)
│   ├── adk_models.py            ADK model wrapper/routing (238L)
│   ├── adk_order_support.py     Order placement utilities (529L)
│   ├── adk_prompting.py         Prompt building + JSON parsing (129L)
│   ├── adk_runner_bootstrap.py  Runner initialization (354L)
│   ├── adk_runner_runtime.py    Runtime execution logic (298L)
│   ├── adk_runner_state.py      Mutable execution state tracking (545L)
│   ├── adk_tool_compaction.py   Tool result compaction (637L)
│   ├── adk_tool_config.py       Tool configuration/selection (140L)
│   ├── adk_tool_helpers.py      ADK tool schema/no-op shims shared by cycle/dev_ui/investment_chat (58L)
│   ├── base.py                  TradingAgent protocol (24L)
│   ├── llm_params.py            Per-provider sampling/token params (165L)
│   ├── memory_compaction_agent.py  Post-cycle lesson synthesis (952L)
│   ├── research_agent.py        Gemini + Google Search Grounding (323L)
│   ├── support_model.py         Helper model builder (247L)
│   ├── investment_chat/          User-facing chat advisor agent (17 files)
│   │   ├── factory.py           ADK Agent builder + chat tool wrapping (125L)
│   │   ├── registry.py          chat tool registry (analysis whitelist + approval-only write tools) (54L)
│   │   ├── tools.py             account/history/order/config entry aggregator (19L)
│   │   ├── account_tools.py     stored snapshot + KIS account refresh + sleeve snapshot (180L)
│   │   ├── history_tools.py     persisted trade history reader (190L)
│   │   ├── order_tools.py       validate_order_draft + submit_approved_order (button-approved bridge) (765L)
│   │   ├── config_tools.py      propose/apply approval-gated runtime config changes (927L)
│   │   ├── drafts.py            order/config approval_token + arena_config-backed draft store (67L)
│   │   ├── audit.py             append_runtime_audit_log + tenant config wrapper (60L)
│   │   ├── memory.py            chat-decision semantic reflection writer (56L)
│   │   ├── scope.py             account vs agent_sleeve scope + strategy_refs (69L)
│   │   ├── market_scope.py      chat account market allowlist parsing (52L)
│   │   ├── locks.py             tenant-keyed write RLock (12L)
│   │   ├── context.py           ContextVars (tenant/user/provider/model) + tenant normalizer (12L)
│   │   ├── constants.py         APP_NAME/AGENT_ID + analysis whitelist + write markers (40L)
│   │   ├── utils.py             snapshot/sources/tenant scoping helpers (120L)
│   │   └── __init__.py
│   └── prompts/                  Compatibility shim → re-exports arena.prompts.prompt_pack
├── prompts/                      Central prompt templates + renderers
│   ├── loader.py                File loader + cached `format_map` renderer (42L)
│   ├── prompt_pack.py           PromptPack: cycle prompts + investment chat instruction (320L)
│   ├── memory.py                Memory compaction + relation extraction prompt builders (51L)
│   ├── adk/                     core_prompt.txt + system_prompt.txt (cycle agent defaults)
│   ├── investment_chat/         system_prompt.txt (chat advisor instruction)
│   └── memory/                  compaction_system / relation_extraction_system / relation_extraction_user_template
├── memory/                       Multi-tier long-term memory system
│   ├── policy.py                Single source for all memory controls — 10 groups (2,817L)
│   ├── store.py                 Write/retrieve events with tier + tagging + graph (887L)
│   ├── vector.py                Vertex AI embeddings + Firestore search (288L)
│   ├── vector_factory.py        GCP/local vector store selector (44L)
│   ├── vector_local.py          ChromaDB + sentence-transformers local store + NullVectorStore fallback (271L)
│   ├── thesis.py                Investment thesis lifecycle tracking (170L)
│   ├── graph.py                 Causal graph node/edge builders (478L)
│   ├── tags.py                  Context tag extraction — regime/strategy/sector/ticker (317L)
│   ├── forgetting.py            Adaptive decay math + batch recompute (245L)
│   ├── tuning.py                Forgetting parameter auto-tuner — shadow/bounded_ema (821L)
│   ├── cleanup.py               Prune stale/low-signal memories (289L)
│   ├── query_builders.py        Tool result → semantic query conversion (166L)
│   ├── candidates.py            Recall candidate assembly + diversity filter (320L)
│   ├── relations.py             Deterministic triple extraction + graph projection (566L)
│   ├── relation_ontology.py     Closed predicate/entity type vocabulary (193L)
│   ├── relation_validation.py   14-step triple validator (235L)
│   ├── semantic_extractor.py    LLM triple extractor + run audit (429L)
│   └── semantic_tuning.py       Shadow↔inject auto-tuner with quality gates (638L)
├── ui/                           Admin dashboard (FastAPI, modular routes)
│   ├── app.py                   Main FastAPI router + investment-chat ADK mount (669L)
│   ├── investment_chat_adk.py   Mounts ADK FastAPI dev-ui at /investment-chat/adk (loader, sessions, static, auth gate) (792L)
│   ├── routes/                   Route modules
│   │   ├── auth.py              Google OAuth (269L)
│   │   ├── board.py             Board viewer (268L)
│   │   ├── nav.py               NAV charts (374L)
│   │   ├── trades.py            Trade history (135L)
│   │   ├── sleeves.py           Sleeve management (859L)
│   │   ├── ops.py               Operations page (224L)
│   │   ├── showcase.py          Public showcase page (564L)
│   │   ├── investment_chat.py   Chat page shell + provider/model selector + draft approval APIs (366L)
│   │   ├── settings_page.py     Settings page render (456L)
│   │   ├── settings_admin.py    Settings CRUD API (751L)
│   │   ├── settings_render.py   Render dispatcher (21L)
│   │   ├── settings_render_agents.py    Agent config panel (178L)
│   │   ├── settings_render_capital.py   Capital management (51L)
│   │   ├── settings_render_credentials.py  KIS/API credentials (638L)
│   │   ├── settings_render_scripts.py   Script management (205L)
│   │   ├── capital_data.py      Capital data API (294L)
│   │   └── viewer.py            Data viewer (49L)
│   ├── templates/               Jinja2 templates (24 files, includes investment_chat_body.jinja2 approval panels)
│   ├── vendor/                   Bundled JS libs (three.min.js, 3d-force-graph.min.js)
│   ├── memory.py                3D memory graph builder + routes (968L)
│   ├── viewer_data.py           Viewer data assembly (1,024L)
│   ├── viewer_analytics.py      Analytics computations (96L)
│   ├── layout.py                Base layout helpers + sidebar nav + collapse toggle (96L)
│   ├── http.py                  JSON/HTML response helpers (22L)
│   ├── access.py                Access control (60L)
│   ├── provisioning.py          Tenant auto-provisioning (194L)
│   ├── app_support.py           App startup support (156L)
│   ├── run_status.py            Run status tracking (144L)
│   ├── runtime.py               UI runtime context (458L)
│   ├── templating.py            Template engine setup (28L)
│   └── server.py                Startup wrapper (5L)
├── tools/                        Agent tool registry (19 core + MCP)
│   ├── default_registry.py      Build registry with all tools (429L)
│   ├── quant_tools.py           Recommend, screen, optimize, forecast, technical, trade perf (2,408L)
│   ├── sentiment_tools.py       Reddit, SEC EDGAR, earnings, VIX, news (771L)
│   ├── macro_tools.py           FRED (US), ECOS (Korea) (240L)
│   ├── allocation.py            Portfolio optimization — Sharpe, HRP, forecast (499L)
│   ├── screening.py             Momentum + discovery ranking (517L)
│   ├── sector_map.py            US 101 + KOSPI 579 sector mapping (688L)
│   ├── _market_scope.py         Per-market ticker/universe scoping helper (152L)
│   └── registry.py              ToolRegistry with two-phase selection (90L)
├── recommendation/               Signal-IC meta-learner + Layer 1 signals
│   ├── ranker.py                Builds opportunity_ranker_scores_latest (713L)
│   └── signals.py               Layer 1 signal definitions + regime features (205L)
├── data/                         Repository layer (BigQuery default, DuckDB for ARENA_MODE=local)
│   ├── bq.py                   BigQueryRepository facade (144L)
│   ├── factory.py               Backend selector — BigQuery vs LocalRepository (73L)
│   ├── protocols.py             Store protocols/interfaces (209L)
│   ├── schema.py                Shared Table DDLs + auto-migration (1,106L)
│   ├── bigquery/                BigQuery store implementations
│   │   ├── session.py           BigQuerySession connection management (277L)
│   │   ├── memory_bq_store.py   Memory events, board posts, graph, briefings, triples (1,901L)
│   │   ├── market_store.py      Price/feature/signals/IC queries (2,497L)
│   │   ├── sleeve_store.py      Virtual account operations + NAV (2,680L)
│   │   ├── execution_store.py   Order intent/execution repository (448L)
│   │   ├── ledger_store.py      Append-only event ledger (381L)
│   │   ├── llm_audit_store.py   LLM call/prompt/response audit log (209L)
│   │   ├── runtime_store.py     Config/credential storage (604L)
│   │   └── backtest_store.py    Backtest persistence (161L)
│   └── local/                   DuckDB store implementations (ARENA_MODE=local)
│       ├── repository.py        LocalRepository facade — store delegation (249L)
│       ├── session.py           DuckDBSession + write file lock (257L)
│       ├── schema.py            Shared table metadata → DuckDB DDL renderer (101L)
│       ├── market_store.py      DuckDB market features + latest view (1,721L)
│       ├── memory_store.py      DuckDB memory/board/graph (1,021L)
│       ├── sleeve_store.py      DuckDB sleeves + NAV replay (572L)
│       ├── execution_store.py   DuckDB intents/executions (400L)
│       └── config_store.py      DuckDB arena_config + credential meta (197L)
├── open_trading/                 Korea Investment API client + fundamentals ingest
│   ├── client.py                REST wrapper — OAuth, account, market data (2,249L)
│   ├── sync.py                  Market data, account, dividend sync (3,375L)
│   ├── kis_fundamentals_ingestor.py  KIS-based KR fundamentals backfill (262L)
│   ├── sec_fundamentals_ingestor.py  SEC EDGAR companyfacts ingest (605L)
│   ├── fmp_fundamentals_ingestor.py  Optional FMP fundamentals source (305L)
│   ├── exchange_codes.py        Exchange code mapping (97L)
│   ├── token_cache.py           Firestore-backed OAuth token (71L)
│   └── token_cache_file.py      Local atomic JSON token cache (77L)
├── broker/                       Order execution abstraction
│   ├── base.py                  BrokerClient protocol (13L)
│   ├── open_trading.py          Live KIS trading — US + KOSPI (567L)
│   └── paper.py                 Paper + HTTP broker (97L)
├── execution/                    Centralized order gateway
│   └── gateway.py               Risk check → broker → memory recording (477L)
├── providers/                    LLM provider registry
│   ├── registry.py              4 providers — GPT/Gemini/Claude/DeepSeek (209L)
│   ├── anthropic_patches.py     Anthropic SDK compatibility patches (69L)
│   └── credentials.py           Secret Manager credential parsing (130L)
├── security/                     Secrets management
│   ├── credential_store.py      Secret Manager + BQ (377L)
│   └── credential_store_env.py  Local JSON credential store (file-mode 0600) (196L)
├── cli_commands/                 Modular CLI command handlers
│   ├── run.py                   Command dispatch routing (49L)
│   ├── run_agent.py             Agent cycle execution (704L)
│   ├── run_pipeline.py          Full sync→forecast→ranker→agent pipeline (1,496L)
│   ├── run_shared.py            Shared sync/forecast/ranker operations (534L)
│   ├── run_reconcile.py         Reconciliation + post-cycle maintenance (330L)
│   ├── serve.py                 UI and MCP server startup (267L)
│   ├── sync.py                  Market/account/fundamentals/ranker sync (647L)
│   ├── admin.py                 Admin operations — tenant, memory (336L)
│   ├── init_local.py            DuckDB bootstrap (`init-local`) (38L)
│   ├── local_demo.py            Deterministic demo seed + local KIS backfill (121L)
│   ├── local_clone.py           BigQuery → DuckDB table clone (`clone-bq-local`) (490L)
│   └── memory_relations.py      Semantic triple extraction CLI (148L)
├── strategy/                     Strategy reference catalog
│   ├── catalog.py               Strategy cards for agents (164L)
│   └── mcp_server.py            MCP server for strategy tool (51L)
├── backtest/                     Walk-forward testing
│   └── walk_forward.py          Stabilization + periodic rebalancing (392L)
├── board/                        Inter-agent communication
│   └── store.py                 Publish/retrieve shared board posts (20L)
├── universe/                     Ticker universe presets
├── forecasting/                  ML forecast pipeline
│   └── stacked.py               7-model ensemble stacking (800L)
├── config.py                     Settings + runtime overrides (1,196L)
├── context.py                    Per-agent context builder (2,758L)
├── orchestrator.py               Multi-agent cycle orchestration (571L)
├── risk.py                       Risk engine policy checks (125L)
├── reconciliation.py             State reconciliation + recovery (1,423L)
├── market_hours.py               Market windows + holidays (357L)
├── market_sources.py             Market source resolution (52L)
├── market_feature_normalization.py  Market feature normalization helpers (150L)
├── runtime_universe.py           Runtime universe resolution (75L)
├── cli.py                        CLI entry point (457L)
├── cli_runtime.py                CLI runtime bootstrap (719L)
├── cloud_run_jobs.py             Cloud Run job dispatch (47L)
├── tenant_leases.py              Firestore execution lease (134L)
├── tenant_leases_local.py        File-locked JSON tenant lease for local mode (132L)
├── models.py                     Core data classes (156L)
├── logging_utils.py              JSON logging for Cloud Run (208L)
└── __main__.py

scripts/                          Operational scripts
├── deploy_cloud_run_job.sh       Deploy trading pipeline
├── deploy_cloud_run_ui.sh        Deploy UI
├── ship.sh                       Build+push+deploy one-command
├── dev-ui.sh                     Local UI dev server
├── cleanup_memory.py             Memory pruning batch
├── daily_mtm_score.py            Memory score update
└── db_migrations/                Schema migration scripts

tests/                            67 test files, pytest
├── test_*.py                     Unit + integration
├── conftest.py                   Pytest fixtures
├── direct_route_client.py        Route testing client
└── integration/                  Integration tests
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

## 3. Data Layer (BigQuery + Local DuckDB)

### 3.1 Architecture

데이터 레이어는 모듈화된 store 패턴 + 백엔드 팩토리로 구성. `arena/data/factory.py:get_repository`가 `Settings.arena_mode`(또는 `ARENA_MODE` env)를 읽어 두 백엔드 중 하나를 인스턴스화하고, 둘 다 `BigQueryRepository`와 동일한 facade 표면을 노출 — store 메서드 이름·시그니처가 동일해서 상위 레이어(orchestrator/agents/tools)는 어떤 백엔드인지 모릅니다.

```
get_repository(settings, tenant_id) ──▶ ARENA_MODE
                                          │
              ┌───────────────────────────┴──────────────────────────┐
              ▼                                                       ▼
BigQueryRepository (bq.py, 144L)                        LocalRepository (data/local/repository.py, 249L)
  └── bigquery/ (9 stores + session)                      └── data/local/ (5 stores + session + schema)
      ├── session.py                                          ├── session.py        DuckDB connection + filelock
      ├── memory_bq_store.py                                  ├── schema.py         Shared table metadata → DuckDB DDL renderer
      ├── market_store.py                                     ├── market_store.py
      ├── sleeve_store.py                                     ├── memory_store.py
      ├── execution_store.py                                  ├── sleeve_store.py
      ├── ledger_store.py                                     ├── execution_store.py
      ├── llm_audit_store.py                                  └── config_store.py
      ├── runtime_store.py
      └── backtest_store.py
```

### 3.1.1 Local Backend (DuckDB)

OSS quickstart / 로컬 평가용. GCP 결제·인증 없이 동일 코드 경로에서 paper 사이클을 돌리려는 목적으로 분리.

| Aspect | BigQuery (default) | Local (`ARENA_MODE=local`) |
|--------|--------------------|----------------------------|
| Storage | BigQuery 데이터셋 (project + dataset + location) | 단일 DuckDB 파일 (`./data/arena.duckdb`, `ARENA_LOCAL_DB_PATH` override) |
| Schema | `arena/data/schema.py` `TABLE_DDLS` (BigQuery 방언) | 동일 테이블/컬럼 메타데이터를 `data/local/schema.py`가 DuckDB DDL로 별도 렌더링 (STRING→VARCHAR, INT64→BIGINT, NUMERIC→DECIMAL(38,9), DATETIME→TIMESTAMP, ARRAY<T>→T[]) |
| Concurrency | BigQuery 서버측 + Firestore 트랜잭션 | `filelock`(optional) + per-write 파일락. 미설치 시 단일 프로세스 가정으로 경고 후 진행 |
| Vector store | Vertex AI 임베딩 + Firestore vector search | ChromaDB(persistent) + sentence-transformers `all-MiniLM-L6-v2` (`memory/vector_local.py`). `chromadb`/`sentence-transformers` 미설치 시 `NullVectorStore`로 폴백 — recency-only 회상 |
| Credentials | Secret Manager + `runtime_credentials` | `~/.llm-arena/credentials.json` (mode 0600) + `runtime_credentials` 메타. `EnvCredentialStore`가 동일 KIS/모델 secret 페이로드 형태 보존 |
| KIS OAuth cache | Firestore-backed `token_cache.py` | Atomic JSON `token_cache_file.py` (`~/.llm-arena/tokens.json`) |
| Tenant leases | `FirestoreTenantLeaseStore` | `LocalTenantLeaseStore` — `./data/tenant_leases.json` + filelock |
| Bootstrap | `llm-arena init-bq` | `llm-arena init-local` (DDL idempotent) → `seed-local-demo`, `backfill-local-market`, 또는 `clone-bq-local` |
| Optional install | (default) | `pip install -e ".[local]"` (duckdb + filelock) · `pip install -e ".[local,local-vector]"` 추가 시 ChromaDB 활성화 |

`LocalRepository.__getattr__` 는 5개 로컬 store에 위임하고, 미구현 surface(예: 일부 BQ-전용 dashboard 쿼리)는 `AttributeError`를 던져 `hasattr(repo, "...")` 기반 feature detection이 그대로 동작합니다.

`DuckDBSession`은 BigQuery 방언 SQL을 그대로 받기 위해 실행 전 정규식 폴리필을 통과시킵니다 — `IN UNNEST(arr) → IN (SELECT unnest(arr))`, `TIMESTAMP_SUB/ADD → ± INTERVAL`, `CURRENT_TIMESTAMP() → CURRENT_TIMESTAMP`, `DATE(ts, "tz") → CAST(ts AS DATE)`, 그리고 bound 파라미터 형태의 `INTERVAL $days DAY → ($days * INTERVAL '1 day')`. memory store에는 사이클 단위 회상을 위한 `memory_events_for_cycle`(payload JSON에서 cycle_id를 fallback 추출)이 추가되어 chat agent와 batch agent가 같은 store를 통해 동일 cycle context를 조회할 수 있습니다.

`clone-bq-local`은 BigQuery table metadata와 로컬 DuckDB schema spec을 맞춰 `arena_config`, runtime credential metadata, market/account/sleeve/memory/execution/audit 테이블을 로컬 파일로 복제합니다. `--dry-run`은 원본 row/byte 규모만 조회하고, 실제 복제는 `--continue-on-error`와 `--tables`/`--exclude-tables`로 좁혀 실행할 수 있습니다.

### 3.2 Schema & Tables (`schema.py`)

40+ 테이블, 날짜 파티셔닝 + tenant/agent/ticker 클러스터링:

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
| `agent_sleeves` | 가상 sleeve 정의 |
| `agent_state_checkpoints` | Recovery 시드 (canonical) |
| `reconciliation_runs` / `reconciliation_issues` | 감사 추적 + 이슈 로그 |
| `positions_current` | Broker 브리핑용 최신 포지션 materialized view |
| `instrument_master` | 종목 메타데이터 마스터 |
| `market_features_latest` | 최신 행 materialized view |
| `arena_config` | 런타임 설정 (append-only KV) |
| `runtime_credentials` / `runtime_migration_states` / `runtime_user_tenants` / `runtime_access_requests` / `runtime_audit_logs` | Tenant 자격증명, 마이그레이션 상태, 사용자 매핑, 승인 요청, 감사 로그 |
| `tenant_run_statuses` | Tenant별 사이클 실행 상태 |
| `universe_candidates` | 런타임 유니버스 후보 스냅샷 |
| `predicted_expected_returns` | 예측 수익률 (7-모델 앙상블) |
| `research_briefings` | 리서치 브리핑 |
| `dividend_events` | 배당 이벤트 |
| `signal_daily_values` | Layer 1 signal + forward label 재료 (point-in-time) |
| `signal_daily_ic` | 각 signal의 cross-section IC 시계열 |
| `regime_daily_features` | 시장 regime 스냅샷 (vol/trend/dispersion/sentiment) |
| `fundamentals_history_raw` | 분기 발표값 원본, `announcement_date` PIT key, 출처 구분 태그 |
| `fundamentals_derived_daily` | 매일 가격과 결합한 PIT-safe ratio (pe/pb/ep/bp/roe/growth/d2e) |
| `fundamentals_snapshot_latest` | 최신 fundamentals 스냅샷 materialized view |
| `fundamentals_ingest_runs` | KIS/SEC/FMP ingest job metadata (status, tickers_attempted, quarters_inserted) |
| `opportunity_ranker_scores_latest` | signal-IC 합산 점수 (런타임 `recommend_opportunities` 소스) |
| `opportunity_ranker_runs` | ranker 학습 run metadata (per-signal OOS accuracy, predicted_IC) |
| `memory_relation_triples` / `memory_relation_extraction_runs` / `memory_relation_tuning_runs` | Semantic triple 저장소 + LLM 추출 감사 + shadow↔inject 튜닝 이력 |
| `alloc_backtest_runs` / `alloc_backtest_allocations` / `alloc_backtest_nav` | Allocation walk-forward 백테스트 아티팩트 |

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
- allocation source는 UI/admin/chat 모두 최종적으로 `agents_config[].capital_krw`로 정규화. 입력 모드는 고정 KRW(`fixed_krw`), 최신 계좌 평가액의 비율(`account_percent`), 계좌 전체(`whole_account`)를 지원하고, 비율/전체 모드는 저장된 최신 `account_snapshots.total_equity_krw`가 필요
- 독립 NAV, P&L, 포지션 추적
- **Chained Returns**: 자본 이벤트 시 새 베이스라인 생성, 수익률 체인 연결

### 3.5 Market Data (`market_store.py`)

- Feature rows: ticker, as_of_ts, OHLCV, returns, volatility
- 소스: `open_trading_*_quote` (장중), `*_daily` (일봉)
- 중복 제거: daily 스냅샷 우선

---

## 4. Agent System

### 4.1 ADK Agent Architecture (Modular Decomposition)

기존 단일 `adk_agents.py`(~3000L)가 13개의 `adk_*` 모듈로 분리됨:

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
  ├── adk_tool_helpers.py       schema metadata + no-op ledger/search shims (cycle/dev_ui/investment_chat 공유)
  ├── llm_params.py             프로바이더별 샘플링/토큰 파라미터
  │
  └── adk_agents.py             최상위 클래스 + 빌더 (1,680L)
```

기본 프롬프트 텍스트는 `arena/prompts/adk/{core_prompt,system_prompt}.txt`(공통 패키지)로 통합되었고, tenant 오버라이드는 `arena_config.system_prompt`로 관리. `arena/agents/prompts/`는 이전 import 경로를 유지하기 위한 compat shim — 실제 로딩/렌더링은 `arena/prompts/{loader,prompt_pack,memory}.py`가 담당. 메모리 compaction과 semantic relation extractor도 동일 패키지의 `memory/*.txt`를 사용해 시스템 프롬프트를 코드에서 분리했습니다.

ADK tool schema는 `FunctionTool`이 callable signature/docstring/type hint를 읽어 `FunctionDeclaration`을 만드는 경로에 맞춰 관리합니다. `adk_tool_helpers.apply_tool_schema_metadata`가 registry description/label과 원 callable의 signature를 wrapper에 이식하고, exposed signatures는 `Optional[...]`, `Literal[...]`, typed list를 사용해 required field, nullable, enum이 모델에 보이도록 유지합니다. 그래서 batch agent, dev UI, investment chat 모두 같은 schema hygiene을 공유하고, 느슨한 `"JSON string 하나"` 형태의 도구는 LLM-facing 표면에서 피합니다.

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
- 시스템 프롬프트는 `arena/prompts/memory/compaction_system.txt`에서 로드

### 4.6 Investment Chat Agent (`agents/investment_chat/`)

운영자/사용자가 동일 ADK 런타임 위에서 채팅으로 계좌·슬리브를 질의하고, 사람 손으로 한 주문을 검증·체결할 수 있게 하는 별도 ADK Agent. 사이클을 도는 batch agent와 LLM·툴·메모리 인프라는 공유하지만, 독립 인스턴스로 빌드되어 자율 사이클 흐름을 침범하지 않습니다.

```
build_investment_chat_agent(repo, settings, tenant_id, registry, provider, model_override)
  │
  ├── prompts/prompt_pack.PromptPack.render_investment_chat_instruction()
  │     └── arena/prompts/investment_chat/system_prompt.txt
  │
  ├── investment_chat/registry.build_chat_registry()
  │     ├── default_registry.clone() + chat용 _ContextTools 재바인딩
  │     ├── CHAT_ANALYSIS_TOOL_IDS 화이트리스트만 통과
  │     ├── WRITE_TOOL_MARKERS(execute/submit/place_order/broker/sync_account/write_/delete_/upsert_) 자동 차단
  │     └── chat 전용 entry 추가 — account/history/order/config
  │
  ├── adk_runner_bootstrap.build_tool_wrapper(...)
  │     └── adk_tool_helpers.{apply_tool_schema_metadata, noop_update_candidate_ledger}
  │
  └── adk_models._resolve_model(provider, settings, model_override)
```

`investment_chat/system_prompt.txt`는 설정 변경 draft를 "적용됨"으로 말하지 말 것, token 복사를 사용자에게 요구하지 말 것 같은 safety policy만 담습니다. 어떤 값을 채워야 하는지는 tool name, description, typed signature/schema가 담당합니다.

| Tool group | Tools | Purpose |
|------------|-------|---------|
| **Chat account** | `get_account_snapshot`, `refresh_account_snapshot`, `get_agent_sleeve_snapshot` | 저장된 총계좌/슬리브 스냅샷 조회 + KIS 계좌 즉시 새로고침 (테넌트 KIS secret 없으면 차단) |
| **Chat history** | `get_trade_history` | 체결/주문 이력 — `judgment_source` 분리(`user+investment_chat` vs 자율 batch) |
| **Chat order** | `validate_order_draft`, `submit_approved_order` | 2단 승인 주문. validate에서 `RiskEngine` 통과한 draft를 arena_config에 저장(`expires_at`, 기본 15분), UI 승인 버튼이 internal bridge로 `CONFIRM <approval_token>`을 전달해야 submit 가능 |
| **Chat config** | `propose_agent_config_change`, `propose_chat_agent_config_change`, `propose_tenant_config_change`, `get_config_change_status` | 투자 에이전트/채팅 에이전트/tenant runtime 설정 변경 초안. LLM은 draft만 만들고, `/investment-chat/config-drafts/{token}/apply`가 승인 버튼 후 internal bridge를 호출 |
| **Inherited analysis** | recommend_opportunities, optimize_portfolio, forecast_returns, technical_signals, sector_summary, get_fundamentals, index_snapshot, fear_greed_index, earnings_calendar, fetch_reddit_sentiment, fetch_sec_filings, macro_snapshot, search_past_experiences, search_peer_lessons, get_research_briefing, portfolio_diagnosis, trade_performance, screen_market | 사이클 도구를 공유 — 단, write/execute 마커는 위 필터로 제거 |

핵심 안전 장치:

- **Scope contract** — 모든 주문은 `scope ∈ {account, agent_sleeve}`로 명시. agent_sleeve는 `agent_id`로 대상 batch agent를 지정해야 하며, account는 `AGENT_ID="investment_chat"`로 강제 매핑. snapshot 미존재 시 `missing_account_snapshot` 반환.
- **Approval token** — `OrderIntent` fingerprint + nonce를 SHA-256으로 압축한 24자 토큰. 동일 draft를 두 번 submit하면 `already_submitted` idempotent 응답.
- **Config draft contract** — settings 변경 도구는 `scope ∈ {agent, chat_agent, tenant}`를 사용하고 `apply_approved_config_change`는 LLM registry에 노출하지 않습니다. agent scope는 `AdminAgentConfigStore`/`build_single_agent_entry`를 재사용해 provider/model/risk/tools/memory/capital을 기존 admin 검증과 같은 규칙으로 정규화합니다.
- **Capital allocation modes** — chat이 sleeve 금액을 조정할 때 `fixed_krw`, `account_percent`, `whole_account` 중 하나를 제안할 수 있고, apply 전 단계에서 최신 account snapshot 기반 `capital_krw`로 resolve됩니다.
- **Tenant write lock** — `locks.tenant_lock(tenant_id)`(threading RLock)로 같은 테넌트의 chat 주문이 직렬화. `repo_tenant_scope`로 store 단위 tenant 변수도 함께 바인딩.
- **Audit + memory trail** — `chat_order_validate`/`chat_order_submit`/`chat_account_refresh`/`chat_config_change_*`는 `runtime_audit_log`에 기록되고, 체결 성공 시 semantic-tier `MemoryEvent`(`source="investment_chat_order_decision"`, `judgment_source="user+investment_chat"`)가 남아 batch agent의 회상 대상에 포함됩니다.

UI 측 마운트는 §12.2 참조.

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
| `manual_note` | 운영자/내부 수동 메모 | episodic | 0.50 |
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

`memory/vector_factory.py`가 `repo.settings.arena_mode`(또는 `ARENA_MODE`)를 보고 임베딩 백엔드를 선택:

| Mode | Backend | Embedding model | Notes |
|------|---------|----------------|-------|
| `gcp` (default) | `VectorStore` (vector.py) | Vertex AI `text-embedding-004` (768-dim) | Firestore vector NN |
| `local` (with `[local-vector]`) | `LocalChromaVectorStore` (vector_local.py) | sentence-transformers `all-MiniLM-L6-v2` | ChromaDB persistent client, `./data/chroma` (`ARENA_LOCAL_VECTOR_DIR` override) |
| `local` (without extras) | `NullVectorStore` | — | 검색은 빈 결과를 반환하고 본 store는 recency + reranking 보너스로만 동작. ChromaDB 또는 sentence-transformers import 실패 시 자동 폴백 |

리랭킹 보너스/계수는 백엔드와 무관하게 동일하게 적용됩니다.

1. **Semantic Search**: vector nearest-neighbor (top-K)
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

### 5.11 Semantic Relation Graph (`relations.py`, `semantic_extractor.py`)

Causal graph 위에 얹힌 concept-level 의미 관계 레이어. 투자 LLM과 분리된 비동기 파이프라인.

**Write Path:**
- Deterministic extractor: memory/board/research 저장 즉시 structured 필드에서 `mentions`/`contains` triple 추출 (SHA1 dedupe).
- Semantic LLM extractor: 별도 job(`extract-memory-relations`)에서 텍스트 → ontology 제약 triple 후보 생성 → 14단계 validator → accepted만 `memory_relation_triples` + graph projection.

**Read Path (mode별):** `shadow` (저장만) → `boost` (entity 공유 기반 vector 점수 보정) → `inject` (`boost` + relation_context를 프롬프트에 주입).

**Auto-tuning (`semantic_tuning.py`):** Wilson interval 기반 sample/quality/safety/diversity/stability/version 게이트 + cooldown. post-cycle에서 shadow↔inject 자동 전환 및 `memory_relation_tuning_runs`에 감사 기록.

**Ontology (`relation_ontology.py`):** 닫힌 predicate 어휘(`risk_to`, `supports`, `contradicts`, `invalidates`, `similar_setup`, `caused_by`, `leads_to`, `outcome_of`, `mentions`, `contains`) + entity type(`ticker`, `sector`, `risk_factor`, `macro_factor`, `theme`, …) + predicate↔type 조합 제약.

---

## 6. Tools System

### 6.1 Registry (`default_registry.py`)

핵심 도구 등록. `tools_config`로 tenant별 enable/label/description 오버레이, `disabled_tools` 설정으로 개별 비활성화 가능.

ToolEntry의 callable은 ADK wrapper를 통과하기 전에 schema metadata를 보강합니다. Registry description은 wrapper docstring으로, 원 callable의 `inspect.signature`와 type hints는 ADK `FunctionTool`이 읽을 수 있는 형태로 유지합니다. Optional scalar/list 파라미터는 `Optional[...]`, 선택지는 `Literal[...]` alias로 표현해 required/default/enum이 모델-visible schema에 남습니다.

### 6.2 Context Tools (5개) — `adk_context_tools.py`

| Tool | Function |
|------|----------|
| `search_past_experiences` | Firestore 벡터 검색 (과거 기억) |
| `search_peer_lessons` | 다른 에이전트의 교훈 검색 |
| `get_research_briefing` | BQ 리서치 브리핑 조회 |
| `portfolio_diagnosis` | 집중도/팩터/스트레스 진단 + HRP 리밸런싱 계획 |
| `trade_performance` | 라운드트립 승률/평균수익률/보유기간/행동 패턴 + 현재 미실현 P&L |

### 6.3 Quant Tools (7개) — `quant_tools.py`

| Tool | Function |
|------|----------|
| `recommend_opportunities` | signal-IC meta-learner 기반 추천. prep 단계에서 signal-IC 학습 → `opportunity_ranker_scores_latest`에 predict_IC × signal 합산 점수 저장, runtime은 읽기만. Tactical ETP 프로필 자동 분리 |
| `screen_market` | 저수준 진단용 bucket screen. 기본 비활성, `recommend_opportunities` 내부에서 사용 |
| `optimize_portfolio` | Max-Sharpe, HRP, forecast-enhanced + graceful degrade |
| `forecast_returns` | 7-model ensemble → prob_up + 컨센서스 |
| `technical_signals` | RSI, MACD, Bollinger, SMA + 거래량 분석 + KOSPI 수급 신호 |
| `sector_summary` | GICS sector returns/volatility |
| `get_fundamentals` | US (P/E, P/B, EPS, BPS), KR (EPS, BPS, ROE, 부채비율, 성장성) |

### 6.4 Macro (4개) — `macro_tools.py`, `sentiment_tools.py`

| Tool | Function |
|------|----------|
| `index_snapshot` | US + KOSPI 주요 지수/원자재/채권 수익률 (마켓별 자동 라우팅) |
| `macro_snapshot` | US→FRED / KR→ECOS 한국은행 |
| `fear_greed_index` | VIX/VKOSPI + breadth + momentum + 수급 복합 지표 |
| `earnings_calendar` | US Nasdaq 실적 / KR KIS 배당 + 컨센서스 |

### 6.5 Sentiment (2개) — `sentiment_tools.py`

| Tool | Function |
|------|----------|
| `fetch_reddit_sentiment` | Reddit 금융 서브레딧 감성 |
| `fetch_sec_filings` | SEC EDGAR 공시 |

### 6.6 Allocation Strategies (`allocation.py`)

| Strategy | Description |
|----------|-------------|
| `optimize_max_sharpe` | Sharpe 최대화 |
| `optimize_min_vol` | Minimum volatility |
| `optimize_hrp` | Hierarchical Risk Parity |
| `optimize_blend` | 60/40 Max-Sharpe + HRP |
| `optimize_forecast_sharpe` | Historical mu + forecast 블렌드 |

### 6.7 Opportunity Ranker (`arena/recommendation/`)

`recommend_opportunities` 도구의 백엔드. Prep 단계에서만 학습/스코어링하고 runtime은 읽기 전용.

- `signals.py` — Layer 1 signal 정의(momentum, pullback, 평균회귀, 저변동성, forecast, RSI/MA/볼린저, EP/BP/SP/ROE/growth/debt) + `REGIME_FEATURES`
- `ranker.py` — Signal별 IC를 regime feature로 조건화하여 학습. Runtime score `= Σ predicted_IC_i(today_regime) × signal_i(ticker)`. 결과를 `opportunity_ranker_scores_latest`에 저장 + per-signal contribution, `opportunity_ranker_runs`에 학습 감사 로그. Tactical ETP(인버스/레버리지/헤지)는 별도 profile로 분리.

### 6.8 Investment Chat Tools (`agents/investment_chat/`)

위 도구들은 사이클 batch agent와 chat agent가 공유합니다. chat에서만 노출되는 도구는 `agents/investment_chat/`이 자체 entry로 구성:

| Category | Tool | Purpose |
|----------|------|---------|
| Account | `get_account_snapshot` / `refresh_account_snapshot` / `get_agent_sleeve_snapshot` | 저장된 총계좌·슬리브 스냅샷 + 테넌트 KIS 자격증명이 있을 때만 동기화 |
| History | `get_trade_history` | 정확한 체결/주문 이력. `judgment_source`가 `user+investment_chat`인지 자율 batch인지 분리 |
| Order | `validate_order_draft` / `submit_approved_order` | 2단 승인 — validate에서 risk 통과한 draft를 arena_config에 TTL 저장, UI order panel이 internal bridge로 `CONFIRM <token>`을 전달할 때만 submit |
| Config | `propose_agent_config_change` / `propose_chat_agent_config_change` / `propose_tenant_config_change` / `get_config_change_status` | 설정 변경 초안 — provider/model/tool/memory/risk/prompt/sleeve allocation을 draft로 저장. UI config panel 승인 전까지 `arena_config` 실제 값은 변경하지 않음 |

도구 어댑팅은 batch agent와 동일하게 `adk_runner_bootstrap.build_tool_wrapper`를 통과하지만, candidate ledger와 ReAct memory injection은 `adk_tool_helpers.noop_*`로 비활성화 (사이클 메트릭에 영향 없도록).

---

## 7. Context Builder

`arena/context.py` (2,758L) — 에이전트별, 사이클별 컨텍스트 조립.

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

### 8.3 Chat-Driven Order Flow

투자챗봇이 내는 주문도 동일한 `ExecutionGateway` + `RiskEngine` + broker 조합을 거치며, 차이는 LLM이 직접 게이트웨이를 호출하지 않고 사용자 confirmation을 사이에 끼운다는 점뿐:

```
chat agent
  │
  ▼
validate_order_draft(scope, agent_id, ticker, side, qty, price_krw, ...)
  ├── snapshot_for_order_scope() → account 또는 agent_sleeve snapshot
  ├── RiskEngine.evaluate(intent, snapshot, recent_metrics)
  ├── approval_token = sha256(fingerprint + nonce)[:24]
  ├── arena_config[draft_key] = { intent, risk, expires_at, approved_by }
  └── return { approval_token, required_confirmation: "CONFIRM <token>" }
        │
        ▼ UI order approval button
        │
submit_approved_order(approval_token, confirmation_text)
  ├── load_draft + 만료/중복 submit 가드
  ├── confirmation_text == "CONFIRM <token>" 확인 (UI bridge가 채움)
  ├── tenant_lock(tenant) + repo_tenant_scope(repo, tenant)
  ├── ExecutionGateway.process(intent, snapshot)         ← batch와 동일 경로
  ├── 성공 시 MemoryStore.record_reflection(semantic, judgment_source="user+investment_chat")
  ├── append_runtime_audit_log(action="chat_order_submit")
  └── return execution_report + memory_warnings
```

검증 단계에서 risk가 떨어지면 draft는 `risk_rejected`로 남고 submit은 거부됩니다. 사이클 batch agent와 chat이 같은 `ExecutionGateway`를 공유하므로 broker, ledger, sleeve replay, NAV 계산은 별도 코드 경로 없이 자동으로 일관 유지.

### 8.4 Chat-Driven Config Flow

투자챗봇의 설정 변경도 주문과 같은 draft/approval 패턴을 사용합니다. LLM은 직접 `INSERT`/`UPDATE` SQL을 만들지 않고, schema가 있는 관리 도구로 변경 의도를 구조화합니다.

```
chat agent
  │
  ├── propose_agent_config_change(agent_id, provider, model, capital_allocation_mode, ...)
  ├── propose_chat_agent_config_change(provider, model, disabled_tools, llm_params_json, ...)
  └── propose_tenant_config_change(system_prompt, risk_policy_json, memory_policy_json, ...)
        │
        ├── validate fields + 기존 admin normalizer 재사용
        ├── account_percent/whole_account이면 latest_account_snapshot 필요
        ├── approval_token + diffs + summary 생성
        └── arena_config[chat_config_draft.<token>] = draft
              │
              ▼ UI config approval button
              │
apply_approved_config_change(approval_token, confirmation_text)
  ├── confirmation_text == "CONFIRM <token>" 확인 (UI bridge가 채움)
  ├── tenant_lock(tenant)
  ├── agents_config 또는 tenant config key append
  ├── runtime/admin cache invalidate
  └── append_runtime_audit_log(action="chat_config_change_apply")
```

노출되는 LLM 도구는 propose/status 계열뿐이고, `apply_approved_config_change`는 `include_internal_bridge=True`로 만든 backend/UI bridge entry에서만 사용합니다. 이 설계 덕분에 설정 변경 권한은 채팅 UX 안에 들어오지만, 적용 권한은 UI의 명시적 사람 액션과 기존 admin validation 경로에 남습니다.

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

`arena/reconciliation.py` (1,423L)

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

### 11.1 OpenTradingClient (`client.py`, 2,249L)

- OAuth 토큰 관리 — `token_cache.py`(Firestore) 또는 `token_cache_file.py`(local atomic JSON, `KIS_TOKEN_CACHE_BACKEND=file`로 강제)
- US + KR 시장 데이터 (일봉, 호가, 지수)
- 계좌 조회 (잔고, 포지션)
- 주문 (해외/국내)
- 배당 (period_rights, KSD)
- Fundamentals (재무비율, EPS/ROE)

### 11.2 Sync Services (`sync.py`, 3,375L)

| Service | Function |
|---------|----------|
| `MarketDataSyncService` | US/KOSPI 일봉 + 실시간 호가 |
| `AccountSyncService` | 계좌 스냅샷 (US+KR 병합) |
| `BrokerTradeSyncService` | 체결 거래 이력 동기화 |
| `BrokerCashSyncService` | 현금 흐름 (수수료, 배당, 이자) |
| `DividendSyncService` | US period_rights + KR KSD 배당 |

### 11.3 Fundamentals Ingestors

`fundamentals_history_raw` / `fundamentals_derived_daily` 백필 전용 커맨드 파이프라인:

| File | Source |
|------|--------|
| `kis_fundamentals_ingestor.py` | KIS 국내 재무비율 API |
| `sec_fundamentals_ingestor.py` | SEC EDGAR companyfacts (keyless, US 기본) |
| `fmp_fundamentals_ingestor.py` | Financial Modeling Prep (옵션) |

---

## 12. UI Layer

FastAPI Admin Dashboard — 모듈화된 라우트 구조.

### 12.1 Route Structure

| Module | Path | Purpose |
|--------|------|---------|
| `auth.py` | `/login`, `/callback`, `/logout` | Google OAuth |
| `board.py` | `/`, `/board` | 대시보드 + 에이전트 간 게시물 |
| `showcase.py` | `/showcase` | 공개 쇼케이스 페이지 |
| `nav.py` | `/nav` | NAV 차트 |
| `trades.py` | `/trades` | 실행 이력 |
| `sleeves.py` | `/sleeves` | 슬리브 관리 |
| `ops.py` | `/ops` | 운영 상태 |
| `investment_chat.py` | `/investment-chat` | 투자챗봇 페이지 (provider/model 선택 + iframe shell) |
| `settings_page.py` | `/settings` | 설정 페이지 |
| `settings_admin.py` | `/admin/*` | 설정 CRUD (에이전트, 리스크, 도구, MCP, 메모리) |
| `settings_render_agents.py` | — | 에이전트 설정 패널 |
| `settings_render_capital.py` | — | 자본 관리 패널 |
| `settings_render_credentials.py` | — | KIS/API 키 관리 |
| `settings_render_scripts.py` | — | 스크립트/프롬프트 편집 |
| `capital_data.py` | `/api/capital/*` | 자본 데이터 API |
| `viewer.py` | `/viewer/*` | 데이터 뷰어 |

사이드바 nav는 `arena/ui/layout.py:tailwind_layout`이 조립합니다. `tenant`가 있으면 `?tenant_id=...`를 자동으로 붙이고, `hide_page_header`/`main_class` 옵션으로 챗봇처럼 헤더 없이 풀폭 iframe을 그릴 수 있습니다. `base_layout.jinja2`에는 데스크톱 전용 사이드바 collapse 토글이 추가되어 있어 `body.sidebar-collapsed` 토글만으로 너비를 `--sidebar-collapsed-w`로 줄입니다.

### 12.2 Investment Chat Mount (`ui/investment_chat_adk.py`)

`/investment-chat`(셸 페이지) + `/investment-chat/adk`(ADK FastAPI dev-ui sub-app)의 2단 구성. 기존 admin UI에 ADK가 제공하는 풀 챗 UX를 끼워 넣되, 인증·테넌트·모델 선택은 admin 측 컨벤션을 그대로 따릅니다.

```
/investment-chat (HTML shell)
  ├── form: provider/model select → query string/세션/`investment_chat_config` 기준
  ├── order approval panel  → /investment-chat/order-drafts/{token}/submit
  ├── config approval panel → /investment-chat/config-drafts/{token}/apply
  └── iframe → /investment-chat/adk/dev-ui/
                       └── google.adk.cli.fast_api.get_fast_api_app(
                              agent_loader=InvestmentChatAgentLoader,
                              session_service_uri=sqlite:///data/arena-investment-chat-adk-sessions.sqlite,
                              artifact_service_uri=memory://,
                              web=False, url_prefix=/investment-chat/adk,
                              auto_create_session=False)
```

| 구성요소 | 역할 |
|---------|------|
| `InvestmentChatAgentLoader` | tenant + provider + model 조합을 base64로 인코딩한 `app_name`으로 ADK에 노출. 같은 조합은 LRU(64)로 캐시. |
| `_install_auth_gate` middleware | 세션/쿼리/env(`ARENA_CHAT_*`)에서 tenant·provider·model을 확정해 `REQUEST_*` ContextVar에 바인딩. `auth_enabled=true`면 미로그인 HTML 요청은 `/auth/google/login`으로 redirect, API 요청은 `401 auth required`. |
| `_mount_adk_static` | ADK 패키지의 `browser/` 정적 자산을 임시 디렉터리로 복사하고, `assets/config/runtime-config.json`의 `backendUrl`을 마운트 prefix로 patch (ADK 버전 + prefix 해시로 캐시). |
| Session/Artifact store | sqlite 파일은 `arena/data/arena-investment-chat-adk-sessions.sqlite`(env로 override 가능), artifact는 in-memory 기본. |
| Provider/Model 선택 | `_CHAT_MODEL_PRESETS`에서 provider별 후보를 노출. 우선순위는 query string → `arena_config.investment_chat_config` → 세션 → `default_model_for_provider(settings, provider)`. |
| Approval panels | `investment_chat_body.jinja2`가 order/config drafts를 2.5초 간격으로 polling. 버튼 클릭 후 backend bridge 결과를 iframe의 ADK chat input에 다시 전달해 모델이 사용자에게 결과를 설명하게 함. |

### 12.3 Memory 3D Graph (`memory.py`, 968L)

10개 그룹 → Branch → Leaf Field 구조. Click node → edit value → save to arena_config. Three.js + 3d-force-graph는 `arena/ui/vendor/`에 번들.

### 12.4 Templates

24개 Jinja2 템플릿: base/board/nav/trades/sleeves/ops/settings + memory panel + 인증 알림 + 자격증명 카드 + `investment_chat_body.jinja2`(provider/model select 폼 + ADK iframe + order/config approval panels).

---

## 13. Market Hours & Scheduling

`arena/market_hours.py` (357L)

| Market | Session | Timezone |
|--------|---------|----------|
| NASDAQ/NYSE | 09:30-16:00 | America/New_York |
| KOSPI/KOSDAQ | 09:00-15:30 | Asia/Seoul |

### Holiday Detection
- **US**: Static 계산 (MLK, Presidents' Day, Easter, etc.)
- **KOSPI**: `korean_lunar_calendar` (설날, 추석, 부처님 오신 날) + 고정 9개 + 대체휴일

---

## 14. Configuration & Runtime Overrides

`arena/config.py` (1,137L)

### Precedence Contract

```
1. load_settings()        .env / 환경변수 → 기본 Settings
2. _build_runtime()       tenant_id 결정 + active Repository(BigQuery/DuckDB)
3. _apply_tenant_runtime_credentials()  Secret Manager 메타 + 자격증명
4. apply_runtime_overrides()            arena_config tenant별 값 → Settings 오버레이
5. apply_distribution_mode()            safety gate
```

### Runtime Source of Truth

| Layer | Storage | Scope | Used For |
|------|---------|-------|----------|
| Boot defaults | `.env` / env vars | process | 기본 Settings 부트스트랩 (`ARENA_MODE` 포함) |
| Secret metadata | `runtime_credentials` (BQ) / 동일 컬럼을 가진 DuckDB 테이블 | tenant | KIS/모델 secret, key availability |
| Secret payload | Secret Manager (gcp) / `~/.llm-arena/credentials.json` mode 0600 (local) | tenant | KIS 계좌 + 모델 API 키 페이로드. `kis_secret_name`이 `local-` 접두면 `arena/security/credential_store_env.load_local_secret_payload`로 동일 JSON 파일을 직접 읽어 GCP Secret Manager 호출을 우회 (cli_runtime, OpenTradingClient 공통). 이 경로는 `_apply_tenant_runtime_credentials`/`_prepare_kis_command_repo`에서 KIS 필드를 명시 클리어한 뒤 적용해, 다른 테넌트의 fallback 자격증명이 새지 않도록 강제합니다. |
| Editable config | `arena_config` | tenant | 프롬프트, 리스크, 에이전트, 메모리 정책 |
| Safety gate | distribution_mode, real_trading_approved | process + tenant | 실거래 허용 여부 |

### Backend Mode (`arena_mode`)

`config.py:load_settings()`는 `ARENA_MODE` env를 정규화해 `Settings.arena_mode in {"gcp","local"}`(unknown은 `gcp`)로 저장. 다음 컴포넌트가 동일 키를 읽어 GCP/로컬 분기:

| Consumer | File | Behaviour |
|----------|------|-----------|
| Repository | `arena/data/factory.py` | `BigQueryRepository` ↔ `LocalRepository` |
| Vector store | `arena/memory/vector_factory.py` | Vertex+Firestore ↔ ChromaDB / NullVectorStore |
| Credential store | UI runtime + admin commands | Secret Manager `CredentialStore` ↔ `EnvCredentialStore` |
| Tenant lease | orchestrator pre-cycle | `FirestoreTenantLeaseStore` ↔ `LocalTenantLeaseStore` |
| KIS token cache | `OpenTradingClient` | Firestore `token_cache.py` ↔ `token_cache_file.py` |

로컬 모드 추가 env (모두 default 있음, 변경 시 단일 머신 워크플로용):
- `ARENA_LOCAL_DB_PATH` — DuckDB 파일 경로 (default `./data/arena.duckdb`)
- `ARENA_LOCAL_VECTOR_DIR` — ChromaDB persist 디렉터리 (default `./data/chroma`)
- `ARENA_LOCAL_CREDENTIALS_FILE` — 자격증명 JSON (default `~/.llm-arena/credentials.json`)
- `ARENA_LOCAL_LEASE_FILE` — tenant lease JSON (default `./data/tenant_leases.json`)
- `KIS_TOKEN_CACHE_BACKEND=file` — Firestore 미사용 강제
- `KIS_TOKEN_CACHE_FILE` — KIS 토큰 캐시 경로 (default `~/.llm-arena/tokens.json`)

### Key Config Keys (`arena_config`)

| Key | Type | Description |
|-----|------|-------------|
| `system_prompt` | text | 에이전트 시스템 프롬프트 |
| `agents_config` | JSON | 에이전트 CRUD (provider/model/capital/risk/tools/memory). chat 승인 flow도 최종적으로 이 key를 append |
| `investment_chat_config` | JSON | 채팅 에이전트 provider/model/disabled_tools/llm_params/memory_compaction_model |
| `investment_chat_account_markets` | scalar | 채팅 계좌/슬리브 조회 허용 market scope |
| `risk_policy` | JSON | Risk 파라미터 |
| `sleeve_capital_krw` | scalar | 기본 sleeve 자본 |
| `disabled_tools` | JSON | 비활성화 도구 |
| `mcp_servers` | JSON | MCP 서버 등록 |
| `memory_policy` | JSON | 메모리 정책 (10 groups) |
| `memory_compactor_prompt` | text | 메모리 compaction helper prompt override |
| `forecast_mode` | scalar | 예측 사용 모드 |
| `kis_target_market` | scalar | tenant별 타깃 시장 |
| `research_enabled` / `research_*` | bool/scalar | 리서치 에이전트 enable 및 ticker/mover/earnings limits |
| `real_trading_approved` | bool | 실거래 승인 스위치 |

`chat_config_draft.<approval_token>`과 order draft keys도 같은 `arena_config` append-only KV에 저장됩니다. draft key는 TTL/status를 가진 임시 승인 객체이고, 승인 후 실제 runtime key(`agents_config`, `investment_chat_config`, `risk_policy`, ...)가 별도 row로 append됩니다.

### tenant / trading_mode Contract

- `tenant_id`는 운영 단위. 설정, 자격증명, UI 접근, BQ 조회 기준.
- `trading_mode`는 실행 레인. paper/live는 별도 데이터 스트림.
- `memory_policy`를 포함한 arena_config는 tenant 단위.

---

## 15. CLI Interface

Entry point: `arena/cli.py` (457L) → `arena/cli_commands/` (12개 실행 모듈)

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

# Forecasting & Ranker
llm-arena build-forecasts                   # 7-model ensemble
llm-arena build-opportunity-ranker          # signal-IC meta-learner + latest scores
llm-arena refresh-signals                   # signal_daily_values만 재계산 (debug)
llm-arena refresh-signal-ic                 # signal_daily_ic만 재계산 (debug)
llm-arena refresh-regime-features           # regime_daily_features만 재계산 (debug)

# Fundamentals
llm-arena fundamentals-backfill-kr          # KIS 기반 KR 재무 백필
llm-arena fundamentals-backfill-us --source sec    # SEC EDGAR 기반 US 재무 백필 (keyless)
llm-arena refresh-fundamentals-derived      # fundamentals_derived_daily 재계산
llm-arena fundamentals-coverage             # 커버리지 리포트

# Trading
llm-arena run-pipeline --live --market us         # Full: sync → forecast → ranker → agents
llm-arena run-shared-prep --live --market us      # Shared sync/forecast/ranker prep only
llm-arena run-agent-cycle --live --all-tenants    # Agent cycle only
llm-arena run-batch --live --all-tenants          # Manual sync + cycle shortcut

# Reconciliation
llm-arena recover-sleeves                         # 자동 복구 + checkpoint 재생성

# Memory
llm-arena enable-memory-forgetting                # tenant forgetting + shadow tuning 활성화
llm-arena run-memory-forgetting-tuner             # forgetting 튜너 스케줄용 실행
llm-arena extract-memory-relations --tenant <id>  # semantic triple 추출

# Admin
llm-arena serve-ui                                # FastAPI UI (port 8080)
llm-arena list-strategies                         # Strategy reference cards
llm-arena serve-strategy-mcp                      # MCP server
llm-arena promote-tenant-live --tenant <id>      # private + live approval
llm-arena approve-live-tenant --tenant <id>      # 실거래 승인/회수
llm-arena set-tenant-simulated --tenant <id>     # simulated-only onboarding
llm-arena backfill-tenant-markets --tenant <id>  # kis_target_market 백필
```

### CLI Command Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `run_agent.py` | 704 | agent cycle only |
| `run_pipeline.py` | 1,496 | sync → forecast → ranker → agent |
| `run_shared.py` | 534 | shared sync/forecast/ranker |
| `sync.py` | 647 | market/account/fundamentals/ranker sync |
| `admin.py` | 336 | tenant/memory admin |
| `run_reconcile.py` | 330 | reconciliation + post-cycle maintenance |
| `serve.py` | 267 | UI + MCP server |
| `memory_relations.py` | 148 | semantic triple 추출 |
| `run.py` | 48 | dispatch routing |

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

87개 테스트 파일, pytest. 100% pass 필수.

### Key Test Files

| Test File | Coverage |
|-----------|----------|
| `test_adk_agents.py` | ADK normalization, model routing, tool loading, schema metadata preservation |
| `test_agents_config.py` | Per-agent config CRUD, capital allocation |
| `test_context.py` | Context builder, memory reranking, cash buffer |
| `test_data_strict_paths.py` | BQ queries, dedup, checkpoint rebuild |
| `test_memory_store.py` | Scoring, dedup, tier assignment, tagging |
| `test_memory_forgetting.py` | Decay math, effective_score, access curves |
| `test_memory_graph.py` | Node/edge builders, causal chain inference |
| `test_memory_tuning.py` | Tuner grid search, objective, auto-promote/demote |
| `test_memory_bq_store.py` | Memory BQ store operations |
| `test_memory_relations.py` | Deterministic triple extraction |
| `test_semantic_relation_extractor.py` | LLM triple extraction + validation |
| `test_semantic_relation_tuning.py` | Shadow↔inject quality gates |
| `test_cli_memory_relations.py` | Semantic triple CLI |
| `test_cli_thesis_compaction_smoke.py` | Thesis chain compaction smoke |
| `test_clone_bq_local.py` / `test_init_local_cli.py` / `test_duckdb_schema.py` | Local DuckDB bootstrap/clone/schema |
| `test_execution_reconcile.py` | Order reconciliation vs. broker |
| `test_forecasting_stacked.py` | 7-model stacking |
| `test_opportunity_ranker.py` | Signal-IC meta-learner 학습/스코어링 |
| `test_signals.py` | Layer 1 signal definitions |
| `test_new_tools.py` | 신규 도구 (`recommend_opportunities`, `trade_performance`) |
| `test_investment_chat_ui.py` | Chat ADK shell, order/config approval bridges, tool schema quality |
| `test_kis_fundamentals_ingestor.py` / `test_sec_fundamentals_ingestor.py` / `test_fmp_fundamentals_ingestor.py` | Fundamentals ingestors |
| `tests/ui/test_*_routes.py` | Settings/admin agent/memory/board/chart/auth UI routes |
| `test_ui_helper_modules.py` | UI helper modules |
| `test_tenant_leases.py` | Firestore execution lease |
| `test_provider_registry.py` | Provider module |
| `test_market_sources.py` | Market source resolution |
| `test_tool_registry.py` | Tool registry overlay + enable/disable |

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
- 13개 `adk_*` 파일로 ADK 에이전트 책임 분리 + base/llm_params + research/compaction/support 3개 파일
- chat 전용 빌더(`agents/investment_chat/`)는 동일 ADK 빌딩 블록(`adk_runner_bootstrap`, `adk_models`, `adk_tool_helpers`, `_ContextTools`)을 재사용해 batch 사이클과 코드 경로를 공유
- 테스트 용이성 + 단일 책임 원칙

### Approval Drafts
- LLM-facing mutating tools는 주문/설정 모두 draft만 생성
- 실제 submit/apply는 UI/backend bridge가 `CONFIRM <token>`을 내부 전달
- draft와 audit은 `arena_config` + `runtime_audit_logs`에 append-only로 남기고, 적용은 기존 gateway/admin normalizer를 재사용

### Virtual Sleeving
- 단일 브로커 계좌 → N개 가상 sleeve
- 고정 금액, 계좌 비율, 계좌 전체 위임 입력을 `capital_krw`로 정규화
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
 5. Forecast + Ranker (Shared Prep)
    │  build_and_store_stacked_forecasts()     ← 7-model ensemble → predicted_expected_returns
    │  refresh signal_daily_values / signal_daily_ic / regime_daily_features
    │  build_and_store_opportunity_ranker()    ← signal-IC meta-learner → opportunity_ranker_scores_latest
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
    │  semantic relation tuner (shadow↔inject 게이트 평가)
    │  forgetting tuner (shadow/bounded_ema)
    │
10. NAV Snapshot (agent_nav_daily + official_nav_daily)
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
6. **Data layer split** — 기존 `bq.py`(1200L+) → `bq.py`(144L facade) + `bigquery/` 디렉토리(9개 store + session).
7. **Agent decomposition** — 기존 `adk_agents.py`(~3000L) → 13개 `adk_*` 파일 + helper/research/compaction 3개.
8. **CLI modularization** — 기존 `cli.py`(~2700L) → `cli.py`(457L) + `cli_commands/`(12개 실행 모듈).
9. **PnL backfeed** — SELL 시 과거 BUY 기억의 outcome_score를 실제 수익률로 역업데이트.
10. **Risk per-agent** — `Settings.risk_policy` + `AgentConfig.risk_overrides` 병합.
11. **ADK tools are async** — LLM 호출은 ReAct 루프에서 블로킹. 동시 에이전트는 ThreadPoolExecutor.
12. **Tenant isolation** — 모든 쿼리 `tenant_id` 필터.
13. **Market hours** — 스케줄링 전 holiday 체크 필수.
14. **Investment chat scope** — chat의 모든 주문은 `judgment_source="user+investment_chat"`로 기록되며 `OrderIntent.agent_id`는 scope에 따라 `investment_chat`(account) 또는 대상 batch agent(`agent_sleeve`)로 강제. batch agent가 자율 판단한 것처럼 보이지 않게 하기 위해 strategy_refs에 `source:investment_chat` + `judgment:user+investment_chat`이 함께 박힙니다.
15. **Investment chat config approval** — 설정 변경 도구는 `propose_*`와 status 조회만 LLM에 노출합니다. `apply_approved_config_change`는 UI bridge 전용이며, 직접 SQL 대신 admin config normalizer와 runtime ops를 사용합니다.
16. **ADK schema shape matters** — ADK는 callable signature/type hints를 schema로 바꿉니다. LLM-facing 도구에 자유형 JSON string을 하나만 주면 모델이 필드 의미를 못 보므로, enum/required가 필요한 인자는 `Literal`/`Optional`/typed list로 드러내야 합니다.
17. **Prompt 패키지 위치** — `arena/agents/prompts/`는 compat shim. 신규 프롬프트는 `arena/prompts/{adk,investment_chat,memory}/*.txt`에 두고, `PromptPack`(또는 `arena.prompts.memory`의 헬퍼)를 통해 로드해야 lru_cache + tenant override 경로가 일관되게 적용됩니다.

---

## 21. Quick Reference

| Layer | Key Files | Lines |
|-------|-----------|-------|
| **Models** | `arena/models.py` | 156 |
| **Config** | `arena/config.py` | 1,196 |
| **Data** | `arena/data/bq.py` + `bigquery/` (9 stores + session) | 144 + 9,158 |
| **Schema** | `arena/data/schema.py` | 1,106 |
| **Agents** | `arena/agents/` (13 adk_* + base + llm_params + support/research/compaction) | 7,619 |
| **Investment Chat** | `arena/agents/investment_chat/` (17 files) + `arena/ui/investment_chat_adk.py` + route + template | 2,753 + 792 + 366 + 433 |
| **Prompts** | `arena/prompts/` (loader/prompt_pack/memory + adk/investment_chat/memory text) | 413 + text |
| **Memory** | `arena/memory/` (18 files) | 9,634 |
| **Tools** | `arena/tools/` (11 files) | 6,213 |
| **Recommendation** | `arena/recommendation/` (ranker + signals) | 918 |
| **Context** | `arena/context.py` | 2,758 |
| **Orchestration** | `arena/orchestrator.py` | 571 |
| **Execution** | `arena/execution/gateway.py` + `risk.py` | 477 + 125 |
| **Broker** | `arena/broker/` (3 files) | 677 |
| **Open Trading** | `arena/open_trading/` (8 files) | 7,113 |
| **Reconciliation** | `arena/reconciliation.py` | 1,423 |
| **CLI** | `arena/cli.py` + `cli_commands/` (12 execution modules) | 457 + 5,628 |
| **UI** | `arena/ui/` (routes + core) | ~11,000 |
| **Providers** | `arena/providers/` (3 files) | 408 |
| **Forecasting** | `arena/forecasting/stacked.py` | 800 |
| **Tests** | `tests/` (67 files) | — |
