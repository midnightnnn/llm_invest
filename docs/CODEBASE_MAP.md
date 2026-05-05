# LLM Arena Codebase Map

> 이 문서는 상세 설계서가 아니라 llm이 작업을 시작할 때 빠르게 방향을 잡기 위한 지도다.
> 최종 업데이트: 2026-05-05

## 읽는 순서

1. 이 파일로 작업 영역과 진입점을 잡는다.
2. 상세 설계, 운영 규칙, 데이터 흐름은 [`ARCHITECTURE.md`](ARCHITECTURE.md)를 본다.
3. 설치, 로컬 실행, 주요 CLI는 [`README.md`](../README.md)를 본다.
4. 메모리 정책을 건드릴 때는 [`memory_system.md`](memory_system.md)를 같이 본다.
5. fundamentals 백필은 [`fundamentals_backfill_runbook.md`](fundamentals_backfill_runbook.md)를 따른다.

이 파일의 내용보다 현재 코드와 테스트가 우선이다. 흐름이나 진입점이 바뀌면 이 파일도 같이 갱신한다.

## 작업별 진입점

| 작업 | 먼저 열 파일 |
| --- | --- |
| CLI 명령 추가/수정 | `arena/cli.py`, `arena/cli_runtime.py`, `arena/cli_commands/run.py`, 관련 `arena/cli_commands/*.py` |
| 에이전트 사이클 | `arena/cli_commands/run_agent.py`, `arena/orchestrator.py`, `arena/agents/adk_agents.py` |
| 전체 파이프라인/공유 준비 | `arena/cli_commands/run_pipeline.py`, `arena/cli_commands/run_shared.py`, `arena/cli_commands/sync.py` |
| Investment Chat UI | `arena/ui/routes/investment_chat.py`, `arena/ui/investment_chat_adk.py`, `arena/ui/templates/investment_chat_body.jinja2` |
| Investment Chat 도구 | `arena/agents/investment_chat/factory.py`, `arena/agents/investment_chat/registry.py`, `arena/agents/investment_chat/*_tools.py` |
| 주문 승인 플로우 | `arena/agents/investment_chat/order_tools.py`, `arena/agents/investment_chat/drafts.py`, `arena/ui/routes/investment_chat.py`, `arena/execution/gateway.py` |
| 설정 변경 승인 플로우 | `arena/agents/investment_chat/config_tools.py`, `arena/agents/investment_chat/drafts.py`, `arena/ui/admin_agent_config.py`, `arena/ui/routes/settings_admin.py` |
| 에이전트/런타임 설정 | `arena/config.py`, `arena/ui/admin_agent_config.py`, `arena/ui/admin_runtime_ops.py`, `arena/data/*/runtime_store.py` |
| 저장소 백엔드 | `arena/data/factory.py`, `arena/data/bq.py`, `arena/data/bigquery/*`, `arena/data/local/*` |
| 메모리 시스템 | `arena/memory/*`, `arena/agents/memory_compaction_agent.py`, `arena/cli_commands/run_memory_relations.py` |
| 퀀트/매크로/센티먼트 도구 | `arena/tools/default_registry.py`, `arena/tools/quant_tools.py`, `arena/tools/macro_tools.py`, `arena/tools/sentiment_tools.py` |
| ADK tool schema 품질 | `arena/agents/adk_tool_helpers.py`, `arena/agents/adk_context_tools.py`, `arena/tools/registry.py`, 각 tool docstring |
| Admin/UI 라우트 | `arena/ui/app.py`, `arena/ui/routes/*`, `arena/ui/templates/*` |
| KIS/브로커/실거래 | `arena/open_trading/*`, `arena/broker/*`, `arena/execution/gateway.py`, `arena/risk.py` |

## 핵심 흐름

### 런타임 설정

`load_settings()`가 env 기본값을 읽고, CLI/UI에서는 repository를 만든 뒤 tenant runtime config를 다시 주입한다.

흐름:

```text
load_settings()
-> get_repository() / _repo_or_exit()
-> _apply_tenant_runtime_credentials()
-> apply_runtime_overrides()
-> apply_distribution_mode()
```

`arena_config`는 append-only에 가깝게 쓰이며 최신 row가 우선한다. UI와 chat 설정 변경은 직접 SQL을 노출하지 말고 admin/helper 경로를 통해 저장해야 감사와 캐시 무효화가 맞는다.

### 에이전트 사이클

`run-agent-cycle`은 런타임을 만들고, agent 설정을 hydrate한 뒤 orchestrator에 넘긴다.

```text
python -m arena.cli run-agent-cycle
-> arena/cli_commands/run_agent.py
-> build_adk_agents()
-> ArenaOrchestrator.run_cycle()
-> draft round
-> execution round
-> ExecutionGateway.process()
-> NAV/memory/post-cycle maintenance
```

주요 분기점은 tenant, market, trading mode, distribution mode, memory policy, disabled tools다.

### Investment Chat 주문 승인

Chat LLM은 주문을 바로 실행하지 않는다. 주문 도구는 draft를 만들고, UI 액션 버튼이 사용자의 승인을 받은 뒤 bridge가 내부 confirmation token을 넣어 실행한다.

```text
/investment-chat
-> ADK iframe
-> validate_order_draft()
-> /investment-chat/order-drafts
-> submit_approved_order()
-> ExecutionGateway
```

이 구조 때문에 프롬프트에 "텍스트로 CONFIRM을 입력하라"는 UX를 넣지 않는다. 승인 UX는 버튼 기반이다.

### Investment Chat 설정 변경 승인

Chat agent가 에이전트 model, sleeve allocation, tools, memory, prompt 같은 설정을 제안할 수 있다. 적용은 주문과 동일하게 draft와 액션 버튼을 거친다.

```text
propose_*_config_change()
-> chat_config_draft.<token>
-> /investment-chat/config-drafts
-> apply_approved_config_change()
-> agents_config / tenant runtime config
```

중요한 원칙:

- LLM-facing tool은 구조화된 필드를 받는다.
- `apply_approved_config_change`는 UI bridge용 internal tool이다.
- 직접 update/insert SQL을 LLM에 맡기지 않는다.
- sleeve 배정은 고정 금액, 계좌 전체, 계좌 비율 모드를 지원해야 한다.

### 로컬 모드

`ARENA_MODE=local`이면 BigQuery 대신 DuckDB 기반 `LocalRepository`가 사용된다.

```text
python -m arena.cli init-local --db-path ./data/arena.duckdb
python -m arena.cli clone-bq-local --db-path ./data/arena.duckdb --project <project> --dataset <dataset>
ARENA_MODE=local ARENA_LOCAL_DB_PATH=./data/arena.duckdb python -m arena.cli run-agent-cycle --market us
```

`clone-bq-local --dry-run`은 복제 계획만 보여주고 데이터를 쓰지 않는다.

## 백엔드 분기

| 영역 | Cloud/BigQuery 모드 | Local 모드 |
| --- | --- | --- |
| Repository | `BigQueryRepository` | `LocalRepository` |
| Market/account/memory stores | `arena/data/bigquery/*` | `arena/data/local/*` |
| Vector memory | Vertex/Firestore 또는 설정된 provider | Chroma 또는 null vector store |
| Credentials | Secret Manager + runtime store | local credential store/env |
| Lease | Firestore lease | file/local lease |
| KIS token cache | Firestore token cache | local token cache |

분기 코드는 대부분 `arena/data/factory.py`, `arena/config.py`, `arena/memory/vector_factory.py`, `arena/tenant_leases.py`에 있다.

## Tool Schema 규칙

ADK `FunctionTool`은 함수 signature, type hint, docstring, helper metadata를 보고 LLM에 tool schema를 전달한다. LLM은 이 schema와 tool description을 보고 인자를 채운다.

점검할 때 보는 것:

- 함수명과 parameter명이 사용자가 말할 법한 개념과 맞는가.
- enum/boolean/number/object가 문자열 JSON blob보다 구조적으로 드러나는가.
- docstring 첫 문장이 tool 목적을 명확히 말하는가.
- dangerous/internal tool은 LLM-facing registry에 노출되지 않는가.
- `apply_tool_schema_metadata()`로 enum, required, field 설명이 보강되어 있는가.

특히 chat 설정 변경 도구는 `change_json` 같은 자유 JSON을 LLM에게 만들게 두지 않는 방향이 맞다. 구조화된 proposal tool이 draft를 만들고, apply tool은 승인 bridge가 호출한다.

## 자주 같이 보는 파일 묶음

### Chat 설정/주문

- `arena/agents/investment_chat/config_tools.py`
- `arena/agents/investment_chat/order_tools.py`
- `arena/agents/investment_chat/drafts.py`
- `arena/agents/investment_chat/factory.py`
- `arena/ui/routes/investment_chat.py`
- `arena/ui/investment_chat_adk.py`
- `tests/ui/test_investment_chat_*.py`

### Agent/Admin 설정

- `arena/ui/admin_agent_config.py`
- `arena/ui/admin_runtime_ops.py`
- `arena/ui/routes/settings_admin.py`
- `arena/data/bigquery/runtime_store.py`
- `arena/data/local/config_store.py`
- `tests/config/test_agents_config_*.py`
- `tests/ui/test_*_routes.py`
- `tests/test_runtime_config.py`

### Batch agent tools

- `arena/tools/default_registry.py`
- `arena/tools/registry.py`
- `arena/tools/quant_tools.py`
- `arena/tools/macro_tools.py`
- `arena/tools/sentiment_tools.py`
- `arena/agents/adk_tool_helpers.py`
- `tests/test_tool_registry.py`
- `tests/test_new_tools.py`
- `tests/tools/test_quant_*.py`

### Local DuckDB

- `arena/data/local/schema.py`
- `arena/data/local/repository.py`
- `arena/data/local/*_store.py`
- `arena/cli_commands/admin.py`
- `tests/data/test_init_local_cli.py`
- `tests/data/test_clone_bq_local.py`
- `tests/data/test_duckdb_schema.py`
- `tests/data/test_local_repository.py`

### Memory

- `arena/memory/policy.py`
- `arena/memory/store.py`
- `arena/memory/graph.py`
- `arena/memory/vector_factory.py`
- `arena/agents/memory_compaction_agent.py`
- `tests/memory/test_memory_*.py`
- `tests/memory/test_semantic_relation_*.py`

## 테스트 매핑

| 변경 영역 | 우선 실행할 테스트 |
| --- | --- |
| Investment Chat UI/approval | `tests/ui/test_investment_chat_*.py` |
| ADK agent/tool schema | `tests/test_adk_agents.py`, `tests/test_new_tools.py`, `tests/test_tool_registry.py` |
| Agent/admin config | `tests/config/test_agents_config_*.py`, `tests/ui/test_settings_routes.py`, `tests/ui/test_admin_agent_routes.py`, `tests/test_runtime_config.py` |
| Local DuckDB | `tests/data/test_init_local_cli.py`, `tests/data/test_clone_bq_local.py`, `tests/data/test_duckdb_schema.py`, `tests/data/test_local_repository.py` |
| Memory | `tests/memory/test_memory_*.py`, `tests/memory/test_semantic_relation_*.py` |
| Execution/reconcile | `tests/trading/test_execution_reconcile.py`, `tests/trading/test_reconciliation_*.py`, `tests/trading/test_risk.py` |
| Pipeline/CLI | `tests/cli/`, `tests/test_repo_or_exit_default.py`, 관련 `tests/test_cli_*.py` |

문서만 바꾼 경우에는 `git diff --check`와 markdown fence 균형 정도면 충분하다. 코드 동작을 바꾸면 해당 영역 테스트를 같이 실행한다.

## 빠른 명령

```bash
rg "keyword" arena tests
python -m arena.cli --help
python -m arena.cli run-agent-cycle --help
python -m arena.cli init-local --db-path ./data/arena.duckdb
python -m arena.cli clone-bq-local --db-path ./data/arena.duckdb --project rising-parser-464807-f6 --dataset llm_arena --dry-run
TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/ui/test_investment_chat_*.py tests/adk -q -p no:cacheprovider
```

## 유지 규칙

- 이 파일은 change_log가 아니다. 
- 새 기능을 추가하면 “새 섹션”만 덧붙이지 말고 기존 작업별 진입점, 흐름, 테스트 매핑 안에 녹인다.
- llm이 처음 읽고 바로 열 파일을 고를 수 있도록 하는게 목적이다. 
