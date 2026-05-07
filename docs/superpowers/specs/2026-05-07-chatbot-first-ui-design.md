# 챗봇 우선 UI 재구성 — 설계 스펙

- 날짜: 2026-05-07
- 작성자: midnightn / Claude (brainstorming)
- 상태: 초안 (구현 계획 작성 전)

## 1. 배경 및 목표

투자챗봇이 설정 변경·주문 승인·기능 설명까지 자연어로 처리할 수 있게 만드는 게 제품 비전이다.
지금의 사이드바는 7개 탭으로 분산되어 있어 "어디서 무얼 하는지"를 사용자가 기억해야 한다.
이번 작업은 그 부담을 없애기 위해 **투자챗봇을 사이트 진입 시 메인 화면**으로 승격하고,
나머지 탭을 단순화한다. 챗봇이 다 할 수 있으니, 직접 만지러 들어가는 사람만 환경설정으로 들어가도 되도록 한다.

### 목표(Goals)

- `/`로 들어오면 투자챗봇이 풀블리드로 뜬다.
- 사이드바는 **4개**로 압축한다: 투자챗봇 / 게시판 / 운용성과 / 환경설정.
- 챗봇 화면에서 별도의 chrome(상단 provider/model selector form, 페이지 헤더)을 제거한다.
- provider/model 선택은 **환경설정 → 에이전트 탭**으로 이전한다.
- 환경설정 페이지 상단에 "투자챗봇으로 변경 가능합니다" 한 줄 안내를 추가한다.
- showcase 모드(`/showcase/{tenant}`)도 진입 시 챗봇(보기 전용)을 메인으로 띄운다.
- ADK dev UI는 그대로 유지한다 (iframe 콘텐츠).

### 비목표(Non-goals)

이번 스펙에서 다루지 않는다:

- 챗봇으로 provider/model을 자연어 변경하는 기능.
- 컨텍스트별 모델 자동 오버라이드(투자 상담 = 고급 / 설정·설명 = 저렴).
- ADK iframe 내부의 색감/타이포를 바깥 frame과 통일하는 CSS 주입.
- showcase 챗봇의 정교한 read-only UX(이번엔 write 도구 차단만).
- 환경설정 자체 구조 개편(현행 4탭 유지).

## 2. 정보 구조 (IA)

**사이드바 (4개, 위 → 아래 순):**

1. 투자챗봇 (`/investment-chat`) — active 기본
2. 게시판 (`/board`)
3. 운용성과 (`/nav`)
4. 환경설정 (`/settings?tab=agents`)

**라우팅 변경:**

- `GET /` 의 리다이렉트 대상을 `/board` → `/investment-chat`으로 변경.
- 기존 `/investment-chat`, `/board`, `/nav`, `/settings?tab=…` URL은 그대로 유지(외부 북마크 호환).
- 기존 `/investment-chat?tenant_id=&provider=&model=` 쿼리 호환도 유지. 단 selector UI는 없으므로
  직접 링크로 들어오는 경우 서버측 우선순위 로직(쿼리 → 챗 config → 세션 → tenant default)이 그대로 적용된다.

**세션:** 기존 세션 키 `investment_chat_provider/model/tenant_id`는 그대로 사용한다.

## 3. 챗봇 메인 화면 (`/investment-chat`)

### 레이아웃

```
┌─ 사이드바(200px, 접기 가능)─┬───────── 메인(풀블리드)───────────┐
│ LLM INVEST                  │                                    │
│ ● Operational               │                                    │
│ • 투자챗봇 (active)         │   <iframe src=".../adk/dev-ui/">  │
│ • 게시판                    │   provider/model selector 없음    │
│ • 운용성과                  │                                    │
│ • 환경설정                  │                                    │
│ [Logout]                    │                                    │
└─────────────────────────────┴────────────────────────────────────┘
                                    ↓ floating bottom-corners
                  [좌측: 설정 변경 승인]   [우측: 주문 승인]
```

### 제거되는 요소

- 페이지 헤더 "투자챗봇" — 이미 `hide_page_header=True`로 숨김 처리 중. 그대로 유지(변경 없음).
- 상단 provider/model selector form (`<form action="/investment-chat">` 블록).
- 그 form을 동작시키던 JS 블록(모델 프리셋 JSON 스크립트, change 핸들러).

### 유지되는 요소

- ADK iframe (`/investment-chat/adk/dev-ui/?tenant_id=…&provider=…&model=…`).
- 좌·우 하단 두 토스트 패널(주문 승인 / 설정 변경 승인) 및 폴링 로직.
- 모바일 사이드바 drawer 동작 및 챗봇 active 시 backdrop 투명화 처리.

### iframe URL 빌드

서버는 환경설정에 저장된 chat config / 세션 / tenant default를 우선순위에 따라 해석해 iframe `src`를 빌드한다.
이 로직(`investment_chat` 라우트의 provider/model 결정 블록)은 변경 없음.

## 4. 환경설정 페이지

### URL 구조

`/settings?tab={agents|capital|mcp|memory}` — 현행 그대로.

### 모든 탭 공통: 상단 안내 배너

```
환경설정
┌──────────────────────────────────┐
│ 💬 투자챗봇으로 변경 가능합니다.  │
└──────────────────────────────────┘
[ 에이전트 │ 자본관리 │ 도구관리 │ 기억관리 ]
```

- 톤: `bg-blue-50` / `border-blue-200` 계열, `rounded-xl`, 기존 안내 배너 스타일과 일치.
- 텍스트: `💬 투자챗봇으로 변경 가능합니다.` 한 줄(예시·CTA 버튼 없음).
- 페이지 헤더와 탭 사이에 위치.

### 에이전트 탭 — 신규 카드: Chat Provider/Model

탭 콘텐츠 최상단에 카드 한 개 추가. 그 아래는 현행 에이전트 카드 그대로.

```
┌─ Chat Provider/Model ─────────────────────────────────────┐
│ 투자챗봇이 사용하는 LLM. 페이지 진입 시 이 값이 적용됩니다.│
│ Provider: [ GPT      ▾ ]   Model: [ gpt-5.5      ▾ ]     │
│                                                  [Apply]  │
└───────────────────────────────────────────────────────────┘
```

- 마크업/옵션 데이터/JS는 기존 `investment_chat_body.jinja2` 상단의 selector form을 옮겨 재사용한다.
- Provider 변경 시 모델 옵션이 프리셋 맵에 따라 갱신되는 동작 그대로.
- Provider 옵션은 `tenant_available_provider_specs(repo, tenant_id=…)` 결과로 필터링(현재 챗 페이지 동작과 동일).
- "Apply" 클릭 시 `POST /settings/chat-model`로 폼 제출.

### POST `/settings/chat-model` (신설)

- 입력: `tenant_id`, `provider`, `model` (form-encoded).
- 동작:
  1. `resolve_viewer_context`로 인증/테넌트 검증.
  2. provider/model 정규화 (`canonical_provider`, `normalize_chat_model_selection`).
  3. provider가 tenant에서 사용 가능한 specs 안에 있는지, model이 해당 provider에서 허용되는지 검증.
  4. 검증 실패 시 400 반환(또는 동일 페이지로 에러 노출).
  5. 검증 성공 시 chat agent config 저장(`load_chat_agent_config`/`save_chat_agent_config` 등 기존 메커니즘 활용).
  6. `302 → /settings?tab=agents&saved=1`.
- 사이드 이펙트: 다음번 `/investment-chat` 진입 시 저장된 값으로 iframe URL이 빌드된다.

### 나머지 3개 탭 (자본관리/도구관리/기억관리)

- 변경 없음. 안내 배너만 상단 공통으로 노출.

## 5. Showcase 모드

### 라우팅 변경

- `GET /showcase/{tenant}` → `302 → /showcase/{tenant}/investment-chat` (현행 `…/board`에서 변경).
- `GET /showcase/{tenant}` 무 trailing-slash 경로도 동일.
- `GET /showcase/{tenant}/investment-chat` (신설) — 챗봇 보기 전용 페이지.

### 사이드바 (showcase)

```
LLM INVEST
• 투자챗봇 (active)
• 게시판
• 운용성과
• 에이전트
• 자본관리
• 도구관리
• 기억관리
```

- showcase는 환경설정이 4탭으로 분리되어 있는 현행 구조 유지(이번 작업 범위 밖).
- 투자챗봇만 최상단으로 추가.

### Read-only 정의

- 방문자는 자기 세션의 챗에 메시지를 보낼 수 있고 도구 호출도 가능하다.
- 단, **write 계열 도구(주문 제출, 설정 변경 적용)는 차단**된다.
- 차단 메커니즘: `/showcase/` prefix의 모든 POST가 이미 403을 반환하고 있어 그 가드에 의존한다(현행).
- 추가로, showcase 챗봇 페이지가 사용하는 ADK iframe 세션은 별도 세션 ID를 사용해 owner 세션과 격리한다.
- write 도구가 호출돼도 ADK 측 도구 실행이 `/showcase/` POST 가드를 만나 403으로 실패하도록 한다.
- 주문/설정 승인 토스트 패널은 showcase 페이지에서는 렌더하지 않는다(어차피 승인 자체가 불가).

## 6. 적용 범위 (touch list)

| 파일 | 변경 |
| --- | --- |
| `arena/ui/app.py` | `_root_redirect` 대상을 `/board` → `/investment-chat`로 변경 |
| `arena/ui/layout.py` | `nav_items`를 4개로 축소(투자챗봇 최상단). showcase용 nav도 투자챗봇 최상단으로 |
| `arena/ui/routes/showcase.py` | `/showcase/{tenant}` 리다이렉트 변경, `/showcase/{tenant}/investment-chat` 신설(보기 전용 챗봇) |
| `arena/ui/templates/investment_chat_body.jinja2` | provider/model selector form 블록 + 관련 `<script>` 제거. iframe + 두 토스트 패널만 남김 |
| `arena/ui/routes/investment_chat.py` | selector 관련 헬퍼 일부를 settings 라우트로 이전(공유), 챗 페이지 렌더 단순화 |
| `arena/ui/templates/settings_body.jinja2` | 페이지 헤더와 탭 사이에 안내 배너 추가 |
| `arena/ui/templates/settings_agents_panel.jinja2` | 상단에 Chat Provider/Model 카드 추가 |
| `arena/ui/routes/settings_render_agents.py` (또는 해당 라우트) | 모델 프리셋/옵션 주입(서버측), Apply 액션 URL 주입 |
| `arena/ui/routes/settings_admin.py` (또는 신규) | `POST /settings/chat-model` 핸들러 신설 |

## 7. 호환성 가드

- `/board`, `/nav`, `/settings?tab=…` URL 모두 기존대로 살아있음 — 외부 북마크/링크 안전.
- `/investment-chat?tenant_id=…&provider=…&model=…` 쿼리도 그대로 유지. 단 selector UI는 없으므로,
  직접 링크는 환경설정 값을 일시 오버라이드하는 효과만 갖는다(서버측 우선순위 로직 그대로).
- 세션 키 `investment_chat_provider/model/tenant_id` 그대로 유지.

## 8. 롤백 전략

- 모든 변경이 템플릿/라우트 수준이므로 PR revert 한 번으로 원복.
- feature flag 없이 진행 — 작은 변경 + 단순화가 목표.
- 위험을 더 줄이려면 환경변수 토글(`ARENA_UI_LANDING=chat|board`)로 `/` 리다이렉트만 가드할 수 있으나,
  이번 스펙에서는 토글 없이 직진을 권장한다.

## 9. 테스트 계획

### 자동화

- `tests/ui/`의 기존 라우트 테스트는 `/board`, `/nav`, `/settings`, `/investment-chat` 모두 직접 호출하므로 그대로 통과.
- 신규 테스트:
  - `GET /` → 302 → `/investment-chat`.
  - `GET /investment-chat` 응답에 selector form 마크업이 없음(예: `data-chat-selector-form` 부재 확인).
  - `GET /settings?tab=agents` 응답에 Chat Provider/Model 카드가 있고, 옵션이 tenant credential에 따라 정확히 필터링됨.
  - `POST /settings/chat-model` 유효 입력 → 302, `chat_agent_config`에 저장.
  - `POST /settings/chat-model` 무효 입력(미허용 provider/model) → 400 또는 검증 실패 응답.
  - `GET /showcase/{tenant}` → 302 → `/showcase/{tenant}/investment-chat`.
  - showcase 챗봇 페이지에서 write 도구 트리거 시 `/showcase/` POST 가드에 의해 403.
- 안내 배너: 모든 settings 탭(`agents|capital|mcp|memory`)에서 배너 마크업이 렌더되는지 확인.

### 수동 검증

- 데스크톱: 200px 사이드바 4개 메뉴, 챗 풀블리드, ADK iframe 정상 로드, 사이드바 collapse 토글 동작.
- 모바일: 사이드바 drawer 동작, 챗 active일 때 backdrop 투명 처리.
- 주문/설정 승인 토스트 정상 출현.
- 환경설정에서 모델 변경 → 챗 페이지 진입 시 iframe URL이 변경된 모델로 빌드되는지 확인.
- showcase 모드 진입 시 챗봇 화면이 뜨고, write 도구는 실패하는지 확인.

## 10. 향후 작업 (이번 스펙 밖)

- 챗봇으로 provider/model을 자연어 변경(별도 스펙).
- 컨텍스트별 모델 자동 오버라이드(투자 상담 = 고급 / 설정·설명 = 저렴).
- ADK iframe 내부 색감/타이포 통일을 위한 CSS 주입.
- showcase 챗봇의 정교한 read-only UX(전용 도구 화이트리스트, 승인 패널 대체 표시 등).
