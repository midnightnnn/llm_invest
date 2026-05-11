# New Local Machine E2E Setup

> 새 로컬 컴퓨터에서 `https://github.com/midnightnnn/llm_invest`를 clone한 뒤,
> 로컬 데모부터 GCP/Cloud Run 운영 경로까지 재현하기 위해 필요한 설정 목록이다.
> 최종 업데이트: 2026-05-11

이 문서는 `git clone` 외에 빠지는 항목을 정리한다. 레포에는 `.env`, `.venv`, `data/`, `logs/`, GCP 인증, Secret Manager 값이 들어오지 않는다.

## 결론

실행 수준은 세 단계로 나뉜다.

| 목표 | GCP 필요 | KIS key 위치 | 최소 준비 |
| --- | --- | --- | --- |
| 로컬 데모 UI | 아니오 | 불필요 | Python 3.12, LLM API key 1개, `.env.local.example` |
| 로컬에서 GCP 데이터 연결 | 예 | Secret Manager 권장, `.env` fallback 가능 | `gcloud` ADC, BigQuery/Firestore 접근, `.env.example` |
| Cloud Run 운영 e2e | 예 | Secret Manager + BigQuery `runtime_credentials` | GCP API/IAM, OAuth, tenant credentials, 배포 스크립트 |

운영 기준으로 KIS key는 `.env`에 넣지 않는다. KIS key는 Secret Manager에 저장하고, 어떤 tenant가 어떤 secret을 쓰는지는 BigQuery `runtime_credentials`가 들고 있다. `.env`의 `KIS_API_KEY`, `KIS_API_SECRET`, `KIS_ACCOUNT_NO`는 Secret Manager 없이 단일 로컬 실행을 하는 fallback 경로다.

## Clone 후 새 컴퓨터에 없는 것

`.gitignore` 기준으로 아래는 레포에서 오지 않는다.

- `.env`
- `.venv/`
- `data/arena.duckdb`, `data/chroma`, tenant lease 파일
- `logs/`
- `~/.llm-arena/credentials.json`
- `~/.llm-arena/tokens.json`
- `gcloud auth login`, `gcloud auth application-default login` 결과
- GCP BigQuery dataset/table data
- GCP Firestore data
- GCP Secret Manager secret versions
- Google OAuth client id/secret

## 1. 로컬 데모만 실행

GCP 없이 UI와 데모 데이터를 확인하는 경로다. BigQuery, Firestore, Secret Manager, Cloud Run이 필요 없다.

필요한 것:

- Python 3.12+
- Git
- LLM API key 최소 1개: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` 중 하나

명령:

```bash
git clone https://github.com/midnightnnn/llm_invest.git
cd llm_invest

python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[local]"

cp .env.local.example .env
# .env에 OPENAI_API_KEY 또는 GEMINI_API_KEY 또는 ANTHROPIC_API_KEY 입력

llm-arena init-local
llm-arena seed-local-demo
llm-arena serve-ui
```

브라우저:

```text
http://127.0.0.1:8080
```

선택 사항:

```bash
pip install -e ".[local,local-vector]"
```

`local-vector`를 설치하면 ChromaDB + sentence-transformers 기반 로컬 벡터 검색을 쓴다. 설치하지 않아도 recency fallback으로 UI/기본 흐름은 동작한다.

## 2. 로컬에서 기존 GCP 프로젝트에 연결

새 컴퓨터에서 운영 GCP 데이터에 붙어 CLI/UI를 실행하는 경로다.

필요한 로컬 도구:

- Python 3.12+
- `gcloud` CLI
- 프로젝트 접근 권한이 있는 Google 계정

인증:

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
gcloud config set project YOUR_PROJECT_ID
```

설치:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

`.env`:

```bash
cp .env.example .env
```

필수 또는 거의 필수:

```env
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
BQ_DATASET=llm_arena
BQ_LOCATION=asia-northeast3
ARENA_AGENT_MODE=adk
ARENA_AGENT_IDS=gemini,gpt,claude
KIS_TARGET_MARKET=us
```

단일 tenant를 로컬에서 빠르게 돌릴 때는 LLM API key를 `.env`에 둘 수 있다.

```env
OPENAI_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
```

다만 Cloud Run 운영과 동일하게 맞추려면 LLM key도 Secret Manager에 tenant별 model secret으로 저장하는 편이 맞다.

초기화/확인:

```bash
llm-arena init-bq
llm-arena serve-ui
```

사이클 실행:

```bash
llm-arena run-pipeline --market us
```

시장 시간/휴장일 가드가 있어서 `run-pipeline`은 시간이 맞지 않으면 정상적으로 skip될 수 있다. UI는 사이클 없이도 뜬다.

## 3. KIS credentials 운영 방식

운영 경로:

```text
Secret Manager payload
-> BigQuery runtime_credentials.kis_secret_name
-> _apply_tenant_runtime_credentials()
-> OpenTradingClient loads Secret Manager
```

즉 새 컴퓨터의 `.env`에 KIS key 값을 넣을 필요가 없다. 필요한 것은 GCP 인증과 Secret Manager 접근 권한이다.

UI 설정 화면에서 KIS 계정을 저장하면 보통 아래가 만들어진다.

```text
Secret Manager:
  llm-arena-<tenant>-kis
  llm-arena-<tenant>-models

BigQuery:
  runtime_credentials row
    tenant_id=<tenant>
    kis_secret_name=llm-arena-<tenant>-kis
    model_secret_name=llm-arena-<tenant>-models
```

KIS secret payload 형태:

```json
{
  "ACCOUNTS": [
    {
      "env": "real",
      "cano": "12345678",
      "prdt_cd": "01",
      "app_key": "real-app-key",
      "app_secret": "real-app-secret",
      "paper_app_key": "paper-app-key",
      "paper_app_secret": "paper-app-secret",
      "key_suffix": "CO"
    }
  ],
  "updated_at": "2026-05-11T00:00:00+00:00"
}
```

`key_suffix`는 여러 계정을 하나의 secret에 넣고 `KIS_ACCOUNT_KEY_SUFFIX`로 선택할 때 쓰는 선택 필드다.

Model secret payload 형태:

```json
{
  "providers": {
    "gpt": { "api_key": "sk-..." },
    "gemini": { "api_key": "AIza..." },
    "claude": { "api_key": "sk-ant-..." }
  },
  "openai_api_key": "sk-...",
  "gemini_api_key": "AIza...",
  "anthropic_api_key": "sk-ant-...",
  "updated_at": "2026-05-11T00:00:00+00:00"
}
```

수동으로 secret을 만들 수는 있지만, 가능하면 UI 설정 화면이나 `CredentialStore` 경로를 사용한다. 그래야 Secret Manager와 `runtime_credentials` metadata가 같이 맞춰진다.

## 4. Cloud Run 운영 e2e

Cloud Run까지 새 컴퓨터에서 배포/운영하려면 로컬 설정뿐 아니라 GCP 리소스/IAM이 필요하다.

로컬 도구:

- `gcloud` CLI
- Docker: `./scripts/ship.sh`를 쓸 때 필요
- Docker 없이 배포하려면 `scripts/deploy_cloud_run_job.sh`, `scripts/deploy_cloud_run_ui.sh`가 Cloud Build를 사용한다

필요 GCP API:

- BigQuery
- Firestore
- Secret Manager
- Cloud Run
- Cloud Build
- Artifact Registry
- Cloud Scheduler
- Vertex AI: Gemini Vertex, text embedding, Anthropic Vertex를 쓸 때

Cloud Run 배포용 `.env` 핵심:

```env
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
BQ_DATASET=llm_arena
BQ_LOCATION=asia-northeast3
CLOUD_RUN_REGION=asia-northeast3

ARENA_UI_AUTH_ENABLED=true
ARENA_UI_SETTINGS_ENABLED=true
ARENA_OPERATOR_EMAILS=you@example.com
ARENA_PUBLIC_DEMO_TENANT=
ARENA_SHOWCASE_TENANT=
ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT=
```

UI OAuth:

```env
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
ARENA_UI_GOOGLE_REDIRECT_URI=https://YOUR_CLOUD_RUN_URL/auth/google/callback
ARENA_UI_SESSION_SECRET=strong-random-secret
```

배포 스크립트가 기본으로 찾는 Secret Manager secret 이름:

```text
GCID                    # Google OAuth client id
GCPASS                  # Google OAuth client secret
ARENA_UI_SESSION_SECRET # UI session secret
```

배포:

```bash
bash scripts/deploy_cloud_run_job.sh
bash scripts/deploy_cloud_run_ui.sh
```

또는 Docker build/push까지 로컬에서 처리:

```bash
./scripts/ship.sh all
```

주의: `deploy_cloud_run_job.sh`의 기본 Cloud Run Job은 `--all-tenants --live` 계열이다. 따라서 `.env`의 KIS key가 아니라 각 tenant의 `runtime_credentials`와 Secret Manager secret이 있어야 한다. tenant credentials가 없으면 `tenant runtime credentials missing`, `tenant model_secret_name missing`, `tenant kis_secret_name missing` 류로 막힌다.

## 5. IAM 체크리스트

배포를 실행하는 사용자에게 필요한 권한:

- Cloud Run Admin
- Cloud Build Editor
- Artifact Registry Admin 또는 Writer
- Secret Manager Admin
- BigQuery Admin
- Service Account User
- Project IAM Admin: 배포 스크립트가 IAM binding을 추가한다
- Cloud Scheduler Admin

Cloud Run runtime service account 기본값:

```text
${GOOGLE_CLOUD_PROJECT}@appspot.gserviceaccount.com
```

runtime service account에 필요한 권한:

- `roles/bigquery.jobUser`
- `roles/bigquery.dataEditor`
- `roles/secretmanager.secretAccessor`
- `roles/datastore.user` 또는 Firestore 접근 가능 권한
- `roles/aiplatform.user`: Vertex AI 사용 시
- `roles/run.invoker`
- `roles/run.jobsExecutorWithOverrides`

배포 스크립트는 일부 권한을 자동으로 부여하지만, 새 프로젝트/새 계정이면 먼저 수동 확인이 필요할 수 있다.

## 6. 빠른 검증 명령

로컬 인증:

```bash
gcloud auth list
gcloud auth application-default print-access-token >/dev/null
gcloud config get-value project
```

GCP 리소스:

```bash
gcloud services list --enabled --filter='name:(bigquery.googleapis.com OR firestore.googleapis.com OR secretmanager.googleapis.com OR run.googleapis.com OR cloudbuild.googleapis.com OR artifactregistry.googleapis.com OR cloudscheduler.googleapis.com)'
gcloud secrets list --filter='name:llm-arena'
bq --project_id=YOUR_PROJECT_ID ls
```

앱:

```bash
llm-arena init-bq
llm-arena serve-ui
```

Cloud Run:

```bash
gcloud run jobs list --region asia-northeast3
gcloud run services list --region asia-northeast3
gcloud scheduler jobs list --location asia-northeast3
```

## 7. 흔한 에러와 원인

| 에러/증상 | 원인 | 처리 |
| --- | --- | --- |
| `GOOGLE_CLOUD_PROJECT is required` | GCP 모드인데 `.env`에 프로젝트가 없음 | `.env` 설정 또는 `ARENA_MODE=local` |
| `BigQuery auth failed` | ADC 없음 | `gcloud auth application-default login` |
| `no runtime tenants resolved` | `runtime_credentials` row 없음 | UI에서 tenant credentials 저장 또는 `ARENA_BATCH_TENANTS`/demo tenant 설정 |
| `tenant model_secret_name missing` | tenant model secret metadata 없음 | UI 설정에서 LLM key 저장 |
| `tenant kis_secret_name missing` | tenant KIS secret metadata 없음 | UI 설정에서 KIS 계정 저장 |
| `KIS credentials are missing` | Secret Manager payload가 비었거나 계정 선택 실패 | secret payload, `KIS_ACCOUNT_KEY_SUFFIX`, `cano/prdt_cd` 확인 |
| UI OAuth redirect 에러 | Google OAuth redirect URI 불일치 | Cloud Run URL의 `/auth/google/callback` 등록 |
| Firestore vector warning | vector index 또는 Firestore 권한 부족 | 검색만 fallback될 수 있음. Firestore 권한/index 확인 |

## 권장 순서

1. `.[local]` + `.env.local.example`로 로컬 데모 UI를 먼저 띄운다.
2. `gcloud` ADC를 설정하고 `.env.example` 기반으로 `llm-arena init-bq`를 실행한다.
3. `llm-arena serve-ui`로 BigQuery 연결과 UI 접근을 확인한다.
4. UI 설정 화면에서 tenant별 LLM/KIS credentials를 저장한다.
5. Cloud Run UI/Job을 배포한다.
6. Scheduler/Job 로그에서 tenant runtime credential hydrate가 성공하는지 확인한다.
