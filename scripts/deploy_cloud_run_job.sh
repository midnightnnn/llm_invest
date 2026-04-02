#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Match scripts/ship.sh behavior when this script is invoked directly.
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

PROJECT="${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT env var}"

REGION="${CLOUD_RUN_REGION:-asia-northeast3}"
JOB_NAME="${CLOUD_RUN_JOB_NAME:-llm-arena-batch}"
AR_REPOSITORY="${AR_REPOSITORY:-llm-arena}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPOSITORY}/${JOB_NAME}:latest"

A_SPLIT_JOBS="${A_SPLIT_JOBS:-true}"
DEFAULT_TARGET_MARKET="${DEFAULT_TARGET_MARKET:-us}"

# ── Dual-market scheduling ────────────────────────────────────────────
# Deploy two scheduler jobs when DUAL_MARKET=true:
#   1) US job  — runs at US market hours (ET)
#   2) KR job  — runs at KOSPI market hours (KST)
# Each passes --market flag so only that market's data is synced/forecast.
DUAL_MARKET="${DUAL_MARKET:-true}"

SCHEDULER_JOB_NAME="${SCHEDULER_JOB_NAME:-${JOB_NAME}-scheduler}"
SCHEDULER_CRON="${SCHEDULER_CRON:-0 15 * * 1-5}"
SCHEDULER_TIMEZONE="${SCHEDULER_TIMEZONE:-America/New_York}"
SCHEDULER_SA_EMAIL="${SCHEDULER_SA_EMAIL:-${PROJECT}@appspot.gserviceaccount.com}"

# US schedule (used when DUAL_MARKET=true)
SCHEDULER_US_CRON="${SCHEDULER_US_CRON:-0 15 * * 1-5}"
SCHEDULER_US_TIMEZONE="${SCHEDULER_US_TIMEZONE:-America/New_York}"
# KOSPI schedule (used when DUAL_MARKET=true)
SCHEDULER_KR_CRON="${SCHEDULER_KR_CRON:-30 14 * * 1-5}"
SCHEDULER_KR_TIMEZONE="${SCHEDULER_KR_TIMEZONE:-Asia/Seoul}"
SCHEDULER_RUN_BODY='{"overrides":{"containerOverrides":[{"env":[{"name":"ARENA_EXECUTION_SOURCE","value":"scheduler"}]}]}}'

RUN_SERVICE_ACCOUNT="${RUN_SERVICE_ACCOUNT:-${PROJECT}@appspot.gserviceaccount.com}"
RUN_ENV_VARS="${RUN_ENV_VARS:-GOOGLE_CLOUD_PROJECT=${PROJECT}@BQ_DATASET=llm_arena@BQ_LOCATION=${REGION}@ARENA_LOG_LEVEL=INFO@ARENA_TRADING_MODE=live@ARENA_ALLOW_LIVE_TRADING=true@ARENA_AGENT_MODE=adk@ARENA_AGENT_IDS=gemini,gpt,claude@OPENAI_MODEL=gpt-5.2@GOOGLE_GENAI_USE_VERTEXAI=true@GOOGLE_CLOUD_LOCATION=global@GEMINI_MODEL=gemini-3-flash-preview@ANTHROPIC_MODEL=claude-sonnet-4-6@ANTHROPIC_USE_VERTEXAI=false@ARENA_LLM_TIMEOUT_SECONDS=600@ARENA_ADK_RETRY_MAX=5@ARENA_ADK_RETRY_BACKOFF_SECONDS=10.0@ARENA_NASDAQ_CYCLE_TIMES_ET=15:00@ARENA_NASDAQ_CYCLE_TOLERANCE_MINUTES=30@ARENA_NASDAQ_DISABLE_SCHEDULE_GUARD=false@ARENA_KOSPI_CYCLE_TIMES_KST=${ARENA_KOSPI_CYCLE_TIMES_KST:-14:30}@ARENA_KOSPI_CYCLE_TOLERANCE_MINUTES=${ARENA_KOSPI_CYCLE_TOLERANCE_MINUTES:-20}@ARENA_KOSPI_DISABLE_SCHEDULE_GUARD=false@ARENA_SLEEVE_CAPITAL_KRW=2000000@ARENA_FORCE_SLEEVE_REINIT=false@ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT=false@ARENA_UNIVERSE_RUN_TOP_N=400@ARENA_UNIVERSE_PER_EXCHANGE_CAP=200@ARENA_US_QUOTE_EXCHANGES=NAS,NYS@ARENA_FORECAST_MODE=all@ARENA_FORECAST_BASE_MODELS=neural@ARENA_AUTONOMY_WORKING_SET_ENABLED=true@ARENA_AUTONOMY_TOOL_DEFAULT_CANDIDATES_ENABLED=true@ARENA_AUTONOMY_OPPORTUNITY_CONTEXT_ENABLED=true@KIS_TOKEN_CACHE_BACKEND=firestore@KIS_TOKEN_CACHE_COLLECTION=api_tokens@KIS_ENV=real@KIS_TARGET_MARKET=us@KIS_SECRET_NAME=KISAPI@KIS_SECRET_VERSION=latest@KIS_ACCOUNT_KEY_SUFFIX=CO@KIS_CONFIRM_FILLS=true@KIS_CONFIRM_TIMEOUT_SECONDS=25@KIS_CONFIRM_POLL_SECONDS=2.0@ARENA_USD_KRW_FX_MARKET_DIV_CODE=${ARENA_USD_KRW_FX_MARKET_DIV_CODE:-X}@ARENA_USD_KRW_FX_SYMBOL=${ARENA_USD_KRW_FX_SYMBOL:-USDKRW}@ARENA_PUBLIC_DEMO_TENANT=${ARENA_PUBLIC_DEMO_TENANT:-}@ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT=${ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT:-}}"
if [[ -n "${FRED_API_KEY:-}" ]]; then
  RUN_ENV_VARS="${RUN_ENV_VARS}@FRED_API_KEY=${FRED_API_KEY}"
fi
if [[ -n "${ECOS_API_KEY:-}" ]]; then
  RUN_ENV_VARS="${RUN_ENV_VARS}@ECOS_API_KEY=${ECOS_API_KEY}"
fi
# Multi-tenant runtime loads model/KIS secrets from runtime_credentials per tenant.
# Keep this empty by default; set explicitly only for single-tenant fallback operation.
RUN_SECRETS="${RUN_SECRETS:-}"
TASK_TIMEOUT="${CLOUD_RUN_TASK_TIMEOUT:-3600s}"
CPU="${CLOUD_RUN_CPU:-4}"
MEMORY="${CLOUD_RUN_MEMORY:-16Gi}"
RUN_COMMAND="${RUN_COMMAND:-python}"
# Single pipeline job: sync → forecast → agent cycle
RUN_ARGS="${RUN_ARGS:--m,arena.cli,run-pipeline,--live,--all-tenants}"

PREP_RUN_COMMAND="${PREP_RUN_COMMAND:-python}"
PREP_RUN_ARGS="${PREP_RUN_ARGS:--m,arena.cli,run-shared-prep,--live}"
PREP_TASK_TIMEOUT="${PREP_TASK_TIMEOUT:-${TASK_TIMEOUT}}"
PREP_CPU="${PREP_CPU:-${CPU}}"
PREP_MEMORY="${PREP_MEMORY:-${MEMORY}}"
PREP_TASKS="${PREP_TASKS:-1}"
PREP_PARALLELISM="${PREP_PARALLELISM:-1}"

AGENT_RUN_COMMAND="${AGENT_RUN_COMMAND:-python}"
AGENT_RUN_ARGS="${AGENT_RUN_ARGS:--m,arena.cli,run-agent-cycle,--live,--all-tenants}"
AGENT_TASK_TIMEOUT="${AGENT_TASK_TIMEOUT:-${TASK_TIMEOUT}}"
AGENT_CPU="${AGENT_CPU:-${CPU}}"
AGENT_MEMORY="${AGENT_MEMORY:-${MEMORY}}"
AGENT_TASKS="${AGENT_TASKS:-10}"
AGENT_PARALLELISM="${AGENT_PARALLELISM:-10}"
AGENT_BATCH_PARALLEL="${AGENT_BATCH_PARALLEL:-1}"
SKIP_BUILD="${SKIP_BUILD:-false}"

KIS_SECRET_NAME="${KIS_SECRET_NAME:-KISAPI}"
GRANT_SECRET_ACCESS="${GRANT_SECRET_ACCESS:-true}"
GRANT_PROJECT_SECRET_ACCESS="${GRANT_PROJECT_SECRET_ACCESS:-true}"

cd "${ROOT_DIR}"

echo "[1/6] Enable required APIs"
gcloud services enable \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com

echo "[2/6] Ensure Artifact Registry"
if ! gcloud artifacts repositories describe "${AR_REPOSITORY}" --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPOSITORY}" \
    --repository-format=docker \
    --location "${REGION}" \
    --description "LLM Arena images" \
    --project "${PROJECT}"
fi

if [[ "${SKIP_BUILD,,}" == "true" ]]; then
  echo "[3/6] Skip build (SKIP_BUILD=true)"
else
  echo "[3/6] Build container image"
  gcloud builds submit --tag "${IMAGE}" --project "${PROJECT}" .
fi

echo "[4/6] Ensure runtime secret access"
if [[ "${GRANT_SECRET_ACCESS,,}" == "true" ]]; then
  if [[ "${GRANT_PROJECT_SECRET_ACCESS,,}" == "true" ]]; then
    gcloud projects add-iam-policy-binding "${PROJECT}" \
      --member "serviceAccount:${RUN_SERVICE_ACCOUNT}" \
      --role "roles/secretmanager.secretAccessor" >/dev/null
    echo "Granted project-level secretAccessor to ${RUN_SERVICE_ACCOUNT}"
  fi

  if gcloud secrets describe "${KIS_SECRET_NAME}" --project "${PROJECT}" >/dev/null 2>&1; then
    gcloud secrets add-iam-policy-binding "${KIS_SECRET_NAME}" \
      --project "${PROJECT}" \
      --member "serviceAccount:${RUN_SERVICE_ACCOUNT}" \
      --role "roles/secretmanager.secretAccessor" >/dev/null
    echo "Granted secretAccessor on ${KIS_SECRET_NAME} to ${RUN_SERVICE_ACCOUNT}"
  else
    echo "Secret ${KIS_SECRET_NAME} not found; skipping secretAccessor binding"
  fi

  if [[ -n "${RUN_SECRETS}" ]]; then
    IFS=',' read -ra SECRET_PAIRS <<< "${RUN_SECRETS}"
    for pair in "${SECRET_PAIRS[@]}"; do
      secret_ref="${pair#*=}"
      secret_name="${secret_ref%%:*}"
      if [[ -z "${secret_name}" ]]; then
        continue
      fi
      if gcloud secrets describe "${secret_name}" --project "${PROJECT}" >/dev/null 2>&1; then
        gcloud secrets add-iam-policy-binding "${secret_name}" \
          --project "${PROJECT}" \
          --member "serviceAccount:${RUN_SERVICE_ACCOUNT}" \
          --role "roles/secretmanager.secretAccessor" >/dev/null
        echo "Granted secretAccessor on ${secret_name} to ${RUN_SERVICE_ACCOUNT}"
      else
        echo "Secret ${secret_name} not found; skipping secretAccessor binding"
      fi
    done
  fi
fi

echo "[5/6] Deploy Cloud Run batch job(s)"
RUN_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${JOB_NAME}:run"

echo "[6/6] Create or update Cloud Scheduler"
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member "serviceAccount:${SCHEDULER_SA_EMAIL}" \
  --role "roles/run.invoker" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member "serviceAccount:${SCHEDULER_SA_EMAIL}" \
  --role "roles/run.jobsExecutorWithOverrides" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member "serviceAccount:${RUN_SERVICE_ACCOUNT}" \
  --role "roles/run.invoker" >/dev/null
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member "serviceAccount:${RUN_SERVICE_ACCOUNT}" \
  --role "roles/run.jobsExecutorWithOverrides" >/dev/null

_upsert_scheduler() {
  local name="$1" cron="$2" tz="$3" uri="$4" body="${5:-}"
  local update_cmd=(
    gcloud scheduler jobs update http "${name}"
    --location "${REGION}" --project "${PROJECT}"
    --schedule "${cron}" --time-zone "${tz}"
    --uri "${uri}" --http-method POST
    --oauth-service-account-email "${SCHEDULER_SA_EMAIL}"
    --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform"
  )
  local create_cmd=(
    gcloud scheduler jobs create http "${name}"
    --location "${REGION}" --project "${PROJECT}"
    --schedule "${cron}" --time-zone "${tz}"
    --uri "${uri}" --http-method POST
    --oauth-service-account-email "${SCHEDULER_SA_EMAIL}"
    --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform"
  )
  if [[ -n "${body}" ]]; then
    update_cmd+=(--update-headers "Content-Type=application/json" --message-body "${body}")
    create_cmd+=(--headers "Content-Type=application/json" --message-body "${body}")
  fi
  if gcloud scheduler jobs describe "${name}" --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1; then
    "${update_cmd[@]}"
  else
    "${create_cmd[@]}"
  fi
  echo "Scheduler: ${name} (${cron} ${tz})"
}

_delete_job_if_exists() {
  local job_name="$1"
  if gcloud run jobs describe "${job_name}" --region "${REGION}" --project "${PROJECT}" >/dev/null 2>&1; then
    gcloud run jobs delete "${job_name}" --region "${REGION}" --project "${PROJECT}" --quiet
    echo "Deleted legacy Cloud Run Job: ${job_name}"
  fi
}

_delete_scheduler_if_exists() {
  local name="$1"
  if gcloud scheduler jobs describe "${name}" --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1; then
    gcloud scheduler jobs delete "${name}" --location "${REGION}" --project "${PROJECT}" --quiet
    echo "Deleted legacy Scheduler: ${name}"
  fi
}

_set_delimited_env_var() {
  local env_vars="$1"
  local key="$2"
  local value="$3"
  local result=""
  local found="false"
  local entry

  IFS='@' read -ra _ENV_ENTRIES <<< "${env_vars}"
  for entry in "${_ENV_ENTRIES[@]}"; do
    [[ -z "${entry}" ]] && continue
    if [[ "${entry}" == "${key}="* ]]; then
      entry="${key}=${value}"
      found="true"
    fi
    if [[ -n "${result}" ]]; then
      result="${result}@${entry}"
    else
      result="${entry}"
    fi
  done
  if [[ "${found}" != "true" ]]; then
    if [[ -n "${result}" ]]; then
      result="${result}@${key}=${value}"
    else
      result="${key}=${value}"
    fi
  fi
  printf '%s' "${result}"
}

_cleanup_dual_market_artifacts() {
  _delete_scheduler_if_exists "${SCHEDULER_JOB_NAME}-us"
  _delete_scheduler_if_exists "${SCHEDULER_JOB_NAME}-kospi"
  _delete_job_if_exists "${JOB_NAME}-us"
  _delete_job_if_exists "${JOB_NAME}-kospi"
  _delete_job_if_exists "${JOB_NAME}-prep-us"
  _delete_job_if_exists "${JOB_NAME}-prep-kospi"
  _delete_job_if_exists "${JOB_NAME}-agent-us"
  _delete_job_if_exists "${JOB_NAME}-agent-kospi"
}

_market_env_vars() {
  local market="$1"
  local env_vars="${RUN_ENV_VARS}"
  local extra_envs="${2:-}"

  env_vars="$(_set_delimited_env_var "${env_vars}" "KIS_TARGET_MARKET" "${market}")"
  env_vars="$(_set_delimited_env_var "${env_vars}" "ARENA_CLOUD_RUN_REGION" "${REGION}")"
  if [[ -n "${extra_envs}" ]]; then
    env_vars="${env_vars}@${extra_envs}"
  fi
  printf '%s' "${env_vars}"
}

_deploy_job() {
  local job_name="$1"
  local command="$2"
  local args="$3"
  local cpu="$4"
  local memory="$5"
  local timeout="$6"
  local tasks="$7"
  local parallelism="$8"
  local env_vars="$9"

  local deploy_cmd=(
    gcloud run jobs deploy "${job_name}"
    --image "${IMAGE}"
    --region "${REGION}"
    --project "${PROJECT}"
    --command "${command}"
    "--args=${args}"
    --service-account "${RUN_SERVICE_ACCOUNT}"
    --cpu "${cpu}"
    --memory "${memory}"
    --task-timeout "${timeout}"
    --max-retries 1
    --tasks "${tasks}"
    --parallelism "${parallelism}"
    --update-env-vars "^@^${env_vars}"
  )

  if [[ -n "${RUN_SECRETS}" ]]; then
    deploy_cmd+=(--set-secrets "${RUN_SECRETS}")
  else
    deploy_cmd+=(--clear-secrets)
  fi

  "${deploy_cmd[@]}"
}

if [[ "${DUAL_MARKET,,}" == "true" ]]; then
  if [[ "${A_SPLIT_JOBS,,}" == "true" ]]; then
    PREP_US_JOB="${JOB_NAME}-prep-us"
    PREP_KR_JOB="${JOB_NAME}-prep-kospi"
    AGENT_US_JOB="${JOB_NAME}-agent-us"
    AGENT_KR_JOB="${JOB_NAME}-agent-kospi"

    # A-plan split mode: scheduler triggers prep jobs only; prep dispatches the agent jobs.
    _delete_scheduler_if_exists "${SCHEDULER_JOB_NAME}"
    _delete_job_if_exists "${JOB_NAME}"
    _delete_job_if_exists "${JOB_NAME}-us"
    _delete_job_if_exists "${JOB_NAME}-kospi"

    _deploy_job "${PREP_US_JOB}" "${PREP_RUN_COMMAND}" "${PREP_RUN_ARGS},--market,us,--dispatch-job,${AGENT_US_JOB}" "${PREP_CPU}" "${PREP_MEMORY}" "${PREP_TASK_TIMEOUT}" "${PREP_TASKS}" "${PREP_PARALLELISM}" "$(_market_env_vars us)"
    _deploy_job "${PREP_KR_JOB}" "${PREP_RUN_COMMAND}" "${PREP_RUN_ARGS},--market,kospi,--dispatch-job,${AGENT_KR_JOB}" "${PREP_CPU}" "${PREP_MEMORY}" "${PREP_TASK_TIMEOUT}" "${PREP_TASKS}" "${PREP_PARALLELISM}" "$(_market_env_vars kospi)"
    _deploy_job "${AGENT_US_JOB}" "${AGENT_RUN_COMMAND}" "${AGENT_RUN_ARGS},--market,us" "${AGENT_CPU}" "${AGENT_MEMORY}" "${AGENT_TASK_TIMEOUT}" "${AGENT_TASKS}" "${AGENT_PARALLELISM}" "$(_market_env_vars us "ARENA_BATCH_PARALLEL=${AGENT_BATCH_PARALLEL}@ARENA_TENANT_LEASE_ENABLED=true")"
    _deploy_job "${AGENT_KR_JOB}" "${AGENT_RUN_COMMAND}" "${AGENT_RUN_ARGS},--market,kospi" "${AGENT_CPU}" "${AGENT_MEMORY}" "${AGENT_TASK_TIMEOUT}" "${AGENT_TASKS}" "${AGENT_PARALLELISM}" "$(_market_env_vars kospi "ARENA_BATCH_PARALLEL=${AGENT_BATCH_PARALLEL}@ARENA_TENANT_LEASE_ENABLED=true")"

    PREP_US_RUN_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${PREP_US_JOB}:run"
    PREP_KR_RUN_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${PREP_KR_JOB}:run"

    _upsert_scheduler "${SCHEDULER_JOB_NAME}-us" "${SCHEDULER_US_CRON}" "${SCHEDULER_US_TIMEZONE}" "${PREP_US_RUN_URL}" "${SCHEDULER_RUN_BODY}"
    _upsert_scheduler "${SCHEDULER_JOB_NAME}-kospi" "${SCHEDULER_KR_CRON}" "${SCHEDULER_KR_TIMEZONE}" "${PREP_KR_RUN_URL}" "${SCHEDULER_RUN_BODY}"

    echo ""
    echo "Done (split A-plan draft)"
    echo "Prep jobs:  ${PREP_US_JOB}, ${PREP_KR_JOB}"
    echo "Agent jobs:  ${AGENT_US_JOB}, ${AGENT_KR_JOB}"
    echo "Schedulers:  ${SCHEDULER_JOB_NAME}-us, ${SCHEDULER_JOB_NAME}-kospi"
    echo "Agent tasks: ${AGENT_TASKS} parallelism=${AGENT_PARALLELISM}"
  else
    # Legacy dual-market behavior: one job per market, each runs the full pipeline.
    US_JOB="${JOB_NAME}-us"
    KR_JOB="${JOB_NAME}-kospi"
    US_ARGS="-m,arena.cli,run-pipeline,--live,--all-tenants,--market,us"
    KR_ARGS="-m,arena.cli,run-pipeline,--live,--all-tenants,--market,kospi"

    # Clean up legacy single-job scheduler if present.
    _delete_scheduler_if_exists "${SCHEDULER_JOB_NAME}"

    # Clean up legacy single Cloud Run Job if present so dual-market mode has only two jobs.
    _delete_job_if_exists "${JOB_NAME}"
    _delete_job_if_exists "${JOB_NAME}-prep-us"
    _delete_job_if_exists "${JOB_NAME}-prep-kospi"
    _delete_job_if_exists "${JOB_NAME}-agent-us"
    _delete_job_if_exists "${JOB_NAME}-agent-kospi"

    _deploy_job "${US_JOB}" "${RUN_COMMAND}" "${US_ARGS}" "${CPU}" "${MEMORY}" "${TASK_TIMEOUT}" "1" "1" "$(_market_env_vars us)"
    _deploy_job "${KR_JOB}" "${RUN_COMMAND}" "${KR_ARGS}" "${CPU}" "${MEMORY}" "${TASK_TIMEOUT}" "1" "1" "$(_market_env_vars kospi)"

    US_RUN_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${US_JOB}:run"
    KR_RUN_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${KR_JOB}:run"

    _upsert_scheduler "${SCHEDULER_JOB_NAME}-us" "${SCHEDULER_US_CRON}" "${SCHEDULER_US_TIMEZONE}" "${US_RUN_URL}" "${SCHEDULER_RUN_BODY}"
    _upsert_scheduler "${SCHEDULER_JOB_NAME}-kospi" "${SCHEDULER_KR_CRON}" "${SCHEDULER_KR_TIMEZONE}" "${KR_RUN_URL}" "${SCHEDULER_RUN_BODY}"

    echo ""
    echo "Done (dual-market)"
    echo "US Job:    ${US_JOB}    schedule=${SCHEDULER_US_CRON} ${SCHEDULER_US_TIMEZONE}"
    echo "KOSPI Job: ${KR_JOB}   schedule=${SCHEDULER_KR_CRON} ${SCHEDULER_KR_TIMEZONE}"
  fi
else
  # Single job (original behavior)
  _cleanup_dual_market_artifacts
  _deploy_job "${JOB_NAME}" "${RUN_COMMAND}" "${RUN_ARGS}" "${CPU}" "${MEMORY}" "${TASK_TIMEOUT}" "1" "1" "$(_market_env_vars "${DEFAULT_TARGET_MARKET}")"

  _upsert_scheduler "${SCHEDULER_JOB_NAME}" "${SCHEDULER_CRON}" "${SCHEDULER_TIMEZONE}" "${RUN_URL}" "${SCHEDULER_RUN_BODY}"

  echo ""
  echo "Done"
  echo "Cloud Run Job: ${JOB_NAME} (${REGION})"
  echo "Run command: ${RUN_COMMAND} ${RUN_ARGS}"
  echo "Scheduler Job: ${SCHEDULER_JOB_NAME}"
fi

echo "Run Service Account: ${RUN_SERVICE_ACCOUNT}"
if [[ -n "${RUN_SECRETS}" ]]; then
  echo "Run secrets: ${RUN_SECRETS}"
fi
