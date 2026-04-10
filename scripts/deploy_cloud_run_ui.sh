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
SERVICE_NAME="${CLOUD_RUN_SERVICE_NAME:-llm-arena-ui}"
AR_REPOSITORY="${AR_REPOSITORY:-llm-arena}"
IMAGE_NAME="${CLOUD_RUN_IMAGE_NAME:-llm-arena-ui}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPOSITORY}/${IMAGE_NAME}:latest"

RUN_SERVICE_ACCOUNT="${RUN_SERVICE_ACCOUNT:-${PROJECT}@appspot.gserviceaccount.com}"
# UI 접근 제어는 앱(OAuth) 레이어에서 처리하고, Cloud Run IAM 403은 기본으로 피한다.
ALLOW_UNAUTHENTICATED="${ALLOW_UNAUTHENTICATED:-true}"
UI_AUTH_ENABLED="${ARENA_UI_AUTH_ENABLED:-true}"
UI_SETTINGS_ENABLED="${ARENA_UI_SETTINGS_ENABLED:-true}"
UI_CLIENT_ID_SECRET_NAME="${UI_CLIENT_ID_SECRET_NAME:-GCID}"
UI_CLIENT_SECRET_SECRET_NAME="${UI_CLIENT_SECRET_SECRET_NAME:-GCPASS}"
UI_SESSION_SECRET_SECRET_NAME="${UI_SESSION_SECRET_SECRET_NAME:-ARENA_UI_SESSION_SECRET}"
UI_CLIENT_ID="${GOOGLE_OAUTH_CLIENT_ID:-}"
UI_CLIENT_SECRET="${GOOGLE_OAUTH_CLIENT_SECRET:-}"
UI_SESSION_SECRET="${ARENA_UI_SESSION_SECRET:-}"
DISTRIBUTION_MODE="${ARENA_DISTRIBUTION_MODE:-simulated_only}"
OPERATOR_EMAILS="${ARENA_OPERATOR_EMAILS:-}"
PUBLIC_DEMO_TENANT="${ARENA_PUBLIC_DEMO_TENANT:-}"
SHOWCASE_TENANT="${ARENA_SHOWCASE_TENANT:-}"
SHARED_RESEARCH_GEMINI_SOURCE_TENANT="${ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT:-}"
UI_MIN_INSTANCES="${CLOUD_RUN_UI_MIN_INSTANCES:-${CLOUD_RUN_MIN_INSTANCES:-0}}"

if ! [[ "${UI_MIN_INSTANCES}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CLOUD_RUN_UI_MIN_INSTANCES must be a non-negative integer."
  exit 1
fi

cd "${ROOT_DIR}"

_read_secret_latest() {
  local secret_name="$1"
  if [[ -z "${secret_name}" ]]; then
    return 0
  fi
  gcloud secrets versions access latest --secret="${secret_name}" --project="${PROJECT}" 2>/dev/null || true
}

_is_insecure_ui_session_secret() {
  local value="$1"
  [[ "${value}" == "local-dev-change-this-session-secret" || "${value}" == "dev-only-change-me" ]]
}

_generate_session_secret() {
  python - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
}

_upsert_secret_latest() {
  local secret_name="$1"
  local secret_value="$2"
  if [[ -z "${secret_name}" || -z "${secret_value}" ]]; then
    return 1
  fi
  if ! gcloud secrets describe "${secret_name}" --project "${PROJECT}" >/dev/null 2>&1; then
    gcloud secrets create "${secret_name}" \
      --project "${PROJECT}" \
      --replication-policy "automatic" >/dev/null
  fi
  printf '%s' "${secret_value}" | gcloud secrets versions add "${secret_name}" \
    --project "${PROJECT}" \
    --data-file=- >/dev/null
}

echo "[1/5] Enable required APIs"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com

echo "[2/5] Resolve UI runtime config"
CURRENT_SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --project "${PROJECT}" --format='value(status.url)' 2>/dev/null || true)"

# Ignore checked-in/dev defaults so ship.sh all can still fall back to Secret Manager.
if _is_insecure_ui_session_secret "${UI_SESSION_SECRET}"; then
  echo "WARN: ignoring insecure local ARENA_UI_SESSION_SECRET and trying Secret Manager"
  UI_SESSION_SECRET=""
fi

# Fallback: read from Secret Manager if env vars are empty.
if [[ -z "${UI_CLIENT_ID}" ]]; then
  UI_CLIENT_ID="$(_read_secret_latest "${UI_CLIENT_ID_SECRET_NAME}")"
fi
if [[ -z "${UI_CLIENT_SECRET}" ]]; then
  UI_CLIENT_SECRET="$(_read_secret_latest "${UI_CLIENT_SECRET_SECRET_NAME}")"
fi
if [[ -z "${UI_SESSION_SECRET}" ]]; then
  UI_SESSION_SECRET="$(_read_secret_latest "${UI_SESSION_SECRET_SECRET_NAME}")"
fi

if [[ -z "${UI_SESSION_SECRET}" ]] || _is_insecure_ui_session_secret "${UI_SESSION_SECRET}"; then
  echo "WARN: UI session secret missing or insecure; provisioning Secret Manager secret ${UI_SESSION_SECRET_SECRET_NAME}"
  UI_SESSION_SECRET="$(_generate_session_secret)"
  _upsert_secret_latest "${UI_SESSION_SECRET_SECRET_NAME}" "${UI_SESSION_SECRET}"
fi

if [[ -z "${UI_SESSION_SECRET}" ]]; then
  echo "ERROR: ARENA_UI_SESSION_SECRET is required. Set it directly or create Secret Manager secret ${UI_SESSION_SECRET_SECRET_NAME}."
  exit 1
fi
if _is_insecure_ui_session_secret "${UI_SESSION_SECRET}"; then
  echo "ERROR: Refusing to deploy with insecure ARENA_UI_SESSION_SECRET default."
  exit 1
fi

UI_REDIRECT_URI="${ARENA_UI_GOOGLE_REDIRECT_URI:-}"
if [[ -z "${UI_REDIRECT_URI}" ]]; then
  if [[ -n "${CURRENT_SERVICE_URL}" ]]; then
    UI_REDIRECT_URI="${CURRENT_SERVICE_URL}/auth/google/callback"
  elif [[ "${UI_AUTH_ENABLED,,}" == "true" ]]; then
    echo "ERROR: ARENA_UI_GOOGLE_REDIRECT_URI is required for first deploy when UI auth is enabled."
    exit 1
  fi
fi
if [[ -n "${CURRENT_SERVICE_URL}" ]] && [[ "${UI_REDIRECT_URI}" =~ ^http://(127\.0\.0\.1|localhost)(:[0-9]+)?/ ]]; then
  echo "WARN: local OAuth redirect URI detected for Cloud Run deploy; overriding to service callback URL."
  UI_REDIRECT_URI="${CURRENT_SERVICE_URL}/auth/google/callback"
fi
if [[ "${UI_AUTH_ENABLED,,}" == "true" ]] && [[ -z "${UI_CLIENT_ID}" || -z "${UI_CLIENT_SECRET}" ]]; then
  echo "ERROR: UI auth is enabled but GOOGLE_OAUTH_CLIENT_ID/GOOGLE_OAUTH_CLIENT_SECRET is empty."
  exit 1
fi

# Minimal env needed for BigQuery access + UI auth/settings defaults.
RUN_ENV_VARS="${RUN_ENV_VARS:-^||^GOOGLE_CLOUD_PROJECT=${PROJECT}||BQ_DATASET=llm_arena||BQ_LOCATION=${REGION}||ARENA_LOG_LEVEL=INFO||ARENA_LOG_FORMAT=rich||ARENA_TRADING_MODE=live||ARENA_DISTRIBUTION_MODE=${DISTRIBUTION_MODE}||ARENA_AGENT_IDS=gemini,gpt,claude||KIS_TARGET_MARKET=us||ARENA_UI_AUTH_ENABLED=${UI_AUTH_ENABLED}||ARENA_UI_SETTINGS_ENABLED=${UI_SETTINGS_ENABLED}||GOOGLE_OAUTH_CLIENT_ID=${UI_CLIENT_ID}||GOOGLE_OAUTH_CLIENT_SECRET=${UI_CLIENT_SECRET}||ARENA_UI_GOOGLE_REDIRECT_URI=${UI_REDIRECT_URI}||ARENA_UI_SESSION_SECRET=${UI_SESSION_SECRET}||ARENA_OPERATOR_EMAILS=${OPERATOR_EMAILS}||ARENA_PUBLIC_DEMO_TENANT=${PUBLIC_DEMO_TENANT}||ARENA_SHOWCASE_TENANT=${SHOWCASE_TENANT}||ARENA_SHARED_RESEARCH_GEMINI_SOURCE_TENANT=${SHARED_RESEARCH_GEMINI_SOURCE_TENANT}}"

echo "[3/5] Ensure Artifact Registry"
if ! gcloud artifacts repositories describe "${AR_REPOSITORY}" --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPOSITORY}" \
    --repository-format=docker \
    --location "${REGION}" \
    --description "LLM Arena images" \
    --project "${PROJECT}"
fi

SKIP_BUILD="${SKIP_BUILD:-false}"
if [[ "${SKIP_BUILD,,}" == "true" ]]; then
  echo "[4/5] Skip build (SKIP_BUILD=true)"
else
  echo "[4/5] Build container image (UI)"
  gcloud builds submit \
    --tag "${IMAGE}" \
    --project "${PROJECT}" \
    --timeout 600s \
    .
  fi

echo "[5/5] Deploy Cloud Run Service"
DEPLOY_CMD=(
  gcloud run deploy "${SERVICE_NAME}"
  --image "${IMAGE}"
  --region "${REGION}"
  --project "${PROJECT}"
  --service-account "${RUN_SERVICE_ACCOUNT}"
  --update-env-vars "${RUN_ENV_VARS}"
  --port 8080
  --min-instances "${UI_MIN_INSTANCES}"
  --memory 512Mi
  --cpu 1
  --concurrency 80
  --timeout 60
)

if [[ "${ALLOW_UNAUTHENTICATED,,}" == "true" ]]; then
  DEPLOY_CMD+=(--allow-unauthenticated)
else
  DEPLOY_CMD+=(--no-allow-unauthenticated)
fi

"${DEPLOY_CMD[@]}"


SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --project "${PROJECT}" --format='value(status.url)')"

echo "Done"
echo "Cloud Run Service: ${SERVICE_NAME}"
echo "Image: ${IMAGE}"
echo "Service Account: ${RUN_SERVICE_ACCOUNT}"
echo "Min instances: ${UI_MIN_INSTANCES}"
echo "URL: ${SERVICE_URL}"
