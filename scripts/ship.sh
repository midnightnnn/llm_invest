#!/usr/bin/env bash
# ship.sh — Build, push, deploy in one command.
#
# Usage:
#   ./scripts/ship.sh              # backend split jobs (prep + agent) only
#   ./scripts/ship.sh ui           # UI (Cloud Run Service) only
#   ./scripts/ship.sh all          # both
#   ./scripts/ship.sh job          # backend split jobs only (explicit)
#   A_SPLIT_JOBS=false ./scripts/ship.sh job   # legacy single-pipeline jobs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Auto-load .env so API keys (FRED, ECOS, etc.) are available to deploy scripts
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

PROJECT="${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT env var}"
REGION="${CLOUD_RUN_REGION:-asia-northeast3}"
AR_REPOSITORY="${AR_REPOSITORY:-llm-arena}"

TARGET="${1:-job}"
A_SPLIT_JOBS="${A_SPLIT_JOBS:-true}"
JOB_IMAGE_NAME="${CLOUD_RUN_JOB_NAME:-llm-arena-batch}"
UI_IMAGE_NAME="${CLOUD_RUN_IMAGE_NAME:-llm-arena-ui}"

_build_and_push() {
  local image_name="$1"
  local image="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPOSITORY}/${image_name}:latest"
  echo "=== Build & Push: ${image} ==="
  docker build --platform linux/amd64 -t "${image}" "${SCRIPT_DIR}/.."
  docker push "${image}"
  echo "=== Pushed: ${image} ==="
}

_deploy_job() {
  _build_and_push "${JOB_IMAGE_NAME}"
  echo "=== Deploy Cloud Run Job (A_SPLIT_JOBS=${A_SPLIT_JOBS}) ==="
  A_SPLIT_JOBS="${A_SPLIT_JOBS}" SKIP_BUILD=true bash "${SCRIPT_DIR}/deploy_cloud_run_job.sh"
}

_deploy_ui() {
  _build_and_push "${UI_IMAGE_NAME}"
  echo "=== Deploy Cloud Run UI ==="
  SKIP_BUILD=true bash "${SCRIPT_DIR}/deploy_cloud_run_ui.sh"
}

case "${TARGET}" in
  job|backend)
    _deploy_job
    ;;
  ui)
    _deploy_ui
    ;;
  all)
    _deploy_job
    _deploy_ui
    ;;
  *)
    echo "Usage: $0 {job|ui|all}"
    exit 1
    ;;
esac

echo ""
echo "Done."
