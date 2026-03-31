#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export ARENA_UI_AUTH_ENABLED=false
export ARENA_UI_SETTINGS_ENABLED=true
unset GOOGLE_OAUTH_CLIENT_ID GOOGLE_OAUTH_CLIENT_SECRET ARENA_UI_GOOGLE_REDIRECT_URI

echo "Starting local UI on http://127.0.0.1:8080/"
exec python -m arena.cli serve-ui
