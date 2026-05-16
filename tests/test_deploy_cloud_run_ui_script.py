from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ui_deploy_script_runs_schema_ensure_before_deploy() -> None:
    script = (ROOT / "scripts" / "deploy_cloud_run_ui.sh").read_text()

    assert "UI_ENSURE_SCHEMA_BEFORE_DEPLOY" in script
    assert "init-bq" in script
    assert "ARENA_MODE=gcp" in script
    assert "Ensure BigQuery schema" in script
    assert script.index("Ensure BigQuery schema") < script.index("Deploy Cloud Run Service")


def test_ui_deploy_script_disables_runtime_schema_ensure_and_uses_configurable_memory() -> None:
    script = (ROOT / "scripts" / "deploy_cloud_run_ui.sh").read_text()

    assert "ARENA_UI_ENSURE_SCHEMA_ON_STARTUP=${UI_ENSURE_SCHEMA_ON_STARTUP}" in script
    assert 'UI_ENSURE_SCHEMA_ON_STARTUP="${ARENA_UI_ENSURE_SCHEMA_ON_STARTUP:-false}"' in script
    assert 'UI_MEMORY="${CLOUD_RUN_UI_MEMORY:-1Gi}"' in script
    assert '--memory "${UI_MEMORY}"' in script
    assert "--set-env-vars" in script
    assert "--update-env-vars" not in script


def test_ui_deploy_script_defaults_chat_sessions_to_firestore_without_min_instance() -> None:
    script = (ROOT / "scripts" / "deploy_cloud_run_ui.sh").read_text()

    assert 'CHAT_SESSION_SERVICE_URI="${ARENA_CHAT_SESSION_SERVICE_URI:-firestore://arena-investment-chat-adk-sessions}"' in script
    assert "ARENA_CHAT_SESSION_SERVICE_URI=${CHAT_SESSION_SERVICE_URI}" in script
    assert 'UI_MIN_INSTANCES="${CLOUD_RUN_UI_MIN_INSTANCES:-${CLOUD_RUN_MIN_INSTANCES:-0}}"' in script


def test_ui_deploy_script_injects_optional_macro_api_keys_from_env() -> None:
    script = (ROOT / "scripts" / "deploy_cloud_run_ui.sh").read_text()

    assert "_append_run_env_var FRED_API_KEY" in script
    assert "_append_run_env_var ECOS_API_KEY" in script
    assert 'RUN_ENV_VARS="${RUN_ENV_VARS}${ENV_PAIR_DELIM}${name}=${value}"' in script
