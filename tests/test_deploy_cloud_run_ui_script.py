from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ui_deploy_script_runs_schema_ensure_before_deploy() -> None:
    script = (ROOT / "scripts" / "deploy_cloud_run_ui.sh").read_text()

    assert "UI_ENSURE_SCHEMA_BEFORE_DEPLOY" in script
    assert "init-bq" in script
    assert "Ensure BigQuery schema" in script
    assert script.index("Ensure BigQuery schema") < script.index("Deploy Cloud Run Service")


def test_ui_deploy_script_disables_runtime_schema_ensure_and_uses_configurable_memory() -> None:
    script = (ROOT / "scripts" / "deploy_cloud_run_ui.sh").read_text()

    assert "ARENA_UI_ENSURE_SCHEMA_ON_STARTUP=${UI_ENSURE_SCHEMA_ON_STARTUP}" in script
    assert 'UI_ENSURE_SCHEMA_ON_STARTUP="${ARENA_UI_ENSURE_SCHEMA_ON_STARTUP:-false}"' in script
    assert 'UI_MEMORY="${CLOUD_RUN_UI_MEMORY:-1Gi}"' in script
    assert '--memory "${UI_MEMORY}"' in script
