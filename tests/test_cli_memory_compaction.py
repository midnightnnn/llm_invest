from __future__ import annotations

import json

import pytest

import arena.cli as cli
from arena.config import load_settings


class _FakeRepo:
    project = "proj"
    location = "asia-northeast3"

    def __init__(self) -> None:
        self.status_rows: list[dict] = []
        self.latest_cycle_args: dict | None = None

    def ensure_dataset(self) -> None:
        pass

    def ensure_tables(self) -> None:
        pass

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return str(tenant_id or "tenant-a").strip().lower()

    def latest_runtime_credentials(self, *, tenant_id: str) -> dict:
        return {"model_secret_name": "model-secret"}

    def latest_config_values(self, *, config_key: str, tenant_ids: list[str] | None = None) -> dict[str, str]:
        _ = config_key
        return {str(tenant_id).strip().lower(): "us" for tenant_id in (tenant_ids or [])}

    def latest_memory_compaction_cycle_id(self, **kwargs) -> str:
        self.latest_cycle_args = dict(kwargs)
        return "cycle_latest"

    def append_tenant_run_status(self, **kwargs) -> None:
        self.status_rows.append(dict(kwargs))


def test_build_parser_supports_run_memory_compaction_command() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "run-memory-compaction",
            "--live",
            "--tenant",
            "tenant-a",
            "--market",
            "us",
            "--cycle-id",
            "latest",
            "--agent",
            "gpt",
            "--dry-run",
        ]
    )

    assert args.command == "run-memory-compaction"
    assert args.live is True
    assert args.tenant == ["tenant-a"]
    assert args.market == "us"
    assert args.cycle_id == "latest"
    assert args.agent == ["gpt"]
    assert args.dry_run is True


def test_run_memory_compaction_resolves_latest_cycle_and_uses_tenant_credentials(monkeypatch, capsys) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj"
    settings.openai_api_key = "env-key-that-should-be-cleared"
    settings.agent_ids = ["gpt"]
    repo = _FakeRepo()

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "_apply_market_override", lambda settings, market: setattr(settings, "kis_target_market", market or settings.kis_target_market))
    monkeypatch.setattr(cli, "apply_runtime_overrides", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_new_run_id", lambda prefix: f"{prefix}_1")
    monkeypatch.setattr(cli, "utc_now", lambda: "now")

    def _apply_runtime_credentials(settings, repo, *, tenant_id=None):
        settings.openai_api_key = "tenant-openai-key"
        return {"model_secret_name": "model-secret"}

    monkeypatch.setattr(cli, "_apply_tenant_runtime_credentials", _apply_runtime_credentials)

    class _FakeCompactor:
        def __init__(self, *, settings, repo, memory_store):
            assert settings.openai_api_key == "tenant-openai-key"

        async def preview(self, *, cycle_id, agent_ids):
            assert cycle_id == "cycle_latest"
            assert agent_ids == ["gpt"]
            return [{"agent_id": "gpt", "cycle_id": cycle_id, "reflection_count": 1, "reflections": []}]

    monkeypatch.setattr("arena.agents.memory_compaction_agent.MemoryCompactionAgent", _FakeCompactor)

    cli.cmd_run_memory_compaction(
        live=True,
        tenant_ids=["tenant-a"],
        cycle_id="latest",
        market_override="us",
        agent_ids=["gpt"],
        dry_run=True,
    )

    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "success"
    assert out["results"][0]["cycle_id"] == "cycle_latest"
    assert repo.latest_cycle_args["agent_ids"] == ["gpt"]
    assert repo.latest_cycle_args["trading_mode"] == "live"
    assert repo.status_rows[0]["run_type"] == "memory_compaction"


def test_run_memory_compaction_raises_on_tenant_failure(monkeypatch, capsys) -> None:
    settings = load_settings()
    settings.google_cloud_project = "proj"
    repo = _FakeRepo()

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_validate_or_exit", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "_repo_or_exit", lambda settings, tenant_id=None: repo)
    monkeypatch.setattr(cli, "_apply_tenant_runtime_credentials", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_run_memory_compaction(live=True, tenant_ids=["tenant-a"], dry_run=True)

    assert exc_info.value.code == 1
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "failed"
    assert out["failure_count"] == 1
