from __future__ import annotations

from types import SimpleNamespace

import pytest

import arena.cli as cli
from arena.config import load_settings
from arena.models import AccountSnapshot


class _FakeReconResult:
    def __init__(self, *, ok: bool, snapshot=None):
        self.ok = ok
        self.status = "ok" if ok else "failed"
        self.issues = [SimpleNamespace(issue_type="position_quantity_mismatch", entity_key="AAPL")] if not ok else []
        self.recoveries = []
        self.account_snapshot = snapshot


class _FakeRecoveryResult:
    def __init__(self, *, ok: bool, snapshot=None):
        self.ok = ok
        self.status = "recovered" if ok else "failed"
        self.applied_checkpoints = 1 if ok else 0
        self.before = _FakeReconResult(ok=False)
        self.after = _FakeReconResult(ok=ok, snapshot=snapshot)


def test_reconciliation_guard_returns_snapshot_when_ok(monkeypatch) -> None:
    settings = load_settings()
    snapshot = AccountSnapshot(cash_krw=1.0, total_equity_krw=1.0, positions={})

    class _FakeService:
        def __init__(self, **kwargs):
            _ = kwargs

        def reconcile_positions(self, **kwargs):
            _ = kwargs
            return _FakeReconResult(ok=True, snapshot=snapshot)

    monkeypatch.setattr(cli, "StateReconciliationService", _FakeService)
    monkeypatch.setenv("ARENA_RECONCILIATION_ENABLED", "true")

    out = cli._run_reconciliation_guard(
        live=True,
        settings=settings,
        repo=object(),
        orchestrator=SimpleNamespace(agents=[SimpleNamespace(agent_id="gpt")]),
        tenant="local",
        snapshot=None,
    )

    assert out is snapshot


def test_reconciliation_guard_fails_closed(monkeypatch) -> None:
    settings = load_settings()

    class _FakeService:
        def __init__(self, **kwargs):
            _ = kwargs

        def reconcile_positions(self, **kwargs):
            _ = kwargs
            return _FakeReconResult(ok=False)

    monkeypatch.setattr(cli, "StateReconciliationService", _FakeService)
    monkeypatch.setenv("ARENA_RECONCILIATION_ENABLED", "true")
    monkeypatch.delenv("ARENA_RECONCILE_FAIL_CLOSED", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli._run_reconciliation_guard(
            live=True,
            settings=settings,
            repo=object(),
            orchestrator=SimpleNamespace(agents=[SimpleNamespace(agent_id="gpt")]),
            tenant="local",
            snapshot=None,
            auto_recover=False,
        )

    assert exc_info.value.code == 3


def test_reconciliation_guard_uses_recovery_before_failing(monkeypatch) -> None:
    settings = load_settings()
    snapshot = AccountSnapshot(cash_krw=1.0, total_equity_krw=1.0, positions={})
    calls: list[dict] = []

    class _FakeReconService:
        def __init__(self, **kwargs):
            _ = kwargs

        def reconcile_positions(self, **kwargs):
            _ = kwargs
            return _FakeReconResult(ok=False, snapshot=snapshot)

    class _FakeRecoveryService:
        def __init__(self, **kwargs):
            _ = kwargs

        def recover_and_reconcile(self, **kwargs):
            calls.append(dict(kwargs))
            return _FakeRecoveryResult(ok=True, snapshot=snapshot)

    monkeypatch.setattr(cli, "StateReconciliationService", _FakeReconService)
    monkeypatch.setattr(cli, "StateRecoveryService", _FakeRecoveryService)
    monkeypatch.setenv("ARENA_RECONCILIATION_ENABLED", "true")
    monkeypatch.setenv("ARENA_RECONCILIATION_RECOVER_ENABLED", "true")

    out = cli._run_reconciliation_guard(
        live=True,
        settings=settings,
        repo=object(),
        orchestrator=SimpleNamespace(agents=[SimpleNamespace(agent_id="gpt")]),
        tenant="local",
        snapshot=snapshot,
        auto_recover=True,
    )

    assert out is snapshot
    assert calls
    assert calls[0]["allow_checkpoint_rebuild"] is False


def test_reconciliation_guard_passes_tolerance_envs_to_services(monkeypatch) -> None:
    settings = load_settings()
    recon_inits: list[dict] = []
    recovery_inits: list[dict] = []

    class _FakeReconService:
        def __init__(self, **kwargs):
            recon_inits.append(dict(kwargs))

        def reconcile_positions(self, **kwargs):
            _ = kwargs
            return _FakeReconResult(ok=False)

    class _FakeRecoveryService:
        def __init__(self, **kwargs):
            recovery_inits.append(dict(kwargs))

        def recover_and_reconcile(self, **kwargs):
            _ = kwargs
            return _FakeRecoveryResult(ok=True)

    monkeypatch.setattr(cli, "StateReconciliationService", _FakeReconService)
    monkeypatch.setattr(cli, "StateRecoveryService", _FakeRecoveryService)
    monkeypatch.setenv("ARENA_RECONCILIATION_ENABLED", "true")
    monkeypatch.setenv("ARENA_RECONCILIATION_RECOVER_ENABLED", "true")
    monkeypatch.setenv("ARENA_RECONCILE_QTY_TOLERANCE", "0.25")
    monkeypatch.setenv("ARENA_RECONCILE_CASH_TOLERANCE_KRW", "12345")
    monkeypatch.setenv("ARENA_RECONCILE_CASH_ENABLED", "true")

    cli._run_reconciliation_guard(
        live=True,
        settings=settings,
        repo=object(),
        orchestrator=SimpleNamespace(agents=[SimpleNamespace(agent_id="gpt")]),
        tenant="local",
        snapshot=None,
        auto_recover=True,
    )

    assert recon_inits
    assert recon_inits[0]["qty_tolerance"] == pytest.approx(0.25)
    assert recon_inits[0]["cash_tolerance_krw"] == pytest.approx(12345.0)
    assert recon_inits[0]["cash_reconciliation_enabled"] is True
    assert recovery_inits
    assert recovery_inits[0]["qty_tolerance"] == pytest.approx(0.25)
    assert recovery_inits[0]["cash_tolerance_krw"] == pytest.approx(12345.0)
    assert recovery_inits[0]["cash_reconciliation_enabled"] is True


def test_cmd_recover_sleeves_runs_recovery_service(monkeypatch) -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    orchestrator = SimpleNamespace(agents=[SimpleNamespace(agent_id="gpt")])
    repo = object()
    calls: list[dict] = []

    class _FakeRecoveryService:
        def __init__(self, **kwargs):
            _ = kwargs

        def recover_and_reconcile(self, **kwargs):
            calls.append(dict(kwargs))
            return _FakeRecoveryResult(ok=True, snapshot=AccountSnapshot(cash_krw=1.0, total_equity_krw=1.0, positions={}))

    monkeypatch.setattr(cli, "_build_runtime", lambda **kwargs: (settings, repo, orchestrator))
    monkeypatch.setattr(cli, "StateRecoveryService", _FakeRecoveryService)
    monkeypatch.setattr(cli, "_tenant_id", lambda: "local")

    cli.cmd_recover_sleeves(live=False)

    assert calls
    assert calls[0]["agent_ids"] == ["gpt"]
    assert calls[0]["tenant_id"] == "local"
    assert calls[0]["allow_checkpoint_rebuild"] is True
