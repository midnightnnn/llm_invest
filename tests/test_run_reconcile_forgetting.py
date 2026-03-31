from __future__ import annotations

from types import SimpleNamespace

import arena.cli_commands.run_reconcile as run_reconcile_module


def test_run_memory_forgetting_tuner_post_cycle_calls_tuner(monkeypatch) -> None:
    calls: list[tuple[object, object, str, str]] = []

    monkeypatch.setattr(
        run_reconcile_module,
        "_cli",
        lambda: SimpleNamespace(_truthy_env=lambda name, default=True: True),
    )

    def _fake_tuner(repo, settings, *, tenant_id, updated_by, **kwargs):
        _ = kwargs
        calls.append((repo, settings, tenant_id, updated_by))
        return {
            "reason": "shadow only",
            "mode": "shadow",
            "effective_mode": "shadow",
            "sample": {"access_events": 12, "prompt_uses": 4, "unique_memories": 3},
            "gates": {"apply_allowed": False},
            "transition": {"action": ""},
        }

    monkeypatch.setattr(run_reconcile_module, "run_memory_forgetting_tuner", _fake_tuner)

    repo = object()
    settings = SimpleNamespace()

    run_reconcile_module._run_memory_forgetting_tuner_post_cycle(
        settings=settings,
        repo=repo,
        tenant="alpha",
    )

    assert calls == [(repo, settings, "alpha", "post-cycle-forgetting-tuner")]


def test_run_memory_forgetting_tuner_post_cycle_skips_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        run_reconcile_module,
        "_cli",
        lambda: SimpleNamespace(_truthy_env=lambda name, default=True: False),
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("tuner should not run")

    monkeypatch.setattr(run_reconcile_module, "run_memory_forgetting_tuner", _should_not_run)

    run_reconcile_module._run_memory_forgetting_tuner_post_cycle(
        settings=SimpleNamespace(),
        repo=object(),
        tenant="alpha",
    )


def test_run_memory_forgetting_tuner_post_cycle_swallows_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        run_reconcile_module,
        "_cli",
        lambda: SimpleNamespace(_truthy_env=lambda name, default=True: True),
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(run_reconcile_module, "run_memory_forgetting_tuner", _boom)

    run_reconcile_module._run_memory_forgetting_tuner_post_cycle(
        settings=SimpleNamespace(),
        repo=object(),
        tenant="alpha",
    )
