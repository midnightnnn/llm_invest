# Test Quality Foundation Design

## Goal

Raise the test suite toward the structure and maintainability level seen in `adk-python` by adding reusable test infrastructure first, without changing production behavior or doing a broad rewrite of existing tests.

## Scope

This first phase builds the foundation only:

- Add a `tests/helpers/` package for reusable test doubles and fixture builders.
- Move a small set of repeated fake BigQuery/session/repository helpers into that package.
- Expand pytest configuration with explicit test markers.
- Upgrade `tests/conftest.py` from import-path setup only to shared test-suite setup.
- Convert one or two low-risk test files to use the new helpers as proof of the pattern.

This phase does not split the largest test files, rewrite the full suite, or add live integration behavior.

## Current Problems

The current test suite has strong domain regression coverage, but the support structure is thin:

- `tests/conftest.py` only inserts the repo root into `sys.path`.
- Repeated fake objects are defined independently across test files.
- `pyproject.toml` defines only the `integration` marker.
- There is no central place for reusable tenant, BigQuery, runtime config, or settings test helpers.
- Large files such as `tests/test_adk_agents.py`, `tests/test_cli_multi_tenant.py`, and `tests/test_ui_admin_routes.py` mix many responsibilities.

## Design

Create a focused helper package under `tests/helpers/`.

Initial modules:

- `tests/helpers/bigquery.py`
  - `FakeInsertClient`
  - `FakeBigQuerySession`
  - `FakeLoadJob`
  - small schema/table test doubles used by store tests
- `tests/helpers/repos.py`
  - lightweight fake repository objects for runtime config and tenant-oriented tests
  - call recording helpers for assertions
- `tests/helpers/settings.py`
  - deterministic settings builders for tests that currently mutate `load_settings()` results directly
- `tests/helpers/__init__.py`
  - intentionally small exports, so tests can import explicit helpers from concrete modules

The helper package should stay test-only and must not be imported by production code.

## Pytest Configuration

Keep `testpaths = ["tests"]` and `pythonpath = ["."]`.

Add explicit markers:

- `unit`: fast isolated tests with no external service dependency
- `integration`: local integration tests or opt-in service-bound tests
- `live`: tests requiring real external credentials or networked services
- `slow`: tests that are deterministic but too slow for default focused runs

No marker filtering is introduced in this phase. The change only documents marker names and prevents unknown-marker drift.

## Conftest Behavior

`tests/conftest.py` should keep its current root path setup.

It may also expose small, broadly useful fixtures such as:

- `fixed_utc_now`
- `tenant_id`
- `fake_bq_session_factory`

Fixtures should be conservative. Domain-specific helpers should live in `tests/helpers/` rather than crowding `conftest.py`.

## Migration Strategy

Start with low-risk files that already define local fake BigQuery/session objects and have narrow scope.

Preferred first candidates:

- `tests/test_ledger_repo.py`
- `tests/test_memory_bq_store.py`
- `tests/test_execution_store.py`

These are better first candidates than `tests/test_adk_agents.py` because they are smaller and exercise the same helper shape needed later.

After the helper pattern is validated, larger files can be split in later phases:

- `tests/test_adk_agents.py` into ADK model, prompt, runner, decision flow, and order support tests
- `tests/test_ui_admin_routes.py` into settings, agents, memory, board, auth, and chart API tests
- `tests/test_cli_multi_tenant.py` into tenant resolution, credential application, runtime build, and batch cycle tests

## Success Criteria

This phase is successful when:

- New helper modules exist and are imported by at least one converted test file.
- Converted tests preserve the existing assertions and behavior.
- Pytest marker names are explicit in `pyproject.toml`.
- Focused tests for converted files pass.
- The change does not modify production behavior.

## Risks And Mitigations

- Risk: helper abstractions become too broad.
  - Mitigation: keep helpers small and move only duplicated mechanics, not test-specific assertions.
- Risk: changing shared test utilities breaks many tests.
  - Mitigation: migrate only one or two files in this phase.
- Risk: dirty working tree contains unrelated user changes.
  - Mitigation: touch only test infrastructure files and selected small test files; do not revert or restage unrelated changes.

## Out Of Scope

- Full test-suite reorganization.
- CI workflow creation.
- Coverage thresholds.
- Live integration test expansion.
- Large-file splitting.
- Production code changes.
