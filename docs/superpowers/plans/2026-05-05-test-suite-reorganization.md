# Test Suite Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the full test suite closer to the `adk-python` style by replacing flat monolith tests with domain-scoped test packages and shared helpers.

**Architecture:** Keep behavior unchanged while moving large test files into focused packages. Start with the UI suite because it has cross-file helper coupling, then apply the same pattern to CLI, data/trading, and memory/reconciliation suites. Each task must preserve the current full-suite count and keep warning output clean.

**Tech Stack:** Python 3.12, pytest, FastAPI `TestClient`, existing `tests.direct_route_client.DirectRouteClient`, existing `tests/helpers/` package.

---

## File Structure

- Create `tests/ui/`
  - Owns UI route, page, investment-chat, settings, ops, and API boundary tests.
- Create `tests/ui/helpers.py`
  - Owns `_DummyRepo`, `_client`, `_client_with_repo`, and credential-store client builders currently embedded in `tests/test_ui_admin_routes.py`.
- Move `tests/test_ui_api_error_boundary.py` to `tests/ui/test_api_error_boundary.py`
  - Replaces imports from `tests.test_ui_admin_routes` with `tests.ui.helpers`.
- Split `tests/test_ui_admin_routes.py`
  - Target modules: `tests/ui/test_settings_routes.py`, `tests/ui/test_admin_agent_routes.py`, `tests/ui/test_memory_routes.py`, `tests/ui/test_board_routes.py`, `tests/ui/test_auth_routes.py`, `tests/ui/test_chart_routes.py`.
- Split `tests/test_investment_chat_ui.py`
  - Target modules: `tests/ui/test_investment_chat_factory.py`, `tests/ui/test_investment_chat_tools.py`, `tests/ui/test_investment_chat_pages.py`, `tests/ui/test_investment_chat_order_drafts.py`, `tests/ui/test_investment_chat_adk_auth.py`.
- Move `tests/test_ops_page.py` to `tests/ui/test_ops_page.py`.
- Later phases:
  - `tests/cli/` for `test_cli_multi_tenant.py` splits.
  - `tests/data/` for strict BigQuery/local repository path tests.
  - `tests/trading/` for broker/sync/dividend/open-trading tests.
  - `tests/memory/` for memory store, compaction, semantic relation, and reconciliation-adjacent helpers.

---

### Task 1: Extract Shared UI Test Helpers

**Files:**
- Create: `tests/ui/__init__.py`
- Create: `tests/ui/helpers.py`
- Modify: `tests/test_ui_admin_routes.py`
- Modify: `tests/test_investment_chat_ui.py`
- Modify: `tests/test_ui_api_error_boundary.py`

- [ ] **Step 1: Create the UI helper package**

Move the `_DummyRepo`, `_client`, `_client_with_repo`, and `_client_with_repo_and_credential_store` definitions from `tests/test_ui_admin_routes.py` into `tests/ui/helpers.py`. Keep method names and return values unchanged.

- [ ] **Step 2: Update imports**

Use:

```python
from tests.ui.helpers import (
    _DummyRepo,
    _client,
    _client_with_repo,
    _client_with_repo_and_credential_store,
)
```

Only import the names each test file actually uses.

- [ ] **Step 3: Run focused UI helper consumers**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  tests/test_ui_admin_routes.py \
  tests/test_investment_chat_ui.py \
  tests/test_ui_api_error_boundary.py \
  -q -p no:cacheprovider
```

Expected: all selected tests pass with no warning summary.

- [ ] **Step 4: Commit**

```bash
git add tests/ui/__init__.py tests/ui/helpers.py tests/test_ui_admin_routes.py tests/test_investment_chat_ui.py tests/test_ui_api_error_boundary.py
git commit -m "test: extract shared ui route helpers"
```

---

### Task 2: Move API Boundary And Ops UI Tests

**Files:**
- Move: `tests/test_ui_api_error_boundary.py` to `tests/ui/test_api_error_boundary.py`
- Move: `tests/test_ops_page.py` to `tests/ui/test_ops_page.py`

- [ ] **Step 1: Move files with `git mv`**

```bash
git mv tests/test_ui_api_error_boundary.py tests/ui/test_api_error_boundary.py
git mv tests/test_ops_page.py tests/ui/test_ops_page.py
```

- [ ] **Step 2: Run moved tests**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/ui/test_api_error_boundary.py tests/ui/test_ops_page.py -q -p no:cacheprovider
```

Expected: both files pass with no warning summary.

- [ ] **Step 3: Commit**

```bash
git add tests/ui/test_api_error_boundary.py tests/ui/test_ops_page.py
git commit -m "test: move focused ui route tests"
```

---

### Task 3: Split Admin Route Tests

**Files:**
- Modify: `tests/test_ui_admin_routes.py`
- Create: `tests/ui/test_settings_routes.py`
- Create: `tests/ui/test_admin_agent_routes.py`
- Create: `tests/ui/test_memory_routes.py`
- Create: `tests/ui/test_board_routes.py`
- Create: `tests/ui/test_auth_routes.py`
- Create: `tests/ui/test_chart_routes.py`

- [ ] **Step 1: Move settings tests**

Move tests whose names start with `test_settings_` or `test_admin_routes_save_config` into `tests/ui/test_settings_routes.py`.

- [ ] **Step 2: Move agent admin tests**

Move tests whose names start with `test_admin_agent`, `test_admin_agents`, or `test_admin_sleeve` into `tests/ui/test_admin_agent_routes.py`.

- [ ] **Step 3: Move memory route tests**

Move tests whose names start with `test_memory_` or `test_api_memory_` into `tests/ui/test_memory_routes.py`.

- [ ] **Step 4: Move board/nav tests**

Move tests whose names start with `test_api_board`, `test_board`, `test_showcase_board`, `test_api_nav`, or `test_nav_page` into `tests/ui/test_board_routes.py`.

- [ ] **Step 5: Move auth tests**

Move tests whose names start with `test_layout_shows_auth`, `test_auth_`, or include `viewer_only_user` into `tests/ui/test_auth_routes.py`.

- [ ] **Step 6: Move chart/card tests**

Move sleeve snapshot, tool frequency, and capital waterfall API/page tests into `tests/ui/test_chart_routes.py`.

- [ ] **Step 7: Run admin route suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/ui tests/test_ui_admin_routes.py -q -p no:cacheprovider
```

Expected: UI tests pass with no warning summary.

- [ ] **Step 8: Commit**

```bash
git add tests/ui tests/test_ui_admin_routes.py
git commit -m "test: split ui admin route tests"
```

---

### Task 4: Split Investment Chat UI Tests

**Files:**
- Modify: `tests/test_investment_chat_ui.py`
- Create: `tests/ui/test_investment_chat_factory.py`
- Create: `tests/ui/test_investment_chat_tools.py`
- Create: `tests/ui/test_investment_chat_pages.py`
- Create: `tests/ui/test_investment_chat_order_drafts.py`
- Create: `tests/ui/test_investment_chat_adk_auth.py`

- [ ] **Step 1: Move factory/layout tests**

Move factory, layout, loader, and model selection tests into `tests/ui/test_investment_chat_factory.py`.

- [ ] **Step 2: Move tool schema and execution tests**

Move account, sleeve, order, config, and refresh tool tests into `tests/ui/test_investment_chat_tools.py`.

- [ ] **Step 3: Move page rendering tests**

Move route/page rendering tests into `tests/ui/test_investment_chat_pages.py`.

- [ ] **Step 4: Move order draft API tests**

Move order/config draft API and approval status tests into `tests/ui/test_investment_chat_order_drafts.py`.

- [ ] **Step 5: Move ADK auth/middleware tests**

Move ADK auth, stale app name, and session default tests into `tests/ui/test_investment_chat_adk_auth.py`.

- [ ] **Step 6: Run investment chat UI suite**

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/ui tests/test_investment_chat_ui.py -q -p no:cacheprovider
```

Expected: UI tests pass with no warning summary.

- [ ] **Step 7: Commit**

```bash
git add tests/ui tests/test_investment_chat_ui.py
git commit -m "test: split investment chat ui tests"
```

---

### Task 5: Repeat Pattern For CLI And Data Suites

**Files:**
- Create: `tests/cli/`
- Create: `tests/data/`
- Create: `tests/trading/`
- Create: `tests/memory/`

- [ ] **Step 1: Split `tests/test_cli_multi_tenant.py` by behavior**

Create modules for tenant resolution, credential materialization, runtime build, batch cycle, and failure handling. Keep shared fake tenant repos in `tests/helpers/repos.py` or `tests/cli/helpers.py`.

- [ ] **Step 2: Split data/trading tests by subsystem**

Move strict path tests into `tests/data/`, broker/sync/open trading tests into `tests/trading/`, and memory tests into `tests/memory/`.

- [ ] **Step 3: Verify after every slice**

Run the moved focused tests first, then:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q -p no:cacheprovider
```

Expected: full suite passes with no warning summary.
