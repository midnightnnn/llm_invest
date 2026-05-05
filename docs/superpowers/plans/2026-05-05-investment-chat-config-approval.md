# Investment Chat Config Approval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the investment chat agent propose tenant and agent setting changes, then apply them only after the user clicks an approval button in the investment chat UI.

**Architecture:** Add a typed config-change draft flow alongside the existing order draft flow. The LLM gets a narrow management tool that can create and inspect config drafts, while the backend approval endpoint applies drafts through existing admin domain logic instead of free-form SQL.

**Tech Stack:** FastAPI routes, Jinja2 templates, `arena_config` runtime config storage, existing admin agent config helpers, pytest.

---

### File Structure

- Create `arena/agents/investment_chat/config_tools.py` for typed draft creation, validation, diffing, and apply helpers.
- Modify `arena/agents/investment_chat/tools.py` to expose config management tools to the chat agent.
- Modify `arena/agents/investment_chat/drafts.py` to support order and config draft key namespaces.
- Modify `arena/ui/routes/investment_chat.py` to list and approve config drafts.
- Modify `arena/ui/templates/investment_chat_body.jinja2` to render a settings approval action panel.
- Modify `tests/test_investment_chat_ui.py` for RED/GREEN coverage of tool creation, approval API, and UI polling.

### Task 1: Config Draft Domain

- [ ] **Step 1: Write failing tests**

Add tests in `tests/test_investment_chat_ui.py` that build raw chat tools, call `propose_config_change` for an agent model/capital update, assert no `agents_config` mutation happens before approval, then call the internal apply helper and assert the config is saved and runtime sync is invoked.

- [ ] **Step 2: Run tests to verify RED**

Run: `TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_investment_chat_ui.py -q -k 'config_change' -p no:cacheprovider`

Expected: FAIL because `propose_config_change` does not exist.

- [ ] **Step 3: Implement minimal domain**

Create `config_tools.py` with:
- `build_config_tool_entries(repo, settings, tenant_id)` exposing `propose_config_change` and `get_config_change_status`.
- `build_config_bridge_tool_entries(repo, settings, tenant_id)` exposing internal `apply_approved_config_change`.
- Typed scope support for `agent`, `chat_agent`, and `tenant`.
- Initial write support for `agent` updates through `build_single_agent_entry`, `serialize_agents_config_entries`, `AdminAgentConfigStore`, and `AdminRuntimeOps`.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same targeted pytest command and confirm the new tests pass.

### Task 2: Approval API

- [ ] **Step 1: Write failing tests**

Add tests for `GET /investment-chat/config-drafts` and `POST /investment-chat/config-drafts/{token}/apply`. Assert draft payloads are surfaced as button-ready rows, applying requires the stored token, and repeated approval is idempotent.

- [ ] **Step 2: Run tests to verify RED**

Run the targeted pytest command. Expected: FAIL because the routes do not exist.

- [ ] **Step 3: Implement routes**

Add route handlers in `arena/ui/routes/investment_chat.py` that mirror order draft behavior but use config draft helpers. The apply route should call the internal bridge tool and return a short `chat_delivery_text` for the iframe chat input.

- [ ] **Step 4: Run tests to verify GREEN**

Run the targeted pytest command and confirm the new API tests pass.

### Task 3: UI Action Button

- [ ] **Step 1: Write failing tests**

Extend investment chat page rendering tests to assert the settings approval panel and endpoints are present.

- [ ] **Step 2: Run tests to verify RED**

Run the targeted pytest command. Expected: FAIL because the template has only the order panel.

- [ ] **Step 3: Implement UI panel**

Add a compact `Config Approval` panel in `investment_chat_body.jinja2` that polls `/investment-chat/config-drafts`, shows the change summary/diff, and applies via the POST approval endpoint. Keep it separate from the order panel state.

- [ ] **Step 4: Run tests to verify GREEN**

Run the targeted pytest command and confirm UI tests pass.

### Task 4: Full Verification

- [ ] Run: `TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_investment_chat_ui.py tests/test_agents_config.py tests/test_ui_admin_routes.py -q -p no:cacheprovider`
- [ ] Check `git diff --check`.
- [ ] Review changed files for accidental unrelated edits.
