# Investment Watch History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the agent persist what it learned from official macro research, track "worth watching" candidates, and keep post-exit sell outcomes so later cycles can learn from what happened after a trim or sale.

**Architecture:** Keep the model contract explicit in the execution JSON, but make persistence happen in code paths that already own the cycle state. Use a small domain helper module for watch-item lifecycle math, a dedicated `agent_watch_items` table for mutable state, and read-only UI/context surfaces that expose the stored watch items without forcing tool usage.

**Tech Stack:** Python 3.11, pytest, BigQuery and DuckDB schema DDL, FastAPI/Jinja UI, ADK runner/tool wrappers, existing memory and execution stores.

---

## File Structure

- Modify: `arena/prompts/adk/execution_format.txt`
  - Add the optional `research_takeaways` and `watch_updates` fields to the final JSON contract.
- Modify: `arena/agents/adk_agent_flow.py`
  - Normalize the new JSON fields so callers can persist them safely.
- Modify: `arena/agents/adk_agents.py`
  - Persist the new execution artifacts during the execution phase and keep the existing explore/order flow intact.
- Create: `arena/memory/watch_items.py`
  - Own watch-item normalization, lot merging, benchmark math, and lifecycle helpers.
- Modify: `arena/memory/store.py`
  - Save explicit watch updates and automatic post-exit SELL tracking from execution reports.
- Modify: `arena/data/schema.py`
  - Add the `agent_watch_items` table DDL.
- Create: `arena/data/bigquery/watch_store.py`
  - Persist and query watch items in BigQuery.
- Create: `arena/data/local/watch_store.py`
  - Mirror the same watch-item API in DuckDB.
- Modify: `arena/data/bq.py`
  - Wire the new BigQuery store into the repository facade.
- Modify: `arena/data/local/repository.py`
  - Wire the new local store into the repository facade.
- Modify: `arena/context.py`
  - Inject a bounded `watch_context` section into the prompt payload.
- Modify: `arena/ui/memory.py`
  - Add a read-only Watch tab API and include watch stats in the memory page.
- Modify: `arena/ui/templates/memory_panel.jinja2`
  - Add the Watch tab and a read-only panel.
- Modify: `arena/ui/templates/memory_panel_script.jinja2`
  - Add the client-side tab switch / fetch logic for watch items.
- Modify: `arena/cli.py`
  - Register the new backfill command.
- Modify: `arena/cli_commands/admin.py`
  - Add the backfill command implementation.
- Modify: `arena/agents/adk_context_tools.py`
  - Keep the `read_official_macro_research` docstring aligned with the new persistence path.
- Modify: `arena/tools/default_registry.py`
  - Update the catalog description text shown in the UI.
- Tests: `tests/adk/test_prompting_flow.py`, `tests/adk/test_memory_pipeline.py`, `tests/memory/test_memory_store_execution.py`, `tests/memory/test_candidate_structured_memory.py`, `tests/data/test_duckdb_schema.py`, `tests/context/test_memory_context_basics.py`, `tests/ui/test_memory_routes.py`, `tests/cli/test_watch_backfill_command.py`, `tests/test_macro_research_pipeline.py`, `tests/test_new_tools.py`
  - Add regressions for parsing, persistence, schema bootstrap, context injection, UI output, CLI parsing, and tool-description text.

## Out Of Scope

- Do not touch `arena/prompts/adk/core_prompt.txt`.
- Do not make tool descriptions user-editable from the settings UI in this pass.
- Do not force tool usage; the agent remains free to ignore the tool.

### Task 1: Extend The Execution JSON Contract

**Files:**
- Modify: `arena/prompts/adk/execution_format.txt`
- Modify: `arena/agents/adk_agent_flow.py`
- Test: `tests/adk/test_prompting_flow.py`

- [ ] **Step 1: Write the failing test**

Add a parser regression that shows the new fields are accepted and normalized:

```python
def test_extract_decision_payload_returns_watch_artifacts():
    explore_summary, orders, research_takeaways, watch_updates = extract_decision_payload(
        {
            "explore_summary": "macro and candidates reviewed",
            "orders": [],
            "research_takeaways": [
                {"source_doc_id": "bok:note:1", "takeaway": "credit transmission still matters"},
            ],
            "watch_updates": [
                {"action": "add", "watch_kind": "candidate", "ticker": "AAPL", "reason": "quality pullback"},
            ],
        }
    )

    assert explore_summary == "macro and candidates reviewed"
    assert orders == []
    assert research_takeaways[0]["source_doc_id"] == "bok:note:1"
    assert watch_updates[0]["ticker"] == "AAPL"
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
python -m pytest tests/adk/test_prompting_flow.py -q -p no:cacheprovider
```

Expected: the new test fails because `extract_decision_payload()` still returns only `(explore_summary, orders)`.

- [ ] **Step 3: Implement the minimal code**

Update the execution format example to include:

```json
{
  "explore_summary": "text",
  "research_takeaways": [
    {
      "source_doc_id": "bok:2026:policy-note-1",
      "takeaway": "Policy transmission remains active.",
      "transmission_channels": ["policy rates", "credit", "liquidity"],
      "watch_indicators": ["base rate", "household credit growth"],
      "horizon_days": 90
    }
  ],
  "watch_updates": [
    {
      "action": "add",
      "watch_kind": "candidate",
      "ticker": "AAPL",
      "reason": "quality pullback with improving margins",
      "confirmation_conditions": ["next earnings beats", "volume confirms"],
      "invalidation_conditions": ["guide cut", "breaks support"],
      "time_horizon_days": 30,
      "source_doc_ids": ["bok:2026:policy-note-1"]
    }
  ],
  "orders": []
}
```

Then return the new fields from `extract_decision_payload()` without breaking existing callers. Keep the function tolerant of non-list / non-dict input.

- [ ] **Step 4: Run the test and verify it passes**

Run the same `pytest` command and confirm the parser regression passes.

- [ ] **Step 5: Commit**

Commit the contract change and parser update before moving on.

### Task 2: Add Watch-Item Storage And Schema

**Files:**
- Create: `arena/memory/watch_items.py`
- Create: `arena/data/bigquery/watch_store.py`
- Create: `arena/data/local/watch_store.py`
- Modify: `arena/data/schema.py`
- Modify: `arena/data/bq.py`
- Modify: `arena/data/local/repository.py`
- Test: `tests/data/test_duckdb_schema.py`, `tests/memory/test_watch_items.py`, `tests/data/test_local_repository_memory.py`

- [ ] **Step 1: Write the failing tests**

Add tests that prove:

```python
def test_watch_item_schema_is_rendered_by_duckdb_bootstrap():
    assert "agent_watch_items" in duckdb_table_names()


def test_apply_watch_updates_upserts_and_lists_active_items():
    ...


def test_normalize_post_exit_watch_item_merges_sell_lots_and_benchmarks():
    ...
```

The important assertions are:

- `agent_watch_items` exists in the generated schema.
- A candidate watch item can be upserted, read back, and closed.
- A post-exit watch item keeps per-lot sell history, not just the latest SELL.

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m pytest tests/data/test_duckdb_schema.py tests/memory/test_watch_items.py tests/data/test_local_repository_memory.py -q -p no:cacheprovider
```

Expected: the new schema and helper tests fail because the table and helper module do not exist yet.

- [ ] **Step 3: Implement the minimal code**

Implement `arena/memory/watch_items.py` as a pure helper module for:

- normalizing watch updates
- validating `source_doc_id` / `source_doc_ids`
- merging lots for repeated SELLs on the same ticker
- computing session-based checkpoints at 5, 20, and 60 trading sessions
- attaching benchmark comparison data without fabricating missing prices or FX

Add `agent_watch_items` to `arena/data/schema.py`, then add the BigQuery and DuckDB store classes plus repository wiring.

- [ ] **Step 4: Run the tests and verify they pass**

Run the same `pytest` command and confirm schema/bootstrap coverage is green.

- [ ] **Step 5: Commit**

Commit the storage layer before wiring it into the cycle runner.

### Task 3: Persist Research Takeaways And Post-Exit Updates

**Files:**
- Modify: `arena/agents/adk_agents.py`
- Modify: `arena/memory/store.py`
- Modify: `arena/memory/semantic_extractor.py`
- Modify: `arena/memory/policy.py`
- Test: `tests/adk/test_memory_pipeline.py`, `tests/memory/test_memory_store_execution.py`, `tests/memory/test_memory_store_thesis.py`

- [ ] **Step 1: Write the failing tests**

Add regressions that prove:

```python
def test_execution_phase_persists_research_takeaways_and_watch_updates():
    ...

def test_record_execution_creates_post_exit_watch_item_for_successful_sell():
    ...

def test_record_execution_promotes_candidate_watch_item_on_buy_fill():
    ...
```

The key checks are:

- a `research_takeaway` with `source_doc_id` is rejected unless the cycle actually read that document text
- explicit `watch_updates` are stored
- successful SELLs create or update a post-exit watch item
- BUY fills promote a matching candidate watch item

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m pytest tests/adk/test_memory_pipeline.py tests/memory/test_memory_store_execution.py tests/memory/test_memory_store_thesis.py -q -p no:cacheprovider
```

Expected: the new tests fail because the runner and memory store do not yet persist the new artifacts.

- [ ] **Step 3: Implement the minimal code**

Update `adk_agents.py` so the execution phase persists the normalized `research_takeaways` and `watch_updates`. Add store methods for:

- `record_research_takeaways()`
- `record_watch_updates()`
- `record_post_exit_watch_item_from_execution()`

Keep the automatic SELL path inside `record_execution()` so post-exit tracking happens even when the model does not emit a watch update.

- [ ] **Step 4: Run the tests and verify they pass**

Run the same `pytest` command and confirm the persistence regressions are green.

- [ ] **Step 5: Commit**

Commit the runner/store integration before touching prompt context or UI.

### Task 4: Surface Watch State In Context And UI

**Files:**
- Modify: `arena/context.py`
- Modify: `arena/ui/memory.py`
- Modify: `arena/ui/templates/memory_panel.jinja2`
- Modify: `arena/ui/templates/memory_panel_script.jinja2`
- Test: `tests/context/test_memory_context_basics.py`, `tests/context/test_memory_context_hydration.py`, `tests/ui/test_memory_routes.py`

- [ ] **Step 1: Write the failing tests**

Add regressions for:

```python
def test_context_builder_injects_watch_context():
    ...

def test_memory_page_exposes_read_only_watch_tab():
    ...

def test_api_memory_watch_items_filters_by_agent_and_kind():
    ...
```

The expected behavior:

- watch items appear in a bounded `watch_context`
- the memory settings page renders a `Watch` tab
- the API returns only tenant-authorized rows and supports filters for `agent_id`, `market`, `kind`, and `status`

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m pytest tests/context/test_memory_context_basics.py tests/context/test_memory_context_hydration.py tests/ui/test_memory_routes.py -q -p no:cacheprovider
```

Expected: the new watch-context and UI tests fail because the routes and prompt sections do not exist yet.

- [ ] **Step 3: Implement the minimal code**

Add `watch_context` to the context payload, render a read-only Watch tab in the memory settings panel, and expose `/api/memory/watch-items` for the settings page. Keep it read-only in v1.

- [ ] **Step 4: Run the tests and verify they pass**

Run the same `pytest` command and confirm the new UI/context assertions pass.

- [ ] **Step 5: Commit**

Commit the context/UI change as a separate step so it is easy to review.

### Task 5: Add Backfill Command And Update Macro Tool Copy

**Files:**
- Modify: `arena/cli.py`
- Modify: `arena/cli_commands/admin.py`
- Modify: `arena/agents/adk_context_tools.py`
- Modify: `arena/tools/default_registry.py`
- Test: `tests/cli/test_watch_backfill_command.py`, `tests/test_macro_research_pipeline.py`, `tests/test_new_tools.py`

- [ ] **Step 1: Write the failing tests**

Add tests for:

```python
def test_backfill_watch_items_parser_options():
    ...

def test_read_official_macro_research_description_mentions_source_doc_id_and_future_value():
    ...
```

The description test should verify that:

- the tool name is `read_official_macro_research`
- the description says the model should list documents first and read full source text by `source_doc_id`
- the copy explains that reading the original text can yield forward-looking signals and future-trend value
- the old `get_macro_research_briefing` name stays absent from the registry

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
python -m pytest tests/cli/test_watch_backfill_command.py tests/test_macro_research_pipeline.py tests/test_new_tools.py -q -p no:cacheprovider
```

Expected: the CLI parser and description regressions fail before the command and text are updated.

- [ ] **Step 3: Implement the minimal code**

Add `backfill-watch-items` to the CLI, wire it through `cmd_backfill_watch_items()`, and update the macro research tool copy in both the callable docstring and the registry catalog text.

- [ ] **Step 4: Run the tests and verify they pass**

Run the same `pytest` command and confirm the parser/description tests are green.

- [ ] **Step 5: Commit**

Commit the CLI and copy updates together so the user-visible behavior stays in sync.

## Verification

After all tasks are green, run the focused regression set that covers the end-to-end feature:

```bash
python -m pytest \
  tests/adk/test_prompting_flow.py \
  tests/adk/test_memory_pipeline.py \
  tests/memory/test_memory_store_execution.py \
  tests/memory/test_watch_items.py \
  tests/context/test_memory_context_basics.py \
  tests/context/test_memory_context_hydration.py \
  tests/ui/test_memory_routes.py \
  tests/cli/test_watch_backfill_command.py \
  tests/test_macro_research_pipeline.py \
  tests/test_new_tools.py \
  -q -p no:cacheprovider
```

If deployment access is available later, verify the Cloud Run job/UI surfaces with the real runtime too:

```bash
gcloud run jobs list
```

and confirm the latest job build exposes `read_official_macro_research` in `available_tools_json` with no `get_macro_research_briefing` entry.
