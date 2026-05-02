# ADK Dev UI Adapters

This directory exposes module-level `root_agent` objects for ADK Web.

Run from the repository root:

```bash
adk web --host 127.0.0.1 --port 18127 arena/agents/dev_ui
```

Optional local-only overrides:

```bash
ARENA_DEV_UI_TENANT_ID=local
ARENA_LOCAL_DB_PATH=data/arena.duckdb
ARENA_DEV_UI_ENSURE_TABLES=1
```

The adapters force local, paper-mode guardrails before building agents. They
also default `ARENA_LOCAL_DB_PATH` to `data/arena.duckdb` when it is unset. They
are intended for prompt and tool-call debugging, not full portfolio-cycle
simulation.
