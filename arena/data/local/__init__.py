"""Local DuckDB-backed storage backend (opt-in via ARENA_MODE=local).

Lazy import: nothing in this package may import duckdb at module load time —
the optional ``local`` extra controls availability. Import duckdb only inside
function bodies that are guaranteed to run after the user opts in.
"""
