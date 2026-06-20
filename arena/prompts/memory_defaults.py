from __future__ import annotations

from arena.prompts.loader import load_prompt_text


DEFAULT_GLOBAL_MEMORY_COMPACTION_PROMPT = load_prompt_text("memory", "default_global_compaction_prompt.txt")
DEFAULT_LOCAL_MEMORY_COMPACTION_PROMPT = load_prompt_text("memory", "default_local_compaction_prompt.txt")


def default_memory_compaction_prompt(scope: str = "global") -> str:
    token = str(scope or "").strip().lower()
    if token in {"local", "tenant"}:
        return DEFAULT_LOCAL_MEMORY_COMPACTION_PROMPT
    return DEFAULT_GLOBAL_MEMORY_COMPACTION_PROMPT
