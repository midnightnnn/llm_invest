from __future__ import annotations

from typing import Any

from arena.prompts.memory_defaults import default_memory_compaction_prompt


def seed_local_memory_compaction_prompts(
    repo: Any,
    *,
    tenant_id: str = "local",
    updated_by: str = "local-bootstrap",
) -> dict[str, bool]:
    getter = getattr(repo, "get_config", None)
    setter = getattr(repo, "set_config", None)
    if not callable(setter):
        return {"global": False, "tenant": False}

    def _has_value(scope: str) -> bool:
        if not callable(getter):
            return False
        try:
            value = getter(scope, "memory_compactor_prompt")
        except Exception:
            return False
        return bool(str(value or "").strip())

    seeded = {"global": False, "tenant": False}
    if not _has_value("global"):
        setter(
            "global",
            "memory_compactor_prompt",
            default_memory_compaction_prompt("global"),
            updated_by,
        )
        seeded["global"] = True

    tenant = str(tenant_id or "").strip().lower()
    if tenant and not _has_value(tenant):
        setter(
            tenant,
            "memory_compactor_prompt",
            default_memory_compaction_prompt("local"),
            updated_by,
        )
        seeded["tenant"] = True

    return seeded
