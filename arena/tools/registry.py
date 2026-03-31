from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable


@dataclass(frozen=True, slots=True)
class ToolEntry:
    """Stores metadata and callable for one tool."""

    tool_id: str
    name: str
    description: str
    category: str
    callable: Callable | None = None
    tier: str = "optional"
    label_ko: str = ""
    description_ko: str = ""
    enabled: bool = True
    sort_order: int = 100


class ToolRegistry:
    """Central registry of tools for two-phase selection."""

    def __init__(self, entries: Iterable[ToolEntry] | None = None):
        self._entries: dict[str, ToolEntry] = {}
        for entry in entries or []:
            self.register(entry)

    def register(self, entry: ToolEntry) -> None:
        """Registers or replaces a tool entry by id."""
        self._entries[entry.tool_id] = entry

    def get(self, tool_id: str) -> ToolEntry | None:
        """Returns one entry by id when present."""
        return self._entries.get(str(tool_id or "").strip())

    def clone(self) -> ToolRegistry:
        """Returns a shallow copy of the registry and its immutable entries."""
        return ToolRegistry(self._entries.values())

    def bind(self, tool_id: str, fn: Callable) -> None:
        """Binds or replaces the callable for an existing tool entry."""
        entry = self.get(tool_id)
        if entry is None:
            raise KeyError(f"unknown tool_id: {tool_id}")
        self._entries[entry.tool_id] = replace(entry, callable=fn)

    def list_entries(
        self,
        *,
        include_disabled: bool = False,
        tier: str | None = None,
        require_callable: bool = False,
    ) -> list[ToolEntry]:
        """Returns a stable list of entries sorted by explicit order then tool_id."""
        entries = []
        for entry in self._entries.values():
            if not include_disabled and not entry.enabled:
                continue
            if tier and entry.tier != tier:
                continue
            if require_callable and entry.callable is None:
                continue
            entries.append(entry)
        return sorted(entries, key=lambda e: (int(e.sort_order), e.tool_id))

    def catalog_for_prompt(self) -> str:
        """Builds phase-1 catalog text: '- tool_id: description'."""
        lines = [f"- {e.tool_id}: {e.description}" for e in self.list_entries()]
        return "\n".join(lines)

    def resolve(self, tool_ids: list[str]) -> list[Callable]:
        """Resolves selected tool ids into callables."""
        out: list[Callable] = []
        for tid in tool_ids:
            entry = self._entries.get(tid)
            if entry and entry.enabled and entry.callable is not None:
                out.append(entry.callable)
        return out

    def set_context(self, context: dict) -> None:
        """Propagates context to registered tools that support set_context."""
        seen: set[int] = set()
        for entry in self.list_entries(include_disabled=True, require_callable=True):
            obj = getattr(entry.callable, "__self__", None)
            if obj is not None and id(obj) not in seen and hasattr(obj, "set_context"):
                obj.set_context(context)
                seen.add(id(obj))
