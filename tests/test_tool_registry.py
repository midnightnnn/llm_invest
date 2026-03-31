from __future__ import annotations

from arena.tools.registry import ToolEntry, ToolRegistry


def test_tool_registry_catalog_and_resolve() -> None:
    reg = ToolRegistry()

    def a():
        return 1

    def b():
        return 2

    reg.register(ToolEntry(tool_id="b", name="b", description="desc b", category="quant", callable=b))
    reg.register(ToolEntry(tool_id="a", name="a", description="desc a", category="quant", callable=a))

    catalog = reg.catalog_for_prompt()
    assert "- a: desc a" in catalog
    assert "- b: desc b" in catalog

    callables = reg.resolve(["b", "a", "missing"])
    assert callables == [b, a]


def test_tool_registry_clone_bind_and_disabled_filtering() -> None:
    reg = ToolRegistry(
        [
            ToolEntry(
                tool_id="core_tool",
                name="core_tool",
                description="core",
                category="context",
                tier="core",
                enabled=True,
                sort_order=10,
            ),
            ToolEntry(
                tool_id="hidden_optional",
                name="hidden_optional",
                description="hidden",
                category="quant",
                enabled=False,
                sort_order=20,
            ),
        ]
    )

    clone = reg.clone()

    def core_tool():
        return 1

    clone.bind("core_tool", core_tool)

    assert [entry.tool_id for entry in reg.list_entries()] == ["core_tool"]
    assert [entry.tool_id for entry in clone.list_entries(include_disabled=True)] == ["core_tool", "hidden_optional"]
    assert clone.resolve(["core_tool", "hidden_optional"]) == [core_tool]
