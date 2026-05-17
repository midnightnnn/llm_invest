from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from arena.agents.adk_agents import _apply_tool_schema_metadata, _load_disabled_tool_ids
from arena.agents.adk_runner_bootstrap import build_tool_wrapper
from arena.config import load_settings
from arena.tools.default_registry import build_default_registry
from arena.tools.registry import ToolEntry
from arena.tools.scratch_workspace import ScratchWorkspace


class _RepoForDisabledTools:
    def __init__(self, disabled: str | None):
        self.disabled = disabled

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        _ = tenant_id, config_key
        return self.disabled


def test_apply_tool_schema_metadata_prefers_registry_description() -> None:
    def original_tool(ticker: str) -> dict[str, str]:
        """Original docstring that should not leak to the model."""
        return {"ticker": ticker}

    entry = ToolEntry(
        tool_id="screen_market",
        name="screen_market",
        description="Canonical registry description for the model schema.",
        category="quant",
        callable=original_tool,
    )

    wrapped = _apply_tool_schema_metadata(
        original_tool,
        entry=entry,
        sig=inspect.signature(original_tool),
    )

    assert wrapped.__name__ == "screen_market"
    assert wrapped.__doc__ == "Canonical registry description for the model schema."


def test_build_tool_wrapper_awaits_async_callables() -> None:
    from arena.agents.adk_tool_helpers import noop_search_tool_memories, noop_update_candidate_ledger

    async def original_tool(ticker: str) -> dict[str, str]:
        return {"ticker": ticker, "status": "fresh"}

    entry = ToolEntry(
        tool_id="sample_async_tool",
        name="sample_async_tool",
        description="Fetch research.",
        category="context",
        callable=original_tool,
    )
    tool_events: list[dict] = []

    wrapped = build_tool_wrapper(
        entry,
        settings=load_settings(),
        agent_id="gpt",
        tool_events=tool_events,
        update_candidate_ledger=noop_update_candidate_ledger,
        search_tool_memories=noop_search_tool_memories,
        apply_tool_schema_metadata=_apply_tool_schema_metadata,
    )

    out = asyncio.run(wrapped(ticker="AAPL"))

    assert out["ticker"] == "AAPL"
    assert out["status"] == "fresh"
    assert "_runtime_clock" in out
    assert tool_events[0]["result"] == {"ticker": "AAPL", "status": "fresh"}


def test_batch_default_tool_schema_preserves_required_fields_and_enums() -> None:
    from google.adk.tools.function_tool import FunctionTool

    from arena.agents.adk_tool_helpers import noop_search_tool_memories, noop_update_candidate_ledger

    settings = load_settings()
    repo = SimpleNamespace(get_config=lambda *args, **kwargs: "")
    registry = build_default_registry(repo, settings, tenant_id="local")
    scratch = ScratchWorkspace(agent_id="gpt", tenant_id="local", tool_events=[])
    registry.bind("scratch_run_python", scratch.run_python)

    def declaration(tool_id: str):
        entry = registry.get(tool_id)
        assert entry is not None
        assert entry.callable is not None
        wrapped = build_tool_wrapper(
            entry,
            settings=settings,
            agent_id="gpt",
            tool_events=[],
            update_candidate_ledger=noop_update_candidate_ledger,
            search_tool_memories=noop_search_tool_memories,
            apply_tool_schema_metadata=_apply_tool_schema_metadata,
        )
        return FunctionTool(wrapped)._get_declaration()

    def enum_values(prop: dict) -> list[str]:
        if "enum" in prop:
            return prop["enum"]
        for option in prop.get("any_of") or []:
            if isinstance(option, dict) and "enum" in option:
                return option["enum"]
        raise AssertionError(f"missing enum in schema property: {prop}")

    def item_enum_values(prop: dict) -> list[str]:
        if isinstance(prop.get("items"), dict) and "enum" in prop["items"]:
            return prop["items"]["enum"]
        for option in prop.get("any_of") or []:
            if isinstance(option, dict) and isinstance(option.get("items"), dict) and "enum" in option["items"]:
                return option["items"]["enum"]
        raise AssertionError(f"missing item enum in schema property: {prop}")

    screen_params = declaration("screen_market").parameters.model_dump(mode="json", exclude_none=True)
    assert enum_values(screen_params["properties"]["bucket"]) == [
        "auto",
        "balanced",
        "momentum",
        "pullback",
        "recovery",
        "defensive",
        "value",
    ]
    assert enum_values(screen_params["properties"]["sort_by"]) == [
        "none",
        "as_of_ts",
        "ret_20d",
        "ret_5d",
        "volatility_20d",
        "sentiment_score",
        "close_price_krw",
    ]
    assert enum_values(screen_params["properties"]["order"]) == ["asc", "desc"]
    assert enum_values(screen_params["properties"]["market_scope"]) == ["us", "kr"]

    optimize_params = declaration("optimize_portfolio").parameters.model_dump(mode="json", exclude_none=True)
    assert "tickers" in optimize_params["required"]
    assert enum_values(optimize_params["properties"]["strategy"]) == ["sharpe", "risk_parity", "forecast"]
    assert enum_values(optimize_params["properties"]["forecast_mode"]) == [
        "default",
        "all",
        "stacked",
        "base",
        "balanced",
        "lgbm",
        "ridge",
        "avg",
    ]

    opportunity_params = declaration("recommend_opportunities").parameters.model_dump(mode="json", exclude_none=True)
    assert item_enum_values(opportunity_params["properties"]["buckets"]) == ["momentum", "pullback", "recovery"]
    assert enum_values(opportunity_params["properties"]["market_scope"]) == ["us", "kr"]
    assert item_enum_values(opportunity_params["properties"]["profiles"]) == [
        "aggressive",
        "balanced",
        "defensive",
        "value",
        "tactical",
        "tactical_leverage",
        "tactical_inverse",
        "tactical_hedge",
    ]

    forecast_params = declaration("forecast_returns").parameters.model_dump(mode="json", exclude_none=True)
    assert enum_values(forecast_params["properties"]["market_scope"]) == ["us", "kr"]

    sector_params = declaration("sector_summary").parameters.model_dump(mode="json", exclude_none=True)
    assert enum_values(sector_params["properties"]["market_scope"]) == ["us", "kr"]

    scratch_params = declaration("scratch_run_python").parameters.model_dump(mode="json", exclude_none=True)
    assert "code" in scratch_params["required"]
    assert scratch_params["properties"]["inputs"]["type"] == "OBJECT"
    assert scratch_params["properties"]["inputs"]["nullable"] is True


def test_load_disabled_tool_ids_uses_tool_id_tokens() -> None:
    repo = _RepoForDisabledTools('["fetch_reddit_sentiment","optimize_portfolio"]')
    out = _load_disabled_tool_ids(repo, "tenant-a")

    assert out == {"fetch_reddit_sentiment", "optimize_portfolio"}
