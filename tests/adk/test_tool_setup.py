from __future__ import annotations

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

    screen_params = declaration("screen_market").parameters.model_dump(mode="json", exclude_none=True)
    assert screen_params["properties"]["bucket"]["enum"] == [
        "auto",
        "balanced",
        "momentum",
        "pullback",
        "recovery",
        "defensive",
        "value",
    ]
    assert screen_params["properties"]["sort_by"]["enum"] == [
        "none",
        "as_of_ts",
        "ret_20d",
        "ret_5d",
        "volatility_20d",
        "sentiment_score",
        "close_price_krw",
    ]
    assert screen_params["properties"]["order"]["enum"] == ["asc", "desc"]

    optimize_params = declaration("optimize_portfolio").parameters.model_dump(mode="json", exclude_none=True)
    assert "tickers" in optimize_params["required"]
    assert optimize_params["properties"]["strategy"]["enum"] == ["sharpe", "risk_parity", "forecast"]
    assert optimize_params["properties"]["forecast_mode"]["enum"] == [
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
    assert opportunity_params["properties"]["buckets"]["items"]["enum"] == ["momentum", "pullback", "recovery"]
    assert opportunity_params["properties"]["profiles"]["items"]["enum"] == [
        "aggressive",
        "balanced",
        "defensive",
        "value",
        "tactical",
        "tactical_leverage",
        "tactical_inverse",
        "tactical_hedge",
    ]

    scratch_params = declaration("scratch_run_python").parameters.model_dump(mode="json", exclude_none=True)
    assert "code" in scratch_params["required"]
    assert scratch_params["properties"]["inputs"]["type"] == "OBJECT"
    assert scratch_params["properties"]["inputs"]["nullable"] is True


def test_load_disabled_tool_ids_uses_tool_id_tokens() -> None:
    repo = _RepoForDisabledTools('["fetch_reddit_sentiment","optimize_portfolio"]')
    out = _load_disabled_tool_ids(repo, "tenant-a")

    assert out == {"fetch_reddit_sentiment", "optimize_portfolio"}
