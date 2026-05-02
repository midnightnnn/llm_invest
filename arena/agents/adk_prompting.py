from __future__ import annotations

import json
import re
from typing import Any

from arena.config import AgentConfig
from arena.data.bq import BigQueryRepository
from arena.prompts.prompt_pack import (
    EXECUTION_FORMAT,
    EXPLORE_SHARED_FORMAT,
    EXPLORE_SOLO_FORMAT,
    PromptPack,
    safe_json,
)
from arena.tools.registry import ToolRegistry


def _safe_json(value: Any) -> Any:
    return safe_json(value)


def _parse_json_text(text: str) -> dict[str, Any]:
    """Extracts a JSON object from model output text."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _file_core_prompt() -> str:
    return PromptPack.file_core_prompt()


def _file_user_prompt_default() -> str:
    return PromptPack.file_user_prompt_default()


def _load_prompt_part(
    config_key: str,
    file_fallback,
    repo: BigQueryRepository | None = None,
    tenant_id: str = "local",
) -> str:
    return PromptPack.load_prompt_part(config_key, file_fallback, repo=repo, tenant_id=tenant_id)


def _system_prompt(
    agent_id: str,
    *,
    repo: BigQueryRepository | None = None,
    tenant_id: str = "local",
    agent_config: AgentConfig | None = None,
    target_market: str = "us",
) -> str:
    return PromptPack.render_system_prompt(
        agent_id,
        repo=repo,
        tenant_id=tenant_id,
        agent_config=agent_config,
        target_market=target_market,
    )


_EXPLORE_SHARED_FORMAT = EXPLORE_SHARED_FORMAT
_EXPLORE_SOLO_FORMAT = EXPLORE_SOLO_FORMAT


def _user_prompt(context: dict[str, Any], default_universe: list[str], *, max_tool_calls: int = 50) -> str:
    return PromptPack.render_decision_prompt(context, default_universe, max_tool_calls=max_tool_calls)


def _tool_category_counts(
    tool_events: list[dict[str, Any]],
    *,
    registry: ToolRegistry | None = None,
) -> dict[str, int]:
    """Builds compact category counts for recent tool usage feedback."""
    category_map: dict[str, str] = {}
    if registry is not None:
        category_map = {
            str(entry.tool_id).strip(): str(entry.category).strip().lower() or "other"
            for entry in registry.list_entries(include_disabled=True)
            if str(entry.tool_id).strip()
        }

    counts: dict[str, int] = {
        "quant": 0,
        "macro": 0,
        "sentiment": 0,
        "performance": 0,
        "context": 0,
        "other": 0,
    }

    for event in tool_events:
        tool = str((event or {}).get("tool") or "").strip()
        if not tool:
            continue
        category = category_map.get(tool, "other")
        bucket = category if category in counts else "other"
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _tool_mix_note(counts: dict[str, int]) -> str:
    """Returns a light-touch note to reduce single-source evidence bias."""
    evidence_axes = [
        "quant",
        "macro",
        "sentiment",
        "performance",
    ]
    used_axes = sum(1 for key in evidence_axes if int(counts.get(key, 0)) > 0)
    if used_axes >= 2:
        return "evidence mix looks balanced."
    if int(counts.get("sentiment", 0)) > 0 and int(counts.get("quant", 0)) == 0 and int(counts.get("macro", 0)) == 0:
        return "news/sentiment-heavy cycle; quant/performance cross-check may help."
    if (int(counts.get("quant", 0)) > 0 or int(counts.get("macro", 0)) > 0) and int(counts.get("sentiment", 0)) == 0:
        return "market-data-heavy cycle; qualitative/news cross-check may help."
    return "single-source tendency detected; consider mixing another evidence type."
