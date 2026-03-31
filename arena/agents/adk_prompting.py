from __future__ import annotations

import functools
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from arena.config import AgentConfig
from arena.data.bq import BigQueryRepository
from arena.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _safe_json(value: Any) -> Any:
    """Converts nested values into JSON-serializable primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    return value


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


@functools.lru_cache(maxsize=1)
def _file_core_prompt() -> str:
    """Loads the default core prompt (format rules, tool rules, etc.) from disk."""
    prompt_path = Path(__file__).resolve().parent / "prompts" / "core_prompt.txt"
    try:
        text = prompt_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load core prompt: path={prompt_path} err={exc}"
        ) from exc
    if "{agent_id}" not in text:
        raise RuntimeError(
            f"Invalid core prompt: missing '{{agent_id}}' placeholder at {prompt_path}"
        )
    return text


@functools.lru_cache(maxsize=1)
def _file_user_prompt_default() -> str:
    """Loads the default user-editable prompt from disk."""
    prompt_path = Path(__file__).resolve().parent / "prompts" / "system_prompt.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load default user prompt: path={prompt_path} err={exc}"
        ) from exc


def _load_prompt_part(
    config_key: str,
    file_fallback: Callable[[], str],
    repo: BigQueryRepository | None = None,
    tenant_id: str = "local",
) -> str:
    """Loads a prompt part from DB first, then local file fallback."""
    if repo is not None:
        try:
            text = repo.get_config(str(tenant_id or "").strip().lower() or "local", config_key)
        except Exception as exc:
            logger.warning(
                "[yellow]DB %s load failed[/yellow] tenant=%s err=%s",
                config_key,
                tenant_id,
                str(exc),
            )
            text = None
        if text is not None and str(text).strip():
            return str(text).strip()
    return file_fallback()


def _system_prompt(
    agent_id: str,
    *,
    repo: BigQueryRepository | None = None,
    tenant_id: str = "local",
    agent_config: AgentConfig | None = None,
    target_market: str = "us",
) -> str:
    """Builds system instructions: core_prompt (global, file-only) + system_prompt (per-agent or per-tenant)."""
    core = _file_core_prompt()
    if agent_config and agent_config.system_prompt:
        user = agent_config.system_prompt
    else:
        user = _load_prompt_part("system_prompt", _file_user_prompt_default, repo=repo, tenant_id=tenant_id)
    return core.replace("{agent_id}", agent_id).replace("{target_market}", target_market) + "\n\n" + user


def _user_prompt(context: dict[str, Any], default_universe: list[str], *, max_tool_calls: int = 50) -> str:
    """Builds compact user prompt payload for one cycle decision."""
    _ = default_universe

    payload = {
        "cycle_phase": context.get("cycle_phase", "execution"),
        "performance_context": context.get("performance_context", ""),
        "active_thesis_context": context.get("active_thesis_context", ""),
        "memory_context": context.get("memory_context", ""),
        "board_context": context.get("board_context", ""),
        "portfolio": context.get("portfolio", {}),
        "ticker_names": context.get("ticker_names", {}),
        "risk_policy": context.get("risk_policy", {}),
        "order_budget": context.get("order_budget", {}),
        "sleeve_state": context.get("sleeve_state", {}),
        "active_theses": context.get("active_theses", []),
        "analysis_funnel": context.get("analysis_funnel", {}),
        "opportunity_working_set": context.get("opportunity_working_set", []),
        "decision_frame": context.get("decision_frame", ""),
        "investment_style_context": context.get("investment_style_context", ""),
        "recent_memory_summaries": [
            str(row.get("summary"))
            for row in (context.get("memory_events") or [])[:6]
            if isinstance(row, dict) and row.get("summary")
        ],
        "tool_budget": {
            "max_tool_calls": max_tool_calls,
            "note": f"You have up to {max_tool_calls} tool calls. Plan accordingly and always output final JSON before exhausting your budget.",
        },
    }
    return (
        "Context payload JSON (follow system instructions; output JSON only):\n"
        + json.dumps(_safe_json(payload), ensure_ascii=False)
    )


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
