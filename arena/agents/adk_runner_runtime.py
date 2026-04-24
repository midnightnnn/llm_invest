from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from google.adk import Runner
from google.genai import types

from arena.agents.adk_prompting import _safe_json
from arena.config import Settings
from arena.logging_utils import event_extra
from arena.memory.policy import memory_event_enabled, memory_vector_search_enabled
from arena.models import MemoryEvent, utc_now

logger = logging.getLogger(__name__)


class AdkToolBudgetExceeded(RuntimeError):
    """Raised when an ADK ReAct run keeps calling tools past its budget."""


def truncate_tool_result(value: Any, max_list: int = 30, max_str: int = 2000) -> Any:
    """Truncates large nested tool payloads before event logging."""
    if isinstance(value, dict):
        return {str(key): truncate_tool_result(val, max_list, max_str) for key, val in value.items()}
    if isinstance(value, list):
        return [truncate_tool_result(val, max_list, max_str) for val in value[:max_list]]
    if isinstance(value, str) and len(value) > max_str:
        return value[:max_str] + "..."
    return value


def append_builtin_tool_event(
    tool_events: list[dict[str, Any]],
    *,
    tool_name: str,
    args_preview: dict[str, Any],
    started_at_ms: int,
    elapsed_ms: int,
    result: Any,
    error: str | None,
) -> None:
    """Appends one builtin tool event to the runner event log."""
    tool_events.append(
        {
            "tool": tool_name,
            "args": args_preview,
            "started_at": started_at_ms,
            "elapsed_ms": elapsed_ms,
            "result": _safe_json(truncate_tool_result(result)),
            "error": error,
            "source": "builtin",
        }
    )


def replace_last_tool_event_result(tool_events: list[dict[str, Any]], result: Any) -> None:
    """Rewrites the most recent tool result after prompt compaction."""
    if not tool_events:
        return
    tool_events[-1]["result"] = _safe_json(truncate_tool_result(result))


def append_mcp_tool_event(
    tool_events: list[dict[str, Any]],
    wrapped_tool_names: set[str],
    *,
    tool_name: str,
    args_preview: dict[str, Any],
) -> None:
    """Records MCP-originated tool calls that are not part of builtin wrappers."""
    if not tool_name or tool_name in wrapped_tool_names:
        return
    tool_events.append(
        {
            "tool": tool_name,
            "args": _safe_json(args_preview),
            "elapsed_ms": 0,
            "result_preview": None,
            "error": None,
            "source": "mcp",
        }
    )


def search_tool_memories(
    *,
    memory_store: Any,
    settings: Settings,
    agent_id: str,
    seen_memory_ids: set[str],
    query: str,
) -> list[dict[str, Any]] | None:
    """Vector search for REACT-time tool memory injection. Returns up to 2 rows."""
    vector_store = getattr(memory_store, "vector_store", None)
    if not vector_store or not memory_vector_search_enabled(settings.memory_policy):
        return None
    memories = vector_store.search_similar_memories(
        agent_id=agent_id,
        query=query,
        limit=2,
        trading_mode=settings.trading_mode,
        tenant_id=memory_store._tenant(),
    )
    if not memories:
        return None

    result: list[dict[str, Any]] = []
    added_ids: set[str] = set()
    for memory in memories:
        event_id = str(memory.get("event_id") or "").strip()
        if event_id and event_id in seen_memory_ids:
            continue

        outcome_score = memory.get("outcome_score")
        outcome_label = ""
        try:
            outcome_value = float(outcome_score) if outcome_score is not None else None
        except (TypeError, ValueError):
            outcome_value = None
        if outcome_value is not None:
            if outcome_value >= 0.65:
                outcome_label = "win"
            elif outcome_value <= 0.35:
                outcome_label = "loss"

        row: dict[str, Any] = {
            "summary": str(memory.get("summary") or "")[:220],
            "importance_score": (
                memory.get("importance_score")
                if memory.get("importance_score") is not None
                else memory.get("score", 0.5)
            ),
        }
        created = memory.get("created_at")
        if created and isinstance(created, datetime):
            row["created_at"] = created.isoformat()
            row["created_date"] = created.date().isoformat()
            row["age_days"] = max(0, (utc_now() - created).days)
        else:
            created_date = str(memory.get("created_date") or "").strip()
            if created_date:
                row["created_date"] = created_date
        if outcome_label:
            row["outcome_label"] = outcome_label
        result.append(row)
        if event_id:
            added_ids.add(event_id)
        if len(result) >= 2:
            break

    if added_ids:
        seen_memory_ids.update(added_ids)
    return result or None


def persist_tool_summary_memory(
    *,
    memory_store: Any,
    settings: Settings,
    repo: Any,
    agent_id: str,
    summary: str,
    payload: dict[str, Any],
) -> str | None:
    """Persists tool summaries through the main memory path when available."""
    if memory_store:
        return memory_store.record_memory(
            agent_id=agent_id,
            summary=summary,
            event_type="react_tools_summary",
            score=0.6,
            payload=payload,
        )

    if not memory_event_enabled(settings.memory_policy, "react_tools_summary", True):
        return None
    event = MemoryEvent(
        agent_id=agent_id,
        event_type="react_tools_summary",
        summary=summary,
        trading_mode=settings.trading_mode,
        payload=payload,
        score=0.6,
    )
    repo.write_memory_event(event)
    return event.event_id


async def collect_response_text(
    *,
    runner: Runner,
    user_id: str,
    session_id: str,
    prompt: str,
    run_config: Any,
    max_tool_events: int,
    wrapped_tool_names: set[str],
    tool_events: list[dict[str, Any]],
    agent_id: str,
) -> tuple[str, dict[str, int]]:
    """Collects final text and token usage from one ADK run."""
    last_text = ""
    tool_calls = 0
    llm_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    total_thinking_tokens = 0
    message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message,
        run_config=run_config,
    ):
        usage_metadata = getattr(event, "usage_metadata", None)
        if usage_metadata is not None:
            llm_calls += 1
            total_prompt_tokens += int(getattr(usage_metadata, "prompt_token_count", 0) or 0)
            total_completion_tokens += int(getattr(usage_metadata, "candidates_token_count", 0) or 0)
            total_cached_tokens += int(getattr(usage_metadata, "cached_content_token_count", 0) or 0)
            total_thinking_tokens += int(getattr(usage_metadata, "thoughts_token_count", 0) or 0)
        if not event.content:
            continue
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                tool_calls += 1
                function_call = getattr(part, "function_call")
                tool_name = str(getattr(function_call, "name", "") or "").strip()
                args = getattr(function_call, "args", None)
                args_preview = dict(args) if isinstance(args, dict) else {}
                append_mcp_tool_event(
                    tool_events,
                    wrapped_tool_names,
                    tool_name=tool_name,
                    args_preview=args_preview,
                )
                if tool_calls > max_tool_events:
                    logger.warning("[yellow]ReAct loop hit max tool calls (%d)[/yellow]", max_tool_events)
                    raise AdkToolBudgetExceeded(
                        f"ADK tool budget exceeded after {tool_calls} tool calls (limit={max_tool_events})"
                    )
            text = getattr(part, "text", None)
            if text:
                last_text = text

    token_usage = {
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "cached_tokens": total_cached_tokens,
        "thinking_tokens": total_thinking_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens + total_thinking_tokens,
    }
    cache_pct = (
        round(total_cached_tokens / total_prompt_tokens * 100, 1)
        if total_prompt_tokens > 0
        else 0.0
    )
    logger.info(
        "[cyan]TOKEN_USAGE[/cyan] agent=%s llm_calls=%d prompt=%d cached=%d (%.1f%%) completion=%d thinking=%d",
        agent_id,
        llm_calls,
        total_prompt_tokens,
        total_cached_tokens,
        cache_pct,
        total_completion_tokens,
        total_thinking_tokens,
        extra=event_extra(
            "adk_token_usage",
            agent_id=agent_id,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            prompt_tokens=total_prompt_tokens,
            cached_tokens=total_cached_tokens,
            cache_pct=cache_pct,
            completion_tokens=total_completion_tokens,
            thinking_tokens=total_thinking_tokens,
            total_tokens=token_usage["total_tokens"],
        ),
    )
    return last_text, token_usage
