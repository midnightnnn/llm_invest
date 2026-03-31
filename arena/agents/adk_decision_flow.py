from __future__ import annotations

import json
import time
from typing import Any

from arena.agents.adk_prompting import (
    _parse_json_text,
    _safe_json,
    _tool_category_counts,
    _tool_mix_note,
    _user_prompt,
)
from arena.tools.registry import ToolRegistry


def safe_int(value: Any, default: int = 0) -> int:
    """Converts arbitrary values to int without raising."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_resume_prompt(
    context: dict[str, Any],
    *,
    analysis_funnel: dict[str, Any],
    max_tool_events: int,
) -> str:
    """Builds the execution-phase delta prompt for a resumed ADK session."""
    board_ctx = str(context.get("board_context") or "").strip()
    parts = [
        "cycle_phase: execution",
        "",
        "이전 draft 단계의 분석과 도구 호출 결과를 바탕으로 최종 주문을 결정합니다.",
        "이전에 호출한 도구 결과를 최대한 활용하세요. 필요시 추가 도구 호출도 가능합니다.",
    ]
    if board_ctx:
        parts += ["", "[다른 에이전트 의견]", board_ctx]
    parts += [
        "",
        json.dumps(
            _safe_json(
                {
                    "order_budget": context.get("order_budget", {}),
                    "risk_policy": context.get("risk_policy", {}),
                    "analysis_funnel": analysis_funnel,
                    "opportunity_working_set": context.get("opportunity_working_set", []),
                    "decision_frame": context.get("decision_frame", ""),
                    "tool_budget": {
                        "max_tool_calls": max_tool_events,
                        "note": f"You have up to {max_tool_events} remaining tool calls.",
                    },
                }
            ),
            ensure_ascii=False,
        ),
    ]
    return "\n".join(parts)


def prepare_decision_prompt(
    context: dict[str, Any],
    default_universe: list[str],
    *,
    phase: str,
    base_session_id: str,
    max_tool_events: int,
    resume_session_id: str | None,
    analysis_funnel: dict[str, Any],
) -> tuple[str, str, bool]:
    """Returns `(session_id, prompt, needs_new_session)` for one decision phase."""
    if resume_session_id:
        return (
            resume_session_id,
            build_resume_prompt(
                context,
                analysis_funnel=analysis_funnel,
                max_tool_events=max_tool_events,
            ),
            False,
        )
    session_id = f"{base_session_id}_{phase}_{int(time.time() * 1000)}"
    prompt = _user_prompt(context, default_universe, max_tool_calls=max_tool_events)
    return session_id, prompt, True


def tag_phase_tool_events(tool_events: list[dict[str, Any]], *, phase: str, start_idx: int) -> None:
    """Annotates tool events created during the current phase."""
    for idx in range(start_idx, len(tool_events)):
        tool_events[idx].setdefault("phase", phase)


def build_tool_summary_memory_record(
    tool_events: list[dict[str, Any]],
    *,
    registry: ToolRegistry,
    phase: str,
    analysis_funnel: dict[str, Any],
    cycle_id: str,
    token_usage: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    """Builds summary text/payload for ReAct tool memory persistence."""
    events = [event for event in tool_events if str(event.get("tool") or "").strip()]
    if not events and safe_int((token_usage or {}).get("llm_calls"), 0) <= 0:
        return None

    mix = _tool_category_counts(events, registry=registry)
    lines = [f"ReAct tools used ({phase}): {len(events)}"]
    for event in events[:6]:
        tool = str(event.get("tool") or "tool")
        preview = event.get("result") or event.get("result_preview")
        try:
            preview_txt = json.dumps(preview, ensure_ascii=False, default=str)
        except Exception:
            preview_txt = str(preview)
        preview_txt = preview_txt.replace("\n", " ")[:240]
        lines.append(f"- {tool}: {preview_txt}")
    lines.append(
        "- tool_mix: "
        + " ".join(
            [
                f"quant={mix.get('quant', 0)}",
                f"macro={mix.get('macro', 0)}",
                f"sentiment={mix.get('sentiment', 0)}",
                f"performance={mix.get('performance', 0)}",
                f"context={mix.get('context', 0)}",
                f"other={mix.get('other', 0)}",
            ]
        )
    )
    lines.append(f"- note: {_tool_mix_note(mix)}")

    summary = "\n".join(lines)[:1200]
    payload = {
        "tool_events": _safe_json(events),
        "tool_mix": mix,
        "analysis_funnel": analysis_funnel,
        "phase": phase,
        "cycle_id": cycle_id,
        "token_usage": _safe_json(token_usage or {}),
    }
    return summary, payload


def build_board_prompt(orders_summary: str) -> str:
    """Builds the board-generation follow-up prompt."""
    rules = [
        "cycle_phase: board",
        "",
        "사실성 규칙:",
        "- 아래에 명시된 실행 결과와 현재 세션에서 확인된 사실만 사용하십시오.",
        "- 종목명은 prompt/context에 명시된 경우에만 사용하십시오. ticker_name이 없으면 종목명을 추정하지 말고 티커만 쓰십시오.",
        "- 전날/어제/지난번 등 상대 날짜 표현은 prompt에 명시된 경우에만 사용하십시오.",
        "- 이전 보유 수량을 언급할 수는 있지만, 매수 시점은 명시된 사실이 없으면 추정하지 마십시오.",
        "",
        orders_summary,
    ]
    return "\n".join(rules)


def parse_board_response(text: str) -> dict[str, Any]:
    """Parses board response JSON, falling back to plain text body."""
    if not text or not text.strip():
        return {}
    try:
        return _parse_json_text(text)
    except Exception:
        return {"board_title": "거래 아이디어", "board_body": text.strip()[:1800]}


def parse_decision_response(text: str) -> dict[str, Any]:
    """Parses decision JSON and leaves failures to the caller."""
    return _parse_json_text(text)
