from __future__ import annotations

import os
from typing import Any

from arena.agents.base import AgentOutput
from arena.models import BoardPost


def retry_policy_from_env() -> tuple[int, float]:
    """Reads retry policy from environment with safe clamping."""
    retry_limit = max(0, min(int(os.getenv("ARENA_ADK_RETRY_MAX", "2") or "2"), 4))
    retry_delay = max(0.5, min(float(os.getenv("ARENA_ADK_RETRY_BACKOFF_SECONDS", "2.0") or "2.0"), 10.0))
    return retry_limit, retry_delay


def cycle_phase(context: dict[str, Any]) -> str:
    """Returns normalized cycle phase."""
    return str(context.get("cycle_phase", "execution")).strip().lower() or "execution"


def execution_resume_session_id(*, phase: str, explore_session_id: str | None) -> str | None:
    """Returns resume session id only for execution phase."""
    return explore_session_id if phase == "execution" else None


def extract_decision_payload(decision: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Extracts normalized explore summary and order list from model output."""
    draft_summary = str(decision.get("draft_summary", "")).strip()[:200]
    orders = decision.get("orders", [])
    if not isinstance(orders, list):
        orders = []
    return draft_summary, orders


def mentioned_tickers(orders: list[dict[str, Any]]) -> list[str]:
    """Returns distinct mentioned tickers in stable order."""
    out: list[str] = []
    for order in orders:
        if not isinstance(order, dict):
            continue
        ticker = str(order.get("ticker", "")).strip().upper()
        if ticker and ticker not in out:
            out.append(ticker)
    return out


def explore_phase_output(
    *,
    agent_id: str,
    cycle_id: str,
    decision: dict[str, Any],
    draft_summary: str,
    orders: list[dict[str, Any]],
    share_summary: bool,
) -> AgentOutput:
    """Builds explore-phase output used for optional peer summary sharing."""
    title_default = "탐색 요약" if share_summary else "내부 탐색"
    body_default = "근거 없음" if share_summary else (draft_summary or "내부 탐색 단계")
    board_title = str(decision.get("board_title", title_default)).strip()[:120] or title_default
    board_body = str(decision.get("board_body", body_default)).strip()[:1800] or body_default
    post = BoardPost(
        agent_id=agent_id,
        title=board_title,
        body=board_body,
        draft_summary=draft_summary,
        cycle_id=cycle_id,
        tickers=mentioned_tickers(orders),
    )
    return AgentOutput(intents=[], board_post=post)


def execution_phase_output(
    *,
    agent_id: str,
    cycle_id: str,
    intents: list[Any],
    tickers_mentioned: set[str] | list[str],
    board_decision: dict[str, Any],
    orders_summary: str,
) -> AgentOutput:
    """Builds final execution-phase output."""
    board_title = str(board_decision.get("board_title", "")).strip()[:120] or "거래 아이디어"
    board_body = str(board_decision.get("board_body", "")).strip()[:1800] or orders_summary
    post = BoardPost(
        agent_id=agent_id,
        title=board_title,
        body=board_body,
        cycle_id=cycle_id,
        tickers=list(tickers_mentioned),
    )
    return AgentOutput(intents=intents, board_post=post)
