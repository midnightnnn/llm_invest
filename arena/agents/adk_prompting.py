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
from arena.agents.adk_runner_state import model_facing_funnel_metrics
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


_DRAFT_FORMAT = """\
## draft phase 규칙
거래를 실행하지 마십시오. 게시판 글을 사용하여 예비 분석 내용을 공유하십시오.
핵심 논리, 계획된 행동과 행동의 근거를 간결하게 요약한 draft_summary 필드를 반드시 포함해야 합니다.
당신을 포함한 에이전트들은 서로 게시글 전체글이 아닌 요약본만을 읽습니다.

## 출력 형식 (반드시 이 JSON 형식을 준수)
```json
{
  "board_title": "게시판 제목",
  "board_body": "게시판 전체글",
  "draft_summary": "핵심 논리와 계획 요약"
}
```"""

EXECUTION_FORMAT = """\
## 주문 규칙
- 정수(whole-share) 단위의 주식 주문만 지원합니다. 소수점 단위 주식(fractional shares)은 주문하지 마십시오.
- side는 BUY, SELL, HOLD 중 하나입니다.
- size_ratio는 0.0~1.0 범위입니다. BUY는 sleeve_equity 대비 목표 익스포저의 공격성, SELL은 보유 수량 대비 매도 비율을 의미합니다.
- size_ratio=1은 무조건적인 풀매수(all-in)가 아니라 가장 강한 확신의 설정치입니다. 최종 체결 가능 수량은 런타임 포트폴리오와 브로커 제약에 따라 조정될 수 있습니다.
- BUY는 예상 주식 수(sleeve_equity * size_ratio / price_per_share)가 최소 1 이상이 되도록 설정하십시오. 1 미만이면 해당 티커는 HOLD를 사용하십시오.
- sleeve_state.buy_blocked가 true이거나 order_budget.max_buy_notional_krw가 0 또는 0에 가까우면 BUY를 내지 말고 SELL 또는 HOLD만 사용하십시오.
- orders 배열에는 여러 종목의 주문을 넣을 수 있습니다. 거래가 필요 없으면 빈 배열 []로 반환하십시오.

## 출력 형식 (반드시 이 JSON 형식을 준수)
```json
{
  "draft_summary": "핵심 논리와 계획 요약",
  "orders": [
    {
      "ticker": "AAPL",
      "side": "BUY",
      "size_ratio": 0.15,
      "rationale": "매수 근거",
      "strategy_refs": ["momentum", "earnings_growth"]
    }
  ]
}
```"""


def _user_prompt(context: dict[str, Any], default_universe: list[str], *, max_tool_calls: int = 50) -> str:
    """Builds compact user prompt payload for one cycle decision."""
    _ = default_universe

    phase = context.get("cycle_phase", "execution")
    phase_format = _DRAFT_FORMAT if phase == "draft" else EXECUTION_FORMAT

    analysis_funnel = context.get("analysis_funnel_prompt")
    if not isinstance(analysis_funnel, dict):
        analysis_funnel = model_facing_funnel_metrics(context.get("analysis_funnel", {}))

    payload = {
        "cycle_phase": phase,
        "performance_context": context.get("performance_context", ""),
        "active_thesis_context": context.get("active_thesis_context", ""),
        "memory_context": context.get("memory_context", ""),
        "board_context": context.get("board_context", ""),
        "market_context": context.get("market_context", context.get("market_features", [])),
        "research_context": context.get("research_context", ""),
        "portfolio": context.get("portfolio", {}),
        "ticker_names": context.get("ticker_names", {}),
        "risk_policy": context.get("risk_policy", {}),
        "order_budget": context.get("order_budget", {}),
        "analysis_funnel": analysis_funnel,
        "candidate_cases": context.get("candidate_cases", []),
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
    relation_context = str(context.get("relation_context") or "").strip()
    if relation_context:
        payload["relation_context"] = relation_context
    return (
        phase_format
        + "\n\nContext payload JSON (output JSON only):\n"
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
