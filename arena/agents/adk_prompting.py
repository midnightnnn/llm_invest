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


_EXPLORE_SHARED_FORMAT = """\
## explore phase 규칙
지금은 도구 탐색과 근거 수집 단계입니다. 거래를 실행하지 마십시오.
**중요: 서로 독립적인 도구 호출은 반드시 하나의 턴에서 병렬로 동시에 호출하십시오.**
순차 호출은 이전 결과가 다음 호출의 입력에 필요한 경우에만 사용하십시오.
도구를 통해 투자 판단에 필요한 정보를 탐색하십시오.
핵심 논리, 계획된 행동과 행동의 근거를 간결하게 요약한 explore_summary 필드를 반드시 포함해야 합니다.
이 요약은 다른 에이전트들과 공유되므로 execution에 필요한 핵심만 압축해서 적으십시오.

## 출력 형식 (반드시 이 JSON 형식을 준수)
```json
{
  "explore_summary": "핵심 논리와 계획 요약"
}
```"""

_EXPLORE_SOLO_FORMAT = """\
## explore phase 규칙
지금은 도구 탐색과 근거 수집 단계입니다. 거래를 실행하지 마십시오.
**중요: 서로 독립적인 도구 호출은 반드시 하나의 턴에서 병렬로 동시에 호출하십시오.**
순차 호출은 이전 결과가 다음 호출의 입력에 필요한 경우에만 사용하십시오.
도구를 통해 투자 판단에 필요한 정보를 탐색하십시오.

## 출력 형식 (반드시 이 JSON 형식을 준수)
```json
{
  "explore_status": "complete"
}
```"""

EXECUTION_FORMAT = """\
## 주문 규칙
- 정수(whole-share) 단위의 주식 주문만 지원합니다. 소수점 단위 주식(fractional shares)은 주문하지 마십시오.
- side는 BUY, SELL, HOLD 중 하나입니다.
- BUY 주문은 target_weight를 사용합니다. target_weight는 sleeve_equity 대비 해당 종목의 최종 목표 비중이며 0.0~1.0 범위입니다.
- SELL 주문은 sell_ratio를 사용합니다. sell_ratio는 현재 보유 수량 대비 매도 비율이며 0.0~1.0 범위입니다.
- BUY에는 target_weight만 포함하고, SELL에는 sell_ratio만 포함하십시오.
- BUY는 예상 추가 주식 수(max(sleeve_equity * target_weight - current_position_value, 0) / price_per_share)가 최소 1 이상이 되도록 설정하십시오. 1 미만이면 해당 티커는 HOLD를 사용하십시오.
- sleeve_state.buy_blocked가 true이거나 order_budget.max_buy_notional_krw가 0 또는 0에 가까우면 BUY를 내지 말고 SELL 또는 HOLD만 사용하십시오.
- orders 배열에는 여러 종목의 주문을 넣을 수 있습니다. 거래가 필요 없으면 빈 배열 []로 반환하십시오.
- 도구의 출력은 최종 판단을 대체하지 않습니다. 도구의 근거와 자신의 분석을 종합해 최적의 투자 결정을 내리십시오.

## 출력 형식 (반드시 이 JSON 형식을 준수)
```json
{
  "explore_summary": "핵심 논리와 계획 요약",
  "orders": [
    {
      "ticker": "AAPL",
      "side": "BUY",
      "target_weight": 0.15,
      "rationale": "매수 근거",
      "strategy_refs": ["momentum", "earnings_growth"]
    }
  ]
}
```"""


def _user_prompt(context: dict[str, Any], default_universe: list[str], *, max_tool_calls: int = 50) -> str:
    """Builds compact user prompt payload for one cycle decision."""
    _ = default_universe

    phase = str(context.get("cycle_phase", "execution") or "").strip().lower() or "execution"
    if phase == "explore":
        phase_format = (
            _EXPLORE_SHARED_FORMAT
            if bool(context.get("share_explore_summary"))
            else _EXPLORE_SOLO_FORMAT
        )
    else:
        phase_format = EXECUTION_FORMAT

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
        "tool_budget": {
            "max_tool_calls": max_tool_calls,
            "note": f"You have up to {max_tool_calls} tool calls. Plan accordingly and always output final JSON before exhausting your budget.",
        },
    }
    relation_context = str(context.get("relation_context") or "").strip()
    if relation_context:
        payload["relation_context"] = relation_context
    graph_context = str(context.get("graph_context") or "").strip()
    if graph_context:
        payload["graph_context"] = graph_context
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
