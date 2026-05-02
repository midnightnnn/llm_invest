from __future__ import annotations

import functools
import json
import logging
from datetime import date, datetime
from typing import Any, Callable

from arena.agents.adk_runner_state import model_facing_funnel_metrics
from arena.config import AgentConfig
from arena.data.bq import BigQueryRepository
from arena.prompts.loader import load_prompt_text, render_prompt_text
from arena.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def safe_json(value: Any) -> Any:
    """Converts nested values into JSON-serializable primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    return value


EXPLORE_SHARED_FORMAT = """\
## explore phase 규칙
지금은 도구 탐색과 근거 수집 단계입니다. 거래를 실행하지 마십시오.
**중요: 서로 독립적인 도구 호출은 반드시 하나의 턴에서 병렬로 동시에 호출하십시오.**
도구를 통해 투자 판단에 필요한 정보를 탐색하십시오.
핵심 논리, 계획된 행동과 행동의 근거를 간결하게 요약한 explore_summary 필드를 반드시 포함해야 합니다.
이 요약은 다른 에이전트들과 공유되며 핵심만 압축해서 적으십시오.

## 출력 형식 (반드시 이 JSON 형식을 준수)
```json
{
  "explore_summary": "핵심 논리와 계획 요약"
}
```"""


EXPLORE_SOLO_FORMAT = """\
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


BOARD_FORMAT = """\
cycle_phase: board

## 게시글 규칙
수치는 사실만을 명시하세요.

## 출력 형식 (반드시 이 JSON 형식을 준수)
```json
{
  "board_title": "게시판 제목",
  "board_body": "게시판 전체글"
}
```"""


class PromptPack:
    """Single entrypoint for ADK prompt sections and rendered prompt payloads."""

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def file_core_prompt() -> str:
        text = load_prompt_text("adk", "core_prompt.txt")
        if "{agent_id}" not in text:
            raise RuntimeError("Invalid core prompt: missing '{agent_id}' placeholder")
        return text

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def file_user_prompt_default() -> str:
        return load_prompt_text("adk", "system_prompt.txt")

    @staticmethod
    def load_prompt_part(
        config_key: str,
        file_fallback: Callable[[], str],
        repo: BigQueryRepository | None = None,
        tenant_id: str = "local",
    ) -> str:
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

    @staticmethod
    def render_system_prompt(
        agent_id: str,
        *,
        repo: BigQueryRepository | None = None,
        tenant_id: str = "local",
        agent_config: AgentConfig | None = None,
        target_market: str = "us",
    ) -> str:
        core = PromptPack.file_core_prompt()
        if agent_config and agent_config.system_prompt:
            user = agent_config.system_prompt
        else:
            user = PromptPack.load_prompt_part(
                "system_prompt",
                PromptPack.file_user_prompt_default,
                repo=repo,
                tenant_id=tenant_id,
            )
        return core.replace("{agent_id}", agent_id).replace("{target_market}", target_market) + "\n\n" + user

    @staticmethod
    def render_investment_chat_instruction(
        *,
        tenant_id: str,
        provider: str = "",
        model_id: str = "",
    ) -> str:
        _ = provider, model_id
        tenant = str(tenant_id or "").strip().lower() or "local"
        return render_prompt_text(
            "investment_chat",
            "system_prompt.txt",
            values={
                "tenant_id": tenant,
                "provider": provider,
                "model_id": model_id,
            },
        )

    @staticmethod
    def phase_format(context: dict[str, Any]) -> str:
        phase = str(context.get("cycle_phase", "execution") or "").strip().lower() or "execution"
        if phase == "explore":
            return EXPLORE_SHARED_FORMAT if bool(context.get("share_explore_summary")) else EXPLORE_SOLO_FORMAT
        return EXECUTION_FORMAT

    @staticmethod
    def decision_payload(
        context: dict[str, Any],
        *,
        max_tool_calls: int,
    ) -> dict[str, Any]:
        phase = str(context.get("cycle_phase", "execution") or "").strip().lower() or "execution"
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
        return payload

    @staticmethod
    def render_decision_prompt(
        context: dict[str, Any],
        default_universe: list[str],
        *,
        max_tool_calls: int = 50,
    ) -> str:
        _ = default_universe
        payload = PromptPack.decision_payload(context, max_tool_calls=max_tool_calls)
        return (
            PromptPack.phase_format(context)
            + "\n\nContext payload JSON (output JSON only):\n"
            + json.dumps(safe_json(payload), ensure_ascii=False)
        )

    @staticmethod
    def render_resume_prompt(
        context: dict[str, Any],
        *,
        analysis_funnel: dict[str, Any],
        max_tool_events: int,
    ) -> str:
        board_ctx = str(context.get("board_context") or "").strip()
        parts = [
            "cycle_phase: execution",
            "",
            "이전 explore 단계의 분석과 도구 호출 결과를 바탕으로 최종 주문을 결정합니다.",
            "필요시 추가 도구 호출도 가능합니다.",
        ]
        if board_ctx:
            parts += ["", "[다른 에이전트 의견]", board_ctx]
        parts += [
            "",
            EXECUTION_FORMAT,
            "",
            json.dumps(
                safe_json(
                    {
                        "order_budget": context.get("order_budget", {}),
                        "risk_policy": context.get("risk_policy", {}),
                        "analysis_funnel": model_facing_funnel_metrics(analysis_funnel),
                        "candidate_cases": context.get("candidate_cases", []),
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

    @staticmethod
    def render_board_prompt(orders_summary: str) -> str:
        return "\n".join([BOARD_FORMAT, "", str(orders_summary or "")])

    @staticmethod
    def tool_catalog_payload(
        registry: ToolRegistry,
        *,
        disabled_tool_ids: set[str],
        mcp_toolset_count: int = 0,
    ) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        disabled = {str(tool_id or "").strip() for tool_id in disabled_tool_ids}
        for entry in registry.list_entries(require_callable=True):
            if str(entry.tool_id or "").strip() in disabled:
                continue
            tools.append(
                {
                    "tool_id": entry.tool_id,
                    "name": entry.name,
                    "category": entry.category,
                    "tier": entry.tier,
                    "description": entry.description,
                }
            )
        if mcp_toolset_count > 0:
            tools.append(
                {
                    "tool_id": "mcp_toolsets",
                    "name": "MCP toolsets",
                    "category": "external",
                    "tier": "optional",
                    "description": f"{mcp_toolset_count} configured MCP toolset(s) exposed through ADK.",
                }
            )
        return tools
