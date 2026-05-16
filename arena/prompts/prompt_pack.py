from __future__ import annotations

import functools
import json
import logging
import math
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


EXPLORE_SHARED_FORMAT = load_prompt_text("adk", "explore_shared_format.txt")
EXPLORE_SOLO_FORMAT = load_prompt_text("adk", "explore_solo_format.txt")
EXECUTION_FORMAT = load_prompt_text("adk", "execution_format.txt")
BOARD_FORMAT = load_prompt_text("adk", "board_format.txt")


class PromptPack:
    """Single entrypoint for ADK prompt sections and rendered prompt payloads."""

    @staticmethod
    def _round_krw(value: Any) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if not math.isfinite(number):
            number = 0.0
        return int(round(number))

    @staticmethod
    def _round_usd(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if not math.isfinite(number):
            number = 0.0
        return round(number, 2)

    @staticmethod
    def _int_or_zero(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _compact_explore_analysis_funnel(value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {"status": "none"}
        skip_reasons = value.get("skip_reasons")
        compact = {
            key: int(raw)
            for key, raw in value.items()
            if key != "skip_reasons"
            and isinstance(raw, int)
            and int(raw) != 0
        }
        if isinstance(skip_reasons, dict) and skip_reasons:
            compact["skip_reasons"] = dict(skip_reasons)
        return compact or {"status": "none"}

    @staticmethod
    def _compact_explore_tool_budget(max_tool_calls: int) -> dict[str, Any]:
        return {
            "max_tool_calls": max_tool_calls,
            "final_json_before_exhaustion": True,
        }

    @staticmethod
    def _compact_explore_risk_policy(value: Any) -> dict[str, Any]:
        raw = value if isinstance(value, dict) else {}
        compact: dict[str, Any] = {}
        for key in (
            "max_position_ratio",
            "min_cash_buffer_ratio",
            "ticker_cooldown_seconds",
            "single_share_buy_exception_enabled",
        ):
            if key in raw and raw.get(key) is not None:
                compact[key] = raw[key]
        return compact

    @staticmethod
    def _compact_explore_order_budget(value: Any) -> dict[str, Any]:
        raw = value if isinstance(value, dict) else {}
        compact: dict[str, Any] = {}
        for key in ("cash_krw", "min_cash_required_krw", "max_buy_notional_krw"):
            if key in raw and raw.get(key) is not None:
                compact[key] = PromptPack._round_krw(raw.get(key))

        caps: dict[str, int] = {}
        cap_sources = (
            ("cash", "max_buy_notional_by_cash_krw"),
            ("sleeve", "max_buy_notional_by_sleeve_krw"),
            ("turnover", "remaining_turnover_krw"),
            ("order", "max_order_krw"),
        )
        for label, key in cap_sources:
            if key in raw and raw.get(key) is not None:
                caps[label] = PromptPack._round_krw(raw.get(key))
        if caps:
            compact["buy_caps_krw"] = caps

        if "today_intents" in raw:
            compact["today_intents"] = PromptPack._int_or_zero(raw.get("today_intents"))
        if "daily_orders_cap" in raw:
            cap = raw.get("daily_orders_cap")
            if cap is None:
                compact["daily_orders"] = "unlimited"
            else:
                daily_orders: dict[str, int] = {"cap": PromptPack._int_or_zero(cap)}
                if raw.get("remaining_daily_orders") is not None:
                    daily_orders["remaining"] = PromptPack._int_or_zero(raw.get("remaining_daily_orders"))
                compact["daily_orders"] = daily_orders
        if raw.get("usd_krw_rate") is not None:
            compact["usd_krw_rate"] = PromptPack._round_krw(raw.get("usd_krw_rate"))
        if raw.get("cash_usd") is not None:
            compact["cash_usd"] = PromptPack._round_usd(raw.get("cash_usd"))
        return compact

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
        utility_agent_name: str = "investment_chat_utility",
        read_only: bool = False,
    ) -> str:
        _ = provider, model_id
        tenant = str(tenant_id or "").strip().lower() or "local"
        text = render_prompt_text(
            "investment_chat",
            "advisor_prompt.txt",
            values={
                "tenant_id": tenant,
                "provider": provider,
                "model_id": model_id,
                "utility_agent_name": utility_agent_name,
            },
        )
        if read_only:
            text = (
                text
                + "\n\n[showcase 보기 전용 세션] "
                "주문 제출과 설정 변경 도구는 이 세션에서 사용할 수 없습니다. "
                "조회/분석 도구만 사용해 답하세요."
            )
        return text

    @staticmethod
    def render_investment_chat_router_instruction(
        *,
        tenant_id: str,
        provider: str = "",
        advisor_model_id: str = "",
        cheap_model_id: str = "",
        router_model_id: str = "",
        utility_model_id: str = "",
        advisor_agent_name: str = "",
        utility_agent_name: str = "",
    ) -> str:
        tenant = str(tenant_id or "").strip().lower() or "local"
        router_model = router_model_id or cheap_model_id
        utility_model = utility_model_id or cheap_model_id
        return render_prompt_text(
            "investment_chat",
            "router_prompt.txt",
            values={
                "tenant_id": tenant,
                "provider": provider,
                "advisor_model_id": advisor_model_id or "provider default",
                "router_model_id": router_model or "provider default",
                "utility_model_id": utility_model or "provider default",
                "cheap_model_id": utility_model or "provider default",
                "advisor_agent_name": advisor_agent_name,
                "utility_agent_name": utility_agent_name,
            },
        )

    @staticmethod
    def render_investment_chat_utility_instruction(
        *,
        tenant_id: str,
        provider: str = "",
        model_id: str = "",
        advisor_agent_name: str = "",
    ) -> str:
        tenant = str(tenant_id or "").strip().lower() or "local"
        return render_prompt_text(
            "investment_chat",
            "utility_prompt.txt",
            values={
                "tenant_id": tenant,
                "provider": provider,
                "model_id": model_id or "provider default",
                "advisor_agent_name": advisor_agent_name,
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
        active_thesis_context = (
            context.get(f"active_thesis_context_{phase}")
            or context.get("active_thesis_context", "")
        )
        analysis_funnel = context.get("analysis_funnel_prompt")
        if not isinstance(analysis_funnel, dict):
            analysis_funnel = model_facing_funnel_metrics(context.get("analysis_funnel", {}))

        tool_budget = {
            "max_tool_calls": max_tool_calls,
            "note": f"You have up to {max_tool_calls} tool calls. Plan accordingly and always output final JSON before exhausting your budget.",
        }
        if phase == "explore":
            payload = {
                "cycle_phase": phase,
                "analysis_funnel": PromptPack._compact_explore_analysis_funnel(analysis_funnel),
                "tool_budget": PromptPack._compact_explore_tool_budget(max_tool_calls),
            }
            for key in (
                "performance_context",
                "active_thesis_context",
                "memory_context",
                "board_context",
                "research_context",
                "ticker_names",
                "candidate_cases",
                "decision_frame",
                "investment_style_context",
            ):
                value = active_thesis_context if key == "active_thesis_context" else context.get(key)
                if value:
                    payload[key] = value
            risk_policy = PromptPack._compact_explore_risk_policy(context.get("risk_policy", {}))
            if risk_policy:
                payload["risk_policy"] = risk_policy
            order_budget = PromptPack._compact_explore_order_budget(context.get("order_budget", {}))
            if order_budget:
                payload["order_budget"] = order_budget
            positions_brief = context.get("positions_brief")
            if positions_brief:
                payload["positions_brief"] = positions_brief
                portfolio = context.get("portfolio", {})
                positions = portfolio.get("positions") if isinstance(portfolio, dict) else {}
                held_tickers = {
                    str(ticker or "").strip().upper()
                    for ticker in (positions.keys() if isinstance(positions, dict) else [])
                    if str(ticker or "").strip()
                }
                market_context = context.get("market_context", context.get("market_features", []))
                if isinstance(market_context, list) and held_tickers:
                    nonheld_market_rows = [
                        row for row in market_context
                        if not isinstance(row, dict)
                        or str(row.get("ticker") or "").strip().upper() not in held_tickers
                    ]
                    if nonheld_market_rows:
                        payload["market_context"] = nonheld_market_rows
            else:
                portfolio = context.get("portfolio", {})
                market_context = context.get("market_context", context.get("market_features", []))
                if portfolio:
                    payload["portfolio"] = portfolio
                if market_context:
                    payload["market_context"] = market_context
        else:
            payload = {
                "cycle_phase": phase,
                "performance_context": context.get("performance_context", ""),
                "active_thesis_context": active_thesis_context,
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
                "tool_budget": tool_budget,
            }
        relation_context = str(context.get("relation_context") or "").strip()
        if relation_context:
            payload["relation_context"] = relation_context
        graph_context = str(context.get("graph_context") or "").strip()
        if graph_context:
            payload["graph_context"] = graph_context
        runtime_clock = context.get("_runtime_clock")
        if isinstance(runtime_clock, dict) and runtime_clock:
            payload["_runtime_clock"] = runtime_clock
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
        payload = {
            "order_budget": PromptPack._compact_explore_order_budget(context.get("order_budget", {})),
            "risk_policy": PromptPack._compact_explore_risk_policy(context.get("risk_policy", {})),
            "analysis_funnel": PromptPack._compact_explore_analysis_funnel(
                model_facing_funnel_metrics(analysis_funnel)
            ),
            "tool_budget": PromptPack._compact_explore_tool_budget(max_tool_events),
        }
        candidate_cases = context.get("candidate_cases", [])
        if candidate_cases:
            payload["candidate_cases"] = candidate_cases
        decision_frame = str(context.get("decision_frame") or "").strip()
        if decision_frame:
            payload["decision_frame"] = decision_frame
        parts += [
            "",
            EXECUTION_FORMAT,
            "",
            json.dumps(safe_json(payload), ensure_ascii=False),
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
