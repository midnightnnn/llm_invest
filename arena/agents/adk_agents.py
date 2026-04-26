from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import threading
import time
import warnings
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from google.adk import Runner

from arena.agents.adk_agent_flow import (
    cycle_phase,
    explore_phase_output,
    execution_phase_output,
    execution_resume_session_id,
    extract_decision_payload,
    retry_policy_from_env,
)
from arena.agents.adk_context_tools import _ContextTools
from arena.agents.adk_models import (
    _has_credentials,
    _normalize_gemini_model,
    _resolve_model,
)
from arena.agents.adk_order_support import (
    build_order_intents,
    format_execution_summary,
    format_orders_summary,
    latest_rows,
    market_row_by_ticker,
)
from arena.agents.adk_prompting import (
    _file_core_prompt,
    _file_user_prompt_default,
    _load_prompt_part,
    _safe_json,
    _system_prompt,
    _tool_category_counts,
    _user_prompt,
)
from arena.agents.adk_decision_flow import (
    build_board_prompt,
    build_tool_summary_memory_record,
    parse_board_response,
    parse_decision_response,
    prepare_decision_prompt,
    tag_phase_tool_events,
)
from arena.agents.adk_runner_runtime import (
    AdkToolBudgetExceeded,
    collect_response_text,
    persist_tool_summary_memory,
    search_tool_memories,
)
from arena.agents.adk_runner_bootstrap import (
    build_agent,
    build_run_config,
    build_runner_runtime,
    initialize_base_session,
    resolve_adk_tools,
    resolve_max_tool_events,
    runner_identity,
)
from arena.agents.adk_runner_state import (
    candidate_cases,
    discovered_candidate_tickers,
    funnel_metrics,
    model_facing_funnel_metrics,
    opportunity_working_set,
    record_candidate_executions,
    record_candidate_order_feedback,
    record_candidate_orders,
    tickers_from_tool_args,
    tickers_from_tool_result,
    unresolved_candidates,
    update_candidate_ledger,
)
from arena.agents.adk_tool_compaction import _compact_tool_result_for_prompt
from arena.agents.adk_tool_config import (
    _load_disabled_tool_ids,
    _resolve_disabled_tool_ids,
)
from arena.agents.base import AgentOutput, TradingAgent
from arena.config import AgentConfig, Settings, normalize_agent_settings
from arena.data.bq import BigQueryRepository
from arena.logging_utils import event_extra, failure_extra
from arena.memory.policy import memory_embed_cache_max
from arena.models import BoardPost, ExecutionReport, OrderIntent
from arena.tools.default_registry import build_default_registry
from arena.tools.registry import ToolEntry, ToolRegistry
from arena.providers.registry import default_model_for_provider, get_provider_spec

logger = logging.getLogger(__name__)

# Reduce noisy third-party warnings during iterative trading cycles.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    message="coroutine 'close_litellm_async_clients' was never awaited",
    category=RuntimeWarning,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stable_json(value: Any) -> str:
    return json.dumps(_safe_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _agent_config_payload(agent_config: AgentConfig | None) -> dict[str, Any]:
    if agent_config is None:
        return {}
    model_dump = getattr(agent_config, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
        return payload if isinstance(payload, dict) else {}
    if is_dataclass(agent_config):
        payload = asdict(agent_config)
        return payload if isinstance(payload, dict) else {}
    return {}


def _short_hash(value: Any, length: int = 16) -> str:
    return _stable_hash(value)[:length]


def _extract_json_tail(text: str, *, marker: str = "") -> dict[str, Any] | None:
    """Extracts the JSON payload embedded at the end of model prompt text."""
    prompt = str(text or "")
    if not prompt:
        return None
    start = -1
    if marker and marker in prompt:
        start = prompt.find("{", prompt.find(marker))
    if start < 0:
        start = prompt.rfind("\n{")
        if start >= 0:
            start += 1
    if start < 0:
        return None
    try:
        parsed = json.loads(prompt[start:])
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _apply_tool_schema_metadata(fn, *, entry: ToolEntry, sig: inspect.Signature):
    """Applies canonical registry metadata to the runtime tool wrapper."""
    name = str(entry.name or entry.tool_id or getattr(fn, "__name__", "tool")).strip() or "tool"
    description = str(entry.description or "").strip()
    fn.__name__ = name
    fn.__qualname__ = name
    if description:
        fn.__doc__ = description
    fn.__signature__ = sig
    return fn


def _is_retryable_adk_error(exc: Exception) -> bool:
    """Returns True when ADK/model failure looks transient."""
    text = f"{type(exc).__name__}: {exc}".strip().lower()
    if not text:
        return False
    if (
        "adk coroutine timed out after" in text
        or "adk call timed out after" in text
        or "adk tool-budget finalization timed out" in text
        or "adk tool budget exceeded" in text
    ):
        return False
    markers = [
        "resource_exhausted",
        "resource exhausted",
        "429",
        "deadline",
        "timed out",
        "unavailable",
        "internal",
        "temporarily",
        "jsondecodeerror",
        "expecting value",
        "extra data",
        "empty response",
    ]
    return any(marker in text for marker in markers)


def _extract_known_ticker_names(context: dict[str, Any]) -> dict[str, str]:
    """Collects explicit ticker_name facts from context for later board validation."""
    out: dict[str, str] = {}

    def add(ticker: object, name: object) -> None:
        token = str(ticker or "").strip().upper()
        label = str(name or "").strip()
        if token and label and token not in out:
            out[token] = label

    portfolio = context.get("portfolio")
    if isinstance(portfolio, dict):
        positions = portfolio.get("positions")
        if isinstance(positions, dict):
            for ticker, row in positions.items():
                if isinstance(row, dict):
                    add(ticker, row.get("ticker_name"))
        elif isinstance(positions, list):
            for row in positions:
                if isinstance(row, dict):
                    add(row.get("ticker"), row.get("ticker_name"))

    performance = context.get("performance")
    if isinstance(performance, dict):
        positions = performance.get("positions")
        if isinstance(positions, list):
            for row in positions:
                if isinstance(row, dict):
                    add(row.get("ticker"), row.get("ticker_name"))

    for key in ("market_features", "opportunity_working_set"):
        rows = context.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                add(row.get("ticker"), row.get("ticker_name"))

    ticker_names = context.get("ticker_names")
    if isinstance(ticker_names, dict):
        for ticker, name in ticker_names.items():
            add(ticker, name)

    return out


class _ADKDecisionRunner:
    """Runs one ReAct decision loop: the LLM autonomously calls tools until ready to decide."""
    _shared_loop: asyncio.AbstractEventLoop | None = None
    _shared_loop_thread: threading.Thread | None = None
    _shared_loop_lock = threading.Lock()

    @classmethod
    def _acquire_shared_loop(cls) -> asyncio.AbstractEventLoop:
        """Returns a process-wide event loop running on a dedicated daemon thread."""
        with cls._shared_loop_lock:
            loop = cls._shared_loop
            thread = cls._shared_loop_thread
            if loop is not None and not loop.is_closed() and thread is not None and thread.is_alive():
                return loop

            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def _run_loop() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()

            thread = threading.Thread(target=_run_loop, name="arena-adk-loop", daemon=True)
            thread.start()
            ready.wait(timeout=2.0)
            cls._shared_loop = loop
            cls._shared_loop_thread = thread
            return loop

    def __init__(
        self,
        *,
        agent_id: str,
        provider: str,
        settings: Settings,
        repo: BigQueryRepository,
        registry: ToolRegistry,
        tenant_id: str = "local",
        agent_config: AgentConfig | None = None,
    ):
        self.agent_id = agent_id
        self.provider = provider
        self.settings = settings
        self.repo = repo
        self._registry = registry.clone()
        self.tenant_id = str(tenant_id or "").strip().lower() or "local"
        self._agent_config = agent_config

        from arena.memory.store import MemoryStore
        from arena.memory.vector import VectorStore
        vector_store = VectorStore(
            project=repo.project,
            location=repo.location,
            embed_cache_max=memory_embed_cache_max(settings.memory_policy),
        )
        self._memory_store = MemoryStore(
            repo=repo,
            vector_store=vector_store,
            trading_mode=settings.trading_mode,
            memory_policy=settings.memory_policy,
        )

        identity = runner_identity(self.agent_id)
        self._app_name = identity.app_name
        self._user_id = identity.user_id
        self._session_id = identity.session_id
        self._max_tool_events = resolve_max_tool_events(self.settings)
        self._run_config = build_run_config(self._max_tool_events)

        self._toolbox = _ContextTools(
            repo=repo,
            settings=settings,
            agent_id=agent_id,
            memory_store=self._memory_store,
            tenant_id=self.tenant_id,
        )
        self._registry.bind("search_past_experiences", self._toolbox.search_past_experiences)
        self._registry.bind("search_peer_lessons", self._toolbox.search_peer_lessons)
        self._registry.bind("get_research_briefing", self._toolbox.get_research_briefing)
        self._registry.bind("portfolio_diagnosis", self._toolbox.portfolio_diagnosis)
        self._registry.bind("trade_performance", self._toolbox.trade_performance)
        self._tool_events: list[dict[str, Any]] = []
        self._seen_memory_ids: set[str] = set()
        self._wrapped_tool_names: set[str] = set()
        self._candidate_ledger: dict[str, dict[str, Any]] = {}
        self._held_tickers_cache: set[str] = set()
        self._current_phase: str = "unknown"
        self._current_context: dict[str, Any] | None = None
        self._prompt_snapshots: list[dict[str, Any]] = []
        self._llm_call_ids_by_phase: dict[str, str] = {}
        self._latest_llm_call_id: str = ""
        tool_resolution = resolve_adk_tools(
            repo=self.repo,
            tenant_id=self.tenant_id,
            agent_config=self._agent_config,
            registry=self._registry,
            settings=self.settings,
            agent_id=self.agent_id,
            tool_events=self._tool_events,
            update_candidate_ledger=self._update_candidate_ledger,
            search_tool_memories=self._search_tool_memories,
            apply_tool_schema_metadata=_apply_tool_schema_metadata,
        )
        self._disabled_tool_ids = set(tool_resolution.disabled_tool_ids)
        self._mcp_toolset_count = int(tool_resolution.mcp_toolset_count)
        self._wrapped_tool_names = tool_resolution.wrapped_tool_names
        self._agent = build_agent(
            agent_id=self.agent_id,
            provider=self.provider,
            settings=self.settings,
            repo=self.repo,
            tenant_id=self.tenant_id,
            agent_config=self._agent_config,
            max_tool_events=self._max_tool_events,
            adk_tools=tool_resolution.adk_tools,
        )
        self._system_prompt_snapshot = str(getattr(self._agent, "instruction", "") or "")

        # Share one event loop across all runners in-process to keep third-party
        # async workers (e.g., litellm logging worker) bound to a single loop.
        self._loop = self._acquire_shared_loop()
        runtime = build_runner_runtime(
            provider=self.provider,
            agent=self._agent,
            identity=identity,
        )
        self._session_service = runtime.session_service
        self._runner = runtime.runner
        initialize_base_session(
            session_service=self._session_service,
            identity=identity,
            run_on_loop=self._run_on_loop,
        )

    def _available_tools_payload(self) -> list[dict[str, Any]]:
        """Returns the built-in tool catalog visible to the model for prompt inspection."""
        tools: list[dict[str, Any]] = []
        for entry in self._registry.list_entries(require_callable=True):
            if str(entry.tool_id or "").strip() in self._disabled_tool_ids:
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
        if self._mcp_toolset_count > 0:
            tools.append(
                {
                    "tool_id": "mcp_toolsets",
                    "name": "MCP toolsets",
                    "category": "external",
                    "tier": "optional",
                    "description": f"{self._mcp_toolset_count} configured MCP toolset(s) exposed through ADK.",
                }
            )
        return tools

    def _configured_model_id(self) -> str:
        if self._agent_config and str(self._agent_config.model or "").strip():
            return str(self._agent_config.model or "").strip()
        provider_token = str(self.provider or "").strip().lower()
        spec = get_provider_spec(provider_token)
        provider_id = spec.provider_id if spec is not None else provider_token
        return default_model_for_provider(self.settings, provider_id)

    def _new_llm_call_id(self, phase: str) -> str:
        phase_token = str(phase or "unknown").strip().lower() or "unknown"
        return f"llm_{phase_token}_{uuid4().hex[:12]}"

    def llm_call_id_for_phase(self, phase: str | None = None) -> str:
        phase_token = str(phase or "").strip().lower()
        if phase_token:
            return self._llm_call_ids_by_phase.get(phase_token, "")
        return self._latest_llm_call_id

    def _append_audit_rows(self, method_name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        writer = getattr(self.repo, method_name, None)
        if not callable(writer):
            return
        try:
            writer(rows, tenant_id=self.tenant_id)
        except Exception as exc:
            logger.warning(
                "[yellow]LLM audit write failed[/yellow] agent=%s method=%s err=%s",
                self.agent_id,
                method_name,
                str(exc),
            )

    def _context_refs_for_audit(
        self,
        *,
        llm_call_id: str,
        context: dict[str, Any] | None,
        phase: str,
        cycle_id: str,
        created_at: datetime,
    ) -> list[dict[str, Any]]:
        if not isinstance(context, dict):
            return []

        rows: list[dict[str, Any]] = []

        def add_ref(
            *,
            source_table: str,
            source_id: str,
            item: Any,
            context_role: str,
            prompt_section: str,
            rank: int,
            source_ts: Any = None,
            used_in_prompt: bool = True,
        ) -> None:
            source_token = str(source_id or "").strip()
            if not source_token:
                return
            ref_id = "ctx_" + _short_hash(
                {
                    "llm_call_id": llm_call_id,
                    "source_table": source_table,
                    "source_id": source_token,
                    "context_role": context_role,
                    "prompt_section": prompt_section,
                    "rank": rank,
                },
                20,
            )
            rows.append(
                {
                    "llm_call_id": llm_call_id,
                    "context_ref_id": ref_id,
                    "cycle_id": cycle_id,
                    "created_at": created_at,
                    "agent_id": self.agent_id,
                    "phase": phase,
                    "source_table": source_table,
                    "source_id": source_token,
                    "source_ts": source_ts,
                    "source_hash": _stable_hash(item),
                    "context_role": context_role,
                    "prompt_section": prompt_section,
                    "rank": rank,
                    "used_in_prompt": used_in_prompt,
                    "detail_json": item if isinstance(item, dict) else {"value": item},
                }
            )

        for idx, row in enumerate(context.get("memory_events") or [], start=1):
            if isinstance(row, dict):
                add_ref(
                    source_table="agent_memory_events",
                    source_id=str(row.get("event_id") or ""),
                    item=row,
                    context_role="memory",
                    prompt_section="memory_context",
                    rank=idx,
                    source_ts=row.get("created_at"),
                )

        for idx, row in enumerate(context.get("board_posts") or [], start=1):
            if isinstance(row, dict):
                add_ref(
                    source_table="board_posts",
                    source_id=str(row.get("post_id") or ""),
                    item=row,
                    context_role="board",
                    prompt_section="board_context",
                    rank=idx,
                    source_ts=row.get("created_at"),
                )

        for idx, row in enumerate(context.get("research_briefings") or [], start=1):
            if isinstance(row, dict):
                add_ref(
                    source_table="research_briefings",
                    source_id=str(row.get("briefing_id") or ""),
                    item=row,
                    context_role="research",
                    prompt_section="research_context",
                    rank=idx,
                    source_ts=row.get("created_at"),
                )

        for idx, row in enumerate(context.get("active_theses") or [], start=1):
            if isinstance(row, dict):
                add_ref(
                    source_table="agent_memory_events",
                    source_id=str(row.get("event_id") or row.get("semantic_key") or ""),
                    item=row,
                    context_role="active_thesis",
                    prompt_section="active_thesis_context",
                    rank=idx,
                    source_ts=row.get("created_at"),
                )

        for idx, row in enumerate(context.get("graph_events") or [], start=1):
            if not isinstance(row, dict):
                continue
            source_table = "memory_graph_edges" if row.get("edge_id") else "memory_graph_nodes"
            add_ref(
                source_table=source_table,
                source_id=str(row.get("edge_id") or row.get("node_id") or row.get("event_id") or ""),
                item=row,
                context_role="graph",
                prompt_section="graph_context",
                rank=idx,
                source_ts=row.get("created_at"),
            )

        for idx, row in enumerate(context.get("market_features") or [], start=1):
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or row.get("symbol") or "").strip().upper()
            source = str(row.get("source") or row.get("data_source") or "").strip().lower()
            as_of = str(
                row.get("date")
                or row.get("as_of_date")
                or row.get("feature_date")
                or row.get("created_at")
                or ""
            ).strip()
            source_id = "|".join(part for part in [ticker, source, as_of] if part) or f"rank_{idx}"
            add_ref(
                source_table="market_features",
                source_id=source_id,
                item=row,
                context_role="market",
                prompt_section="market_context",
                rank=idx,
                source_ts=row.get("created_at") or row.get("as_of_ts"),
            )

        return rows

    def _tool_event_rows_for_audit(
        self,
        *,
        llm_call_id: str,
        events: list[dict[str, Any]],
        phase: str,
        cycle_id: str,
        default_created_at: datetime,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                continue
            tool_name = str(event.get("tool") or "").strip()
            if not tool_name:
                continue
            started_at = event.get("started_at")
            created_at = default_created_at
            try:
                if started_at is not None:
                    created_at = datetime.fromtimestamp(float(started_at) / 1000.0, tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                created_at = default_created_at
            result = event.get("result") if event.get("result") is not None else event.get("result_preview")
            tool_event_id = "tool_" + _short_hash(
                {
                    "llm_call_id": llm_call_id,
                    "idx": idx,
                    "tool": tool_name,
                    "started_at": started_at,
                    "args": event.get("args"),
                },
                20,
            )
            rows.append(
                {
                    "llm_call_id": llm_call_id,
                    "tool_event_id": tool_event_id,
                    "cycle_id": cycle_id,
                    "created_at": created_at,
                    "agent_id": self.agent_id,
                    "phase": str(event.get("phase") or phase or "").strip().lower() or phase,
                    "tool_name": tool_name,
                    "source": event.get("source"),
                    "args_json": event.get("args") or {},
                    "model_visible_result_json": result,
                    "raw_result_hash": _stable_hash(result) if result is not None else None,
                    "elapsed_ms": event.get("elapsed_ms"),
                    "error": event.get("error"),
                }
            )
        return rows

    def _record_llm_interaction_audit(
        self,
        *,
        llm_call_id: str,
        phase: str,
        session_id: str,
        resumed: bool,
        context: dict[str, Any] | None,
        prompt: str,
        created_at: datetime,
        completed_at: datetime,
        status: str,
        response_text: str = "",
        response_json: dict[str, Any] | None = None,
        token_usage: dict[str, Any] | None = None,
        error_message: str = "",
        tool_events: list[dict[str, Any]] | None = None,
    ) -> None:
        cycle_id = str((context or {}).get("cycle_id") or "").strip()
        context_payload = _extract_json_tail(prompt, marker="Context payload JSON")
        context_sections = self._prompt_context_sections(context)
        available_tools = self._available_tools_payload()
        row = {
            "llm_call_id": llm_call_id,
            "cycle_id": cycle_id,
            "created_at": created_at,
            "completed_at": completed_at,
            "agent_id": self.agent_id,
            "provider": self.provider,
            "model": self._configured_model_id(),
            "phase": phase,
            "session_id": session_id,
            "resume_session": resumed,
            "trading_mode": self.settings.trading_mode,
            "status": status,
            "system_prompt": self._system_prompt_snapshot,
            "user_prompt": prompt,
            "context_payload_json": context_payload,
            "context_sections_json": context_sections,
            "available_tools_json": available_tools,
            "response_text": response_text or None,
            "response_json": response_json,
            "token_usage_json": token_usage or {},
            "request_hash": _stable_hash(
                {
                    "system_prompt": self._system_prompt_snapshot,
                    "user_prompt": prompt,
                    "available_tools": available_tools,
                }
            ),
            "prompt_version": "adk_prompt_v1",
            "context_builder_version": "context_builder_v1",
            "settings_hash": _stable_hash(
                {
                    "trading_mode": self.settings.trading_mode,
                    "kis_target_market": getattr(self.settings, "kis_target_market", ""),
                    "memory_policy": getattr(self.settings, "memory_policy", None),
                    "agent_config": _agent_config_payload(self._agent_config),
                }
            ),
            "latency_ms": int(max(0.0, (completed_at - created_at).total_seconds() * 1000.0)),
            "error_message": error_message,
        }
        self._append_audit_rows("append_llm_interactions", [row])
        self._append_audit_rows(
            "append_llm_context_refs",
            self._context_refs_for_audit(
                llm_call_id=llm_call_id,
                context=context,
                phase=phase,
                cycle_id=cycle_id,
                created_at=created_at,
            ),
        )
        self._append_audit_rows(
            "append_llm_tool_events",
            self._tool_event_rows_for_audit(
                llm_call_id=llm_call_id,
                events=tool_events or [],
                phase=phase,
                cycle_id=cycle_id,
                default_created_at=completed_at,
            ),
        )

    def record_artifact_links(
        self,
        *,
        phase: str,
        cycle_id: str,
        artifacts: list[dict[str, Any]],
    ) -> None:
        phase_token = str(phase or "").strip().lower() or "unknown"
        llm_call_id = self.llm_call_id_for_phase(phase_token)
        if not llm_call_id or not artifacts:
            return
        created_at = _utc_now()
        rows: list[dict[str, Any]] = []
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            artifact_table = str(artifact.get("artifact_table") or "").strip()
            artifact_id = str(artifact.get("artifact_id") or "").strip()
            if not artifact_table or not artifact_id:
                continue
            link_id = "alink_" + _short_hash(
                {
                    "llm_call_id": llm_call_id,
                    "artifact_table": artifact_table,
                    "artifact_id": artifact_id,
                    "artifact_role": artifact.get("artifact_role"),
                },
                20,
            )
            rows.append(
                {
                    "llm_call_id": llm_call_id,
                    "artifact_link_id": link_id,
                    "cycle_id": str(cycle_id or "").strip(),
                    "created_at": created_at,
                    "agent_id": self.agent_id,
                    "phase": phase_token,
                    "artifact_table": artifact_table,
                    "artifact_id": artifact_id,
                    "artifact_role": artifact.get("artifact_role"),
                    "detail_json": artifact.get("detail_json"),
                }
            )
        self._append_audit_rows("append_llm_artifact_links", rows)

    @staticmethod
    def _prompt_context_sections(context: dict[str, Any] | None) -> dict[str, Any]:
        """Captures the high-level context blocks operators expect to see in prompt details."""
        if not isinstance(context, dict):
            return {}
        return _safe_json(
            {
                "portfolio_context": context.get("portfolio", {}),
                "market_context": context.get("market_context", context.get("market_features", [])),
                "board_context": context.get("board_context") or context.get("board_posts", []),
                "research_context": context.get("research_context", ""),
                "relation_context": context.get("relation_context", ""),
                "graph_context": context.get("graph_context", ""),
                "memory_context": context.get("memory_context", ""),
            }
        )

    def _seed_seen_memory_ids(self, context: dict[str, Any]) -> None:
        """Seeds per-cycle seen memory ids from initial prompt-injected memories."""
        rows = context.get("memory_events") or []
        seeded: set[str] = set()
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                event_id = str(row.get("event_id") or "").strip()
                if event_id:
                    seeded.add(event_id)
        self._seen_memory_ids.update(seeded)

    def _extract_held_tickers(self, context: dict[str, Any]) -> set[str]:
        """Returns normalized held tickers from the current portfolio context."""
        portfolio = context.get("portfolio") or {}
        if not isinstance(portfolio, dict):
            return set()
        positions = portfolio.get("positions") or {}
        if not isinstance(positions, dict):
            return set()
        return {str(t).strip().upper() for t in positions if str(t).strip()}

    @staticmethod
    def _tickers_from_tool_args(args: dict[str, Any]) -> list[str]:
        return tickers_from_tool_args(args)

    @staticmethod
    def _tickers_from_tool_result(result: Any) -> list[str]:
        return tickers_from_tool_result(result)

    def _sync_pipeline_context(self) -> None:
        """Syncs candidate/funnel telemetry into the live per-cycle context object."""
        if not isinstance(self._current_context, dict):
            return
        self._current_context["_candidate_tickers"] = self._unresolved_candidates()
        self._current_context["_discovered_candidate_tickers"] = self._discovered_candidates()
        self._current_context["opportunity_working_set"] = self._opportunity_working_set()
        metrics = self._funnel_metrics()
        self._current_context["analysis_funnel"] = metrics
        self._current_context["analysis_funnel_prompt"] = model_facing_funnel_metrics(metrics)
        self._current_context["candidate_cases"] = self._candidate_cases()
        self._current_context["decision_frame"] = self._decision_frame()

    def _update_candidate_ledger(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        update_candidate_ledger(
            self._candidate_ledger,
            self._held_tickers_cache,
            self._current_phase,
            tool_name=tool_name,
            args=args,
            result=result,
        )
        self._sync_pipeline_context()

    def _unresolved_candidates(self) -> list[str]:
        return unresolved_candidates(self._candidate_ledger)

    def _discovered_candidates(self) -> list[str]:
        return discovered_candidate_tickers(self._candidate_ledger)

    def _opportunity_working_set(self) -> list[dict[str, Any]]:
        if not bool(getattr(getattr(self, "settings", None), "autonomy_working_set_enabled", True)):
            return []
        return opportunity_working_set(self._candidate_ledger)

    def _candidate_cases(self) -> list[dict[str, Any]]:
        if not bool(getattr(getattr(self, "settings", None), "autonomy_working_set_enabled", True)):
            return []
        return candidate_cases(self._candidate_ledger)

    def _decision_frame(self) -> str:
        """Returns a light-touch decision frame without forcing exploration."""
        if not bool(getattr(getattr(self, "settings", None), "autonomy_opportunity_context_enabled", True)):
            return ""
        working_set = self._opportunity_working_set()
        if not working_set:
            return ""
        order_budget = self._current_context.get("order_budget") or {}
        try:
            max_buy_notional = float(order_budget.get("max_buy_notional_krw") or 0.0)
        except (TypeError, ValueError):
            max_buy_notional = 0.0
        if max_buy_notional > 0:
            return (
                "Compare any self-discovered opportunities against cash or the weakest current exposure. "
                "Keeping existing holdings is valid if alternatives are inferior."
            )
        return (
            "Self-discovered opportunities exist, but buying budget is currently constrained. "
            "Use them mainly for triage and relative comparison."
        )

    def _funnel_metrics(self) -> dict[str, int]:
        return funnel_metrics(
            self._candidate_ledger,
            self._tool_events,
            self._held_tickers_cache,
        )

    def _funnel_metrics_for_prompt(self) -> dict[str, Any]:
        return model_facing_funnel_metrics(self._funnel_metrics())

    def record_candidate_orders(self, orders: list[dict[str, Any]]) -> None:
        """Syncs raw model orders into the candidate ledger for funnel telemetry."""
        record_candidate_orders(
            self._candidate_ledger,
            current_phase=self._current_phase,
            orders=orders,
        )
        self._sync_pipeline_context()

    def record_candidate_order_feedback(self, feedback_events: list[dict[str, Any]]) -> None:
        """Syncs post-validation order feedback into the candidate ledger."""
        record_candidate_order_feedback(
            self._candidate_ledger,
            current_phase=self._current_phase,
            feedback_events=feedback_events,
        )
        self._sync_pipeline_context()

    def record_candidate_executions(self, intents: list[OrderIntent], reports: list[ExecutionReport]) -> None:
        """Syncs final execution outcomes into the candidate ledger."""
        record_candidate_executions(
            self._candidate_ledger,
            current_phase=self._current_phase,
            intents=intents,
            reports=reports,
        )
        self._sync_pipeline_context()

    def _search_tool_memories(self, query: str) -> list[dict[str, Any]] | None:
        return search_tool_memories(
            memory_store=self._memory_store,
            settings=self.settings,
            agent_id=self.agent_id,
            seen_memory_ids=self._seen_memory_ids,
            query=query,
        )

    def _persist_tool_summary_memory(self, *, summary: str, payload: dict[str, Any]) -> str | None:
        return persist_tool_summary_memory(
            memory_store=self._memory_store,
            settings=self.settings,
            repo=self.repo,
            agent_id=self.agent_id,
            summary=summary,
            payload=payload,
        )

    def _persist_candidate_memories(self, *, cycle_id: str = "") -> int:
        writer = getattr(self._memory_store, "record_candidate_memories", None)
        if not callable(writer) or not self._candidate_ledger:
            return 0
        try:
            return int(
                writer(
                    agent_id=self.agent_id,
                    candidate_ledger=self._candidate_ledger,
                    held_tickers=self._held_tickers_cache,
                    cycle_id=str(cycle_id or "").strip(),
                    phase=self._current_phase,
                )
                or 0
            )
        except Exception as exc:
            logger.warning(
                "[yellow]Candidate memory write failed[/yellow] agent=%s err=%s",
                self.agent_id,
                str(exc),
            )
            return 0

    def _record_prompt_snapshot(
        self,
        *,
        phase: str,
        session_id: str,
        prompt: str,
        resumed: bool,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Stores one application-visible prompt snapshot per phase."""
        phase_token = str(phase or "").strip().lower() or "unknown"
        prompt_text = str(prompt or "")
        if not prompt_text.strip():
            return
        snapshot = {
            "phase": phase_token,
            "session_id": str(session_id or "").strip(),
            "resume_session": bool(resumed),
            "prompt": prompt_text,
        }
        context_sections = self._prompt_context_sections(context)
        if context_sections:
            snapshot["context_sections"] = context_sections
        retained = [
            dict(item)
            for item in self._prompt_snapshots
            if str((item or {}).get("phase") or "").strip().lower() != phase_token
        ]
        retained.append(snapshot)
        phase_order = {"explore": 0, "execution": 1, "board": 2}
        retained.sort(
            key=lambda item: (
                phase_order.get(str((item or {}).get("phase") or "").strip().lower(), 99),
                str((item or {}).get("phase") or ""),
            )
        )
        self._prompt_snapshots = retained

    def _prompt_bundle_payload(self) -> dict[str, Any]:
        """Builds an app-visible prompt bundle for post-board inspection."""
        phases = [
            dict(item)
            for item in self._prompt_snapshots
            if isinstance(item, dict) and str(item.get("prompt") or "").strip()
        ]
        return {
            "system_prompt": self._system_prompt_snapshot,
            "available_tools": _safe_json(self._available_tools_payload()),
            "phases": _safe_json(phases),
            "note": (
                "Application-visible prompt bundle captured around board generation. "
                "Prior tool transcript is preserved separately in compacted tool_events."
            ),
        }

    def _run_on_loop(self, coro):
        """Schedules a coroutine on the shared ADK loop and waits for result."""
        if self._loop.is_closed():
            self._loop = self._acquire_shared_loop()
        timeout = max(float(self.settings.timeout_for("trading")) + 30.0, 60.0)
        future = None
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            if future is not None:
                future.cancel()
            raise TimeoutError(f"ADK coroutine timed out after {int(timeout)}s") from exc
        except RuntimeError as exc:
            if future is None:
                try:
                    coro.close()
                except Exception:
                    pass
                raise RuntimeError("Failed to schedule coroutine on shared ADK loop") from exc
            raise

    def _tool_budget_closing_prompt(self) -> str:
        """Builds a one-shot finalization prompt after the ReAct tool budget is exhausted."""
        phase = str(getattr(self, "_current_phase", "") or "execution").strip().lower() or "execution"
        if phase == "board":
            schema = '{\n  "board_title": "게시판 제목",\n  "board_body": "게시판 전체글"\n}'
        elif phase == "explore":
            schema = '{\n  "explore_summary": "지금까지 확인한 핵심 근거와 다음 행동 계획"\n}'
        else:
            schema = (
                '{\n'
                '  "explore_summary": "지금까지 확인한 핵심 근거와 주문 판단",\n'
                '  "orders": []\n'
                '}'
            )
        return "\n".join(
            [
                f"cycle_phase: {phase}",
                "",
                "도구 호출 예산이 끝났습니다. 더 이상 도구를 호출하지 마십시오.",
                "이미 같은 세션에 있는 컨텍스트와 도구 결과만 사용해 지금 최종 JSON을 반환하십시오.",
                "확신이 부족하면 새 도구를 호출하지 말고 보수적으로 HOLD 또는 빈 orders를 선택하십시오.",
                "",
                "## 출력 형식 (JSON only)",
                "```json",
                schema,
                "```",
            ]
        )

    async def _collect_response_text(
        self,
        runner: Runner,
        session_id: str,
        prompt: str,
        *,
        max_tool_events: int | None = None,
        run_config: Any | None = None,
    ) -> str:
        last_text, token_usage = await collect_response_text(
            runner=runner,
            user_id=self._user_id,
            session_id=session_id,
            prompt=prompt,
            run_config=run_config or self._run_config,
            max_tool_events=self._max_tool_events if max_tool_events is None else max_tool_events,
            wrapped_tool_names=self._wrapped_tool_names,
            tool_events=self._tool_events,
            agent_id=self.agent_id,
        )
        self._last_token_usage = token_usage
        return last_text

    async def _run_async(self, runner: Runner, session_id: str, prompt: str) -> str:
        """Runs one ADK call with a hard timeout and returns final text."""
        trading_timeout = int(self.settings.timeout_for("trading"))
        try:
            return await asyncio.wait_for(
                self._collect_response_text(runner, session_id, prompt),
                timeout=trading_timeout,
            )
        except AdkToolBudgetExceeded as exc:
            closing_timeout = min(max(float(trading_timeout) * 0.2, 60.0), 300.0)
            logger.warning(
                "[yellow]ADK tool budget exhausted; requesting final JSON[/yellow] agent=%s provider=%s timeout=%.0fs err=%s",
                self.agent_id,
                self.provider,
                closing_timeout,
                str(exc),
            )
            try:
                return await asyncio.wait_for(
                    self._collect_response_text(
                        runner,
                        session_id,
                        self._tool_budget_closing_prompt(),
                        max_tool_events=0,
                        run_config=build_run_config(1),
                    ),
                    timeout=closing_timeout,
                )
            except asyncio.TimeoutError as close_exc:
                raise TimeoutError(f"ADK tool-budget finalization timed out after {int(closing_timeout)}s") from close_exc
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"ADK call timed out after {trading_timeout}s") from exc

    def decide_orders(
        self,
        context: dict[str, Any],
        default_universe: list[str],
        *,
        resume_session_id: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        """Runs the ReAct loop: LLM freely explores tools, then returns (decision JSON, session_id).

        When *resume_session_id* is provided the runner continues the existing
        ADK session (explore → execution continuity) so that the model retains
        all prior tool-call history in its conversation context.
        """
        phase = str(context.get("cycle_phase") or "execution").strip().lower() or "execution"
        phase_start_idx = len(self._tool_events)
        if not resume_session_id:
            # Keep the list identity stable because wrapped tools capture it at init time.
            self._tool_events.clear()
            phase_start_idx = 0
            self._seen_memory_ids = set()
            self._candidate_ledger = {}
            self._prompt_snapshots = []
            self._llm_call_ids_by_phase = {}
            self._latest_llm_call_id = ""
        self._current_phase = phase
        self._current_context = context
        set_seen_memory_ids = getattr(self._toolbox, "set_seen_memory_ids", None)
        if callable(set_seen_memory_ids):
            set_seen_memory_ids(self._seen_memory_ids)
        self._seed_seen_memory_ids(context)
        self._toolbox.set_context(context)
        self._registry.set_context(context)
        self._held_tickers_cache = self._extract_held_tickers(context)
        self._sync_pipeline_context()
        # Clear per-cycle embedding cache to avoid stale vectors across decisions
        if self._memory_store and self._memory_store.vector_store:
            self._memory_store.vector_store.clear_embed_cache()

        session_id, prompt, needs_new_session = prepare_decision_prompt(
            context,
            default_universe,
            phase=phase,
            base_session_id=self._session_id,
            max_tool_events=self._max_tool_events,
            resume_session_id=resume_session_id,
            analysis_funnel=self._funnel_metrics_for_prompt(),
        )
        if needs_new_session:
            self._run_on_loop(
                self._session_service.create_session(
                    app_name=self._app_name,
                    user_id=self._user_id,
                    session_id=session_id,
                )
            )
        self._record_prompt_snapshot(
            phase=phase,
            session_id=session_id,
            prompt=prompt,
            resumed=bool(resume_session_id),
            context=context,
        )

        llm_call_id = self._new_llm_call_id(phase)
        self._llm_call_ids_by_phase[phase] = llm_call_id
        self._latest_llm_call_id = llm_call_id
        created_at = _utc_now()
        text = ""
        decision: dict[str, Any] | None = None
        try:
            text = self._run_on_loop(self._run_async(self._runner, session_id, prompt))
            if not text or not text.strip():
                raise RuntimeError("ADK runner returned empty response")
            decision = parse_decision_response(text)
        except Exception as exc:
            completed_at = _utc_now()
            tag_phase_tool_events(self._tool_events, phase=phase, start_idx=phase_start_idx)
            token_usage = getattr(self, "_last_token_usage", None)
            self._record_llm_interaction_audit(
                llm_call_id=llm_call_id,
                phase=phase,
                session_id=session_id,
                resumed=bool(resume_session_id),
                context=context,
                prompt=prompt,
                created_at=created_at,
                completed_at=completed_at,
                status="error",
                response_text=text,
                token_usage=token_usage if isinstance(token_usage, dict) else None,
                error_message=str(exc),
                tool_events=self._tool_events[phase_start_idx:],
            )
            raise

        completed_at = _utc_now()
        tag_phase_tool_events(self._tool_events, phase=phase, start_idx=phase_start_idx)
        token_usage = getattr(self, "_last_token_usage", None)
        self._record_llm_interaction_audit(
            llm_call_id=llm_call_id,
            phase=phase,
            session_id=session_id,
            resumed=bool(resume_session_id),
            context=context,
            prompt=prompt,
            created_at=created_at,
            completed_at=completed_at,
            status="ok",
            response_text=text,
            response_json=decision,
            token_usage=token_usage if isinstance(token_usage, dict) else None,
            tool_events=self._tool_events[phase_start_idx:],
        )

        # Persist a compact summary of ReAct tool usage for next-cycle reference.
        # Saved for every phase (explore + execution) so the board shows the full picture.
        summary_record = build_tool_summary_memory_record(
            self._tool_events,
            registry=self._registry,
            phase=phase,
            analysis_funnel=self._funnel_metrics(),
            cycle_id=str(context.get("cycle_id") or "").strip(),
            token_usage=token_usage if isinstance(token_usage, dict) else None,
        )
        if summary_record is not None:
            summary, payload = summary_record or ("", {})
            payload["llm_call_id"] = llm_call_id
            try:
                event_id = self._persist_tool_summary_memory(summary=summary, payload=payload)
                if event_id:
                    self.record_artifact_links(
                        phase=phase,
                        cycle_id=str(context.get("cycle_id") or "").strip(),
                        artifacts=[
                            {
                                "artifact_table": "agent_memory_events",
                                "artifact_id": event_id,
                                "artifact_role": "tool_summary_memory",
                            }
                        ],
                    )
            except Exception as exc:
                logger.warning(
                    "[yellow]Tool summary memory write failed[/yellow] agent=%s err=%s",
                    self.agent_id,
                    str(exc),
                )

        return decision, session_id

    def decide_board(self, session_id: str, orders_summary: str, *, cycle_id: str = "") -> dict[str, Any]:
        """Step 2: 주문 내역 기반 게시글 작성. 같은 세션 컨텍스트 활용."""
        phase = "board"
        phase_start_idx = len(self._tool_events)
        board_prompt = build_board_prompt(orders_summary)
        audit_context = dict(self._current_context or {})
        if cycle_id and not str(audit_context.get("cycle_id") or "").strip():
            audit_context["cycle_id"] = str(cycle_id or "").strip()
        self._record_prompt_snapshot(
            phase=phase,
            session_id=session_id,
            prompt=board_prompt,
            resumed=True,
            context=audit_context,
        )
        llm_call_id = self._new_llm_call_id(phase)
        self._llm_call_ids_by_phase[phase] = llm_call_id
        self._latest_llm_call_id = llm_call_id
        created_at = _utc_now()
        text = ""
        try:
            text = self._run_on_loop(self._run_async(self._runner, session_id, board_prompt))
            board_decision = parse_board_response(text)
        except Exception as exc:
            completed_at = _utc_now()
            tag_phase_tool_events(self._tool_events, phase=phase, start_idx=phase_start_idx)
            token_usage = getattr(self, "_last_token_usage", None)
            self._record_llm_interaction_audit(
                llm_call_id=llm_call_id,
                phase=phase,
                session_id=session_id,
                resumed=True,
                context=audit_context,
                prompt=board_prompt,
                created_at=created_at,
                completed_at=completed_at,
                status="error",
                response_text=text,
                token_usage=token_usage if isinstance(token_usage, dict) else None,
                error_message=str(exc),
                tool_events=self._tool_events[phase_start_idx:],
            )
            raise

        completed_at = _utc_now()
        tag_phase_tool_events(self._tool_events, phase=phase, start_idx=phase_start_idx)
        token_usage = getattr(self, "_last_token_usage", None)
        self._record_llm_interaction_audit(
            llm_call_id=llm_call_id,
            phase=phase,
            session_id=session_id,
            resumed=True,
            context=audit_context,
            prompt=board_prompt,
            created_at=created_at,
            completed_at=completed_at,
            status="ok",
            response_text=text,
            response_json=board_decision,
            token_usage=token_usage if isinstance(token_usage, dict) else None,
            tool_events=self._tool_events[phase_start_idx:],
        )
        try:
            events = [event for event in self._tool_events if str(event.get("tool") or "").strip()]
            payload = {
                "phase": "board",
                "cycle_id": str(cycle_id or "").strip(),
                "llm_call_id": llm_call_id,
                "analysis_funnel": self._funnel_metrics(),
                "tool_events": _safe_json(events),
                "tool_mix": _tool_category_counts(events, registry=self._registry),
                "token_usage": _safe_json(token_usage if isinstance(token_usage, dict) else {}),
                "prompt_bundle": self._prompt_bundle_payload(),
            }
            event_id = self._persist_tool_summary_memory(
                summary="Board prompt bundle snapshot before post generation.",
                payload=payload,
            )
            if event_id:
                self.record_artifact_links(
                    phase=phase,
                    cycle_id=str(cycle_id or "").strip(),
                    artifacts=[
                        {
                            "artifact_table": "agent_memory_events",
                            "artifact_id": event_id,
                            "artifact_role": "board_prompt_bundle_memory",
                        }
                    ],
                )
        except Exception as exc:
            logger.warning(
                "[yellow]Board prompt bundle memory write failed[/yellow] agent=%s err=%s",
                self.agent_id,
                str(exc),
            )
        return board_decision


class AdkTradingAgent:
    """Produces order intent and board post via ADK-native orchestration."""

    def __init__(
        self,
        *,
        agent_id: str,
        provider: str,
        settings: Settings,
        repo: BigQueryRepository,
        registry: ToolRegistry,
        tenant_id: str = "local",
        agent_config: AgentConfig | None = None,
    ):
        self.agent_id = agent_id
        self.provider = provider
        self.settings = settings
        self.repo = repo
        self.registry = registry
        self.tenant_id = str(tenant_id or "").strip().lower() or "local"
        self.agent_config = agent_config
        self.sleeve_capital_krw = settings.agent_capitals.get(agent_id, settings.sleeve_capital_krw)
        self.default_universe = settings.default_universe
        self._runner_settings = settings
        self._explore_session_id: str | None = None
        self._execution_session_id: str | None = None
        self._board_ticker_names: dict[str, str] = {}
        self.runner = self._build_runner(settings=settings)

    def _build_runner(self, *, settings: Settings) -> _ADKDecisionRunner:
        """Builds one ADK decision runner for current provider/settings."""
        return _ADKDecisionRunner(
            agent_id=self.agent_id,
            provider=self.provider,
            settings=settings,
            repo=self.repo,
            registry=self.registry,
            tenant_id=self.tenant_id,
            agent_config=self.agent_config,
        )

    def generate(self, context: dict[str, Any]) -> AgentOutput:
        """Generates one board post and optional trade intent."""
        self._board_ticker_names = _extract_known_ticker_names(context)
        rows = latest_rows(context.get("market_features", []))
        row_map = market_row_by_ticker(rows)
        if not rows:
            raise RuntimeError(
                f"market_features missing for agent={self.agent_id} cycle_id={str(context.get('cycle_id') or '').strip()}"
            )

        retry_limit, retry_delay = retry_policy_from_env()
        decision: dict[str, Any] | None = None
        session_id: str | None = None
        last_exc: Exception | None = None
        phase = cycle_phase(context)
        cycle_id = str(context.get("cycle_id", "")).strip()
        logger.info(
            "[blue]ADK decision start[/blue] agent=%s provider=%s",
            self.agent_id,
            self.provider,
            extra=event_extra(
                "adk_decision_start",
                agent_id=self.agent_id,
                provider=self.provider,
                phase=phase,
                cycle_id=cycle_id,
                resume_session_id=self._explore_session_id if phase == "execution" else None,
            ),
        )
        # Resume explore session for execution phase (session continuity).
        resume_sid = execution_resume_session_id(
            phase=phase,
            explore_session_id=self._explore_session_id,
        )
        for attempt in range(retry_limit + 1):
            try:
                decision, session_id = self.runner.decide_orders(
                    context=context,
                    default_universe=self.default_universe,
                    resume_session_id=resume_sid,
                )
                break
            except Exception as exc:
                last_exc = exc
                llm_call_lookup = getattr(self.runner, "llm_call_id_for_phase", None)
                llm_call_id = llm_call_lookup(phase) if callable(llm_call_lookup) else ""
                token_usage = getattr(self.runner, "_last_token_usage", None)
                tool_calls = token_usage.get("tool_calls") if isinstance(token_usage, dict) else None
                should_retry = _is_retryable_adk_error(exc) and attempt < retry_limit
                if should_retry:
                    sleep_s = retry_delay * float(attempt + 1)
                    logger.warning(
                        "[yellow]ADK decision retry[/yellow] agent=%s provider=%s attempt=%d sleep=%.1fs err=%s",
                        self.agent_id,
                        self.provider,
                        attempt + 1,
                        sleep_s,
                        str(exc),
                        extra=failure_extra(
                            "adk_decision_retry",
                            exc,
                            agent_id=self.agent_id,
                            provider=self.provider,
                            phase=phase,
                            cycle_id=cycle_id,
                            llm_call_id=llm_call_id,
                            tool_calls=tool_calls,
                            attempt=attempt + 1,
                            sleep_s=round(sleep_s, 1),
                        ),
                    )
                    time.sleep(sleep_s)
                    continue
                logger.exception(
                    "[red]ADK decision failed[/red] agent=%s provider=%s err=%s",
                    self.agent_id,
                    self.provider,
                    str(exc),
                    extra=failure_extra(
                        "adk_decision_failed",
                        exc,
                        agent_id=self.agent_id,
                        provider=self.provider,
                        phase=phase,
                        cycle_id=cycle_id,
                        llm_call_id=llm_call_id,
                        tool_calls=tool_calls,
                        attempt=attempt + 1,
                        resumed_session=bool(resume_sid),
                    ),
                )
                break
        if decision is None:
            reason = str(last_exc) if last_exc else "unknown"
            raise RuntimeError(f"ADK decision failed for agent={self.agent_id}: {reason}") from last_exc
        llm_call_lookup = getattr(self.runner, "llm_call_id_for_phase", None)
        llm_call_id = llm_call_lookup(phase) if callable(llm_call_lookup) else ""
        token_usage = getattr(self.runner, "_last_token_usage", None)
        logger.info(
            "[blue]ADK decision received[/blue] agent=%s provider=%s",
            self.agent_id,
            self.provider,
            extra=event_extra(
                "adk_decision_received",
                agent_id=self.agent_id,
                provider=self.provider,
                phase=phase,
                cycle_id=cycle_id,
                llm_call_id=llm_call_id,
                llm_calls=token_usage.get("llm_calls") if isinstance(token_usage, dict) else None,
                tool_calls=token_usage.get("tool_calls") if isinstance(token_usage, dict) else None,
            ),
        )

        explore_summary, orders = extract_decision_payload(decision)

        # Explore phase is board-sync only: never emit intents.
        # Preserve session_id so execution can continue the same conversation.
        if phase == "explore":
            self._explore_session_id = session_id
            self._execution_session_id = None
            return explore_phase_output(
                agent_id=self.agent_id,
                cycle_id=cycle_id,
                explore_summary=explore_summary,
                orders=orders,
                share_summary=bool(context.get("share_explore_summary")),
            )

        record_candidate_orders = getattr(self.runner, "record_candidate_orders", None)
        if callable(record_candidate_orders):
            record_candidate_orders(orders)
        order_feedback: list[dict[str, Any]] = []
        intents, tickers_mentioned = build_order_intents(
            repo=self.repo,
            settings=self.settings,
            agent_id=self.agent_id,
            sleeve_capital_krw=self.sleeve_capital_krw,
            cycle_id=cycle_id,
            context=context,
            orders=orders,
            row_map=row_map,
            feedback_events=order_feedback,
        )
        llm_call_lookup = getattr(self.runner, "llm_call_id_for_phase", None)
        execution_llm_call_id = llm_call_lookup("execution") if callable(llm_call_lookup) else ""
        if execution_llm_call_id:
            for intent in intents:
                intent.llm_call_id = execution_llm_call_id
        artifact_recorder = getattr(self.runner, "record_artifact_links", None)
        if callable(artifact_recorder) and intents:
            artifact_recorder(
                phase="execution",
                cycle_id=cycle_id,
                artifacts=[
                    {
                        "artifact_table": "agent_order_intents",
                        "artifact_id": intent.intent_id,
                        "artifact_role": "order_intent",
                        "detail_json": {"ticker": intent.ticker, "side": intent.side.value},
                    }
                    for intent in intents
                ],
            )
        record_candidate_order_feedback = getattr(self.runner, "record_candidate_order_feedback", None)
        if callable(record_candidate_order_feedback):
            record_candidate_order_feedback(order_feedback)

        # Save a placeholder board; final board generation happens after broker outcomes are known.
        orders_summary = format_orders_summary(intents, orders, ticker_names=self._board_ticker_names)
        self._execution_session_id = session_id
        self._explore_session_id = None

        return execution_phase_output(
            agent_id=self.agent_id,
            cycle_id=cycle_id,
            intents=intents,
            tickers_mentioned=tickers_mentioned,
            board_decision={},
            orders_summary=orders_summary,
        )

    def finalize_board_post(
        self,
        *,
        cycle_id: str,
        initial_post: BoardPost,
        intents: list[OrderIntent],
        reports: list[ExecutionReport],
    ) -> BoardPost:
        """Generates the final board post from actual execution outcomes."""
        execution_summary = format_execution_summary(intents, reports, ticker_names=self._board_ticker_names)
        board_decision: dict[str, Any] = {}
        session_id = self._execution_session_id
        try:
            if session_id:
                board_decision = self.runner.decide_board(
                    session_id,
                    execution_summary,
                    cycle_id=str(cycle_id or "").strip(),
                )
                logger.info("[blue]Board generation complete[/blue] agent=%s", self.agent_id)
        except Exception as exc:
            raise RuntimeError(f"board generation failed for agent={self.agent_id}: {exc}") from exc
        finally:
            self._execution_session_id = None

        record_candidate_executions = getattr(self.runner, "record_candidate_executions", None)
        if callable(record_candidate_executions):
            record_candidate_executions(intents, reports)
        persist_candidate_memories = getattr(self.runner, "_persist_candidate_memories", None)
        if callable(persist_candidate_memories):
            persist_candidate_memories(cycle_id=cycle_id)

        title = str(board_decision.get("board_title", "")).strip()[:120] or str(initial_post.title or "").strip()[:120] or "거래 아이디어"
        body = str(board_decision.get("board_body", "")).strip()[:1800] or execution_summary
        tickers: list[str] = []
        for token in [*list(getattr(initial_post, "tickers", [])), *[intent.ticker for intent in intents]]:
            clean = str(token or "").strip().upper()
            if clean and clean not in tickers:
                tickers.append(clean)
        self._board_ticker_names = {}
        llm_call_lookup = getattr(self.runner, "llm_call_id_for_phase", None)
        board_llm_call_id = llm_call_lookup("board") if callable(llm_call_lookup) else ""
        post = BoardPost(
            agent_id=self.agent_id,
            title=title,
            body=body,
            explore_summary=str(getattr(initial_post, "explore_summary", "") or "").strip()[:200],
            trading_mode=getattr(initial_post, "trading_mode", "paper"),
            tickers=tickers,
            cycle_id=str(cycle_id or "").strip(),
            llm_call_id=board_llm_call_id,
        )
        artifact_recorder = getattr(self.runner, "record_artifact_links", None)
        if callable(artifact_recorder):
            artifact_recorder(
                phase="board",
                cycle_id=str(cycle_id or "").strip(),
                artifacts=[
                    {
                        "artifact_table": "board_posts",
                        "artifact_id": post.post_id,
                        "artifact_role": "board_post",
                        "detail_json": {"title": post.title, "tickers": post.tickers},
                    }
                ],
            )
        return post

def build_adk_agents(
    settings: Settings,
    repo: BigQueryRepository,
    *,
    tenant_id: str = "local",
) -> list[TradingAgent]:
    """Builds ADK-backed agents from normalized per-agent configs."""
    normalize_agent_settings(settings)
    agents: list[TradingAgent] = []
    registry = build_default_registry(repo=repo, settings=settings, tenant_id=tenant_id)

    for agent_id in settings.agent_ids:
        ac = settings.agent_configs.get(agent_id)
        if not ac or not str(ac.provider).strip():
            logger.warning(
                "[yellow]Skipping agent without normalized config[/yellow] tenant=%s agent=%s",
                tenant_id,
                agent_id,
            )
            continue
        provider = ac.provider
        spec = get_provider_spec(provider)

        if not provider or spec is None or not spec.supports_adk or not _has_credentials(provider, settings):
            continue

        agents.append(
            AdkTradingAgent(
                agent_id=agent_id,
                provider=provider,
                settings=settings,
                repo=repo,
                registry=registry,
                tenant_id=tenant_id,
                agent_config=ac,
            )
        )

    return agents
