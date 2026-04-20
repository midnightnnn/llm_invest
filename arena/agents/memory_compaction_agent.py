from __future__ import annotations

import asyncio
import json
import logging
from string import Formatter
from typing import Any

import litellm

from arena.agents.adk_agent_flow import retry_policy_from_env
from arena.agents.support_model import (
    resolve_helper_api_key,
    resolve_helper_base_url,
    resolve_helper_model_token,
    select_helper_provider,
)
from arena.config import Settings
from arena.data.bq import BigQueryRepository
from arena.memory.policy import get_memory_policy_value, memory_event_enabled, resolve_compaction_prompt
from arena.memory.store import MemoryStore
from arena.memory.thesis import THESIS_EVENT_TYPES
from arena.models import utc_now

logger = logging.getLogger(__name__)

_GLOBAL_PROMPT_TENANT = "global"


_COMPACTION_INSTRUCTION = (
    "당신은 투자 에이전트의 장기기억 정리 담당입니다. "
    "입력된 사이클 산출물에서 다음 사이클에도 재사용 가치가 있는 교훈만 추려야 합니다. "
    "주문 로그를 반복하지 말고, 반복 가능한 lesson만 남기세요. "
    "닫힌 thesis chain이 주어지면 thesis 단위 post-mortem을 우선하고, 같은 thesis에 대한 reflection은 최대 1개만 만드세요. "
    "사실을 꾸며내지 말고 입력에 있는 정보만 사용하세요. "
    "반드시 JSON만 반환하세요."
)


def _is_retryable_compaction_error(exc: Exception) -> bool:
    """Returns True when helper-model failures look transient."""
    text = f"{type(exc).__name__}: {exc}".strip().lower()
    if not text:
        return False
    markers = [
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "429",
        "503",
        "service unavailable",
        "serviceunavailable",
        "unavailable",
        "high demand",
        "deadline",
        "timed out",
        "timeout",
        "temporarily",
        "temporary",
        "internal",
        "empty response",
    ]
    return any(marker in text for marker in markers)


def _model_accepts_temperature(model: str) -> bool:
    token = str(model or "").strip().lower()
    if not token:
        return True
    if "gpt-5" in token:
        return False
    if token.startswith("anthropic/") or token.startswith("claude-") or "/claude-" in token:
        return False
    return True


def _trim_text(value: Any, *, max_len: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}


def _extract_response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    message = choices[0].get("message") if isinstance(choices[0], dict) else getattr(choices[0], "message", None)
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    return ""


class MemoryCompactionAgent:
    """Summarizes recent cycle artifacts into a few retrieval-friendly lesson memories."""

    def __init__(self, settings: Settings, repo: BigQueryRepository, memory_store: MemoryStore):
        self.settings = settings
        self.repo = repo
        self.memory_store = memory_store
        self.agent_id = "memory_compactor"
        resolver = getattr(repo, "resolve_tenant_id", None)
        if callable(resolver):
            self.tenant_id = str(resolver()).strip().lower() or "local"
        else:
            self.tenant_id = str(getattr(repo, "tenant_id", "") or "").strip().lower() or "local"

        self.provider = select_helper_provider(self.settings, direct_only=True)
        self.model = resolve_helper_model_token(self.settings, self.provider, direct_only=True)
        self.api_key = resolve_helper_api_key(self.settings, self.provider)
        self.base_url = resolve_helper_base_url(self.settings, self.provider)
        if not self.api_key:
            raise ValueError(f"Missing direct API key for helper provider={self.provider}")

    def _load_prompt_template(self) -> str:
        """Loads the tenant compactor prompt template, falling back to global."""
        return resolve_compaction_prompt(self.repo, self.tenant_id, policy=self.settings.memory_policy)

    def _request_temperature(self) -> float | None:
        if not _model_accepts_temperature(self.model):
            return None
        return 0.1

    def _policy_value(self, path: str, default: Any) -> Any:
        policy = getattr(self.settings, "memory_policy", None)
        if not isinstance(policy, dict):
            return default
        return get_memory_policy_value(policy, path, default)

    def _thesis_chain_enabled(self) -> bool:
        return bool(self._policy_value("compaction.thesis_chain_enabled", True))

    def _thesis_chain_max_chains_per_cycle(self) -> int:
        try:
            value = int(self._policy_value("compaction.thesis_chain_max_chains_per_cycle", 2))
        except (TypeError, ValueError):
            value = 2
        return max(1, min(value, 8))

    def _thesis_chain_max_events_per_chain(self) -> int:
        try:
            value = int(self._policy_value("compaction.thesis_chain_max_events_per_chain", 6))
        except (TypeError, ValueError):
            value = 6
        return max(2, min(value, 12))

    @staticmethod
    def _thesis_chain_reflection_key(thesis_id: str) -> str:
        return f"reflection:{str(thesis_id or '').strip()}"

    @staticmethod
    def _render_prompt_template(template: str, values: dict[str, Any]) -> str:
        """Formats known placeholders while tolerating partially customized templates."""
        safe_values = {key: str(value) for key, value in values.items()}
        fields = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name
        }
        if fields:
            try:
                rendered = template.format_map({key: safe_values.get(key, "") for key in fields})
            except Exception:
                rendered = template
        else:
            rendered = template
        if "{payload_json}" not in template and safe_values.get("payload_json"):
            rendered = rendered.rstrip() + "\n\n" + safe_values["payload_json"]
        return rendered

    @staticmethod
    def _parse_payload(row: dict[str, Any]) -> dict[str, Any]:
        payload_raw = row.get("payload_json")
        if isinstance(payload_raw, dict):
            return payload_raw
        if isinstance(payload_raw, str) and payload_raw.strip():
            try:
                parsed = json.loads(payload_raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    def _compact_cycle_memories(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for row in rows:
            payload = self._parse_payload(row)
            event_type = str(row.get("event_type") or "").strip()
            compact: dict[str, Any] = {
                "event_id": str(row.get("event_id") or "").strip(),
                "event_type": event_type,
                "summary": _trim_text(row.get("summary"), max_len=220),
            }
            if row.get("importance_score") is not None:
                compact["importance_score"] = float(row.get("importance_score") or 0.0)
            if row.get("outcome_score") is not None:
                compact["outcome_score"] = float(row.get("outcome_score") or 0.0)

            if event_type == "trade_execution":
                intent = payload.get("intent") if isinstance(payload.get("intent"), dict) else {}
                report = payload.get("report") if isinstance(payload.get("report"), dict) else {}
                decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
                compact.update(
                    {
                        "ticker": str(intent.get("ticker") or "").strip().upper(),
                        "side": str(intent.get("side") or "").strip().upper(),
                        "rationale": _trim_text(intent.get("rationale"), max_len=120),
                        "status": str(report.get("status") or "").strip().upper(),
                        "policy_hits": list(decision.get("policy_hits") or [])[:4],
                    }
                )
            elif event_type == "react_tools_summary":
                tool_mix = payload.get("tool_mix") if isinstance(payload.get("tool_mix"), dict) else {}
                tool_events = payload.get("tool_events") if isinstance(payload.get("tool_events"), list) else []
                compact.update(
                    {
                        "phase": str(payload.get("phase") or "").strip().lower(),
                        "tool_mix": tool_mix,
                        "tool_count": len(tool_events),
                        "tools": [
                            str(evt.get("tool") or "").strip()
                            for evt in tool_events[:6]
                            if isinstance(evt, dict) and str(evt.get("tool") or "").strip()
                        ],
                    }
                )
            elif event_type.startswith("thesis_"):
                compact.update(
                    {
                        "ticker": str(payload.get("ticker") or "").strip().upper(),
                        "state": str(payload.get("state") or "").strip().lower(),
                        "thesis_summary": _trim_text(payload.get("thesis_summary"), max_len=140),
                        "position_action": str(payload.get("position_action") or "").strip().lower(),
                        "strategy_refs": list(payload.get("strategy_refs") or [])[:4],
                    }
                )
            compacted.append(compact)
        return compacted

    def _compact_board_posts(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for row in rows:
            compacted.append(
                {
                    "post_id": str(row.get("post_id") or "").strip(),
                    "title": _trim_text(row.get("title"), max_len=120),
                    "draft_summary": _trim_text(row.get("draft_summary"), max_len=180),
                    "body": _trim_text(row.get("body"), max_len=240),
                    "tickers": list(row.get("tickers") or [])[:6],
                }
            )
        return compacted

    def _compact_environment_research(self) -> list[dict[str, Any]]:
        loader = getattr(self.repo, "get_research_briefings", None)
        if not callable(loader):
            return []
        try:
            rows = list(
                loader(
                    categories=["global_market", "geopolitical"],
                    limit=3,
                    trading_mode=self.settings.trading_mode,
                )
            )
        except Exception as exc:
            logger.warning("[yellow]compaction research load failed[/yellow] err=%s", str(exc))
            return []

        compacted: list[dict[str, Any]] = []
        for row in rows:
            compacted.append(
                {
                    "briefing_id": str(row.get("briefing_id") or "").strip(),
                    "category": str(row.get("category") or "").strip().lower(),
                    "headline": _trim_text(row.get("headline"), max_len=120),
                    "summary": _trim_text(row.get("summary"), max_len=220),
                }
            )
        return compacted

    def _recent_lessons(self, agent_id: str) -> list[dict[str, Any]]:
        rows = self.memory_store.recent(
            agent_id=agent_id,
            limit=max(int(self.settings.memory_compaction_recent_lessons_limit) * 3, 8),
        )
        compacted: list[dict[str, Any]] = []
        for row in rows:
            event_type = str(row.get("event_type") or "").strip()
            if not memory_event_enabled(self.settings.memory_policy, event_type, True):
                continue
            if event_type not in {"strategy_reflection", "manual_note"}:
                continue
            compacted.append(
                {
                    "event_id": str(row.get("event_id") or "").strip(),
                    "event_type": event_type,
                    "summary": _trim_text(row.get("summary"), max_len=180),
                }
            )
            if len(compacted) >= max(1, int(self.settings.memory_compaction_recent_lessons_limit)):
                break
        return compacted

    def _load_closed_thesis_chains(self, agent_id: str, cycle_id: str) -> list[dict[str, Any]]:
        if not self._thesis_chain_enabled():
            return []

        key_loader = getattr(self.repo, "closed_thesis_keys_for_cycle", None)
        row_loader = getattr(self.repo, "memory_events_by_semantic_keys", None)
        exists_loader = getattr(self.repo, "memory_event_exists_by_semantic_key", None)
        if not callable(key_loader) or not callable(row_loader):
            return []

        max_chains = self._thesis_chain_max_chains_per_cycle()
        max_events = self._thesis_chain_max_events_per_chain()
        try:
            semantic_keys = list(
                key_loader(
                    agent_id=agent_id,
                    cycle_id=cycle_id,
                    limit=max_chains,
                    trading_mode=self.settings.trading_mode,
                )
            )
        except Exception as exc:
            logger.warning(
                "[yellow]closed thesis key load failed[/yellow] agent=%s cycle_id=%s err=%s",
                agent_id,
                cycle_id,
                str(exc),
            )
            return []

        filtered_keys: list[str] = []
        for token in semantic_keys:
            semantic_key = str(token or "").strip()
            if not semantic_key:
                continue
            duplicate_exists = False
            if callable(exists_loader):
                try:
                    duplicate_exists = bool(
                        exists_loader(
                            agent_id=agent_id,
                            event_type="strategy_reflection",
                            semantic_key=self._thesis_chain_reflection_key(semantic_key),
                            trading_mode=self.settings.trading_mode,
                        )
                    )
                except TypeError:
                    duplicate_exists = bool(
                        exists_loader(
                            agent_id=agent_id,
                            event_type="strategy_reflection",
                            semantic_key=self._thesis_chain_reflection_key(semantic_key),
                        )
                    )
                except Exception:
                    duplicate_exists = False
            if not duplicate_exists:
                filtered_keys.append(semantic_key)
            if len(filtered_keys) >= max_chains:
                break
        if not filtered_keys:
            return []

        try:
            rows = list(
                row_loader(
                    agent_id=agent_id,
                    semantic_keys=filtered_keys,
                    event_types=sorted(THESIS_EVENT_TYPES),
                    trading_mode=self.settings.trading_mode,
                )
            )
        except Exception as exc:
            logger.warning(
                "[yellow]thesis chain row load failed[/yellow] agent=%s cycle_id=%s err=%s",
                agent_id,
                cycle_id,
                str(exc),
            )
            return []

        grouped_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in filtered_keys}
        for row in rows:
            payload = self._parse_payload(row)
            semantic_key = str(row.get("semantic_key") or payload.get("thesis_id") or "").strip()
            if semantic_key in grouped_rows:
                grouped_rows[semantic_key].append(row)

        compacted: list[dict[str, Any]] = []
        for semantic_key in filtered_keys:
            chain_rows = grouped_rows.get(semantic_key) or []
            if not chain_rows:
                continue
            compact_events = self._compact_cycle_memories(chain_rows)
            if not compact_events:
                continue
            compact_events = compact_events[-max_events:]
            terminal_row = chain_rows[-1]
            terminal_payload = self._parse_payload(terminal_row)
            thesis_id = str(terminal_row.get("semantic_key") or terminal_payload.get("thesis_id") or semantic_key).strip()
            ticker = str(terminal_payload.get("ticker") or "").strip().upper()
            thesis_summary = _trim_text(terminal_payload.get("thesis_summary"), max_len=140)
            event_ids = [
                str(item.get("event_id") or "").strip()
                for item in compact_events
                if str(item.get("event_id") or "").strip()
            ][:max_events]
            compacted.append(
                {
                    "thesis_id": thesis_id,
                    "semantic_key": semantic_key,
                    "ticker": ticker,
                    "thesis_summary": thesis_summary,
                    "terminal_event_type": str(terminal_row.get("event_type") or "").strip().lower(),
                    "event_ids": event_ids,
                    "events": compact_events,
                }
            )
        return compacted

    def _load_agent_inputs(self, agent_id: str, cycle_id: str) -> dict[str, Any]:
        event_types = [
            event_type
            for event_type in (
                "trade_execution",
                "react_tools_summary",
                "thesis_open",
                "thesis_update",
                "thesis_invalidated",
                "thesis_realized",
            )
            if memory_event_enabled(self.settings.memory_policy, event_type, True)
        ]
        cycle_rows = self.repo.memory_events_for_cycle(
            agent_id=agent_id,
            cycle_id=cycle_id,
            event_types=event_types,
            limit=max(4, int(self.settings.memory_compaction_cycle_event_limit)),
            trading_mode=self.settings.trading_mode,
        )
        board_rows = self.repo.board_posts_for_cycle(
            cycle_id=cycle_id,
            agent_id=agent_id,
            limit=2,
            trading_mode=self.settings.trading_mode,
        )
        return {
            "closed_thesis_chains": self._load_closed_thesis_chains(agent_id, cycle_id),
            "cycle_memories": self._compact_cycle_memories(cycle_rows),
            "board_posts": self._compact_board_posts(board_rows),
            "environment_research": self._compact_environment_research(),
            "prior_lessons": self._recent_lessons(agent_id),
        }

    def _build_prompt(self, *, agent_id: str, cycle_id: str, inputs: dict[str, Any]) -> str:
        max_reflections = max(1, int(self.settings.memory_compaction_max_reflections))
        payload = {
            "cycle_id": cycle_id,
            "agent_id": agent_id,
            "closed_thesis_chains": inputs.get("closed_thesis_chains") or [],
            "cycle_memories": inputs.get("cycle_memories") or [],
            "board_posts": inputs.get("board_posts") or [],
            "environment_research": inputs.get("environment_research") or [],
            "prior_lessons": inputs.get("prior_lessons") or [],
        }
        template = self._load_prompt_template()
        return self._render_prompt_template(
            template,
            {
                "agent_id": agent_id,
                "cycle_id": cycle_id,
                "max_reflections": max_reflections,
                "payload_json": json.dumps(payload, ensure_ascii=False),
            },
        )

    async def _collect_response_text(self, *, prompt: str) -> str:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "timeout": self.settings.llm_timeout_seconds,
            "messages": [
                {"role": "system", "content": _COMPACTION_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        }
        temperature = self._request_temperature()
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if self.base_url:
            request_kwargs["base_url"] = self.base_url
        retry_limit, retry_delay = retry_policy_from_env()
        for attempt in range(retry_limit + 1):
            try:
                response = await asyncio.wait_for(
                    litellm.acompletion(**request_kwargs),
                    timeout=self.settings.llm_timeout_seconds,
                )
                text = _extract_response_text(response)
                if not text.strip():
                    raise ValueError("empty response")
                return text
            except Exception as exc:
                should_retry = _is_retryable_compaction_error(exc) and attempt < retry_limit
                if should_retry:
                    sleep_s = retry_delay * float(attempt + 1)
                    logger.warning(
                        "[yellow]Memory compaction retry[/yellow] provider=%s model=%s attempt=%d sleep=%.1fs err=%s",
                        self.provider,
                        self.model,
                        attempt + 1,
                        sleep_s,
                        str(exc),
                    )
                    await asyncio.sleep(sleep_s)
                    continue
                raise

    def _sanitize_reflections(
        self,
        raw: dict[str, Any],
        *,
        known_event_ids: set[str],
        known_post_ids: set[str],
        known_briefing_ids: set[str],
        known_thesis_chains: dict[str, dict[str, Any]],
        existing_summaries: set[str],
    ) -> list[dict[str, Any]]:
        reflections_raw = raw.get("reflections")
        if not isinstance(reflections_raw, list):
            return []

        sanitized: list[dict[str, Any]] = []
        max_reflections = max(1, int(self.settings.memory_compaction_max_reflections))
        event_to_thesis: dict[str, str] = {}
        for thesis_id, chain in known_thesis_chains.items():
            for event_id in list(chain.get("event_ids") or [])[:12]:
                token = str(event_id or "").strip()
                if token:
                    event_to_thesis[token] = thesis_id
        used_thesis_ids: set[str] = set()
        for item in reflections_raw:
            if not isinstance(item, dict):
                continue
            summary = _trim_text(item.get("summary"), max_len=160)
            if len(summary) < 18:
                continue
            key = summary.lower()
            if key in existing_summaries:
                continue
            existing_summaries.add(key)
            try:
                score = float(item.get("importance_score") or 0.7)
            except (TypeError, ValueError):
                score = 0.7
            tags_raw = item.get("tags")
            tags = []
            if isinstance(tags_raw, list):
                tags = [str(tag).strip().lower() for tag in tags_raw if str(tag).strip()][:4]
            source_event_ids = []
            raw_event_ids = item.get("source_event_ids")
            if isinstance(raw_event_ids, list):
                source_event_ids = [
                    str(token).strip()
                    for token in raw_event_ids
                    if str(token).strip() in known_event_ids
                ][:6]
            source_post_ids = []
            raw_post_ids = item.get("source_post_ids")
            if isinstance(raw_post_ids, list):
                source_post_ids = [
                    str(token).strip()
                    for token in raw_post_ids
                    if str(token).strip() in known_post_ids
                ][:4]
            source_briefing_ids = []
            raw_briefing_ids = item.get("source_briefing_ids")
            if isinstance(raw_briefing_ids, list):
                source_briefing_ids = [
                    str(token).strip()
                    for token in raw_briefing_ids
                    if str(token).strip() in known_briefing_ids
                ][:4]
            thesis_id = str(item.get("thesis_id") or "").strip()
            if thesis_id and thesis_id not in known_thesis_chains:
                thesis_id = ""
            if not thesis_id and source_event_ids:
                matched_thesis_ids = {event_to_thesis[event_id] for event_id in source_event_ids if event_id in event_to_thesis}
                if len(matched_thesis_ids) == 1:
                    thesis_id = next(iter(matched_thesis_ids))
            if thesis_id:
                if thesis_id in used_thesis_ids:
                    continue
                used_thesis_ids.add(thesis_id)
                chain = known_thesis_chains.get(thesis_id) or {}
                if not source_event_ids:
                    source_event_ids = list(chain.get("event_ids") or [])[:6]
                terminal_event_type = str(chain.get("terminal_event_type") or "").strip().lower()
            else:
                terminal_event_type = ""
            sanitized_item = {
                "summary": summary,
                "importance_score": max(0.0, min(score, 1.0)),
                "tags": tags,
                "source_event_ids": source_event_ids,
                "source_post_ids": source_post_ids,
                "source_briefing_ids": source_briefing_ids,
            }
            if thesis_id:
                sanitized_item["thesis_id"] = thesis_id
                sanitized_item["terminal_event_type"] = terminal_event_type
            sanitized.append(sanitized_item)
            if len(sanitized) >= max_reflections:
                break
        return sanitized

    async def _compact_one(self, *, agent_id: str, cycle_id: str, inputs: dict[str, Any]) -> list[dict[str, Any]]:
        if not inputs.get("cycle_memories") and not inputs.get("board_posts") and not inputs.get("closed_thesis_chains"):
            return []

        prompt = self._build_prompt(agent_id=agent_id, cycle_id=cycle_id, inputs=inputs)
        raw_text = await self._collect_response_text(prompt=prompt)
        parsed = _parse_json_object(raw_text)
        known_event_ids = {
            str(row.get("event_id") or "").strip()
            for row in (inputs.get("cycle_memories") or [])
            if str(row.get("event_id") or "").strip()
        }
        for chain in inputs.get("closed_thesis_chains") or []:
            for event_id in list(chain.get("event_ids") or [])[:12]:
                token = str(event_id or "").strip()
                if token:
                    known_event_ids.add(token)
        known_post_ids = {
            str(row.get("post_id") or "").strip()
            for row in (inputs.get("board_posts") or [])
            if str(row.get("post_id") or "").strip()
        }
        known_briefing_ids = {
            str(row.get("briefing_id") or "").strip()
            for row in (inputs.get("environment_research") or [])
            if str(row.get("briefing_id") or "").strip()
        }
        existing_summaries = {
            str(row.get("summary") or "").strip().lower()
            for row in (inputs.get("prior_lessons") or [])
            if str(row.get("summary") or "").strip()
        }
        known_thesis_chains = {
            str(row.get("thesis_id") or "").strip(): row
            for row in (inputs.get("closed_thesis_chains") or [])
            if str(row.get("thesis_id") or "").strip()
        }
        return self._sanitize_reflections(
            parsed,
            known_event_ids=known_event_ids,
            known_post_ids=known_post_ids,
            known_briefing_ids=known_briefing_ids,
            known_thesis_chains=known_thesis_chains,
            existing_summaries=existing_summaries,
        )

    async def run(self, *, cycle_id: str, agent_ids: list[str]) -> list[dict[str, Any]]:
        """Compacts one completed cycle into a few lesson memories per agent."""
        if not self.settings.memory_compaction_enabled:
            return []

        saved: list[dict[str, Any]] = []
        deduped_agent_ids: list[str] = []
        for token in agent_ids:
            clean = str(token or "").strip()
            if clean and clean not in deduped_agent_ids:
                deduped_agent_ids.append(clean)

        for agent_id in deduped_agent_ids:
            inputs = self._load_agent_inputs(agent_id, cycle_id)
            if not inputs.get("cycle_memories") and not inputs.get("board_posts") and not inputs.get("closed_thesis_chains"):
                continue
            logger.info(
                "[cyan]Memory compaction input[/cyan] agent=%s cycle_id=%s cycle_memories=%d board_posts=%d closed_thesis_chains=%d prior_lessons=%d",
                agent_id,
                cycle_id,
                len(inputs.get("cycle_memories") or []),
                len(inputs.get("board_posts") or []),
                len(inputs.get("closed_thesis_chains") or []),
                len(inputs.get("prior_lessons") or []),
            )
            try:
                reflections = await self._compact_one(agent_id=agent_id, cycle_id=cycle_id, inputs=inputs)
            except Exception as exc:
                logger.warning(
                    "[yellow]Memory compaction failed[/yellow] agent=%s cycle_id=%s err=%s",
                    agent_id,
                    cycle_id,
                    str(exc),
                )
                continue
            for reflection in reflections:
                thesis_id = str(reflection.get("thesis_id") or "").strip()
                terminal_event_type = str(reflection.get("terminal_event_type") or "").strip().lower()
                semantic_key = ""
                if thesis_id:
                    semantic_key = self._thesis_chain_reflection_key(thesis_id)
                    payload = {
                        "source": "thesis_chain_compaction",
                        "cycle_id": cycle_id,
                        "thesis_id": thesis_id,
                        "terminal_event_type": terminal_event_type,
                        "tags": list(reflection.get("tags") or [])[:4],
                        "source_event_ids": list(reflection.get("source_event_ids") or [])[:6],
                        "source_post_ids": list(reflection.get("source_post_ids") or [])[:4],
                        "source_briefing_ids": list(reflection.get("source_briefing_ids") or [])[:4],
                    }
                else:
                    payload = {
                        "source": "memory_compaction",
                        "cycle_id": cycle_id,
                        "tags": list(reflection.get("tags") or [])[:4],
                        "source_event_ids": list(reflection.get("source_event_ids") or [])[:6],
                        "source_post_ids": list(reflection.get("source_post_ids") or [])[:4],
                        "source_briefing_ids": list(reflection.get("source_briefing_ids") or [])[:4],
                    }
                self.memory_store.record_reflection(
                    agent_id=agent_id,
                    summary=str(reflection.get("summary") or ""),
                    score=float(reflection.get("importance_score") or 0.7),
                    payload=payload,
                    semantic_key=semantic_key or None,
                )
                saved.append(
                    {
                        "agent_id": agent_id,
                        "cycle_id": cycle_id,
                        "summary": str(reflection.get("summary") or ""),
                        "importance_score": float(reflection.get("importance_score") or 0.7),
                        "thesis_id": thesis_id or None,
                    }
                )
            logger.info(
                "[cyan]Memory compaction output[/cyan] agent=%s cycle_id=%s reflections=%d thesis_reflections=%d",
                agent_id,
                cycle_id,
                len(reflections),
                sum(1 for row in reflections if str(row.get("thesis_id") or "").strip()),
            )
        return saved
