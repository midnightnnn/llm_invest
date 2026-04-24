from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import litellm

from arena.agents.adk_agent_flow import retry_policy_from_env
from arena.agents.support_model import (
    resolve_helper_api_key,
    resolve_helper_base_url,
    resolve_helper_model_token,
    select_helper_provider,
)
from arena.config import Settings
from arena.memory.relation_ontology import ONTOLOGY_VERSION, ontology_prompt_block
from arena.memory.relation_validation import RelationSource, RejectedRelation, validate_extracted_relations
from arena.models import utc_now
from arena.providers.registry import canonical_provider, get_provider_spec

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "semantic_relation_extractor_v1"
PROMPT_VERSION = "semantic_relation_prompt_v1"

DEFAULT_MEMORY_EVENT_TYPES: tuple[str, ...] = (
    "strategy_reflection",
    "manual_note",
    "thesis_open",
    "thesis_update",
    "thesis_invalidated",
    "thesis_realized",
)

_SYSTEM_INSTRUCTION = (
    "You extract source-grounded semantic relation triples for an investment memory graph. "
    "Use only facts explicitly present in the source. "
    "Return JSON only."
)


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    source: RelationSource
    run_row: dict[str, Any]
    accepted: list[dict[str, Any]]
    rejected: list[RejectedRelation]
    raw_output: dict[str, Any]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


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
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item.get("text")))
        return "".join(parts).strip()
    return ""


def _is_retryable_extractor_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".strip().lower()
    markers = [
        "resource_exhausted",
        "quota",
        "429",
        "503",
        "service unavailable",
        "unavailable",
        "deadline",
        "timed out",
        "timeout",
        "temporarily",
        "internal",
        "empty response",
    ]
    return any(marker in text for marker in markers)


def _request_temperature(model: str) -> float | None:
    token = str(model or "").strip().lower()
    if not token:
        return 0.0
    if "gpt-5" in token:
        return None
    if token.startswith("anthropic/") or token.startswith("claude-") or "/claude-" in token:
        return None
    return 0.0


def _model_token_for_override(settings: Settings, provider: str, model: str) -> str:
    model_id = _text(model)
    if not model_id:
        return resolve_helper_model_token(settings, provider, direct_only=True)
    if "/" in model_id:
        return model_id
    spec = get_provider_spec(provider)
    if spec is None:
        return model_id
    if spec.transport == "gemini_native":
        return f"gemini/{model_id}"
    if spec.transport == "anthropic_native":
        return f"anthropic/{model_id}"
    prefix = _text(spec.litellm_provider) or "openai"
    return f"{prefix}/{model_id}"


def source_from_pending_row(row: dict[str, Any], *, tenant_id: str | None = None) -> RelationSource:
    source_text = _text(row.get("source_text"))
    source_hash = _text(row.get("source_hash")) or _sha256_text(source_text)
    return RelationSource(
        tenant_id=_text(tenant_id or row.get("tenant_id")) or "local",
        source_table=_text(row.get("source_table")),
        source_id=_text(row.get("source_id")),
        source_node_id=_text(row.get("source_node_id")),
        source_created_at=row.get("source_created_at"),
        agent_id=_text(row.get("agent_id")) or None,
        trading_mode=_text(row.get("trading_mode")).lower() or "paper",
        cycle_id=_text(row.get("cycle_id")) or None,
        source_label=_text(row.get("source_label") or row.get("source_id")),
        source_text=source_text,
        source_hash=source_hash,
        detail_json=dict(row.get("detail_json") or {}),
    )


def build_extraction_prompt(source: RelationSource, *, max_triples: int = 6) -> str:
    source_payload = {
        "source_table": source.source_table,
        "source_id": source.source_id,
        "source_label": source.source_label,
        "agent_id": source.agent_id,
        "trading_mode": source.trading_mode,
        "cycle_id": source.cycle_id,
        "text": source.source_text,
    }
    schema = {
        "triples": [
            {
                "subject": {"label": "exact entity label", "type": "ticker|risk|catalyst|..."},
                "predicate": "supports|contradicts|risk_to|caused_by|leads_to|similar_setup|invalidates|outcome_of|mentions|contains",
                "object": {"label": "exact entity label", "type": "ticker|thesis|outcome|..."},
                "confidence": 0.0,
                "evidence_text": "copy one exact span from source text",
            }
        ]
    }
    return "\n\n".join(
        [
            "Extract up to "
            + str(max(1, int(max_triples)))
            + " high-signal semantic relation triples.",
            "Prefer causal, risk, support, contradiction, invalidation, and outcome relations. Use mentions only when no stronger predicate is justified.",
            "Every evidence_text must be copied verbatim from the source text. Do not use outside knowledge.",
            "Use uppercase labels for ticker entities.",
            ontology_prompt_block(),
            "Return JSON matching this schema:\n" + json.dumps(schema, ensure_ascii=False),
            "Source:\n" + json.dumps(source_payload, ensure_ascii=False, default=str),
        ]
    )


class SemanticRelationExtractor:
    """LLM-backed relation extraction for memory graph triples."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: Any,
        provider: str | None = None,
        model: str | None = None,
        min_confidence: float = 0.65,
        max_triples_per_source: int = 6,
    ) -> None:
        self.settings = settings
        self.repo = repo
        selected_provider = canonical_provider(provider) if provider else ""
        self.provider = selected_provider or select_helper_provider(settings, direct_only=True)
        self.model = _model_token_for_override(settings, self.provider, _text(model))
        self.api_key = resolve_helper_api_key(settings, self.provider)
        self.base_url = resolve_helper_base_url(settings, self.provider)
        if not self.api_key:
            raise ValueError(f"Missing direct API key for helper provider={self.provider}")
        self.min_confidence = max(0.0, min(float(min_confidence), 1.0))
        self.max_triples_per_source = max(1, min(int(max_triples_per_source), 12))

    async def _collect_response_text(self, *, prompt: str) -> str:
        extractor_timeout = int(self.settings.timeout_for("compaction"))
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "timeout": extractor_timeout,
            "messages": [
                {"role": "system", "content": _SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        }
        temperature = _request_temperature(self.model)
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if self.base_url:
            request_kwargs["base_url"] = self.base_url

        retry_limit, retry_delay = retry_policy_from_env()
        for attempt in range(retry_limit + 1):
            try:
                response = await asyncio.wait_for(
                    litellm.acompletion(**request_kwargs),
                    timeout=extractor_timeout,
                )
                text = _extract_response_text(response)
                if not text:
                    raise ValueError("empty response")
                return text
            except Exception as exc:
                if _is_retryable_extractor_error(exc) and attempt < retry_limit:
                    sleep_s = retry_delay * float(attempt + 1)
                    logger.warning(
                        "[yellow]relation extractor retry[/yellow] provider=%s model=%s attempt=%d sleep=%.1fs err=%s",
                        self.provider,
                        self.model,
                        attempt + 1,
                        sleep_s,
                        str(exc),
                    )
                    await asyncio.sleep(sleep_s)
                    continue
                raise
        return ""

    def _run_row(
        self,
        *,
        source: RelationSource,
        run_id: str,
        started_at: Any,
        finished_at: Any,
        status: str,
        accepted_count: int = 0,
        rejected: list[RejectedRelation] | None = None,
        raw_output: dict[str, Any] | None = None,
        error_message: str = "",
    ) -> dict[str, Any]:
        rejected_rows = [
            {"reason": item.reason, "raw": item.raw}
            for item in list(rejected or [])
        ]
        return {
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "source_table": source.source_table,
            "source_id": source.source_id,
            "source_hash": source.source_hash,
            "source_created_at": source.source_created_at,
            "agent_id": source.agent_id,
            "trading_mode": source.trading_mode,
            "cycle_id": source.cycle_id,
            "extractor_version": EXTRACTOR_VERSION,
            "prompt_version": PROMPT_VERSION,
            "ontology_version": ONTOLOGY_VERSION,
            "provider": self.provider,
            "model": self.model,
            "status": status,
            "accepted_count": int(accepted_count),
            "rejected_count": len(rejected_rows),
            "raw_output_json": raw_output or {},
            "error_message": error_message or None,
            "detail_json": {
                "source_label": source.source_label,
                "source_node_id": source.source_node_id,
                "source_text_chars": len(source.source_text),
                "rejected": rejected_rows[:20],
            },
        }

    async def extract_source(self, source: RelationSource) -> ExtractionResult:
        started_at = utc_now()
        run_id = f"rel_extract_{uuid4().hex[:12]}"
        raw_output: dict[str, Any] = {}
        rejected: list[RejectedRelation] = []
        accepted: list[dict[str, Any]] = []

        try:
            prompt = build_extraction_prompt(source, max_triples=self.max_triples_per_source)
            response_text = await self._collect_response_text(prompt=prompt)
            raw_output = _parse_json_object(response_text)
            triples_raw = raw_output.get("triples")
            if not isinstance(triples_raw, list):
                triples_raw = []
                rejected.append(RejectedRelation(reason="missing_triples_array", raw=raw_output))
                status = "invalid_output"
            else:
                validated = validate_extracted_relations(
                    triples_raw,
                    source=source,
                    extractor_version=EXTRACTOR_VERSION,
                    prompt_version=PROMPT_VERSION,
                    min_confidence=self.min_confidence,
                )
                accepted = validated.accepted
                rejected.extend(validated.rejected)
                status = "success"
            error_message = ""
        except Exception as exc:
            status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "[yellow]relation extraction failed[/yellow] source=%s:%s err=%s",
                source.source_table,
                source.source_id,
                error_message,
            )

        finished_at = utc_now()
        run_row = self._run_row(
            source=source,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            accepted_count=len(accepted),
            rejected=rejected,
            raw_output=raw_output,
            error_message=error_message,
        )
        return ExtractionResult(
            source=source,
            run_row=run_row,
            accepted=accepted,
            rejected=rejected,
            raw_output=raw_output,
        )

    async def run_pending(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 25,
        source_table: str | None = None,
        event_types: list[str] | None = None,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        clean_event_types = [
            _text(token)
            for token in (event_types if event_types is not None else DEFAULT_MEMORY_EVENT_TYPES)
            if _text(token)
        ]
        pending = self.repo.relation_extraction_pending_sources(
            tenant_id=tenant_id,
            limit=max(1, int(limit)),
            source_table=_text(source_table) or None,
            event_types=clean_event_types,
            trading_mode=self.settings.trading_mode,
            extractor_version=EXTRACTOR_VERSION,
            prompt_version=PROMPT_VERSION,
            ontology_version=ONTOLOGY_VERSION,
        )
        out: list[dict[str, Any]] = []
        for row in pending:
            source = source_from_pending_row(row, tenant_id=tenant_id)
            result = await self.extract_source(source)
            if not dry_run:
                if result.accepted:
                    self.repo.upsert_memory_relation_triples_with_graph(
                        result.accepted,
                        tenant_id=source.tenant_id,
                    )
                self.repo.append_memory_relation_extraction_runs([result.run_row], tenant_id=source.tenant_id)
            out.append(
                {
                    "tenant_id": source.tenant_id,
                    "source_table": source.source_table,
                    "source_id": source.source_id,
                    "source_hash": source.source_hash,
                    "status": result.run_row.get("status"),
                    "accepted_count": len(result.accepted),
                    "rejected_count": len(result.rejected),
                    "dry_run": bool(dry_run),
                }
            )
        return out
