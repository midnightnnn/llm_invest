from __future__ import annotations

import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from arena.memory.relation_ontology import (
    ONTOLOGY_VERSION,
    canonical_entity_type,
    canonical_predicate,
    is_allowed_entity_type,
    is_allowed_predicate,
    predicate_allows,
    predicate_min_confidence,
)
from arena.memory.relations import make_relation_triple, semantic_entity_node_id, ticker_node_id


@dataclass(frozen=True, slots=True)
class RelationSource:
    tenant_id: str
    source_table: str
    source_id: str
    source_node_id: str
    source_created_at: datetime | None
    agent_id: str | None
    trading_mode: str
    cycle_id: str | None
    source_label: str
    source_text: str
    source_hash: str = ""
    detail_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RejectedRelation:
    reason: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ValidatedRelations:
    accepted: list[dict[str, Any]]
    rejected: list[RejectedRelation]


_SPACE_RE = re.compile(r"\s+")
_MATCH_TOKEN_RE = re.compile(r"[^\w가-힣]+", re.UNICODE)
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,14}$")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _compact(value: Any, *, max_len: int) -> str:
    text = _SPACE_RE.sub(" ", _text(value)).strip()
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)].rstrip() + "..."


def _normalized_for_match(value: Any) -> str:
    return _SPACE_RE.sub(" ", _text(value)).strip().lower()


def _canonical_for_match(value: Any) -> str:
    text = _MATCH_TOKEN_RE.sub(" ", _normalized_for_match(value))
    return _SPACE_RE.sub(" ", text).strip()


def _fuzzy_span_supported(evidence: str, source: str) -> bool:
    evidence_tokens = _canonical_for_match(evidence).split()
    source_tokens = _canonical_for_match(source).split()
    if len(evidence_tokens) < 4 or len(source_tokens) < len(evidence_tokens):
        return False
    evidence_norm = " ".join(evidence_tokens)
    best = 0.0
    min_len = max(1, len(evidence_tokens) - 2)
    max_len = min(len(source_tokens), len(evidence_tokens) + 2)
    for window_len in range(min_len, max_len + 1):
        for idx in range(0, len(source_tokens) - window_len + 1):
            window = " ".join(source_tokens[idx : idx + window_len])
            ratio = SequenceMatcher(None, evidence_norm, window).ratio()
            if ratio > best:
                best = ratio
            if ratio >= 0.88:
                return True
    return False


def evidence_supported(evidence_text: str, source_text: str) -> bool:
    evidence = _normalized_for_match(evidence_text)
    source = _normalized_for_match(source_text)
    if len(evidence) < 12 or not source:
        return False
    if evidence in source:
        return True
    canonical_evidence = _canonical_for_match(evidence_text)
    canonical_source = _canonical_for_match(source_text)
    if canonical_evidence and canonical_evidence in canonical_source:
        return True
    return _fuzzy_span_supported(evidence_text, source_text)


def relation_confidence_threshold(predicate: str, global_min_confidence: float = 0.65) -> float:
    global_threshold = max(0.0, min(float(global_min_confidence), 1.0))
    return max(global_threshold, predicate_min_confidence(predicate, default=global_threshold))


def _entity(raw: Any, prefix: str) -> tuple[str, str]:
    if isinstance(raw, dict):
        label = _text(raw.get("label") or raw.get("name") or raw.get("text"))
        entity_type = _text(raw.get("type") or raw.get("entity_type"))
    else:
        label = ""
        entity_type = ""
    label = label or _text(raw.get(f"{prefix}_label")) if isinstance(raw, dict) else label
    entity_type = entity_type or _text(raw.get(f"{prefix}_type")) if isinstance(raw, dict) else entity_type
    return _compact(label, max_len=180), canonical_entity_type(entity_type)


def _entity_node_id(entity_type: str, label: str) -> str:
    if entity_type == "ticker":
        token = _text(label).upper()
        if not _TICKER_RE.match(token):
            return ""
        return ticker_node_id(token)
    return semantic_entity_node_id(entity_type, label)


def _reject(raw: dict[str, Any], reason: str) -> RejectedRelation:
    return RejectedRelation(reason=reason, raw=dict(raw))


def validate_extracted_relations(
    raw_triples: list[Any],
    *,
    source: RelationSource,
    extractor_version: str,
    prompt_version: str,
    min_confidence: float = 0.65,
) -> ValidatedRelations:
    accepted: list[dict[str, Any]] = []
    rejected: list[RejectedRelation] = []
    seen: set[str] = set()
    threshold = max(0.0, min(float(min_confidence), 1.0))

    for raw_item in raw_triples:
        if not isinstance(raw_item, dict):
            rejected.append(_reject({"value": raw_item}, "not_object"))
            continue
        raw = dict(raw_item)
        subject_label, subject_type = _entity(raw.get("subject") or raw, "subject")
        object_label, object_type = _entity(raw.get("object") or raw, "object")
        predicate = canonical_predicate(raw.get("predicate"))
        evidence_text = _compact(raw.get("evidence_text") or raw.get("evidence") or raw.get("quote"), max_len=360)
        try:
            confidence = float(raw.get("confidence"))
        except (TypeError, ValueError):
            confidence = 0.0

        if not subject_label or not object_label:
            rejected.append(_reject(raw, "missing_entity_label"))
            continue
        if not subject_type or not object_type:
            rejected.append(_reject(raw, "missing_entity_type"))
            continue
        if not is_allowed_entity_type(subject_type) or not is_allowed_entity_type(object_type):
            rejected.append(_reject(raw, "unsupported_entity_type"))
            continue
        if not is_allowed_predicate(predicate):
            rejected.append(_reject(raw, "unsupported_predicate"))
            continue
        if not predicate_allows(predicate, subject_type, object_type):
            rejected.append(_reject(raw, "predicate_type_mismatch"))
            continue
        relation_threshold = relation_confidence_threshold(predicate, threshold)
        if confidence < relation_threshold:
            rejected.append(_reject(raw, "low_confidence"))
            continue
        if not evidence_supported(evidence_text, source.source_text):
            rejected.append(_reject(raw, "evidence_not_found"))
            continue

        subject_node_id = _entity_node_id(subject_type, subject_label)
        object_node_id = _entity_node_id(object_type, object_label)
        if not subject_node_id or not object_node_id:
            rejected.append(_reject(raw, "invalid_entity_node"))
            continue
        if subject_node_id == object_node_id:
            rejected.append(_reject(raw, "self_relation"))
            continue

        triple = make_relation_triple(
            source_table=source.source_table,
            source_id=source.source_id,
            source_node_id=source.source_node_id,
            source_label=source.source_label,
            source_created_at=source.source_created_at,
            agent_id=source.agent_id,
            trading_mode=source.trading_mode,
            cycle_id=source.cycle_id,
            subject_node_id=subject_node_id,
            subject_label=subject_label.upper() if subject_type == "ticker" else subject_label,
            subject_type=subject_type,
            predicate=predicate,
            object_node_id=object_node_id,
            object_label=object_label.upper() if object_type == "ticker" else object_label,
            object_type=object_type,
            evidence_text=evidence_text,
            confidence=confidence,
            extraction_method="llm",
            extraction_version=extractor_version,
            status="accepted",
            detail_json={
                "ontology_version": ONTOLOGY_VERSION,
                "prompt_version": prompt_version,
                "confidence_threshold": relation_threshold,
                "source_hash": source.source_hash,
                "raw": raw,
            },
        )
        if not triple:
            rejected.append(_reject(raw, "triple_build_failed"))
            continue
        triple_id = _text(triple.get("triple_id"))
        if not triple_id or triple_id in seen:
            continue
        seen.add(triple_id)
        accepted.append(triple)

    return ValidatedRelations(accepted=accepted, rejected=rejected)
