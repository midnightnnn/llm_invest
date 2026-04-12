from __future__ import annotations

from dataclasses import dataclass


ONTOLOGY_VERSION = "semantic_relation_ontology_v1"


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    name: str
    description: str
    subject_types: tuple[str, ...] = ("*",)
    object_types: tuple[str, ...] = ("*",)
    min_confidence: float = 0.65


ENTITY_TYPES: tuple[str, ...] = (
    "ticker",
    "company",
    "sector",
    "industry",
    "market",
    "macro_factor",
    "risk",
    "catalyst",
    "event",
    "indicator",
    "metric",
    "strategy_tag",
    "regime",
    "scenario",
    "thesis",
    "outcome",
    "price_action",
    "research_category",
    "passage",
)

_ENTITY_TYPE_ALIASES: dict[str, str] = {
    "stock": "ticker",
    "symbol": "ticker",
    "equity": "ticker",
    "corp": "company",
    "corporation": "company",
    "macro": "macro_factor",
    "macrofactor": "macro_factor",
    "risk_factor": "risk",
    "driver": "catalyst",
    "trigger": "catalyst",
    "signal": "indicator",
    "setup": "scenario",
    "memory": "passage",
    "source": "passage",
    "briefing": "passage",
}

PREDICATES: tuple[PredicateSpec, ...] = (
    PredicateSpec(
        "supports",
        "The subject is evidence or a condition that strengthens the object.",
        ("catalyst", "event", "indicator", "metric", "macro_factor", "regime", "strategy_tag", "thesis", "scenario"),
        ("ticker", "company", "sector", "industry", "strategy_tag", "thesis", "scenario", "outcome", "price_action"),
        0.70,
    ),
    PredicateSpec(
        "contradicts",
        "The subject weakens or conflicts with the object.",
        ("risk", "event", "indicator", "metric", "macro_factor", "regime", "thesis", "scenario"),
        ("ticker", "company", "sector", "industry", "strategy_tag", "thesis", "scenario", "outcome", "price_action"),
        0.80,
    ),
    PredicateSpec(
        "risk_to",
        "The subject is a risk for the object.",
        ("risk", "event", "macro_factor", "regime", "scenario", "indicator", "metric"),
        ("ticker", "company", "sector", "industry", "strategy_tag", "thesis", "scenario"),
        0.80,
    ),
    PredicateSpec(
        "caused_by",
        "The subject is explained by the object.",
        ("event", "outcome", "price_action", "regime", "thesis", "scenario"),
        ("risk", "catalyst", "event", "macro_factor", "indicator", "metric", "regime"),
        0.75,
    ),
    PredicateSpec(
        "leads_to",
        "The subject tends to produce or precede the object.",
        ("risk", "catalyst", "event", "macro_factor", "indicator", "metric", "regime", "scenario", "strategy_tag"),
        ("outcome", "price_action", "regime", "scenario", "thesis", "ticker", "company", "sector", "industry"),
        0.75,
    ),
    PredicateSpec(
        "similar_setup",
        "The subject and object describe comparable investment setups.",
        ("ticker", "company", "sector", "industry", "strategy_tag", "regime", "scenario", "thesis"),
        ("ticker", "company", "sector", "industry", "strategy_tag", "regime", "scenario", "thesis"),
        0.78,
    ),
    PredicateSpec(
        "invalidates",
        "The subject breaks the object thesis or setup.",
        ("risk", "event", "indicator", "metric", "macro_factor", "regime", "scenario", "thesis"),
        ("thesis", "scenario", "strategy_tag", "ticker", "company", "sector", "industry"),
        0.88,
    ),
    PredicateSpec(
        "outcome_of",
        "The subject is an observed result of the object.",
        ("outcome", "price_action", "event", "thesis"),
        ("thesis", "scenario", "strategy_tag", "event", "ticker", "company", "sector", "industry"),
        0.75,
    ),
    PredicateSpec(
        "mentions",
        "The subject explicitly mentions the object without stronger semantics.",
        ("*",),
        ("*",),
        0.60,
    ),
    PredicateSpec(
        "contains",
        "A passage or grouping contains the object.",
        ("passage", "research_category", "sector", "industry", "market"),
        ("*",),
        0.65,
    ),
)

PREDICATE_BY_NAME = {spec.name: spec for spec in PREDICATES}
ALLOWED_ENTITY_TYPES = frozenset(ENTITY_TYPES)
ALLOWED_PREDICATES = frozenset(PREDICATE_BY_NAME)


def _slug(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def canonical_entity_type(value: str) -> str:
    token = _slug(value)
    return _ENTITY_TYPE_ALIASES.get(token, token)


def canonical_predicate(value: str) -> str:
    return _slug(value)


def is_allowed_entity_type(value: str) -> bool:
    return canonical_entity_type(value) in ALLOWED_ENTITY_TYPES


def is_allowed_predicate(value: str) -> bool:
    return canonical_predicate(value) in ALLOWED_PREDICATES


def predicate_allows(predicate: str, subject_type: str, object_type: str) -> bool:
    spec = PREDICATE_BY_NAME.get(canonical_predicate(predicate))
    if spec is None:
        return False
    subj = canonical_entity_type(subject_type)
    obj = canonical_entity_type(object_type)
    subject_ok = "*" in spec.subject_types or subj in spec.subject_types
    object_ok = "*" in spec.object_types or obj in spec.object_types
    return subject_ok and object_ok


def predicate_min_confidence(predicate: str, default: float = 0.65) -> float:
    spec = PREDICATE_BY_NAME.get(canonical_predicate(predicate))
    if spec is None:
        return max(0.0, min(float(default), 1.0))
    return max(0.0, min(float(spec.min_confidence), 1.0))


def ontology_prompt_block() -> str:
    lines = [
        f"ontology_version: {ONTOLOGY_VERSION}",
        "entity_types: " + ", ".join(ENTITY_TYPES),
        "predicates:",
    ]
    for spec in PREDICATES:
        lines.append(
            "- "
            + spec.name
            + " | subject_types="
            + ",".join(spec.subject_types)
            + " | object_types="
            + ",".join(spec.object_types)
            + f" | min_confidence={spec.min_confidence:.2f}"
            + " | "
            + spec.description
        )
    return "\n".join(lines)
