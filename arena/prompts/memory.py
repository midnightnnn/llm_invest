from __future__ import annotations

import json
from typing import Any

from arena.prompts.loader import load_prompt_text, render_prompt_text

COMPACTION_SYSTEM_INSTRUCTION = load_prompt_text("memory", "compaction_system.txt")
RELATION_EXTRACTOR_SYSTEM_INSTRUCTION = load_prompt_text("memory", "relation_extraction_system.txt")


def _text(value: Any) -> str:
    return str(value or "").strip()


def build_relation_extraction_prompt(
    source: Any,
    *,
    max_triples: int = 6,
    ontology_block: str,
) -> str:
    source_payload = {
        "source_table": getattr(source, "source_table", ""),
        "source_id": getattr(source, "source_id", ""),
        "source_label": getattr(source, "source_label", ""),
        "agent_id": getattr(source, "agent_id", None),
        "trading_mode": getattr(source, "trading_mode", ""),
        "cycle_id": getattr(source, "cycle_id", None),
        "text": getattr(source, "source_text", ""),
    }
    schema = {
        "triples": [
            {
                "subject": {"label": "compact canonical graph node name", "type": "ticker|risk|catalyst|..."},
                "predicate": "supports|contradicts|risk_to|caused_by|leads_to|similar_setup|invalidates|outcome_of|mentions|contains",
                "object": {"label": "compact canonical graph node name", "type": "ticker|thesis|outcome|..."},
                "confidence": 0.0,
                "evidence_text": "copy one exact span from source text",
            }
        ]
    }
    return render_prompt_text(
        "memory",
        "relation_extraction_user_template.txt",
        values={
            "max_triples": max(1, int(max_triples)),
            "ontology_block": _text(ontology_block),
            "schema_json": json.dumps(schema, ensure_ascii=False),
            "source_json": json.dumps(source_payload, ensure_ascii=False, default=str),
        },
    )
