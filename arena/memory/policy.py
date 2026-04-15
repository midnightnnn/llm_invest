from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass
from typing import Any

from arena.memory.query_builders import MEMORY_INJECTABLE_TOOLS

logger = logging.getLogger(__name__)

MEMORY_POLICY_CONFIG_KEY = "memory_policy"
GLOBAL_MEMORY_PROMPT_CONFIG_KEY = "memory_compactor_prompt"
GLOBAL_MEMORY_PROMPT_TENANT = "global"
MEMORY_FORGETTING_TUNING_STATE_CONFIG_KEY = "memory_forgetting_tuning_state"
MEMORY_RELATION_TUNING_STATE_CONFIG_KEY = "memory_relation_tuning_state"


@dataclass(frozen=True, slots=True)
class MemoryFieldSpec:
    path: str
    label: str
    group: str
    value_type: str
    options: tuple[str, ...] = ()
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    description: str = ""
    scope: str = "tenant"


GROUP_SPECS: tuple[dict[str, str], ...] = (
    {"id": "storage", "label": "Storage", "color": "#60a5fa", "description": "Memory persistence layers and cache behavior."},
    {"id": "event_types", "label": "Event Types", "color": "#34d399", "description": "Which memory event categories are persisted and retrievable."},
    {"id": "hierarchy", "label": "Hierarchy", "color": "#22c55e", "description": "Temporal tiering and semantic promotion controls."},
    {"id": "tagging", "label": "Tagging", "color": "#14b8a6", "description": "Context tags and regime-aware retrieval shaping."},
    {"id": "forgetting", "label": "Forgetting", "color": "#f59e0b", "description": "Access logging and adaptive decay groundwork."},
    {"id": "graph", "label": "Graph", "color": "#ef4444", "description": "Full causal graph rollout controls and traversal limits."},
    {"id": "compaction", "label": "Compaction", "color": "#a78bfa", "description": "Post-cycle lesson synthesis and compaction controls."},
    {"id": "retrieval", "label": "Retrieval", "color": "#fb923c", "description": "Vector search, reranking, and final context assembly."},
    {"id": "react_injection", "label": "REACT Injection", "color": "#f87171", "description": "Tool-result memory injection behavior during REACT."},
    {"id": "cleanup", "label": "Cleanup", "color": "#94a3b8", "description": "Pruning policy for stale or low-signal memories."},
)

BRANCH_SPECS: tuple[dict[str, str], ...] = (
    {
        "id": "storage.bigquery",
        "label": "BigQuery",
        "group": "storage",
        "parent": "storage",
        "description": "Structured long-term memory events stored in BigQuery.",
    },
    {
        "id": "storage.firestore",
        "label": "Firestore",
        "group": "storage",
        "parent": "storage",
        "description": "Vector index and embedding cache behavior for Firestore search.",
    },
    {
        "id": "retrieval.vector_search",
        "label": "Vector Search",
        "group": "retrieval",
        "parent": "retrieval",
        "description": "Nearest-neighbor retrieval before reranking.",
    },
    {
        "id": "retrieval.reranking",
        "label": "Reranking",
        "group": "retrieval",
        "parent": "retrieval",
        "description": "Score shaping for event type, recency, ticker overlap, and outcomes.",
    },
    {
        "id": "react_injection.tools",
        "label": "Injectable Tools",
        "group": "react_injection",
        "parent": "react_injection",
        "description": "Per-tool memory injection gates during REACT execution.",
    },
    {
        "id": "forgetting.tuning",
        "label": "Tuning",
        "group": "forgetting",
        "parent": "forgetting",
        "description": "Shadow recommendations, EMA rollout, and auto-promotion guardrails.",
    },
    {
        "id": "graph.semantic_triples",
        "label": "Semantic Triples",
        "group": "graph",
        "parent": "graph",
        "description": "Shadow/boost/inject controls for source-grounded relation triples.",
    },
    {
        "id": "graph.semantic_triples.tuning",
        "label": "Triple Tuning",
        "group": "graph",
        "parent": "graph.semantic_triples",
        "description": "Automatic shadow/inject transitions based on relation quality, safety, and impact metrics.",
    },
)


FIELD_SPECS: tuple[MemoryFieldSpec, ...] = (
    MemoryFieldSpec(
        path="storage.embed_cache_max",
        label="Embed Cache Max",
        group="storage",
        value_type="int",
        min_value=16,
        max_value=4096,
        step=1,
        description="Per-process in-memory embedding cache size.",
    ),
    MemoryFieldSpec(
        path="event_types.trade_execution",
        label="trade_execution",
        group="event_types",
        value_type="toggle",
        description="Persist and retrieve trade execution memories.",
    ),
    MemoryFieldSpec(
        path="event_types.strategy_reflection",
        label="strategy_reflection",
        group="event_types",
        value_type="toggle",
        description="Persist compacted lesson memories.",
    ),
    MemoryFieldSpec(
        path="event_types.manual_note",
        label="manual_note",
        group="event_types",
        value_type="toggle",
        description="Persist manual operator notes.",
    ),
    MemoryFieldSpec(
        path="event_types.react_tools_summary",
        label="react_tools_summary",
        group="event_types",
        value_type="toggle",
        description="Persist REACT tool summary memories.",
    ),
    MemoryFieldSpec(
        path="event_types.thesis_open",
        label="thesis_open",
        group="event_types",
        value_type="toggle",
        description="Persist thesis-open memories created on executed entries.",
    ),
    MemoryFieldSpec(
        path="event_types.thesis_update",
        label="thesis_update",
        group="event_types",
        value_type="toggle",
        description="Persist thesis-update memories when the active investment logic materially changes.",
    ),
    MemoryFieldSpec(
        path="event_types.thesis_invalidated",
        label="thesis_invalidated",
        group="event_types",
        value_type="toggle",
        description="Persist thesis-invalidated memories when a sell explicitly breaks the investment case.",
    ),
    MemoryFieldSpec(
        path="event_types.thesis_realized",
        label="thesis_realized",
        group="event_types",
        value_type="toggle",
        description="Persist thesis-realized memories when the investment case closes successfully.",
    ),
    MemoryFieldSpec(
        path="event_types.candidate_screen_hit",
        label="candidate_screen_hit",
        group="event_types",
        value_type="toggle",
        description="Persist short-lived non-held candidates surfaced by screening.",
    ),
    MemoryFieldSpec(
        path="event_types.candidate_watchlist",
        label="candidate_watchlist",
        group="event_types",
        value_type="toggle",
        description="Persist non-held candidates that repeated or received follow-up analysis.",
    ),
    MemoryFieldSpec(
        path="event_types.candidate_rejected",
        label="candidate_rejected",
        group="event_types",
        value_type="toggle",
        description="Persist rejected candidate notes so repeated screen hits keep their negative prior.",
    ),
    MemoryFieldSpec(
        path="event_types.candidate_thesis",
        label="candidate_thesis",
        group="event_types",
        value_type="toggle",
        description="Persist non-held candidate thesis memories when explicit thesis evidence exists.",
    ),
    MemoryFieldSpec(
        path="hierarchy.enabled",
        label="Hierarchy Enabled",
        group="hierarchy",
        value_type="toggle",
        description="Enable temporal working/episodic/semantic tiering.",
    ),
    MemoryFieldSpec(
        path="hierarchy.working_ttl_hours",
        label="Working TTL Hours",
        group="hierarchy",
        value_type="int",
        min_value=1,
        max_value=336,
        step=1,
        description="Retention window for working-tier memories.",
    ),
    MemoryFieldSpec(
        path="hierarchy.episodic_ttl_days",
        label="Episodic TTL Days",
        group="hierarchy",
        value_type="int",
        min_value=7,
        max_value=3650,
        step=1,
        description="Default retention window for episodic memories.",
    ),
    MemoryFieldSpec(
        path="hierarchy.semantic_promotion_min_support",
        label="Semantic Support Min",
        group="hierarchy",
        value_type="int",
        min_value=2,
        max_value=16,
        step=1,
        description="Minimum repeated support before semantic promotion can occur.",
    ),
    MemoryFieldSpec(
        path="tagging.enabled",
        label="Tagging Enabled",
        group="tagging",
        value_type="toggle",
        description="Enable structured context tags on memory rows.",
    ),
    MemoryFieldSpec(
        path="tagging.max_tags",
        label="Max Tags",
        group="tagging",
        value_type="int",
        min_value=1,
        max_value=16,
        step=1,
        description="Upper bound on stored tags per memory event.",
    ),
    MemoryFieldSpec(
        path="tagging.regime_bonus",
        label="Regime Bonus",
        group="tagging",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Retrieval bonus for matching market regime tags.",
    ),
    MemoryFieldSpec(
        path="tagging.strategy_bonus",
        label="Strategy Bonus",
        group="tagging",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Retrieval bonus for matching strategy-family tags.",
    ),
    MemoryFieldSpec(
        path="tagging.sector_bonus",
        label="Sector Bonus",
        group="tagging",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Retrieval bonus for matching sector tags.",
    ),
    MemoryFieldSpec(
        path="forgetting.enabled",
        label="Forgetting Enabled",
        group="forgetting",
        value_type="toggle",
        description="Enable adaptive forgetting metadata and decay-aware scoring.",
    ),
    MemoryFieldSpec(
        path="forgetting.access_log_enabled",
        label="Access Log",
        group="forgetting",
        value_type="toggle",
        description="Write memory access events for future decay computation.",
    ),
    MemoryFieldSpec(
        path="forgetting.default_decay_factor",
        label="Decay Factor",
        group="forgetting",
        value_type="float",
        min_value=0.9,
        max_value=1.0,
        step=0.001,
        description="Default per-period decay multiplier before access adjustments.",
    ),
    MemoryFieldSpec(
        path="forgetting.access_curve",
        label="Access Curve",
        group="forgetting",
        value_type="select",
        options=("sqrt", "log", "capped_linear"),
        description="How repeated access slows forgetting decay.",
    ),
    MemoryFieldSpec(
        path="forgetting.tier_weight_working",
        label="Working Weight",
        group="forgetting",
        value_type="float",
        min_value=0.1,
        max_value=4.0,
        step=0.01,
        description="Decay multiplier applied to working-tier memories.",
    ),
    MemoryFieldSpec(
        path="forgetting.tier_weight_episodic",
        label="Episodic Weight",
        group="forgetting",
        value_type="float",
        min_value=0.1,
        max_value=4.0,
        step=0.01,
        description="Decay multiplier applied to episodic-tier memories.",
    ),
    MemoryFieldSpec(
        path="forgetting.tier_weight_semantic",
        label="Semantic Weight",
        group="forgetting",
        value_type="float",
        min_value=0.05,
        max_value=2.0,
        step=0.01,
        description="Decay multiplier applied to semantic-tier memories.",
    ),
    MemoryFieldSpec(
        path="forgetting.min_effective_score",
        label="Min Effective Score",
        group="forgetting",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Lower bound for decayed score retention.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.enabled",
        label="Tuning Enabled",
        group="forgetting",
        value_type="toggle",
        description="Enable shadow recommendation and bounded EMA apply for forgetting knobs.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.mode",
        label="Tuning Mode",
        group="forgetting",
        value_type="select",
        options=("shadow", "bounded_ema"),
        description="Current forgetting tuner operating mode.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.lookback_days",
        label="Lookback Days",
        group="forgetting",
        value_type="int",
        min_value=7,
        max_value=180,
        step=1,
        description="Trailing window used to evaluate access-history candidates.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.stability_window_days",
        label="Stability Days",
        group="forgetting",
        value_type="int",
        min_value=3,
        max_value=60,
        step=1,
        description="Short trailing window used to penalize unstable recommendations.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.min_access_events",
        label="Min Access Events",
        group="forgetting",
        value_type="int",
        min_value=1,
        max_value=50000,
        step=1,
        description="Minimum access events required before tuner recommendations are eligible.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.min_prompt_uses",
        label="Min Prompt Uses",
        group="forgetting",
        value_type="int",
        min_value=1,
        max_value=50000,
        step=1,
        description="Minimum prompt insertions required before tuner recommendations are eligible.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.min_unique_memories",
        label="Min Unique Memories",
        group="forgetting",
        value_type="int",
        min_value=1,
        max_value=50000,
        step=1,
        description="Minimum distinct memories required before tuner recommendations are eligible.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.ema_alpha",
        label="EMA Alpha",
        group="forgetting",
        value_type="float",
        min_value=0.01,
        max_value=0.5,
        step=0.01,
        description="Bounded EMA blend used when applying tuner recommendations.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.max_decay_factor_delta",
        label="Max Decay Delta",
        group="forgetting",
        value_type="float",
        min_value=0.0005,
        max_value=0.02,
        step=0.0005,
        description="Maximum bounded EMA step allowed for decay_factor per tuning run.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.max_min_effective_score_delta",
        label="Max Floor Delta",
        group="forgetting",
        value_type="float",
        min_value=0.005,
        max_value=0.20,
        step=0.005,
        description="Maximum bounded EMA step allowed for min_effective_score per tuning run.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.max_tier_weight_delta",
        label="Max Tier Delta",
        group="forgetting",
        value_type="float",
        min_value=0.01,
        max_value=1.0,
        step=0.01,
        description="Maximum bounded EMA step allowed for tier weights per tuning run.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.objective_topk",
        label="Objective Top-K",
        group="forgetting",
        value_type="int",
        min_value=4,
        max_value=128,
        step=1,
        description="Top-K depth used when scoring tuning candidates.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.auto_promote_enabled",
        label="Auto Promote",
        group="forgetting",
        value_type="toggle",
        description="Automatically move shadow tuning into bounded EMA when stable enough.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.auto_promote_min_shadow_days",
        label="Promote Shadow Days",
        group="forgetting",
        value_type="int",
        min_value=1,
        max_value=90,
        step=1,
        description="Minimum number of shadow days before auto-promotion is allowed.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.auto_promote_min_shadow_runs",
        label="Promote Shadow Runs",
        group="forgetting",
        value_type="int",
        min_value=1,
        max_value=100,
        step=1,
        description="Minimum number of shadow tuning evaluations before auto-promotion is allowed.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.auto_promote_required_stable_runs",
        label="Promote Stable Runs",
        group="forgetting",
        value_type="int",
        min_value=1,
        max_value=100,
        step=1,
        description="Required consecutive stable tuning runs before auto-promotion.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.auto_promote_required_improving_runs",
        label="Promote Improve Runs",
        group="forgetting",
        value_type="int",
        min_value=1,
        max_value=100,
        step=1,
        description="Required consecutive improving recommendations before auto-promotion.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.auto_promote_max_recommendation_drift",
        label="Promote Drift Max",
        group="forgetting",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Maximum recommendation drift allowed before auto-promotion is blocked.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.auto_demote_enabled",
        label="Auto Demote",
        group="forgetting",
        value_type="toggle",
        description="Automatically move bounded EMA back to shadow when tuning health degrades.",
    ),
    MemoryFieldSpec(
        path="forgetting.tuning.auto_demote_unhealthy_runs",
        label="Demote Unhealthy Runs",
        group="forgetting",
        value_type="int",
        min_value=1,
        max_value=100,
        step=1,
        description="Consecutive unhealthy runs required before auto-demotion.",
    ),
    MemoryFieldSpec(
        path="graph.enabled",
        label="Graph Enabled",
        group="graph",
        value_type="toggle",
        description="Enable graph node/edge projection and graph-aware retrieval.",
    ),
    MemoryFieldSpec(
        path="graph.max_expansion_hops",
        label="Expansion Hops",
        group="graph",
        value_type="int",
        min_value=1,
        max_value=4,
        step=1,
        description="Maximum causal graph expansion depth from seed memories.",
    ),
    MemoryFieldSpec(
        path="graph.max_expanded_nodes",
        label="Expanded Nodes Max",
        group="graph",
        value_type="int",
        min_value=2,
        max_value=128,
        step=1,
        description="Upper bound on graph nodes expanded into retrieval context.",
    ),
    MemoryFieldSpec(
        path="graph.inferred_edge_min_confidence",
        label="Inferred Edge Confidence",
        group="graph",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Minimum confidence required before inferred graph edges are used.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.enabled",
        label="Semantic Triples",
        group="graph",
        value_type="toggle",
        description="Collect and optionally use source-grounded relation triples.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.mode",
        label="Triple Mode",
        group="graph",
        value_type="select",
        options=("shadow", "boost", "inject"),
        description="shadow stores only, boost affects retrieval score, inject also adds relation hints to prompts.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.min_confidence",
        label="Triple Confidence",
        group="graph",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Minimum relation confidence used for boost/inject read paths.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.max_candidates",
        label="Triple Candidates",
        group="graph",
        value_type="int",
        min_value=1,
        max_value=64,
        step=1,
        description="Maximum relation-linked memory candidates considered per retrieval track.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.boost_bonus_base",
        label="Triple Boost Base",
        group="graph",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Base retrieval bonus multiplied by relation confidence.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.boost_bonus_cap",
        label="Triple Boost Cap",
        group="graph",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Maximum retrieval bonus from semantic relation triples.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.max_relation_context_items",
        label="Relation Hints",
        group="graph",
        value_type="int",
        min_value=0,
        max_value=16,
        step=1,
        description="Maximum relation hints injected into prompt context when mode is inject.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.tuning.enabled",
        label="Triple Auto Tuning",
        group="graph",
        value_type="toggle",
        description="Automatically evaluate semantic triple metrics for shadow/inject transitions.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.tuning.auto_transition_enabled",
        label="Triple Auto Mode",
        group="graph",
        value_type="toggle",
        description="Allow metric gates to switch semantic triples between shadow and inject without manual approval.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.tuning.lookback_days",
        label="Triple Lookback Days",
        group="graph",
        value_type="int",
        min_value=3,
        max_value=180,
        step=1,
        description="Window used to evaluate relation extraction health.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.tuning.stability_window_days",
        label="Triple Stability Days",
        group="graph",
        value_type="int",
        min_value=1,
        max_value=60,
        step=1,
        description="Recent sub-window compared with the longer baseline for drift.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.tuning.required_healthy_evaluations",
        label="Healthy Runs",
        group="graph",
        value_type="int",
        min_value=2,
        max_value=30,
        step=1,
        description="Consecutive healthy evaluations required before shadow can auto-promote to inject.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.tuning.demote_unhealthy_evaluations",
        label="Unhealthy Runs",
        group="graph",
        value_type="int",
        min_value=1,
        max_value=30,
        step=1,
        description="Consecutive unhealthy evaluations required before inject auto-demotes to shadow.",
    ),
    MemoryFieldSpec(
        path="graph.semantic_triples.tuning.post_demote_cooldown_evaluations",
        label="Demote Cooldown",
        group="graph",
        value_type="int",
        min_value=0,
        max_value=30,
        step=1,
        description="Evaluations to wait after demotion before promotion can happen again. Zero derives from Healthy Runs.",
    ),
    MemoryFieldSpec(
        path="compaction.enabled",
        label="Compaction Enabled",
        group="compaction",
        value_type="toggle",
        description="Run the post-cycle memory compactor.",
    ),
    MemoryFieldSpec(
        path="compaction.cycle_event_limit",
        label="Cycle Event Limit",
        group="compaction",
        value_type="int",
        min_value=4,
        max_value=64,
        step=1,
        description="How many cycle artifacts feed the compactor prompt.",
    ),
    MemoryFieldSpec(
        path="compaction.recent_lessons_limit",
        label="Recent Lessons Limit",
        group="compaction",
        value_type="int",
        min_value=1,
        max_value=12,
        step=1,
        description="How many prior lessons the compactor sees.",
    ),
    MemoryFieldSpec(
        path="compaction.max_reflections",
        label="Max Reflections",
        group="compaction",
        value_type="int",
        min_value=1,
        max_value=12,
        step=1,
        description="Upper bound on lessons saved per cycle.",
    ),
    MemoryFieldSpec(
        path="compaction.thesis_chain_enabled",
        label="Thesis Chains",
        group="compaction",
        value_type="toggle",
        description="Prioritize closed thesis chains during compaction when available.",
    ),
    MemoryFieldSpec(
        path="compaction.thesis_chain_max_chains_per_cycle",
        label="Max Thesis Chains",
        group="compaction",
        value_type="int",
        min_value=1,
        max_value=12,
        step=1,
        description="Upper bound on closed thesis chains included in one compaction prompt.",
    ),
    MemoryFieldSpec(
        path="compaction.thesis_chain_max_events_per_chain",
        label="Max Events / Chain",
        group="compaction",
        value_type="int",
        min_value=2,
        max_value=16,
        step=1,
        description="Upper bound on thesis-lifecycle events retained per chain in compaction input.",
    ),
    MemoryFieldSpec(
        path="retrieval.context_limit",
        label="Max Events",
        group="retrieval",
        value_type="int",
        min_value=1,
        max_value=100,
        step=1,
        description="Final number of memories injected into cycle context.",
    ),
    MemoryFieldSpec(
        path="retrieval.vector_search_enabled",
        label="Vector Search",
        group="retrieval",
        value_type="toggle",
        description="Allow vector search retrieval for memory context and tools.",
    ),
    MemoryFieldSpec(
        path="retrieval.vector_search_limit",
        label="Search Limit",
        group="retrieval",
        value_type="int",
        min_value=2,
        max_value=10,
        step=1,
        description="Per-query nearest-neighbor fetch limit before reranking.",
    ),
    MemoryFieldSpec(
        path="retrieval.peer_lessons_enabled",
        label="Peer Lessons",
        group="retrieval",
        value_type="toggle",
        description="Allow cross-agent peer lesson lookup.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.type_bonus_reflection",
        label="Reflection Bonus",
        group="retrieval",
        value_type="float",
        min_value=-1.0,
        max_value=1.0,
        step=0.01,
        description="Bonus applied to strategy_reflection memories.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.type_bonus_trade",
        label="Trade Bonus",
        group="retrieval",
        value_type="float",
        min_value=-1.0,
        max_value=1.0,
        step=0.01,
        description="Bonus applied to trade_execution memories.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.type_bonus_manual",
        label="Manual Bonus",
        group="retrieval",
        value_type="float",
        min_value=-1.0,
        max_value=1.0,
        step=0.01,
        description="Bonus applied to manual_note memories.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.type_bonus_react_tools",
        label="REACT Summary Bonus",
        group="retrieval",
        value_type="float",
        min_value=-1.0,
        max_value=1.0,
        step=0.01,
        description="Bonus applied to react_tools_summary memories.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.recency_bonus_3d",
        label="Recency 3d",
        group="retrieval",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Recency bonus for memories newer than 3 days.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.recency_bonus_14d",
        label="Recency 14d",
        group="retrieval",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Recency bonus for memories newer than 14 days.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.recency_bonus_45d",
        label="Recency 45d",
        group="retrieval",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Recency bonus for memories newer than 45 days.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.ticker_bonus_base",
        label="Ticker Bonus Base",
        group="retrieval",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Base bonus when a memory overlaps an active ticker.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.ticker_bonus_step",
        label="Ticker Bonus Step",
        group="retrieval",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Additional bonus for each extra overlapping ticker.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.ticker_bonus_max",
        label="Ticker Bonus Max",
        group="retrieval",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Upper bound for ticker overlap bonus.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.outcome_bonus_max",
        label="Outcome Bonus Max",
        group="retrieval",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Upper bound for decisiveness bonus from outcome_score.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.effective_score_bonus_scale",
        label="Effective Scale",
        group="retrieval",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Weak reranking lift for memories with strong adaptive effective_score.",
    ),
    MemoryFieldSpec(
        path="retrieval.reranking.effective_score_bonus_cap",
        label="Effective Cap",
        group="retrieval",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Upper bound for the effective_score reranking bonus.",
    ),
    MemoryFieldSpec(
        path="react_injection.enabled",
        label="REACT Memory Injection",
        group="react_injection",
        value_type="toggle",
        description="Inject relevant memories into tool results during REACT.",
    ),
    MemoryFieldSpec(
        path="cleanup.enabled",
        label="Auto Cleanup",
        group="cleanup",
        value_type="toggle",
        description="Allow scheduled cleanup runs to prune stale low-signal memories.",
    ),
    MemoryFieldSpec(
        path="cleanup.max_age_days",
        label="Max Age Days",
        group="cleanup",
        value_type="int",
        min_value=30,
        max_value=3650,
        step=1,
        description="Delete memories older than this age threshold.",
    ),
    MemoryFieldSpec(
        path="cleanup.min_score",
        label="Min Score",
        group="cleanup",
        value_type="float",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        description="Delete memories whose effective score falls below this threshold.",
    ),
) + tuple(
    MemoryFieldSpec(
        path=f"react_injection.tools.{tool_name}",
        label=tool_name,
        group="react_injection",
        value_type="toggle",
        description=f"Allow REACT-time memory injection for {tool_name}.",
    )
    for tool_name in sorted(MEMORY_INJECTABLE_TOOLS)
)


_PATH_ALIASES: dict[str, str] = {
    "context_max_memory_events": "retrieval.context_limit",
    "embed_cache_max": "storage.embed_cache_max",
    "compaction_enabled": "compaction.enabled",
    "cycle_event_limit": "compaction.cycle_event_limit",
    "recent_lessons_limit": "compaction.recent_lessons_limit",
    "max_reflections": "compaction.max_reflections",
    "vector_search_enabled": "retrieval.vector_search_enabled",
    "search_limit": "retrieval.vector_search_limit",
    "peer_lessons_enabled": "retrieval.peer_lessons_enabled",
    "type_bonus_reflection": "retrieval.reranking.type_bonus_reflection",
    "type_bonus_trade": "retrieval.reranking.type_bonus_trade",
    "type_bonus_manual": "retrieval.reranking.type_bonus_manual",
    "type_bonus_react_tools": "retrieval.reranking.type_bonus_react_tools",
    "recency_bonus_3d": "retrieval.reranking.recency_bonus_3d",
    "recency_bonus_14d": "retrieval.reranking.recency_bonus_14d",
    "recency_bonus_45d": "retrieval.reranking.recency_bonus_45d",
    "ticker_bonus_base": "retrieval.reranking.ticker_bonus_base",
    "ticker_bonus_step": "retrieval.reranking.ticker_bonus_step",
    "ticker_bonus_max": "retrieval.reranking.ticker_bonus_max",
    "outcome_bonus_max": "retrieval.reranking.outcome_bonus_max",
    "effective_score_bonus_scale": "retrieval.reranking.effective_score_bonus_scale",
    "effective_score_bonus_cap": "retrieval.reranking.effective_score_bonus_cap",
    "react_memory_injection": "react_injection.enabled",
    "auto_cleanup_enabled": "cleanup.enabled",
    "max_age_days": "cleanup.max_age_days",
    "min_score": "cleanup.min_score",
}


FIELD_PARENT_OVERRIDES: dict[str, str] = {
    "storage.embed_cache_max": "storage.firestore",
    "retrieval.context_limit": "storage.bigquery",
    "forgetting.tuning.enabled": "forgetting.tuning",
    "forgetting.tuning.mode": "forgetting.tuning",
    "forgetting.tuning.lookback_days": "forgetting.tuning",
    "forgetting.tuning.stability_window_days": "forgetting.tuning",
    "forgetting.tuning.min_access_events": "forgetting.tuning",
    "forgetting.tuning.min_prompt_uses": "forgetting.tuning",
    "forgetting.tuning.min_unique_memories": "forgetting.tuning",
    "forgetting.tuning.ema_alpha": "forgetting.tuning",
    "forgetting.tuning.max_decay_factor_delta": "forgetting.tuning",
    "forgetting.tuning.max_min_effective_score_delta": "forgetting.tuning",
    "forgetting.tuning.max_tier_weight_delta": "forgetting.tuning",
    "forgetting.tuning.objective_topk": "forgetting.tuning",
    "forgetting.tuning.auto_promote_enabled": "forgetting.tuning",
    "forgetting.tuning.auto_promote_min_shadow_days": "forgetting.tuning",
    "forgetting.tuning.auto_promote_min_shadow_runs": "forgetting.tuning",
    "forgetting.tuning.auto_promote_required_stable_runs": "forgetting.tuning",
    "forgetting.tuning.auto_promote_required_improving_runs": "forgetting.tuning",
    "forgetting.tuning.auto_promote_max_recommendation_drift": "forgetting.tuning",
    "forgetting.tuning.auto_demote_enabled": "forgetting.tuning",
    "forgetting.tuning.auto_demote_unhealthy_runs": "forgetting.tuning",
    "retrieval.vector_search_enabled": "retrieval.vector_search",
    "retrieval.vector_search_limit": "retrieval.vector_search",
    "retrieval.peer_lessons_enabled": "retrieval.vector_search",
    "retrieval.reranking.type_bonus_reflection": "retrieval.reranking",
    "retrieval.reranking.type_bonus_trade": "retrieval.reranking",
    "retrieval.reranking.type_bonus_manual": "retrieval.reranking",
    "retrieval.reranking.type_bonus_react_tools": "retrieval.reranking",
    "retrieval.reranking.recency_bonus_3d": "retrieval.reranking",
    "retrieval.reranking.recency_bonus_14d": "retrieval.reranking",
    "retrieval.reranking.recency_bonus_45d": "retrieval.reranking",
    "retrieval.reranking.ticker_bonus_base": "retrieval.reranking",
    "retrieval.reranking.ticker_bonus_step": "retrieval.reranking",
    "retrieval.reranking.ticker_bonus_max": "retrieval.reranking",
    "retrieval.reranking.outcome_bonus_max": "retrieval.reranking",
    "retrieval.reranking.effective_score_bonus_scale": "retrieval.reranking",
    "retrieval.reranking.effective_score_bonus_cap": "retrieval.reranking",
    "graph.semantic_triples.enabled": "graph.semantic_triples",
    "graph.semantic_triples.mode": "graph.semantic_triples",
    "graph.semantic_triples.min_confidence": "graph.semantic_triples",
    "graph.semantic_triples.max_candidates": "graph.semantic_triples",
    "graph.semantic_triples.boost_bonus_base": "graph.semantic_triples",
    "graph.semantic_triples.boost_bonus_cap": "graph.semantic_triples",
    "graph.semantic_triples.max_relation_context_items": "graph.semantic_triples",
    "graph.semantic_triples.tuning.enabled": "graph.semantic_triples.tuning",
    "graph.semantic_triples.tuning.auto_transition_enabled": "graph.semantic_triples.tuning",
    "graph.semantic_triples.tuning.lookback_days": "graph.semantic_triples.tuning",
    "graph.semantic_triples.tuning.stability_window_days": "graph.semantic_triples.tuning",
    "graph.semantic_triples.tuning.required_healthy_evaluations": "graph.semantic_triples.tuning",
    "graph.semantic_triples.tuning.demote_unhealthy_evaluations": "graph.semantic_triples.tuning",
    "graph.semantic_triples.tuning.post_demote_cooldown_evaluations": "graph.semantic_triples.tuning",
}


def default_memory_policy(
    *,
    context_limit: int = 32,
    embed_cache_max: int = 128,
    compaction_enabled: bool = True,
    cycle_event_limit: int = 12,
    recent_lessons_limit: int = 4,
    max_reflections: int = 3,
) -> dict[str, Any]:
    return {
        "storage": {
            "embed_cache_max": max(16, int(embed_cache_max)),
        },
        "event_types": {
            "trade_execution": True,
            "strategy_reflection": True,
            "manual_note": True,
            "react_tools_summary": True,
            "thesis_open": True,
            "thesis_update": True,
            "thesis_invalidated": True,
            "thesis_realized": True,
            "candidate_screen_hit": True,
            "candidate_watchlist": True,
            "candidate_rejected": True,
            "candidate_thesis": True,
        },
        "hierarchy": {
            "enabled": True,
            "working_ttl_hours": 36,
            "episodic_ttl_days": 90,
            "semantic_promotion_min_support": 3,
        },
        "tagging": {
            "enabled": True,
            "max_tags": 6,
            "regime_bonus": 0.25,
            "strategy_bonus": 0.18,
            "sector_bonus": 0.10,
        },
        "forgetting": {
            "enabled": True,
            "access_log_enabled": True,
            "default_decay_factor": 0.985,
            "access_curve": "sqrt",
            "tier_weight_working": 2.0,
            "tier_weight_episodic": 1.0,
            "tier_weight_semantic": 0.35,
            "min_effective_score": 0.15,
            "tuning": {
                "enabled": True,
                "mode": "shadow",
                "lookback_days": 30,
                "stability_window_days": 7,
                "min_access_events": 500,
                "min_prompt_uses": 100,
                "min_unique_memories": 100,
                "ema_alpha": 0.10,
                "max_decay_factor_delta": 0.003,
                "max_min_effective_score_delta": 0.03,
                "max_tier_weight_delta": 0.10,
                "objective_topk": 24,
                "auto_promote_enabled": False,
                "auto_promote_min_shadow_days": 14,
                "auto_promote_min_shadow_runs": 10,
                "auto_promote_required_stable_runs": 5,
                "auto_promote_required_improving_runs": 3,
                "auto_promote_max_recommendation_drift": 0.25,
                "auto_demote_enabled": False,
                "auto_demote_unhealthy_runs": 3,
            },
        },
        "graph": {
            "enabled": True,
            "max_expansion_hops": 1,
            "max_expanded_nodes": 12,
            "inferred_edge_min_confidence": 0.75,
            "semantic_triples": {
                "enabled": True,
                "mode": "shadow",
                "min_confidence": 0.75,
                "max_candidates": 8,
                "boost_bonus_base": 0.12,
                "boost_bonus_cap": 0.18,
                "max_relation_context_items": 4,
                "tuning": {
                    "enabled": True,
                    "auto_transition_enabled": True,
                    "lookback_days": 30,
                    "stability_window_days": 7,
                    "min_sources": 0,
                    "min_accepted_triples": 0,
                    "required_healthy_evaluations": 3,
                    "demote_unhealthy_evaluations": 2,
                    "post_demote_cooldown_evaluations": 0,
                    "demote_on_version_change": True,
                },
            },
        },
        "compaction": {
            "enabled": bool(compaction_enabled),
            "cycle_event_limit": max(4, int(cycle_event_limit)),
            "recent_lessons_limit": max(1, int(recent_lessons_limit)),
            "max_reflections": max(1, int(max_reflections)),
            "thesis_chain_enabled": True,
            "thesis_chain_max_chains_per_cycle": 2,
            "thesis_chain_max_events_per_chain": 6,
        },
        "retrieval": {
            "context_limit": max(1, int(context_limit)),
            "vector_search_enabled": True,
            "vector_search_limit": 5,
            "peer_lessons_enabled": True,
            "reranking": {
                "type_bonus_reflection": 0.45,
                "type_bonus_trade": 0.28,
                "type_bonus_manual": 0.16,
                "type_bonus_react_tools": -0.12,
                "recency_bonus_3d": 0.08,
                "recency_bonus_14d": 0.05,
                "recency_bonus_45d": 0.02,
                "ticker_bonus_base": 0.30,
                "ticker_bonus_step": 0.05,
                "ticker_bonus_max": 0.40,
                "outcome_bonus_max": 0.18,
                "effective_score_bonus_scale": 0.08,
                "effective_score_bonus_cap": 0.08,
            },
        },
        "react_injection": {
            "enabled": True,
            "tools": {tool_name: True for tool_name in sorted(MEMORY_INJECTABLE_TOOLS)},
        },
        "cleanup": {
            "enabled": False,
            "max_age_days": 180,
            "min_score": 0.30,
        },
    }


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _parse_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return copy.deepcopy(raw)
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = [part for part in str(path or "").split(".") if part]
    if not parts:
        return
    cursor = target
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def get_memory_policy_value(policy: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    cursor: Any = policy or {}
    for part in [token for token in str(path or "").split(".") if token]:
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    token = str(value).strip().lower()
    if not token:
        return default
    return token in {"1", "true", "yes", "y", "on"}


def _coerce_int(value: Any, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        out = default
    if min_value is not None:
        out = max(min_value, out)
    if max_value is not None:
        out = min(max_value, out)
    return out


def _coerce_float(value: Any, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if min_value is not None:
        out = max(min_value, out)
    if max_value is not None:
        out = min(max_value, out)
    return out


def _apply_aliases(raw: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    for alias, path in _PATH_ALIASES.items():
        if alias not in raw:
            continue
        _set_path(out, path, raw[alias])
    return out


def normalize_memory_policy(raw: Any, *, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    base = copy.deepcopy(
        _deep_merge(default_memory_policy(), _parse_json_object(defaults))
        if defaults is not None
        else default_memory_policy()
    )
    parsed = _apply_aliases(_parse_json_object(raw))
    merged = _deep_merge(base, parsed)

    normalized = copy.deepcopy(base)

    normalized["storage"]["embed_cache_max"] = _coerce_int(
        get_memory_policy_value(merged, "storage.embed_cache_max", base["storage"]["embed_cache_max"]),
        base["storage"]["embed_cache_max"],
        min_value=16,
        max_value=4096,
    )

    for event_type, default in base["event_types"].items():
        normalized["event_types"][event_type] = _coerce_bool(
            get_memory_policy_value(merged, f"event_types.{event_type}", default),
            default,
        )

    normalized["hierarchy"]["enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "hierarchy.enabled", base["hierarchy"]["enabled"]),
        base["hierarchy"]["enabled"],
    )
    normalized["hierarchy"]["working_ttl_hours"] = _coerce_int(
        get_memory_policy_value(merged, "hierarchy.working_ttl_hours", base["hierarchy"]["working_ttl_hours"]),
        base["hierarchy"]["working_ttl_hours"],
        min_value=1,
        max_value=336,
    )
    normalized["hierarchy"]["episodic_ttl_days"] = _coerce_int(
        get_memory_policy_value(merged, "hierarchy.episodic_ttl_days", base["hierarchy"]["episodic_ttl_days"]),
        base["hierarchy"]["episodic_ttl_days"],
        min_value=7,
        max_value=3650,
    )
    normalized["hierarchy"]["semantic_promotion_min_support"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "hierarchy.semantic_promotion_min_support",
            base["hierarchy"]["semantic_promotion_min_support"],
        ),
        base["hierarchy"]["semantic_promotion_min_support"],
        min_value=2,
        max_value=16,
    )

    normalized["tagging"]["enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "tagging.enabled", base["tagging"]["enabled"]),
        base["tagging"]["enabled"],
    )
    normalized["tagging"]["max_tags"] = _coerce_int(
        get_memory_policy_value(merged, "tagging.max_tags", base["tagging"]["max_tags"]),
        base["tagging"]["max_tags"],
        min_value=1,
        max_value=16,
    )
    normalized["tagging"]["regime_bonus"] = _coerce_float(
        get_memory_policy_value(merged, "tagging.regime_bonus", base["tagging"]["regime_bonus"]),
        base["tagging"]["regime_bonus"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["tagging"]["strategy_bonus"] = _coerce_float(
        get_memory_policy_value(merged, "tagging.strategy_bonus", base["tagging"]["strategy_bonus"]),
        base["tagging"]["strategy_bonus"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["tagging"]["sector_bonus"] = _coerce_float(
        get_memory_policy_value(merged, "tagging.sector_bonus", base["tagging"]["sector_bonus"]),
        base["tagging"]["sector_bonus"],
        min_value=0.0,
        max_value=1.0,
    )

    normalized["forgetting"]["enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "forgetting.enabled", base["forgetting"]["enabled"]),
        base["forgetting"]["enabled"],
    )
    normalized["forgetting"]["access_log_enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "forgetting.access_log_enabled", base["forgetting"]["access_log_enabled"]),
        base["forgetting"]["access_log_enabled"],
    )
    normalized["forgetting"]["default_decay_factor"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.default_decay_factor",
            base["forgetting"]["default_decay_factor"],
        ),
        base["forgetting"]["default_decay_factor"],
        min_value=0.9,
        max_value=1.0,
    )
    access_curve = str(
        get_memory_policy_value(merged, "forgetting.access_curve", base["forgetting"]["access_curve"]) or ""
    ).strip().lower()
    normalized["forgetting"]["access_curve"] = access_curve if access_curve in {"sqrt", "log", "capped_linear"} else base["forgetting"]["access_curve"]
    normalized["forgetting"]["tier_weight_working"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.tier_weight_working",
            base["forgetting"]["tier_weight_working"],
        ),
        base["forgetting"]["tier_weight_working"],
        min_value=0.1,
        max_value=4.0,
    )
    normalized["forgetting"]["tier_weight_episodic"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.tier_weight_episodic",
            base["forgetting"]["tier_weight_episodic"],
        ),
        base["forgetting"]["tier_weight_episodic"],
        min_value=0.1,
        max_value=4.0,
    )
    normalized["forgetting"]["tier_weight_semantic"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.tier_weight_semantic",
            base["forgetting"]["tier_weight_semantic"],
        ),
        base["forgetting"]["tier_weight_semantic"],
        min_value=0.05,
        max_value=2.0,
    )
    normalized["forgetting"]["min_effective_score"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.min_effective_score",
            base["forgetting"]["min_effective_score"],
        ),
        base["forgetting"]["min_effective_score"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["forgetting"]["tuning"]["enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "forgetting.tuning.enabled", base["forgetting"]["tuning"]["enabled"]),
        base["forgetting"]["tuning"]["enabled"],
    )
    mode = str(get_memory_policy_value(merged, "forgetting.tuning.mode", base["forgetting"]["tuning"]["mode"]) or "").strip().lower()
    normalized["forgetting"]["tuning"]["mode"] = mode if mode in {"shadow", "bounded_ema"} else base["forgetting"]["tuning"]["mode"]
    normalized["forgetting"]["tuning"]["lookback_days"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.lookback_days",
            base["forgetting"]["tuning"]["lookback_days"],
        ),
        base["forgetting"]["tuning"]["lookback_days"],
        min_value=7,
        max_value=180,
    )
    normalized["forgetting"]["tuning"]["stability_window_days"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.stability_window_days",
            base["forgetting"]["tuning"]["stability_window_days"],
        ),
        base["forgetting"]["tuning"]["stability_window_days"],
        min_value=3,
        max_value=60,
    )
    normalized["forgetting"]["tuning"]["min_access_events"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.min_access_events",
            base["forgetting"]["tuning"]["min_access_events"],
        ),
        base["forgetting"]["tuning"]["min_access_events"],
        min_value=1,
        max_value=50000,
    )
    normalized["forgetting"]["tuning"]["min_prompt_uses"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.min_prompt_uses",
            base["forgetting"]["tuning"]["min_prompt_uses"],
        ),
        base["forgetting"]["tuning"]["min_prompt_uses"],
        min_value=1,
        max_value=50000,
    )
    normalized["forgetting"]["tuning"]["min_unique_memories"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.min_unique_memories",
            base["forgetting"]["tuning"]["min_unique_memories"],
        ),
        base["forgetting"]["tuning"]["min_unique_memories"],
        min_value=1,
        max_value=50000,
    )
    normalized["forgetting"]["tuning"]["ema_alpha"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.ema_alpha",
            base["forgetting"]["tuning"]["ema_alpha"],
        ),
        base["forgetting"]["tuning"]["ema_alpha"],
        min_value=0.01,
        max_value=0.5,
    )
    normalized["forgetting"]["tuning"]["max_decay_factor_delta"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.max_decay_factor_delta",
            base["forgetting"]["tuning"]["max_decay_factor_delta"],
        ),
        base["forgetting"]["tuning"]["max_decay_factor_delta"],
        min_value=0.0005,
        max_value=0.02,
    )
    normalized["forgetting"]["tuning"]["max_min_effective_score_delta"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.max_min_effective_score_delta",
            base["forgetting"]["tuning"]["max_min_effective_score_delta"],
        ),
        base["forgetting"]["tuning"]["max_min_effective_score_delta"],
        min_value=0.005,
        max_value=0.20,
    )
    normalized["forgetting"]["tuning"]["max_tier_weight_delta"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.max_tier_weight_delta",
            base["forgetting"]["tuning"]["max_tier_weight_delta"],
        ),
        base["forgetting"]["tuning"]["max_tier_weight_delta"],
        min_value=0.01,
        max_value=1.0,
    )
    normalized["forgetting"]["tuning"]["objective_topk"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.objective_topk",
            base["forgetting"]["tuning"]["objective_topk"],
        ),
        base["forgetting"]["tuning"]["objective_topk"],
        min_value=4,
        max_value=128,
    )
    normalized["forgetting"]["tuning"]["auto_promote_enabled"] = _coerce_bool(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.auto_promote_enabled",
            base["forgetting"]["tuning"]["auto_promote_enabled"],
        ),
        base["forgetting"]["tuning"]["auto_promote_enabled"],
    )
    normalized["forgetting"]["tuning"]["auto_promote_min_shadow_days"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.auto_promote_min_shadow_days",
            base["forgetting"]["tuning"]["auto_promote_min_shadow_days"],
        ),
        base["forgetting"]["tuning"]["auto_promote_min_shadow_days"],
        min_value=1,
        max_value=90,
    )
    normalized["forgetting"]["tuning"]["auto_promote_min_shadow_runs"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.auto_promote_min_shadow_runs",
            base["forgetting"]["tuning"]["auto_promote_min_shadow_runs"],
        ),
        base["forgetting"]["tuning"]["auto_promote_min_shadow_runs"],
        min_value=1,
        max_value=100,
    )
    normalized["forgetting"]["tuning"]["auto_promote_required_stable_runs"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.auto_promote_required_stable_runs",
            base["forgetting"]["tuning"]["auto_promote_required_stable_runs"],
        ),
        base["forgetting"]["tuning"]["auto_promote_required_stable_runs"],
        min_value=1,
        max_value=100,
    )
    normalized["forgetting"]["tuning"]["auto_promote_required_improving_runs"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.auto_promote_required_improving_runs",
            base["forgetting"]["tuning"]["auto_promote_required_improving_runs"],
        ),
        base["forgetting"]["tuning"]["auto_promote_required_improving_runs"],
        min_value=1,
        max_value=100,
    )
    normalized["forgetting"]["tuning"]["auto_promote_max_recommendation_drift"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.auto_promote_max_recommendation_drift",
            base["forgetting"]["tuning"]["auto_promote_max_recommendation_drift"],
        ),
        base["forgetting"]["tuning"]["auto_promote_max_recommendation_drift"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["forgetting"]["tuning"]["auto_demote_enabled"] = _coerce_bool(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.auto_demote_enabled",
            base["forgetting"]["tuning"]["auto_demote_enabled"],
        ),
        base["forgetting"]["tuning"]["auto_demote_enabled"],
    )
    normalized["forgetting"]["tuning"]["auto_demote_unhealthy_runs"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "forgetting.tuning.auto_demote_unhealthy_runs",
            base["forgetting"]["tuning"]["auto_demote_unhealthy_runs"],
        ),
        base["forgetting"]["tuning"]["auto_demote_unhealthy_runs"],
        min_value=1,
        max_value=100,
    )

    normalized["graph"]["enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "graph.enabled", base["graph"]["enabled"]),
        base["graph"]["enabled"],
    )
    normalized["graph"]["max_expansion_hops"] = _coerce_int(
        get_memory_policy_value(merged, "graph.max_expansion_hops", base["graph"]["max_expansion_hops"]),
        base["graph"]["max_expansion_hops"],
        min_value=1,
        max_value=4,
    )
    normalized["graph"]["max_expanded_nodes"] = _coerce_int(
        get_memory_policy_value(merged, "graph.max_expanded_nodes", base["graph"]["max_expanded_nodes"]),
        base["graph"]["max_expanded_nodes"],
        min_value=2,
        max_value=128,
    )
    normalized["graph"]["inferred_edge_min_confidence"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "graph.inferred_edge_min_confidence",
            base["graph"]["inferred_edge_min_confidence"],
        ),
        base["graph"]["inferred_edge_min_confidence"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["graph"]["semantic_triples"]["enabled"] = _coerce_bool(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.enabled",
            base["graph"]["semantic_triples"]["enabled"],
        ),
        base["graph"]["semantic_triples"]["enabled"],
    )
    triple_mode = str(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.mode",
            base["graph"]["semantic_triples"]["mode"],
        )
        or ""
    ).strip().lower()
    normalized["graph"]["semantic_triples"]["mode"] = (
        triple_mode if triple_mode in {"shadow", "boost", "inject"} else base["graph"]["semantic_triples"]["mode"]
    )
    normalized["graph"]["semantic_triples"]["min_confidence"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.min_confidence",
            base["graph"]["semantic_triples"]["min_confidence"],
        ),
        base["graph"]["semantic_triples"]["min_confidence"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["graph"]["semantic_triples"]["max_candidates"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.max_candidates",
            base["graph"]["semantic_triples"]["max_candidates"],
        ),
        base["graph"]["semantic_triples"]["max_candidates"],
        min_value=1,
        max_value=64,
    )
    normalized["graph"]["semantic_triples"]["boost_bonus_base"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.boost_bonus_base",
            base["graph"]["semantic_triples"]["boost_bonus_base"],
        ),
        base["graph"]["semantic_triples"]["boost_bonus_base"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["graph"]["semantic_triples"]["boost_bonus_cap"] = _coerce_float(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.boost_bonus_cap",
            base["graph"]["semantic_triples"]["boost_bonus_cap"],
        ),
        base["graph"]["semantic_triples"]["boost_bonus_cap"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["graph"]["semantic_triples"]["max_relation_context_items"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.max_relation_context_items",
            base["graph"]["semantic_triples"]["max_relation_context_items"],
        ),
        base["graph"]["semantic_triples"]["max_relation_context_items"],
        min_value=0,
        max_value=16,
    )
    normalized["graph"]["semantic_triples"].setdefault("tuning", {})
    normalized["graph"]["semantic_triples"]["tuning"]["enabled"] = _coerce_bool(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.enabled",
            base["graph"]["semantic_triples"]["tuning"]["enabled"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["enabled"],
    )
    normalized["graph"]["semantic_triples"]["tuning"]["auto_transition_enabled"] = _coerce_bool(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.auto_transition_enabled",
            base["graph"]["semantic_triples"]["tuning"]["auto_transition_enabled"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["auto_transition_enabled"],
    )
    normalized["graph"]["semantic_triples"]["tuning"]["lookback_days"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.lookback_days",
            base["graph"]["semantic_triples"]["tuning"]["lookback_days"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["lookback_days"],
        min_value=3,
        max_value=180,
    )
    normalized["graph"]["semantic_triples"]["tuning"]["stability_window_days"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.stability_window_days",
            base["graph"]["semantic_triples"]["tuning"]["stability_window_days"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["stability_window_days"],
        min_value=1,
        max_value=60,
    )
    normalized["graph"]["semantic_triples"]["tuning"]["min_sources"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.min_sources",
            base["graph"]["semantic_triples"]["tuning"]["min_sources"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["min_sources"],
        min_value=0,
        max_value=10000,
    )
    normalized["graph"]["semantic_triples"]["tuning"]["min_accepted_triples"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.min_accepted_triples",
            base["graph"]["semantic_triples"]["tuning"]["min_accepted_triples"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["min_accepted_triples"],
        min_value=0,
        max_value=100000,
    )
    normalized["graph"]["semantic_triples"]["tuning"]["required_healthy_evaluations"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.required_healthy_evaluations",
            base["graph"]["semantic_triples"]["tuning"]["required_healthy_evaluations"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["required_healthy_evaluations"],
        min_value=2,
        max_value=30,
    )
    normalized["graph"]["semantic_triples"]["tuning"]["demote_unhealthy_evaluations"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.demote_unhealthy_evaluations",
            base["graph"]["semantic_triples"]["tuning"]["demote_unhealthy_evaluations"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["demote_unhealthy_evaluations"],
        min_value=1,
        max_value=30,
    )
    normalized["graph"]["semantic_triples"]["tuning"]["post_demote_cooldown_evaluations"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.post_demote_cooldown_evaluations",
            base["graph"]["semantic_triples"]["tuning"]["post_demote_cooldown_evaluations"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["post_demote_cooldown_evaluations"],
        min_value=0,
        max_value=30,
    )
    normalized["graph"]["semantic_triples"]["tuning"]["demote_on_version_change"] = _coerce_bool(
        get_memory_policy_value(
            merged,
            "graph.semantic_triples.tuning.demote_on_version_change",
            base["graph"]["semantic_triples"]["tuning"]["demote_on_version_change"],
        ),
        base["graph"]["semantic_triples"]["tuning"]["demote_on_version_change"],
    )

    normalized["compaction"]["enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "compaction.enabled", base["compaction"]["enabled"]),
        base["compaction"]["enabled"],
    )
    normalized["compaction"]["cycle_event_limit"] = _coerce_int(
        get_memory_policy_value(merged, "compaction.cycle_event_limit", base["compaction"]["cycle_event_limit"]),
        base["compaction"]["cycle_event_limit"],
        min_value=4,
        max_value=64,
    )
    normalized["compaction"]["recent_lessons_limit"] = _coerce_int(
        get_memory_policy_value(merged, "compaction.recent_lessons_limit", base["compaction"]["recent_lessons_limit"]),
        base["compaction"]["recent_lessons_limit"],
        min_value=1,
        max_value=16,
    )
    normalized["compaction"]["max_reflections"] = _coerce_int(
        get_memory_policy_value(merged, "compaction.max_reflections", base["compaction"]["max_reflections"]),
        base["compaction"]["max_reflections"],
        min_value=1,
        max_value=16,
    )
    normalized["compaction"]["thesis_chain_enabled"] = _coerce_bool(
        get_memory_policy_value(
            merged,
            "compaction.thesis_chain_enabled",
            base["compaction"]["thesis_chain_enabled"],
        ),
        base["compaction"]["thesis_chain_enabled"],
    )
    normalized["compaction"]["thesis_chain_max_chains_per_cycle"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "compaction.thesis_chain_max_chains_per_cycle",
            base["compaction"]["thesis_chain_max_chains_per_cycle"],
        ),
        base["compaction"]["thesis_chain_max_chains_per_cycle"],
        min_value=1,
        max_value=16,
    )
    normalized["compaction"]["thesis_chain_max_events_per_chain"] = _coerce_int(
        get_memory_policy_value(
            merged,
            "compaction.thesis_chain_max_events_per_chain",
            base["compaction"]["thesis_chain_max_events_per_chain"],
        ),
        base["compaction"]["thesis_chain_max_events_per_chain"],
        min_value=2,
        max_value=24,
    )
    normalized["retrieval"]["context_limit"] = _coerce_int(
        get_memory_policy_value(merged, "retrieval.context_limit", base["retrieval"]["context_limit"]),
        base["retrieval"]["context_limit"],
        min_value=1,
        max_value=128,
    )
    normalized["retrieval"]["vector_search_enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "retrieval.vector_search_enabled", base["retrieval"]["vector_search_enabled"]),
        base["retrieval"]["vector_search_enabled"],
    )
    normalized["retrieval"]["vector_search_limit"] = _coerce_int(
        get_memory_policy_value(merged, "retrieval.vector_search_limit", base["retrieval"]["vector_search_limit"]),
        base["retrieval"]["vector_search_limit"],
        min_value=2,
        max_value=10,
    )
    normalized["retrieval"]["peer_lessons_enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "retrieval.peer_lessons_enabled", base["retrieval"]["peer_lessons_enabled"]),
        base["retrieval"]["peer_lessons_enabled"],
    )

    for key, default in base["retrieval"]["reranking"].items():
        normalized["retrieval"]["reranking"][key] = _coerce_float(
            get_memory_policy_value(merged, f"retrieval.reranking.{key}", default),
            default,
            min_value=-1.0,
            max_value=1.0,
        )
    normalized["retrieval"]["reranking"]["recency_bonus_3d"] = _coerce_float(
        normalized["retrieval"]["reranking"]["recency_bonus_3d"],
        base["retrieval"]["reranking"]["recency_bonus_3d"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["retrieval"]["reranking"]["recency_bonus_14d"] = _coerce_float(
        normalized["retrieval"]["reranking"]["recency_bonus_14d"],
        base["retrieval"]["reranking"]["recency_bonus_14d"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["retrieval"]["reranking"]["recency_bonus_45d"] = _coerce_float(
        normalized["retrieval"]["reranking"]["recency_bonus_45d"],
        base["retrieval"]["reranking"]["recency_bonus_45d"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["retrieval"]["reranking"]["ticker_bonus_base"] = _coerce_float(
        normalized["retrieval"]["reranking"]["ticker_bonus_base"],
        base["retrieval"]["reranking"]["ticker_bonus_base"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["retrieval"]["reranking"]["ticker_bonus_step"] = _coerce_float(
        normalized["retrieval"]["reranking"]["ticker_bonus_step"],
        base["retrieval"]["reranking"]["ticker_bonus_step"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["retrieval"]["reranking"]["ticker_bonus_max"] = _coerce_float(
        normalized["retrieval"]["reranking"]["ticker_bonus_max"],
        base["retrieval"]["reranking"]["ticker_bonus_max"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["retrieval"]["reranking"]["outcome_bonus_max"] = _coerce_float(
        normalized["retrieval"]["reranking"]["outcome_bonus_max"],
        base["retrieval"]["reranking"]["outcome_bonus_max"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["retrieval"]["reranking"]["effective_score_bonus_scale"] = _coerce_float(
        normalized["retrieval"]["reranking"]["effective_score_bonus_scale"],
        base["retrieval"]["reranking"]["effective_score_bonus_scale"],
        min_value=0.0,
        max_value=1.0,
    )
    normalized["retrieval"]["reranking"]["effective_score_bonus_cap"] = _coerce_float(
        normalized["retrieval"]["reranking"]["effective_score_bonus_cap"],
        base["retrieval"]["reranking"]["effective_score_bonus_cap"],
        min_value=0.0,
        max_value=1.0,
    )

    normalized["react_injection"]["enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "react_injection.enabled", base["react_injection"]["enabled"]),
        base["react_injection"]["enabled"],
    )
    for tool_name, default in base["react_injection"]["tools"].items():
        normalized["react_injection"]["tools"][tool_name] = _coerce_bool(
            get_memory_policy_value(merged, f"react_injection.tools.{tool_name}", default),
            default,
        )

    normalized["cleanup"]["enabled"] = _coerce_bool(
        get_memory_policy_value(merged, "cleanup.enabled", base["cleanup"]["enabled"]),
        base["cleanup"]["enabled"],
    )
    normalized["cleanup"]["max_age_days"] = _coerce_int(
        get_memory_policy_value(merged, "cleanup.max_age_days", base["cleanup"]["max_age_days"]),
        base["cleanup"]["max_age_days"],
        min_value=30,
        max_value=3650,
    )
    normalized["cleanup"]["min_score"] = _coerce_float(
        get_memory_policy_value(merged, "cleanup.min_score", base["cleanup"]["min_score"]),
        base["cleanup"]["min_score"],
        min_value=0.0,
        max_value=1.0,
    )

    return normalized


def serialize_memory_policy(policy: dict[str, Any]) -> str:
    return json.dumps(normalize_memory_policy(policy), ensure_ascii=False, separators=(",", ":"))


def apply_memory_policy_to_settings(settings: Any, policy: dict[str, Any]) -> None:
    normalized = normalize_memory_policy(policy, defaults=getattr(settings, "memory_policy", None) or default_memory_policy())
    setattr(settings, "memory_policy", normalized)
    setattr(settings, "context_max_memory_events", int(get_memory_policy_value(normalized, "retrieval.context_limit", getattr(settings, "context_max_memory_events", 32))))
    setattr(settings, "memory_compaction_enabled", bool(get_memory_policy_value(normalized, "compaction.enabled", getattr(settings, "memory_compaction_enabled", True))))
    setattr(settings, "memory_compaction_cycle_event_limit", int(get_memory_policy_value(normalized, "compaction.cycle_event_limit", getattr(settings, "memory_compaction_cycle_event_limit", 12))))
    setattr(settings, "memory_compaction_recent_lessons_limit", int(get_memory_policy_value(normalized, "compaction.recent_lessons_limit", getattr(settings, "memory_compaction_recent_lessons_limit", 4))))
    setattr(settings, "memory_compaction_max_reflections", int(get_memory_policy_value(normalized, "compaction.max_reflections", getattr(settings, "memory_compaction_max_reflections", 3))))


def load_memory_policy(repo: Any, tenant_id: str, *, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    tenant = str(tenant_id or "").strip().lower() or "local"
    getter = getattr(repo, "get_config", None)
    raw = None
    if callable(getter):
        try:
            raw = getter(tenant, MEMORY_POLICY_CONFIG_KEY)
        except Exception as exc:
            logger.warning("[yellow]memory_policy load failed[/yellow] tenant=%s err=%s", tenant, str(exc))
    return normalize_memory_policy(raw, defaults=defaults or default_memory_policy())


def load_global_compaction_prompt(repo: Any) -> str:
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return ""
    try:
        value = getter(GLOBAL_MEMORY_PROMPT_TENANT, GLOBAL_MEMORY_PROMPT_CONFIG_KEY)
    except Exception as exc:
        logger.warning(
            "[yellow]global memory_compactor_prompt load failed[/yellow] tenant=%s err=%s",
            GLOBAL_MEMORY_PROMPT_TENANT,
            str(exc),
        )
        return ""
    return str(value or "").strip()


def load_tenant_compaction_prompt(repo: Any, tenant_id: str) -> str:
    tenant = str(tenant_id or "").strip().lower() or "local"
    getter = getattr(repo, "get_config", None)
    if not callable(getter):
        return ""
    try:
        value = getter(tenant, GLOBAL_MEMORY_PROMPT_CONFIG_KEY)
    except Exception as exc:
        logger.warning(
            "[yellow]tenant memory_compactor_prompt load failed[/yellow] tenant=%s err=%s",
            tenant,
            str(exc),
        )
        return ""
    return str(value or "").strip()


def resolve_compaction_prompt(repo: Any, tenant_id: str, *, policy: dict[str, Any] | None = None) -> str:
    tenant_prompt = load_tenant_compaction_prompt(repo, tenant_id)
    if tenant_prompt:
        return tenant_prompt
    global_prompt = load_global_compaction_prompt(repo)
    if global_prompt:
        return global_prompt
    raise RuntimeError(
        f"Missing required runtime prompt: tenant={GLOBAL_MEMORY_PROMPT_TENANT} key={GLOBAL_MEMORY_PROMPT_CONFIG_KEY}"
    )


def memory_event_enabled(policy: dict[str, Any] | None, event_type: str, default: bool = True) -> bool:
    token = str(event_type or "").strip().lower()
    if not token:
        return default
    return _coerce_bool(get_memory_policy_value(policy, f"event_types.{token}", default), default)


def memory_vector_search_enabled(policy: dict[str, Any] | None, default: bool = True) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "retrieval.vector_search_enabled", default), default)


def memory_vector_search_limit(policy: dict[str, Any] | None, default: int = 5) -> int:
    return _coerce_int(get_memory_policy_value(policy, "retrieval.vector_search_limit", default), default, min_value=2, max_value=10)


def memory_peer_lessons_enabled(policy: dict[str, Any] | None, default: bool = True) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "retrieval.peer_lessons_enabled", default), default)


def memory_embed_cache_max(policy: dict[str, Any] | None, default: int = 128) -> int:
    return _coerce_int(get_memory_policy_value(policy, "storage.embed_cache_max", default), default, min_value=16, max_value=4096)


def memory_hierarchy_enabled(policy: dict[str, Any] | None, default: bool = False) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "hierarchy.enabled", default), default)


def memory_hierarchy_working_ttl_hours(policy: dict[str, Any] | None, default: int = 36) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "hierarchy.working_ttl_hours", default),
        default,
        min_value=1,
        max_value=336,
    )


def memory_hierarchy_episodic_ttl_days(policy: dict[str, Any] | None, default: int = 90) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "hierarchy.episodic_ttl_days", default),
        default,
        min_value=7,
        max_value=3650,
    )


def memory_tagging_enabled(policy: dict[str, Any] | None, default: bool = False) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "tagging.enabled", default), default)


def memory_tagging_max_tags(policy: dict[str, Any] | None, default: int = 6) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "tagging.max_tags", default),
        default,
        min_value=1,
        max_value=32,
    )


def memory_tagging_regime_bonus(policy: dict[str, Any] | None, default: float = 0.25) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "tagging.regime_bonus", default),
        default,
        min_value=0.0,
        max_value=1.0,
    )


def memory_tagging_strategy_bonus(policy: dict[str, Any] | None, default: float = 0.18) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "tagging.strategy_bonus", default),
        default,
        min_value=0.0,
        max_value=1.0,
    )


def memory_tagging_sector_bonus(policy: dict[str, Any] | None, default: float = 0.10) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "tagging.sector_bonus", default),
        default,
        min_value=0.0,
        max_value=1.0,
    )


def memory_forgetting_enabled(policy: dict[str, Any] | None, default: bool = False) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "forgetting.enabled", default), default)


def memory_forgetting_access_log_enabled(policy: dict[str, Any] | None, default: bool = False) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "forgetting.access_log_enabled", default), default)


def memory_forgetting_default_decay_factor(policy: dict[str, Any] | None, default: float = 0.985) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "forgetting.default_decay_factor", default),
        default,
        min_value=0.90,
        max_value=1.0,
    )


def memory_forgetting_access_curve(policy: dict[str, Any] | None, default: str = "sqrt") -> str:
    token = str(get_memory_policy_value(policy, "forgetting.access_curve", default) or "").strip().lower()
    return token if token in {"sqrt", "log", "capped_linear"} else default


def memory_forgetting_tier_weight(policy: dict[str, Any] | None, memory_tier: str, default: float | None = None) -> float:
    tier = str(memory_tier or "").strip().lower()
    fallback_map = {
        "working": 2.0,
        "semantic": 0.35,
        "episodic": 1.0,
    }
    fallback = float(default if default is not None else fallback_map.get(tier, 1.0))
    path = {
        "working": "forgetting.tier_weight_working",
        "semantic": "forgetting.tier_weight_semantic",
        "episodic": "forgetting.tier_weight_episodic",
    }.get(tier, "forgetting.tier_weight_episodic")
    max_value = 2.0 if tier == "semantic" else 4.0
    min_value = 0.05 if tier == "semantic" else 0.1
    return _coerce_float(
        get_memory_policy_value(policy, path, fallback),
        fallback,
        min_value=min_value,
        max_value=max_value,
    )


def memory_forgetting_min_effective_score(policy: dict[str, Any] | None, default: float = 0.15) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "forgetting.min_effective_score", default),
        default,
        min_value=0.0,
        max_value=1.0,
    )


def memory_forgetting_tuning_enabled(policy: dict[str, Any] | None, default: bool = False) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "forgetting.tuning.enabled", default), default)


def memory_forgetting_tuning_mode(policy: dict[str, Any] | None, default: str = "shadow") -> str:
    token = str(get_memory_policy_value(policy, "forgetting.tuning.mode", default) or "").strip().lower()
    return token if token in {"shadow", "bounded_ema"} else default


def memory_forgetting_tuning_lookback_days(policy: dict[str, Any] | None, default: int = 30) -> int:
    return _coerce_int(get_memory_policy_value(policy, "forgetting.tuning.lookback_days", default), default, min_value=7, max_value=180)


def memory_forgetting_tuning_stability_window_days(policy: dict[str, Any] | None, default: int = 7) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.stability_window_days", default),
        default,
        min_value=3,
        max_value=60,
    )


def memory_forgetting_tuning_min_access_events(policy: dict[str, Any] | None, default: int = 500) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.min_access_events", default),
        default,
        min_value=1,
        max_value=50000,
    )


def memory_forgetting_tuning_min_prompt_uses(policy: dict[str, Any] | None, default: int = 100) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.min_prompt_uses", default),
        default,
        min_value=1,
        max_value=50000,
    )


def memory_forgetting_tuning_min_unique_memories(policy: dict[str, Any] | None, default: int = 100) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.min_unique_memories", default),
        default,
        min_value=1,
        max_value=50000,
    )


def memory_forgetting_tuning_ema_alpha(policy: dict[str, Any] | None, default: float = 0.10) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "forgetting.tuning.ema_alpha", default),
        default,
        min_value=0.01,
        max_value=0.5,
    )


def memory_forgetting_tuning_max_decay_factor_delta(policy: dict[str, Any] | None, default: float = 0.003) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "forgetting.tuning.max_decay_factor_delta", default),
        default,
        min_value=0.0005,
        max_value=0.02,
    )


def memory_forgetting_tuning_max_min_effective_score_delta(policy: dict[str, Any] | None, default: float = 0.03) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "forgetting.tuning.max_min_effective_score_delta", default),
        default,
        min_value=0.005,
        max_value=0.20,
    )


def memory_forgetting_tuning_max_tier_weight_delta(policy: dict[str, Any] | None, default: float = 0.10) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "forgetting.tuning.max_tier_weight_delta", default),
        default,
        min_value=0.01,
        max_value=1.0,
    )


def memory_forgetting_tuning_objective_topk(policy: dict[str, Any] | None, default: int = 24) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.objective_topk", default),
        default,
        min_value=4,
        max_value=128,
    )


def memory_forgetting_tuning_auto_promote_enabled(policy: dict[str, Any] | None, default: bool = False) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "forgetting.tuning.auto_promote_enabled", default), default)


def memory_forgetting_tuning_auto_promote_min_shadow_days(policy: dict[str, Any] | None, default: int = 14) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.auto_promote_min_shadow_days", default),
        default,
        min_value=1,
        max_value=90,
    )


def memory_forgetting_tuning_auto_promote_min_shadow_runs(policy: dict[str, Any] | None, default: int = 10) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.auto_promote_min_shadow_runs", default),
        default,
        min_value=1,
        max_value=100,
    )


def memory_forgetting_tuning_auto_promote_required_stable_runs(policy: dict[str, Any] | None, default: int = 5) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.auto_promote_required_stable_runs", default),
        default,
        min_value=1,
        max_value=100,
    )


def memory_forgetting_tuning_auto_promote_required_improving_runs(policy: dict[str, Any] | None, default: int = 3) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.auto_promote_required_improving_runs", default),
        default,
        min_value=1,
        max_value=100,
    )


def memory_forgetting_tuning_auto_promote_max_recommendation_drift(policy: dict[str, Any] | None, default: float = 0.25) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "forgetting.tuning.auto_promote_max_recommendation_drift", default),
        default,
        min_value=0.0,
        max_value=1.0,
    )


def memory_forgetting_tuning_auto_demote_enabled(policy: dict[str, Any] | None, default: bool = False) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "forgetting.tuning.auto_demote_enabled", default), default)


def memory_forgetting_tuning_auto_demote_unhealthy_runs(policy: dict[str, Any] | None, default: int = 3) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "forgetting.tuning.auto_demote_unhealthy_runs", default),
        default,
        min_value=1,
        max_value=100,
    )


def memory_graph_enabled(policy: dict[str, Any] | None, default: bool = False) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "graph.enabled", default), default)


def memory_graph_max_expansion_hops(policy: dict[str, Any] | None, default: int = 1) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.max_expansion_hops", default),
        default,
        min_value=1,
        max_value=4,
    )


def memory_graph_max_expanded_nodes(policy: dict[str, Any] | None, default: int = 12) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.max_expanded_nodes", default),
        default,
        min_value=2,
        max_value=128,
    )


def memory_graph_inferred_edge_min_confidence(policy: dict[str, Any] | None, default: float = 0.75) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "graph.inferred_edge_min_confidence", default),
        default,
        min_value=0.0,
        max_value=1.0,
    )


def memory_graph_semantic_triples_enabled(policy: dict[str, Any] | None, default: bool = True) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "graph.semantic_triples.enabled", default), default)


def memory_graph_semantic_triples_mode(policy: dict[str, Any] | None, default: str = "shadow") -> str:
    mode = str(get_memory_policy_value(policy, "graph.semantic_triples.mode", default) or "").strip().lower()
    return mode if mode in {"shadow", "boost", "inject"} else default


def memory_graph_semantic_triples_boost_enabled(policy: dict[str, Any] | None) -> bool:
    if not memory_graph_semantic_triples_enabled(policy):
        return False
    return memory_graph_semantic_triples_mode(policy) in {"boost", "inject"}


def memory_graph_semantic_triples_inject_enabled(policy: dict[str, Any] | None) -> bool:
    if not memory_graph_semantic_triples_enabled(policy):
        return False
    return memory_graph_semantic_triples_mode(policy) == "inject"


def memory_graph_semantic_triples_min_confidence(policy: dict[str, Any] | None, default: float = 0.75) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "graph.semantic_triples.min_confidence", default),
        default,
        min_value=0.0,
        max_value=1.0,
    )


def memory_graph_semantic_triples_max_candidates(policy: dict[str, Any] | None, default: int = 8) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.semantic_triples.max_candidates", default),
        default,
        min_value=1,
        max_value=64,
    )


def memory_graph_semantic_triples_boost_bonus_base(policy: dict[str, Any] | None, default: float = 0.12) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "graph.semantic_triples.boost_bonus_base", default),
        default,
        min_value=0.0,
        max_value=1.0,
    )


def memory_graph_semantic_triples_boost_bonus_cap(policy: dict[str, Any] | None, default: float = 0.18) -> float:
    return _coerce_float(
        get_memory_policy_value(policy, "graph.semantic_triples.boost_bonus_cap", default),
        default,
        min_value=0.0,
        max_value=1.0,
    )


def memory_graph_semantic_triples_max_relation_context_items(
    policy: dict[str, Any] | None,
    default: int = 4,
) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.semantic_triples.max_relation_context_items", default),
        default,
        min_value=0,
        max_value=16,
    )


def memory_graph_semantic_triples_tuning_enabled(policy: dict[str, Any] | None, default: bool = True) -> bool:
    return _coerce_bool(get_memory_policy_value(policy, "graph.semantic_triples.tuning.enabled", default), default)


def memory_graph_semantic_triples_tuning_auto_transition_enabled(
    policy: dict[str, Any] | None,
    default: bool = True,
) -> bool:
    return _coerce_bool(
        get_memory_policy_value(policy, "graph.semantic_triples.tuning.auto_transition_enabled", default),
        default,
    )


def memory_graph_semantic_triples_tuning_lookback_days(policy: dict[str, Any] | None, default: int = 30) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.semantic_triples.tuning.lookback_days", default),
        default,
        min_value=3,
        max_value=180,
    )


def memory_graph_semantic_triples_tuning_stability_window_days(
    policy: dict[str, Any] | None,
    default: int = 7,
) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.semantic_triples.tuning.stability_window_days", default),
        default,
        min_value=1,
        max_value=60,
    )


def memory_graph_semantic_triples_tuning_min_sources(policy: dict[str, Any] | None, default: int = 0) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.semantic_triples.tuning.min_sources", default),
        default,
        min_value=0,
        max_value=10000,
    )


def memory_graph_semantic_triples_tuning_min_accepted_triples(
    policy: dict[str, Any] | None,
    default: int = 0,
) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.semantic_triples.tuning.min_accepted_triples", default),
        default,
        min_value=0,
        max_value=100000,
    )


def memory_graph_semantic_triples_tuning_required_healthy_evaluations(
    policy: dict[str, Any] | None,
    default: int = 3,
) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.semantic_triples.tuning.required_healthy_evaluations", default),
        default,
        min_value=2,
        max_value=30,
    )


def memory_graph_semantic_triples_tuning_demote_unhealthy_evaluations(
    policy: dict[str, Any] | None,
    default: int = 2,
) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.semantic_triples.tuning.demote_unhealthy_evaluations", default),
        default,
        min_value=1,
        max_value=30,
    )


def memory_graph_semantic_triples_tuning_post_demote_cooldown_evaluations(
    policy: dict[str, Any] | None,
    default: int = 0,
) -> int:
    return _coerce_int(
        get_memory_policy_value(policy, "graph.semantic_triples.tuning.post_demote_cooldown_evaluations", default),
        default,
        min_value=0,
        max_value=30,
    )


def memory_graph_semantic_triples_tuning_demote_on_version_change(
    policy: dict[str, Any] | None,
    default: bool = True,
) -> bool:
    return _coerce_bool(
        get_memory_policy_value(policy, "graph.semantic_triples.tuning.demote_on_version_change", default),
        default,
    )


def memory_react_injection_enabled(policy: dict[str, Any] | None, tool_name: str | None = None, default: bool = True) -> bool:
    enabled = _coerce_bool(get_memory_policy_value(policy, "react_injection.enabled", default), default)
    if not enabled:
        return False
    token = str(tool_name or "").strip()
    if not token:
        return enabled
    tool_default = token in MEMORY_INJECTABLE_TOOLS
    return _coerce_bool(get_memory_policy_value(policy, f"react_injection.tools.{token}", tool_default), tool_default)


def _size_from_count(count: int, *, base: int, step: float, max_size: int) -> int:
    if count <= 0:
        return base
    return min(max_size, int(round(base + (count * step))))


def _format_label(spec: MemoryFieldSpec, value: Any, *, prompt_source: str = "") -> str:
    if spec.value_type == "toggle":
        return f"{spec.label}: {'ON' if bool(value) else 'OFF'}"
    if spec.value_type == "prompt":
        return spec.label
    if spec.value_type == "select":
        return f"{spec.label}: {value}"
    if spec.value_type == "float":
        return f"{spec.label}: {float(value):.2f}"
    return f"{spec.label}: {value}"


def _child_positions(
    parent: tuple[float, float, float],
    *,
    count: int,
    radius: float,
    y_offset: float,
) -> list[tuple[float, float, float]]:
    if count <= 0:
        return []
    out: list[tuple[float, float, float]] = []
    for idx in range(count):
        angle = (2.0 * math.pi * idx) / count
        out.append(
            (
                parent[0] + (math.cos(angle) * radius),
                parent[1] + y_offset + (math.sin(angle * 0.5) * (radius * 0.1)),
                parent[2] + (math.sin(angle) * radius),
            )
        )
    return out


def build_memory_graph(
    policy: dict[str, Any],
    *,
    tenant_id: str,
    stats: dict[str, Any] | None = None,
    tenant_compaction_prompt: str = "",
    global_compaction_prompt: str = "",
    effective_compaction_prompt: str = "",
    prompt_source: str = "global",
    compaction_prompt_editable: bool = True,
) -> dict[str, Any]:
    normalized = normalize_memory_policy(policy)
    counts = {
        str(k): int(v or 0)
        for k, v in ((stats or {}).get("counts_by_event_type") or {}).items()
    }
    total_count = int((stats or {}).get("total_events") or sum(counts.values()) or 0)
    group_colors = {str(group["id"]): str(group["color"]) for group in GROUP_SPECS}

    top_level_positions: dict[str, tuple[float, float, float]] = {}
    group_count = max(len(GROUP_SPECS), 1)
    orbit = 140.0
    for idx, group in enumerate(GROUP_SPECS):
        angle = (2.0 * math.pi * idx) / group_count
        top_level_positions[str(group["id"])] = (
            math.cos(angle) * orbit,
            math.sin(angle * 1.7) * 18.0,
            math.sin(angle) * orbit,
        )

    branch_specs_by_parent: dict[str, list[dict[str, str]]] = {}
    for branch in BRANCH_SPECS:
        branch_specs_by_parent.setdefault(str(branch["parent"]), []).append(branch)

    nodes: list[dict[str, Any]] = [
        {
            "id": "root",
            "label": "Memory System",
            "group": "root",
            "size": _size_from_count(total_count, base=30, step=0.04, max_size=42),
            "editable": False,
            "kind": "root",
            "description": "Central hub for tenant memory policy, storage, retrieval, and cleanup.",
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "fx": 0.0,
            "fy": 0.0,
            "fz": 0.0,
        }
    ]
    links: list[dict[str, str]] = []
    branch_positions: dict[str, tuple[float, float, float]] = {}

    for group in GROUP_SPECS:
        group_id = group["id"]
        group_total = 0
        if group_id == "event_types":
            group_total = sum(counts.values())
        group_pos = top_level_positions[str(group_id)]
        nodes.append(
            {
                "id": group_id,
                "label": group["label"],
                "group": group_id,
                "color": group["color"],
                "size": _size_from_count(group_total, base=18, step=0.03, max_size=28),
                "editable": False,
                "kind": "group",
                "description": group.get("description") or "",
                "count": group_total,
                "x": group_pos[0],
                "y": group_pos[1],
                "z": group_pos[2],
                "fx": group_pos[0],
                "fy": group_pos[1],
                "fz": group_pos[2],
            }
        )
        links.append({"source": "root", "target": group_id})

        child_branches = branch_specs_by_parent.get(str(group_id), [])
        branch_positions_list = _child_positions(group_pos, count=len(child_branches), radius=52.0, y_offset=-16.0)
        for branch, branch_pos in zip(child_branches, branch_positions_list):
            branch_id = str(branch["id"])
            branch_positions[branch_id] = branch_pos
            nodes.append(
                {
                    "id": branch_id,
                    "label": branch["label"],
                    "group": branch["group"],
                    "color": group["color"],
                    "size": 14,
                    "editable": False,
                    "kind": "branch",
                    "description": branch.get("description") or "",
                    "x": branch_pos[0],
                    "y": branch_pos[1],
                    "z": branch_pos[2],
                }
            )
            links.append({"source": group_id, "target": branch_id})

    # Handle nested branches (branches whose parent is another branch, not a group)
    for branch in BRANCH_SPECS:
        branch_id = str(branch["id"])
        if branch_id in branch_positions:
            continue
        parent_id = str(branch["parent"])
        parent_pos = branch_positions.get(parent_id)
        if not parent_pos:
            continue
        nested_pos = _child_positions(parent_pos, count=1, radius=42.0, y_offset=-16.0)[0]
        branch_positions[branch_id] = nested_pos
        nodes.append(
            {
                "id": branch_id,
                "label": branch["label"],
                "group": branch["group"],
                "color": group_colors.get(branch["group"], "#94a3b8"),
                "size": 12,
                "editable": False,
                "kind": "branch",
                "description": branch.get("description") or "",
                "x": nested_pos[0],
                "y": nested_pos[1],
                "z": nested_pos[2],
            }
        )
        links.append({"source": parent_id, "target": branch_id})

    field_specs_by_parent: dict[str, list[MemoryFieldSpec]] = {}
    for spec in FIELD_SPECS:
        field_parent = FIELD_PARENT_OVERRIDES.get(spec.path, spec.group)
        field_specs_by_parent.setdefault(field_parent, []).append(spec)

    for parent_id, specs in field_specs_by_parent.items():
        parent_position = branch_positions.get(parent_id) or top_level_positions.get(parent_id) or (0.0, 0.0, 0.0)
        leaf_positions = _child_positions(parent_position, count=len(specs), radius=42.0, y_offset=-20.0)
        for spec, leaf_pos in zip(specs, leaf_positions):
            value = get_memory_policy_value(normalized, spec.path)
            count = 0
            if spec.path.startswith("event_types."):
                event_type = spec.path.split(".", 1)[1]
                count = counts.get(event_type, 0)
            nodes.append(
                {
                    "id": spec.path,
                    "label": _format_label(spec, value, prompt_source=prompt_source),
                    "group": spec.group,
                    "color": group_colors.get(spec.group, "#94a3b8"),
                    "size": _size_from_count(count, base=10 if spec.value_type != "prompt" else 12, step=0.02, max_size=18),
                    "editable": True,
                    "kind": "leaf",
                    "path": spec.path,
                    "value": value,
                    "type": spec.value_type,
                    "options": list(spec.options),
                    "min": spec.min_value,
                    "max": spec.max_value,
                    "step": spec.step,
                    "description": spec.description,
                    "scope": spec.scope,
                    "count": count,
                    "parent": parent_id,
                    "x": leaf_pos[0],
                    "y": leaf_pos[1],
                    "z": leaf_pos[2],
                }
            )
            links.append({"source": parent_id, "target": spec.path})

    prompt_parent = "compaction"
    prompt_position = branch_positions.get(prompt_parent) or top_level_positions.get(prompt_parent) or (0.0, 0.0, 0.0)
    prompt_leaf_position = _child_positions(prompt_position, count=1, radius=50.0, y_offset=-24.0)[0]
    nodes.append(
        {
            "id": "compaction.global_prompt",
            "label": "Compaction Prompt",
            "group": "compaction",
            "color": "#a78bfa",
            "size": 12,
            "editable": bool(compaction_prompt_editable),
            "kind": "leaf",
            "path": "compaction.global_prompt",
            "value": str(effective_compaction_prompt or tenant_compaction_prompt or global_compaction_prompt or ""),
            "type": "prompt",
            "description": "Tenant compaction prompt template. If empty, the global default is inherited.",
            "scope": "tenant",
            "tenant_value": str(tenant_compaction_prompt or ""),
            "effective_value": str(effective_compaction_prompt or ""),
            "prompt_source": str(prompt_source or "global"),
            "parent": prompt_parent,
            "x": prompt_leaf_position[0],
            "y": prompt_leaf_position[1],
            "z": prompt_leaf_position[2],
        }
    )
    links.append({"source": prompt_parent, "target": "compaction.global_prompt"})

    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "tenant_id": str(tenant_id or "").strip().lower() or "local",
            "policy": normalized,
            "tenant_compaction_prompt": str(tenant_compaction_prompt or ""),
            "global_compaction_prompt": str(global_compaction_prompt or ""),
            "effective_compaction_prompt": str(effective_compaction_prompt or ""),
            "prompt_source": str(prompt_source or "global"),
            "compaction_prompt_editable": bool(compaction_prompt_editable),
        },
    }
