from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import math
import re
from typing import Any

from arena.data.bq import BigQueryRepository
from arena.memory.candidates import CANDIDATE_MEMORY_EVENT_TYPES, candidate_memory_records
from arena.memory.graph import ensure_memory_event_graph_ids, infer_memory_event_causal_chain_id, memory_event_node_id
from arena.models import ExecutionReport, MemoryEvent, OrderIntent, RiskDecision, utc_now
from arena.memory.policy import (
    memory_embed_cache_max,
    memory_event_enabled,
    memory_hierarchy_enabled,
    memory_hierarchy_episodic_ttl_days,
    memory_hierarchy_working_ttl_hours,
    memory_tagging_enabled,
    memory_tagging_max_tags,
    normalize_memory_policy,
)
from arena.memory.thesis import (
    build_thesis_id,
    build_thesis_payload,
    is_material_thesis_update,
    is_thesis_broken,
    thesis_event_summary,
)
from arena.memory.tags import extract_context_tags
from arena.memory.vector import VectorStore

logger = logging.getLogger(__name__)


class MemoryStore:
    """Provides long-memory write and retrieval helpers with dynamic score feedback."""

    _MANUAL_NOTE_DUPLICATE_LOOKBACK_DAYS = 14
    _MANUAL_NOTE_DUPLICATE_FETCH_LIMIT = 64

    def __init__(
        self,
        repo: BigQueryRepository,
        vector_store: VectorStore | None = None,
        trading_mode: str = "paper",
        memory_policy: dict[str, Any] | None = None,
    ):
        self.repo = repo
        self.trading_mode = trading_mode
        self.memory_policy = normalize_memory_policy(memory_policy or {})
        self.vector_store = vector_store or VectorStore(
            project=repo.project,
            location=repo.location,
            embed_cache_max=memory_embed_cache_max(self.memory_policy),
        )

    def _tenant(self) -> str:
        resolver = getattr(self.repo, "resolve_tenant_id", None)
        if callable(resolver):
            return str(resolver())
        raw = str(getattr(self.repo, "tenant_id", "") or "").strip().lower()
        return raw or "local"

    @staticmethod
    def _safe_event_count(payload: dict[str, Any] | None) -> int:
        tool_events = payload.get("tool_events") if isinstance(payload, dict) else None
        if isinstance(tool_events, list):
            return len(tool_events)
        return 0

    @staticmethod
    def _memory_source(payload: dict[str, Any] | None) -> str:
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("source") or "").strip()

    def _memory_tier(self, *, event_type: str, payload: dict[str, Any] | None) -> str | None:
        """Assigns a temporal tier when hierarchy mode is enabled."""
        if not memory_hierarchy_enabled(self.memory_policy):
            return None
        if isinstance(payload, dict):
            explicit = str(payload.get("memory_tier") or "").strip().lower()
            if explicit in {"working", "episodic", "semantic"}:
                return explicit
        token = str(event_type or "").strip().lower()
        if token == "strategy_reflection":
            return "semantic"
        if token == "react_tools_summary":
            return "working"
        return "episodic"

    def _memory_expiry(self, *, created_at: datetime, memory_tier: str | None) -> datetime | None:
        """Computes expiry for non-semantic tiers when hierarchy mode is enabled."""
        if not memory_hierarchy_enabled(self.memory_policy):
            return None
        tier = str(memory_tier or "").strip().lower()
        if tier == "working":
            return created_at + timedelta(hours=memory_hierarchy_working_ttl_hours(self.memory_policy))
        if tier == "episodic":
            return created_at + timedelta(days=memory_hierarchy_episodic_ttl_days(self.memory_policy))
        return None

    def _context_tags(
        self,
        *,
        event_type: str,
        summary: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not memory_tagging_enabled(self.memory_policy):
            return {}
        return extract_context_tags(
            event_type=event_type,
            summary=summary,
            payload=payload,
            max_tags=memory_tagging_max_tags(self.memory_policy),
        )

    @staticmethod
    def _normalize_summary_key(summary: str) -> str:
        """Normalizes free-form text for duplicate detection."""
        tokens = re.findall(r"[A-Za-z0-9$]+(?:[._:/-][A-Za-z0-9$]+)*", str(summary or "").lower())
        return " ".join(token.strip() for token in tokens if token.strip())

    @staticmethod
    def _manual_note_has_signal(summary: str) -> bool:
        """Accepts short notes when they still carry clear retrieval signal."""
        text = str(summary or "").strip()
        if not text:
            return False
        tokens = re.findall(r"[A-Za-z0-9$._:/-]+", text)
        if len(tokens) >= 2:
            return True
        for token in tokens:
            plain = token.strip("$")
            if not plain:
                continue
            if any(ch.isdigit() for ch in plain):
                return True
            if plain.isalpha() and plain.upper() == plain and len(plain) <= 5:
                return True
            if len(plain) >= 8:
                return True
        return False

    def _is_duplicate_manual_note(
        self,
        *,
        agent_id: str,
        summary: str,
        current_event_id: str = "",
    ) -> bool:
        """Skips indexing for recent notes that normalize to the same text."""
        normalized = self._normalize_summary_key(summary)
        if not normalized:
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._MANUAL_NOTE_DUPLICATE_LOOKBACK_DAYS)
        try:
            recent_rows = self.repo.recent_memory_events(
                agent_id=agent_id,
                limit=self._MANUAL_NOTE_DUPLICATE_FETCH_LIMIT,
                trading_mode=self.trading_mode,
            )
        except Exception:
            recent_rows = []
        for row in recent_rows:
            if str(row.get("event_id") or "").strip() == current_event_id:
                continue
            if str(row.get("event_type") or "").strip().lower() != "manual_note":
                continue
            created_raw = row.get("created_at")
            if isinstance(created_raw, datetime):
                created_at = created_raw
            else:
                try:
                    created_at = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
                except Exception:
                    created_at = None
            if created_at is None:
                continue
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if created_at < cutoff:
                continue
            if self._normalize_summary_key(str(row.get("summary") or "")) == normalized:
                return True
        return False

    def _should_index_memory_event(
        self,
        *,
        agent_id: str,
        event_id: str = "",
        event_type: str,
        summary: str,
        payload: dict[str, Any] | None,
    ) -> bool:
        """Keeps BQ as the full log while only indexing memories with search value."""
        text = str(summary or "").strip()
        if not text:
            return False

        kind = str(event_type or "").strip().lower()
        if not memory_event_enabled(self.memory_policy, kind, True):
            return False
        data = payload if isinstance(payload, dict) else {}
        if kind == "strategy_reflection":
            return True
        if kind == "manual_note":
            return self._manual_note_has_signal(text) and not self._is_duplicate_manual_note(
                agent_id=agent_id,
                summary=text,
                current_event_id=event_id,
            )
        if kind == "trade_execution":
            intent = data.get("intent")
            report = data.get("report")
            ticker = str(intent.get("ticker") or "").strip().upper() if isinstance(intent, dict) else ""
            status = str(report.get("status") or "").strip().upper() if isinstance(report, dict) else ""
            return bool(ticker and status in {"FILLED", "SIMULATED"})
        if kind in CANDIDATE_MEMORY_EVENT_TYPES:
            ticker = str(data.get("ticker") or "").strip().upper()
            source = str(data.get("source") or "").strip().lower()
            return bool(ticker and source == "candidate_discovery")
        if kind in {"thesis_invalidated", "thesis_realized"}:
            thesis_id = str(data.get("thesis_id") or "").strip()
            ticker = str(data.get("ticker") or "").strip().upper()
            return bool(thesis_id and ticker)
        return False

    @staticmethod
    def _payload_from_row(row: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(row, dict):
            return {}
        payload_raw = row.get("payload_json")
        if isinstance(payload_raw, dict):
            return dict(payload_raw)
        if isinstance(payload_raw, str) and payload_raw.strip():
            try:
                parsed = json.loads(payload_raw)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _active_thesis_row(self, *, agent_id: str, ticker: str) -> dict[str, Any] | None:
        loader = getattr(self.repo, "active_thesis_events", None)
        if not callable(loader):
            return None
        try:
            rows = list(
                loader(
                    agent_id=agent_id,
                    tickers=[ticker],
                    trading_mode=self.trading_mode,
                )
            )
        except Exception:
            return None
        return rows[0] if rows else None

    @staticmethod
    def _snapshot_position_quantity(snapshot: Any, ticker: str) -> float:
        if snapshot is None:
            return 0.0
        positions = getattr(snapshot, "positions", None)
        if not isinstance(positions, dict):
            positions = snapshot.get("positions") if isinstance(snapshot, dict) else None
        if not isinstance(positions, dict):
            return 0.0
        position = positions.get(str(ticker or "").strip().upper())
        if position is None:
            position = positions.get(str(ticker or "").strip())
        if position is None:
            return 0.0
        if hasattr(position, "quantity"):
            try:
                return float(position.quantity or 0.0)
            except (TypeError, ValueError):
                return 0.0
        if isinstance(position, dict):
            try:
                return float(position.get("quantity") or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def record_thesis_lifecycle(
        self,
        *,
        intent: OrderIntent,
        decision: RiskDecision,
        report: ExecutionReport,
        snapshot_before: Any | None = None,
    ) -> None:
        """Tracks the lifecycle of an investment thesis alongside execution memories."""
        _ = decision
        status = str(report.status.value or "").strip().upper()
        if status not in {"FILLED", "SIMULATED"}:
            return
        ticker = str(intent.ticker or "").strip().upper()
        if not ticker:
            return

        active_row = self._active_thesis_row(agent_id=intent.agent_id, ticker=ticker)
        previous_payload = self._payload_from_row(active_row)
        previous_thesis_id = str(
            (active_row or {}).get("semantic_key")
            or previous_payload.get("thesis_id")
            or ""
        ).strip()

        quantity_before = self._snapshot_position_quantity(snapshot_before, ticker)
        quantity_after = max(quantity_before, 0.0)
        try:
            quantity_after = max(quantity_before - float(report.filled_qty or 0.0), 0.0)
        except (TypeError, ValueError):
            quantity_after = max(quantity_before, 0.0)

        event_type = ""
        thesis_id = previous_thesis_id
        position_action = ""

        if intent.side.value == "BUY":
            if thesis_id:
                if not is_material_thesis_update(
                    previous_payload,
                    rationale=intent.rationale,
                    strategy_refs=intent.strategy_refs,
                ):
                    return
                event_type = "thesis_update"
                position_action = "add"
            else:
                event_type = "thesis_open"
                thesis_id = build_thesis_id(
                    agent_id=intent.agent_id,
                    ticker=ticker,
                    trading_mode=self.trading_mode,
                    intent_id=intent.intent_id,
                    created_at=report.created_at,
                )
                position_action = "entry"
            quantity_after = max(quantity_before + float(report.filled_qty or 0.0), 0.0)
        else:
            if not thesis_id:
                return
            if is_thesis_broken(intent.strategy_refs):
                event_type = "thesis_invalidated"
                position_action = "exit" if quantity_after <= 1e-9 else "trim"
            elif quantity_before > 0 and quantity_after <= 1e-9:
                event_type = "thesis_realized"
                position_action = "exit"
            else:
                if not is_material_thesis_update(
                    previous_payload,
                    rationale=intent.rationale,
                    strategy_refs=intent.strategy_refs,
                ):
                    return
                event_type = "thesis_update"
                position_action = "trim"

        payload = build_thesis_payload(
            event_type=event_type,
            thesis_id=thesis_id,
            intent=intent,
            decision=decision,
            report=report,
            previous_payload=previous_payload,
            position_action=position_action,
            position_qty_before=quantity_before,
            position_qty_after=quantity_after,
        )
        summary = thesis_event_summary(
            event_type=event_type,
            payload=payload,
            report=report,
        )
        score_map = {
            "thesis_open": 0.58,
            "thesis_update": 0.62,
            "thesis_invalidated": 0.78,
            "thesis_realized": 0.74,
        }
        self.record_memory(
            agent_id=intent.agent_id,
            summary=summary,
            event_type=event_type,
            score=score_map.get(event_type, 0.6),
            payload=payload,
            semantic_key=thesis_id,
        )
        logger.info(
            "[cyan]Thesis memory[/cyan] agent=%s ticker=%s event=%s thesis_id=%s action=%s qty_before=%.4f qty_after=%.4f",
            intent.agent_id,
            ticker,
            event_type,
            thesis_id,
            position_action or "-",
            float(quantity_before or 0.0),
            float(quantity_after or 0.0),
        )

    @staticmethod
    def _execution_importance_score(report: ExecutionReport) -> float:
        filled = report.status.value in {"FILLED", "SIMULATED"}
        return 0.75 if filled else 0.35

    @staticmethod
    def _execution_outcome_score(report: ExecutionReport) -> float:
        status = report.status.value
        if status in {"FILLED", "SIMULATED"}:
            return 0.5
        if status == "SUBMITTED":
            return 0.4
        return 0.25

    @staticmethod
    def _execution_summary(
        *,
        intent: OrderIntent,
        status: str,
        policy_reason: str,
        broker_reason: str,
    ) -> str:
        broker = str(broker_reason or "").strip() or "-"
        if len(broker) > 120:
            broker = broker[:117] + "..."
        base = (
            f"{intent.ticker} {intent.side.value} qty={intent.quantity:.4f} "
            f"status={status} policy={policy_reason} broker={broker}"
        )
        rationale = str(getattr(intent, "rationale", "") or "").strip()
        if rationale:
            base += f" rationale={rationale[:120]}"
        return base

    @staticmethod
    def _execution_payload(
        *,
        intent: OrderIntent,
        decision: RiskDecision,
        report: ExecutionReport,
    ) -> dict:
        return {
            "intent": intent.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json"),
            "report": report.model_dump(mode="json"),
        }

    def _upsert_trade_execution_memory(
        self,
        *,
        intent: OrderIntent,
        summary: str,
        payload: dict,
        importance_score: float,
        outcome_score: float | None,
    ) -> None:
        if not memory_event_enabled(self.memory_policy, "trade_execution", True):
            return
        tenant = self._tenant()
        order_id = str(payload.get("report", {}).get("order_id") or "").strip()

        existing_event_id = ""
        existing_created_at: datetime | None = None

        find_fn = getattr(self.repo, "find_trade_execution_memory_event", None)
        if callable(find_fn) and order_id:
            try:
                existing = find_fn(
                    agent_id=intent.agent_id,
                    order_id=order_id,
                    trading_mode=self.trading_mode,
                )
            except Exception:
                existing = None
            if isinstance(existing, dict):
                existing_event_id = str(existing.get("event_id") or "").strip()
                created_raw = existing.get("created_at")
                if isinstance(created_raw, datetime):
                    existing_created_at = created_raw
                elif created_raw:
                    try:
                        existing_created_at = datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
                    except Exception:
                        existing_created_at = None

        if existing_event_id:
            update_fn = getattr(self.repo, "update_memory_event", None)
            if callable(update_fn):
                created_at = existing_created_at or datetime.now(timezone.utc)
                memory_tier = self._memory_tier(event_type="trade_execution", payload=payload)
                context_tags = self._context_tags(event_type="trade_execution", summary=summary, payload=payload)
                graph_node_id = memory_event_node_id(existing_event_id)
                causal_chain_id = infer_memory_event_causal_chain_id(
                    {
                        "event_id": existing_event_id,
                        "agent_id": intent.agent_id,
                        "event_type": "trade_execution",
                        "summary": summary,
                        "trading_mode": self.trading_mode,
                        "payload_json": payload,
                    }
                )
                update_fn(
                    event_id=existing_event_id,
                    summary=summary,
                    payload=payload,
                    importance_score=importance_score,
                    outcome_score=outcome_score,
                    score=importance_score,
                    memory_tier=memory_tier,
                    expires_at=self._memory_expiry(created_at=created_at, memory_tier=memory_tier),
                    context_tags=context_tags,
                    primary_regime=(context_tags.get("regimes") or [None])[0],
                    primary_strategy_tag=(context_tags.get("strategies") or [None])[0],
                    primary_sector=(context_tags.get("sectors") or [None])[0],
                    graph_node_id=graph_node_id,
                    causal_chain_id=causal_chain_id,
                )
                if self._should_index_memory_event(
                    agent_id=intent.agent_id,
                    event_type="trade_execution",
                    summary=summary,
                    payload=payload,
                ):
                    self.vector_store.save_memory_vector(
                        event_id=existing_event_id,
                        agent_id=intent.agent_id,
                        summary=summary,
                        score=importance_score,
                        importance_score=importance_score,
                        outcome_score=outcome_score,
                        trading_mode=self.trading_mode,
                        created_at=existing_created_at,
                        tenant_id=tenant,
                        event_type="trade_execution",
                        memory_source=self._memory_source(payload),
                        memory_tier=memory_tier or "",
                        primary_regime=str((context_tags.get("regimes") or [""])[0] or ""),
                        primary_strategy_tag=str((context_tags.get("strategies") or [""])[0] or ""),
                        primary_sector=str((context_tags.get("sectors") or [""])[0] or ""),
                        context_tags=context_tags or None,
                        graph_node_id=graph_node_id,
                        causal_chain_id=causal_chain_id,
                    )
                return

        created_at = datetime.now(timezone.utc)
        memory_tier = self._memory_tier(event_type="trade_execution", payload=payload)
        context_tags = self._context_tags(event_type="trade_execution", summary=summary, payload=payload)
        event = MemoryEvent(
            agent_id=intent.agent_id,
            event_type="trade_execution",
            summary=summary,
            payload=payload,
            trading_mode=self.trading_mode,
            importance_score=importance_score,
            outcome_score=outcome_score,
            score=importance_score,
            created_at=created_at,
            memory_tier=memory_tier,
            expires_at=self._memory_expiry(created_at=created_at, memory_tier=memory_tier),
            context_tags=context_tags,
            primary_regime=(context_tags.get("regimes") or [None])[0],
            primary_strategy_tag=(context_tags.get("strategies") or [None])[0],
            primary_sector=(context_tags.get("sectors") or [None])[0],
        )
        ensure_memory_event_graph_ids(event)
        self.repo.write_memory_event(event)
        if self._should_index_memory_event(
            agent_id=event.agent_id,
            event_type=event.event_type,
            summary=event.summary,
            payload=payload,
        ):
            self.vector_store.save_memory_vector(
                event_id=event.event_id,
                agent_id=event.agent_id,
                summary=event.summary,
                score=event.score,
                importance_score=event.importance_score,
                outcome_score=event.outcome_score,
                trading_mode=self.trading_mode,
                created_at=event.created_at,
                tenant_id=tenant,
                event_type="trade_execution",
                memory_source=self._memory_source(payload),
                memory_tier=event.memory_tier or "",
                primary_regime=event.primary_regime or "",
                primary_strategy_tag=event.primary_strategy_tag or "",
                primary_sector=event.primary_sector or "",
                context_tags=event.context_tags or None,
                graph_node_id=event.graph_node_id or "",
                causal_chain_id=event.causal_chain_id or "",
            )

    def record_execution(
        self,
        intent: OrderIntent,
        decision: RiskDecision,
        report: ExecutionReport,
    ) -> None:
        """Stores execution memory and triggers score feedback on SELL."""
        policy_reason = str(decision.reason or "").strip() or "-"
        summary = self._execution_summary(
            intent=intent,
            status=report.status.value,
            policy_reason=policy_reason,
            broker_reason=str(report.message or ""),
        )
        payload = self._execution_payload(intent=intent, decision=decision, report=report)
        importance_score = self._execution_importance_score(report)
        outcome_score = self._execution_outcome_score(report)
        self._upsert_trade_execution_memory(
            intent=intent,
            summary=summary,
            payload=payload,
            importance_score=importance_score,
            outcome_score=outcome_score,
        )

        filled = report.status.value in {"FILLED", "SIMULATED"}
        if filled and intent.side.value == "SELL":
            self._feedback_buy_score(intent, report)

    def _feedback_buy_score(self, sell_intent: OrderIntent, sell_report: ExecutionReport) -> None:
        """SELL 체결 시 해당 종목의 과거 BUY 기억 score를 실현 수익률로 갱신."""
        try:
            sell_price = float(sell_report.avg_price_krw or sell_intent.price_krw)
            buy_memories = getattr(self.repo, "find_buy_memories_for_ticker", lambda *args, **kwargs: [])(
                agent_id=sell_intent.agent_id,
                ticker=sell_intent.ticker,
                limit=3,
                trading_mode=self.trading_mode,
            )
            for mem in buy_memories:
                buy_price = self._extract_buy_price(mem)
                if buy_price <= 0:
                    continue

                pnl_ratio = (sell_price - buy_price) / buy_price
                # tanh 기반 비선형 변환: 수익 구간 변별력 확보 (+10% ≠ +50%)
                new_score = max(0.1, min(0.5 + 0.5 * math.tanh(pnl_ratio * 3), 1.0))
                event_id = str(mem.get("event_id", "")).strip()
                if event_id:
                    update_fn = getattr(self.repo, "update_memory_score", None)
                    if callable(update_fn):
                        update_fn(event_id, new_score)
                        
                        # Firestore에 저장된 벡터 쪽 score도 갱신 (선택적)
                        # 여기서는 outcome_score 필드만 업데이트
                        if self.vector_store and self.vector_store.db:
                            try:
                                doc_ref = self.vector_store.db.collection("agent_memories").document(event_id)
                                doc_ref.update({"outcome_score": float(new_score)})
                            except Exception as fs_exc:
                                logger.warning("[yellow]Firestore score sync failed[/yellow] event=%s err=%s", event_id[:8], str(fs_exc))

                        logger.info(
                            "[cyan]Memory outcome updated[/cyan] event=%s ticker=%s pnl=%.2f%% outcome=%.2f→%.2f",
                            event_id[:8], sell_intent.ticker, pnl_ratio * 100,
                            float(mem.get("outcome_score") or mem.get("score") or 0.5), new_score,
                        )
        except Exception as exc:
            logger.warning("[yellow]Buy score feedback failed[/yellow] ticker=%s err=%s", sell_intent.ticker, str(exc))

    @staticmethod
    def _extract_buy_price(memory_row: dict) -> float:
        """Extracts buy price from a memory event's payload."""
        try:
            raw = memory_row.get("payload_json", "")
            if isinstance(raw, str) and raw.strip():
                payload = json.loads(raw)
            elif isinstance(raw, dict):
                payload = raw
            else:
                return 0.0
            return float(payload.get("intent", {}).get("price_krw") or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0

    def record_memory(
        self,
        agent_id: str,
        summary: str,
        event_type: str = "manual_note",
        score: float = 0.5,
        payload: dict | None = None,
        semantic_key: str | None = None,
        memory_tier: str | None = None,
        expires_at: datetime | None = None,
    ) -> str | None:
        """Stores a memory event."""
        if not memory_event_enabled(self.memory_policy, event_type, True):
            return None
        context_tags = self._context_tags(event_type=event_type, summary=summary, payload=payload)
        tier = str(memory_tier or "").strip().lower() or self._memory_tier(event_type=event_type, payload=payload)
        event = MemoryEvent(
            agent_id=agent_id,
            event_type=event_type,
            summary=summary[:600],
            payload=payload or {},
            trading_mode=self.trading_mode,
            importance_score=max(0.0, min(float(score), 1.0)),
            outcome_score=None,
            score=max(0.0, min(float(score), 1.0)),
            memory_tier=tier,
            semantic_key=str(semantic_key or "").strip() or None,
            context_tags=context_tags,
            primary_regime=(context_tags.get("regimes") or [None])[0],
            primary_strategy_tag=(context_tags.get("strategies") or [None])[0],
            primary_sector=(context_tags.get("sectors") or [None])[0],
        )
        event.expires_at = expires_at or self._memory_expiry(created_at=event.created_at, memory_tier=event.memory_tier)
        ensure_memory_event_graph_ids(event)
        self.repo.write_memory_event(event)
        if self._should_index_memory_event(
            agent_id=event.agent_id,
            event_id=event.event_id,
            event_type=event.event_type,
            summary=event.summary,
            payload=event.payload,
        ):
            tenant = self._tenant()
            self.vector_store.save_memory_vector(
                event_id=event.event_id,
                agent_id=event.agent_id,
                summary=event.summary,
                score=event.score,
                importance_score=event.importance_score,
                outcome_score=event.outcome_score,
                trading_mode=self.trading_mode,
                created_at=event.created_at,
                tenant_id=tenant,
                event_type=event_type,
                memory_source=self._memory_source(event.payload),
                memory_tier=event.memory_tier or "",
                primary_regime=event.primary_regime or "",
                primary_strategy_tag=event.primary_strategy_tag or "",
                primary_sector=event.primary_sector or "",
                context_tags=event.context_tags or None,
                graph_node_id=event.graph_node_id or "",
                causal_chain_id=event.causal_chain_id or "",
            )
        return event.event_id

    def _candidate_memory_exists(self, *, agent_id: str, semantic_key: str) -> bool:
        key = str(semantic_key or "").strip()
        if not key:
            return False
        loader = getattr(self.repo, "memory_events_by_semantic_keys", None)
        if callable(loader):
            try:
                rows = loader(
                    agent_id=agent_id,
                    semantic_keys=[key],
                    event_types=list(CANDIDATE_MEMORY_EVENT_TYPES),
                    trading_mode=self.trading_mode,
                )
                return bool(rows)
            except Exception:
                pass
        exists = getattr(self.repo, "memory_event_exists_by_semantic_key", None)
        if callable(exists):
            for event_type in CANDIDATE_MEMORY_EVENT_TYPES:
                try:
                    if exists(
                        agent_id=agent_id,
                        event_type=event_type,
                        semantic_key=key,
                        trading_mode=self.trading_mode,
                    ):
                        return True
                except Exception:
                    continue
        return False

    def record_candidate_memories(
        self,
        *,
        agent_id: str,
        candidate_ledger: dict[str, dict[str, Any]],
        held_tickers: set[str] | None = None,
        cycle_id: str = "",
        phase: str = "",
        limit: int = 5,
    ) -> int:
        """Persists bounded non-held screening candidates as short-lived memory."""
        records = candidate_memory_records(
            candidate_ledger,
            held_tickers=held_tickers or set(),
            agent_id=agent_id,
            trading_mode=self.trading_mode,
            cycle_id=cycle_id,
            phase=phase,
            as_of=utc_now().date(),
            limit=limit,
        )
        written = 0
        for record in records:
            semantic_key = str(record.get("semantic_key") or "").strip()
            if self._candidate_memory_exists(agent_id=agent_id, semantic_key=semantic_key):
                continue
            ttl_days = record.get("ttl_days")
            expires_at = None
            try:
                if ttl_days is not None:
                    expires_at = utc_now() + timedelta(days=max(1, int(ttl_days)))
            except (TypeError, ValueError):
                expires_at = None
            self.record_memory(
                agent_id=agent_id,
                summary=str(record.get("summary") or ""),
                event_type=str(record.get("event_type") or "candidate_screen_hit"),
                score=float(record.get("score") or 0.25),
                payload=record.get("payload") if isinstance(record.get("payload"), dict) else {},
                semantic_key=semantic_key,
                memory_tier="episodic",
                expires_at=expires_at,
            )
            written += 1
        if written:
            logger.info("[cyan]Candidate memory[/cyan] agent=%s written=%d", agent_id, written)
        return written

    def record_reflection(
        self,
        agent_id: str,
        summary: str,
        *,
        score: float = 0.5,
        payload: dict | None = None,
        semantic_key: str | None = None,
    ) -> None:
        """Stores a strategy reflection / lesson memory."""
        source = self._memory_source(payload)
        self.record_memory(
            agent_id=agent_id,
            summary=summary,
            event_type="strategy_reflection",
            score=score,
            payload=payload,
            semantic_key=semantic_key,
        )
        logger.info(
            "[cyan]Reflection memory[/cyan] agent=%s source=%s semantic_key=%s summary=%s",
            agent_id,
            source or "-",
            str(semantic_key or "").strip() or "-",
            self._trim_log_text(summary, max_len=120),
        )

    @staticmethod
    def _trim_log_text(value: Any, *, max_len: int = 120) -> str:
        text = str(value or "").replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def record_manual_note(
        self,
        agent_id: str,
        summary: str,
        *,
        score: float = 0.5,
        payload: dict | None = None,
    ) -> None:
        """Stores a manual exception/note separate from compacted lessons."""
        merged_payload = {"source": "manual_note"}
        if isinstance(payload, dict):
            merged_payload.update(payload)
        self.record_memory(
            agent_id=agent_id,
            summary=summary,
            event_type="manual_note",
            score=score,
            payload=merged_payload,
        )

    def recent(self, agent_id: str, limit: int) -> list[dict]:
        """Fetches recent memory events for one agent."""
        return self.repo.recent_memory_events(agent_id=agent_id, limit=limit, trading_mode=self.trading_mode)
