from __future__ import annotations

import json
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

from arena.config import Settings
from arena.models import AccountSnapshot, utc_now

logger = logging.getLogger(__name__)

_ZERO_QTY_EPSILON = 1e-9


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_order_token(value: Any) -> str:
    return str(value or "").strip().upper()


@dataclass(slots=True)
class ReconciliationIssue:
    """Represents one reconciliation mismatch or missing prerequisite."""

    severity: str
    issue_type: str
    entity_type: str
    entity_key: str
    expected: dict[str, Any] | None = None
    actual: dict[str, Any] | None = None
    detail: dict[str, Any] | None = None


@dataclass(slots=True)
class ReconciliationResult:
    """Summarizes one reconciliation run."""

    run_id: str
    ok: bool
    status: str
    snapshot_at: datetime | None
    issues: list[ReconciliationIssue] = field(default_factory=list)
    recoveries: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    account_snapshot: AccountSnapshot | None = None


@dataclass(slots=True)
class RecoveryResult:
    """Summarizes one deterministic recovery attempt."""

    ok: bool
    before: ReconciliationResult
    after: ReconciliationResult
    applied_checkpoints: int
    status: str
    recoveries: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class StateReconciliationService:
    """Compares seed-plus-ledger positions against the latest broker snapshot."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: Any,
        excluded_tickers: list[str] | None = None,
        qty_tolerance: float = 1e-9,
        cash_tolerance_krw: float = 1.0,
        cash_reconciliation_enabled: bool = False,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.excluded_tickers = {
            str(token or "").strip().upper()
            for token in (excluded_tickers or [])
            if str(token or "").strip()
        }
        self.qty_tolerance = max(float(qty_tolerance), 0.0)
        self.cash_tolerance_krw = max(float(cash_tolerance_krw), 0.0)
        self.cash_reconciliation_enabled = bool(cash_reconciliation_enabled)

    def _tenant_token(self, tenant_id: str | None = None) -> str:
        resolver = getattr(self.repo, "resolve_tenant_id", None)
        if callable(resolver):
            return str(resolver(tenant_id))
        raw = str(tenant_id or getattr(self.repo, "tenant_id", "") or "").strip().lower()
        return raw or "local"

    def _virtual_total_cash_krw(self, agent_ids: list[str]) -> float:
        agent_tokens = [str(token).strip() for token in agent_ids if str(token).strip()]
        if self.settings.agent_capitals:
            total = sum(
                float(value)
                for key, value in self.settings.agent_capitals.items()
                if str(key).strip() in agent_tokens
            )
            if total > 0:
                return float(total)
        per_agent = max(float(self.settings.sleeve_capital_krw), 0.0)
        return per_agent * float(max(len(agent_tokens), 1))

    def _latest_snapshot_at(self, *, tenant_id: str | None = None) -> datetime | None:
        if not hasattr(self.repo, "fetch_rows") or not getattr(self.repo, "dataset_fqn", ""):
            return None
        tenant = self._tenant_token(tenant_id)
        sql = f"""
        SELECT snapshot_at
        FROM `{self.repo.dataset_fqn}.account_snapshots`
        WHERE tenant_id = @tenant_id
        ORDER BY snapshot_at DESC
        LIMIT 1
        """
        rows = self.repo.fetch_rows(sql, {"tenant_id": tenant})
        if not rows:
            return None
        snapshot_at = rows[0].get("snapshot_at")
        return snapshot_at if isinstance(snapshot_at, datetime) else None

    def _account_position_map(self, snapshot: AccountSnapshot | None) -> dict[str, float]:
        if snapshot is None:
            return {}
        out: dict[str, float] = {}
        for ticker, pos in (snapshot.positions or {}).items():
            token = str(ticker or "").strip().upper()
            if not token or token in self.excluded_tickers:
                continue
            qty = float(getattr(pos, "quantity", 0.0) or 0.0)
            if qty <= 0:
                continue
            out[token] = out.get(token, 0.0) + qty
        return out

    @staticmethod
    def _parse_positions_payload(raw_positions: Any) -> tuple[list[dict[str, Any]], str | None]:
        parsed = raw_positions
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed or "[]")
            except Exception as exc:
                return [], str(exc)
        if isinstance(parsed, dict):
            parsed = list(parsed.values())
        if not isinstance(parsed, list):
            return [], None
        return [item for item in parsed if isinstance(item, dict)], None

    @staticmethod
    def _checkpoint_event_id(
        *,
        agent_id: str,
        checkpoint_at: datetime,
        cash_krw: float,
        positions_payload: list[dict[str, Any]],
        source: str,
    ) -> str:
        payload = json.dumps(
            {
                "agent_id": str(agent_id or "").strip(),
                "checkpoint_at": checkpoint_at.isoformat(),
                "cash_krw": float(cash_krw),
                "positions": positions_payload,
                "source": str(source or "").strip(),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"chk_{hashlib.md5(payload.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _issues_have_errors(issues: list[ReconciliationIssue]) -> bool:
        return any(str(issue.severity or "").strip().lower() == "error" for issue in issues)

    def _seed_positions_from_latest_checkpoints(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str,
    ) -> tuple[dict[str, float], datetime | None, list[str], list[ReconciliationIssue]]:
        if not hasattr(self.repo, "latest_agent_state_checkpoints"):
            return {}, None, list(agent_ids), []

        configs = self.repo.latest_agent_state_checkpoints(agent_ids=agent_ids, tenant_id=tenant_id)
        missing = [agent_id for agent_id in agent_ids if agent_id not in configs]
        issues: list[ReconciliationIssue] = []
        timestamps: list[datetime] = []
        positions: dict[str, float] = defaultdict(float)

        for agent_id in agent_ids:
            cfg = configs.get(agent_id)
            if not cfg:
                continue
            checkpoint_at = cfg.get("checkpoint_at")
            if isinstance(checkpoint_at, datetime):
                timestamps.append(checkpoint_at)

            parsed, error = self._parse_positions_payload(cfg.get("positions_json"))
            if error:
                issues.append(
                    ReconciliationIssue(
                        severity="error",
                        issue_type="invalid_checkpoint_positions_json",
                        entity_type="agent",
                        entity_key=agent_id,
                        detail={"error": error},
                    )
                )
                continue

            for item in parsed:
                ticker = str(item.get("ticker") or "").strip().upper()
                if not ticker or ticker in self.excluded_tickers:
                    continue
                try:
                    qty = float(item.get("quantity") or 0.0)
                except (TypeError, ValueError):
                    qty = 0.0
                if qty <= 0:
                    continue
                positions[ticker] = positions.get(ticker, 0.0) + qty

        seed_at = max(timestamps) if timestamps else None
        if timestamps:
            earliest = min(timestamps)
            latest = max(timestamps)
            if earliest != latest:
                issues.append(
                    ReconciliationIssue(
                        severity="error",
                        issue_type="checkpoint_seed_timestamp_mismatch",
                        entity_type="checkpoint_seed",
                        entity_key="latest",
                        expected={"checkpoint_at": earliest.isoformat()},
                        actual={"checkpoint_at": latest.isoformat()},
                        detail={"agent_count": len(agent_ids)},
                    )
                )

        return dict(positions), seed_at, missing, issues

    def _seed_positions_from_latest_sleeves(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str,
    ) -> tuple[dict[str, float], datetime | None, list[str], list[ReconciliationIssue]]:
        configs = self.repo.latest_agent_sleeves(agent_ids=agent_ids, tenant_id=tenant_id)
        missing = [agent_id for agent_id in agent_ids if agent_id not in configs]
        issues: list[ReconciliationIssue] = []
        timestamps: list[datetime] = []
        positions: dict[str, float] = defaultdict(float)

        for agent_id in agent_ids:
            cfg = configs.get(agent_id)
            if not cfg:
                continue
            initialized_at = cfg.get("initialized_at")
            if isinstance(initialized_at, datetime):
                timestamps.append(initialized_at)

            parsed, error = self._parse_positions_payload(cfg.get("initial_positions_json"))
            if error:
                issues.append(
                    ReconciliationIssue(
                        severity="error",
                        issue_type="invalid_initial_positions_json",
                        entity_type="agent",
                        entity_key=agent_id,
                        detail={"error": error},
                    )
                )
                continue

            for item in parsed:
                ticker = str(item.get("ticker") or "").strip().upper()
                if not ticker or ticker in self.excluded_tickers:
                    continue
                try:
                    qty = float(item.get("quantity") or 0.0)
                except (TypeError, ValueError):
                    qty = 0.0
                if qty <= 0:
                    continue
                positions[ticker] = positions.get(ticker, 0.0) + qty

        seed_at = max(timestamps) if timestamps else None
        if timestamps:
            earliest = min(timestamps)
            latest = max(timestamps)
            if earliest != latest:
                issues.append(
                    ReconciliationIssue(
                        severity="error",
                        issue_type="sleeve_seed_timestamp_mismatch",
                        entity_type="sleeve_seed",
                        entity_key="latest",
                        expected={"initialized_at": earliest.isoformat()},
                        actual={"initialized_at": latest.isoformat()},
                        detail={"agent_count": len(agent_ids)},
                    )
                )

        return dict(positions), seed_at, missing, issues

    def _bootstrap_checkpoints_from_sleeves(
        self,
        *,
        configs: dict[str, dict[str, Any]],
        agent_ids: list[str],
        tenant_id: str,
    ) -> bool:
        append_checkpoints = getattr(self.repo, "append_agent_state_checkpoints", None)
        if not callable(append_checkpoints):
            return False

        rows: list[dict[str, Any]] = []
        for agent_id in agent_ids:
            cfg = configs.get(agent_id)
            if not cfg:
                return False
            checkpoint_at = cfg.get("initialized_at")
            if not isinstance(checkpoint_at, datetime):
                return False
            positions_payload, error = self._parse_positions_payload(cfg.get("initial_positions_json"))
            if error:
                return False
            try:
                cash_krw = float(cfg.get("initial_cash_krw") or 0.0)
            except (TypeError, ValueError):
                cash_krw = 0.0
            rows.append(
                {
                    "event_id": self._checkpoint_event_id(
                        agent_id=agent_id,
                        checkpoint_at=checkpoint_at,
                        cash_krw=cash_krw,
                        positions_payload=positions_payload,
                        source="legacy_agent_sleeve",
                    ),
                    "checkpoint_at": checkpoint_at,
                    "agent_id": agent_id,
                    "cash_krw": cash_krw,
                    "positions_json": positions_payload,
                    "source": "legacy_agent_sleeve",
                    "created_by": "reconciliation",
                    "detail_json": {"bootstrap_reason": "legacy_fallback_seed"},
                }
            )

        if not rows:
            return False
        append_checkpoints(rows, tenant_id=tenant_id)
        return True

    def _replay_position_deltas_from_ledger(
        self,
        *,
        base_positions: dict[str, float],
        since: datetime | None,
        tenant_id: str,
    ) -> tuple[dict[str, float], dict[str, float], set[str], list[ReconciliationIssue]]:
        positions = defaultdict(float)
        ai_evidence_tickers: set[str] = set()
        for ticker, qty in (base_positions or {}).items():
            token = str(ticker or "").strip().upper()
            if token and token not in self.excluded_tickers and float(qty or 0.0) > 0:
                positions[token] += float(qty)
                ai_evidence_tickers.add(token)

        if since is None:
            return dict(positions), {}, set(ai_evidence_tickers), []

        issues: list[ReconciliationIssue] = []
        external_positions = defaultdict(float)

        # --- Step 1: execution_reports = primary source for AI trades ---
        # Uses created_at (system clock) which is consistent with checkpoint_at,
        # eliminating the occurred_at (broker clock) boundary mismatch.
        ai_order_ids: set[str] = set()
        fetch_filled = getattr(self.repo, "filled_execution_reports_since", None)
        if callable(fetch_filled):
            execution_report_rows = list(fetch_filled(since=since, tenant_id=tenant_id))
            for row in execution_report_rows:
                order_id = _normalize_order_token(row.get("order_id"))
                if order_id:
                    ai_order_ids.add(order_id)
                ticker = str(row.get("ticker") or "").strip().upper()
                if not ticker or ticker in self.excluded_tickers:
                    continue
                side = str(row.get("side") or "").strip().upper()
                try:
                    qty = float(row.get("filled_qty") or 0.0)
                except (TypeError, ValueError):
                    qty = 0.0
                if qty <= 0:
                    continue
                if side == "BUY":
                    positions[ticker] += qty
                    ai_evidence_tickers.add(ticker)
                elif side == "SELL":
                    positions[ticker] -= qty
                    ai_evidence_tickers.add(ticker)
                else:
                    issues.append(
                        ReconciliationIssue(
                            severity="warning",
                            issue_type="unknown_trade_side",
                            entity_type="execution_report",
                            entity_key=order_id or ticker,
                            detail={"side": side, "ticker": ticker},
                        )
                    )

        # --- Step 2: broker_trade_events = external trade detection only ---
        # AI trades (order matched to execution_report) are skipped since
        # they were already applied from execution_reports above.
        trade_rows = self.repo.broker_trade_events_since(since=since, tenant_id=tenant_id, statuses=["FILLED"])
        for row in trade_rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker or ticker in self.excluded_tickers:
                continue
            broker_order_id = _normalize_order_token(row.get("broker_order_id"))
            if ai_order_ids and broker_order_id and broker_order_id in ai_order_ids:
                continue
            side = str(row.get("side") or "").strip().upper()
            try:
                qty = float(row.get("quantity") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty <= 0:
                continue
            if side == "BUY":
                external_positions[ticker] += qty
            elif side == "SELL":
                external_positions[ticker] -= qty
            else:
                issues.append(
                    ReconciliationIssue(
                        severity="warning",
                        issue_type="unknown_trade_side",
                        entity_type="broker_trade_event",
                        entity_key=str(row.get("event_id") or ticker),
                        detail={"side": side, "ticker": ticker},
                    )
                )

        adjust_rows = self.repo.manual_position_adjustments_since(since=since, tenant_id=tenant_id)
        for row in adjust_rows:
            ticker = str(row.get("ticker") or "").strip().upper()
            if not ticker or ticker in self.excluded_tickers:
                continue
            try:
                delta = float(row.get("delta_quantity") or 0.0)
            except (TypeError, ValueError):
                delta = 0.0
            if delta == 0:
                continue
            positions[ticker] += delta
            ai_evidence_tickers.add(ticker)

        out: dict[str, float] = {}
        for ticker, qty in positions.items():
            if abs(qty) <= _ZERO_QTY_EPSILON:
                continue
            if qty < 0:
                issues.append(
                    ReconciliationIssue(
                        severity="warning",
                        issue_type="negative_replayed_quantity",
                        entity_type="ticker",
                        entity_key=ticker,
                        actual={"ledger_quantity": qty},
                        detail={"clamped_to_zero": True},
                    )
                )
                continue
            out[ticker] = qty
        external_out = {
            ticker: qty
            for ticker, qty in external_positions.items()
            if abs(qty) > _ZERO_QTY_EPSILON
        }
        return out, external_out, ai_evidence_tickers, issues

    def _derived_agent_cash_map(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str,
        include_simulated: bool,
    ) -> tuple[dict[str, float], list[ReconciliationIssue]]:
        build_snapshot = getattr(self.repo, "build_agent_sleeve_snapshot", None)
        if not callable(build_snapshot):
            return {}, []

        cash_by_agent: dict[str, float] = {}
        issues: list[ReconciliationIssue] = []
        for agent_id in agent_ids:
            try:
                snapshot, _, _ = build_snapshot(
                    agent_id=agent_id,
                    include_simulated=include_simulated,
                    tenant_id=tenant_id,
                )
            except Exception as exc:
                issues.append(
                    ReconciliationIssue(
                        severity="error",
                        issue_type="agent_cash_snapshot_build_failed",
                        entity_type="agent",
                        entity_key=agent_id,
                        detail={"error": str(exc)},
                    )
                )
                continue
            cash = float(getattr(snapshot, "cash_krw", 0.0) or 0.0)
            cash_by_agent[agent_id] = cash
            if cash < -self.cash_tolerance_krw:
                issues.append(
                    ReconciliationIssue(
                        severity="error",
                        issue_type="negative_agent_cash",
                        entity_type="agent",
                        entity_key=agent_id,
                        actual={"cash_krw": cash},
                    )
                )
        return cash_by_agent, issues

    def _cash_event_coverage_summary(
        self,
        *,
        since: datetime | None,
        tenant_id: str,
    ) -> dict[str, Any]:
        loader = getattr(self.repo, "broker_cash_events_since", None)
        if not callable(loader) or since is None:
            return {
                "cash_event_count": 0,
                "raw_cash_event_count": 0,
                "inferred_cash_event_count": 0,
                "raw_cash_amount_krw": 0.0,
                "inferred_cash_amount_krw": 0.0,
                "cash_event_basis": "unknown",
            }

        try:
            rows = loader(since=since, tenant_id=tenant_id)
        except Exception as exc:
            logger.warning("[yellow]Broker cash coverage load skipped[/yellow] tenant=%s err=%s", tenant_id, str(exc))
            return {
                "cash_event_count": 0,
                "raw_cash_event_count": 0,
                "inferred_cash_event_count": 0,
                "raw_cash_amount_krw": 0.0,
                "inferred_cash_amount_krw": 0.0,
                "cash_event_basis": "unknown",
                "cash_event_load_error": str(exc),
            }

        raw_count = 0
        inferred_count = 0
        raw_amount_krw = 0.0
        inferred_amount_krw = 0.0
        for row in rows:
            payload = row.get("raw_payload_json")
            payload_dict = payload if isinstance(payload, dict) else {}
            source = str(row.get("source") or "").strip()
            inferred = bool(payload_dict.get("inferred")) or source == "account_cash_history_residual"
            amount_krw = _to_float(row.get("amount_krw"))
            if inferred:
                inferred_count += 1
                inferred_amount_krw += amount_krw
            else:
                raw_count += 1
                raw_amount_krw += amount_krw

        basis = "raw_only"
        if inferred_count and raw_count:
            basis = "mixed_inferred"
        elif inferred_count:
            basis = "inferred_only"
        elif not rows:
            basis = "no_cash_events"

        return {
            "cash_event_count": int(len(rows)),
            "raw_cash_event_count": int(raw_count),
            "inferred_cash_event_count": int(inferred_count),
            "raw_cash_amount_krw": float(raw_amount_krw),
            "inferred_cash_amount_krw": float(inferred_amount_krw),
            "cash_event_basis": basis,
        }

    def reconcile_positions(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str | None = None,
        include_simulated: bool = False,
        auto_recover: bool = True,
        account_snapshot: AccountSnapshot | None = None,
        sync_account_snapshot: Callable[[], AccountSnapshot] | None = None,
    ) -> ReconciliationResult:
        """Runs position-only reconciliation and persists run/issue rows."""
        tenant = self._tenant_token(tenant_id)
        agent_tokens = [str(token).strip() for token in agent_ids if str(token).strip()]
        agent_tokens = list(dict.fromkeys(agent_tokens))
        run_id = f"recon_{uuid4().hex[:12]}"
        recoveries: list[str] = []
        issues: list[ReconciliationIssue] = []

        snapshot = account_snapshot
        if snapshot is None:
            snapshot = self.repo.latest_account_snapshot(tenant_id=tenant)
        if snapshot is None and auto_recover and callable(sync_account_snapshot):
            snapshot = sync_account_snapshot()
            recoveries.append("sync_account_snapshot")

        if snapshot is None:
            issues.append(
                ReconciliationIssue(
                    severity="error",
                    issue_type="missing_account_snapshot",
                    entity_type="account_snapshot",
                    entity_key="latest",
                    detail={"tenant_id": tenant},
                )
            )

        checkpoint_positions, checkpoint_at, checkpoint_missing, checkpoint_issues = self._seed_positions_from_latest_checkpoints(
            agent_ids=agent_tokens,
            tenant_id=tenant,
        )
        seed_source = "agent_state_checkpoints"
        seed_positions = checkpoint_positions
        seed_at = checkpoint_at
        legacy_configs: dict[str, dict[str, Any]] = {}

        if checkpoint_missing and auto_recover:
            latest_sleeves = getattr(self.repo, "latest_agent_sleeves", None)
            if callable(latest_sleeves):
                legacy_configs = latest_sleeves(agent_ids=agent_tokens, tenant_id=tenant)
            if legacy_configs and self._bootstrap_checkpoints_from_sleeves(
                configs=legacy_configs,
                agent_ids=[agent_id for agent_id in agent_tokens if agent_id in legacy_configs],
                tenant_id=tenant,
            ):
                recoveries.append("bootstrap_agent_state_checkpoints")
                checkpoint_positions, checkpoint_at, checkpoint_missing, checkpoint_issues = self._seed_positions_from_latest_checkpoints(
                    agent_ids=agent_tokens,
                    tenant_id=tenant,
                )
                seed_positions = checkpoint_positions
                seed_at = checkpoint_at
            else:
                ensure_checkpoints = getattr(self.repo, "ensure_agent_state_checkpoints", None)
                if callable(ensure_checkpoints):
                    ensure_checkpoints(
                        agent_ids=agent_tokens,
                        total_cash_krw=self._virtual_total_cash_krw(agent_tokens),
                        capital_per_agent=self.settings.agent_capitals or None,
                        tenant_id=tenant,
                    )
                    recoveries.append("ensure_agent_state_checkpoints")
                    checkpoint_positions, checkpoint_at, checkpoint_missing, checkpoint_issues = self._seed_positions_from_latest_checkpoints(
                        agent_ids=agent_tokens,
                        tenant_id=tenant,
                    )
                    seed_positions = checkpoint_positions
                    seed_at = checkpoint_at
                elif hasattr(self.repo, "ensure_agent_sleeves"):
                    self.repo.ensure_agent_sleeves(
                        agent_ids=agent_tokens,
                        total_cash_krw=self._virtual_total_cash_krw(agent_tokens),
                        capital_per_agent=self.settings.agent_capitals or None,
                        tenant_id=tenant,
                    )
                    recoveries.append("ensure_agent_sleeves")

        if checkpoint_missing or self._issues_have_errors(checkpoint_issues):
            latest_sleeves = getattr(self.repo, "latest_agent_sleeves", None)
            if callable(latest_sleeves):
                legacy_configs = latest_sleeves(agent_ids=agent_tokens, tenant_id=tenant)
            issues.append(
                ReconciliationIssue(
                    severity="warning",
                    issue_type="checkpoint_seed_fallback_to_legacy_sleeve",
                    entity_type="checkpoint_seed",
                    entity_key="latest",
                    detail={
                        "missing_agents": list(checkpoint_missing),
                        "checkpoint_issue_types": [issue.issue_type for issue in checkpoint_issues],
                    },
                )
            )
            if checkpoint_missing or checkpoint_issues:
                for issue in checkpoint_issues:
                    severity = str(issue.severity or "").strip().lower()
                    if severity == "error":
                        issues.append(
                            ReconciliationIssue(
                                severity="warning",
                                issue_type=issue.issue_type,
                                entity_type=issue.entity_type,
                                entity_key=issue.entity_key,
                                expected=issue.expected,
                                actual=issue.actual,
                                detail={
                                    **(issue.detail or {}),
                                    "fallback_to": "agent_sleeves",
                                },
                            )
                        )
                    else:
                        issues.append(issue)

            seed_source = "agent_sleeves"
            seed_positions, seed_at, missing, seed_issues = self._seed_positions_from_latest_sleeves(
                agent_ids=agent_tokens,
                tenant_id=tenant,
            )
            issues.extend(seed_issues)

            if checkpoint_missing and legacy_configs and auto_recover and self._bootstrap_checkpoints_from_sleeves(
                configs=legacy_configs,
                agent_ids=[agent_id for agent_id in agent_tokens if agent_id in legacy_configs],
                tenant_id=tenant,
            ):
                recoveries.append("bootstrap_agent_state_checkpoints")
        else:
            missing = checkpoint_missing

        for agent_id in missing:
            issues.append(
                ReconciliationIssue(
                    severity="error",
                    issue_type="missing_agent_sleeve",
                    entity_type="agent",
                    entity_key=agent_id,
                    detail={"tenant_id": tenant},
                )
            )

        ledger_positions, external_positions, ai_evidence_tickers, ledger_issues = self._replay_position_deltas_from_ledger(
            base_positions=seed_positions,
            since=seed_at,
            tenant_id=tenant,
        )
        issues.extend(ledger_issues)

        broker_positions = dict(self._account_position_map(snapshot))
        external_trade_ticker_count = 0
        external_trade_quantity_total = 0.0
        for ticker, external_qty in external_positions.items():
            raw_qty = float(broker_positions.get(ticker) or 0.0)
            if abs(external_qty) <= _ZERO_QTY_EPSILON or raw_qty <= 0:
                continue
            excluded_qty = min(raw_qty, max(external_qty, 0.0))
            if excluded_qty <= 0:
                continue
            broker_positions[ticker] = max(raw_qty - excluded_qty, 0.0)
            external_trade_ticker_count += 1
            external_trade_quantity_total += excluded_qty
            issues.append(
                ReconciliationIssue(
                    severity="warning",
                    issue_type="external_broker_trade_excluded",
                    entity_type="ticker",
                    entity_key=ticker,
                    expected={"ledger_quantity": float(ledger_positions.get(ticker) or 0.0)},
                    actual={"broker_quantity": raw_qty},
                    detail={
                        "excluded_quantity": excluded_qty,
                        "remaining_broker_quantity": broker_positions[ticker],
                        "reason": "unmatched_broker_trade_event",
                    },
                )
            )

        external_carry_ticker_count = 0
        external_carry_quantity_total = 0.0
        for ticker, broker_qty in list(broker_positions.items()):
            broker_qty = float(broker_qty or 0.0)
            if broker_qty <= _ZERO_QTY_EPSILON:
                continue
            sleeve_qty = float(ledger_positions.get(ticker) or 0.0)
            if sleeve_qty > _ZERO_QTY_EPSILON:
                continue
            if ticker in ai_evidence_tickers:
                continue
            broker_positions[ticker] = 0.0
            external_carry_ticker_count += 1
            external_carry_quantity_total += broker_qty
            issues.append(
                ReconciliationIssue(
                    severity="warning",
                    issue_type="external_broker_position_excluded",
                    entity_type="ticker",
                    entity_key=ticker,
                    expected={"ledger_quantity": 0.0},
                    actual={"broker_quantity": broker_qty},
                    detail={
                        "excluded_quantity": broker_qty,
                        "reason": "no_ai_ledger_or_execution_evidence",
                    },
                )
            )

        external_overlap_ticker_count = 0
        external_overlap_quantity_total = 0.0
        for ticker, broker_qty in list(broker_positions.items()):
            broker_qty = float(broker_qty or 0.0)
            if broker_qty <= _ZERO_QTY_EPSILON:
                continue
            sleeve_qty = float(ledger_positions.get(ticker) or 0.0)
            if sleeve_qty <= _ZERO_QTY_EPSILON:
                continue
            excluded_qty = broker_qty - sleeve_qty
            if excluded_qty <= self.qty_tolerance:
                continue
            broker_positions[ticker] = sleeve_qty
            external_overlap_ticker_count += 1
            external_overlap_quantity_total += excluded_qty
            issues.append(
                ReconciliationIssue(
                    severity="warning",
                    issue_type="external_broker_position_overlap_excluded",
                    entity_type="ticker",
                    entity_key=ticker,
                    expected={"ledger_quantity": sleeve_qty},
                    actual={"broker_quantity": broker_qty},
                    detail={
                        "excluded_quantity": excluded_qty,
                        "remaining_broker_quantity": broker_positions[ticker],
                        "reason": "broker_quantity_exceeds_ai_ledger",
                    },
                )
            )

        for ticker in sorted(set(ledger_positions) | set(broker_positions)):
            sleeve_qty = float(ledger_positions.get(ticker) or 0.0)
            broker_qty = float(broker_positions.get(ticker) or 0.0)
            delta_quantity = broker_qty - sleeve_qty
            abs_delta_quantity = abs(delta_quantity)
            if abs_delta_quantity <= self.qty_tolerance:
                if abs_delta_quantity > 0:
                    issues.append(
                        ReconciliationIssue(
                            severity="warning",
                            issue_type="position_quantity_mismatch",
                            entity_type="ticker",
                            entity_key=ticker,
                            expected={"ledger_quantity": sleeve_qty},
                            actual={"broker_quantity": broker_qty},
                            detail={
                                "delta_quantity": delta_quantity,
                                "within_tolerance": True,
                                "qty_tolerance": self.qty_tolerance,
                            },
                        )
                    )
                continue
            issues.append(
                ReconciliationIssue(
                    severity="error",
                    issue_type="position_quantity_mismatch",
                    entity_type="ticker",
                    entity_key=ticker,
                    expected={"ledger_quantity": sleeve_qty},
                    actual={"broker_quantity": broker_qty},
                    detail={
                        "delta_quantity": delta_quantity,
                        "within_tolerance": False,
                        "qty_tolerance": self.qty_tolerance,
                    },
                )
            )

        broker_cash = float(getattr(snapshot, "cash_krw", 0.0) or 0.0) if snapshot is not None else 0.0
        agent_cash_by_agent: dict[str, float] = {}
        derived_agent_cash = 0.0
        unallocated_cash = broker_cash
        cash_event_summary = self._cash_event_coverage_summary(since=seed_at, tenant_id=tenant)
        if self.cash_reconciliation_enabled:
            agent_cash_by_agent, cash_issues = self._derived_agent_cash_map(
                agent_ids=agent_tokens,
                tenant_id=tenant,
                include_simulated=include_simulated,
            )
            issues.extend(cash_issues)
            derived_agent_cash = sum(float(value) for value in agent_cash_by_agent.values())
            unallocated_cash = broker_cash - derived_agent_cash
            abs_unallocated_cash = abs(unallocated_cash)
            if abs_unallocated_cash <= self.cash_tolerance_krw:
                if abs_unallocated_cash > 0:
                    issues.append(
                        ReconciliationIssue(
                            severity="warning",
                            issue_type="broker_cash_unallocated" if unallocated_cash >= 0 else "broker_cash_overallocated",
                            entity_type="cash",
                            entity_key="broker_cash",
                            expected={"broker_cash_krw": broker_cash},
                            actual={"derived_agent_cash_krw": derived_agent_cash},
                            detail={
                                "unallocated_cash_krw": unallocated_cash,
                                "within_tolerance": True,
                                "cash_tolerance_krw": self.cash_tolerance_krw,
                                "cash_event_basis": cash_event_summary["cash_event_basis"],
                                "inferred_cash_event_count": cash_event_summary["inferred_cash_event_count"],
                            },
                        )
                    )
            elif unallocated_cash < 0:
                issues.append(
                    ReconciliationIssue(
                        severity="warning",
                        issue_type="broker_cash_overallocated",
                        entity_type="cash",
                        entity_key="broker_cash",
                        expected={"broker_cash_krw": broker_cash},
                        actual={"derived_agent_cash_krw": derived_agent_cash},
                        detail={
                            "unallocated_cash_krw": unallocated_cash,
                            "within_tolerance": False,
                            "cash_tolerance_krw": self.cash_tolerance_krw,
                            "cash_event_basis": cash_event_summary["cash_event_basis"],
                            "inferred_cash_event_count": cash_event_summary["inferred_cash_event_count"],
                        },
                    )
                )
            else:
                issues.append(
                    ReconciliationIssue(
                        severity="warning",
                        issue_type="broker_cash_unallocated",
                        entity_type="cash",
                        entity_key="broker_cash",
                        expected={"broker_cash_krw": broker_cash},
                        actual={"derived_agent_cash_krw": derived_agent_cash},
                        detail={
                            "unallocated_cash_krw": unallocated_cash,
                            "within_tolerance": False,
                            "cash_tolerance_krw": self.cash_tolerance_krw,
                            "cash_event_basis": cash_event_summary["cash_event_basis"],
                            "inferred_cash_event_count": cash_event_summary["inferred_cash_event_count"],
                        },
                    )
                )

        # --- FX / price integrity checks on execution_reports ---
        fx_issues = self._check_execution_fx_integrity(
            agent_ids=agent_tokens,
            tenant_id=tenant,
            since=seed_at,
        )
        issues.extend(fx_issues)

        snapshot_at = self._latest_snapshot_at(tenant_id=tenant)
        ok = not self._issues_have_errors(issues)
        status = "recovered" if ok and recoveries else ("ok" if ok else "failed")
        summary = {
            "tenant_id": tenant,
            "agent_count": len(agent_tokens),
            "excluded_tickers": sorted(self.excluded_tickers),
            "recoveries": list(recoveries),
            "issue_count": len(issues),
            "error_count": sum(1 for issue in issues if str(issue.severity or "").strip().lower() == "error"),
            "warning_count": sum(1 for issue in issues if str(issue.severity or "").strip().lower() == "warning"),
            "seed_source": seed_source,
            "seed_initialized_at": seed_at.isoformat() if isinstance(seed_at, datetime) else None,
            "seed_ticker_count": len(seed_positions),
            "ledger_ticker_count": len(ledger_positions),
            "broker_ticker_count": len(broker_positions),
            "external_trade_ticker_count": int(external_trade_ticker_count),
            "external_trade_quantity_total": float(external_trade_quantity_total),
            "external_carry_ticker_count": int(external_carry_ticker_count),
            "external_carry_quantity_total": float(external_carry_quantity_total),
            "external_overlap_ticker_count": int(external_overlap_ticker_count),
            "external_overlap_quantity_total": float(external_overlap_quantity_total),
            "qty_tolerance": float(self.qty_tolerance),
            "cash_tolerance_krw": float(self.cash_tolerance_krw),
            "broker_cash_krw": float(broker_cash),
            "derived_agent_cash_krw": float(derived_agent_cash),
            "unallocated_cash_krw": float(unallocated_cash),
            "agent_cash_count": len(agent_cash_by_agent),
            "cash_event_basis": str(cash_event_summary["cash_event_basis"]),
            "cash_event_count": int(cash_event_summary["cash_event_count"]),
            "raw_cash_event_count": int(cash_event_summary["raw_cash_event_count"]),
            "inferred_cash_event_count": int(cash_event_summary["inferred_cash_event_count"]),
            "raw_cash_amount_krw": float(cash_event_summary["raw_cash_amount_krw"]),
            "inferred_cash_amount_krw": float(cash_event_summary["inferred_cash_amount_krw"]),
        }

        self.repo.append_reconciliation_run(
            run_id=run_id,
            status=status,
            snapshot_at=snapshot_at,
            summary=summary,
            tenant_id=tenant,
        )
        if issues:
            created_at = utc_now()
            self.repo.append_reconciliation_issues(
                [
                    {
                        "run_id": run_id,
                        "issue_id": f"issue_{uuid4().hex[:12]}",
                        "created_at": created_at,
                        "severity": issue.severity,
                        "issue_type": issue.issue_type,
                        "entity_type": issue.entity_type,
                        "entity_key": issue.entity_key,
                        "expected_json": issue.expected or None,
                        "actual_json": issue.actual or None,
                        "detail_json": issue.detail or None,
                    }
                    for issue in issues
                ],
                tenant_id=tenant,
            )

        if ok:
            logger.info(
                "[green]Reconciliation ok[/green] tenant=%s status=%s recoveries=%s",
                tenant,
                status,
                ",".join(recoveries) or "-",
            )
        else:
            logger.warning(
                "[yellow]Reconciliation failed[/yellow] tenant=%s issues=%d recoveries=%s",
                tenant,
                len(issues),
                ",".join(recoveries) or "-",
            )

        return ReconciliationResult(
            run_id=run_id,
            ok=ok,
            status=status,
            snapshot_at=snapshot_at,
            issues=issues,
            recoveries=recoveries,
            summary=summary,
            account_snapshot=snapshot,
        )


    def _check_execution_fx_integrity(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str,
        since: datetime | None,
    ) -> list[ReconciliationIssue]:
        """Detects execution_reports with missing FX or price divergence vs broker_trade_events."""
        issues: list[ReconciliationIssue] = []
        if not since:
            return issues

        # 1. FILLED executions with missing fx_rate or avg_price_native
        try:
            sql = f"""
            SELECT order_id, agent_id, ticker, side, filled_qty, avg_price_krw, avg_price_native, fx_rate, created_at
            FROM `{self.repo.dataset_fqn}.execution_reports`
            WHERE tenant_id = @tenant_id
              AND agent_id IN UNNEST(@agent_ids)
              AND created_at >= @since
              AND status = 'FILLED'
              AND (fx_rate IS NULL OR fx_rate = 0 OR avg_price_native IS NULL)
            ORDER BY created_at ASC
            LIMIT 50
            """
            rows = self.repo.fetch_rows(sql, {
                "tenant_id": tenant_id,
                "agent_ids": agent_ids,
                "since": since,
            })
            for r in rows:
                issues.append(ReconciliationIssue(
                    severity="warning",
                    issue_type="execution_missing_fx",
                    entity_type="execution_report",
                    entity_key=str(r.get("order_id") or ""),
                    expected={"fx_rate": "> 0", "avg_price_native": "not null"},
                    actual={
                        "fx_rate": float(r.get("fx_rate") or 0),
                        "avg_price_native": r.get("avg_price_native"),
                    },
                    detail={
                        "agent_id": r.get("agent_id"),
                        "ticker": r.get("ticker"),
                        "created_at": str(r.get("created_at")),
                    },
                ))
        except Exception as exc:
            logger.warning("[yellow]execution fx integrity check skipped[/yellow] err=%s", str(exc))

        # 2. Price divergence: execution vs broker_trade_events (>1%)
        try:
            sql = f"""
            WITH broker_agg AS (
              SELECT broker_order_id, ticker, side,
                SUM(quantity * price_native) / NULLIF(SUM(quantity), 0) AS wavg_native,
                MAX(fx_rate) AS broker_fx
              FROM `{self.repo.dataset_fqn}.broker_trade_events`
              WHERE tenant_id = @tenant_id
                AND price_native IS NOT NULL AND price_native > 0
              GROUP BY broker_order_id, ticker, side
            )
            SELECT e.order_id, e.agent_id, e.ticker, e.created_at,
              e.avg_price_krw AS exec_krw,
              b.wavg_native * COALESCE(NULLIF(b.broker_fx, 0), NULLIF(e.fx_rate, 0), 1) AS broker_krw
            FROM `{self.repo.dataset_fqn}.execution_reports` e
            JOIN broker_agg b
              ON e.order_id = b.broker_order_id AND e.ticker = b.ticker AND e.side = b.side
            WHERE e.tenant_id = @tenant_id
              AND e.agent_id IN UNNEST(@agent_ids)
              AND e.created_at >= @since
              AND e.status = 'FILLED'
              AND e.avg_price_krw > 0
              AND ABS(e.avg_price_krw - b.wavg_native * COALESCE(NULLIF(b.broker_fx, 0), NULLIF(e.fx_rate, 0), 1))
                  / GREATEST(e.avg_price_krw, 1) > 0.01
            LIMIT 50
            """
            rows = self.repo.fetch_rows(sql, {
                "tenant_id": tenant_id,
                "agent_ids": agent_ids,
                "since": since,
            })
            for r in rows:
                issues.append(ReconciliationIssue(
                    severity="warning",
                    issue_type="execution_price_divergence",
                    entity_type="execution_report",
                    entity_key=str(r.get("order_id") or ""),
                    expected={"broker_price_krw": float(r.get("broker_krw") or 0)},
                    actual={"execution_price_krw": float(r.get("exec_krw") or 0)},
                    detail={
                        "agent_id": r.get("agent_id"),
                        "ticker": r.get("ticker"),
                        "created_at": str(r.get("created_at")),
                        "divergence_pct": round(
                            abs(float(r.get("exec_krw") or 0) - float(r.get("broker_krw") or 0))
                            / max(float(r.get("exec_krw") or 1), 1) * 100, 2
                        ),
                    },
                ))
        except Exception as exc:
            logger.warning("[yellow]execution price divergence check skipped[/yellow] err=%s", str(exc))

        return issues


class StateRecoveryService:
    """Runs deterministic checkpoint rebuilds before giving up on reconciliation."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: Any,
        excluded_tickers: list[str] | None = None,
        qty_tolerance: float = 1e-9,
        cash_tolerance_krw: float = 1.0,
        cash_reconciliation_enabled: bool = True,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.excluded_tickers = list(excluded_tickers or [])
        self.qty_tolerance = max(float(qty_tolerance), 0.0)
        self.cash_tolerance_krw = max(float(cash_tolerance_krw), 0.0)
        self.cash_reconciliation_enabled = bool(cash_reconciliation_enabled)

    def _tenant_token(self, tenant_id: str | None = None) -> str:
        resolver = getattr(self.repo, "resolve_tenant_id", None)
        if callable(resolver):
            return str(resolver(tenant_id))
        raw = str(tenant_id or getattr(self.repo, "tenant_id", "") or "").strip().lower()
        return raw or "local"

    def _checkpoint_rows_from_current_state(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str,
        include_simulated: bool,
        created_by: str,
    ) -> list[dict[str, Any]]:
        build_snapshot = getattr(self.repo, "build_agent_sleeve_snapshot", None)
        if not callable(build_snapshot):
            raise RuntimeError("build_agent_sleeve_snapshot is not available on this repository")

        checkpoint_at = utc_now()
        rows: list[dict[str, Any]] = []
        for agent_id in agent_ids:
            snapshot, _, meta = build_snapshot(
                agent_id=agent_id,
                include_simulated=include_simulated,
                tenant_id=tenant_id,
            )
            positions_payload: list[dict[str, Any]] = []
            for pos in sorted(snapshot.positions.values(), key=lambda item: str(item.ticker or "")):
                try:
                    qty = float(pos.quantity)
                except (TypeError, ValueError):
                    qty = 0.0
                if qty <= 0:
                    continue
                row: dict[str, Any] = {
                    "ticker": str(pos.ticker or "").strip().upper(),
                    "exchange_code": str(pos.exchange_code or ""),
                    "instrument_id": str(pos.instrument_id or ""),
                    "quantity": qty,
                    "avg_price_krw": float(pos.avg_price_krw or 0.0),
                }
                if pos.avg_price_native is not None:
                    row["avg_price_native"] = pos.avg_price_native
                if pos.quote_currency:
                    row["quote_currency"] = pos.quote_currency
                if float(pos.fx_rate or 0.0) > 0:
                    row["fx_rate"] = pos.fx_rate
                positions_payload.append(row)

            payload = json.dumps(
                {
                    "agent_id": str(agent_id or "").strip(),
                    "checkpoint_at": checkpoint_at.isoformat(),
                    "cash_krw": float(snapshot.cash_krw or 0.0),
                    "positions": positions_payload,
                    "source": "recovery_rebuild",
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            rows.append(
                {
                    "event_id": f"chk_{hashlib.md5(payload.encode('utf-8')).hexdigest()}",
                    "checkpoint_at": checkpoint_at,
                    "agent_id": str(agent_id or "").strip(),
                    "cash_krw": float(snapshot.cash_krw or 0.0),
                    "positions_json": positions_payload,
                    "source": "recovery_rebuild",
                    "created_by": str(created_by or "").strip() or "system",
                    "detail_json": {
                        "seed_source": str((meta or {}).get("seed_source") or ""),
                        "trade_count_total": int((meta or {}).get("trade_count_total") or 0),
                        "capital_event_count": int((meta or {}).get("capital_event_count") or 0),
                    },
                }
            )
        return rows

    def recover_checkpoints_from_current_state(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str | None = None,
        include_simulated: bool = False,
        created_by: str = "system",
    ) -> int:
        tenant = self._tenant_token(tenant_id)
        append_checkpoints = getattr(self.repo, "append_agent_state_checkpoints", None)
        if not callable(append_checkpoints):
            raise RuntimeError("append_agent_state_checkpoints is not available on this repository")
        rows = self._checkpoint_rows_from_current_state(
            agent_ids=agent_ids,
            tenant_id=tenant,
            include_simulated=include_simulated,
            created_by=created_by,
        )
        append_checkpoints(rows, tenant_id=tenant)
        return len(rows)

    def recover_and_reconcile(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str | None = None,
        include_simulated: bool = False,
        auto_recover: bool = True,
        account_snapshot: AccountSnapshot | None = None,
        sync_account_snapshot: Callable[[], AccountSnapshot] | None = None,
        created_by: str = "system",
        allow_checkpoint_rebuild: bool = True,
    ) -> RecoveryResult:
        tenant = self._tenant_token(tenant_id)
        recon = StateReconciliationService(
            settings=self.settings,
            repo=self.repo,
            excluded_tickers=self.excluded_tickers,
            qty_tolerance=self.qty_tolerance,
            cash_tolerance_krw=self.cash_tolerance_krw,
            cash_reconciliation_enabled=self.cash_reconciliation_enabled,
        )
        before = recon.reconcile_positions(
            agent_ids=agent_ids,
            tenant_id=tenant,
            include_simulated=include_simulated,
            auto_recover=auto_recover,
            account_snapshot=account_snapshot,
            sync_account_snapshot=sync_account_snapshot,
        )
        if before.ok:
            return RecoveryResult(
                ok=True,
                before=before,
                after=before,
                applied_checkpoints=0,
                status="noop",
                recoveries=["already_reconciled"],
                summary={"tenant_id": tenant},
            )

        if not allow_checkpoint_rebuild:
            return RecoveryResult(
                ok=False,
                before=before,
                after=before,
                applied_checkpoints=0,
                status="failed",
                recoveries=["checkpoint_rebuild_disabled"],
                summary={
                    "tenant_id": tenant,
                    "applied_checkpoints": 0,
                    "before_status": before.status,
                    "after_status": before.status,
                },
            )

        applied = self.recover_checkpoints_from_current_state(
            agent_ids=agent_ids,
            tenant_id=tenant,
            include_simulated=include_simulated,
            created_by=created_by,
        )
        after = recon.reconcile_positions(
            agent_ids=agent_ids,
            tenant_id=tenant,
            include_simulated=include_simulated,
            auto_recover=False,
            account_snapshot=before.account_snapshot,
        )
        return RecoveryResult(
            ok=after.ok,
            before=before,
            after=after,
            applied_checkpoints=applied,
            status="recovered" if after.ok else "failed",
            recoveries=["checkpoint_rebuild"],
            summary={
                "tenant_id": tenant,
                "applied_checkpoints": applied,
                "before_status": before.status,
                "after_status": after.status,
            },
        )
