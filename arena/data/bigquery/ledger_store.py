"""Ledger store — broker trades, cash events, capital events, reconciliation."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from arena.models import utc_now

if TYPE_CHECKING:
    from arena.data.bigquery.session import BigQuerySession


def _json_safe(value: Any) -> Any:
    """Converts nested values into BigQuery-safe primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class LedgerStore:
    """Append-only ledger and reconciliation store operations."""

    _JSON_COLUMN_NAMES = frozenset(
        {
            "raw_payload_json",
            "positions_json",
            "detail_json",
            "summary_json",
            "expected_json",
            "actual_json",
        }
    )

    def __init__(self, session: BigQuerySession) -> None:
        self.session = session

    def _tenant_token(self, tenant_id: str | None = None) -> str:
        return self.session.resolve_tenant_id(tenant_id)

    def _append_rows(
        self,
        table_name: str,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        if not rows:
            return
        tenant = self._tenant_token(tenant_id)
        deduped_rows = list(rows)
        event_ids = [str(r.get("event_id") or "").strip() for r in rows if str(r.get("event_id") or "").strip()]
        if event_ids:
            existing = self.existing_event_ids(table_name, event_ids, tenant_id=tenant)
            seen: set[str] = set()
            deduped_rows = []
            for row in rows:
                event_id = str(row.get("event_id") or "").strip()
                if event_id:
                    if event_id in seen or event_id in existing:
                        continue
                    seen.add(event_id)
                deduped_rows.append(row)
            if not deduped_rows:
                return
        table_id = f"{self.session.dataset_fqn}.{table_name}"
        payloads = []
        for row in deduped_rows:
            payload = dict(row)
            payload["tenant_id"] = tenant
            for key, value in list(payload.items()):
                if key not in self._JSON_COLUMN_NAMES:
                    continue
                if value is None or isinstance(value, str):
                    continue
                payload[key] = json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
            payloads.append(_json_safe(payload))
        errors = self.session.client.insert_rows_json(table_id, payloads)
        if errors:
            raise RuntimeError(f"{table_name} insert failed: {errors}")

    def existing_event_ids(
        self,
        table_name: str,
        event_ids: list[str],
        *,
        tenant_id: str | None = None,
    ) -> set[str]:
        """Returns event_ids that already exist in the target table."""
        tokens = [str(token).strip() for token in event_ids if str(token).strip()]
        if not tokens:
            return set()
        tenant = self._tenant_token(tenant_id)
        sql = f"""
        SELECT event_id
        FROM `{self.session.dataset_fqn}.{table_name}`
        WHERE tenant_id = @tenant_id
          AND event_id IN UNNEST(@event_ids)
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant, "event_ids": tokens})
        return {str(row.get("event_id") or "").strip() for row in rows if str(row.get("event_id") or "").strip()}

    def append_broker_trade_events(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends raw broker fill/trade events."""
        self._append_rows("broker_trade_events", rows, tenant_id=tenant_id)

    def append_broker_cash_events(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends raw broker cash movement events."""
        self._append_rows("broker_cash_events", rows, tenant_id=tenant_id)

    def append_capital_events(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends agent-level capital injections or withdrawals."""
        self._append_rows("capital_events", rows, tenant_id=tenant_id)

    def append_agent_transfer_events(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends explicit transfers between virtual sleeves."""
        self._append_rows("agent_transfer_events", rows, tenant_id=tenant_id)

    def append_manual_adjustments(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends auditable manual correction events."""
        self._append_rows("manual_adjustments", rows, tenant_id=tenant_id)

    def append_agent_state_checkpoints(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends agent-level checkpoint rows for deterministic recovery."""
        self._append_rows("agent_state_checkpoints", rows, tenant_id=tenant_id)

    def append_reconciliation_run(
        self,
        *,
        run_id: str,
        status: str,
        snapshot_at: datetime | None = None,
        summary: dict[str, Any] | None = None,
        run_at: datetime | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Appends one reconciliation run summary row."""
        self._append_rows(
            "reconciliation_runs",
            [
                {
                    "run_id": str(run_id or "").strip(),
                    "run_at": run_at or utc_now(),
                    "snapshot_at": snapshot_at,
                    "status": str(status or "").strip().lower() or "unknown",
                    "summary_json": summary or {},
                }
            ],
            tenant_id=tenant_id,
        )

    def append_reconciliation_issues(
        self,
        rows: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
    ) -> None:
        """Appends issue rows produced during reconciliation."""
        self._append_rows("reconciliation_issues", rows, tenant_id=tenant_id)

    def latest_reconciliation_run(
        self,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Returns the latest reconciliation run for the tenant."""
        tenant = self._tenant_token(tenant_id)
        sql = f"""
        SELECT run_id, run_at, snapshot_at, status, summary_json
        FROM `{self.session.dataset_fqn}.reconciliation_runs`
        WHERE tenant_id = @tenant_id
        ORDER BY run_at DESC
        LIMIT 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant})
        return rows[0] if rows else None

    def latest_agent_state_checkpoints(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Returns the latest checkpoint row per agent."""
        tenant = self._tenant_token(tenant_id)
        tokens = [str(a).strip() for a in agent_ids if str(a).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        sql = f"""
        SELECT agent_id, event_id, checkpoint_at, cash_krw, positions_json, source, created_by, detail_json
        FROM (
          SELECT
            agent_id, event_id, checkpoint_at, cash_krw, positions_json, source, created_by, detail_json,
            ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY checkpoint_at DESC, event_id DESC) AS rn
          FROM `{self.session.dataset_fqn}.agent_state_checkpoints`
          WHERE tenant_id = @tenant_id
            AND agent_id IN UNNEST(@agent_ids)
        )
        WHERE rn = 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant, "agent_ids": tokens})
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            agent_id = str(row.get("agent_id") or "").strip()
            if agent_id:
                out[agent_id] = row
        return out

    def broker_trade_events_since(
        self,
        *,
        since: datetime,
        tenant_id: str | None = None,
        statuses: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns broker trade events on or after a timestamp."""
        tenant = self._tenant_token(tenant_id)
        tokens = [str(token).strip().upper() for token in (statuses or ["FILLED"]) if str(token).strip()]
        sql = f"""
        SELECT
          event_id, occurred_at, broker_order_id, broker_fill_id, ticker,
          exchange_code, instrument_id, side, quantity, price_krw, price_native,
          quote_currency, fx_rate, fee_krw, status, source, raw_payload_json
        FROM `{self.session.dataset_fqn}.broker_trade_events`
        WHERE tenant_id = @tenant_id
          AND occurred_at >= @since
          AND status IN UNNEST(@statuses)
        ORDER BY occurred_at ASC, event_id ASC
        """
        return self.session.fetch_rows(sql, {"tenant_id": tenant, "since": since, "statuses": tokens})

    def broker_cash_events_since(
        self,
        *,
        since: datetime,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns broker cash events on or after a timestamp."""
        tenant = self._tenant_token(tenant_id)
        sql = f"""
        SELECT
          event_id, occurred_at, account_id, currency, amount_native, amount_krw,
          fx_rate, event_type, source, raw_payload_json
        FROM `{self.session.dataset_fqn}.broker_cash_events`
        WHERE tenant_id = @tenant_id
          AND occurred_at >= @since
        ORDER BY occurred_at ASC, event_id ASC
        """
        return self.session.fetch_rows(sql, {"tenant_id": tenant, "since": since})

    def manual_position_adjustments_since(
        self,
        *,
        since: datetime,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns quantity-changing manual adjustments on or after a timestamp."""
        tenant = self._tenant_token(tenant_id)
        sql = f"""
        SELECT event_id, occurred_at, agent_id, ticker, delta_quantity, reason, raw_payload_json
        FROM `{self.session.dataset_fqn}.manual_adjustments`
        WHERE tenant_id = @tenant_id
          AND occurred_at >= @since
          AND delta_quantity IS NOT NULL
          AND delta_quantity != 0
        ORDER BY occurred_at ASC, event_id ASC
        """
        return self.session.fetch_rows(sql, {"tenant_id": tenant, "since": since})

    def manual_cash_adjustments_since(
        self,
        *,
        agent_id: str,
        since: datetime,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns cash-changing manual adjustments on or after a timestamp."""
        tenant = self._tenant_token(tenant_id)
        agent = str(agent_id).strip()
        sql = f"""
        SELECT event_id, occurred_at, agent_id, delta_cash_krw, reason, raw_payload_json
        FROM `{self.session.dataset_fqn}.manual_adjustments`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND occurred_at >= @since
          AND delta_cash_krw IS NOT NULL
          AND delta_cash_krw != 0
        ORDER BY occurred_at ASC, event_id ASC
        """
        return self.session.fetch_rows(sql, {"tenant_id": tenant, "agent_id": agent, "since": since})

    def capital_events_since(
        self,
        *,
        agent_id: str,
        since: datetime,
        tenant_id: str | None = None,
        event_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns agent capital-flow events on or after a timestamp."""
        tenant = self._tenant_token(tenant_id)
        agent = str(agent_id).strip()
        tokens = [str(token).strip().upper() for token in (event_types or []) if str(token).strip()]
        filters = [
            "tenant_id = @tenant_id",
            "agent_id = @agent_id",
            "occurred_at >= @since",
        ]
        params: dict[str, Any] = {"tenant_id": tenant, "agent_id": agent, "since": since}
        if tokens:
            filters.append("event_type IN UNNEST(@event_types)")
            params["event_types"] = tokens
        where = " AND ".join(filters)
        sql = f"""
        SELECT event_id, occurred_at, agent_id, amount_krw, event_type, reason, created_by
        FROM `{self.session.dataset_fqn}.capital_events`
        WHERE {where}
        ORDER BY occurred_at ASC, event_id ASC
        """
        return self.session.fetch_rows(sql, params)

    def agent_transfer_events_since(
        self,
        *,
        agent_id: str,
        since: datetime,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns transfer events that affect the given agent on or after a timestamp."""
        tenant = self._tenant_token(tenant_id)
        agent = str(agent_id).strip()
        sql = f"""
        SELECT
          event_id, occurred_at, transfer_type, from_agent_id, to_agent_id,
          ticker, quantity, price_krw, amount_krw, reason, created_by
        FROM `{self.session.dataset_fqn}.agent_transfer_events`
        WHERE tenant_id = @tenant_id
          AND occurred_at >= @since
          AND (@agent_id = from_agent_id OR @agent_id = to_agent_id)
        ORDER BY occurred_at ASC, event_id ASC
        """
        return self.session.fetch_rows(sql, {"tenant_id": tenant, "agent_id": agent, "since": since})
