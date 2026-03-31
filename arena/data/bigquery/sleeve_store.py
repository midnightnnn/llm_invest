"""Sleeve store — account snapshots, agent sleeves, NAV, dividends, position management."""

from __future__ import annotations

import json
import hashlib
import logging
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from arena.models import AccountSnapshot, Position, utc_now

if TYPE_CHECKING:
    from arena.data.bigquery.session import BigQuerySession

logger = logging.getLogger(__name__)


class SleeveStore:
    """Sleeve/account snapshot + agent NAV store operations."""

    def __init__(
        self,
        session: BigQuerySession,
        *,
        ledger: Any | None = None,
        market: Any | None = None,
    ) -> None:
        self.session = session
        self._ledger = ledger
        self._market = market

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _chained_nav_index(
        cls,
        nav_vals: list[float | None],
        pnl_krw_vals: list[float | None],
        pnl_ratio_vals: list[float | None],
    ) -> list[float | None]:
        """Chains NAV returns across retargets by neutralizing baseline changes."""
        out: list[float | None] = []
        cum = 1.0
        prev_baseline: float | None = None
        prev_nav: float | None = None
        for nav, pnl_krw, pnl_ratio in zip(nav_vals, pnl_krw_vals, pnl_ratio_vals):
            if nav is None or float(nav) <= 0:
                out.append(None)
                continue
            nav_f = float(nav)
            pnl_krw_f = float(pnl_krw) if pnl_krw is not None else 0.0
            pnl_ratio_f = float(pnl_ratio) if pnl_ratio is not None else 0.0
            baseline = nav_f - pnl_krw_f
            if prev_nav is not None:
                if prev_baseline is not None and abs(prev_baseline) > 0 and abs(baseline - prev_baseline) / abs(prev_baseline) > 0.05:
                    cum *= (1.0 + pnl_ratio_f)
                else:
                    cum *= (nav_f / prev_nav)
            out.append(100.0 * cum)
            prev_baseline = baseline
            prev_nav = nav_f
        return out

    def fetch_agent_nav_history(
        self,
        *,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        agent_ids: list[str] | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Loads agent NAV history rows ordered for chained-return calculation."""
        tenant = self._tenant_token(tenant_id)
        filters = ["tenant_id = @tenant_id"]
        params: dict[str, Any] = {"tenant_id": tenant, "limit": max(int(limit), 1)}

        if agent_ids:
            ids = [str(token).strip().lower() for token in agent_ids if str(token).strip()]
            if ids:
                filters.append("LOWER(agent_id) IN UNNEST(@agent_ids)")
                params["agent_ids"] = ids
        if agent_id:
            filters.append("LOWER(agent_id) = @agent_id")
            params["agent_id"] = str(agent_id).strip().lower()

        where = " AND ".join(filters)
        sql = f"""
        WITH nav_rows AS (
          SELECT nav_date, agent_id, nav_krw, pnl_krw, pnl_ratio, 0 AS source_priority
          FROM `{self.session.dataset_fqn}.official_nav_daily`
          WHERE {where}
          UNION ALL
          SELECT nav_date, agent_id, nav_krw, pnl_krw, pnl_ratio, 1 AS source_priority
          FROM `{self.session.dataset_fqn}.agent_nav_daily`
          WHERE {where}
        )
        SELECT nav_date, agent_id, nav_krw, pnl_krw, pnl_ratio
        FROM nav_rows
        QUALIFY ROW_NUMBER() OVER (PARTITION BY nav_date, agent_id ORDER BY source_priority ASC) = 1
        ORDER BY nav_date ASC, agent_id ASC
        LIMIT @limit
        """
        return self.session.fetch_rows(sql, params)

    def latest_agent_chained_returns(
        self,
        *,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        agent_ids: list[str] | None = None,
        limit: int = 10000,
    ) -> dict[str, dict[str, float]]:
        """Returns latest chained return stats per agent based on agent_nav_daily."""
        rows = self.fetch_agent_nav_history(
            tenant_id=tenant_id,
            agent_id=agent_id,
            agent_ids=agent_ids,
            limit=limit,
        )
        rows_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            aid = str(row.get("agent_id") or "").strip()
            if not aid:
                continue
            rows_by_agent[aid].append(row)

        out: dict[str, dict[str, float]] = {}
        for aid, agent_rows in rows_by_agent.items():
            nav_vals = [self._safe_float(r.get("nav_krw"), 0.0) or None for r in agent_rows]
            pnl_krw_vals = [r.get("pnl_krw") for r in agent_rows]
            pnl_ratio_vals = [r.get("pnl_ratio") for r in agent_rows]
            idx_vals = self._chained_nav_index(nav_vals, pnl_krw_vals, pnl_ratio_vals)

            last_index = None
            last_nav = None
            first_baseline = None
            latest_baseline = None
            first_nav_date = None
            latest_nav_date = None
            for row, idx in zip(agent_rows, idx_vals):
                nav = self._safe_float(row.get("nav_krw"), 0.0)
                pnl = self._safe_float(row.get("pnl_krw"), 0.0)
                baseline = nav - pnl if nav > 0 else 0.0
                nav_date = row.get("nav_date")
                if nav_date is not None and first_nav_date is None:
                    first_nav_date = nav_date
                if nav_date is not None:
                    latest_nav_date = nav_date
                if baseline > 0 and first_baseline is None:
                    first_baseline = baseline
                if baseline > 0:
                    latest_baseline = baseline
                if idx is not None:
                    last_index = float(idx)
                if nav > 0:
                    last_nav = nav

            if last_index is None:
                continue

            return_ratio = (last_index / 100.0) - 1.0
            base_for_pnl = first_baseline if first_baseline and first_baseline > 0 else latest_baseline
            return_pnl_krw = float(base_for_pnl or 0.0) * return_ratio if base_for_pnl else 0.0
            out[aid] = {
                "return_ratio": return_ratio,
                "return_index": last_index,
                "return_pnl_krw": return_pnl_krw,
                "first_baseline_krw": float(first_baseline or 0.0),
                "latest_baseline_krw": float(latest_baseline or 0.0),
                "latest_nav_krw": float(last_nav or 0.0),
                "started_at": str(first_nav_date) if first_nav_date is not None else "",
                "latest_nav_date": str(latest_nav_date) if latest_nav_date is not None else "",
            }
        return out

    def _tenant_token(self, tenant_id: str | None = None) -> str:
        return str(self.session.resolve_tenant_id(tenant_id))

    def latest_account_snapshot(self, *, tenant_id: str | None = None) -> AccountSnapshot | None:
        """Loads the latest account cash, equity, and current positions."""
        tenant = self._tenant_token(tenant_id)
        sql = f"""
        SELECT snapshot_at, cash_krw, total_equity_krw, usd_krw_rate,
               cash_foreign, cash_foreign_currency
        FROM `{self.session.dataset_fqn}.account_snapshots`
        WHERE tenant_id = @tenant_id
        ORDER BY snapshot_at DESC
        LIMIT 1
        """
        head = self.session.fetch_rows(sql, {"tenant_id": tenant})
        if not head:
            return None

        snapshot_at = head[0]["snapshot_at"]
        pos_sql = f"""
        SELECT ticker, exchange_code, instrument_id, quantity,
               avg_price_krw, market_price_krw,
               avg_price_native, market_price_native, quote_currency, fx_rate
        FROM `{self.session.dataset_fqn}.positions_current`
        WHERE tenant_id = @tenant_id
          AND snapshot_at = @snapshot_at
        """
        positions_rows = self.session.fetch_rows(pos_sql, {"tenant_id": tenant, "snapshot_at": snapshot_at})
        positions = {
            row["ticker"]: Position(
                ticker=row["ticker"],
                exchange_code=str(row.get("exchange_code") or ""),
                instrument_id=str(row.get("instrument_id") or ""),
                quantity=float(row["quantity"]),
                avg_price_krw=float(row["avg_price_krw"]),
                market_price_krw=float(row["market_price_krw"]),
                avg_price_native=float(row["avg_price_native"]) if row.get("avg_price_native") is not None else None,
                market_price_native=float(row["market_price_native"]) if row.get("market_price_native") is not None else None,
                quote_currency=str(row.get("quote_currency") or ""),
                fx_rate=float(row.get("fx_rate") or 0.0),
            )
            for row in positions_rows
        }
        h = head[0]
        return AccountSnapshot(
            cash_krw=float(h["cash_krw"]),
            total_equity_krw=float(h["total_equity_krw"]),
            positions=positions,
            usd_krw_rate=float(h.get("usd_krw_rate") or 0.0),
            cash_foreign=float(h.get("cash_foreign") or 0.0),
            cash_foreign_currency=str(h.get("cash_foreign_currency") or ""),
        )

    def account_cash_history(
        self,
        *,
        start_at: datetime,
        end_at: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns account cash checkpoints across a time range."""
        tenant = self._tenant_token(tenant_id)
        filters = ["tenant_id = @tenant_id", "snapshot_at >= @start_at"]
        params: dict[str, Any] = {"tenant_id": tenant, "start_at": start_at}
        if end_at is not None:
            filters.append("snapshot_at <= @end_at")
            params["end_at"] = end_at

        sql = f"""
        SELECT snapshot_at, cash_krw, total_equity_krw, usd_krw_rate, cash_foreign, cash_foreign_currency
        FROM `{self.session.dataset_fqn}.account_snapshots`
        WHERE {' AND '.join(filters)}
        ORDER BY snapshot_at ASC
        """
        return self.session.fetch_rows(sql, params)

    def account_holdings_at_date(
        self,
        *,
        as_of_date: date,
        ticker: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, float]:
        """Returns broker-account holdings from the latest snapshot on or before a date."""
        tenant = self._tenant_token(tenant_id)
        filters = ["tenant_id = @tenant_id", "snapshot_at < TIMESTAMP(DATE_ADD(@as_of_date, INTERVAL 1 DAY))"]
        params: dict[str, Any] = {"tenant_id": tenant, "as_of_date": as_of_date}
        token = str(ticker or "").strip().upper()
        if token:
            filters.append("ticker = @ticker")
            params["ticker"] = token

        sql = f"""
        WITH latest_snapshot AS (
          SELECT snapshot_at
          FROM `{self.session.dataset_fqn}.account_snapshots`
          WHERE tenant_id = @tenant_id
            AND snapshot_at < TIMESTAMP(DATE_ADD(@as_of_date, INTERVAL 1 DAY))
          ORDER BY snapshot_at DESC
          LIMIT 1
        )
        SELECT ticker, quantity
        FROM `{self.session.dataset_fqn}.positions_current`
        WHERE {' AND '.join(filters)}
          AND snapshot_at = (SELECT snapshot_at FROM latest_snapshot)
          AND quantity > 0
        """
        rows = self.session.fetch_rows(sql, params)
        out: dict[str, float] = {}
        for row in rows:
            t = str(row.get("ticker") or "").strip().upper()
            if not t:
                continue
            try:
                qty = float(row.get("quantity") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if qty > 0:
                out[t] = qty
        return out

    def write_account_snapshot(
        self,
        snapshot: AccountSnapshot,
        snapshot_at: datetime | None = None,
        *,
        tenant_id: str | None = None,
    ) -> datetime:
        """Persists one full account snapshot and matching positions rows."""
        tenant = self._tenant_token(tenant_id)
        ts = snapshot_at or utc_now()

        snapshots_table = f"{self.session.dataset_fqn}.account_snapshots"
        snapshot_row = {
            "tenant_id": tenant,
            "snapshot_at": ts.isoformat(),
            "cash_krw": snapshot.cash_krw,
            "total_equity_krw": snapshot.total_equity_krw,
            "usd_krw_rate": snapshot.usd_krw_rate if snapshot.usd_krw_rate > 0 else None,
            "cash_foreign": snapshot.cash_foreign if snapshot.cash_foreign > 0 else None,
            "cash_foreign_currency": snapshot.cash_foreign_currency or None,
        }
        errors = self.session.client.insert_rows_json(snapshots_table, [snapshot_row])
        if errors:
            raise RuntimeError(f"account_snapshots insert failed: {errors}")

        if snapshot.positions:
            positions_table = f"{self.session.dataset_fqn}.positions_current"
            rows = [
                {
                    "tenant_id": tenant,
                    "snapshot_at": ts.isoformat(),
                    "ticker": pos.ticker,
                    "exchange_code": pos.exchange_code,
                    "instrument_id": pos.instrument_id,
                    "quantity": pos.quantity,
                    "avg_price_krw": pos.avg_price_krw,
                    "market_price_krw": pos.market_price_krw,
                    "avg_price_native": pos.avg_price_native,
                    "market_price_native": pos.market_price_native,
                    "quote_currency": pos.quote_currency or None,
                    "fx_rate": pos.fx_rate if pos.fx_rate > 0 else None,
                }
                for pos in snapshot.positions.values()
            ]
            errors = self.session.client.insert_rows_json(positions_table, rows)
            if errors:
                logger.error(
                    "[red]Account snapshot positions insert failed[/red] snapshot_at=%s errors=%s",
                    ts.isoformat(),
                    errors,
                )
                try:
                    self.session.execute(
                        f"DELETE FROM `{positions_table}` WHERE tenant_id = @tenant_id AND snapshot_at = @snapshot_at",
                        {"tenant_id": tenant, "snapshot_at": ts},
                    )
                    self.session.execute(
                        f"DELETE FROM `{snapshots_table}` WHERE tenant_id = @tenant_id AND snapshot_at = @snapshot_at",
                        {"tenant_id": tenant, "snapshot_at": ts},
                    )
                except Exception as rollback_exc:
                    logger.error(
                        "[red]Account snapshot rollback failed[/red] snapshot_at=%s err=%s",
                        ts.isoformat(),
                        str(rollback_exc),
                    )
                raise RuntimeError(f"positions_current insert failed: {errors}")

        append_cash_events = getattr(self._ledger, "append_broker_cash_events", None) if self._ledger else None
        if callable(append_cash_events):
            raw_payload = {
                "cash_krw": float(snapshot.cash_krw),
                "total_equity_krw": float(snapshot.total_equity_krw),
                "usd_krw_rate": float(snapshot.usd_krw_rate or 0.0),
                "cash_foreign": float(snapshot.cash_foreign or 0.0),
                "cash_foreign_currency": str(snapshot.cash_foreign_currency or ""),
            }
            event_id = f"cashchk_{hashlib.md5(json.dumps({'tenant_id': tenant, 'snapshot_at': ts.isoformat(), **raw_payload}, sort_keys=True).encode('utf-8')).hexdigest()}"
            append_cash_events(
                [
                    {
                        "event_id": event_id,
                        "occurred_at": ts,
                        "account_id": None,
                        "currency": str(snapshot.cash_foreign_currency or "KRW").strip() or "KRW",
                        "amount_native": float(snapshot.cash_foreign) if float(snapshot.cash_foreign or 0.0) > 0 else None,
                        "amount_krw": float(snapshot.cash_krw),
                        "fx_rate": float(snapshot.usd_krw_rate) if float(snapshot.usd_krw_rate or 0.0) > 0 else None,
                        "event_type": "CASH_CHECKPOINT",
                        "source": "account_snapshot",
                        "raw_payload_json": raw_payload,
                    }
                ],
                tenant_id=tenant,
            )

        return ts

    def get_all_held_tickers(self, *, tenant_id: str | None = None, market: str = "") -> list[str]:
        """Returns distinct tickers with positive net positions across all agents.

        Args:
            market: If 'kospi', return only 6-digit numeric tickers.
                    If 'us', return only alphabetic tickers.
                    Empty string returns all.
        """
        tenant = self._tenant_token(tenant_id)
        sql = f"""
        SELECT ticker
        FROM (
          SELECT ticker,
                 SUM(CASE WHEN side = 'BUY' THEN filled_qty ELSE -filled_qty END) AS net_qty
          FROM `{self.session.dataset_fqn}.execution_reports`
          WHERE tenant_id = @tenant_id
            AND status IN ('FILLED', 'SIMULATED')
          GROUP BY ticker
        )
        WHERE net_qty > 0
        """
        try:
            rows = self.session.fetch_rows(sql, {"tenant_id": tenant})
        except Exception as exc:
            logger.warning("[yellow]get_all_held_tickers failed[/yellow] err=%s", str(exc))
            return []
        tickers = [str(r["ticker"]).strip().upper() for r in rows if r.get("ticker")]
        m = market.lower().strip()
        if m == "kospi":
            return [t for t in tickers if t.isdigit() and len(t) == 6]
        if m == "us":
            return [t for t in tickers if t and not t[:1].isdigit()]
        return tickers

    def get_latest_position_tickers(
        self,
        *,
        tenant_id: str | None = None,
        market: str = "",
        all_tenants: bool = False,
    ) -> list[str]:
        """Returns distinct tickers from the latest account snapshot positions.

        This differs from ``get_all_held_tickers`` which infers holdings from
        execution reports. Forecast construction should prefer current positions
        from ``positions_current`` so stale fill history does not silently drop
        live holdings.
        """
        params: dict[str, Any] = {}
        if all_tenants:
            sql = f"""
            WITH latest AS (
              SELECT tenant_id, MAX(snapshot_at) AS snapshot_at
              FROM `{self.session.dataset_fqn}.account_snapshots`
              GROUP BY tenant_id
            )
            SELECT DISTINCT p.ticker
            FROM `{self.session.dataset_fqn}.positions_current` p
            JOIN latest l
              ON p.tenant_id = l.tenant_id
             AND p.snapshot_at = l.snapshot_at
            WHERE p.quantity > 0
            """
        else:
            tenant = self._tenant_token(tenant_id)
            params["tenant_id"] = tenant
            sql = f"""
            WITH latest AS (
              SELECT MAX(snapshot_at) AS snapshot_at
              FROM `{self.session.dataset_fqn}.account_snapshots`
              WHERE tenant_id = @tenant_id
            )
            SELECT DISTINCT p.ticker
            FROM `{self.session.dataset_fqn}.positions_current` p
            CROSS JOIN latest l
            WHERE p.tenant_id = @tenant_id
              AND p.snapshot_at = l.snapshot_at
              AND p.quantity > 0
            """
        try:
            rows = self.session.fetch_rows(sql, params)
        except Exception as exc:
            logger.warning("[yellow]get_latest_position_tickers failed[/yellow] err=%s", str(exc))
            return []
        tickers = [str(r["ticker"]).strip().upper() for r in rows if r.get("ticker")]
        m = market.lower().strip()
        if m == "kospi":
            return [t for t in tickers if t.isdigit() and len(t) == 6]
        if m == "us":
            return [t for t in tickers if t and not t[:1].isdigit()]
        return tickers

    def latest_agent_sleeves(
        self,
        *,
        agent_ids: list[str],
        tenant_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Returns latest sleeve configs per agent_id."""
        tenant = self._tenant_token(tenant_id)
        tokens = [str(a).strip() for a in agent_ids if str(a).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        sql = f"""
        SELECT agent_id, initialized_at, initial_cash_krw, initial_positions_json
        FROM (
          SELECT agent_id, initialized_at, initial_cash_krw, initial_positions_json,
                 ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY initialized_at DESC) AS rn
          FROM `{self.session.dataset_fqn}.agent_sleeves`
          WHERE tenant_id = @tenant_id
            AND agent_id IN UNNEST(@agent_ids)
        )
        WHERE rn = 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant, "agent_ids": tokens})
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            aid = str(r.get("agent_id", "")).strip()
            if aid:
                out[aid] = r
        return out

    @staticmethod
    def _truthy_env(name: str) -> bool:
        return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _as_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_seed_positions_payload(value: Any) -> tuple[list[dict[str, Any]], str | None]:
        parsed = value
        if isinstance(parsed, str):
            raw = parsed.strip() or "[]"
            try:
                parsed = json.loads(raw)
            except Exception as exc:
                return [], str(exc)
        if isinstance(parsed, dict):
            parsed = list(parsed.values())
        if not isinstance(parsed, list):
            return [], None
        return [item for item in parsed if isinstance(item, dict)], None

    @staticmethod
    def _parse_seed_positions_json(value: Any) -> list[dict[str, Any]]:
        """Parses seed positions into a normalized list payload."""
        parsed, _ = SleeveStore._parse_seed_positions_payload(value)
        return parsed

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
    def _signed_capital_amount(event_type: Any, amount_krw: Any) -> float:
        """Converts capital event type + amount into a signed KRW delta."""
        try:
            amount = float(amount_krw or 0.0)
        except (TypeError, ValueError):
            amount = 0.0
        token = str(event_type or "").strip().upper()
        if token in {"WITHDRAWAL", "DEBIT", "REDUCTION"}:
            return -abs(amount)
        if token in {"INJECTION", "CREDIT", "DEPOSIT"}:
            return abs(amount)
        return amount

    @staticmethod
    def _is_cash_transfer_event(row: dict[str, Any]) -> bool:
        """Returns True when a transfer row represents a cash-only capital move."""
        token = str(row.get("transfer_type") or "").strip().upper()
        return token in {"CASH", "CASH_TRANSFER", "CASH_ONLY", "WITHDRAWAL", "DEPOSIT"} or not str(row.get("ticker") or "").strip()

    @staticmethod
    def _signed_transfer_cash_amount(agent_id: str, row: dict[str, Any]) -> float:
        """Returns the signed cash delta for one transfer event from one agent's perspective."""
        agent = str(agent_id or "").strip()
        from_agent = str(row.get("from_agent_id") or "").strip()
        to_agent = str(row.get("to_agent_id") or "").strip()
        if not agent or agent not in {from_agent, to_agent}:
            return 0.0

        try:
            qty = float(row.get("quantity") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            price = float(row.get("price_krw") or 0.0)
        except (TypeError, ValueError):
            price = 0.0
        try:
            amount = float(row.get("amount_krw") or 0.0)
        except (TypeError, ValueError):
            amount = 0.0
        if abs(amount) <= 1e-9 and qty > 0 and price > 0:
            amount = qty * price
        if abs(amount) <= 1e-9:
            return 0.0

        is_cash_transfer = SleeveStore._is_cash_transfer_event(row)
        if is_cash_transfer:
            if agent == from_agent:
                return -abs(amount)
            if agent == to_agent:
                return abs(amount)
            return 0.0

        if agent == from_agent:
            return abs(amount)
        if agent == to_agent:
            return -abs(amount)
        return 0.0

    def _apply_transfer_event(
        self,
        *,
        agent_id: str,
        row: dict[str, Any],
        positions: dict[str, Position],
        cash: float,
        opened_at: dict[str, datetime | None],
        default_ts: datetime,
    ) -> float:
        """Applies one agent transfer event to sleeve state."""
        ticker = str(row.get("ticker") or "").strip().upper()
        from_agent = str(row.get("from_agent_id") or "").strip()
        to_agent = str(row.get("to_agent_id") or "").strip()
        transfer_ts = self._as_datetime(row.get("occurred_at")) or default_ts
        try:
            qty = float(row.get("quantity") or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        try:
            price = float(row.get("price_krw") or 0.0)
        except (TypeError, ValueError):
            price = 0.0

        cash += self._signed_transfer_cash_amount(agent_id, row)

        if not ticker or qty <= 0:
            return cash

        pos = positions.get(ticker)
        if agent_id == from_agent:
            if not pos or pos.quantity <= 0:
                return cash
            qty_out = min(float(pos.quantity), qty)
            pos.quantity = max(0.0, float(pos.quantity) - qty_out)
            if price > 0:
                pos.market_price_krw = price
            if pos.quantity == 0.0:
                positions.pop(ticker, None)
                opened_at.pop(ticker, None)
            return cash

        if agent_id != to_agent:
            return cash

        if pos:
            old_qty = float(pos.quantity)
            new_qty = old_qty + qty
            if price > 0:
                old_cost = float(pos.avg_price_krw or 0.0) * old_qty
                pos.avg_price_krw = ((old_cost + price * qty) / new_qty) if new_qty > 0 else float(pos.avg_price_krw or 0.0)
                pos.market_price_krw = price
            pos.quantity = new_qty
        else:
            positions[ticker] = Position(
                ticker=ticker,
                exchange_code=str(row.get("exchange_code") or ""),
                instrument_id=str(row.get("instrument_id") or ""),
                quantity=qty,
                avg_price_krw=max(price, 0.0),
                market_price_krw=max(price, 0.0),
            )
            opened_at[ticker] = transfer_ts
        opened_at.setdefault(ticker, transfer_ts)
        return cash

    def _checkpoints_as_of(
        self,
        *,
        agent_id: str,
        tenant_id: str,
        as_of_ts: datetime,
    ) -> dict[str, dict[str, Any]]:
        """Returns the latest checkpoint for *agent_id* at or before *as_of_ts*."""
        sql = f"""
        SELECT agent_id, event_id, checkpoint_at, cash_krw, positions_json, source, created_by, detail_json
        FROM (
          SELECT
            agent_id, event_id, checkpoint_at, cash_krw, positions_json, source, created_by, detail_json,
            ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY checkpoint_at DESC, event_id DESC) AS rn
          FROM `{self.session.dataset_fqn}.agent_state_checkpoints`
          WHERE tenant_id = @tenant_id
            AND agent_id = @agent_id
            AND checkpoint_at <= @as_of_ts
        )
        WHERE rn = 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant_id, "agent_id": agent_id, "as_of_ts": as_of_ts})
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            aid = str(row.get("agent_id") or "").strip()
            if aid:
                out[aid] = row
        return out

    def _sleeves_as_of(
        self,
        *,
        agent_id: str,
        tenant_id: str,
        as_of_ts: datetime,
    ) -> dict[str, dict[str, Any]]:
        """Returns the latest sleeve row for *agent_id* at or before *as_of_ts*."""
        sql = f"""
        SELECT agent_id, initialized_at, initial_cash_krw, initial_positions_json
        FROM (
          SELECT agent_id, initialized_at, initial_cash_krw, initial_positions_json,
                 ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY initialized_at DESC) AS rn
          FROM `{self.session.dataset_fqn}.agent_sleeves`
          WHERE tenant_id = @tenant_id
            AND agent_id = @agent_id
            AND initialized_at <= @as_of_ts
        )
        WHERE rn = 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant_id, "agent_id": agent_id, "as_of_ts": as_of_ts})
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            aid = str(row.get("agent_id") or "").strip()
            if aid:
                out[aid] = row
        return out

    def _load_agent_seed_state(
        self,
        *,
        agent_id: str,
        tenant_id: str | None = None,
        as_of_ts: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Loads the latest agent seed from checkpoint first, then legacy sleeve row.

        When *as_of_ts* is given, only checkpoints at or before that timestamp
        are considered — this allows historical NAV recomputation.
        """
        tenant = self._tenant_token(tenant_id)
        agent = str(agent_id).strip()
        if not agent:
            return None

        checkpoint_loader = getattr(self._ledger, "latest_agent_state_checkpoints", None) if self._ledger else None
        if callable(checkpoint_loader):
            if as_of_ts is not None:
                checkpoint_cfgs = self._checkpoints_as_of(agent_id=agent, tenant_id=tenant, as_of_ts=as_of_ts)
            else:
                checkpoint_cfgs = checkpoint_loader(agent_ids=[agent], tenant_id=tenant)
            checkpoint = checkpoint_cfgs.get(agent)
            checkpoint_at = self._as_datetime((checkpoint or {}).get("checkpoint_at"))
            if checkpoint and checkpoint_at is not None:
                positions_payload, error = self._parse_seed_positions_payload(checkpoint.get("positions_json"))
                if error is None:
                    return {
                        "source": str(checkpoint.get("source") or "agent_state_checkpoint"),
                        "since": checkpoint_at,
                        "cash_krw": self._safe_float(checkpoint.get("cash_krw")),
                        "positions_payload": positions_payload,
                        "positions_error": None,
                    }
                logger.warning(
                    "[yellow]Agent checkpoint seed parse failed; falling back to sleeve[/yellow] agent=%s err=%s",
                    agent,
                    error,
                )

        if as_of_ts is not None:
            cfgs = self._sleeves_as_of(agent_id=agent, tenant_id=tenant, as_of_ts=as_of_ts)
        else:
            cfgs = self.latest_agent_sleeves(agent_ids=[agent], tenant_id=tenant)
        cfg = cfgs.get(agent)
        if not cfg:
            return None
        initialized_at = self._as_datetime(cfg.get("initialized_at"))
        if initialized_at is None:
            initialized_at = utc_now()
        positions_payload, error = self._parse_seed_positions_payload(cfg.get("initial_positions_json"))
        return {
            "source": "agent_sleeves",
            "since": initialized_at,
            "cash_krw": self._safe_float(cfg.get("initial_cash_krw")),
            "positions_payload": positions_payload,
            "positions_error": error,
        }

    def _first_agent_state_checkpoint(
        self,
        *,
        agent_id: str,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        """Returns the earliest checkpoint row for one agent."""
        sql = f"""
        SELECT agent_id, event_id, checkpoint_at, cash_krw, positions_json, source, created_by, detail_json
        FROM `{self.session.dataset_fqn}.agent_state_checkpoints`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
        ORDER BY checkpoint_at ASC, event_id ASC
        LIMIT 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant_id, "agent_id": str(agent_id).strip()})
        return rows[0] if rows else None

    def _first_agent_sleeve(
        self,
        *,
        agent_id: str,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        """Returns the earliest legacy sleeve row for one agent."""
        sql = f"""
        SELECT agent_id, initialized_at, initial_cash_krw, initial_positions_json
        FROM `{self.session.dataset_fqn}.agent_sleeves`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
        ORDER BY initialized_at ASC
        LIMIT 1
        """
        rows = self.session.fetch_rows(sql, {"tenant_id": tenant_id, "agent_id": str(agent_id).strip()})
        return rows[0] if rows else None

    @staticmethod
    def _seed_basis_components(seed_state: dict[str, Any] | None) -> tuple[float, float]:
        """Returns (seed_cash, seed_positions_cost) from one seed payload."""
        if not isinstance(seed_state, dict):
            return 0.0, 0.0
        seed_cash = SleeveStore._safe_float(seed_state.get("cash_krw"))
        seed_positions_cost = 0.0
        for item in list(seed_state.get("positions_payload") or []):
            if not isinstance(item, dict):
                continue
            try:
                qty = float(item.get("quantity") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            try:
                avg = float(item.get("avg_price_krw") or 0.0)
            except (TypeError, ValueError):
                avg = 0.0
            if qty <= 0 or avg <= 0:
                continue
            seed_positions_cost += qty * avg
        return seed_cash, seed_positions_cost

    @staticmethod
    def _normalize_detail_json(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _to_kst_date_token(value: Any) -> str:
        dt = SleeveStore._as_datetime(value)
        if dt is None:
            if isinstance(value, date):
                return value.isoformat()
            token = str(value or "").strip()
            return token[:10] if len(token) >= 10 else token
        return dt.date().isoformat()

    def _load_agent_origin_state(
        self,
        *,
        agent_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Loads the lineage origin used for actual invested-capital tracing.

        Prefer the earliest checkpoint when one exists. This keeps the current
        active sleeve lineage separate from older experimental sleeve rows.
        """
        tenant = self._tenant_token(tenant_id)
        agent = str(agent_id).strip()
        if not agent:
            return None

        checkpoint = self._first_agent_state_checkpoint(agent_id=agent, tenant_id=tenant)
        checkpoint_at = self._as_datetime((checkpoint or {}).get("checkpoint_at"))
        if checkpoint and checkpoint_at is not None:
            positions_payload, error = self._parse_seed_positions_payload(checkpoint.get("positions_json"))
            if error is None:
                return {
                    "source": str(checkpoint.get("source") or "agent_state_checkpoint"),
                    "since": checkpoint_at,
                    "cash_krw": self._safe_float(checkpoint.get("cash_krw")),
                    "positions_payload": positions_payload,
                    "positions_error": None,
                    "detail_json": self._normalize_detail_json(checkpoint.get("detail_json")),
                }
            logger.warning(
                "[yellow]Agent origin checkpoint parse failed; falling back to first sleeve[/yellow] agent=%s err=%s",
                agent,
                error,
            )

        cfg = self._first_agent_sleeve(agent_id=agent, tenant_id=tenant)
        if not cfg:
            return None
        initialized_at = self._as_datetime(cfg.get("initialized_at"))
        if initialized_at is None:
            initialized_at = utc_now()
        positions_payload, error = self._parse_seed_positions_payload(cfg.get("initial_positions_json"))
        return {
            "source": "agent_sleeves",
            "since": initialized_at,
            "cash_krw": self._safe_float(cfg.get("initial_cash_krw")),
            "positions_payload": positions_payload,
            "positions_error": error,
            "detail_json": {},
        }

    def trace_agent_actual_capital_basis(
        self,
        *,
        agent_id: str,
        tenant_id: str | None = None,
        as_of_ts: datetime | None = None,
    ) -> dict[str, Any]:
        """Traces actual invested capital from the lineage origin plus real cash events."""
        tenant = self._tenant_token(tenant_id)
        agent = str(agent_id).strip()
        if not agent:
            return {}

        origin = self._load_agent_origin_state(agent_id=agent, tenant_id=tenant)
        if not origin:
            return {}

        origin_at = self._as_datetime(origin.get("since")) or utc_now()
        seed_cash, seed_positions_cost = self._seed_basis_components(origin)
        basis = seed_cash + seed_positions_cost

        detail = self._normalize_detail_json(origin.get("detail_json"))
        if str(origin.get("source") or "").strip() == "capital_events.retarget":
            basis_before = detail.get("baseline_equity_krw_before_adjustment")
            flow = detail.get("capital_flow_krw")
            if basis_before is not None and flow is not None:
                basis = self._safe_float(basis_before) + self._safe_float(flow)

        capital_flow_krw = 0.0
        capital_event_count = 0
        transfer_equity_krw = 0.0
        transfer_event_count = 0
        manual_cash_adjustment_krw = 0.0
        manual_cash_adjustment_count = 0
        basis_events: list[dict[str, Any]] = []

        cap_loader = getattr(self._ledger, "capital_events_since", None) if self._ledger else None
        if callable(cap_loader):
            for event in cap_loader(agent_id=agent, since=origin_at, tenant_id=tenant):
                event_ts = self._as_datetime(event.get("occurred_at")) or origin_at
                if as_of_ts is not None and event_ts > as_of_ts:
                    continue
                delta = self._signed_capital_amount(event.get("event_type"), event.get("amount_krw"))
                if abs(delta) <= 1e-9:
                    continue
                capital_flow_krw += delta
                capital_event_count += 1
                basis_events.append(
                    {
                        "occurred_at": event_ts,
                        "type": "capital",
                        "event_type": str(event.get("event_type") or ""),
                        "amount_krw": float(delta),
                        "reason": str(event.get("reason") or ""),
                    }
                )

        manual_loader = getattr(self._ledger, "manual_cash_adjustments_since", None) if self._ledger else None
        if callable(manual_loader):
            for row in manual_loader(agent_id=agent, since=origin_at, tenant_id=tenant):
                event_ts = self._as_datetime(row.get("occurred_at")) or origin_at
                if as_of_ts is not None and event_ts > as_of_ts:
                    continue
                delta_cash = self._safe_float(row.get("delta_cash_krw"))
                if abs(delta_cash) <= 1e-9:
                    continue
                manual_cash_adjustment_krw += delta_cash
                manual_cash_adjustment_count += 1
                basis_events.append(
                    {
                        "occurred_at": event_ts,
                        "type": "manual_adjustment",
                        "event_type": "MANUAL_ADJUSTMENT",
                        "amount_krw": float(delta_cash),
                        "reason": str(row.get("reason") or ""),
                    }
                )

        transfer_loader = getattr(self._ledger, "agent_transfer_events_since", None) if self._ledger else None
        if callable(transfer_loader):
            for row in transfer_loader(agent_id=agent, since=origin_at, tenant_id=tenant):
                event_ts = self._as_datetime(row.get("occurred_at")) or origin_at
                if as_of_ts is not None and event_ts > as_of_ts:
                    continue
                if not self._is_cash_transfer_event(row):
                    continue
                delta_cash = self._signed_transfer_cash_amount(agent, row)
                if abs(delta_cash) <= 1e-9:
                    continue
                transfer_equity_krw += delta_cash
                transfer_event_count += 1
                basis_events.append(
                    {
                        "occurred_at": event_ts,
                        "type": "transfer",
                        "event_type": str(row.get("transfer_type") or "CASH_TRANSFER"),
                        "amount_krw": float(delta_cash),
                        "reason": str(row.get("reason") or ""),
                        "from_agent_id": str(row.get("from_agent_id") or ""),
                        "to_agent_id": str(row.get("to_agent_id") or ""),
                    }
                )

        basis_events.sort(
            key=lambda item: (
                self._as_datetime(item.get("occurred_at")) or origin_at,
                str(item.get("type") or ""),
                str(item.get("event_type") or ""),
            )
        )
        for item in basis_events:
            basis += self._safe_float(item.get("amount_krw"))

        return {
            "origin_at": origin_at,
            "origin_source": str(origin.get("source") or ""),
            "seed_cash_krw": float(seed_cash),
            "seed_positions_cost_krw": float(seed_positions_cost),
            "baseline_equity_krw": float(basis),
            "capital_flow_krw": float(capital_flow_krw),
            "capital_event_count": int(capital_event_count),
            "transfer_equity_krw": float(transfer_equity_krw),
            "transfer_event_count": int(transfer_event_count),
            "manual_cash_adjustment_krw": float(manual_cash_adjustment_krw),
            "manual_cash_adjustment_count": int(manual_cash_adjustment_count),
            "events": basis_events,
        }

    def actual_capital_basis_by_date(
        self,
        *,
        agent_id: str,
        nav_dates: list[Any],
        tenant_id: str | None = None,
    ) -> dict[str, float | None]:
        """Returns actual invested capital by KST nav_date token."""
        tokens = sorted({self._to_kst_date_token(value) for value in nav_dates if self._to_kst_date_token(value)})
        if not tokens:
            return {}
        trace = self.trace_agent_actual_capital_basis(
            agent_id=agent_id,
            tenant_id=tenant_id,
        )
        if not trace:
            return {}

        origin_at = self._as_datetime(trace.get("origin_at"))
        origin_token = self._to_kst_date_token(origin_at)
        running = self._safe_float(trace.get("seed_cash_krw")) + self._safe_float(trace.get("seed_positions_cost_krw"))
        deltas_by_date: dict[str, float] = defaultdict(float)
        for event in list(trace.get("events") or []):
            token = self._to_kst_date_token(event.get("occurred_at"))
            if token:
                deltas_by_date[token] += self._safe_float(event.get("amount_krw"))

        out: dict[str, float | None] = {}
        delta_dates = sorted(deltas_by_date.keys())
        delta_idx = 0
        for token in tokens:
            if origin_token and token < origin_token:
                out[token] = None
                continue
            while delta_idx < len(delta_dates) and delta_dates[delta_idx] <= token:
                running += deltas_by_date[delta_dates[delta_idx]]
                delta_idx += 1
            out[token] = float(running)
        return out

    def fetch_actual_agent_nav_history(
        self,
        *,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        agent_ids: list[str] | None = None,
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        """Fetches NAV rows with actual invested-capital PnL fields overlaid."""
        rows = self.fetch_agent_nav_history(
            tenant_id=tenant_id,
            agent_id=agent_id,
            agent_ids=agent_ids,
            limit=limit,
        )
        if not rows:
            return []

        tenant = self._tenant_token(tenant_id)
        rows_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            agent = str(row.get("agent_id") or "").strip()
            if agent:
                rows_by_agent[agent].append(row)

        basis_maps: dict[str, dict[str, float | None]] = {}
        for agent, agent_rows in rows_by_agent.items():
            basis_maps[agent] = self.actual_capital_basis_by_date(
                agent_id=agent,
                nav_dates=[row.get("nav_date") for row in agent_rows],
                tenant_id=tenant,
            )

        adjusted: list[dict[str, Any]] = []
        for row in rows:
            agent = str(row.get("agent_id") or "").strip()
            token = self._to_kst_date_token(row.get("nav_date"))
            basis = (basis_maps.get(agent) or {}).get(token)
            cloned = dict(row)
            nav = self._safe_float(row.get("nav_krw"), 0.0)
            if basis is not None and basis > 0 and nav > 0:
                pnl = nav - float(basis)
                cloned["baseline_equity_krw"] = float(basis)
                cloned["pnl_krw"] = float(pnl)
                cloned["pnl_ratio"] = float(pnl / float(basis))
            adjusted.append(cloned)
        return adjusted

    def _append_agent_state_checkpoints_from_seed_payloads(
        self,
        *,
        payloads: list[dict[str, Any]],
        source: str,
        tenant_id: str | None = None,
        created_by: str = "system",
        detail_by_agent: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Mirrors a sleeve seed write into append-only checkpoint rows when available."""
        append_checkpoints = getattr(self._ledger, "append_agent_state_checkpoints", None) if self._ledger else None
        if not callable(append_checkpoints):
            return

        def _as_dt(value: Any) -> datetime | None:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str) and value.strip():
                try:
                    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
                except ValueError:
                    return None
            return None

        rows: list[dict[str, Any]] = []
        for payload in payloads:
            agent_id = str(payload.get("agent_id") or "").strip()
            if not agent_id:
                continue
            checkpoint_at = _as_dt(payload.get("initialized_at")) or utc_now()
            positions_payload = self._parse_seed_positions_json(payload.get("initial_positions_json"))
            cash_krw = self._safe_float(payload.get("initial_cash_krw"))
            rows.append(
                {
                    "event_id": self._checkpoint_event_id(
                        agent_id=agent_id,
                        checkpoint_at=checkpoint_at,
                        cash_krw=cash_krw,
                        positions_payload=positions_payload,
                        source=source,
                    ),
                    "checkpoint_at": checkpoint_at,
                    "agent_id": agent_id,
                    "cash_krw": cash_krw,
                    "positions_json": positions_payload,
                    "source": str(source or "").strip() or "unknown",
                    "created_by": str(created_by or "").strip() or "system",
                    "detail_json": (detail_by_agent or {}).get(agent_id),
                }
            )

        if rows:
            append_checkpoints(rows, tenant_id=tenant_id)

    def ensure_agent_state_checkpoints(
        self,
        *,
        agent_ids: list[str],
        total_cash_krw: float,
        capital_per_agent: dict[str, float] | None = None,
        checkpoint_at: datetime | None = None,
        tenant_id: str | None = None,
        excluded_tickers: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Ensures each agent has a checkpoint seed without requiring legacy sleeve rows."""
        tenant = self._tenant_token(tenant_id)
        tokens = [str(a).strip() for a in agent_ids if str(a).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        checkpoint_loader = getattr(self._ledger, "latest_agent_state_checkpoints", None) if self._ledger else None
        if not callable(checkpoint_loader):
            raise RuntimeError("latest_agent_state_checkpoints is not available on this repository")
        existing = checkpoint_loader(agent_ids=tokens, tenant_id=tenant)
        missing = [a for a in tokens if a not in existing]
        if not missing:
            return existing

        per_agent, initial_positions_json, seed_source, seed_positions = self._sleeve_seed_config(
            tenant_id=tenant,
            agent_count=len(tokens),
            total_cash_krw=total_cash_krw,
            excluded_tickers=excluded_tickers,
        )
        ts = checkpoint_at or utc_now()
        payloads = [
            {
                "tenant_id": tenant,
                "agent_id": a,
                "initialized_at": ts.isoformat(),
                "initial_cash_krw": float(capital_per_agent.get(a, per_agent)) if capital_per_agent else per_agent,
                "initial_positions_json": initial_positions_json,
            }
            for a in missing
        ]
        self._append_agent_state_checkpoints_from_seed_payloads(
            payloads=payloads,
            source="agent_state_checkpoints.ensure",
            tenant_id=tenant,
            detail_by_agent={
                agent_id: {
                    "seed_source": seed_source,
                    "seed_positions": seed_positions,
                    "mode": "ensure",
                }
                for agent_id in missing
            },
        )
        logger.info(
            "[green]Agent checkpoints initialized[/green] agents=%s per_agent_cash=%.0f seed=%s seed_positions=%d",
            ",".join(missing),
            per_agent,
            seed_source,
            seed_positions,
        )
        return checkpoint_loader(agent_ids=tokens, tenant_id=tenant)

    def reinitialize_agent_state_checkpoints(
        self,
        *,
        agent_ids: list[str],
        total_cash_krw: float,
        checkpoint_at: datetime | None = None,
        tenant_id: str | None = None,
        excluded_tickers: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Appends fresh checkpoint seeds for all agents without rewriting legacy sleeve rows."""
        tenant = self._tenant_token(tenant_id)
        tokens = [str(a).strip() for a in agent_ids if str(a).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        checkpoint_loader = getattr(self._ledger, "latest_agent_state_checkpoints", None) if self._ledger else None
        if not callable(checkpoint_loader):
            raise RuntimeError("latest_agent_state_checkpoints is not available on this repository")

        per_agent, initial_positions_json, seed_source, seed_positions = self._sleeve_seed_config(
            tenant_id=tenant,
            agent_count=len(tokens),
            total_cash_krw=total_cash_krw,
            excluded_tickers=excluded_tickers,
        )
        ts = checkpoint_at or utc_now()
        payloads = [
            {
                "tenant_id": tenant,
                "agent_id": a,
                "initialized_at": ts.isoformat(),
                "initial_cash_krw": per_agent,
                "initial_positions_json": initial_positions_json,
            }
            for a in tokens
        ]
        self._append_agent_state_checkpoints_from_seed_payloads(
            payloads=payloads,
            source="agent_state_checkpoints.reinitialize",
            tenant_id=tenant,
            detail_by_agent={
                agent_id: {
                    "seed_source": seed_source,
                    "seed_positions": seed_positions,
                    "mode": "reinitialize",
                }
                for agent_id in tokens
            },
        )
        logger.warning(
            "[yellow]Agent checkpoints reinitialized[/yellow] agents=%s per_agent_cash=%.0f seed=%s seed_positions=%d",
            ",".join(tokens),
            per_agent,
            seed_source,
            seed_positions,
        )
        return checkpoint_loader(agent_ids=tokens, tenant_id=tenant)

    def _sleeve_seed_config(
        self,
        *,
        tenant_id: str,
        agent_count: int,
        total_cash_krw: float,
        excluded_tickers: list[str] | None = None,
    ) -> tuple[float, str, str, int]:
        """Builds initial sleeve seed (cash + positions) for first-time/reinit rows."""
        n = max(int(agent_count), 1)
        default_per_agent_cash = float(total_cash_krw) / float(n) if float(total_cash_krw) > 0 else 0.0
        excluded = {
            str(token or "").strip().upper()
            for token in (excluded_tickers or [])
            if str(token or "").strip()
        }

        # Keep legacy behavior unless explicitly enabled for live migration/bootstrap.
        if not self._truthy_env("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT"):
            return default_per_agent_cash, "[]", "virtual_cash", 0

        try:
            snapshot = self.latest_account_snapshot(tenant_id=tenant_id)
        except Exception as exc:
            logger.warning(
                "[yellow]Sleeve seed from account snapshot skipped[/yellow] tenant=%s err=%s",
                tenant_id,
                str(exc),
            )
            return default_per_agent_cash, "[]", "virtual_cash", 0

        if snapshot is None:
            return default_per_agent_cash, "[]", "virtual_cash", 0

        seed_rows: list[dict[str, Any]] = []
        for pos in sorted(snapshot.positions.values(), key=lambda p: str(p.ticker or "")):
            ticker = str(pos.ticker or "").strip().upper()
            if ticker in excluded:
                continue
            try:
                qty = float(pos.quantity) / float(n)
                avg = float(pos.avg_price_krw)
            except (TypeError, ValueError):
                continue
            if qty <= 0 or avg <= 0:
                continue
            seed_row: dict[str, Any] = {
                "ticker": ticker,
                "exchange_code": str(pos.exchange_code or ""),
                "instrument_id": str(pos.instrument_id or ""),
                "quantity": qty,
                "avg_price_krw": avg,
            }
            if pos.avg_price_native is not None:
                seed_row["avg_price_native"] = pos.avg_price_native
            if pos.quote_currency:
                seed_row["quote_currency"] = pos.quote_currency
            if pos.fx_rate > 0:
                seed_row["fx_rate"] = pos.fx_rate
            seed_rows.append(seed_row)

        per_agent_cash = max(float(snapshot.cash_krw), 0.0) / float(n)
        initial_positions_json = json.dumps(seed_rows, ensure_ascii=False, separators=(",", ":")) if seed_rows else "[]"
        return per_agent_cash, initial_positions_json, "account_snapshot", len(seed_rows)

    def ensure_agent_sleeves(
        self,
        *,
        agent_ids: list[str],
        total_cash_krw: float,
        capital_per_agent: dict[str, float] | None = None,
        initialized_at: datetime | None = None,
        tenant_id: str | None = None,
        excluded_tickers: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Ensures each agent has a sleeve config row; initializes missing ones.

        If *capital_per_agent* is provided, each agent gets its specified capital.
        Otherwise falls back to equal distribution of *total_cash_krw*.
        """
        tenant = self._tenant_token(tenant_id)
        tokens = [str(a).strip() for a in agent_ids if str(a).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        existing = self.latest_agent_sleeves(agent_ids=tokens, tenant_id=tenant)
        missing = [a for a in tokens if a not in existing]
        if not missing:
            return existing

        per_agent, initial_positions_json, seed_source, seed_positions = self._sleeve_seed_config(
            tenant_id=tenant,
            agent_count=len(tokens),
            total_cash_krw=total_cash_krw,
            excluded_tickers=excluded_tickers,
        )
        ts = initialized_at or utc_now()

        table_id = f"{self.session.dataset_fqn}.agent_sleeves"
        payloads = [
            {
                "tenant_id": tenant,
                "agent_id": a,
                "initialized_at": ts.isoformat(),
                "initial_cash_krw": float(capital_per_agent.get(a, per_agent)) if capital_per_agent else per_agent,
                "initial_positions_json": initial_positions_json,
            }
            for a in missing
        ]
        errors = self.session.client.insert_rows_json(table_id, payloads)
        if errors:
            raise RuntimeError(f"agent_sleeves insert failed: {errors}")
        self._append_agent_state_checkpoints_from_seed_payloads(
            payloads=payloads,
            source="agent_sleeves.ensure",
            tenant_id=tenant,
            detail_by_agent={
                agent_id: {
                    "seed_source": seed_source,
                    "seed_positions": seed_positions,
                    "mode": "ensure",
                }
                for agent_id in missing
            },
        )

        logger.info(
            "[green]Agent sleeves initialized[/green] agents=%s per_agent_cash=%.0f seed=%s seed_positions=%d",
            ",".join(missing),
            per_agent,
            seed_source,
            seed_positions,
        )
        return self.latest_agent_sleeves(agent_ids=tokens, tenant_id=tenant)

    def reinitialize_agent_sleeves(
        self,
        *,
        agent_ids: list[str],
        total_cash_krw: float,
        initialized_at: datetime | None = None,
        tenant_id: str | None = None,
        excluded_tickers: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Appends a fresh sleeve baseline for all agents (keeps history rows)."""
        tenant = self._tenant_token(tenant_id)
        tokens = [str(a).strip() for a in agent_ids if str(a).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        per_agent, initial_positions_json, seed_source, seed_positions = self._sleeve_seed_config(
            tenant_id=tenant,
            agent_count=len(tokens),
            total_cash_krw=total_cash_krw,
            excluded_tickers=excluded_tickers,
        )
        ts = initialized_at or utc_now()

        table_id = f"{self.session.dataset_fqn}.agent_sleeves"
        payloads = [
            {
                "tenant_id": tenant,
                "agent_id": a,
                "initialized_at": ts.isoformat(),
                "initial_cash_krw": per_agent,
                "initial_positions_json": initial_positions_json,
            }
            for a in tokens
        ]
        errors = self.session.client.insert_rows_json(table_id, payloads)
        if errors:
            raise RuntimeError(f"agent_sleeves reinit failed: {errors}")
        self._append_agent_state_checkpoints_from_seed_payloads(
            payloads=payloads,
            source="agent_sleeves.reinitialize",
            tenant_id=tenant,
            detail_by_agent={
                agent_id: {
                    "seed_source": seed_source,
                    "seed_positions": seed_positions,
                    "mode": "reinitialize",
                }
                for agent_id in tokens
            },
        )

        logger.warning(
            "[yellow]Agent sleeves reinitialized[/yellow] agents=%s per_agent_cash=%.0f seed=%s seed_positions=%d",
            ",".join(tokens),
            per_agent,
            seed_source,
            seed_positions,
        )
        return self.latest_agent_sleeves(agent_ids=tokens, tenant_id=tenant)

    def retarget_agent_capitals_preserve_positions(
        self,
        *,
        agent_ids: list[str],
        target_sleeve_capital_krw: float,
        target_capitals: dict[str, float] | None = None,
        occurred_at: datetime | None = None,
        include_simulated: bool = True,
        sources: list[str] | None = None,
        tenant_id: str | None = None,
        created_by: str = "system",
    ) -> dict[str, dict[str, Any]]:
        """Adjusts sleeve cash via capital events while preserving holdings and cumulative PnL."""
        tenant = self._tenant_token(tenant_id)
        tokens = [str(a).strip() for a in agent_ids if str(a).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        try:
            target = float(target_sleeve_capital_krw)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_sleeve_capital_krw must be numeric") from exc
        if target <= 0:
            raise ValueError("target_sleeve_capital_krw must be > 0")

        append_capital_events = getattr(self._ledger, "append_capital_events", None) if self._ledger else None
        if not callable(append_capital_events):
            raise RuntimeError("append_capital_events is not available on this repository")

        ts = occurred_at or utc_now()
        rows: list[dict[str, Any]] = []
        result: dict[str, dict[str, Any]] = {}
        snapshots: dict[str, AccountSnapshot] = {}

        for aid in tokens:
            snapshot, baseline_equity, _ = self.build_agent_sleeve_snapshot(
                agent_id=aid,
                sources=sources,
                include_simulated=include_simulated,
                tenant_id=tenant,
            )
            positions_value = sum(pos.market_value_krw() for pos in snapshot.positions.values())
            current_cash = float(snapshot.cash_krw)
            current_equity = current_cash + positions_value
            agent_target = float(target_capitals.get(aid, target)) if target_capitals else target
            old_baseline = float(baseline_equity)

            # Preserve existing P&L: inject/withdraw only the capital delta
            # so that new_equity = agent_target + existing_pnl.
            delta_cash = agent_target - old_baseline
            new_cash = current_cash + delta_cash
            if new_cash < 0:
                # Can't withdraw more cash than available — clamp and flag.
                delta_cash = -current_cash
                new_cash = 0.0
                over_target = True
            else:
                over_target = False
            target_cash = new_cash

            result[aid] = {
                "target_sleeve_capital_krw": float(agent_target),
                "effective_target_equity_krw": float(current_equity + delta_cash),
                "positions_value_krw": float(positions_value),
                "current_cash_krw": float(current_cash),
                "target_cash_krw": float(target_cash),
                "capital_flow_krw": float(delta_cash),
                "event_type": "NOOP",
                "over_target": bool(over_target),
                "equity_krw_before_adjustment": float(snapshot.total_equity_krw),
                "baseline_equity_krw_before_adjustment": float(baseline_equity),
            }

            if abs(delta_cash) <= 1e-9:
                continue

            snapshots[aid] = snapshot
            event_type = "INJECTION" if delta_cash > 0 else "WITHDRAWAL"
            rows.append(
                {
                    "event_id": f"cap_{uuid4().hex[:20]}",
                    "occurred_at": ts,
                    "agent_id": aid,
                    "amount_krw": abs(float(delta_cash)),
                    "event_type": event_type,
                    "reason": "retarget_preserve_positions",
                    "created_by": str(created_by or "").strip() or "system",
                }
            )
            result[aid]["event_type"] = event_type

        if rows:
            append_capital_events(rows, tenant_id=tenant)

            # Sync checkpoints so seed state reflects the capital event.
            # Without this, a later checkpoint created from stale sleeve data
            # would push checkpoint_at past the capital event, causing the
            # event to be silently dropped by capital_events_since().
            checkpoint_payloads: list[dict[str, Any]] = []
            for row in rows:
                aid = str(row["agent_id"])
                snap = snapshots.get(aid)
                if snap is None:
                    continue
                meta = result.get(aid, {})
                new_cash = float(meta.get("current_cash_krw", 0.0)) + float(meta.get("capital_flow_krw", 0.0))
                pos_rows: list[dict[str, Any]] = []
                for pos in sorted(snap.positions.values(), key=lambda p: str(p.ticker or "")):
                    if pos.quantity <= 0:
                        continue
                    seed_row: dict[str, Any] = {
                        "ticker": str(pos.ticker or "").strip().upper(),
                        "exchange_code": str(pos.exchange_code or ""),
                        "instrument_id": str(pos.instrument_id or ""),
                        "quantity": pos.quantity,
                        "avg_price_krw": pos.avg_price_krw,
                    }
                    if pos.avg_price_native is not None:
                        seed_row["avg_price_native"] = pos.avg_price_native
                    if pos.quote_currency:
                        seed_row["quote_currency"] = pos.quote_currency
                    if pos.fx_rate > 0:
                        seed_row["fx_rate"] = pos.fx_rate
                    pos_rows.append(seed_row)
                # Place checkpoint 1μs after the capital event so that
                # capital_events_since(>= checkpoint_at) does NOT pick up
                # the event that is already baked into checkpoint cash.
                checkpoint_ts = ts + timedelta(microseconds=1)
                checkpoint_payloads.append({
                    "tenant_id": tenant,
                    "agent_id": aid,
                    "initialized_at": checkpoint_ts.isoformat(),
                    "initial_cash_krw": new_cash,
                    "initial_positions_json": json.dumps(pos_rows, ensure_ascii=False, separators=(",", ":")) if pos_rows else "[]",
                })
            if checkpoint_payloads:
                self._append_agent_state_checkpoints_from_seed_payloads(
                    payloads=checkpoint_payloads,
                    source="capital_events.retarget",
                    tenant_id=tenant,
                    created_by=str(created_by or "").strip() or "system",
                    detail_by_agent={
                        aid: {**result.get(aid, {}), "mode": "capital_retarget"}
                        for aid in [str(r["agent_id"]) for r in rows]
                    },
                )

        logger.info(
            "[green]Agent capitals retargeted[/green] agents=%d target=%.0f events=%d",
            len(tokens),
            target,
            len(rows),
        )
        return result

    def retarget_agent_sleeves_preserve_positions(
        self,
        *,
        agent_ids: list[str],
        target_sleeve_capital_krw: float,
        target_capitals: dict[str, float] | None = None,
        initialized_at: datetime | None = None,
        include_simulated: bool = True,
        sources: list[str] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Re-bases each sleeve to a new target while preserving current holdings.

        The method appends one fresh row per agent to `agent_sleeves`.
        - Holdings are copied from the latest reconstructed sleeve snapshot.
        - If holdings market value exceeds target, cash is clamped to 0 (no forced sell).
        """
        tenant = self._tenant_token(tenant_id)
        tokens = [str(a).strip() for a in agent_ids if str(a).strip()]
        tokens = list(dict.fromkeys(tokens))
        if not tokens:
            return {}

        try:
            target = float(target_sleeve_capital_krw)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_sleeve_capital_krw must be numeric") from exc
        if target <= 0:
            raise ValueError("target_sleeve_capital_krw must be > 0")

        ts = initialized_at or utc_now()
        table_id = f"{self.session.dataset_fqn}.agent_sleeves"
        payloads: list[dict[str, Any]] = []
        result: dict[str, dict[str, Any]] = {}

        for aid in tokens:
            snapshot, _, _ = self.build_agent_sleeve_snapshot(
                agent_id=aid,
                sources=sources,
                include_simulated=include_simulated,
                tenant_id=tenant,
            )
            position_rows: list[dict[str, Any]] = []
            positions_value = 0.0
            for pos in sorted(snapshot.positions.values(), key=lambda p: str(p.ticker or "")):
                try:
                    qty = float(pos.quantity)
                except (TypeError, ValueError):
                    qty = 0.0
                if qty <= 0:
                    continue

                mark_px = 0.0
                try:
                    mark_px = float(pos.market_price_krw or 0.0)
                except (TypeError, ValueError):
                    mark_px = 0.0
                try:
                    avg_px = float(pos.avg_price_krw or 0.0)
                except (TypeError, ValueError):
                    avg_px = 0.0
                seed_px = mark_px if mark_px > 0 else avg_px
                if seed_px <= 0:
                    continue

                positions_value += qty * seed_px
                seed_row: dict[str, Any] = {
                    "ticker": str(pos.ticker or "").strip().upper(),
                    "exchange_code": str(pos.exchange_code or ""),
                    "instrument_id": str(pos.instrument_id or ""),
                    "quantity": qty,
                    "avg_price_krw": seed_px,
                }
                if pos.avg_price_native is not None:
                    seed_row["avg_price_native"] = pos.avg_price_native
                if pos.quote_currency:
                    seed_row["quote_currency"] = pos.quote_currency
                if pos.fx_rate > 0:
                    seed_row["fx_rate"] = pos.fx_rate
                position_rows.append(seed_row)

            agent_target = float(target_capitals.get(aid, target)) if target_capitals else target
            initial_cash = max(agent_target - positions_value, 0.0)
            over_target = positions_value > agent_target
            initial_positions_json = json.dumps(position_rows, ensure_ascii=False, separators=(",", ":")) if position_rows else "[]"

            payloads.append(
                {
                    "tenant_id": tenant,
                    "agent_id": aid,
                    "initialized_at": ts.isoformat(),
                    "initial_cash_krw": initial_cash,
                    "initial_positions_json": initial_positions_json,
                }
            )
            result[aid] = {
                "target_sleeve_capital_krw": float(agent_target),
                "positions_value_krw": float(positions_value),
                "initial_cash_krw": float(initial_cash),
                "over_target": bool(over_target),
                "equity_krw_before_rebase": float(snapshot.total_equity_krw),
            }

        errors = self.session.client.insert_rows_json(table_id, payloads)
        if errors:
            raise RuntimeError(f"agent_sleeves retarget failed: {errors}")
        self._append_agent_state_checkpoints_from_seed_payloads(
            payloads=payloads,
            source="agent_sleeves.retarget",
            tenant_id=tenant,
            detail_by_agent={
                agent_id: {
                    **details,
                    "mode": "retarget",
                }
                for agent_id, details in result.items()
            },
        )

        over_count = sum(1 for item in result.values() if bool(item.get("over_target")))
        logger.info(
            "[green]Agent sleeves retargeted[/green] agents=%d target=%.0f over_target=%d",
            len(tokens),
            target,
            over_count,
        )
        return result

    def build_agent_sleeve_snapshot(
        self,
        *,
        agent_id: str,
        sources: list[str] | None = None,
        include_simulated: bool = True,
        tenant_id: str | None = None,
        as_of_ts: datetime | None = None,
    ) -> tuple[AccountSnapshot, float, dict[str, Any]]:
        """Builds an agent-local sleeve snapshot from seed state + execution/capital history.

        When *as_of_ts* is given, only events up to that timestamp are replayed
        and market prices are resolved as of that date rather than "now".

        Returns (snapshot, baseline_equity_krw, meta).
        """
        tenant = self._tenant_token(tenant_id)
        agent_id = str(agent_id).strip()
        seed_state = self._load_agent_seed_state(agent_id=agent_id, tenant_id=tenant, as_of_ts=as_of_ts)
        if not seed_state:
            empty = AccountSnapshot(cash_krw=0.0, total_equity_krw=0.0, positions={})
            return empty, 0.0, {}

        init_cash = float(seed_state.get("cash_krw") or 0.0)
        since = seed_state.get("since")
        init_opened_at = self._as_datetime(since) or utc_now()
        parsed = list(seed_state.get("positions_payload") or [])
        seed_source = str(seed_state.get("source") or "unknown")
        positions_error = str(seed_state.get("positions_error") or "").strip()
        if positions_error:
            raise RuntimeError(f"invalid initial_positions_json for agent={agent_id}: {positions_error}")

        positions: dict[str, Position] = {}
        opened_at: dict[str, datetime] = {}
        baseline_equity = init_cash
        seed_positions_cost = 0.0
        for item in parsed:
            if not isinstance(item, dict):
                continue
            t = str(item.get("ticker", "")).strip().upper()
            if not t:
                continue
            try:
                qty = float(item.get("quantity") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            try:
                avg = float(item.get("avg_price_krw") or 0.0)
            except (TypeError, ValueError):
                avg = 0.0
            if qty <= 0 or avg <= 0:
                continue
            try:
                seed_native = float(item.get("avg_price_native")) if item.get("avg_price_native") is not None else None
            except (TypeError, ValueError):
                seed_native = None
            seed_ccy = str(item.get("quote_currency") or "")
            try:
                seed_fx = float(item.get("fx_rate") or 0.0)
            except (TypeError, ValueError):
                seed_fx = 0.0

            positions[t] = Position(
                ticker=t,
                exchange_code=str(item.get("exchange_code") or ""),
                instrument_id=str(item.get("instrument_id") or ""),
                quantity=qty,
                avg_price_krw=avg,
                market_price_krw=avg,
                avg_price_native=seed_native,
                market_price_native=seed_native,
                quote_currency=seed_ccy,
                fx_rate=seed_fx,
            )
            opened_at[t] = init_opened_at
            seed_positions_cost += qty * avg
            baseline_equity += qty * avg

        cash = init_cash
        realized_total = 0.0
        realized_by_ticker: dict[str, float] = {}
        sell_trades = 0
        sell_wins = 0
        worst_trade: dict[str, Any] | None = None

        statuses = ["FILLED"]
        if include_simulated:
            statuses.append("SIMULATED")

        cutoff_clause = ""
        cutoff_params: dict[str, Any] = {}
        if as_of_ts is not None:
            cutoff_clause = " AND created_at <= @as_of_ts"
            cutoff_params["as_of_ts"] = as_of_ts

        sql = f"""
        SELECT created_at, ticker, exchange_code, instrument_id, side,
               filled_qty, avg_price_krw, avg_price_native, quote_currency, fx_rate,
               status
        FROM `{self.session.dataset_fqn}.execution_reports`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND created_at >= @since
          AND status IN UNNEST(@statuses)
          {cutoff_clause}
        ORDER BY created_at ASC
        """
        fills = self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "agent_id": agent_id,
                "since": since,
                "statuses": statuses,
                **cutoff_params,
            },
        )

        transfer_rows: list[dict[str, Any]] = []
        agent_transfer_events_since = getattr(self._ledger, "agent_transfer_events_since", None) if self._ledger else None
        if callable(agent_transfer_events_since):
            transfer_rows = agent_transfer_events_since(agent_id=agent_id, since=since, tenant_id=tenant)

        replay_events: list[tuple[datetime, int, dict[str, Any]]] = []
        for r in fills:
            replay_events.append((self._as_datetime(r.get("created_at")) or init_opened_at, 0, {"kind": "fill", "row": r}))
        for row in transfer_rows:
            replay_events.append((self._as_datetime(row.get("occurred_at")) or init_opened_at, 1, {"kind": "transfer", "row": row}))
        replay_events.sort(key=lambda item: (item[0], item[1], str(item[2]["row"].get("event_id") or item[2]["row"].get("ticker") or "")))

        trade_count_total = 0
        transfer_event_count = 0
        transfer_cash_krw = 0.0
        transfer_equity_krw = 0.0
        for _, _, event in replay_events:
            kind = str(event.get("kind") or "")
            r = dict(event.get("row") or {})
            if kind == "transfer":
                transfer_equity_delta = 0.0
                if self._is_cash_transfer_event(r):
                    transfer_equity_delta = self._signed_transfer_cash_amount(agent_id, r)
                    baseline_equity += transfer_equity_delta
                cash_before = float(cash)
                cash = self._apply_transfer_event(
                    agent_id=agent_id,
                    row=r,
                    positions=positions,
                    cash=cash,
                    opened_at=opened_at,
                    default_ts=init_opened_at,
                )
                transfer_cash_krw += float(cash - cash_before)
                transfer_equity_krw += float(transfer_equity_delta)
                transfer_event_count += 1
                continue

            t = str(r.get("ticker", "")).strip().upper()
            side = str(r.get("side", "")).strip().upper()
            ts = self._as_datetime(r.get("created_at"))
            try:
                qty = float(r.get("filled_qty") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            try:
                px = float(r.get("avg_price_krw") or 0.0)
            except (TypeError, ValueError):
                px = 0.0
            if not t or qty <= 0 or px <= 0:
                continue

            # Native currency fields from execution report
            try:
                px_native = float(r.get("avg_price_native")) if r.get("avg_price_native") is not None else None
            except (TypeError, ValueError):
                px_native = None
            fill_ccy = str(r.get("quote_currency") or "").strip().upper()
            try:
                fill_fx = float(r.get("fx_rate") or 0.0)
            except (TypeError, ValueError):
                fill_fx = 0.0

            trade_count_total += 1

            pos = positions.get(t)

            if side == "BUY":
                notional = qty * px
                cash -= notional
                if pos:
                    if pos.quantity <= 0:
                        opened_at[t] = ts or init_opened_at
                    new_qty = pos.quantity + qty
                    new_cost = pos.avg_price_krw * pos.quantity + notional
                    pos.quantity = new_qty
                    pos.avg_price_krw = new_cost / new_qty if new_qty > 0 else pos.avg_price_krw
                    pos.market_price_krw = px
                    # Update native currency cost basis
                    if px_native is not None and pos.avg_price_native is not None:
                        old_native_cost = pos.avg_price_native * (new_qty - qty)
                        pos.avg_price_native = (old_native_cost + px_native * qty) / new_qty if new_qty > 0 else pos.avg_price_native
                    elif px_native is not None:
                        pos.avg_price_native = px_native
                    pos.market_price_native = px_native
                    if fill_ccy:
                        pos.quote_currency = fill_ccy
                    if fill_fx > 0:
                        pos.fx_rate = fill_fx
                else:
                    positions[t] = Position(
                        ticker=t,
                        exchange_code=str(r.get("exchange_code") or ""),
                        instrument_id=str(r.get("instrument_id") or ""),
                        quantity=qty,
                        avg_price_krw=px,
                        market_price_krw=px,
                        avg_price_native=px_native,
                        market_price_native=px_native,
                        quote_currency=fill_ccy,
                        fx_rate=fill_fx,
                    )
                    opened_at[t] = ts or init_opened_at

            elif side == "SELL":
                if not pos or pos.quantity <= 0:
                    continue

                qty_sold = min(float(qty), float(pos.quantity))
                if qty_sold <= 0:
                    continue

                notional = qty_sold * px
                cash += notional

                realized = (px - float(pos.avg_price_krw)) * qty_sold
                realized_total += realized
                realized_by_ticker[t] = realized_by_ticker.get(t, 0.0) + realized

                sell_trades += 1
                if realized > 0:
                    sell_wins += 1

                if worst_trade is None or float(realized) < float(worst_trade.get("pnl_krw") or 0.0):
                    worst_trade = {
                        "ticker": t,
                        "pnl_krw": float(realized),
                        "qty": float(qty_sold),
                        "sell_px_krw": float(px),
                        "cost_basis_krw": float(pos.avg_price_krw),
                        "opened_at": opened_at.get(t),
                        "closed_at": ts,
                    }

                pos.quantity = max(0.0, float(pos.quantity) - qty_sold)
                pos.market_price_krw = px
                if px_native is not None:
                    pos.market_price_native = px_native
                if fill_fx > 0:
                    pos.fx_rate = fill_fx
                if pos.quantity == 0.0:
                    positions.pop(t, None)
                    opened_at.pop(t, None)

        capital_flow_krw = 0.0
        capital_event_count = 0
        capital_events_since = getattr(self._ledger, "capital_events_since", None) if self._ledger else None
        if callable(capital_events_since):
            for event in capital_events_since(agent_id=agent_id, since=init_opened_at, tenant_id=tenant):
                delta = self._signed_capital_amount(event.get("event_type"), event.get("amount_krw"))
                if abs(delta) <= 1e-9:
                    continue
                cash += delta
                baseline_equity += delta
                capital_flow_krw += delta
                capital_event_count += 1

        manual_cash_adjustment_krw = 0.0
        manual_cash_adjustment_count = 0
        manual_cash_adjustments_since = getattr(self._ledger, "manual_cash_adjustments_since", None) if self._ledger else None
        if callable(manual_cash_adjustments_since):
            for row in manual_cash_adjustments_since(agent_id=agent_id, since=init_opened_at, tenant_id=tenant):
                try:
                    delta_cash = float(row.get("delta_cash_krw") or 0.0)
                except (TypeError, ValueError):
                    delta_cash = 0.0
                if abs(delta_cash) <= 1e-9:
                    continue
                cash += delta_cash
                baseline_equity += delta_cash
                manual_cash_adjustment_krw += delta_cash
                manual_cash_adjustment_count += 1

        # --- Dividend credits ---
        dividend_income_krw = 0.0
        dividend_count = 0
        try:
            div_credits = self.get_dividend_credits(
                agent_id=agent_id,
                since=since,
                tenant_id=tenant,
            )
            for dc in div_credits:
                try:
                    amt = float(dc.get("net_amount_krw") or 0.0)
                except (TypeError, ValueError):
                    amt = 0.0
                if amt > 0:
                    dividend_income_krw += amt
                    dividend_count += 1
            cash += dividend_income_krw
        except Exception as exc:
            logger.warning(
                "[yellow]Dividend credit injection skipped[/yellow] agent=%s err=%s",
                agent_id, str(exc),
            )

        if positions:
            price_kwargs: dict[str, Any] = {"tickers": list(positions.keys()), "sources": sources}
            if as_of_ts is not None:
                price_kwargs["as_of_date"] = as_of_ts.date() if hasattr(as_of_ts, "date") else as_of_ts
            price_data = self._market.latest_close_prices_with_currency(**price_kwargs)
            instrument_map: dict[str, dict[str, Any]] = {}
            latest_instrument_map = getattr(self._market, "latest_instrument_map", None) if self._market else None
            if callable(latest_instrument_map):
                try:
                    instrument_map = latest_instrument_map(list(positions.keys()))
                except Exception as exc:
                    logger.warning(
                        "[yellow]Instrument metadata refresh skipped[/yellow] agent=%s err=%s",
                        agent_id,
                        str(exc),
                    )
            for t, pos in positions.items():
                pd = price_data.get(t)
                if pd is None:
                    pd = {}
                px = pd.get("close_price_krw")
                if px is not None and float(px) > 0:
                    pos.market_price_krw = float(px)
                native = pd.get("close_price_native")
                if native is not None:
                    pos.market_price_native = float(native)
                ccy = pd.get("quote_currency")
                if ccy:
                    pos.quote_currency = str(ccy)
                fx = pd.get("fx_rate_used")
                if fx and float(fx) > 0:
                    pos.fx_rate = float(fx)

                instrument_row = instrument_map.get(t) or {}
                exchange_code = str(instrument_row.get("exchange_code") or "").strip().upper()
                instrument_id = str(instrument_row.get("instrument_id") or "").strip()
                if exchange_code:
                    pos.exchange_code = exchange_code
                elif instrument_id and ":" in instrument_id and not pos.exchange_code:
                    pos.exchange_code = instrument_id.split(":", 1)[0].strip().upper()
                if instrument_id:
                    pos.instrument_id = instrument_id

        positions_market_value = sum(p.market_value_krw() for p in positions.values())
        snapshot = AccountSnapshot(
            cash_krw=float(cash),
            total_equity_krw=float(cash) + positions_market_value,
            positions=positions,
        )

        meta: dict[str, Any] = {
            "seed_source": seed_source,
            "initialized_at": init_opened_at,
            "seed_cash_krw": float(init_cash),
            "seed_positions_cost_krw": float(seed_positions_cost),
            "baseline_equity_krw": float(baseline_equity),
            "trade_count_total": int(trade_count_total),
            "transfer_event_count": int(transfer_event_count),
            "transfer_cash_krw": float(transfer_cash_krw),
            "transfer_equity_krw": float(transfer_equity_krw),
            "realized_pnl_krw": float(realized_total),
            "realized_pnl_by_ticker": {k: float(v) for k, v in realized_by_ticker.items()},
            "sell_trade_count": int(sell_trades),
            "sell_win_rate": (float(sell_wins) / float(sell_trades)) if sell_trades > 0 else None,
            "worst_trade": worst_trade,
            "position_opened_at": {k: opened_at[k] for k in positions.keys() if k in opened_at},
            "capital_flow_krw": float(capital_flow_krw),
            "capital_event_count": int(capital_event_count),
            "manual_cash_adjustment_krw": float(manual_cash_adjustment_krw),
            "manual_cash_adjustment_count": int(manual_cash_adjustment_count),
            "dividend_income_krw": float(dividend_income_krw),
            "dividend_count": int(dividend_count),
            "current_cash_krw": float(cash),
            "current_positions_value_krw": float(positions_market_value),
            "fx_source": "market_features_latest.fx_rate_used" if positions else "",
            "valuation_source": ",".join(sources or []) if sources else "market_features_latest",
        }

        return snapshot, float(baseline_equity), meta

    def agent_holdings_at_date(
        self,
        *,
        agent_id: str,
        as_of_date: date,
        include_simulated: bool = True,
        tenant_id: str | None = None,
    ) -> dict[str, float]:
        """Returns ``{ticker: qty}`` held by *agent_id* on *as_of_date*.

        This is a lightweight version of ``build_agent_sleeve_snapshot`` that
        only tracks positions (no cash, PnL, or market-price lookups).  Fills
        are replayed in chronological order from the agent's sleeve
        initialization date up to ``as_of_date`` (inclusive).
        """
        tenant = self._tenant_token(tenant_id)
        agent_id = str(agent_id).strip()
        seed_state = self._load_agent_seed_state(agent_id=agent_id, tenant_id=tenant)
        if not seed_state:
            return {}

        since = seed_state.get("since")

        # Seed positions from sleeve config.
        positions: dict[str, float] = {}
        for item in list(seed_state.get("positions_payload") or []):
            if not isinstance(item, dict):
                continue
            t = str(item.get("ticker", "")).strip().upper()
            try:
                qty = float(item.get("quantity") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if t and qty > 0:
                positions[t] = positions.get(t, 0.0) + qty

        statuses = ["FILLED"]
        if include_simulated:
            statuses.append("SIMULATED")

        sql = f"""
        SELECT ticker, side, filled_qty
        FROM `{self.session.dataset_fqn}.execution_reports`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND created_at >= @since
          AND DATE(created_at) <= @as_of_date
          AND status IN UNNEST(@statuses)
        ORDER BY created_at ASC
        """
        fills = self.session.fetch_rows(
            sql,
            {
                "tenant_id": tenant,
                "agent_id": agent_id,
                "since": since,
                "as_of_date": as_of_date,
                "statuses": statuses,
            },
        )

        transfer_rows: list[dict[str, Any]] = []
        agent_transfer_events_since = getattr(self._ledger, "agent_transfer_events_since", None) if self._ledger else None
        if callable(agent_transfer_events_since):
            transfer_rows = agent_transfer_events_since(agent_id=agent_id, since=since, tenant_id=tenant)

        replay_events: list[tuple[datetime, int, dict[str, Any]]] = []
        for r in fills:
            replay_events.append((self._as_datetime(r.get("created_at")) or since, 0, {"kind": "fill", "row": r}))
        for row in transfer_rows:
            replay_events.append((self._as_datetime(row.get("occurred_at")) or since, 1, {"kind": "transfer", "row": row}))
        replay_events.sort(key=lambda item: (item[0], item[1], str(item[2]["row"].get("event_id") or item[2]["row"].get("ticker") or "")))

        for _, _, event in replay_events:
            row = dict(event.get("row") or {})
            if str(event.get("kind") or "") == "transfer":
                t = str(row.get("ticker", "")).strip().upper()
                try:
                    qty = float(row.get("quantity") or 0.0)
                except (TypeError, ValueError):
                    qty = 0.0
                if not t or qty <= 0:
                    continue
                from_agent = str(row.get("from_agent_id") or "").strip()
                to_agent = str(row.get("to_agent_id") or "").strip()
                if agent_id == from_agent:
                    positions[t] = max(0.0, positions.get(t, 0.0) - qty)
                elif agent_id == to_agent:
                    positions[t] = positions.get(t, 0.0) + qty
                continue

            t = str(row.get("ticker", "")).strip().upper()
            side = str(row.get("side", "")).strip().upper()
            try:
                qty = float(row.get("filled_qty") or 0.0)
            except (TypeError, ValueError):
                qty = 0.0
            if not t or qty <= 0:
                continue
            if side == "BUY":
                positions[t] = positions.get(t, 0.0) + qty
            elif side == "SELL":
                cur = positions.get(t, 0.0)
                positions[t] = max(0.0, cur - qty)

        return {t: q for t, q in positions.items() if q > 0}

    def insert_dividend_events(self, rows: list[dict[str, Any]]) -> None:
        """Batch-inserts dividend event rows into BigQuery."""
        if not rows:
            return
        table_id = f"{self.session.dataset_fqn}.dividend_events"
        errors = self.session.client.insert_rows_json(table_id, rows)
        if errors:
            raise RuntimeError(f"dividend_events insert failed: {errors}")

    def get_dividend_credits(
        self,
        *,
        agent_id: str,
        since: datetime | Any,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns dividend credits for *agent_id* since the sleeve init time."""
        tenant = self._tenant_token(tenant_id)
        sql = f"""
        SELECT event_id, ticker, ex_date, shares_held,
               gross_per_share_usd, net_amount_usd, usd_krw_rate, net_amount_krw
        FROM `{self.session.dataset_fqn}.dividend_events`
        WHERE tenant_id = @tenant_id
          AND agent_id = @agent_id
          AND created_at >= @since
        ORDER BY ex_date ASC
        """
        try:
            return self.session.fetch_rows(sql, {"tenant_id": tenant, "agent_id": agent_id, "since": since})
        except Exception as exc:
            logger.warning("[yellow]get_dividend_credits failed[/yellow] agent=%s err=%s", agent_id, str(exc))
            return []

    def dividend_event_exists(
        self,
        *,
        event_ids: list[str],
        tenant_id: str | None = None,
    ) -> set[str]:
        """Returns the subset of *event_ids* that already exist in BQ."""
        if not event_ids:
            return set()
        tenant = self._tenant_token(tenant_id)
        sql = f"""
        SELECT DISTINCT event_id
        FROM `{self.session.dataset_fqn}.dividend_events`
        WHERE tenant_id = @tenant_id
          AND event_id IN UNNEST(@event_ids)
        """
        try:
            rows = self.session.fetch_rows(sql, {"tenant_id": tenant, "event_ids": event_ids})
            return {str(r["event_id"]) for r in rows if r.get("event_id")}
        except Exception as exc:
            logger.warning("[yellow]dividend_event_exists check failed[/yellow] err=%s", str(exc))
            return set()

    def upsert_agent_nav_daily(
        self,
        *,
        nav_date: date,
        agent_id: str,
        nav_krw: float,
        baseline_equity_krw: float,
        cash_krw: float | None = None,
        market_value_krw: float | None = None,
        capital_flow_krw: float | None = None,
        fx_source: str | None = None,
        valuation_source: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Upserts one daily NAV row for an agent in legacy and official tables."""
        tenant = self._tenant_token(tenant_id)
        agent_id = str(agent_id).strip()
        base = float(baseline_equity_krw)
        nav = float(nav_krw)
        if cash_krw is not None:
            cash = float(cash_krw)
        elif market_value_krw is not None:
            cash = float(nav - float(market_value_krw))
        else:
            cash = 0.0
        if market_value_krw is None:
            market_value = max(nav - cash, 0.0)
        else:
            market_value = max(float(market_value_krw), 0.0)
        capital_flow = float(capital_flow_krw) if capital_flow_krw is not None else 0.0

        pnl = nav - base if base > 0 else 0.0
        pnl_ratio = (pnl / base) if base > 0 else 0.0

        delete_sql = f"""
        DELETE FROM `{self.session.dataset_fqn}.agent_nav_daily`
        WHERE tenant_id = @tenant_id
          AND nav_date = @nav_date
          AND agent_id = @agent_id
        """
        self.session.execute(delete_sql, {"tenant_id": tenant, "nav_date": nav_date, "agent_id": agent_id})

        insert_sql = f"""
        INSERT INTO `{self.session.dataset_fqn}.agent_nav_daily`
        (tenant_id, nav_date, agent_id, nav_krw, pnl_krw, pnl_ratio)
        VALUES
        (@tenant_id, @nav_date, @agent_id, @nav_krw, @pnl_krw, @pnl_ratio)
        """
        self.session.execute(
            insert_sql,
            {
                "tenant_id": tenant,
                "nav_date": nav_date,
                "agent_id": agent_id,
                "nav_krw": nav,
                "pnl_krw": pnl,
                "pnl_ratio": pnl_ratio,
            },
        )

        delete_official_sql = f"""
        DELETE FROM `{self.session.dataset_fqn}.official_nav_daily`
        WHERE tenant_id = @tenant_id
          AND nav_date = @nav_date
          AND agent_id = @agent_id
        """
        self.session.execute(delete_official_sql, {"tenant_id": tenant, "nav_date": nav_date, "agent_id": agent_id})

        insert_official_sql = f"""
        INSERT INTO `{self.session.dataset_fqn}.official_nav_daily`
        (tenant_id, nav_date, agent_id, nav_krw, cash_krw, market_value_krw, capital_flow_krw, pnl_krw, pnl_ratio, fx_source, valuation_source)
        VALUES
        (@tenant_id, @nav_date, @agent_id, @nav_krw, @cash_krw, @market_value_krw, @capital_flow_krw, @pnl_krw, @pnl_ratio, @fx_source, @valuation_source)
        """
        self.session.execute(
            insert_official_sql,
            {
                "tenant_id": tenant,
                "nav_date": nav_date,
                "agent_id": agent_id,
                "nav_krw": nav,
                "cash_krw": cash,
                "market_value_krw": market_value,
                "capital_flow_krw": capital_flow,
                "pnl_krw": pnl,
                "pnl_ratio": pnl_ratio,
                "fx_source": str(fx_source or "").strip() or None,
                "valuation_source": str(valuation_source or "").strip() or "agent_sleeve_snapshot",
            },
        )
