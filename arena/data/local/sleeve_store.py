"""Local DuckDB-backed sleeve/account snapshot helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from typing import Any
from uuid import uuid4

from arena.data.local.session import DuckDBSession
from arena.models import AccountSnapshot, Position, utc_now


class LocalSleeveStore:
    """Minimal sleeve state replay for local paper/demo cycles."""

    def __init__(self, session: DuckDBSession, *, market: Any | None = None) -> None:
        self.session = session
        self._market = market

    def _tenant_token(self, tenant_id: str | None = None) -> str:
        return self.session.resolve_tenant_id(tenant_id)

    @staticmethod
    def _parse_positions(raw: Any) -> list[dict[str, Any]]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return [dict(item) for item in raw if isinstance(item, dict)]
        text = str(raw or "").strip()
        if not text:
            return []
        parsed = json.loads(text)
        return [dict(item) for item in parsed] if isinstance(parsed, list) else []

    @staticmethod
    def _position_to_payload(pos: Position) -> dict[str, Any]:
        return {
            "ticker": pos.ticker,
            "exchange_code": pos.exchange_code,
            "instrument_id": pos.instrument_id,
            "quantity": pos.quantity,
            "avg_price_krw": pos.avg_price_krw,
            "market_price_krw": pos.market_price_krw,
            "avg_price_native": pos.avg_price_native,
            "market_price_native": pos.market_price_native,
            "quote_currency": pos.quote_currency,
            "fx_rate": pos.fx_rate,
        }

    @staticmethod
    def _as_utc_naive(value: Any) -> datetime | None:
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def _signed_capital_amount(event_type: Any, amount_krw: Any) -> float:
        try:
            amount = abs(float(amount_krw or 0.0))
        except (TypeError, ValueError):
            amount = 0.0
        token = str(event_type or "").strip().upper()
        if "WITHDRAW" in token or token in {"OUTFLOW", "DEBIT"}:
            return -amount
        if amount <= 0:
            return 0.0
        return amount

    def latest_agent_sleeves(self, *, agent_ids: list[str] | None = None, tenant_id: str | None = None) -> dict[str, dict[str, Any]]:
        tenant = self._tenant_token(tenant_id)
        params: dict[str, Any] = {"tenant_id": tenant}
        filters = ["tenant_id = $tenant_id"]
        tokens = [str(a or "").strip() for a in (agent_ids or []) if str(a or "").strip()]
        if tokens:
            filters.append("agent_id IN (SELECT unnest($agent_ids))")
            params["agent_ids"] = tokens
        rows = self.session.fetch_rows(
            f"""
            WITH ranked AS (
              SELECT tenant_id, agent_id, initialized_at, initial_cash_krw, initial_positions_json,
                     ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY initialized_at DESC) AS rn
              FROM agent_sleeves
              WHERE {' AND '.join(filters)}
            )
            SELECT tenant_id, agent_id, initialized_at, initial_cash_krw, initial_positions_json
            FROM ranked
            WHERE rn = 1
            """,
            params,
        )
        return {str(row.get("agent_id")): row for row in rows if row.get("agent_id")}

    def latest_agent_state_checkpoints(
        self,
        *,
        agent_ids: list[str] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        tenant = self._tenant_token(tenant_id)
        params: dict[str, Any] = {"tenant_id": tenant}
        filters = ["tenant_id = $tenant_id"]
        tokens = [str(a or "").strip() for a in (agent_ids or []) if str(a or "").strip()]
        if tokens:
            filters.append("agent_id IN (SELECT unnest($agent_ids))")
            params["agent_ids"] = tokens
        rows = self.session.fetch_rows(
            f"""
            WITH ranked AS (
              SELECT tenant_id, event_id, checkpoint_at, agent_id, cash_krw, positions_json,
                     source, created_by, detail_json,
                     ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY checkpoint_at DESC) AS rn
              FROM agent_state_checkpoints
              WHERE {' AND '.join(filters)}
            )
            SELECT tenant_id, event_id, checkpoint_at, agent_id, cash_krw, positions_json,
                   source, created_by, detail_json
            FROM ranked
            WHERE rn = 1
            """,
            params,
        )
        return {str(row.get("agent_id")): row for row in rows if row.get("agent_id")}

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
        _ = excluded_tickers
        tenant = self._tenant_token(tenant_id)
        tokens = list(dict.fromkeys(str(a or "").strip() for a in agent_ids if str(a or "").strip()))
        if not tokens:
            return {}
        existing = self.latest_agent_sleeves(agent_ids=tokens, tenant_id=tenant)
        missing = [agent_id for agent_id in tokens if agent_id not in existing]
        if missing:
            per_agent = float(total_cash_krw) / float(max(len(tokens), 1)) if float(total_cash_krw) > 0 else 0.0
            ts = initialized_at or utc_now()
            for agent_id in missing:
                cash = float((capital_per_agent or {}).get(agent_id, per_agent))
                self.session.insert_dict(
                    "agent_sleeves",
                    {
                        "tenant_id": tenant,
                        "agent_id": agent_id,
                        "initialized_at": ts,
                        "initial_cash_krw": cash,
                        "initial_positions_json": "[]",
                    },
                )
                self._write_checkpoint(
                    tenant=tenant,
                    agent_id=agent_id,
                    checkpoint_at=ts,
                    cash_krw=cash,
                    positions=[],
                    source="agent_sleeves.ensure",
                )
        return self.latest_agent_sleeves(agent_ids=tokens, tenant_id=tenant)

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
        _ = excluded_tickers
        tenant = self._tenant_token(tenant_id)
        tokens = list(dict.fromkeys(str(a or "").strip() for a in agent_ids if str(a or "").strip()))
        if not tokens:
            return {}
        existing = self.latest_agent_state_checkpoints(agent_ids=tokens, tenant_id=tenant)
        missing = [agent_id for agent_id in tokens if agent_id not in existing]
        if missing:
            per_agent = float(total_cash_krw) / float(max(len(tokens), 1)) if float(total_cash_krw) > 0 else 0.0
            ts = checkpoint_at or utc_now()
            for agent_id in missing:
                cash = float((capital_per_agent or {}).get(agent_id, per_agent))
                self._write_checkpoint(
                    tenant=tenant,
                    agent_id=agent_id,
                    checkpoint_at=ts,
                    cash_krw=cash,
                    positions=[],
                    source="agent_state_checkpoints.ensure",
                )
        return self.latest_agent_state_checkpoints(agent_ids=tokens, tenant_id=tenant)

    def _write_checkpoint(
        self,
        *,
        tenant: str,
        agent_id: str,
        checkpoint_at: datetime,
        cash_krw: float,
        positions: list[dict[str, Any]],
        source: str,
        created_by: str = "local",
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.session.insert_dict(
            "agent_state_checkpoints",
            {
                "tenant_id": tenant,
                "event_id": f"checkpoint_{agent_id}_{uuid4().hex[:20]}",
                "checkpoint_at": checkpoint_at,
                "agent_id": agent_id,
                "cash_krw": float(cash_krw),
                "positions_json": json.dumps(positions, ensure_ascii=False, default=str),
                "source": source,
                "created_by": str(created_by or "").strip() or "local",
                "detail_json": json.dumps({"source": source, **dict(detail or {})}, ensure_ascii=False),
            },
        )

    def append_capital_events(self, rows: list[dict[str, Any]], *, tenant_id: str | None = None) -> int:
        tenant = self._tenant_token(tenant_id)
        payloads: list[dict[str, Any]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            agent = str(row.get("agent_id") or "").strip()
            if not agent:
                continue
            try:
                amount = abs(float(row.get("amount_krw") or 0.0))
            except (TypeError, ValueError):
                amount = 0.0
            if amount <= 0:
                continue
            payloads.append(
                {
                    "tenant_id": tenant,
                    "event_id": str(row.get("event_id") or f"cap_{uuid4().hex[:20]}"),
                    "occurred_at": row.get("occurred_at") or utc_now(),
                    "agent_id": agent,
                    "amount_krw": amount,
                    "event_type": str(row.get("event_type") or "INJECTION").strip().upper() or "INJECTION",
                    "reason": str(row.get("reason") or "").strip() or None,
                    "created_by": str(row.get("created_by") or "").strip() or None,
                }
            )
        return self.session.insert_dicts("capital_events", payloads)

    def capital_events_since(
        self,
        *,
        agent_id: str,
        since: datetime,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        tenant = self._tenant_token(tenant_id)
        agent = str(agent_id or "").strip()
        if not agent:
            return []
        return self.session.fetch_rows(
            """
            SELECT tenant_id, event_id, occurred_at, agent_id, amount_krw, event_type, reason, created_by
            FROM capital_events
            WHERE tenant_id = $tenant_id
              AND agent_id = $agent_id
              AND occurred_at >= $since
            ORDER BY occurred_at ASC, event_id ASC
            """,
            {"tenant_id": tenant, "agent_id": agent, "since": since},
        )

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
        tenant = self._tenant_token(tenant_id)
        tokens = list(dict.fromkeys(str(a or "").strip() for a in agent_ids if str(a or "").strip()))
        if not tokens:
            return {}
        try:
            default_target = float(target_sleeve_capital_krw)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_sleeve_capital_krw must be numeric") from exc
        if default_target <= 0:
            raise ValueError("target_sleeve_capital_krw must be > 0")

        ts = occurred_at or utc_now()
        result: dict[str, dict[str, Any]] = {}
        event_rows: list[dict[str, Any]] = []
        checkpoint_rows: dict[str, tuple[AccountSnapshot, float]] = {}

        for agent in tokens:
            snapshot, baseline_equity, _meta = self.build_agent_sleeve_snapshot(
                agent_id=agent,
                sources=sources,
                include_simulated=include_simulated,
                tenant_id=tenant,
            )
            positions_value = sum(pos.market_value_krw() for pos in snapshot.positions.values())
            current_cash = float(snapshot.cash_krw)
            current_equity = current_cash + positions_value
            agent_target = float((target_capitals or {}).get(agent, default_target))
            delta_cash = agent_target - float(baseline_equity)
            new_cash = current_cash + delta_cash
            over_target = False
            if new_cash < 0:
                delta_cash = -current_cash
                new_cash = 0.0
                over_target = True

            result[agent] = {
                "target_sleeve_capital_krw": float(agent_target),
                "effective_target_equity_krw": float(current_equity + delta_cash),
                "positions_value_krw": float(positions_value),
                "current_cash_krw": float(current_cash),
                "target_cash_krw": float(new_cash),
                "capital_flow_krw": float(delta_cash),
                "event_type": "NOOP",
                "over_target": bool(over_target),
                "equity_krw_before_adjustment": float(snapshot.total_equity_krw),
                "baseline_equity_krw_before_adjustment": float(baseline_equity),
            }
            if abs(delta_cash) <= 1e-9:
                continue
            event_type = "INJECTION" if delta_cash > 0 else "WITHDRAWAL"
            event_rows.append(
                {
                    "event_id": f"cap_{uuid4().hex[:20]}",
                    "occurred_at": ts,
                    "agent_id": agent,
                    "amount_krw": abs(float(delta_cash)),
                    "event_type": event_type,
                    "reason": "retarget_preserve_positions",
                    "created_by": str(created_by or "").strip() or "system",
                }
            )
            result[agent]["event_type"] = event_type
            checkpoint_rows[agent] = (snapshot, float(new_cash))

        if event_rows:
            self.append_capital_events(event_rows, tenant_id=tenant)
            checkpoint_at = ts + timedelta(microseconds=1)
            for agent, (snapshot, new_cash) in checkpoint_rows.items():
                positions = [
                    self._position_to_payload(pos)
                    for pos in sorted(snapshot.positions.values(), key=lambda p: str(p.ticker or ""))
                    if float(pos.quantity or 0.0) > 0
                ]
                self._write_checkpoint(
                    tenant=tenant,
                    agent_id=agent,
                    checkpoint_at=checkpoint_at,
                    cash_krw=new_cash,
                    positions=positions,
                    source="capital_events.retarget",
                    created_by=created_by,
                    detail={**result.get(agent, {}), "mode": "capital_retarget"},
                )
        return result

    def _seed_for_agent(self, *, agent_id: str, tenant_id: str, as_of_ts: datetime | None = None) -> tuple[datetime, float, list[dict[str, Any]], str]:
        checkpoint_filters = ["tenant_id = $tenant_id", "agent_id = $agent_id"]
        params: dict[str, Any] = {"tenant_id": tenant_id, "agent_id": agent_id}
        if as_of_ts is not None:
            checkpoint_filters.append("checkpoint_at <= $as_of_ts")
            params["as_of_ts"] = as_of_ts
        checkpoints = self.session.fetch_rows(
            f"""
            SELECT checkpoint_at, cash_krw, positions_json, source
            FROM agent_state_checkpoints
            WHERE {' AND '.join(checkpoint_filters)}
            ORDER BY checkpoint_at DESC
            LIMIT 1
            """,
            params,
        )
        if checkpoints:
            row = checkpoints[0]
            return (
                row["checkpoint_at"],
                float(row.get("cash_krw") or 0.0),
                self._parse_positions(row.get("positions_json")),
                str(row.get("source") or "agent_state_checkpoints"),
            )
        sleeves = self.latest_agent_sleeves(agent_ids=[agent_id], tenant_id=tenant_id)
        row = sleeves.get(agent_id)
        if row:
            return (
                row["initialized_at"],
                float(row.get("initial_cash_krw") or 0.0),
                self._parse_positions(row.get("initial_positions_json")),
                "agent_sleeves",
            )
        return utc_now(), 0.0, [], "empty"

    def _positions_from_payload(self, rows: list[dict[str, Any]]) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for item in rows:
            ticker = str(item.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            qty = float(item.get("quantity") or 0.0)
            avg = float(item.get("avg_price_krw") or item.get("market_price_krw") or 0.0)
            if qty <= 0 or avg <= 0:
                continue
            out[ticker] = Position(
                ticker=ticker,
                exchange_code=str(item.get("exchange_code") or ""),
                instrument_id=str(item.get("instrument_id") or ""),
                quantity=qty,
                avg_price_krw=avg,
                market_price_krw=float(item.get("market_price_krw") or avg),
                avg_price_native=item.get("avg_price_native"),
                market_price_native=item.get("market_price_native") or item.get("avg_price_native"),
                quote_currency=str(item.get("quote_currency") or ""),
                fx_rate=float(item.get("fx_rate") or 0.0),
            )
        return out

    def _apply_execution(self, *, positions: dict[str, Position], cash_krw: float, row: dict[str, Any]) -> float:
        ticker = str(row.get("ticker") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        if not ticker or side not in {"BUY", "SELL"}:
            return cash_krw
        qty = float(row.get("filled_qty") or row.get("requested_qty") or 0.0)
        price = float(row.get("avg_price_krw") or 0.0)
        if qty <= 0 or price <= 0:
            return cash_krw
        if side == "BUY":
            existing = positions.get(ticker)
            old_qty = float(existing.quantity) if existing else 0.0
            old_cost = old_qty * float(existing.avg_price_krw) if existing else 0.0
            new_qty = old_qty + qty
            avg = (old_cost + qty * price) / new_qty if new_qty > 0 else price
            positions[ticker] = Position(
                ticker=ticker,
                exchange_code=str(row.get("exchange_code") or (existing.exchange_code if existing else "")),
                instrument_id=str(row.get("instrument_id") or (existing.instrument_id if existing else "")),
                quantity=new_qty,
                avg_price_krw=avg,
                market_price_krw=price,
                avg_price_native=row.get("avg_price_native"),
                market_price_native=row.get("avg_price_native"),
                quote_currency=str(row.get("quote_currency") or ""),
                fx_rate=float(row.get("fx_rate") or 0.0),
            )
            return cash_krw - qty * price
        existing = positions.get(ticker)
        if not existing:
            return cash_krw + qty * price
        remaining = max(float(existing.quantity) - qty, 0.0)
        if remaining <= 0:
            positions.pop(ticker, None)
        else:
            existing.quantity = remaining
            existing.market_price_krw = price
            positions[ticker] = existing
        return cash_krw + qty * price

    def build_agent_sleeve_snapshot(
        self,
        *,
        agent_id: str,
        sources: list[str] | None = None,
        include_simulated: bool = True,
        tenant_id: str | None = None,
        as_of_ts: datetime | None = None,
    ) -> tuple[AccountSnapshot, float, dict[str, Any]]:
        tenant = self._tenant_token(tenant_id)
        agent = str(agent_id or "").strip()
        since, cash, seed_positions, source = self._seed_for_agent(agent_id=agent, tenant_id=tenant, as_of_ts=as_of_ts)
        positions = self._positions_from_payload(seed_positions)
        baseline = cash + sum(pos.quantity * pos.avg_price_krw for pos in positions.values())

        statuses = ["FILLED"]
        if include_simulated:
            statuses.append("SIMULATED")
        params: dict[str, Any] = {
            "tenant_id": tenant,
            "agent_id": agent,
            "since": since,
            "statuses": statuses,
        }
        time_clause = ""
        if as_of_ts is not None:
            time_clause = "AND created_at <= $as_of_ts"
            params["as_of_ts"] = as_of_ts
        executions = self.session.fetch_rows(
            f"""
            SELECT *
            FROM execution_reports
            WHERE tenant_id = $tenant_id
              AND agent_id = $agent_id
              AND created_at >= $since
              AND status IN (SELECT unnest($statuses))
              {time_clause}
            ORDER BY created_at ASC, order_id ASC
            """,
            params,
        )
        for row in executions:
            cash = self._apply_execution(positions=positions, cash_krw=cash, row=row)

        capital_flow_krw = 0.0
        capital_event_count = 0
        as_of_naive = self._as_utc_naive(as_of_ts)
        for event in self.capital_events_since(agent_id=agent, since=since, tenant_id=tenant):
            event_ts = self._as_utc_naive(event.get("occurred_at"))
            if as_of_naive is not None and event_ts is not None and event_ts > as_of_naive:
                continue
            delta = self._signed_capital_amount(event.get("event_type"), event.get("amount_krw"))
            if abs(delta) <= 1e-9:
                continue
            cash += delta
            baseline += delta
            capital_flow_krw += delta
            capital_event_count += 1

        if positions and self._market is not None:
            try:
                prices = self._market.latest_close_prices_with_currency(
                    tickers=list(positions.keys()),
                    sources=sources,
                    as_of_date=as_of_ts.date() if isinstance(as_of_ts, datetime) else None,
                )
            except Exception:
                prices = {}
            for ticker, pos in positions.items():
                px = prices.get(ticker) or {}
                if px.get("close_price_krw"):
                    pos.market_price_krw = float(px["close_price_krw"])
                    pos.market_price_native = px.get("close_price_native")
                    pos.quote_currency = str(px.get("quote_currency") or pos.quote_currency)
                    pos.fx_rate = float(px.get("fx_rate_used") or pos.fx_rate or 0.0)
        total = cash + sum(pos.market_value_krw() for pos in positions.values())
        return (
            AccountSnapshot(cash_krw=cash, total_equity_krw=total, positions=positions),
            baseline,
            {
                "seed_source": source,
                "valuation_source": "local_sleeve_replay",
                "capital_flow_krw": float(capital_flow_krw),
                "capital_event_count": int(capital_event_count),
            },
        )

    @staticmethod
    def _normalize_market_scope(market_scope: str | None = None) -> str:
        tokens: list[str] = []
        for token in str(market_scope or "").replace("|", ",").replace(";", ",").split(","):
            clean = token.strip().lower()
            if clean and clean not in tokens:
                tokens.append(clean)
        return ",".join(tokens)

    @staticmethod
    def _market_scope_tokens(market: str | None = None) -> list[str]:
        tokens: list[str] = []
        for token in str(market or "").replace("|", ",").replace(";", ",").split(","):
            clean = token.strip().lower()
            if not clean:
                continue
            mapped = {
                "nasdaq": ["us"],
                "nyse": ["us"],
                "amex": ["us"],
                "arca": ["us"],
                "usa": ["us"],
                "kr": ["kospi", "kosdaq"],
                "krx": ["kospi", "kosdaq"],
                "korea": ["kospi", "kosdaq"],
                "domestic": ["kospi", "kosdaq"],
            }.get(clean, [clean])
            for item in mapped:
                if item and item not in tokens:
                    tokens.append(item)
        return tokens

    @classmethod
    def _filter_tickers_for_market(cls, tickers: list[str], market: str | None = None) -> list[str]:
        tokens = set(cls._market_scope_tokens(market))
        wants_kr = bool(tokens.intersection({"kospi", "kosdaq"}))
        wants_us = "us" in tokens
        if wants_kr and not wants_us:
            return [ticker for ticker in tickers if ticker.isdigit() and len(ticker) == 6]
        if wants_us and not wants_kr:
            return [ticker for ticker in tickers if ticker and not ticker[:1].isdigit()]
        return tickers

    def write_account_snapshot(
        self,
        snapshot: AccountSnapshot,
        *,
        tenant_id: str | None = None,
        market_scope: str | None = None,
    ) -> None:
        tenant = self._tenant_token(tenant_id)
        ts = utc_now()
        scope = self._normalize_market_scope(market_scope)
        self.session.insert_dict(
            "account_snapshots",
            {
                "tenant_id": tenant,
                "snapshot_at": ts,
                "market_scope": scope or None,
                "cash_krw": snapshot.cash_krw,
                "total_equity_krw": snapshot.total_equity_krw,
                "usd_krw_rate": snapshot.usd_krw_rate,
                "cash_foreign": snapshot.cash_foreign,
                "cash_foreign_currency": snapshot.cash_foreign_currency or None,
            },
        )
        for pos in snapshot.positions.values():
            self.session.insert_dict(
                "positions_current",
                {
                    "tenant_id": tenant,
                    "snapshot_at": ts,
                    "ticker": pos.ticker,
                    "exchange_code": pos.exchange_code or None,
                    "instrument_id": pos.instrument_id or None,
                    "quantity": pos.quantity,
                    "avg_price_krw": pos.avg_price_krw,
                    "market_price_krw": pos.market_price_krw,
                    "avg_price_native": pos.avg_price_native,
                    "market_price_native": pos.market_price_native,
                    "quote_currency": pos.quote_currency or None,
                    "fx_rate": pos.fx_rate,
                },
            )

    def latest_account_snapshot(
        self,
        *,
        tenant_id: str | None = None,
        market_scope: str | None = None,
    ) -> AccountSnapshot | None:
        tenant = self._tenant_token(tenant_id)
        scope = self._normalize_market_scope(market_scope)
        filters = ["tenant_id = $tenant_id"]
        params: dict[str, Any] = {"tenant_id": tenant}
        if scope:
            filters.append("market_scope = $market_scope")
            params["market_scope"] = scope
        rows = self.session.fetch_rows(
            f"""
            SELECT *
            FROM account_snapshots
            WHERE {' AND '.join(filters)}
            ORDER BY snapshot_at DESC
            LIMIT 1
            """,
            params,
        )
        if not rows:
            return None
        snap = rows[0]
        positions_rows = self.session.fetch_rows(
            """
            SELECT *
            FROM positions_current
            WHERE tenant_id = $tenant_id
              AND snapshot_at = $snapshot_at
            """,
            {"tenant_id": tenant, "snapshot_at": snap["snapshot_at"]},
        )
        positions = self._positions_from_payload(positions_rows)
        return AccountSnapshot(
            cash_krw=float(snap.get("cash_krw") or 0.0),
            total_equity_krw=float(snap.get("total_equity_krw") or 0.0),
            positions=positions,
            usd_krw_rate=float(snap.get("usd_krw_rate") or 0.0),
            cash_foreign=float(snap.get("cash_foreign") or 0.0),
            cash_foreign_currency=str(snap.get("cash_foreign_currency") or ""),
        )

    def get_all_held_tickers(self, *, tenant_id: str | None = None, market: str = "") -> list[str]:
        """Returns distinct tickers with positive local ledger positions."""
        tenant = self._tenant_token(tenant_id)
        rows = self.session.fetch_rows(
            """
            SELECT ticker
            FROM (
              SELECT ticker,
                     SUM(CASE WHEN side = 'BUY' THEN filled_qty ELSE -filled_qty END) AS net_qty
              FROM execution_reports
              WHERE tenant_id = $tenant_id
                AND status IN ('FILLED', 'SIMULATED')
              GROUP BY ticker
            )
            WHERE net_qty > 0
            ORDER BY ticker
            """,
            {"tenant_id": tenant},
        )
        tickers = [str(row.get("ticker") or "").strip().upper() for row in rows if str(row.get("ticker") or "").strip()]
        m = str(market or "").strip().lower()
        if m in {"kospi", "kosdaq", "kr", "korea"}:
            return [ticker for ticker in tickers if ticker.isdigit() and len(ticker) == 6]
        if m in {"us", "nasdaq", "nyse", "amex"}:
            return [ticker for ticker in tickers if ticker and not ticker[:1].isdigit()]
        return tickers

    def get_latest_position_tickers(
        self,
        *,
        tenant_id: str | None = None,
        market: str = "",
        all_tenants: bool = False,
    ) -> list[str]:
        """Returns distinct tickers from latest account snapshot positions."""
        params: dict[str, Any] = {}
        market_tokens = self._market_scope_tokens(market)
        scope_match_sql = ""
        if market_tokens:
            scope_param_idx = 0

            def like_expr(token: str) -> str:
                nonlocal scope_param_idx
                key = f"market_scope_like_{scope_param_idx}"
                scope_param_idx += 1
                params[key] = f"%,{token},%"
                return (
                    "(',' || lower(replace(replace(coalesce(market_scope, ''), '|', ','), ';', ',')) || ',') "
                    f"LIKE ${key}"
                )

            groups: list[str] = []
            if "us" in market_tokens:
                groups.append(f"({like_expr('us')})")
            kr_tokens = [token for token in market_tokens if token in {"kospi", "kosdaq"}]
            if kr_tokens:
                groups.append("(" + " OR ".join(like_expr(token) for token in kr_tokens) + ")")
            for token in market_tokens:
                if token not in {"us", "kospi", "kosdaq"}:
                    groups.append(f"({like_expr(token)})")
            scope_match_sql = "(" + " AND ".join(groups) + ")"
        if all_tenants:
            if scope_match_sql:
                latest_sql = f"""
            WITH ranked_snapshots AS (
              SELECT tenant_id,
                     snapshot_at,
                     ROW_NUMBER() OVER (
                       PARTITION BY tenant_id
                       ORDER BY CASE WHEN {scope_match_sql} THEN 0 ELSE 1 END, snapshot_at DESC
                     ) AS rn
              FROM account_snapshots
              WHERE {scope_match_sql}
                 OR market_scope IS NULL
                 OR trim(market_scope) = ''
            ),
            latest AS (
              SELECT tenant_id, snapshot_at
              FROM ranked_snapshots
              WHERE rn = 1
            )
                """
            else:
                latest_sql = """
            WITH latest AS (
              SELECT tenant_id, MAX(snapshot_at) AS snapshot_at
              FROM account_snapshots
              GROUP BY tenant_id
            )
                """
            sql = f"""
            {latest_sql}
            SELECT DISTINCT p.ticker
            FROM positions_current p
            JOIN latest l
              ON p.tenant_id = l.tenant_id
             AND p.snapshot_at = l.snapshot_at
            WHERE p.quantity > 0
            ORDER BY p.ticker
            """
        else:
            tenant = self._tenant_token(tenant_id)
            params["tenant_id"] = tenant
            if scope_match_sql:
                latest_sql = f"""
            WITH ranked_snapshots AS (
              SELECT snapshot_at,
                     ROW_NUMBER() OVER (
                       ORDER BY CASE WHEN {scope_match_sql} THEN 0 ELSE 1 END, snapshot_at DESC
                     ) AS rn
              FROM account_snapshots
              WHERE tenant_id = $tenant_id
                AND ({scope_match_sql}
                  OR market_scope IS NULL
                  OR trim(market_scope) = '')
            ),
            latest AS (
              SELECT snapshot_at
              FROM ranked_snapshots
              WHERE rn = 1
            )
                """
            else:
                latest_sql = """
            WITH latest AS (
              SELECT MAX(snapshot_at) AS snapshot_at
              FROM account_snapshots
              WHERE tenant_id = $tenant_id
            )
                """
            sql = f"""
            {latest_sql}
            SELECT DISTINCT p.ticker
            FROM positions_current p
            CROSS JOIN latest l
            WHERE p.tenant_id = $tenant_id
              AND p.snapshot_at = l.snapshot_at
              AND p.quantity > 0
            ORDER BY p.ticker
            """
        rows = self.session.fetch_rows(sql, params)
        tickers = [str(row.get("ticker") or "").strip().upper() for row in rows if str(row.get("ticker") or "").strip()]
        return self._filter_tickers_for_market(tickers, market)

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
        tenant = self._tenant_token(tenant_id)
        agent = str(agent_id or "").strip()
        base = float(baseline_equity_krw)
        nav = float(nav_krw)
        pnl = nav - base
        pnl_ratio = pnl / base if base else 0.0
        self.session.execute(
            "DELETE FROM agent_nav_daily WHERE tenant_id = $tenant_id AND nav_date = $nav_date AND agent_id = $agent_id",
            {"tenant_id": tenant, "nav_date": nav_date, "agent_id": agent},
        )
        self.session.insert_dict(
            "agent_nav_daily",
            {
                "tenant_id": tenant,
                "nav_date": nav_date,
                "agent_id": agent,
                "nav_krw": nav,
                "pnl_krw": pnl,
                "pnl_ratio": pnl_ratio,
            },
        )
        cash = float(cash_krw) if cash_krw is not None else 0.0
        market_value = float(market_value_krw) if market_value_krw is not None else max(nav - cash, 0.0)
        flow = float(capital_flow_krw) if capital_flow_krw is not None else 0.0
        self.session.execute(
            "DELETE FROM official_nav_daily WHERE tenant_id = $tenant_id AND nav_date = $nav_date AND agent_id = $agent_id",
            {"tenant_id": tenant, "nav_date": nav_date, "agent_id": agent},
        )
        self.session.insert_dict(
            "official_nav_daily",
            {
                "tenant_id": tenant,
                "nav_date": nav_date,
                "agent_id": agent,
                "nav_krw": nav,
                "cash_krw": cash,
                "market_value_krw": market_value,
                "capital_flow_krw": flow,
                "pnl_krw": pnl,
                "pnl_ratio": pnl_ratio,
                "fx_source": fx_source or None,
                "valuation_source": valuation_source or "local_sleeve_replay",
            },
        )
