"""Allocation backtest store — manages backtest run, NAV and allocation tables."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from arena.models import utc_now

if TYPE_CHECKING:
    from arena.data.bigquery.session import BigQuerySession


class BacktestStore:
    """Reads and writes allocation-backtest data via a shared session."""

    def __init__(self, session: BigQuerySession) -> None:
        self.session = session

    # ------------------------------------------------------------------

    def upsert_alloc_backtest_run(
        self,
        *,
        run_id: str,
        start_date: date,
        end_date: date,
        rebalance_freq: str,
        lookback_days: int,
        fee_bps: float,
        tickers: list[str],
        strategies: list[str],
        notes: str | None = None,
        created_at: datetime | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Upserts one backtest run metadata row."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        run_id = str(run_id).strip()
        if not run_id:
            raise ValueError("run_id is required")

        ts = created_at or utc_now()
        table_id = f"{self.session.dataset_fqn}.alloc_backtest_runs"

        self.session.execute(
            f"DELETE FROM `{table_id}` WHERE tenant_id = @tenant_id AND run_id = @run_id",
            {"tenant_id": tenant, "run_id": run_id},
        )

        row = {
            "tenant_id": tenant,
            "run_id": run_id,
            "created_at": ts.isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "rebalance_freq": str(rebalance_freq),
            "lookback_days": int(lookback_days),
            "fee_bps": float(fee_bps),
            "tickers": [str(t).strip().upper() for t in tickers if str(t).strip()],
            "strategies": [str(s).strip() for s in strategies if str(s).strip()],
            "notes": str(notes) if notes else None,
        }
        errors = self.session.client.insert_rows_json(table_id, [row])
        if errors:
            raise RuntimeError(f"alloc_backtest_runs insert failed: {errors}")

    def write_alloc_backtest_nav(
        self,
        *,
        run_id: str,
        strategy: str,
        rows: list[dict[str, Any]],
        tenant_id: str | None = None,
    ) -> None:
        """Replaces alloc_backtest_nav rows for (run_id, strategy)."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        run_id = str(run_id).strip()
        strategy = str(strategy).strip()
        table_id = f"{self.session.dataset_fqn}.alloc_backtest_nav"

        self.session.execute(
            f"DELETE FROM `{table_id}` WHERE tenant_id = @tenant_id AND run_id = @run_id AND strategy = @strategy",
            {"tenant_id": tenant, "run_id": run_id, "strategy": strategy},
        )

        if not rows:
            return

        payloads: list[dict[str, Any]] = []
        for r in rows:
            nav_date = r.get("nav_date")
            if isinstance(nav_date, date):
                nav_date = nav_date.isoformat()
            else:
                nav_date = str(nav_date or "")[:10]
            payloads.append(
                {
                    "tenant_id": tenant,
                    "run_id": run_id,
                    "strategy": strategy,
                    "nav_date": nav_date,
                    "nav": float(r.get("nav") or 0.0),
                    "daily_return": float(r.get("daily_return") or 0.0),
                    "cum_return": float(r.get("cum_return") or 0.0),
                    "drawdown": float(r.get("drawdown") or 0.0),
                }
            )

        chunk = 500
        for i in range(0, len(payloads), chunk):
            errors = self.session.client.insert_rows_json(table_id, payloads[i : i + chunk])
            if errors:
                raise RuntimeError(f"alloc_backtest_nav insert failed: {errors}")

    def write_alloc_backtest_allocations(
        self,
        *,
        run_id: str,
        strategy: str,
        rows: list[dict[str, Any]],
        tenant_id: str | None = None,
    ) -> None:
        """Replaces alloc_backtest_allocations rows for (run_id, strategy)."""
        tenant = self.session.resolve_tenant_id(tenant_id)
        run_id = str(run_id).strip()
        strategy = str(strategy).strip()
        table_id = f"{self.session.dataset_fqn}.alloc_backtest_allocations"

        self.session.execute(
            f"DELETE FROM `{table_id}` WHERE tenant_id = @tenant_id AND run_id = @run_id AND strategy = @strategy",
            {"tenant_id": tenant, "run_id": run_id, "strategy": strategy},
        )

        if not rows:
            return

        payloads: list[dict[str, Any]] = []
        for r in rows:
            rebalance_date = str(r.get("rebalance_date") or "")[:10]
            ticker = str(r.get("ticker") or "").strip().upper()
            if not rebalance_date or not ticker:
                continue
            payloads.append(
                {
                    "tenant_id": tenant,
                    "run_id": run_id,
                    "strategy": strategy,
                    "rebalance_date": rebalance_date,
                    "ticker": ticker,
                    "weight": float(r.get("weight") or 0.0),
                    "turnover": float(r.get("turnover") or 0.0),
                    "cost_ratio": float(r.get("cost_ratio") or 0.0),
                }
            )

        chunk = 500
        for i in range(0, len(payloads), chunk):
            errors = self.session.client.insert_rows_json(table_id, payloads[i : i + chunk])
            if errors:
                raise RuntimeError(f"alloc_backtest_allocations insert failed: {errors}")
