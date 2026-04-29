"""File-backed tenant lease store for local mode."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any

from arena.tenant_leases import LeaseAcquireResult, _as_utc_datetime, _lease_doc_id


def default_lease_path() -> Path:
    raw = os.getenv("ARENA_LOCAL_LEASE_FILE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.cwd() / "data" / "tenant_leases.json").resolve()


class LocalTenantLeaseStore:
    """Coordinates local tenant execution using a JSON file plus file lock."""

    def __init__(self, *, path: str | Path | None = None, collection: str = "tenant_cycle_leases") -> None:
        _ = collection
        self.path = Path(path) if path is not None else default_lease_path()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _lock(self):
        from filelock import FileLock

        self.path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.lock_path), timeout=30)

    def _read(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def acquire(
        self,
        *,
        tenant_id: str,
        market: str,
        trading_date: date,
        run_type: str,
        execution_source: str = "",
        owner_execution: str,
        run_id: str,
        lease_ttl_minutes: int = 120,
        detail: dict[str, Any] | None = None,
    ) -> LeaseAcquireResult:
        tenant = str(tenant_id or "").strip().lower() or "local"
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=max(1, int(lease_ttl_minutes)))
        lease_id = _lease_doc_id(
            run_type=run_type,
            market=market,
            trading_date=trading_date,
            tenant_id=tenant,
            execution_source=execution_source,
        )
        with self._lock():
            data = self._read()
            current = data.get(lease_id)
            if isinstance(current, dict):
                status = str(current.get("status") or "").strip().lower()
                current_owner = str(current.get("owner_execution") or "").strip()
                try:
                    parsed_expiry = datetime.fromisoformat(str(current.get("lease_expires_at"))) if current.get("lease_expires_at") else None
                except ValueError:
                    parsed_expiry = None
                current_expiry = _as_utc_datetime(parsed_expiry)
                if status == "success":
                    return LeaseAcquireResult(acquired=False, reason="already_completed", lease_id=lease_id)
                if current_owner == str(owner_execution or "").strip() and status in {"failed", "blocked", "warning", "skipped"}:
                    return LeaseAcquireResult(acquired=False, reason="same_execution_replay", lease_id=lease_id)
                if status == "running" and current_expiry and current_expiry > now:
                    return LeaseAcquireResult(acquired=False, reason="lease_held", lease_id=lease_id)
            data[lease_id] = {
                "tenant_id": tenant,
                "market": str(market or "").strip().lower() or "unknown",
                "trading_date": trading_date.isoformat(),
                "run_type": str(run_type or "").strip().lower() or "run",
                "execution_source": str(execution_source or "").strip().lower() or None,
                "run_id": str(run_id or "").strip() or "unknown",
                "owner_execution": str(owner_execution or "").strip() or "unknown",
                "status": "running",
                "started_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "lease_expires_at": expires_at.isoformat(),
                "detail": detail or {},
            }
            self._write(data)
        return LeaseAcquireResult(acquired=True, reason="acquired", lease_id=lease_id)

    def complete(
        self,
        *,
        lease_id: str,
        status: str,
        owner_execution: str,
        message: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._lock():
            data = self._read()
            row = dict(data.get(str(lease_id or "").strip()) or {})
            row.update(
                {
                    "status": str(status or "").strip().lower() or "unknown",
                    "owner_execution": str(owner_execution or "").strip() or "unknown",
                    "finished_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "lease_expires_at": now.isoformat(),
                    "message": str(message or "").strip() or None,
                    "detail": detail or {},
                }
            )
            data[str(lease_id or "").strip()] = row
            self._write(data)
