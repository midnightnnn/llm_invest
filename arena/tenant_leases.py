from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class LeaseAcquireResult:
    acquired: bool
    reason: str
    lease_id: str


def _as_utc_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return None


def _lease_doc_id(*, run_type: str, market: str, trading_date: date, tenant_id: str, execution_source: str = "") -> str:
    clean_source = str(execution_source or "").strip().lower()
    source_prefix = f"{clean_source}_" if clean_source else ""
    return (
        f"{str(run_type or '').strip().lower() or 'run'}_"
        f"{source_prefix}"
        f"{str(market or '').strip().lower() or 'unknown'}_"
        f"{trading_date.isoformat()}_"
        f"{str(tenant_id or '').strip().lower() or 'local'}"
    )


class FirestoreTenantLeaseStore:
    """Coordinates one tenant execution per market/trading date."""

    def __init__(self, *, project: str, collection: str = "tenant_cycle_leases"):
        self.project = str(project or "").strip()
        self.collection = str(collection or "").strip() or "tenant_cycle_leases"

    def _client(self):
        from google.cloud import firestore

        return firestore.Client(project=self.project)

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
        from google.cloud import firestore

        tenant = str(tenant_id or "").strip().lower() or "local"
        clean_market = str(market or "").strip().lower() or "unknown"
        clean_run_type = str(run_type or "").strip().lower() or "run"
        clean_source = str(execution_source or "").strip().lower()
        clean_owner = str(owner_execution or "").strip() or "unknown"
        clean_run_id = str(run_id or "").strip() or "unknown"
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=max(1, int(lease_ttl_minutes)))
        lease_id = _lease_doc_id(
            run_type=clean_run_type,
            market=clean_market,
            trading_date=trading_date,
            tenant_id=tenant,
            execution_source=clean_source,
        )

        client = self._client()
        doc = client.collection(self.collection).document(lease_id)
        payload = {
            "tenant_id": tenant,
            "market": clean_market,
            "trading_date": trading_date.isoformat(),
            "run_type": clean_run_type,
            "execution_source": clean_source or None,
            "run_id": clean_run_id,
            "owner_execution": clean_owner,
            "status": "running",
            "started_at": now,
            "updated_at": now,
            "lease_expires_at": expires_at,
            "detail": detail or {},
        }

        @firestore.transactional
        def _txn(transaction):
            snap = doc.get(transaction=transaction)
            if getattr(snap, "exists", False):
                current = snap.to_dict() or {}
                status = str(current.get("status") or "").strip().lower()
                current_owner = str(current.get("owner_execution") or "").strip()
                current_expiry = _as_utc_datetime(current.get("lease_expires_at"))
                if status == "success":
                    return LeaseAcquireResult(acquired=False, reason="already_completed", lease_id=lease_id)
                if current_owner == clean_owner and status in {"failed", "blocked", "warning", "skipped"}:
                    return LeaseAcquireResult(acquired=False, reason="same_execution_replay", lease_id=lease_id)
                if status == "running" and current_expiry and current_expiry > now:
                    return LeaseAcquireResult(acquired=False, reason="lease_held", lease_id=lease_id)
            transaction.set(doc, payload)
            return LeaseAcquireResult(acquired=True, reason="acquired", lease_id=lease_id)

        return _txn(client.transaction())

    def complete(
        self,
        *,
        lease_id: str,
        status: str,
        owner_execution: str,
        message: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        doc = self._client().collection(self.collection).document(str(lease_id or "").strip())
        now = datetime.now(timezone.utc)
        doc.set(
            {
                "status": str(status or "").strip().lower() or "unknown",
                "owner_execution": str(owner_execution or "").strip() or "unknown",
                "finished_at": now,
                "updated_at": now,
                "lease_expires_at": now,
                "message": str(message or "").strip() or None,
                "detail": detail or {},
            },
            merge=True,
        )
