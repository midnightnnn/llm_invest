from __future__ import annotations

import types
from datetime import date, datetime, timedelta, timezone

from arena.tenant_leases import FirestoreTenantLeaseStore


class _FakeSnap:
    def __init__(self, data: dict | None) -> None:
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> dict:
        return dict(self._data or {})


class _FakeDoc:
    def __init__(self, *, existing: dict | None = None, fail_create: bool = False) -> None:
        self.existing = dict(existing or {}) if existing is not None else None
        self.fail_create = fail_create
        self.create_calls: list[dict] = []
        self.get_calls = 0
        self.set_calls: list[tuple[dict, bool]] = []

    def create(self, payload: dict) -> None:
        self.create_calls.append(dict(payload))
        if self.fail_create:
            raise RuntimeError("already exists")
        self.existing = dict(payload)

    def get(self, transaction=None):
        _ = transaction
        self.get_calls += 1
        return _FakeSnap(self.existing)

    def set(self, payload: dict, merge: bool = False) -> None:
        self.set_calls.append((dict(payload), merge))
        if merge and self.existing is not None:
            self.existing.update(payload)
        else:
            self.existing = dict(payload)


class _FakeCollection:
    def __init__(self, doc: _FakeDoc) -> None:
        self.doc = doc

    def document(self, doc_id: str) -> _FakeDoc:
        _ = doc_id
        return self.doc


class _FakeClient:
    def __init__(self, doc: _FakeDoc) -> None:
        self.doc = doc
        self.collection_calls: list[str] = []

    def collection(self, name: str) -> _FakeCollection:
        self.collection_calls.append(name)
        return _FakeCollection(self.doc)

    def transaction(self):
        class _FakeTransaction:
            def set(self, doc: _FakeDoc, payload: dict) -> None:
                doc.set(payload)

        return _FakeTransaction()


def _install_fake_firestore(monkeypatch, client: _FakeClient) -> None:
    fake_module = types.SimpleNamespace(Client=lambda project=None: client, transactional=lambda fn: fn)

    import google.cloud

    monkeypatch.setattr(google.cloud, "firestore", fake_module, raising=False)


def test_tenant_lease_acquire_creates_document(monkeypatch) -> None:
    doc = _FakeDoc()
    client = _FakeClient(doc)
    _install_fake_firestore(monkeypatch, client)

    store = FirestoreTenantLeaseStore(project="proj-x", collection="tenant_cycle_leases")
    result = store.acquire(
        tenant_id="Tenant-A",
        market="us",
        trading_date=date(2026, 3, 19),
        run_type="agent_cycle",
        owner_execution="exec-1",
        run_id="run-1",
        lease_ttl_minutes=30,
        detail={"job_name": "job-a"},
    )

    assert result.acquired is True
    assert result.reason == "acquired"
    assert result.lease_id == "agent_cycle_us_2026-03-19_tenant-a"
    assert client.collection_calls == ["tenant_cycle_leases"]
    assert doc.set_calls
    payload, merge = doc.set_calls[0]
    assert merge is False
    assert payload["tenant_id"] == "tenant-a"
    assert payload["market"] == "us"


def test_tenant_lease_complete_updates_document(monkeypatch) -> None:
    doc = _FakeDoc(existing={"status": "running", "run_id": "run-1"})
    client = _FakeClient(doc)
    _install_fake_firestore(monkeypatch, client)

    store = FirestoreTenantLeaseStore(project="proj-x", collection="tenant_cycle_leases")
    store.complete(
        lease_id="agent_cycle_us_2026-03-19_tenant-a",
        status="success",
        owner_execution="exec-1",
        message="done",
        detail={"report_count": 3},
    )

    assert doc.set_calls
    payload, merge = doc.set_calls[-1]
    assert merge is True
    assert payload["status"] == "success"
    assert payload["message"] == "done"
    assert payload["detail"] == {"report_count": 3}


def test_tenant_lease_acquire_scopes_by_execution_source(monkeypatch) -> None:
    doc = _FakeDoc()
    client = _FakeClient(doc)
    _install_fake_firestore(monkeypatch, client)

    store = FirestoreTenantLeaseStore(project="proj-x", collection="tenant_cycle_leases")
    result = store.acquire(
        tenant_id="Tenant-A",
        market="us",
        trading_date=date(2026, 3, 19),
        run_type="agent_cycle",
        execution_source="scheduler",
        owner_execution="exec-1",
        run_id="run-1",
        lease_ttl_minutes=30,
        detail={"job_name": "job-a"},
    )

    assert result.acquired is True
    assert result.lease_id == "agent_cycle_scheduler_us_2026-03-19_tenant-a"
    payload, merge = doc.set_calls[0]
    assert merge is False
    assert payload["execution_source"] == "scheduler"


def test_tenant_lease_acquire_blocks_same_execution_replay_after_failure(monkeypatch) -> None:
    doc = _FakeDoc(
        existing={
            "status": "failed",
            "owner_execution": "exec-1",
            "lease_expires_at": datetime.now(timezone.utc),
        }
    )
    client = _FakeClient(doc)
    _install_fake_firestore(monkeypatch, client)

    store = FirestoreTenantLeaseStore(project="proj-x", collection="tenant_cycle_leases")
    result = store.acquire(
        tenant_id="Tenant-A",
        market="us",
        trading_date=date(2026, 3, 19),
        run_type="agent_cycle",
        owner_execution="exec-1",
        run_id="run-2",
    )

    assert result.acquired is False
    assert result.reason == "same_execution_replay"
    assert doc.get_calls == 1
    assert doc.set_calls == []


def test_tenant_lease_acquire_allows_new_execution_after_failure(monkeypatch) -> None:
    doc = _FakeDoc(
        existing={
            "status": "failed",
            "owner_execution": "exec-1",
            "lease_expires_at": datetime.now(timezone.utc),
        }
    )
    client = _FakeClient(doc)
    _install_fake_firestore(monkeypatch, client)

    store = FirestoreTenantLeaseStore(project="proj-x", collection="tenant_cycle_leases")
    result = store.acquire(
        tenant_id="Tenant-A",
        market="us",
        trading_date=date(2026, 3, 19),
        run_type="agent_cycle",
        owner_execution="exec-2",
        run_id="run-2",
    )

    assert result.acquired is True
    assert result.reason == "acquired"
    assert doc.set_calls
