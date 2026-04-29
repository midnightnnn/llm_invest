from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from arena.config import load_settings
from arena.data.local.repository import LocalRepository
from arena.memory.vector_factory import build_vector_store
from arena.memory.vector_local import NullVectorStore
from arena.open_trading.token_cache import TokenRecord
from arena.open_trading.token_cache_file import FileTokenCache
from arena.tenant_leases_local import LocalTenantLeaseStore


def test_local_vector_factory_falls_back_without_local_vector_extra(monkeypatch, tmp_path):
    monkeypatch.setenv("ARENA_MODE", "local")
    monkeypatch.setenv("ARENA_LOCAL_DB_PATH", str(tmp_path / "arena.duckdb"))
    settings = load_settings()
    repo = LocalRepository(settings=settings, tenant_id="t1", db_path=str(tmp_path / "arena.duckdb"))

    store = build_vector_store(repo, settings.memory_policy)

    # CI intentionally does not need chromadb/sentence-transformers installed
    # for local quickstart to work.
    if isinstance(store, NullVectorStore):
        assert store.search_similar_memories("gpt", "hello") == []
    else:
        assert hasattr(store, "save_memory_vector")


def test_file_token_cache_round_trips(tmp_path):
    cache = FileTokenCache(path=tmp_path / "tokens.json")
    record = TokenRecord(token="tok", expires_at=datetime.now(timezone.utc) + timedelta(hours=1))

    cache.set(base_url="https://example.test", app_key="app", record=record)
    loaded = cache.get(base_url="https://example.test", app_key="app")

    assert loaded is not None
    assert loaded.token == "tok"


def test_file_token_cache_ignores_expired(tmp_path):
    cache = FileTokenCache(path=tmp_path / "tokens.json")
    cache.set(
        base_url="https://example.test",
        app_key="app",
        record=TokenRecord(token="tok", expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
    )

    assert cache.get(base_url="https://example.test", app_key="app") is None


def test_local_tenant_lease_acquire_and_complete(tmp_path):
    store = LocalTenantLeaseStore(path=tmp_path / "leases.json")
    first = store.acquire(
        tenant_id="tenant-a",
        market="us",
        trading_date=date(2026, 4, 28),
        run_type="agent_cycle",
        owner_execution="owner-1",
        run_id="run-1",
    )
    second = store.acquire(
        tenant_id="tenant-a",
        market="us",
        trading_date=date(2026, 4, 28),
        run_type="agent_cycle",
        owner_execution="owner-2",
        run_id="run-2",
    )

    assert first.acquired is True
    assert second.acquired is False
    assert second.reason == "lease_held"

    store.complete(lease_id=first.lease_id, status="success", owner_execution="owner-1")
    third = store.acquire(
        tenant_id="tenant-a",
        market="us",
        trading_date=date(2026, 4, 28),
        run_type="agent_cycle",
        owner_execution="owner-3",
        run_id="run-3",
    )
    assert third.acquired is False
    assert third.reason == "already_completed"
