from __future__ import annotations

import pytest

from arena.agents.investment_chat.drafts import draft_key, load_draft, save_draft
from tests.data.local_repository_helpers import repo


def test_ensure_tables_creates_arena_schema(repo):
    rows = repo.fetch_rows(
        "SELECT COUNT(*) AS n FROM information_schema.tables WHERE table_schema='main'"
    )
    assert int(rows[0]["n"]) > 50  # 54 arena tables today, future-proof.


def test_dataset_fqn_is_empty_for_facade_parity(repo):
    assert repo.dataset_fqn == ""


def test_resolve_tenant_id_normalises(repo):
    assert repo.resolve_tenant_id(None) == "tenant-a"
    assert repo.resolve_tenant_id("Tenant-B") == "tenant-b"


def test_set_and_get_config(repo):
    repo.set_config("tenant-a", "risk_policy", "aggressive", updated_by="midnight")
    assert repo.get_config("tenant-a", "risk_policy") == "aggressive"
    assert repo.get_config("tenant-a", "unset_key") is None
    assert repo.get_config("other-tenant", "risk_policy") is None


def test_chat_order_draft_key_is_valid_local_config_key(repo):
    key = draft_key("abc123")
    assert ":" not in key

    save_draft(repo, tenant_id="tenant-a", token="abc123", draft={"status": "draft"})

    assert repo.get_config("tenant-a", key)
    assert load_draft(repo, tenant_id="tenant-a", token="abc123") == {"status": "draft"}


def test_get_configs_returns_latest_per_key(repo):
    repo.set_config("tenant-a", "k1", "old")
    repo.set_config("tenant-a", "k1", "new")
    repo.set_config("tenant-a", "k2", "v2")

    out = repo.get_configs("tenant-a", ["k1", "k2", "k3"])
    assert out == {"k1": "new", "k2": "v2"}


def test_set_config_rejects_blank_inputs(repo):
    with pytest.raises(ValueError):
        repo.set_config("", "k", "v")
    with pytest.raises(ValueError):
        repo.set_config("tenant-a", "", "v")


def test_unimplemented_method_raises_attribute_error_for_hasattr(repo):
    with pytest.raises(AttributeError, match="not implemented yet"):
        repo.not_a_local_method()
    assert hasattr(repo, "not_a_local_method") is False


def test_attribute_lookup_traverses_stores_first(repo):
    # ``recent_memory_events`` is implemented on LocalMemoryStore; the
    # __getattr__ fallback must locate it via _STORE_ATTRS instead of raising.
    bound = repo.recent_memory_events  # must not raise
    assert callable(bound)
