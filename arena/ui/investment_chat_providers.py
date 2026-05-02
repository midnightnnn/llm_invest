from __future__ import annotations

from typing import Any

from arena.agents.investment_chat.context import normalize_tenant
from arena.providers.credentials import runtime_row_api_key_status
from arena.providers.registry import ProviderSpec, list_adk_provider_specs


def tenant_available_provider_specs(repo: Any, *, tenant_id: str) -> tuple[list[ProviderSpec], bool]:
    """Returns ADK providers allowed by the tenant runtime credential row.

    The boolean is True when a runtime credential row exists and therefore the
    result is intentionally credential-scoped. Missing rows keep the historical
    permissive behavior for local/dev setups that have not been provisioned.
    """
    specs = list(list_adk_provider_specs())
    tenant = normalize_tenant(tenant_id)
    loader = getattr(repo, "latest_runtime_credentials", None)
    if not callable(loader):
        return specs, False
    try:
        row = loader(tenant_id=tenant) or {}
    except Exception:
        return specs, False
    if not row:
        return specs, False
    status = runtime_row_api_key_status(row)
    allowed = [spec for spec in specs if bool(status.get(spec.provider_id))]
    return allowed, True


def tenant_available_provider_ids(repo: Any, *, tenant_id: str) -> tuple[list[str], bool]:
    specs, credential_scoped = tenant_available_provider_specs(repo, tenant_id=tenant_id)
    return [spec.provider_id for spec in specs], credential_scoped
