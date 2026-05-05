from __future__ import annotations

from typing import Any


class FakeRuntimeConfigRepo:
    """Runtime config repo fake with call recording and bounded key lookup."""

    def __init__(
        self,
        values: dict[str, str] | None = None,
        *,
        universe_rows: list[str] | None = None,
    ) -> None:
        self.values = dict(values or {})
        self.universe_rows = list(universe_rows or ["AAPL", "MSFT"])
        self.get_config_calls: list[tuple[str, str]] = []
        self.get_configs_calls: list[tuple[str, list[str]]] = []

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        tenant = str(tenant_id or "").strip().lower()
        key = str(config_key or "").strip().lower()
        self.get_config_calls.append((tenant, key))
        return self.values.get(key)

    def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]:
        tenant = str(tenant_id or "").strip().lower()
        keys = [str(key or "").strip().lower() for key in config_keys or [] if str(key or "").strip()]
        self.get_configs_calls.append((tenant, keys))
        return {key: self.values[key] for key in keys if key in self.values}

    def latest_universe_candidate_tickers(self, *, limit: int = 200) -> list[str]:
        return list(self.universe_rows[:limit])


class FakeRuntimeCredentialsRepo:
    """Runtime credential repo fake used by credential-store tests."""

    def __init__(self, latest: dict[str, Any] | None = None) -> None:
        self.latest = dict(latest or {})
        self.upserts: list[dict[str, Any]] = []
        self.latest_calls: list[str] = []

    def upsert_runtime_credentials(self, **kwargs: Any) -> None:
        self.upserts.append(dict(kwargs))

    def latest_runtime_credentials(self, *, tenant_id: str) -> dict[str, Any]:
        self.latest_calls.append(str(tenant_id or "").strip())
        return dict(self.latest)
