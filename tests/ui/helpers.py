from __future__ import annotations

from arena.config import load_settings
from arena.models import AccountSnapshot
from arena.ui.server import _build_app
from tests.direct_route_client import DirectRouteClient


class _DummyRepo:
    dataset_fqn = "proj.ds"
    project = "proj"
    location = "asia-northeast3"

    def __init__(self) -> None:
        self.cfg: dict[tuple[str, str], str] = {}
        self.runtime_credentials: dict[str, dict[str, str]] = {}
        self.fetch_calls: list[tuple[str, dict | None]] = []
        self.sleeve_sync_calls: list[dict[str, object]] = []
        self.capital_sync_calls: list[dict[str, object]] = []
        self.nav_upsert_calls: list[dict[str, object]] = []
        self.snapshot_calls: list[dict[str, object]] = []
        self.latest_run_status_row: dict[str, object] | None = None
        self.latest_recon_row: dict[str, object] | None = None
        self.recon_issue_rows: list[dict[str, object]] = []
        self.user_tenants: dict[str, list[dict[str, str]]] = {}
        self.access_requests: list[dict[str, str]] = []

    def list_runtime_user_tenants(self, *, user_email: str) -> list[dict[str, str]]:
        return [dict(row) for row in self.user_tenants.get(str(user_email or "").strip().lower(), [])]

    def ensure_runtime_user_tenant(self, **kwargs) -> None:
        user_email = str(kwargs.get("user_email") or "").strip().lower()
        tenant_id = str(kwargs.get("tenant_id") or "").strip().lower()
        role = str(kwargs.get("role") or "owner").strip().lower() or "owner"
        if not user_email or not tenant_id:
            return None
        rows = self.user_tenants.setdefault(user_email, [])
        if not any(str(row.get("tenant_id") or "").strip().lower() == tenant_id for row in rows):
            rows.append({"user_email": user_email, "tenant_id": tenant_id, "role": role})
        return None

    def latest_runtime_credentials(self, *, tenant_id: str) -> dict[str, str]:
        return dict(self.runtime_credentials.get(str(tenant_id or "").strip().lower(), {}))

    def upsert_runtime_credentials(self, **kwargs) -> None:
        tenant = str(kwargs.get("tenant_id") or "").strip().lower()
        if not tenant:
            return None
        current = dict(self.runtime_credentials.get(tenant, {}))
        current.update(kwargs)
        self.runtime_credentials[tenant] = current
        return None

    def recent_runtime_credentials(self, *, limit: int = 20) -> list[dict[str, str]]:
        _ = limit
        return []

    def has_runtime_user_tenant(self, *, user_email: str, tenant_id: str) -> bool:
        user = str(user_email or "").strip().lower()
        tenant = str(tenant_id or "").strip().lower()
        return any(str(row.get("tenant_id") or "").strip().lower() == tenant for row in self.user_tenants.get(user, []))

    def latest_runtime_access_request(self, *, user_email: str) -> dict[str, str] | None:
        user = str(user_email or "").strip().lower()
        matches = [row for row in self.access_requests if str(row.get("user_email") or "").strip().lower() == user]
        return dict(matches[-1]) if matches else None

    def ensure_runtime_access_request_pending(
        self,
        *,
        user_email: str,
        user_name: str | None = None,
        google_sub: str | None = None,
    ):
        latest = self.latest_runtime_access_request(user_email=user_email)
        if latest and str(latest.get("status") or "").strip().lower() == "pending":
            return latest
        row = {
            "user_email": str(user_email or "").strip().lower(),
            "user_name": str(user_name or "").strip(),
            "google_sub": str(google_sub or "").strip(),
            "requested_at": "2026-03-21T00:00:00+00:00",
            "status": "pending",
            "note": "",
        }
        self.access_requests.append(row)
        return dict(row)

    def append_runtime_audit_log(self, **kwargs) -> None:
        _ = kwargs
        return None

    def set_config(self, tenant_id: str, config_key: str, value: str, updated_by: str | None = None, **kwargs) -> None:
        _ = updated_by, kwargs
        self.cfg[(tenant_id, config_key)] = value

    def get_config(self, tenant_id: str, config_key: str) -> str | None:
        return self.cfg.get((tenant_id, config_key))

    def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]:
        return {
            key: self.cfg[(tenant_id, key)]
            for key in config_keys
            if (tenant_id, key) in self.cfg
        }

    def fetch_rows(self, sql: str, params: dict | None = None) -> list[dict]:
        self.fetch_calls.append((sql, params))
        if "FROM `proj.ds.reconciliation_issues`" in sql:
            return [dict(row) for row in self.recon_issue_rows]
        return []

    def latest_tenant_run_status(self, *, tenant_id: str, run_type: str | None = None):
        _ = tenant_id, run_type
        return self.latest_run_status_row

    def latest_reconciliation_run(self, *, tenant_id: str | None = None):
        _ = tenant_id
        return self.latest_recon_row

    def retarget_agent_sleeves_preserve_positions(
        self,
        *,
        agent_ids: list[str],
        target_sleeve_capital_krw: float,
        target_capitals: dict[str, float] | None = None,
        initialized_at=None,
        include_simulated: bool = True,
        sources=None,
        tenant_id: str | None = None,
    ) -> dict[str, dict[str, object]]:
        _ = initialized_at
        self.sleeve_sync_calls.append(
            {
                "agent_ids": list(agent_ids),
                "target_sleeve_capital_krw": float(target_sleeve_capital_krw),
                "target_capitals": dict(target_capitals) if target_capitals else None,
                "include_simulated": include_simulated,
                "sources": sources,
                "tenant_id": tenant_id,
            }
        )
        return {
            str(a): {
                "over_target": False,
            }
            for a in agent_ids
        }

    def retarget_agent_capitals_preserve_positions(
        self,
        *,
        agent_ids: list[str],
        target_sleeve_capital_krw: float,
        target_capitals: dict[str, float] | None = None,
        occurred_at=None,
        include_simulated: bool = True,
        sources=None,
        tenant_id: str | None = None,
        created_by: str = "system",
    ) -> dict[str, dict[str, object]]:
        _ = occurred_at
        self.capital_sync_calls.append(
            {
                "agent_ids": list(agent_ids),
                "target_sleeve_capital_krw": float(target_sleeve_capital_krw),
                "target_capitals": dict(target_capitals) if target_capitals else None,
                "include_simulated": include_simulated,
                "sources": sources,
                "tenant_id": tenant_id,
                "created_by": created_by,
            }
        )
        return {
            str(a): {
                "over_target": False,
                "capital_flow_krw": 0.0,
            }
            for a in agent_ids
        }

    def build_agent_sleeve_snapshot(
        self,
        *,
        agent_id: str,
        sources=None,
        include_simulated: bool = True,
        tenant_id: str | None = None,
    ):
        self.snapshot_calls.append(
            {
                "agent_id": agent_id,
                "sources": list(sources) if isinstance(sources, list) else sources,
                "include_simulated": include_simulated,
                "tenant_id": tenant_id,
            }
        )
        return (
            AccountSnapshot(cash_krw=500000.0, total_equity_krw=500000.0, positions={}),
            500000.0,
            {"agent_id": agent_id},
        )

    def upsert_agent_nav_daily(
        self,
        *,
        nav_date,
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
        self.nav_upsert_calls.append(
            {
                "nav_date": nav_date,
                "agent_id": agent_id,
                "nav_krw": float(nav_krw),
                "baseline_equity_krw": float(baseline_equity_krw),
                "cash_krw": cash_krw,
                "market_value_krw": market_value_krw,
                "capital_flow_krw": capital_flow_krw,
                "fx_source": fx_source,
                "valuation_source": valuation_source,
                "tenant_id": tenant_id,
            }
        )


def _client(monkeypatch) -> DirectRouteClient:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _DummyRepo()
    app = _build_app(repo=repo, settings=load_settings())
    return DirectRouteClient(app)


def _client_with_repo(monkeypatch) -> tuple[DirectRouteClient, _DummyRepo]:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _DummyRepo()
    app = _build_app(repo=repo, settings=load_settings())
    return DirectRouteClient(app), repo


def _client_with_repo_and_credential_store(monkeypatch, store_cls) -> tuple[DirectRouteClient, _DummyRepo]:
    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    monkeypatch.setattr("arena.ui.app.CredentialStore", store_cls)
    repo = _DummyRepo()
    settings = load_settings()
    settings.arena_mode = "gcp"
    app = _build_app(repo=repo, settings=settings)
    return DirectRouteClient(app), repo
