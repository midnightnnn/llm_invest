from __future__ import annotations

from arena.data.bigquery.sleeve_store import SleeveStore
from arena.models import AccountSnapshot
from tests.direct_route_client import DirectRouteClient


class _FakeConfigRepo:
    def __init__(self, values: dict[str, str]):
        self._values = values

    def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]:
        _ = tenant_id, config_keys
        return dict(self._values)


class _InsertClient:
    def __init__(self):
        self.payloads: list[dict[str, object]] = []

    def insert_rows_json(self, table_id: str, rows: list[dict[str, object]]):
        _ = table_id
        self.payloads.extend(rows)
        return []


class _FakeSession:
    """Minimal stand-in for BigQuerySession used by SleeveStore."""

    def __init__(self, *, fetch_rows_fn=None):
        self.dataset_fqn = "proj.ds"
        self.client = _InsertClient()
        self._fetch_rows_fn = fetch_rows_fn or (lambda sql, params=None: [])

    def resolve_tenant_id(self, tenant_id=None):
        return str(tenant_id or "local").strip().lower() or "local"

    def fetch_rows(self, sql, params=None):
        return self._fetch_rows_fn(sql, params)

    def execute(self, sql, params=None):
        pass


def _make_init_store():
    """Create a SleeveStore whose session returns empty rows (no existing sleeves)."""
    session = _FakeSession()
    store = SleeveStore(session)
    return store


class _RetargetSleeveStore(SleeveStore):
    """SleeveStore subclass that stubs build_agent_sleeve_snapshot with canned data."""

    def __init__(self, snapshots: dict[str, AccountSnapshot]):
        session = _FakeSession()
        super().__init__(session)
        self._snapshots = snapshots

    def build_agent_sleeve_snapshot(
        self,
        *,
        agent_id,
        sources=None,
        include_simulated=True,
        tenant_id=None,
        as_of_ts=None,
    ):
        snapshot = self._snapshots.get(
            str(agent_id),
            AccountSnapshot(cash_krw=0.0, total_equity_krw=0.0, positions={}),
        )
        return snapshot, float(snapshot.total_equity_krw), {}


def _build_test_client(monkeypatch):
    """Builds a direct-call UI client with settings and a dummy repo."""
    from arena.config import load_settings
    from arena.models import AccountSnapshot
    from arena.ui.server import _build_app

    class _DummyRepo:
        dataset_fqn = "proj.ds"
        project = "proj"
        location = "asia-northeast3"

        def __init__(self):
            self.cfg: dict[tuple[str, str], str] = {}
            self.sleeve_sync_calls = []
            self.nav_upsert_calls = []

        def list_runtime_user_tenants(self, *, user_email):
            return []

        def ensure_runtime_user_tenant(self, **kwargs):
            return None

        def latest_runtime_credentials(self, *, tenant_id):
            return {}

        def recent_runtime_credentials(self, *, limit=20):
            return []

        def upsert_runtime_credentials(self, **kwargs):
            return None

        def has_runtime_user_tenant(self, *, user_email, tenant_id):
            return True

        def append_runtime_audit_log(self, **kwargs):
            return None

        def set_config(self, tenant_id, config_key, value, updated_by=None, **kwargs):
            self.cfg[(tenant_id, config_key)] = value

        def get_config(self, tenant_id, config_key):
            return self.cfg.get((tenant_id, config_key))

        def get_configs(self, tenant_id, config_keys):
            return {k: self.cfg[(tenant_id, k)] for k in config_keys if (tenant_id, k) in self.cfg}

        def fetch_rows(self, sql, params=None):
            return []

        def retarget_agent_sleeves_preserve_positions(
            self, *, agent_ids, target_sleeve_capital_krw,
            target_capitals=None, initialized_at=None,
            include_simulated=True, sources=None, tenant_id=None,
        ):
            self.sleeve_sync_calls.append({
                "agent_ids": list(agent_ids),
                "target_sleeve_capital_krw": float(target_sleeve_capital_krw),
                "target_capitals": dict(target_capitals) if target_capitals else None,
                "tenant_id": tenant_id,
            })
            return {a: {"over_target": False} for a in agent_ids}

        def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True, tenant_id=None):
            return (
                AccountSnapshot(cash_krw=500_000, total_equity_krw=500_000, positions={}),
                500_000, {"agent_id": agent_id},
            )

        def upsert_agent_nav_daily(self, *, nav_date, agent_id, nav_krw, baseline_equity_krw, tenant_id=None, **kwargs):
            self.nav_upsert_calls.append({"agent_id": agent_id, "nav_krw": nav_krw, "tenant_id": tenant_id, **kwargs})

    monkeypatch.setenv("ARENA_UI_SETTINGS_ENABLED", "true")
    monkeypatch.setenv("ARENA_UI_AUTH_ENABLED", "false")
    repo = _DummyRepo()
    app = _build_app(repo=repo, settings=load_settings())
    return DirectRouteClient(app), repo
