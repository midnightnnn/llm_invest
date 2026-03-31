"""Tests for the unified agents_config feature.

Covers:
- config.py: agents_config parsing, per-agent capitals, model override
- orchestrator.py: per-agent capital total cash
- sleeve.py: capital_per_agent / target_capitals params
- server.py: POST /admin/agents with agents_config_json, GET /admin/agents,
  settings page rendering of unified agents panel
"""
from __future__ import annotations

import json
import math

import pytest

from arena.config import AgentConfig, apply_runtime_overrides, load_settings, merge_agent_risk_settings
from arena.models import AccountSnapshot, Position
from arena.data.bigquery.sleeve_store import SleeveStore
from tests.direct_route_client import DirectRouteClient


# ──────────────────────────────────────────────────
# Helpers / Fakes
# ──────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────
# config.py: agents_config parsing
# ──────────────────────────────────────────────────

def test_agents_config_overrides_agent_ids_and_capitals() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 500_000

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gemini", "model": "gemini-3-flash", "capital_krw": 1_000_000},
                {"id": "claude", "model": "claude-sonnet-4-6", "capital_krw": 2_000_000},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gemini", "claude"]
    assert out.agent_capitals["gemini"] == 1_000_000
    assert out.agent_capitals["claude"] == 2_000_000
    assert out.gemini_model == "gemini-3-flash"
    assert out.anthropic_model == "claude-sonnet-4-6"


def test_agents_config_fallback_to_sleeve_capital_when_no_capital() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 750_000

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gpt", "model": "gpt-4.1"},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gpt"]
    # capital_krw not specified → fallback to sleeve_capital_krw
    assert out.agent_capitals["gpt"] == 750_000


def test_agents_config_missing_ignores_legacy_agent_keys() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 100_000

    repo = _FakeConfigRepo(
        {
            "agent_ids": json.dumps(["gemini", "claude"]),
            "agent_models": json.dumps({"gemini": "gemini-3-flash"}),
            "sleeve_capital_krw": "200000",
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gpt"]
    assert out.sleeve_capital_krw == 200_000
    assert out.openai_model == settings.openai_model
    assert out.agent_capitals["gpt"] == 200_000


def test_agents_config_skips_invalid_entries() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 500_000

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gemini", "model": "m1", "capital_krw": 1_000_000},
                "not-a-dict",
                {"id": "", "model": "m2"},
                {"model": "m3"},
                {"id": "claude", "capital_krw": "not-a-number"},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gemini", "claude"]
    assert out.agent_capitals["gemini"] == 1_000_000
    # claude had invalid capital → fallback to sleeve_capital_krw
    assert out.agent_capitals["claude"] == 500_000


def test_agents_config_per_provider_model_override() -> None:
    settings = load_settings()
    settings.openai_model = "gpt-old"
    settings.gemini_model = "gemini-old"
    settings.anthropic_model = "claude-old"

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gpt", "model": "gpt-5"},
                {"id": "gemini", "model": "gemini-3-ultra"},
                {"id": "claude", "model": "claude-opus-4-6"},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.openai_model == "gpt-5"
    assert out.gemini_model == "gemini-3-ultra"
    assert out.anthropic_model == "claude-opus-4-6"


# ──────────────────────────────────────────────────
# sleeve.py: capital_per_agent in ensure_agent_sleeves
# ──────────────────────────────────────────────────

def test_ensure_agent_sleeves_uses_capital_per_agent(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)
    store = _make_init_store()

    store.ensure_agent_sleeves(
        agent_ids=["gpt", "gemini"],
        total_cash_krw=3_000_000,
        capital_per_agent={"gpt": 1_000_000, "gemini": 2_000_000},
    )

    assert len(store.session.client.payloads) == 2
    payloads_by_agent = {p["agent_id"]: p for p in store.session.client.payloads}
    assert float(payloads_by_agent["gpt"]["initial_cash_krw"]) == 1_000_000
    assert float(payloads_by_agent["gemini"]["initial_cash_krw"]) == 2_000_000


def test_ensure_agent_sleeves_without_capital_per_agent_uses_equal_split(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_SLEEVE_BOOTSTRAP_FROM_ACCOUNT", raising=False)
    store = _make_init_store()

    store.ensure_agent_sleeves(
        agent_ids=["gpt", "gemini"],
        total_cash_krw=2_000_000,
    )

    assert len(store.session.client.payloads) == 2
    for p in store.session.client.payloads:
        assert float(p["initial_cash_krw"]) == 1_000_000


# ──────────────────────────────────────────────────
# sleeve.py: target_capitals in retarget
# ──────────────────────────────────────────────────

def test_retarget_uses_target_capitals_per_agent() -> None:
    store = _RetargetSleeveStore(
        {
            "gpt": AccountSnapshot(
                cash_krw=100_000,
                total_equity_krw=400_000,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000,
                        market_price_krw=150_000,
                    )
                },
            ),
            "gemini": AccountSnapshot(
                cash_krw=200_000,
                total_equity_krw=500_000,
                positions={
                    "MSFT": Position(
                        ticker="MSFT",
                        exchange_code="NASD",
                        instrument_id="NASD:MSFT",
                        quantity=1.0,
                        avg_price_krw=300_000,
                        market_price_krw=300_000,
                    )
                },
            ),
        }
    )

    out = store.retarget_agent_sleeves_preserve_positions(
        agent_ids=["gpt", "gemini"],
        target_sleeve_capital_krw=500_000,
        target_capitals={"gpt": 600_000, "gemini": 800_000},
    )

    payloads_by_agent = {p["agent_id"]: p for p in store.session.client.payloads}
    # gpt: positions_value=300_000, target=600_000, cash=300_000
    assert float(payloads_by_agent["gpt"]["initial_cash_krw"]) == pytest.approx(300_000)
    assert out["gpt"]["over_target"] is False
    # gemini: positions_value=300_000, target=800_000, cash=500_000
    assert float(payloads_by_agent["gemini"]["initial_cash_krw"]) == pytest.approx(500_000)
    assert out["gemini"]["over_target"] is False


def test_retarget_without_target_capitals_uses_uniform_target() -> None:
    store = _RetargetSleeveStore(
        {
            "gpt": AccountSnapshot(
                cash_krw=100_000,
                total_equity_krw=400_000,
                positions={
                    "AAPL": Position(
                        ticker="AAPL",
                        exchange_code="NASD",
                        instrument_id="NASD:AAPL",
                        quantity=2.0,
                        avg_price_krw=120_000,
                        market_price_krw=150_000,
                    )
                },
            ),
        }
    )

    out = store.retarget_agent_sleeves_preserve_positions(
        agent_ids=["gpt"],
        target_sleeve_capital_krw=500_000,
    )

    first = store.session.client.payloads[0]
    # positions_value=300_000, target=500_000, cash=200_000
    assert float(first["initial_cash_krw"]) == pytest.approx(200_000)
    assert out["gpt"]["over_target"] is False


# ──────────────────────────────────────────────────
# orchestrator.py: per-agent capital total cash
# ──────────────────────────────────────────────────

def test_orchestrator_per_agent_capital_total_cash() -> None:
    from arena.agents.base import AgentOutput
    from arena.config import Settings
    from arena.models import BoardPost
    from arena.orchestrator import ArenaOrchestrator

    class _FakeRepo:
        def __init__(self):
            self.ensure_calls = []

        def ensure_agent_sleeves(self, *, agent_ids, total_cash_krw, capital_per_agent=None, initialized_at=None):
            _ = initialized_at
            self.ensure_calls.append({
                "agent_ids": list(agent_ids),
                "total_cash_krw": float(total_cash_krw),
                "capital_per_agent": dict(capital_per_agent) if capital_per_agent else None,
            })
            return {a: {} for a in agent_ids}

        def build_agent_sleeve_snapshot(self, *, agent_id, sources=None, include_simulated=True):
            return AccountSnapshot(cash_krw=1_000_000, total_equity_krw=1_000_000, positions={}), 1_000_000, {}

        def upsert_agent_nav_daily(self, **kwargs):
            pass

    class _FakeGateway:
        def __init__(self, repo):
            self.repo = repo

    class _FakeCtx:
        def build(self, agent_id, snapshot, sleeve_baseline_equity_krw=None, sleeve_meta=None):
            return {"portfolio": {}, "market_features": [], "memory_events": [], "board_posts": []}

    class _FakeBoard:
        def publish(self, post):
            pass
        def recent(self, limit):
            return []

    class _DummyAgent:
        def __init__(self, agent_id):
            self.agent_id = agent_id
        def generate(self, context):
            return AgentOutput(
                intents=[],
                board_post=BoardPost(agent_id=self.agent_id, title="t", body="b", tickers=[]),
            )

    s = load_settings()
    s.agent_ids = ["gpt", "gemini"]
    s.sleeve_capital_krw = 1_000_000
    s.agent_capitals = {"gpt": 1_500_000, "gemini": 500_000}
    s.trading_mode = "paper"

    repo = _FakeRepo()
    orch = ArenaOrchestrator(
        settings=s,
        context_builder=_FakeCtx(),
        board_store=_FakeBoard(),
        gateway=_FakeGateway(repo),
        agents=[_DummyAgent("gpt"), _DummyAgent("gemini")],
    )

    orch.run_cycle(snapshot=None)

    assert repo.ensure_calls
    call = repo.ensure_calls[0]
    # Total should be sum of per-agent capitals: 1_500_000 + 500_000 = 2_000_000
    assert call["total_cash_krw"] == 2_000_000
    assert call["capital_per_agent"] == {"gpt": 1_500_000, "gemini": 500_000}


# ──────────────────────────────────────────────────
# server.py: UI endpoints
# ──────────────────────────────────────────────────

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


def test_admin_agents_save_with_agents_config_json(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    agents_config = [
        {"id": "gemini", "model": "gemini-3-flash", "capital_krw": 1_500_000, "api_key": ""},
        {"id": "gpt", "model": "gpt-4.1", "capital_krw": 1_000_000, "api_key": ""},
        {"id": "claude", "model": "claude-sonnet-4-6", "capital_krw": 2_000_000, "api_key": ""},
    ]

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "agents_config_json": json.dumps(agents_config),
        },
        follow_redirects=False,
    )

    assert response.status_code == 303

    # Verify agents_config saved to DB
    saved_raw = repo.get_config("local", "agents_config")
    assert saved_raw is not None
    saved = json.loads(saved_raw)
    assert len(saved) == 3
    assert saved[0]["id"] == "gemini"
    assert saved[0]["capital_krw"] == 1_500_000
    # api_key should NOT be in DB config
    assert "api_key" not in saved[0]

    assert repo.get_config("local", "agent_ids") is None
    assert repo.get_config("local", "agent_models") is None

    # Verify sleeve retarget called with per-agent capitals
    assert repo.sleeve_sync_calls
    last = repo.sleeve_sync_calls[-1]
    assert last["target_capitals"]["gemini"] == 1_500_000
    assert last["target_capitals"]["gpt"] == 1_000_000
    assert last["target_capitals"]["claude"] == 2_000_000
    assert last["tenant_id"] == "local"

    # Verify NAV upsert happened
    assert len(repo.nav_upsert_calls) == 3


def test_admin_agents_save_rejects_empty_config(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "agents_config_json": "[]",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    # Should redirect with error
    location = response.headers.get("location", "")
    assert "ok=0" in location


def test_admin_agents_save_rejects_unknown_provider(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "agents_config_json": json.dumps([
                {"id": "unknown_provider", "model": "model-x", "capital_krw": 500_000},
            ]),
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    location = response.headers.get("location", "")
    assert "ok=0" in location


def test_admin_agents_get_returns_agents_config(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    # Pre-save agents_config
    agents_config = [
        {"id": "gpt", "model": "gpt-4.1", "capital_krw": 800_000},
    ]
    repo.cfg[("local", "agents_config")] = json.dumps(agents_config)

    response = client.get("/admin/agents", params={"tenant_id": "local"})
    assert response.status_code == 200
    payload = response.json()
    assert "agents_config" in payload
    assert "api_key_status" in payload
    assert "research_status" in payload
    assert payload["agents_config"][0]["id"] == "gpt"
    assert payload["agents_config"][0]["capital_krw"] == 800_000


def test_settings_page_renders_unified_agents_panel(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    response = client.get("/settings")
    assert response.status_code == 200
    html = response.text

    assert "Agents" in html
    assert "agent-card" in html
    assert "agent-toggle-btn" in html
    assert "agent-save-btn" in html
    # Global checkboxes should be gone — per-agent only
    assert "agent-global-prompt" not in html
    assert "agent-global-risk" not in html
    assert "agent-global-tools" not in html
    # The old sleeve panel tab should be gone
    assert "Sleeve Capital" not in html


def test_admin_agents_save_does_not_sync_global_sleeve_capital(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    agents_config = [
        {"id": "gpt", "model": "gpt-4.1", "capital_krw": 600_000},
        {"id": "gemini", "model": "gemini-3", "capital_krw": 400_000},
    ]

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "agents_config_json": json.dumps(agents_config),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    assert repo.get_config("local", "sleeve_capital_krw") is None


def test_admin_agents_save_single_agent(monkeypatch) -> None:
    """User can configure just one agent."""
    client, repo = _build_test_client(monkeypatch)

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "agents_config_json": json.dumps([
                {"id": "claude", "model": "claude-opus-4-6", "capital_krw": 5_000_000},
            ]),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    saved = json.loads(repo.get_config("local", "agents_config"))
    assert len(saved) == 1
    assert saved[0]["id"] == "claude"
    assert saved[0]["capital_krw"] == 5_000_000

    assert repo.get_config("local", "agent_ids") is None
    assert repo.sleeve_sync_calls
    assert repo.sleeve_sync_calls[-1]["target_capitals"] == {"claude": 5_000_000}


# ──────────────────────────────────────────────────
# Per-Agent Settings: AgentConfig parsing
# ──────────────────────────────────────────────────

def test_agents_config_parses_per_agent_provider_and_fields() -> None:
    """agents_config with explicit provider + per-agent prompt/risk/tools."""
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.sleeve_capital_krw = 500_000

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {
                    "id": "aggressive-gpt",
                    "provider": "gpt",
                    "model": "gpt-5.2",
                    "capital_krw": 2_000_000,
                    "system_prompt": "Be aggressive.",
                    "risk_policy": {"max_order_krw": 50_000_000, "max_daily_orders": 20},
                    "disabled_tools": ["screen_market"],
                },
                {
                    "id": "safe-gpt",
                    "provider": "gpt",
                    "model": "gpt-4.1",
                    "capital_krw": 1_000_000,
                },
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["aggressive-gpt", "safe-gpt"]
    assert "aggressive-gpt" in out.agent_configs
    assert "safe-gpt" in out.agent_configs

    ac_agg = out.agent_configs["aggressive-gpt"]
    assert ac_agg.provider == "gpt"
    assert ac_agg.model == "gpt-5.2"
    assert ac_agg.capital_krw == 2_000_000
    assert ac_agg.system_prompt == "Be aggressive."
    assert ac_agg.risk_overrides == {"max_order_krw": 50_000_000, "max_daily_orders": 20}
    assert ac_agg.disabled_tools == ["screen_market"]

    ac_safe = out.agent_configs["safe-gpt"]
    assert ac_safe.provider == "gpt"
    assert ac_safe.system_prompt is None
    assert ac_safe.risk_overrides is None
    assert ac_safe.disabled_tools is None


def test_agents_config_infers_provider_from_id() -> None:
    """Legacy agents_config without explicit provider infers from id."""
    settings = load_settings()
    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gpt", "model": "gpt-5.2"},
                {"id": "gemini", "model": "gemini-3-flash"},
                {"id": "claude", "model": "claude-sonnet-4-6"},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_configs["gpt"].provider == "gpt"
    assert out.agent_configs["gemini"].provider == "gemini"
    assert out.agent_configs["claude"].provider == "claude"


def test_agents_config_same_provider_duplicate() -> None:
    """Two agents with same provider (gpt) but different ids."""
    settings = load_settings()
    settings.openai_api_key = "sk-test"

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps([
                {"id": "gpt-a", "provider": "gpt", "model": "gpt-5.2", "capital_krw": 1_000_000},
                {"id": "gpt-b", "provider": "gpt", "model": "gpt-4.1", "capital_krw": 500_000},
            ]),
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gpt-a", "gpt-b"]
    assert out.agent_configs["gpt-a"].model == "gpt-5.2"
    assert out.agent_configs["gpt-b"].model == "gpt-4.1"
    assert out.agent_capitals["gpt-a"] == 1_000_000
    assert out.agent_capitals["gpt-b"] == 500_000


def test_agents_config_empty_keeps_env_agents_normalized() -> None:
    """When agents_config is absent, env/default agents remain normalized."""
    settings = load_settings()
    settings.agent_ids = ["gpt", "gemini"]

    repo = _FakeConfigRepo({})

    out = apply_runtime_overrides(settings, repo, tenant_id="t")

    assert out.agent_ids == ["gpt", "gemini"]
    assert out.agent_configs["gpt"].provider == "gpt"
    assert out.agent_configs["gemini"].provider == "gemini"


# ──────────────────────────────────────────────────
# merge_agent_risk_settings
# ──────────────────────────────────────────────────

def test_merge_agent_risk_settings_applies_overrides() -> None:
    settings = load_settings()
    settings.max_order_krw = 100_000
    settings.max_daily_orders = 5

    ac = AgentConfig(
        agent_id="agg",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        risk_overrides={"max_order_krw": 500_000, "max_daily_orders": 20},
    )

    merged = merge_agent_risk_settings(settings, ac)

    assert merged.max_order_krw == 500_000
    assert merged.max_daily_orders == 20
    # Original unchanged
    assert settings.max_order_krw == 100_000
    assert settings.max_daily_orders == 5


def test_merge_agent_risk_settings_returns_original_when_none() -> None:
    settings = load_settings()
    settings.max_order_krw = 100_000

    result = merge_agent_risk_settings(settings, None)
    assert result is settings

    ac = AgentConfig(
        agent_id="x",
        provider="gpt",
        model="gpt-5.2",
        capital_krw=1_000_000,
        risk_overrides=None,
    )
    result = merge_agent_risk_settings(settings, ac)
    assert result is settings


# ──────────────────────────────────────────────────
# UI: per-agent settings save with provider
# ──────────────────────────────────────────────────

def test_admin_agents_save_with_custom_id_and_provider(monkeypatch) -> None:
    client, repo = _build_test_client(monkeypatch)

    agents_config = [
        {"id": "aggressive-gpt", "provider": "gpt", "model": "gpt-5.2", "capital_krw": 2_000_000},
        {"id": "safe-claude", "provider": "claude", "model": "claude-sonnet-4-6", "capital_krw": 1_000_000},
    ]

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "agents_config_json": json.dumps(agents_config),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    saved = json.loads(repo.get_config("local", "agents_config"))
    assert len(saved) == 2
    assert saved[0]["id"] == "aggressive-gpt"
    assert saved[0]["provider"] == "gpt"
    assert saved[1]["id"] == "safe-claude"
    assert saved[1]["provider"] == "claude"

    assert repo.get_config("local", "agent_ids") is None


def test_admin_agents_save_with_per_agent_fields(monkeypatch) -> None:
    """Per-agent system_prompt, risk_policy, disabled_tools are saved to DB."""
    client, repo = _build_test_client(monkeypatch)

    agents_config = [
        {
            "id": "custom-agent",
            "provider": "gpt",
            "model": "gpt-5.2",
            "capital_krw": 1_000_000,
            "system_prompt": "Be aggressive trader.",
            "risk_policy": {"max_order_krw": 50_000_000},
            "disabled_tools": ["screen_market"],
        },
    ]

    response = client.post(
        "/admin/agents",
        data={
            "tenant_id": "local",
            "updated_by": "tester",
            "agents_config_json": json.dumps(agents_config),
        },
        follow_redirects=False,
    )
    assert response.status_code == 303

    saved = json.loads(repo.get_config("local", "agents_config"))
    assert len(saved) == 1
    assert saved[0]["system_prompt"] == "Be aggressive trader."
    assert saved[0]["risk_policy"] == {"max_order_krw": 50_000_000}
    assert saved[0]["disabled_tools"] == ["screen_market"]
