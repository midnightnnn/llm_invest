from __future__ import annotations

import json

from arena.config import apply_runtime_overrides, load_settings
from arena.data.bigquery.runtime_store import RuntimeStore


class _FakeConfigRepo:
    def __init__(self, values: dict[str, str]):
        self._values = values
        self.universe_rows: list[str] = ["AAPL", "MSFT"]

    def get_configs(self, tenant_id: str, config_keys: list[str]) -> dict[str, str]:
        _ = tenant_id, config_keys
        return dict(self._values)

    def latest_universe_candidate_tickers(self, *, limit=200):
        return list(self.universe_rows[:limit])


class _DummyClient:
    def __init__(self) -> None:
        self.insert_calls: list[tuple[str, list[dict[str, object]]]] = []

    def insert_rows_json(self, table_id: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        self.insert_calls.append((table_id, rows))
        return []


class _FakeSession:
    dataset_fqn = "proj.ds"
    tenant_id = "local"

    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.client = _DummyClient()
        self._rows = rows or []
        self.last_fetch: tuple[str, dict[str, object]] | None = None

    def resolve_tenant_id(self, tenant_id: str | None = None) -> str:
        return str(tenant_id or self.tenant_id or "local").strip().lower() or "local"

    def fetch_rows(self, sql: str, params: dict[str, object] | None = None) -> list[dict[str, object]]:
        self.last_fetch = (sql, params or {})
        return list(self._rows)

    def execute(self, sql: str, params: dict[str, object] | None = None) -> None:
        pass


def test_apply_runtime_overrides_applies_agents_config_risk_and_sleeve() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt"]
    settings.openai_model = "gpt-default"
    settings.gemini_model = "gemini-default"
    settings.anthropic_model = "claude-default"
    settings.max_order_krw = 10.0
    settings.sleeve_capital_krw = 100.0

    repo = _FakeConfigRepo(
        {
            "agents_config": json.dumps(
                [
                    {"id": "gemini", "provider": "gemini", "model": "gemini-3-flash", "capital_krw": 777777.0},
                    {"id": "claude", "provider": "claude", "model": "claude-sonnet-4-5", "capital_krw": 777777.0},
                ]
            ),
            "risk_policy": json.dumps(
                {
                    "max_order_krw": 123456.0,
                    "max_daily_turnover_ratio": 0.42,
                    "max_position_ratio": 0.33,
                    "min_cash_buffer_ratio": 0.12,
                    "ticker_cooldown_seconds": 45,
                    "max_daily_orders": 7,
                    "estimated_fee_bps": 4.5,
                }
            ),
            "sleeve_capital_krw": "777777.0",
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="tenant-a")

    assert out.agent_ids == ["gemini", "claude"]
    assert out.openai_model == "gpt-default"
    assert out.gemini_model == "gemini-3-flash"
    assert out.anthropic_model == "claude-sonnet-4-5"
    assert out.max_order_krw == 123456.0
    assert out.max_daily_turnover_ratio == 0.42
    assert out.max_position_ratio == 0.33
    assert out.min_cash_buffer_ratio == 0.12
    assert out.ticker_cooldown_seconds == 45
    assert out.max_daily_orders == 7
    assert out.estimated_fee_bps == 4.5
    assert out.sleeve_capital_krw == 777777.0


def test_apply_runtime_overrides_respects_explicit_empty_agents_config() -> None:
    settings = load_settings()
    settings.agent_ids = ["gpt", "gemini", "claude"]

    repo = _FakeConfigRepo({"agents_config": "[]"})

    out = apply_runtime_overrides(settings, repo, tenant_id="tenant-a")

    assert out.agent_ids == []
    assert out.agent_configs == {}
    assert out.agent_capitals == {}


def test_apply_runtime_overrides_ignores_invalid_numeric_tokens() -> None:
    settings = load_settings()
    settings.max_order_krw = 11.0
    settings.sleeve_capital_krw = 22.0

    repo = _FakeConfigRepo(
        {
            "risk_policy": json.dumps(
                {
                    "max_order_krw": "not-a-number",
                    "max_daily_orders": "oops",
                }
            ),
            "sleeve_capital_krw": "not-a-number",
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="tenant-a")
    assert out.max_order_krw == 11.0
    assert out.max_daily_orders == settings.max_daily_orders
    assert out.sleeve_capital_krw == 22.0


def test_runtime_repo_set_config_writes_normalized_row() -> None:
    session = _FakeSession()
    store = RuntimeStore(session)
    store.set_config("Tenant-A", " Agent_Ids ", "[\"gpt\"]", "tester")

    assert len(session.client.insert_calls) == 1
    table_id, rows = session.client.insert_calls[0]
    assert table_id == "proj.ds.arena_config"
    assert len(rows) == 1
    row = rows[0]
    assert row["tenant_id"] == "tenant-a"
    assert row["config_key"] == "agent_ids"
    assert row["config_value"] == "[\"gpt\"]"
    assert row["updated_by"] == "tester"
    assert "updated_at" in row


def test_runtime_repo_get_config_returns_latest_value() -> None:
    session = _FakeSession(rows=[{"config_value": "abc"}])
    store = RuntimeStore(session)
    value = store.get_config("tenant-a", "system_prompt")
    assert value == "abc"
    assert session.last_fetch is not None
    _, params = session.last_fetch
    assert params["tenant_id"] == "tenant-a"
    assert params["config_key"] == "system_prompt"


def test_runtime_repo_get_configs_returns_key_value_map() -> None:
    session = _FakeSession(
        rows=[
            {"config_key": "agent_ids", "config_value": "[\"gpt\"]"},
            {"config_key": "risk_policy", "config_value": "{\"max_order_krw\":1}"},
        ]
    )
    store = RuntimeStore(session)
    values = store.get_configs("tenant-a", ["agent_ids", "risk_policy"])
    assert values["agent_ids"] == "[\"gpt\"]"
    assert values["risk_policy"] == "{\"max_order_krw\":1}"


def test_runtime_repo_latest_config_values_returns_latest_value_per_tenant() -> None:
    session = _FakeSession(
        rows=[
            {"tenant_id": "Tenant-A", "config_value": "us"},
            {"tenant_id": "tenant-b", "config_value": "kospi"},
        ]
    )
    store = RuntimeStore(session)

    values = store.latest_config_values(config_key="kis_target_market", tenant_ids=["Tenant-A", "tenant-b", "tenant-a"])

    assert values == {"tenant-a": "us", "tenant-b": "kospi"}
    assert session.last_fetch is not None
    _, params = session.last_fetch
    assert params["config_key"] == "kis_target_market"
    assert params["tenant_ids"] == ["tenant-a", "tenant-b"]


def test_runtime_repo_list_runtime_tenants_normalizes_tokens() -> None:
    session = _FakeSession(
        rows=[
            {"tenant_id": "Tenant-A"},
            {"tenant_id": " local "},
            {"tenant_id": ""},
            {"tenant_id": None},
            {"tenant_id": "tenant-a"},
        ]
    )
    store = RuntimeStore(session)
    out = store.list_runtime_tenants(limit=100)
    assert out == ["tenant-a", "local"]


def test_apply_runtime_overrides_applies_kis_target_market() -> None:
    settings = load_settings()
    settings.kis_target_market = "nasdaq"

    repo = _FakeConfigRepo({"kis_target_market": "kospi"})
    out = apply_runtime_overrides(settings, repo, tenant_id="tenant-kr")

    assert out.kis_target_market == "kospi"


def test_apply_runtime_overrides_keeps_market_when_not_set() -> None:
    settings = load_settings()
    settings.kis_target_market = "us"

    repo = _FakeConfigRepo({})
    out = apply_runtime_overrides(settings, repo, tenant_id="tenant-us")

    assert out.kis_target_market == "us"


def test_apply_runtime_overrides_does_not_populate_default_universe_from_repo() -> None:
    settings = load_settings()
    settings.default_universe = []
    settings.universe_run_top_n = 2

    repo = _FakeConfigRepo({})
    out = apply_runtime_overrides(settings, repo, tenant_id="tenant-a")

    assert out.default_universe == []


def test_apply_runtime_overrides_merges_reconcile_excluded_tickers() -> None:
    settings = load_settings()
    settings.reconcile_excluded_tickers = ["PLTD"]

    repo = _FakeConfigRepo({"reconcile_excluded_tickers": " tsll , pltd "})
    out = apply_runtime_overrides(settings, repo, tenant_id="tenant-us")

    assert out.reconcile_excluded_tickers == ["PLTD", "TSLL"]


def test_apply_runtime_overrides_applies_runtime_strategy_knobs() -> None:
    settings = load_settings()
    settings.universe_run_top_n = 400
    settings.universe_per_exchange_cap = 200
    settings.forecast_mode = "all"
    settings.reddit_sentiment_enabled = False
    settings.research_max_tickers = 5
    settings.research_mover_top_n = 3
    settings.research_earnings_lookahead_days = 7

    repo = _FakeConfigRepo(
        {
            "universe_run_top_n": "120",
            "universe_per_exchange_cap": "80",
            "forecast_mode": "base",
            "reddit_sentiment_enabled": "true",
            "research_max_tickers": "9",
            "research_mover_top_n": "6",
            "research_earnings_lookahead_days": "14",
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="tenant-a")

    assert out.universe_run_top_n == 120
    assert out.universe_per_exchange_cap == 80
    assert out.forecast_mode == "base"
    assert out.reddit_sentiment_enabled is True
    assert out.research_max_tickers == 9
    assert out.research_mover_top_n == 6
    assert out.research_earnings_lookahead_days == 14


def test_apply_runtime_overrides_applies_agent_autonomy_config() -> None:
    settings = load_settings()
    settings.autonomy_working_set_enabled = False
    settings.autonomy_tool_default_candidates_enabled = False
    settings.autonomy_opportunity_context_enabled = False

    repo = _FakeConfigRepo(
        {
            "agent_autonomy_config": json.dumps(
                {
                    "working_set_enabled": True,
                    "tool_default_candidates_enabled": True,
                    "opportunity_context_enabled": True,
                }
            )
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="midnightnnn")

    assert out.autonomy_working_set_enabled is True
    assert out.autonomy_tool_default_candidates_enabled is True
    assert out.autonomy_opportunity_context_enabled is True


def test_apply_runtime_overrides_can_disable_agent_autonomy_config() -> None:
    settings = load_settings()
    settings.autonomy_working_set_enabled = True
    settings.autonomy_tool_default_candidates_enabled = True
    settings.autonomy_opportunity_context_enabled = True

    repo = _FakeConfigRepo(
        {
            "agent_autonomy_config": json.dumps(
                {
                    "working_set_enabled": False,
                    "tool_default_candidates_enabled": False,
                    "opportunity_context_enabled": False,
                }
            )
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="midnightnnn")

    assert out.autonomy_working_set_enabled is False
    assert out.autonomy_tool_default_candidates_enabled is False
    assert out.autonomy_opportunity_context_enabled is False


def test_apply_runtime_overrides_applies_kis_account_selection() -> None:
    settings = load_settings()
    settings.kis_account_no = ""
    settings.kis_account_product_code = "01"
    settings.kis_account_key_suffix = ""

    repo = _FakeConfigRepo(
        {
            "kis_account_no": "6431760301",
            "kis_account_product_code": "01",
            "kis_account_key_suffix": "MIDNIGHTNNN",
        }
    )

    out = apply_runtime_overrides(settings, repo, tenant_id="midnightnnn")

    assert out.kis_account_no == "6431760301"
    assert out.kis_account_product_code == "01"
    assert out.kis_account_key_suffix == "MIDNIGHTNNN"


def test_apply_runtime_overrides_applies_real_trading_approval() -> None:
    settings = load_settings()
    settings.real_trading_approved = False

    repo = _FakeConfigRepo({"real_trading_approved": "true"})
    out = apply_runtime_overrides(settings, repo, tenant_id="tenant-a")

    assert out.real_trading_approved is True
