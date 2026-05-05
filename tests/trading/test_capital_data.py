"""Unit tests for arena.ui.routes.capital_data — pure transformation functions."""
from __future__ import annotations

from arena.ui.routes.capital_data import (
    build_event_timeline_data,
    build_recon_dashboard_data,
    build_sankey_data,
    build_treemap_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _sample_snapshots() -> dict[str, dict]:
    return {
        "gpt": {
            "cash_krw": 500_000,
            "total_equity_krw": 1_000_000,
            "positions": [
                {"ticker": "AAPL", "market_value_krw": 300_000, "return_pct": 5.2},
                {"ticker": "MSFT", "market_value_krw": 200_000, "return_pct": -2.1},
            ],
        },
        "claude": {
            "cash_krw": 800_000,
            "total_equity_krw": 900_000,
            "positions": [
                {"ticker": "GOOG", "market_value_krw": 100_000, "return_pct": 0.0},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Sankey
# ---------------------------------------------------------------------------
def test_sankey_basic_structure() -> None:
    data = build_sankey_data(_sample_snapshots(), "ok")
    assert data["recon_status"] == "ok"
    names = {n["name"] for n in data["nodes"]}
    assert "총 NAV" in names
    assert "gpt" in names
    assert "claude" in names
    assert "gpt:Cash" in names
    assert "gpt:AAPL" in names
    assert "claude:GOOG" in names
    # Links should connect total -> agent -> positions
    link_sources = {l["source"] for l in data["links"]}
    assert "총 NAV" in link_sources
    assert "gpt" in link_sources


def test_sankey_empty_snapshots() -> None:
    data = build_sankey_data({}, "unknown")
    assert data["nodes"] == [{"name": "총 NAV", "itemStyle": {"color": "#0f172a"}}]
    assert data["links"] == []


def test_sankey_skips_zero_equity() -> None:
    data = build_sankey_data({"x": {"cash_krw": 0, "total_equity_krw": 0, "positions": []}})
    assert len(data["nodes"]) == 1  # only root


def test_sankey_skips_zero_positions() -> None:
    snaps = {"gpt": {"cash_krw": 100, "total_equity_krw": 100, "positions": [{"ticker": "X", "market_value_krw": 0}]}}
    data = build_sankey_data(snaps)
    pos_nodes = [n for n in data["nodes"] if ":" in n["name"] and "Cash" not in n["name"]]
    assert len(pos_nodes) == 0


# ---------------------------------------------------------------------------
# Treemap
# ---------------------------------------------------------------------------
def test_treemap_basic_structure() -> None:
    data = build_treemap_data(_sample_snapshots())
    assert len(data["children"]) == 2
    gpt = next(c for c in data["children"] if c["name"] == "gpt")
    assert gpt["value"] == 1_000_000
    assert len(gpt["children"]) == 3  # Cash + AAPL + MSFT


def test_treemap_return_pct_colors() -> None:
    data = build_treemap_data(_sample_snapshots())
    gpt = next(c for c in data["children"] if c["name"] == "gpt")
    aapl = next(c for c in gpt["children"] if c["name"] == "AAPL")
    assert aapl["return_pct"] == 5.2
    # Positive return should have green-ish colour
    assert "129" in aapl["itemStyle"]["color"]  # rgba(16,G,129,...)


def test_treemap_empty() -> None:
    data = build_treemap_data({})
    assert data["children"] == []


def test_treemap_cash_only_agent() -> None:
    snaps = {"bot": {"cash_krw": 500, "total_equity_krw": 500, "positions": []}}
    data = build_treemap_data(snaps)
    assert len(data["children"]) == 1
    assert data["children"][0]["children"][0]["name"] == "Cash"


# ---------------------------------------------------------------------------
# Event Timeline
# ---------------------------------------------------------------------------
def test_timeline_basic() -> None:
    events = [
        {"date": "2026-03-20", "type": "deposit", "label": "입금 1,000,000원", "amount": 1_000_000},
        {"date": "2026-03-25", "type": "deposit", "label": "입금 1,000,000원", "amount": 1_000_000},
    ]
    data = build_event_timeline_data("gpt", 1_000_000, events, 3_100_000)
    assert data["agent_id"] == "gpt"
    assert data["initial_cash"] == 1_000_000
    assert data["total_invested"] == 3_000_000
    assert data["current_equity"] == 3_100_000
    assert data["event_count"] == 2
    # Chart: 시작 + 2 deposits = 3 points (no trailing "현재")
    assert len(data["chart"]["dates"]) == 3
    assert data["chart"]["balances"][0] == 1_000_000  # start
    assert data["chart"]["balances"][1] == 2_000_000  # after deposit 1
    assert data["chart"]["balances"][2] == 3_000_000  # after deposit 2
    # Events have running balance
    assert data["events"][0]["balance"] == 2_000_000
    assert data["events"][1]["balance"] == 3_000_000


def test_timeline_no_events() -> None:
    data = build_event_timeline_data("x", 500_000, [], 500_000)
    assert data["event_count"] == 0
    assert len(data["chart"]["dates"]) == 1  # 시작 only
    assert data["chart"]["balances"] == [500_000]
    assert data["total_invested"] == 500_000


def test_timeline_uses_canonical_baseline_summary_fields() -> None:
    events = [
        {"date": "2026-03-20", "type": "deposit", "label": "입금 1,000,000원", "amount": 1_000_000},
        {"date": "2026-03-21", "type": "manual_adjustment", "label": "수동 조정 +200,000원", "amount": 200_000},
    ]
    data = build_event_timeline_data(
        "gpt",
        1_000_000,
        events,
        2_950_195,
        baseline_equity=3_000_000,
        seed_positions_cost_krw=938_568,
        capital_flow_krw=1_061_432,
        capital_event_count=1,
        transfer_equity_krw=0.0,
        transfer_event_count=0,
        manual_cash_adjustment_krw=0.0,
        manual_cash_adjustment_count=1,
        current_cash_krw=200_000,
        current_positions_value_krw=2_750_195,
        seed_source="agent_state_checkpoint",
        initialized_at="2026-03-01T00:00:00+00:00",
    )

    assert data["chart"]["balances"][0] == 1_938_568
    assert data["total_invested"] == 3_000_000
    assert data["summary"]["seed_cash_krw"] == 1_000_000
    assert data["summary"]["seed_positions_cost_krw"] == 938_568
    assert data["summary"]["capital_flow_krw"] == 1_061_432
    assert data["summary"]["capital_event_count"] == 1
    assert data["summary"]["transfer_and_adjustment_count"] == 1
    assert data["summary"]["baseline_equity_krw"] == 3_000_000
    assert data["summary"]["current_positions_value_krw"] == 2_750_195
    assert data["summary"]["pnl_krw"] == -49_805


# ---------------------------------------------------------------------------
# Recon Dashboard
# ---------------------------------------------------------------------------
def test_recon_no_data() -> None:
    data = build_recon_dashboard_data(None)
    assert data["has_data"] is False
    assert data["status"] == "no_data"
    assert data["gauge_value"] == 0


def test_recon_ok() -> None:
    run = {"status": "ok", "run_at": "2026-03-26T10:00:00Z", "summary_json": {}}
    data = build_recon_dashboard_data(run, issues=[])
    assert data["has_data"] is True
    assert data["gauge_value"] == 100.0
    assert data["status"] == "ok"


def test_recon_with_errors() -> None:
    run = {"status": "error", "run_at": "2026-03-26T10:00:00Z", "summary_json": "{}"}
    issues = [
        {"severity": "error", "issue_type": "position_quantity_mismatch", "entity_key": "AAPL", "expected_json": {}, "actual_json": {}, "detail": "qty diff"},
        {"severity": "warning", "issue_type": "cash_mismatch", "entity_key": "", "expected_json": {}, "actual_json": {}, "detail": "cash diff"},
    ]
    data = build_recon_dashboard_data(run, issues)
    assert data["error_count"] == 1
    assert data["warning_count"] == 1
    assert data["gauge_value"] < 100
    assert len(data["issues"]) == 2


def test_recon_recovered() -> None:
    run = {"status": "recovered", "run_at": "2026-03-26T10:00:00Z", "summary_json": {}}
    data = build_recon_dashboard_data(run, issues=[])
    assert data["status"] == "recovered"
    assert data["gauge_value"] >= 80


def test_recon_summary_json_string() -> None:
    """summary_json may arrive as a JSON string from BigQuery."""
    run = {"status": "ok", "run_at": "2026-03-26T00:00:00Z", "summary_json": '{"agents": 3}'}
    data = build_recon_dashboard_data(run, [])
    assert data["has_data"] is True
