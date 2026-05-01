from __future__ import annotations

from arena.tools.scratch_workspace import ScratchWorkspace


def test_scratch_run_python_persists_tables_within_cycle() -> None:
    workspace = ScratchWorkspace(agent_id="gpt", tenant_id="local", tool_events=[])
    workspace.set_context(
        {
            "cycle_id": "cycle_1",
            "cycle_phase": "explore",
            "market_features": [{"ticker": "AAPL", "score": 0.5}],
        }
    )

    first = workspace.run_python(
        """
df = pd.DataFrame(inputs["rows"])
df["risk_adjusted"] = np.array(df["expected"]) / np.maximum(np.array(df["vol"]), 0.01)
save_table("ranked_candidates", df.sort_values("risk_adjusted", ascending=False))
result = df[["ticker", "risk_adjusted"]].to_dict("records")
""",
        inputs={
            "rows": [
                {"ticker": "AAPL", "expected": 0.04, "vol": 0.20},
                {"ticker": "MSFT", "expected": 0.03, "vol": 0.10},
            ]
        },
    )

    assert first["status"] == "ok"
    assert first["saved_artifacts"] == [{"type": "table", "name": "ranked_candidates", "rows": 2}]

    second = workspace.run_python(
        """
ranked = load_table("ranked_candidates")
result = {
    "tables": list_tables(),
    "top": ranked.iloc[0]["ticker"],
    "context_cycle": context["cycle_id"],
}
"""
    )

    assert second["status"] == "ok"
    assert second["result"] == {
        "tables": ["ranked_candidates"],
        "top": "MSFT",
        "context_cycle": "cycle_1",
    }


def test_scratch_workspace_resets_when_cycle_changes() -> None:
    workspace = ScratchWorkspace(agent_id="gpt", tenant_id="local", tool_events=[])
    workspace.set_context({"cycle_id": "cycle_1", "cycle_phase": "explore"})
    assert workspace.run_python('save_note("thesis", "first"); result = list_notes()')["result"] == ["thesis"]

    workspace.set_context({"cycle_id": "cycle_2", "cycle_phase": "explore"})

    assert workspace.run_python("result = list_notes()")["result"] == []


def test_scratch_run_python_captures_final_expression() -> None:
    workspace = ScratchWorkspace(agent_id="gpt", tenant_id="local", tool_events=[])
    workspace.set_context({"cycle_id": "cycle_1", "cycle_phase": "explore"})

    out = workspace.run_python(
        """
positions = {"HCAI": {"qty": 2, "px": 10}, "AKAN": {"qty": 4, "px": 5}}
vals = {ticker: row["qty"] * row["px"] for ticker, row in positions.items()}
total = sum(vals.values())
weights = {ticker: value / total for ticker, value in vals.items()}
weights, total, vals
"""
    )

    assert out["status"] == "ok"
    assert out["result"] == [
        {"HCAI": 0.5, "AKAN": 0.5},
        40,
        {"HCAI": 20, "AKAN": 20},
    ]


def test_scratch_run_python_blocks_network_imports() -> None:
    workspace = ScratchWorkspace(agent_id="gpt", tenant_id="local", tool_events=[])
    workspace.set_context({"cycle_id": "cycle_1", "cycle_phase": "explore"})

    out = workspace.run_python("import socket\nresult = 'unreachable'")

    assert out["status"] == "error"
    assert "blocked import" in out["error"]
    assert "socket" in out["error"]
